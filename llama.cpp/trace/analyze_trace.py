#!/usr/bin/env python3
"""
LLM Memory Trace Analysis & Visualization Pipeline
===================================================
Parses the JSONL trace output from llama.cpp MEM_TRACE instrumentation,
generates visualizations and an interactive HTML analysis report with
optimization recommendations for physical memory usage.
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "llmop-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from trace_metrics import collect_core_metrics, inference_latency_records
from stage_scheduling_analysis import analyze_stage_scheduling_opportunity


# ─────────────────────────────────────────────
#  JSONL Parsing
# ─────────────────────────────────────────────

def load_jsonl(path: str, max_records: int | None = None) -> list[dict]:
    """Load a JSONL file, returning a list of parsed dicts.
    If max_records is set, sample uniformly to not exceed that count.
    For very large files, this avoids OOM and excessive processing time."""
    if not os.path.exists(path):
        print(f"  [WARN] missing: {path}")
        return []

    file_size = os.path.getsize(path)
    records = []

    # If file is huge (>500MB), use sampling
    if file_size > 500 * 1024 * 1024 and max_records is None:
        max_records = 100000  # default cap for huge files
        print(f"  [INFO] Large file ({file_size / (1024**3):.1f} GB), sampling to {max_records:,} records")

    if max_records is not None and file_size > 100 * 1024 * 1024:
        # Count lines first (fast, line-oriented)
        line_count = 0
        with open(path, "r") as f:
            for _ in f:
                line_count += 1

        if line_count <= max_records:
            # Can load all
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        else:
            # Uniform sampling: take every Nth line
            step = line_count / max_records
            with open(path, "r") as f:
                for i, line in enumerate(f):
                    # Use deterministic sampling
                    if int(i / step) != int((i - 1) / step) if i > 0 else True:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            print(f"  [INFO] Sampled {len(records):,} records from {line_count:,} lines ({len(records) / line_count * 100:.1f}%)")
    else:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return records


def load_all(trace_dir: str) -> dict[str, list[dict]]:
    """Load all four trace JSONL files + summary.
    Tensor trace uses a dedicated first-touch pass + sampling for other events."""
    tensor_path = os.path.join(trace_dir, "tensor_trace.jsonl")
    tensor_records = load_jsonl(tensor_path, max_records=200000)

    # Also do a targeted pass for tensor load / first-touch / residency events
    # from the full file. These are sparse but critical for mmap analysis.
    key_records = load_key_tensor_events(tensor_path)
    if key_records:
        print(f"  [INFO] Tensor key-event pass: {len(key_records)} events from full file")
        sampled_keys = {json.dumps(r, sort_keys=True) for r in tensor_records
                        if r.get("event") == "TENSOR_LOAD" or r.get("first_touch") is True or "page_count" in r}
        for record in key_records:
            if json.dumps(record, sort_keys=True) not in sampled_keys:
                tensor_records.append(record)

    data = {
        "tensor": tensor_records,
        "kv":     load_jsonl(os.path.join(trace_dir, "kv_trace.jsonl")),
        "expert": load_jsonl(os.path.join(trace_dir, "expert_trace.jsonl")),
        "memory": load_jsonl(os.path.join(trace_dir, "memory_trace.jsonl")),
    }
    run_id = Path(trace_dir).resolve().name
    manifest_path = Path(trace_dir) / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest.get("run_name") or run_id)
    except (OSError, json.JSONDecodeError):
        pass
    for record in data["memory"]:
        record.setdefault("run_id", run_id)
    return data


def load_key_tensor_events(path: str) -> list[dict]:
    """Fast streaming pass to extract tensor load / first-touch / residency events."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            # Fast string check before JSON parsing
            if '"first_touch":true' not in line and '"TENSOR_LOAD"' not in line and '"page_count"' not in line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("event") == "TENSOR_LOAD" or r.get("first_touch") is True or "page_count" in r:
                    records.append(r)
            except json.JSONDecodeError:
                continue
    return records


# ─────────────────────────────────────────────
#  Data Enrichment
# ─────────────────────────────────────────────

def phase_color(phase: str) -> str:
    return "#2196F3" if phase == "PREFILL" else "#FF9800" if phase == "DECODE" else "#9E9E9E"


def add_relative_time(records: list[dict]) -> list[dict]:
    """Add relative time in ms from first event."""
    if not records:
        return records
    t0 = min(r.get("ts_ns", 0) for r in records if "ts_ns" in r)
    for r in records:
        r["t_ms"] = (r.get("ts_ns", 0) - t0) / 1e6
    return records


def has_residency(record: dict) -> bool:
    return "page_count" in record and "resident_pages" in record and record.get("page_count", 0) > 0


def residency_pct(record: dict) -> float:
    if not has_residency(record):
        return float("nan")
    return record.get("resident_pages", 0) / max(1, record.get("page_count", 0)) * 100


def residency_weighted_pct(records: list[dict]) -> float:
    pages = sum(r.get("page_count", 0) for r in records if has_residency(r))
    resident = sum(r.get("resident_pages", 0) for r in records if has_residency(r))
    return resident / max(1, pages) * 100


def residency_nonresident_bytes(records: list[dict]) -> int:
    total = 0
    for r in records:
        if not has_residency(r):
            continue
        page_size = r.get("page_size", 4096)
        missing_pages = max(0, r.get("page_count", 0) - r.get("resident_pages", 0))
        total += missing_pages * page_size
    return total


def is_os_hint_syscall(record: dict) -> bool:
    action = str(record.get("action", ""))
    decision = str(record.get("decision", ""))
    if decision in {"hit", "skip", "keep"} or action.startswith("expert_cache_"):
        return False
    return "madvise" in action or "fadvise" in action


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


EXPERT_TOPK_MAX = env_int("LLM_MEM_TRACE_EXPERT_TOPK_MAX", 16)
EXPERT_ID_MAX = env_int("LLM_MEM_TRACE_MAX_EXPERT_ID", 255)


def valid_experts(record: dict) -> list[int]:
    experts = record.get("experts", [])
    if not isinstance(experts, list) or not experts:
        return []
    if len(experts) > EXPERT_TOPK_MAX:
        return []
    out: list[int] = []
    for eid in experts:
        if not isinstance(eid, int) or eid < 0 or eid > EXPERT_ID_MAX:
            return []
        out.append(eid)
    return out


def kv_append_key(record: dict) -> tuple:
    return (
        record.get("phase"),
        record.get("step"),
        record.get("layer"),
        record.get("kind"),
        record.get("kv_addr"),
        tuple(record.get("token_ids", [])),
        record.get("kv_bytes", 0),
    )


def dedupe_kv_appends(records: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for record in records:
        key = kv_append_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


# ─────────────────────────────────────────────
#  Plot 1: Memory Timeline (RSS / VMS / Page Faults)
# ─────────────────────────────────────────────

def plot_memory_timeline(memory_records: list[dict], out_dir: str):
    if not memory_records:
        print("  [SKIP] No memory events")
        return None

    mem = [r for r in memory_records if r.get("event") == "MEMORY_STAT" and "rss_bytes" in r]
    if not mem:
        print("  [SKIP] No MEMORY_STAT events with RSS data")
        return None

    mem = add_relative_time(mem)
    df = pd.DataFrame(mem)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Physical Memory Timeline During LLM Inference", fontsize=14, fontweight="bold")

    # --- RSS & VMS ---
    ax = axes[0][0]
    rss_gb = df["rss_bytes"] / (1024**3)
    vms_gb = df["vms_bytes"] / (1024**3)

    # Color by phase
    colors = [phase_color(p) for p in df.get("phase", ["UNKNOWN"] * len(df))]
    ax.scatter(df["t_ms"], rss_gb, c=colors, s=20, alpha=0.7, edgecolors="none")
    ax.plot(df["t_ms"], rss_gb, alpha=0.3, linewidth=1, color="#2196F3")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("RSS (GB)")
    ax.set_title("Resident Set Size (Physical Memory)")
    ax.grid(True, alpha=0.3)

    # Add phase transition line if exists
    phases_phase = df["phase"].tolist()
    for i in range(1, len(phases_phase)):
        if phases_phase[i] != phases_phase[i - 1]:
            ax.axvline(x=df["t_ms"].iloc[i], color="red", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.text(df["t_ms"].iloc[i], ax.get_ylim()[1] * 0.95, phases_phase[i],
                    fontsize=8, color="red", ha="left", rotation=90)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", markersize=8, label="PREFILL"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9800", markersize=8, label="DECODE"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(df["t_ms"], vms_gb, color="purple", alpha=0.5, linewidth=1, linestyle="--")
    ax2.set_ylabel("VMS (GB)", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    # --- Page Faults (minor) ---
    ax = axes[0][1]
    if "minor_faults_delta" in df.columns:
        ax.bar(df["t_ms"], df["minor_faults_delta"], width=max(1, (df["t_ms"].max() - df["t_ms"].min()) / max(1, len(df)) * 0.8),
               color=["#2196F3" if p == "PREFILL" else "#FF9800" for p in df.get("phase", [])],
               alpha=0.7, edgecolor="none")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Minor Page Faults (delta)")
    ax.set_title("Minor Page Fault Burst Analysis (Demand Paging)")
    ax.grid(True, alpha=0.3)

    # --- Minor Faults Cumulative ---
    ax = axes[1][0]
    if "minor_faults" in df.columns:
        ax.plot(df["t_ms"], df["minor_faults"] / 1000, color="#2196F3", linewidth=1.5)
        # Mark phase transitions
        for i in range(1, len(phases_phase)):
            if phases_phase[i] != phases_phase[i - 1]:
                ax.axvline(x=df["t_ms"].iloc[i], color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Cumulative Minor Faults (K)")
    ax.set_title("Cumulative Page Faults — mmap Lazy Loading Indicator")
    ax.grid(True, alpha=0.3)

    # --- mmap Count ---
    ax = axes[1][1]
    if "mmap_count" in df.columns:
        ax.plot(df["t_ms"], df["mmap_count"], color="green", linewidth=1.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("mmap Regions")
    ax.set_title("Memory Mapping Regions Over Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "01_memory_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


# ─────────────────────────────────────────────
#  Plot 2: KV Cache Growth
# ─────────────────────────────────────────────

def plot_kv_cache(kv_records: list[dict], out_dir: str):
    appends_raw = [r for r in kv_records if r.get("event") == "KV_APPEND"]
    appends = dedupe_kv_appends(appends_raw)
    reuses = [r for r in kv_records if r.get("event") == "KV_REUSE"]

    if not appends and not reuses:
        print("  [SKIP] No KV cache events")
        return None
    if len(appends) != len(appends_raw):
        print(f"  [INFO] KV append dedupe: {len(appends):,} unique from {len(appends_raw):,} events")

    appends = add_relative_time(appends)
    reuses = add_relative_time(reuses)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("KV Cache Behavior Analysis", fontsize=14, fontweight="bold")

    # --- KV bytes over time (by layer) ---
    ax = axes[0]
    if appends:
        df = pd.DataFrame(appends)
        layers = sorted(set(r.get("layer", -1) for r in appends if r.get("layer", -1) >= 0))
        if not layers:
            layers = [-1]
        cmap = plt.cm.viridis
        for i, layer in enumerate(layers[:20]):  # max 20 layers
            ldf = df[df["layer"] == layer].sort_values("t_ms")
            if len(ldf) == 0:
                continue
            color = cmap(i / max(1, len(layers) - 1))
            ax.plot(ldf["t_ms"], ldf["kv_bytes"].cumsum() / (1024**2), linewidth=1, color=color, alpha=0.7,
                    label=f"L{layer}" if i % 3 == 0 else "")

        total_kv = df.groupby("t_ms")["kv_bytes"].sum().cumsum() / (1024**2)
        ax.plot(total_kv.index, total_kv.values, color="red", linewidth=2, label="Total KV (MB)")
        ax.legend(fontsize=7, loc="upper left")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("KV Size (MB)")
    ax.set_title("KV Cache Growth per Layer")
    ax.grid(True, alpha=0.3)

    # --- Context length vs time ---
    ax = axes[1]
    if appends:
        df = pd.DataFrame(appends)
        ctx_per_layer = df.groupby("t_ms")["ctx_len"].max()
        ax.plot(ctx_per_layer.index, ctx_per_layer.values, color="#2196F3", linewidth=1.5)
        # Color prefill vs decode
        for r in appends:
            ax.axvline(x=r["t_ms"], color=phase_color(r.get("phase", "UNKNOWN")), alpha=0.1, linewidth=0.5)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Context Length (tokens)")
    ax.set_title("Context Length Growth")
    ax.grid(True, alpha=0.3)

    # --- KV Reuse stats ---
    ax = axes[2]
    if reuses:
        df = pd.DataFrame(reuses)
        reuse_rate = df["reused"] / df["n_tokens"].replace(0, 1) * 100
        ax.bar(range(len(reuse_rate)), reuse_rate.values,
               color=["#2196F3" if p == "PREFILL" else "#FF9800" for p in df.get("phase", [])],
               alpha=0.7)
        ax.axhline(y=reuse_rate.mean(), color="red", linestyle="--", linewidth=1,
                   label=f"Avg: {reuse_rate.mean():.1f}%")
        ax.legend()

    ax.set_xlabel("Event #")
    ax.set_ylabel("KV Reuse Rate (%)")
    ax.set_title("KV Cache Slot Reuse Rate")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "02_kv_cache.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


# ─────────────────────────────────────────────
#  Plot 3: Expert Activation Heatmap
# ─────────────────────────────────────────────

def plot_expert_activation(expert_records: list[dict], out_dir: str):
    routes = [r for r in expert_records if r.get("event") == "EXPERT_ROUTE"]
    if not routes:
        print("  [SKIP] No expert route events")
        return None

    # Build expert frequency per layer
    # Structure: {layer: {expert_id: count}}
    layer_expert_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    layer_token_counts: dict[int, int] = defaultdict(int)

    for r in routes:
        layer = r.get("layer", -1)
        if layer < 0:
            continue
        experts = valid_experts(r)
        if not experts:
            continue
        layer_token_counts[layer] += 1
        for expert in experts:
            layer_expert_counts[layer][expert] += 1

    if not layer_expert_counts:
        print("  [SKIP] No layer-expert data")
        return None

    layers = sorted(layer_expert_counts.keys())

    # Find all expert IDs
    all_experts = set()
    for lc in layer_expert_counts.values():
        all_experts.update(lc.keys())
    if not all_experts:
        return None

    max_expert = max(all_experts)
    n_experts = min(max_expert + 1, EXPERT_ID_MAX + 1)

    # Build heatmap matrix: rows=layers, cols=experts
    matrix = np.zeros((len(layers), n_experts))
    for i, layer in enumerate(layers):
        total = layer_token_counts[layer]
        for eid in range(n_experts):
            count = layer_expert_counts[layer].get(eid, 0)
            matrix[i, eid] = count / max(1, total) * 100  # percentage

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(layers) * 0.3)))
    fig.suptitle("MoE Expert Activation Analysis", fontsize=14, fontweight="bold")

    # --- Heatmap ---
    ax = axes[0]
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Layer")
    ax.set_title("Expert Activation Frequency per Layer (%)")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_xticks(range(0, n_experts, max(1, n_experts // 16)))
    plt.colorbar(im, ax=ax, label="Activation %")

    # --- Top experts bar chart ---
    ax = axes[1]
    # Aggregate across all layers
    global_counts: dict[int, int] = defaultdict(int)
    for r in routes:
        for eid in valid_experts(r):
            global_counts[eid] += 1

    top_n = 20
    sorted_experts = sorted(global_counts.items(), key=lambda x: -x[1])[:top_n]
    expert_ids = [e[0] for e in sorted_experts]
    expert_freqs = [e[1] for e in sorted_experts]

    bars = ax.bar(range(len(expert_ids)), expert_freqs, color=plt.cm.YlOrRd(
        np.array(expert_freqs) / max(expert_freqs)))
    ax.set_xticks(range(len(expert_ids)))
    ax.set_xticklabels([f"E{e}" for e in expert_ids], rotation=45, fontsize=8)
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Total Activations")
    ax.set_title(f"Top {top_n} Most Activated Experts (Global)")
    ax.grid(True, alpha=0.3, axis="y")

    # Add count labels on bars
    for bar, freq in zip(bars, expert_freqs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(expert_freqs) * 0.01,
                str(freq), ha="center", fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, "03_expert_activation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")
    return path, sorted_experts, global_counts


# ─────────────────────────────────────────────
#  Plot 4: Tensor Access Locality
# ─────────────────────────────────────────────

def plot_tensor_access(tensor_records: list[dict], out_dir: str):
    accesses = [r for r in tensor_records if r.get("event") == "TENSOR_ACCESS"]
    loads = [r for r in tensor_records if r.get("event") == "TENSOR_LOAD"]

    if not accesses and not loads:
        print("  [SKIP] No tensor events")
        return None

    accesses = add_relative_time(accesses)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Tensor Memory Access Analysis", fontsize=14, fontweight="bold")

    # --- First-touch timeline ---
    ax = axes[0][0]
    first_touches = [r for r in accesses if r.get("first_touch") is True]
    if first_touches:
        ft = add_relative_time(first_touches)
        df_ft = pd.DataFrame(ft)
        sizes_gb = df_ft["size"] / (1024**3)
        colors = [phase_color(p) for p in df_ft.get("phase", [])]
        ax.scatter(df_ft["t_ms"], sizes_gb, c=colors, s=30, alpha=0.6, edgecolors="none")
        ax.set_ylabel("Tensor Size (GB)")

        # Mark prefill/decode boundary
        phases_list = df_ft["phase"].tolist()
        for i in range(1, len(phases_list)):
            if phases_list[i] != phases_list[i - 1]:
                ax.axvline(x=df_ft["t_ms"].iloc[i], color="red", linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (ms)")
    ax.set_title("First-Touch Timeline (mmap Lazy Loading)")
    ax.grid(True, alpha=0.3)

    # --- Access frequency by layer ---
    ax = axes[0][1]
    layer_freq: dict[int, int] = defaultdict(int)
    for r in accesses:
        layer = r.get("layer", -1)
        if layer >= 0:
            layer_freq[layer] += 1

    if layer_freq:
        layers = sorted(layer_freq.keys())
        counts = [layer_freq[l] for l in layers]
        colors = plt.cm.viridis(np.array(counts) / max(counts))
        ax.bar(layers, counts, color=colors)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Access Count")
        ax.set_title("Tensor Access Frequency per Layer")

    ax.grid(True, alpha=0.3, axis="y")

    # --- Size distribution ---
    ax = axes[1][0]
    sizes = [r.get("size", 0) for r in accesses if r.get("size", 0) > 0]
    if sizes:
        sizes_mb = np.array(sizes) / (1024**2)
        # Log-scale histogram
        log_sizes = np.log10(np.clip(sizes_mb, 0.001, None))
        ax.hist(log_sizes, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
        ax.set_xlabel("log10(Size in MB)")
        ax.set_ylabel("Count")
        ax.set_title("Tensor Size Distribution (log scale)")
        ax.grid(True, alpha=0.3)

    # --- Backend distribution ---
    ax = axes[1][1]
    backend_counts: dict[str, int] = defaultdict(int)
    for r in accesses:
        backend_counts[r.get("backend", "unknown")] += 1

    if backend_counts:
        backends = sorted(backend_counts.keys(), key=lambda b: -backend_counts[b])
        counts = [backend_counts[b] for b in backends]
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"][:len(backends)]
        wedges, texts, autotexts = ax.pie(counts, labels=backends, autopct="%1.1f%%",
                                           colors=colors, startangle=90)
        for at in autotexts:
            at.set_fontsize(8)
        ax.set_title("Tensor Access by Backend")

    plt.tight_layout()
    path = os.path.join(out_dir, "04_tensor_access.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")

    return path, layer_freq, backend_counts


# ─────────────────────────────────────────────
#  Plot 5: Token Latency & Prefill vs Decode
# ─────────────────────────────────────────────

def plot_token_latency(memory_records: list[dict], out_dir: str):
    latency_records, latency_source = inference_latency_records(memory_records)
    if not latency_records:
        print("  [SKIP] No inference latency data")
        return None

    latency_records = add_relative_time(latency_records)

    prefill_tokens = [r for r in latency_records if r.get("phase") == "PREFILL"]
    decode_tokens = [r for r in latency_records if r.get("phase") == "DECODE"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Inference Step Latency Analysis ({latency_source})", fontsize=14, fontweight="bold")

    # --- Latency timeline ---
    ax = axes[0]
    df = pd.DataFrame(latency_records)
    latency_ms = df["latency_ns"] / 1e6
    colors = [phase_color(p) for p in df.get("phase", [])]
    ax.scatter(df["t_ms"], latency_ms, c=colors, s=15, alpha=0.6, edgecolors="none")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Ubatch Latency Timeline")
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", markersize=8, label="PREFILL"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9800", markersize=8, label="DECODE"),
    ])

    # --- Prefill vs Decode latency distribution ---
    ax = axes[1]
    plot_data = []
    labels = []
    if prefill_tokens:
        pf_lat = [t["latency_ns"] / 1e3 for t in prefill_tokens]  # us
        plot_data.append(pf_lat)
        labels.append(f"PREFILL\n(n={len(pf_lat)}, avg={np.mean(pf_lat):.0f}μs)")
    if decode_tokens:
        dc_lat = [t["latency_ns"] / 1e3 for t in decode_tokens]  # us
        plot_data.append(dc_lat)
        labels.append(f"DECODE\n(n={len(dc_lat)}, avg={np.mean(dc_lat):.0f}μs)")

    if plot_data:
        bp = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True)
        colors_box = ["#2196F3", "#FF9800"][:len(plot_data)]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax.set_ylabel("Latency (μs)")
    ax.set_title("Ubatch Latency: Prefill vs Decode")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Token latency by position ---
    ax = axes[2]
    pos_lat = defaultdict(list)
    for r in latency_records:
        pos = r.get("pos")
        if pos is not None:
            pos_lat[pos].append(r["latency_ns"] / 1e3)  # us

    if pos_lat:
        positions = sorted(pos_lat.keys())
        avg_lat = [np.mean(pos_lat[p]) for p in positions]
        colors_pos = ["#2196F3" if p < positions[len(positions) // 2] else "#FF9800" for p in positions]
        ax.bar(positions[:50], avg_lat[:50], color=colors_pos[:50], alpha=0.7)
        ax.set_xlabel("Position (legacy traces only)")
        ax.set_ylabel("Avg Latency (μs)")
        ax.set_title("Latency by Token Position")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "05_token_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")

    # Return summary stats
    pf_mean = np.mean([t["latency_ns"] / 1e3 for t in prefill_tokens]) if prefill_tokens else 0
    dc_mean = np.mean([t["latency_ns"] / 1e3 for t in decode_tokens]) if decode_tokens else 0
    return path, pf_mean, dc_mean


# ─────────────────────────────────────────────
#  Plot 6: Summary Dashboard
# ─────────────────────────────────────────────

def plot_summary_dashboard(data: dict[str, list[dict]], out_dir: str):
    """Create a summary dashboard combining key metrics."""
    mem_events = [r for r in data["memory"] if r.get("event") == "MEMORY_STAT" and "rss_bytes" in r]
    kv_appends = dedupe_kv_appends([r for r in data["kv"] if r.get("event") == "KV_APPEND"])
    expert_routes = [r for r in data["expert"] if r.get("event") == "EXPERT_ROUTE"]
    tensor_accesses = [r for r in data["tensor"] if r.get("event") == "TENSOR_ACCESS"]
    latency_records, latency_source = inference_latency_records(data["memory"])
    first_touches = [r for r in tensor_accesses if r.get("first_touch") is True]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("LLM Memory Behavior Summary Dashboard", fontsize=16, fontweight="bold")

    # 1. RSS peak
    ax = axes[0][0]
    if mem_events:
        mem = add_relative_time(mem_events)
        df = pd.DataFrame(mem)
        ax.fill_between(df["t_ms"], 0, df["rss_bytes"] / (1024**3), color="#2196F3", alpha=0.3)
        ax.plot(df["t_ms"], df["rss_bytes"] / (1024**3), color="#2196F3", linewidth=1.5)
        ax.set_title(f"RSS Timeline (Peak: {df['rss_bytes'].max() / (1024**3):.2f} GB)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("RSS (GB)")
    ax.grid(True, alpha=0.3)

    # 2. KV Cache total
    ax = axes[0][1]
    if kv_appends:
        kv = add_relative_time(kv_appends)
        df = pd.DataFrame(kv)
        kv_cumsum = df.groupby("t_ms")["kv_bytes"].sum().cumsum() / (1024**2)
        ax.fill_between(kv_cumsum.index, 0, kv_cumsum.values, color="#4CAF50", alpha=0.3)
        ax.plot(kv_cumsum.index, kv_cumsum.values, color="#4CAF50", linewidth=1.5)
        ax.set_title(f"KV Cache Growth (Total: {kv_cumsum.values[-1]:.1f} MB)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("KV Size (MB)")
    ax.grid(True, alpha=0.3)

    # 3. Page Faults
    ax = axes[0][2]
    if mem_events:
        df = pd.DataFrame(mem)
        if "minor_faults_delta" in df.columns:
            pf = df[df["minor_faults_delta"] > 0]
            ax.bar(pf["t_ms"], pf["minor_faults_delta"], color="#F44336", alpha=0.6, width=50)
            ax.set_title(f"Page Fault Bursts (Total: {df['minor_faults_delta'].sum():,})")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Minor Faults (delta)")
    ax.grid(True, alpha=0.3)

    # 4. Expert distribution
    ax = axes[1][0]
    if expert_routes:
        expert_counts: dict[int, int] = defaultdict(int)
        for r in expert_routes:
            for eid in valid_experts(r):
                expert_counts[eid] += 1
        top15 = sorted(expert_counts.items(), key=lambda x: -x[1])[:15]
        if top15:
            eids = [str(e[0]) for e in top15]
            freqs = [e[1] for e in top15]
            ax.bar(range(len(eids)), freqs, color=plt.cm.YlOrRd(np.array(freqs) / max(freqs)))
            ax.set_xticks(range(len(eids)))
            ax.set_xticklabels([f"E{e}" for e in eids], rotation=45, fontsize=8)
            ax.set_title(f"Top 15 Hot Experts")
    ax.set_ylabel("Activations")
    ax.grid(True, alpha=0.3, axis="y")

    # 5. First-touch memory
    ax = axes[1][1]
    if first_touches:
        ft = add_relative_time(first_touches)
        df = pd.DataFrame(ft)
        ft_cumsum = df.sort_values("t_ms")["size"].cumsum() / (1024**3)
        ax.plot(df.sort_values("t_ms")["t_ms"], ft_cumsum, color="#9C27B0", linewidth=1.5)
        ax.set_title(f"Lazy Loading: {ft_cumsum.values[-1]:.2f} GB via mmap faults")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Cumulative Size (GB)")
    ax.grid(True, alpha=0.3)

    # 6. Inference step latency summary
    ax = axes[1][2]
    prefill_lat = [t["latency_ns"] / 1e3 for t in latency_records if t.get("phase") == "PREFILL"]
    decode_lat = [t["latency_ns"] / 1e3 for t in latency_records if t.get("phase") == "DECODE"]
    labels = []
    values = []
    colors = []
    if prefill_lat:
        labels.append(f"PREFILL\n{np.mean(prefill_lat):.0f}μs avg")
        values.append(np.mean(prefill_lat))
        colors.append("#2196F3")
    if decode_lat:
        labels.append(f"DECODE\n{np.mean(decode_lat):.0f}μs avg")
        values.append(np.mean(decode_lat))
        colors.append("#FF9800")

    if values:
        bars = ax.bar(labels, values, color=colors, alpha=0.7)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    f"{val:.0f}μs", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Avg Latency (μs)")
    ax.set_title(f"Ubatch Latency: Prefill vs Decode ({latency_source})")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "06_summary_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


# ─────────────────────────────────────────────
#  Plot 7: Page Residency & PSS Analysis
# ─────────────────────────────────────────────

def plot_page_residency(data: dict[str, list[dict]], out_dir: str):
    tensor_records = data["tensor"]
    memory_records = data["memory"]
    residency_records = [r for r in tensor_records if has_residency(r)]
    mem_pss = [r for r in memory_records if r.get("event") == "MEMORY_STAT" and "pss_bytes" in r]

    if not residency_records and not mem_pss:
        print("  [SKIP] No page residency or smaps_rollup data")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Page Residency and Process Memory Analysis", fontsize=14, fontweight="bold")

    # --- First-touch residency scatter ---
    ax = axes[0][0]
    first_res = [r for r in residency_records if r.get("first_touch") is True]
    if first_res:
        ft = add_relative_time(first_res)
        df = pd.DataFrame(ft)
        pct = df.apply(residency_pct, axis=1)
        sizes_mb = df["size"] / (1024**2)
        sc = ax.scatter(df["t_ms"], sizes_mb, c=pct, cmap="RdYlGn", vmin=0, vmax=100,
                        s=35, alpha=0.75, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="Resident Pages (%)")
        ax.set_title("First-Touch Tensor Residency")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Tensor Size (MB)")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title("First-Touch Tensor Residency (no data)")
        ax.axis("off")

    # --- Residency distribution ---
    ax = axes[0][1]
    if residency_records:
        load_res = [r for r in residency_records if r.get("event") == "TENSOR_LOAD"]
        access_res = [r for r in residency_records if r.get("first_touch") is True]
        if load_res:
            ax.hist([residency_pct(r) for r in load_res], bins=20, range=(0, 100),
                    alpha=0.55, label="TENSOR_LOAD", color="#2196F3")
        if access_res:
            ax.hist([residency_pct(r) for r in access_res], bins=20, range=(0, 100),
                    alpha=0.55, label="FIRST_TOUCH", color="#FF9800")
        ax.set_title("Residency Ratio Distribution")
        ax.set_xlabel("Resident Pages (%)")
        ax.set_ylabel("Event Count")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.set_title("Residency Ratio Distribution (no data)")
        ax.axis("off")

    # --- Process memory from smaps_rollup ---
    ax = axes[1][0]
    if mem_pss:
        mem = add_relative_time(mem_pss)
        df = pd.DataFrame(mem)
        ax.plot(df["t_ms"], df["rss_bytes"] / (1024**3), label="RSS (/proc/stat)", linewidth=1.4)
        ax.plot(df["t_ms"], df["pss_bytes"] / (1024**3), label="PSS", linewidth=1.4)
        ax.plot(df["t_ms"], df["private_dirty_bytes"] / (1024**3), label="Private Dirty", linewidth=1.2)
        ax.plot(df["t_ms"], df["shared_clean_bytes"] / (1024**3), label="Shared Clean", linewidth=1.2)
        if df["swap_bytes"].max() > 0:
            ax.plot(df["t_ms"], df["swap_bytes"] / (1024**3), label="Swap", linewidth=1.2)
        ax.set_title("Process Memory Breakdown (smaps_rollup)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("GiB")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title("Process Memory Breakdown (no smaps data)")
        ax.axis("off")

    # --- Top non-resident first-touch tensors ---
    ax = axes[1][1]
    if first_res:
        by_tensor: dict[str, int] = defaultdict(int)
        for r in first_res:
            if not has_residency(r):
                continue
            missing_pages = max(0, r.get("page_count", 0) - r.get("resident_pages", 0))
            by_tensor[r.get("tensor", "<unnamed>")] += missing_pages * r.get("page_size", 4096)
        top = sorted(by_tensor.items(), key=lambda kv: -kv[1])[:12]
        if top:
            names = [name if len(name) <= 32 else name[:29] + "..." for name, _ in top]
            vals = [v / (1024**2) for _, v in top]
            y = np.arange(len(names))
            ax.barh(y, vals, color="#F44336", alpha=0.75)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.set_title("Top First-Touch Non-Resident Tensors")
            ax.set_xlabel("Estimated Non-Resident MB")
            ax.grid(True, alpha=0.3, axis="x")
        else:
            ax.set_title("Top First-Touch Non-Resident Tensors (none)")
            ax.axis("off")
    else:
        ax.set_title("Top First-Touch Non-Resident Tensors (no data)")
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "07_page_residency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


# ─────────────────────────────────────────────
#  Collect Key Metrics for Report
# ─────────────────────────────────────────────

def _stage_pair_stats(deltas_ns: list[int]) -> dict:
    paired = len(deltas_ns)
    late_after = sum(delta > 0 for delta in deltas_ns)
    late_before = sum(delta < 0 for delta in deltas_ns)
    equal = sum(delta == 0 for delta in deltas_ns)
    percentiles = {
        percentile: (float(np.percentile(deltas_ns, value)) if deltas_ns else None)
        for percentile, value in (("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))
    }
    return {
        "paired_experts": paired,
        "late_after_early_count": late_after,
        "late_before_early_count": late_before,
        "equal_timestamp_count": equal,
        "late_after_early_ratio": late_after / paired if paired else 0.0,
        "delta_ns": percentiles,
    }


def collect_expert_stage_pairing(first_use_events: list[dict]) -> dict:
    """Pair trace-provided EARLY/LATE logical first-use timestamps.

    Stage is deliberately never inferred from tensor names here. Multiple detail
    records emitted for a one-to-many Task match are collapsed back to their
    single logical first-use observation before pairing.
    """
    observations: dict[tuple, dict] = {}
    invalid_stage_records = 0
    for record in first_use_events:
        stage = record.get("stage")
        if stage not in {"EARLY", "LATE", "UNKNOWN"}:
            invalid_stage_records += 1
            continue
        observation_key = (
            str(record.get("run_id", "single_run")),
            int(record.get("step", -1)),
            int(record.get("layer", -1)),
            int(record.get("expert", -1)),
            str(record.get("tensor", "")),
            stage,
            int(record.get("first_use_ts_ns", record.get("ts_ns", 0))),
        )
        observations.setdefault(observation_key, record)

    groups: dict[tuple, dict[str, list[dict]]] = defaultdict(
        lambda: {"EARLY": [], "LATE": [], "UNKNOWN": []}
    )
    for record in observations.values():
        key = (
            str(record.get("run_id", "single_run")),
            int(record.get("step", -1)),
            int(record.get("layer", -1)),
            int(record.get("expert", -1)),
        )
        groups[key][str(record["stage"])].append(record)

    overall_deltas: list[int] = []
    phase_deltas: dict[str, list[int]] = {"PREFILL": [], "DECODE": []}
    layer_deltas: dict[int, list[int]] = defaultdict(list)
    unmatched_reasons: Counter[str] = Counter()
    multiple_early_groups = 0
    multiple_late_groups = 0
    phase_mismatch_pairs = 0

    for key, by_stage in groups.items():
        early = by_stage["EARLY"]
        late = by_stage["LATE"]
        if not early and not late:
            unmatched_reasons["unknown_stage_only"] += 1
            continue
        if not early:
            unmatched_reasons["missing_early"] += 1
            continue
        if not late:
            unmatched_reasons["missing_late"] += 1
            continue

        multiple_early_groups += int(len(early) > 1)
        multiple_late_groups += int(len(late) > 1)
        early_record = min(early, key=lambda item: int(item.get("first_use_ts_ns", item.get("ts_ns", 0))))
        late_record = min(late, key=lambda item: int(item.get("first_use_ts_ns", item.get("ts_ns", 0))))
        early_ts = int(early_record.get("first_use_ts_ns", early_record.get("ts_ns", 0)))
        late_ts = int(late_record.get("first_use_ts_ns", late_record.get("ts_ns", 0)))
        delta = late_ts - early_ts
        overall_deltas.append(delta)
        layer_deltas[int(key[2])].append(delta)

        early_phase = str(early_record.get("phase", "UNKNOWN"))
        late_phase = str(late_record.get("phase", "UNKNOWN"))
        if early_phase == late_phase and early_phase in phase_deltas:
            phase_deltas[early_phase].append(delta)
        else:
            phase_mismatch_pairs += 1

    result = _stage_pair_stats(overall_deltas)
    result["pairing_key"] = ["run_id", "step", "layer", "expert"]
    result["stage_source"] = "trace_field"
    result["by_phase"] = {
        phase: _stage_pair_stats(phase_deltas[phase]) for phase in ("PREFILL", "DECODE")
    }
    result["by_layer"] = {
        str(layer): _stage_pair_stats(layer_deltas[layer]) for layer in sorted(layer_deltas)
    }
    result["unmatched_reasons"] = dict(sorted(unmatched_reasons.items()))
    result["invalid_stage_records"] = invalid_stage_records
    result["multiple_early_observation_groups"] = multiple_early_groups
    result["multiple_late_observation_groups"] = multiple_late_groups
    result["phase_mismatch_pairs"] = phase_mismatch_pairs
    result["logical_first_use_records"] = len(observations)
    result["duplicate_match_records_collapsed"] = len(first_use_events) - invalid_stage_records - len(observations)
    return result

def collect_metrics(data: dict[str, list[dict]]) -> dict:
    """Extract key numeric metrics for the report."""
    metrics = collect_core_metrics(data["memory"])

    # KV Cache
    kv_appends = [r for r in data["kv"] if r.get("event") == "KV_APPEND"]
    if kv_appends:
        kv_unique = dedupe_kv_appends(kv_appends)
        kv_bytes_total = sum(r.get("kv_bytes", 0) for r in kv_unique)
        metrics["kv_event_bytes_mb"] = sum(r.get("kv_bytes", 0) for r in kv_appends) / (1024**2)
        metrics["kv_total_mb"] = kv_bytes_total / (1024**2)
        metrics["kv_append_events"] = len(kv_appends)
        metrics["kv_append_events_unique"] = len(kv_unique)
        metrics["kv_append_duplicate_events"] = len(kv_appends) - len(kv_unique)
        kv_layers = set(r.get("layer", -1) for r in kv_unique)
        metrics["kv_layers"] = len(kv_layers)
        metrics["kv_ctx_len_capacity"] = max(r.get("ctx_len", 0) for r in kv_unique)

        kv_by_kind: dict[str, int] = defaultdict(int)
        kv_by_phase: dict[str, int] = defaultdict(int)
        kv_by_layer: dict[int, int] = defaultdict(int)
        tokens_by_step: dict[tuple[str, int], int] = defaultdict(int)
        decode_tokens_by_step: dict[tuple[str, int], int] = defaultdict(int)
        for r in kv_unique:
            kv_bytes = int(r.get("kv_bytes", 0))
            kind = str(r.get("kind", "unknown")).lower()
            phase = str(r.get("phase", "UNKNOWN"))
            layer = int(r.get("layer", -1))
            step = int(r.get("step", 0))
            n_tokens = int(r.get("n_tokens", 0))
            kv_by_kind[kind] += kv_bytes
            kv_by_phase[phase] += kv_bytes
            if layer >= 0:
                kv_by_layer[layer] += kv_bytes
            tokens_by_step[(phase, step)] = max(tokens_by_step[(phase, step)], n_tokens)
            if phase == "DECODE":
                decode_tokens_by_step[(phase, step)] = max(decode_tokens_by_step[(phase, step)], n_tokens)

        metrics["kv_k_mb"] = kv_by_kind.get("k", 0) / (1024**2)
        metrics["kv_v_mb"] = kv_by_kind.get("v", 0) / (1024**2)
        metrics["kv_prefill_mb"] = kv_by_phase.get("PREFILL", 0) / (1024**2)
        metrics["kv_decode_mb"] = kv_by_phase.get("DECODE", 0) / (1024**2)
        tokens_appended = sum(tokens_by_step.values())
        decode_tokens_appended = sum(decode_tokens_by_step.values())
        metrics["kv_tokens_appended_est"] = tokens_appended
        if tokens_appended > 0:
            bytes_per_token = kv_bytes_total / tokens_appended
            metrics["kv_bytes_per_token_est"] = bytes_per_token
            metrics["kv_mb_per_1k_tokens_est"] = bytes_per_token * 1000 / (1024**2)
            for ctx in (2048, 4096, 8192):
                metrics[f"kv_projected_{ctx}_mb"] = bytes_per_token * ctx / (1024**2)
        if decode_tokens_appended > 0:
            metrics["kv_decode_bytes_per_token_est"] = kv_by_phase.get("DECODE", 0) / decode_tokens_appended
        if kv_by_layer:
            layer_values = list(kv_by_layer.values())
            metrics["kv_layer_avg_mb"] = float(np.mean(layer_values)) / (1024**2)
            metrics["kv_layer_min_mb"] = min(layer_values) / (1024**2)
            metrics["kv_layer_max_mb"] = max(layer_values) / (1024**2)
            metrics["kv_layer_imbalance_ratio"] = max(layer_values) / max(1, min(layer_values))

    kv_reuses = [r for r in data["kv"] if r.get("event") == "KV_REUSE"]
    if kv_reuses:
        total_n = sum(r.get("n_tokens", 0) for r in kv_reuses)
        total_r = sum(r.get("reused", 0) for r in kv_reuses)
        metrics["kv_reuse_rate_pct"] = total_r / max(1, total_n) * 100

    # Experts — IDs are per-layer (0-255 per MoE layer for Qwen3.5-A3B).
    # For the report, count unique (layer, expert_id) pairs and also
    # aggregate globally across all layers for frequency analysis.
    expert_routes = [r for r in data["expert"] if r.get("event") == "EXPERT_ROUTE"]
    if expert_routes:
        metrics["expert_route_events"] = len(expert_routes)

        # Per-layer unique expert count
        layer_experts: dict[int, set[int]] = defaultdict(set)
        # Global expert activation count (expert ID only, across all layers)
        expert_counts: dict[int, int] = defaultdict(int)
        valid_route_events = 0
        for r in expert_routes:
            layer = r.get("layer", -1)
            experts = valid_experts(r)
            if not experts:
                continue
            valid_route_events += 1
            for eid in experts:
                if layer >= 0:
                    layer_experts[layer].add(eid)
                expert_counts[eid] += 1

        total_unique = sum(len(s) for s in layer_experts.values())
        metrics["expert_route_events_valid"] = valid_route_events
        metrics["expert_route_events_skipped"] = len(expert_routes) - valid_route_events
        metrics["unique_experts_per_layer"] = total_unique
        metrics["expert_layers"] = len(layer_experts)
        # Global: how many distinct expert IDs appear across all layers
        metrics["unique_experts_global"] = len(expert_counts)
        metrics["max_expert_id"] = max(expert_counts.keys()) if expert_counts else 0

    # Tensor
    tensor_accesses = [r for r in data["tensor"] if r.get("event") == "TENSOR_ACCESS"]
    if tensor_accesses:
        metrics["tensor_access_events_sampled"] = len(tensor_accesses)
        first_touches = [r for r in tensor_accesses if r.get("first_touch") is True]
        metrics["first_touch_events"] = len(first_touches)
        metrics["first_touch_gb"] = sum(r.get("size", 0) for r in first_touches) / (1024**3)

    tensor_residency = [r for r in data["tensor"] if has_residency(r)]
    if tensor_residency:
        first_touch_residency = [r for r in tensor_residency if r.get("first_touch") is True]
        load_residency = [r for r in tensor_residency if r.get("event") == "TENSOR_LOAD"]
        mmap_load_residency = [r for r in load_residency if r.get("stage") == "mmap"]
        metrics["tensor_residency_events"] = len(tensor_residency)
        metrics["tensor_residency_exact_events"] = sum(1 for r in tensor_residency if r.get("resident_exact") is True)
        metrics["first_touch_residency_events"] = len(first_touch_residency)
        metrics["first_touch_resident_pct_weighted"] = residency_weighted_pct(first_touch_residency)
        metrics["first_touch_nonresident_gb_est"] = residency_nonresident_bytes(first_touch_residency) / (1024**3)
        metrics["tensor_load_resident_pct_weighted"] = residency_weighted_pct(load_residency)
        metrics["mmap_load_resident_pct_weighted"] = residency_weighted_pct(mmap_load_residency)

    os_hints = [r for r in data["memory"] if r.get("event") == "OS_HINT"]
    if os_hints:
        os_hint_syscalls = [r for r in os_hints if is_os_hint_syscall(r)]
        metrics["os_hint_records"] = len(os_hints)
        metrics["os_hint_events"] = len(os_hint_syscalls)
        metrics["os_hint_errors"] = sum(1 for r in os_hint_syscalls if r.get("result", 0) != 0)
        metrics["os_hint_advised_mb"] = sum(r.get("advised_bytes", 0) for r in os_hint_syscalls) / (1024**2)
        metrics["expert_cache_hits"] = sum(1 for r in os_hints if r.get("decision") == "hit" or r.get("cache_hit") is True)
        metrics["expert_cache_prefetches"] = sum(1 for r in os_hints if r.get("decision") == "prefetch")
        metrics["expert_cache_evictions"] = sum(1 for r in os_hints if r.get("decision") == "evict")
        metrics["expert_cache_skips"] = sum(1 for r in os_hints if r.get("decision") == "skip")
        cache_decisions = metrics["expert_cache_hits"] + metrics["expert_cache_prefetches"]
        if cache_decisions > 0:
            metrics["expert_cache_hit_rate_pct"] = metrics["expert_cache_hits"] / cache_decisions * 100
        cache_bytes = [r.get("cache_bytes", 0) for r in os_hints if "cache_bytes" in r]
        cache_capacity = [r.get("cache_capacity_bytes", 0) for r in os_hints if "cache_capacity_bytes" in r]
        if cache_bytes:
            metrics["expert_cache_peak_mb"] = max(cache_bytes) / (1024**2)
        if cache_capacity:
            metrics["expert_cache_capacity_mb"] = max(cache_capacity) / (1024**2)
        for action, count in Counter(r.get("action", "unknown") for r in os_hints).items():
            key = "os_hint_" + "".join(ch if ch.isalnum() else "_" for ch in action.lower()) + "_events"
            metrics[key] = count
        for policy, count in Counter(r.get("policy", "none") for r in os_hints if r.get("policy")).items():
            key = "expert_cache_policy_" + "".join(ch if ch.isalnum() else "_" for ch in policy.lower()) + "_events"
            metrics[key] = count
        controlled_hints = [r for r in os_hints if "route_confidence" in r]
        predicted_hints = [r for r in controlled_hints if r.get("predicted") is True]
        metrics["expert_controller_records"] = len(controlled_hints)
        metrics["expert_predicted_records"] = len(predicted_hints)
        metrics["expert_predicted_prefetches"] = sum(
            1 for r in predicted_hints if r.get("decision") == "prefetch"
        )
        metrics["expert_predicted_skips"] = sum(
            1 for r in predicted_hints if r.get("decision") == "skip"
        )
        value_ratios = [float(r["value_ratio"]) for r in controlled_hints if isinstance(r.get("value_ratio"), (int, float))]
        if value_ratios:
            metrics["expert_controller_value_ratio_avg"] = float(np.mean(value_ratios))
        slack_values = [int(r["slack_ns"]) for r in controlled_hints if isinstance(r.get("slack_ns"), (int, float))]
        if slack_values:
            metrics["expert_controller_slack_p50_us"] = float(np.percentile(slack_values, 50)) / 1e3
        controller_cancel_actions = {
            "expert_prefetch_cancel_expired",
            "expert_prefetch_cancel_pressure",
            "expert_prefetch_cancel_value",
            "expert_prefetch_cancel_queue_full",
            "expert_prefetch_skip_pressure",
            "expert_prefetch_skip_value",
        }
        metrics["expert_controller_cancelled_total"] = sum(
            1 for r in controlled_hints if r.get("action") in controller_cancel_actions
        )

    task_events = [r for r in data["memory"] if r.get("event") == "EXPERT_TASK"]
    task_summaries = [r for r in data["memory"] if r.get("event") == "EXPERT_TASK_SUMMARY"]
    if task_summaries:
        metrics["expert_task_summary_events"] = len(task_summaries)
        task_trace_modes = sorted({
            str(r.get("trace_mode", "unknown")) for r in task_summaries
        })
        metrics["expert_task_trace_mode"] = ",".join(task_trace_modes)
        metrics["expert_task_detail_events_enabled"] = 1 if any(
            r.get("detail_events_enabled") is True for r in task_summaries
        ) else 0
        for field in (
            "created",
            "admitted",
            "rejected",
            "enqueued",
            "dequeued",
            "issued",
            "cancelled",
            "terminal",
            "in_flight",
            "invalid_transitions",
            "rejected_pressure",
            "rejected_value",
            "cancelled_pressure",
            "cancelled_value",
            "cancelled_expired",
            "cancelled_queue_full",
            "issue_groups",
            "coalesced_issue_groups",
            "same_stage_issue_groups",
            "cross_stage_issue_groups",
            "early_task_count",
            "late_task_count",
            "unknown_task_count",
        ):
            metrics[f"expert_task_{field}"] = sum(int(r.get(field, 0)) for r in task_summaries)
        metrics["expert_task_queue_wait_ns_by_stage"] = {
            stage: {
                field: sum(
                    int(r.get("queue_wait_ns_by_stage", {}).get(stage, {}).get(field, 0))
                    for r in task_summaries
                )
                for field in ("count", "total_ns", "min_ns", "max_ns")
            }
            for stage in ("EARLY", "LATE", "UNKNOWN")
        }
        metrics["expert_controller_cancelled_total"] = max(
            int(metrics.get("expert_controller_cancelled_total", 0)),
            metrics["expert_task_rejected"] + metrics["expert_task_cancelled"],
        )

    if task_events:
        metrics["expert_task_detail_records"] = len(task_events)
        event_counts = Counter(str(r.get("lifecycle_event", "UNKNOWN")) for r in task_events)
        for event, count in event_counts.items():
            key = "".join(ch if ch.isalnum() else "_" for ch in event.lower())
            metrics[f"expert_task_trace_{key}"] = count
        for reason, count in Counter(
            str(r.get("reason")) for r in task_events if r.get("reason")
        ).items():
            key = "".join(ch if ch.isalnum() else "_" for ch in reason.lower())
            metrics[f"expert_task_reason_{key}"] = count

        required_fields = {
            "task_id", "step", "layer", "expert", "phase", "stage",
            "tensor", "addr", "nbytes", "score", "sequence", "deadline_ts_ns",
            "queue_wait_ns",
        }
        metrics["expert_task_records_missing_fields"] = sum(
            1 for r in task_events if not required_fields.issubset(r)
        )

        tasks: dict[int, list[dict]] = defaultdict(list)
        invalid_task_id_records = 0
        for record in task_events:
            task_id = record.get("task_id")
            if not isinstance(task_id, int) or task_id <= 0:
                invalid_task_id_records += 1
                continue
            tasks[task_id].append(record)
        metrics["expert_task_invalid_id_records"] = invalid_task_id_records
        metrics["expert_task_unique_ids"] = len(tasks)
        metrics["expert_task_duplicate_create_ids"] = sum(
            max(0, sum(1 for r in records if r.get("lifecycle_event") == "CREATE") - 1)
            for records in tasks.values()
        )
        metrics["expert_task_invalid_stage_records"] = sum(
            1 for record in task_events
            if record.get("stage") not in {"EARLY", "LATE", "UNKNOWN"}
        )
        metrics["expert_task_stage_changes"] = sum(
            int(len({record.get("stage") for record in records}) > 1)
            for records in tasks.values()
        )

        transitions = {
            None: {"CREATE": "CREATED"},
            "CREATED": {"ADMIT": "ADMITTED", "REJECT": "REJECTED"},
            "ADMITTED": {"ENQUEUE": "ENQUEUED", "ISSUE": "ISSUED", "CANCEL": "CANCELLED"},
            "ENQUEUED": {"DEQUEUE": "DEQUEUED"},
            "DEQUEUED": {"ISSUE": "ISSUED", "CANCEL": "CANCELLED"},
            "REJECTED": {},
            "ISSUED": {},
            "CANCELLED": {},
        }
        terminal_states = {"REJECTED", "ISSUED", "CANCELLED"}
        invalid_transitions = 0
        state_mismatches = 0
        incomplete_tasks = 0
        timestamp_regressions = 0
        queue_wait_ns: list[int] = []
        create_to_issue_ns: list[int] = []
        enqueue_to_issue_ns: list[int] = []
        hint_return_ns: list[int] = []

        for records in tasks.values():
            state: str | None = None
            previous_ts = 0
            for record in records:
                event = str(record.get("lifecycle_event", "UNKNOWN"))
                next_state = transitions.get(state, {}).get(event)
                if next_state is None:
                    invalid_transitions += 1
                    continue
                state = next_state
                if record.get("state") != state:
                    state_mismatches += 1
                ts_ns = int(record.get("ts_ns", 0))
                if previous_ts and ts_ns < previous_ts:
                    timestamp_regressions += 1
                previous_ts = max(previous_ts, ts_ns)
            if state not in terminal_states:
                incomplete_tasks += 1

            final = records[-1]
            created_ts = int(final.get("created_ts_ns", 0))
            enqueued_ts = int(final.get("enqueued_ts_ns", 0))
            dequeued_ts = int(final.get("dequeued_ts_ns", 0))
            issued_ts = int(final.get("issued_ts_ns", 0))
            returned_ts = int(final.get("returned_ts_ns", 0))
            if enqueued_ts and dequeued_ts >= enqueued_ts:
                queue_wait_ns.append(dequeued_ts - enqueued_ts)
            if created_ts and issued_ts >= created_ts:
                create_to_issue_ns.append(issued_ts - created_ts)
            if enqueued_ts and issued_ts >= enqueued_ts:
                enqueue_to_issue_ns.append(issued_ts - enqueued_ts)
            if issued_ts and returned_ts >= issued_ts:
                hint_return_ns.append(returned_ts - issued_ts)

        metrics["expert_task_trace_invalid_transitions"] = invalid_transitions
        metrics["expert_task_trace_state_mismatches"] = state_mismatches
        metrics["expert_task_trace_incomplete"] = incomplete_tasks
        metrics["expert_task_trace_timestamp_regressions"] = timestamp_regressions
        metrics["expert_task_hint_returned_records"] = sum(
            1 for r in task_events
            if r.get("lifecycle_event") == "ISSUE" and r.get("hint_status") == "returned"
        )

        issue_records = [r for r in task_events if r.get("lifecycle_event") == "ISSUE"]
        issue_tasks: dict[int, list[dict]] = defaultdict(list)
        invalid_issue_id_records = 0
        for record in issue_records:
            issue_id = record.get("issue_id")
            if not isinstance(issue_id, int) or issue_id <= 0:
                invalid_issue_id_records += 1
                continue
            issue_tasks[issue_id].append(record)
        metrics["expert_task_invalid_issue_id_records"] = invalid_issue_id_records
        metrics["expert_issue_unique_ids"] = len(issue_tasks)
        metrics["expert_issue_coalesced_groups"] = sum(
            1 for records in issue_tasks.values() if len(records) > 1
        )
        metrics["expert_issue_coalesced_tasks"] = sum(
            len(records) for records in issue_tasks.values() if len(records) > 1
        )
        metrics["expert_issue_max_tasks_per_group"] = max(
            (len(records) for records in issue_tasks.values()), default=0
        )
        metrics["expert_issue_task_count_mismatches"] = sum(
            1 for records in issue_tasks.values()
            if any(int(r.get("issue_task_count", 0)) != len(records) for r in records)
        )
        metrics["expert_issue_cross_stage_groups_from_detail"] = sum(
            int(len({str(record.get("stage", "UNKNOWN")) for record in records}) > 1)
            for records in issue_tasks.values()
        )

        linked_syscalls = [
            r for r in os_hints
            if is_os_hint_syscall(r) and isinstance(r.get("issue_id"), int) and r["issue_id"] > 0
        ]
        syscall_issue_ids = {int(r["issue_id"]) for r in linked_syscalls}
        task_issue_ids = set(issue_tasks)
        metrics["expert_issue_linked_syscalls"] = len(linked_syscalls)
        metrics["expert_issue_ids_with_syscalls"] = len(syscall_issue_ids)
        metrics["expert_issue_ids_without_syscalls"] = len(task_issue_ids - syscall_issue_ids)
        metrics["expert_syscall_issue_ids_without_tasks"] = len(syscall_issue_ids - task_issue_ids)

        def add_latency_percentiles(prefix: str, values_ns: list[int]) -> None:
            if not values_ns:
                return
            metrics[f"{prefix}_p50_us"] = float(np.percentile(values_ns, 50)) / 1e3
            metrics[f"{prefix}_p95_us"] = float(np.percentile(values_ns, 95)) / 1e3

        add_latency_percentiles("expert_task_queue_wait", queue_wait_ns)
        add_latency_percentiles("expert_task_create_to_issue", create_to_issue_ns)
        add_latency_percentiles("expert_task_enqueue_to_issue", enqueue_to_issue_ns)
        add_latency_percentiles("expert_task_hint_return", hint_return_ns)

    first_use_events = [r for r in data["memory"] if r.get("event") == "EXPERT_FIRST_USE"]
    first_use_summaries = [
        r for r in data["memory"] if r.get("event") == "EXPERT_FIRST_USE_SUMMARY"
    ]
    if first_use_summaries:
        metrics["expert_first_use_summary_events"] = len(first_use_summaries)
        metrics["expert_first_use_summary_semantic_violations"] = sum(
            1 for r in first_use_summaries
            if r.get("semantics") != "logical_first_use" or
            r.get("physical_load_observed") is not False
        )
        for field in (
            "eligible_tasks",
            "logical_first_uses",
            "matched_tasks",
            "unmatched_tasks",
            "unmatched_first_uses",
            "ambiguous_matches",
            "duplicate_first_use_ignored",
            "matcher_peak_live_tasks",
            "matcher_expired_tasks",
            "late_issued_tasks",
            "pending_issued_tasks",
            "ignored_old_uses",
        ):
            metrics[f"expert_first_use_{field}"] = sum(
                int(r.get(field, 0)) for r in first_use_summaries
            )
        eligible = metrics["expert_first_use_eligible_tasks"]
        uses = metrics["expert_first_use_logical_first_uses"]
        matched = metrics["expert_first_use_matched_tasks"]
        unmatched_uses = metrics["expert_first_use_unmatched_first_uses"]
        metrics["expert_first_use_task_match_rate_pct"] = (
            100.0 * matched / eligible if eligible else 0.0
        )
        metrics["expert_first_use_coverage_pct"] = (
            100.0 * (uses - unmatched_uses) / uses if uses else 0.0
        )
        metrics["expert_first_use_create_to_first_use_ns_by_stage"] = {
            stage: {
                field: sum(
                    int(r.get("create_to_first_use_ns_by_stage", {}).get(stage, {}).get(field, 0))
                    for r in first_use_summaries
                )
                for field in ("count", "total_ns", "min_ns", "max_ns")
            }
            for stage in ("EARLY", "LATE", "UNKNOWN")
        }

    if first_use_events:
        metrics["expert_first_use_records"] = len(first_use_events)
        matched_first_use_events = [
            r for r in first_use_events if r.get("matched") is True or "task_id" in r
        ]
        task_ids = [int(r.get("task_id", 0)) for r in matched_first_use_events]
        issue_ids = [int(r.get("issue_id", 0)) for r in matched_first_use_events]
        metrics["expert_first_use_unique_task_ids"] = len({value for value in task_ids if value > 0})
        metrics["expert_first_use_unique_issue_ids"] = len({value for value in issue_ids if value > 0})
        metrics["expert_first_use_duplicate_task_matches"] = len(task_ids) - len(set(task_ids))
        metrics["expert_first_use_invalid_identity_records"] = sum(
            1 for task_id, issue_id in zip(task_ids, issue_ids) if task_id <= 0 or issue_id <= 0
        )
        metrics["expert_first_use_semantic_violations"] = sum(
            1 for r in first_use_events
            if r.get("semantics") != "logical_first_use" or r.get("physical_load_observed") is not False
        )
        metrics["expert_first_use_invalid_stage_records"] = sum(
            1 for record in first_use_events
            if record.get("stage") not in {"EARLY", "LATE", "UNKNOWN"}
        )
        issue_to_first_use = [
            int(r["issue_to_first_use_ns"]) for r in matched_first_use_events
            if isinstance(r.get("issue_to_first_use_ns"), (int, float)) and
            int(r["issue_to_first_use_ns"]) >= 0
        ]
        if issue_to_first_use:
            metrics["expert_first_use_issue_to_first_use_p50_us"] = (
                float(np.percentile(issue_to_first_use, 50)) / 1e3
            )
            metrics["expert_first_use_issue_to_first_use_p95_us"] = (
                float(np.percentile(issue_to_first_use, 95)) / 1e3
            )

        issued_by_task = {
            int(r["task_id"]): int(r.get("issue_id", 0))
            for r in task_events
            if r.get("lifecycle_event") == "ISSUE" and isinstance(r.get("task_id"), int)
        }
        metrics["expert_first_use_task_link_mismatches"] = sum(
            1 for task_id, issue_id in zip(task_ids, issue_ids)
            if issued_by_task.get(task_id) != issue_id
        )

        metrics["expert_stage_pairing"] = collect_expert_stage_pairing(first_use_events)

    metrics["expert_stage_scheduling_opportunity"] = analyze_stage_scheduling_opportunity(
        data["memory"]
    )

    async_summaries = [r for r in data["memory"] if r.get("event") == "EXPERT_ASYNC_SUMMARY"]
    if async_summaries:
        metrics["expert_async_summary_events"] = len(async_summaries)
        metrics["expert_async_enqueued"] = sum(int(r.get("enqueued", 0)) for r in async_summaries)
        metrics["expert_async_issued"] = sum(int(r.get("issued", 0)) for r in async_summaries)
        metrics["expert_async_priority_enabled"] = 1 if any(r.get("priority_enabled") is True for r in async_summaries) else 0
        metrics["expert_async_priority_heap_enabled"] = 1 if any(r.get("priority_heap_enabled") is True for r in async_summaries) else 0
        priority_modes = sorted({str(r.get("priority_mode", "")) for r in async_summaries if r.get("priority_mode")})
        if priority_modes:
            metrics["expert_async_priority_mode"] = ",".join(priority_modes)
        metrics["expert_async_priority_pops"] = sum(int(r.get("priority_pops", 0)) for r in async_summaries)
        metrics["expert_async_priority_heap_pops"] = sum(int(r.get("priority_heap_pops", 0)) for r in async_summaries)
        metrics["expert_async_fallback"] = sum(int(r.get("fallback", 0)) for r in async_summaries)
        metrics["expert_async_queue_full_fallbacks"] = sum(int(r.get("queue_full_fallbacks", 0)) for r in async_summaries)
        metrics["expert_async_start_fail_fallbacks"] = sum(int(r.get("start_fail_fallbacks", 0)) for r in async_summaries)
        metrics["expert_async_max_queue_depth"] = max(int(r.get("max_queue_depth", 0)) for r in async_summaries)
        metrics["expert_async_max_queued_mb"] = max(int(r.get("max_queued_bytes", 0)) for r in async_summaries) / (1024**2)
        metrics["expert_async_queue_capacity"] = max(int(r.get("queue_capacity", 0)) for r in async_summaries)
        metrics["expert_async_workers"] = max(int(r.get("workers", 0)) for r in async_summaries)
        metrics["expert_async_cancelled_expired"] = sum(int(r.get("cancelled_expired", 0)) for r in async_summaries)
        metrics["expert_async_cancelled_pressure"] = sum(int(r.get("cancelled_pressure", 0)) for r in async_summaries)
        metrics["expert_async_cancelled_value"] = sum(int(r.get("cancelled_value", 0)) for r in async_summaries)
        metrics["expert_async_cancelled_queue_full"] = sum(int(r.get("cancelled_queue_full", 0)) for r in async_summaries)
        metrics["expert_async_worker_batches"] = sum(int(r.get("worker_batches", 0)) for r in async_summaries)
        metrics["expert_async_batched_candidates"] = sum(int(r.get("batched_candidates", 0)) for r in async_summaries)
        metrics["expert_async_coalesced_syscalls_saved"] = sum(
            int(r.get("coalesced_syscalls_saved", 0)) for r in async_summaries
        )
        metrics["expert_async_batch_size"] = max(int(r.get("batch_size", 1)) for r in async_summaries)
        metrics["expert_async_batch_wait_us"] = max(int(r.get("batch_wait_us", 0)) for r in async_summaries)

    route_hint_summaries = [r for r in data["memory"] if r.get("event") == "EXPERT_ROUTE_HINT_SUMMARY"]
    if route_hint_summaries:
        metrics["expert_route_hint_summary_events"] = len(route_hint_summaries)
        metrics["expert_route_hint_ttl_steps"] = max(int(r.get("ttl_steps", 0)) for r in route_hint_summaries)
        metrics["expert_route_hint_candidates"] = sum(int(r.get("candidates", 0)) for r in route_hint_summaries)
        metrics["expert_route_hint_issued"] = sum(int(r.get("issued", 0)) for r in route_hint_summaries)
        metrics["expert_route_hint_skipped"] = sum(int(r.get("skipped", 0)) for r in route_hint_summaries)
        metrics["expert_route_hint_duplicate_skipped"] = sum(int(r.get("duplicate_skipped", 0)) for r in route_hint_summaries)
        metrics["expert_route_hint_ttl_skipped"] = sum(int(r.get("ttl_skipped", 0)) for r in route_hint_summaries)

    pressure_events = [r for r in data["memory"] if r.get("event") == "EXPERT_PRESSURE"]
    if pressure_events:
        metrics["expert_pressure_samples"] = len(pressure_events)
        for level, count in Counter(str(r.get("level", "unknown")) for r in pressure_events).items():
            metrics[f"expert_pressure_{level}_samples"] = count
        metrics["expert_pressure_high_or_critical_samples"] = sum(
            1 for r in pressure_events if r.get("level") in {"high", "critical"}
        )
        metrics["expert_pressure_memory_ratio_peak_pct"] = max(
            float(r.get("memory_ratio_pct", 0.0)) for r in pressure_events
        )
        metrics["expert_pressure_psi_some_peak"] = max(float(r.get("psi_some_avg10", 0.0)) for r in pressure_events)
        metrics["expert_pressure_psi_full_peak"] = max(float(r.get("psi_full_avg10", 0.0)) for r in pressure_events)
        metrics["expert_pressure_refault_delta_total"] = sum(int(r.get("refault_delta", 0)) for r in pressure_events)
        budgets = [int(r.get("prefetch_budget_bytes", 0)) for r in pressure_events]
        metrics["expert_pressure_budget_min_mb"] = min(budgets) / (1024**2)
        metrics["expert_pressure_budget_max_mb"] = max(budgets) / (1024**2)

    prediction_summaries = [r for r in data["memory"] if r.get("event") == "EXPERT_PREDICT_SUMMARY"]
    if prediction_summaries:
        metrics["expert_prediction_summary_events"] = len(prediction_summaries)
        for field in (
            "observed_routes",
            "learned_transitions",
            "transition_buckets",
            "prediction_sets",
            "prediction_candidates",
            "evaluated_sets",
            "evaluated_candidates",
            "prediction_hits",
            "prediction_set_hits",
            "actual_experts_evaluated",
            "unevaluated_sets",
            "capacity_skips",
            "destination_replacements",
        ):
            metric_name = f"expert_{field}" if field.startswith("prediction_") else f"expert_prediction_{field}"
            metrics[metric_name] = sum(int(r.get(field, 0)) for r in prediction_summaries)
        evaluated = metrics["expert_prediction_evaluated_candidates"]
        actual = metrics["expert_prediction_actual_experts_evaluated"]
        evaluated_sets = metrics["expert_prediction_evaluated_sets"]
        metrics["expert_prediction_precision_pct"] = (
            100.0 * metrics["expert_prediction_hits"] / evaluated if evaluated else 0.0
        )
        metrics["expert_prediction_recall_pct"] = (
            100.0 * metrics["expert_prediction_hits"] / actual if actual else 0.0
        )
        metrics["expert_prediction_set_hit_rate_pct"] = (
            100.0 * metrics["expert_prediction_set_hits"] / evaluated_sets if evaluated_sets else 0.0
        )

    return metrics


# ─────────────────────────────────────────────
#  Generate Optimization Recommendations
# ─────────────────────────────────────────────

def generate_recommendations(metrics: dict, data: dict[str, list[dict]]) -> list[dict]:
    """Analyze metrics and generate targeted optimization recommendations."""
    recs = []

    # 1. RSS peak analysis
    rss_peak = metrics.get("rss_peak_gb", 0)
    if rss_peak > 0:
        recs.append({
            "priority": "HIGH",
            "title": "降低峰值物理内存 RSS",
            "finding": f"推理过程中峰值 RSS 达到 {rss_peak:.2f} GB。",
            "recommendation": (
                "实现逐层权重流式释放：每层计算完成后，对该层权重 tensor 使用 "
                "madvise(MADV_DONTNEED)。由于 layer 计算是顺序执行的，已经完成计算的层可以释放其物理页，"
                "不影响正确性。"
            ),
            "expected_benefit": "对使用 mmap 加载的大模型，理论上可降低 40-60% 的峰值 RSS。"
        })

    # 2. Page fault analysis
    total_pf = metrics.get("total_minor_faults", 0)
    ft_gb = metrics.get("first_touch_gb", 0)
    if total_pf > 0 and ft_gb > 0:
        pf_per_gb = total_pf / ft_gb if ft_gb > 0 else 0
        recs.append({
            "priority": "HIGH",
            "title": "优化 mmap Page Fault 模式",
            "finding": (
                f"在 first-touch {ft_gb:.2f} GB tensor 数据时，观测到 {total_pf:,} 次 minor page-fault delta "
                f"（约 {pf_per_gb:,.0f} faults/GB）。每次 fault 都会触发内核页分配或页表处理。"
            ),
            "recommendation": (
                "1. 对热点层，在模型加载时使用 MAP_POPULATE 预先触发关键权重页加载。\n"
                "2. 对 mmap 区域使用 madvise(MADV_SEQUENTIAL)，向内核提示顺序访问模式，改善 readahead。\n"
                "3. 考虑为权重 tensor 使用 huge pages（2MB 或 1GB），减少 TLB miss 和页级 fault 数量。"
            ),
            "expected_benefit": "有机会显著减少 minor faults，并改善 TLB 覆盖率。"
        })

    # 2b. Page residency analysis
    ft_res_pct = metrics.get("first_touch_resident_pct_weighted")
    ft_nonresident = metrics.get("first_touch_nonresident_gb_est", 0)
    if ft_res_pct is not None:
        recs.append({
            "priority": "HIGH" if ft_res_pct < 50 else "MEDIUM",
            "title": "基于页驻留率优化权重预取",
            "finding": (
                f"first-touch tensor 的加权页驻留率为 {ft_res_pct:.1f}%，"
                f"估算未驻留 first-touch 页面为 {ft_nonresident:.2f} GiB。"
            ),
            "recommendation": (
                "优先对未驻留比例高、且会在 PREFILL 或高频 DECODE 路径访问的 tensor 做预取实验。"
                "可比较 madvise(MADV_WILLNEED)、posix_fadvise(POSIX_FADV_WILLNEED) 和按层/按 expert 预取。"
            ),
            "expected_benefit": "减少首次访问时的 page fault 突发，并降低 PREFILL 阶段尾部延迟。"
        })

    os_hint_events = metrics.get("os_hint_events", 0)
    if os_hint_events > 0:
        os_hint_errors = metrics.get("os_hint_errors", 0)
        advised_mb = metrics.get("os_hint_advised_mb", 0)
        os_hint_records = metrics.get("os_hint_records", os_hint_events)
        recs.append({
            "priority": "MEDIUM" if os_hint_errors == 0 else "HIGH",
            "title": "对比 OS Hint 原型效果",
            "finding": (
                f"本次运行触发 {os_hint_events:,} 次实际 OS hint（原始决策记录 {os_hint_records:,} 条），"
                f"累计 hint 范围约 {advised_mb:.1f} MiB，"
                f"失败 {os_hint_errors:,} 次。"
            ),
            "recommendation": (
                "将本次运行与未启用 OS hint 的基线比较，重点观察 major/minor fault delta、"
                "first-touch 驻留率、PREFILL 延迟和 swap 峰值是否改善。"
            ),
            "expected_benefit": "验证 madvise、posix_fadvise、THP 和 expert-aware prefetch 是否能把观测结论转化为性能收益。"
        })

    # 3. Prefill vs Decode
    pf_us = metrics.get("prefill_avg_latency_us", 0)
    dc_us = metrics.get("decode_avg_latency_us", 0)
    if pf_us > 0 and dc_us > 0 and pf_us > dc_us:
        recs.append({
            "priority": "MEDIUM",
            "title": "Prefill 阶段内存预取",
            "finding": (
                f"Prefill traced-token latency（{pf_us:.0f}μs）高于 decode（{dc_us:.0f}μs）。"
                "这里应把 prefill 看作 ubatch/prompt-processing 延迟，而不是与生成 token 延迟直接对比。"
                "Prefill 会在短时间内连续触碰大量权重页。"
            ),
            "recommendation": (
                "在 prefill 期间使用异步预取（prefetch() 或软件流水线），在计算当前层时提前把下一层权重带入 cache。"
                "这样可以重叠计算和内存访问。"
            ),
            "expected_benefit": "通过隐藏部分内存延迟，可能带来 10-20% 的 prefill 加速。"
        })

    # 4. KV Cache
    kv_mb = metrics.get("kv_total_mb", 0)
    reuse = metrics.get("kv_reuse_rate_pct", 0)
    if kv_mb > 0:
        kv_tokens = metrics.get("kv_tokens_appended_est", 0)
        kv_per_1k = metrics.get("kv_mb_per_1k_tokens_est", 0)
        kv_proj_4k = metrics.get("kv_projected_4096_mb", 0)
        kv_proj_8k = metrics.get("kv_projected_8192_mb", 0)
        recs.append({
            "priority": "HIGH",
            "title": "KV Cache 内存管理",
            "finding": (
                f"KV cache 在 {metrics.get('kv_layers', 0)} 个层上增长到 {kv_mb:.1f} MB，"
                f"估算追加 token 数为 {kv_tokens:,}，约 {kv_per_1k:.1f} MB/1k tokens。"
                f"按当前增长率，4k/8k 上下文约为 {kv_proj_4k:.1f}/{kv_proj_8k:.1f} MB。"
                f"KV reuse rate 为 {reuse:.1f}%；这里表示旧 KV slot 复用情况，"
                f"{'当前复用情况较好' if reuse > 30 else '这次运行主要是在追加新 KV'}。"
            ),
            "recommendation": (
                "1. 实现 KV cache 量化（例如 int8/fp8 KV cache），降低内存占用。\n"
                "2. 对长上下文场景，考虑 sliding window attention 或 token eviction 策略，限制 KV cache 增长。\n"
                "3. 为 KV cache 使用专用内存池和预分配 buffer，减少运行时分配开销。"
            ),
            "expected_benefit": "KV 量化通常可显著降低 KV cache 内存，从而支持更长上下文或更大 batch。"
        })

    # 5. Expert skew
    expert_routes = [r for r in data["expert"] if r.get("event") == "EXPERT_ROUTE"]
    if expert_routes:
        expert_counts: dict[int, int] = defaultdict(int)
        for r in expert_routes:
            for eid in valid_experts(r):
                expert_counts[eid] += 1

        if expert_counts:
            total_activations = sum(expert_counts.values())
            n_experts_global = len(expert_counts)
            top5_pct = sum(c for _, c in sorted(expert_counts.items(), key=lambda x: -x[1])[:5]) / total_activations * 100

            # Check if expert IDs suggest per-layer numbering
            max_eid = max(expert_counts.keys())
            expert_layers = len(set(r.get("layer", -1) for r in expert_routes if r.get("layer", -1) >= 0))
            note = (
                f"Expert ID 范围为 0-{max_eid}，分布在 {expert_layers} 个 MoE 层中。"
                f"这些 ID 看起来是每层内部编号（例如每层 0-255）。"
                if max_eid <= 255 else ""
            )

            recs.append({
                "priority": "MEDIUM",
                "title": "MoE Expert 放置优化",
                "finding": (
                    f"在 {expert_layers} 个 MoE 层中观测到 {n_experts_global} 个有效 expert ID。"
                    f"全局 top 5 expert ID 占全部激活的 {top5_pct:.1f}%，说明 expert 激活偏斜"
                    f"{'很强' if top5_pct > 60 else '比较明显' if top5_pct > 20 else '中等'}。"
                    f"{note}"
                ),
                "recommendation": (
                    "对 hot experts（高频激活）：尽量让权重常驻物理内存，或放到更快的存储层级（GPU/HBM）。\n"
                    "对 cold experts（低频激活）：保留 mmap 懒加载，需要时再 fault in。"
                    "这种 expert-aware 的分层放置可以降低 MoE 模型的平均物理内存压力。"
                ),
                "expected_benefit": (
                    "如果某些层内 expert 热点足够集中，按 (layer, expert) pin 热点权重会比全局 top expert 更有效。"
                )
            })

    # 6. General
    recs.append({
        "priority": "LOW",
        "title": "操作系统层调优",
        "finding": "当前为 WSL2/Linux 环境，内核参数会影响页面回收和 fault 行为。",
        "recommendation": (
            "1. 设置 vm.swappiness=1，尽量减少 swap。\n"
            "2. 启用 transparent huge pages：echo madvise > /sys/kernel/mm/transparent_hugepage/enabled\n"
            "3. 如果 mmap_count 很高，提高 vm.max_map_count：sysctl -w vm.max_map_count=262144\n"
            "4. 使用 cgroups v2 memory limit 测试受限内存场景。"
        ),
        "expected_benefit": "提升系统层稳定性；THP 场景下可能减少一部分 TLB miss。"
    })

    return recs


# ─────────────────────────────────────────────
#  Generate HTML Report
# ─────────────────────────────────────────────

def generate_html_report(
    out_dir: str,
    metrics: dict,
    recommendations: list[dict],
    plot_paths: dict[str, str | None],
    data_summary: dict,
    num_generate: int,
):
    """Generate a self-contained HTML analysis report."""

    def _opt(val, fmt=".2f"):
        if val is None or val == 0:
            return "N/A"
        if isinstance(val, float):
            return f"{val:{fmt}}"
        return str(val)

    # Build recommendation cards
    rec_cards = ""
    priority_order = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
    priority_colors = {"HIGH": "#F44336", "MEDIUM": "#FF9800", "LOW": "#4CAF50"}
    priority_labels = {"HIGH": "高", "MEDIUM": "中", "LOW": "低"}
    for rec in sorted(recommendations, key=lambda r: priority_order.get(r["priority"], 99)):
        bg = priority_colors.get(rec["priority"], "#9E9E9E")
        rec_cards += f"""
        <div class="rec-card" style="border-left: 4px solid {bg};">
            <div class="rec-header">
                <span class="rec-priority" style="background: {bg};">{priority_labels.get(rec['priority'], rec['priority'])}</span>
                <span class="rec-title">{rec['title']}</span>
            </div>
            <div class="rec-finding"><strong>🔍 发现：</strong>{rec['finding']}</div>
            <div class="rec-action"><strong>💡 建议：</strong><br>{rec['recommendation'].replace(chr(10), '<br>')}</div>
            <div class="rec-benefit"><strong>📊 预期收益：</strong>{rec['expected_benefit']}</div>
        </div>
        """

    # Build metric cards
    kv_reuse = metrics.get("kv_reuse_rate_pct")
    kv_reuse_display = "N/A" if kv_reuse is None else f"{kv_reuse:.1f}"
    ft_res = metrics.get("first_touch_resident_pct_weighted")
    ft_res_display = "N/A" if ft_res is None else f"{ft_res:.1f}%"
    pss_peak = metrics.get("pss_peak_gb")
    pss_peak_display = "N/A" if pss_peak is None else f"{pss_peak:.2f} GB"
    os_hint_events = metrics.get("os_hint_events", 0)
    os_hint_errors = metrics.get("os_hint_errors", 0)
    os_hint_records = metrics.get("os_hint_records", os_hint_events)
    cache_hit_rate = metrics.get("expert_cache_hit_rate_pct")
    cache_hit_rate_display = "N/A" if cache_hit_rate is None else f"{cache_hit_rate:.1f}%"
    cache_peak = metrics.get("expert_cache_peak_mb")
    cache_peak_display = "N/A" if cache_peak is None else f"{cache_peak:.1f} MiB"
    metric_cards = f"""
    <div class="metric-card"><div class="metric-value">{_opt(metrics.get('rss_peak_gb'))} GB</div><div class="metric-label">峰值 RSS</div></div>
    <div class="metric-card"><div class="metric-value">{pss_peak_display}</div><div class="metric-label">峰值 PSS</div></div>
    <div class="metric-card"><div class="metric-value">{_opt(metrics.get('rss_avg_gb'))} GB</div><div class="metric-label">平均 RSS</div></div>
    <div class="metric-card"><div class="metric-value">{metrics.get('total_minor_faults', 0):,}</div><div class="metric-label">Minor 缺页次数</div></div>
    <div class="metric-card"><div class="metric-value">{_opt(metrics.get('kv_total_mb'))} MB</div><div class="metric-label">KV Cache 大小</div></div>
    <div class="metric-card"><div class="metric-value">{kv_reuse_display}%</div><div class="metric-label">KV Slot 复用率</div></div>
    <div class="metric-card"><div class="metric-value">{_opt(metrics.get('prefill_avg_latency_us'), '.0f')} μs</div><div class="metric-label">Prefill 延迟</div></div>
    <div class="metric-card"><div class="metric-value">{_opt(metrics.get('decode_avg_latency_us'), '.0f')} μs</div><div class="metric-label">Decode 延迟</div></div>
    <div class="metric-card"><div class="metric-value">{metrics.get('unique_experts_global', 'N/A')}</div><div class="metric-label">唯一 Expert ID</div></div>
    <div class="metric-card"><div class="metric-value">{metrics.get('expert_layers', 'N/A')}</div><div class="metric-label">MoE 层数</div></div>
    <div class="metric-card"><div class="metric-value">{_opt(metrics.get('first_touch_gb'))} GB</div><div class="metric-label">mmap 懒加载量</div></div>
    <div class="metric-card"><div class="metric-value">{ft_res_display}</div><div class="metric-label">First-touch 驻留率</div></div>
    <div class="metric-card"><div class="metric-value">{os_hint_events:,}</div><div class="metric-label">OS Hint 系统调用</div></div>
    <div class="metric-card"><div class="metric-value">{os_hint_records:,}</div><div class="metric-label">OS Hint 决策记录</div></div>
    <div class="metric-card"><div class="metric-value">{os_hint_errors:,}</div><div class="metric-label">OS Hint 失败</div></div>
    <div class="metric-card"><div class="metric-value">{cache_hit_rate_display}</div><div class="metric-label">Expert Cache 命中率</div></div>
    <div class="metric-card"><div class="metric-value">{cache_peak_display}</div><div class="metric-label">Expert Cache 峰值</div></div>
    <div class="metric-card"><div class="metric-value">{metrics.get('mmap_count', 'N/A')}</div><div class="metric-label">mmap 区域数</div></div>
    """

    # Plot gallery
    plot_gallery = ""
    plot_labels = {
        "Memory Timeline": "内存时间线",
        "KV Cache": "KV Cache 行为",
        "Expert Activation": "Expert 激活分布",
        "Tensor Access": "Tensor 访存",
        "Token Latency": "Token 延迟",
        "Summary Dashboard": "总览仪表盘",
        "Page Residency": "页面驻留分析",
    }
    for name, path in plot_paths.items():
        if path:
            rel = os.path.relpath(path, out_dir)
            label = plot_labels.get(name, name)
            plot_gallery += f"""
            <div class="plot-card">
                <img src="{rel}" alt="{label}" loading="lazy">
                <div class="plot-caption">{label}</div>
            </div>
            """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM 访存行为分析报告</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f5f5f5; color: #333; line-height: 1.6;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
    .header {{
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        color: white; padding: 40px 30px; border-radius: 12px; margin-bottom: 30px;
    }}
    .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
    .header p {{ opacity: 0.9; font-size: 14px; }}
    .test-case {{
        background: #e8eaf6; padding: 20px; border-radius: 8px; margin-bottom: 30px;
        border-left: 4px solid #3949ab;
    }}
    .test-case h3 {{ color: #1a237e; margin-bottom: 8px; }}
    .metrics-grid {{
        display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 15px; margin-bottom: 30px;
    }}
    .metric-card {{
        background: white; padding: 20px; border-radius: 10px; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: transform 0.2s;
    }}
    .metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.12); }}
    .metric-value {{ font-size: 24px; font-weight: 700; color: #1a237e; }}
    .metric-label {{ font-size: 12px; color: #666; margin-top: 4px; }}
    .section-title {{
        font-size: 22px; font-weight: 700; color: #1a237e; margin: 30px 0 15px;
        padding-bottom: 8px; border-bottom: 2px solid #e0e0e0;
    }}
    .plot-gallery {{
        display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
        gap: 20px; margin-bottom: 30px;
    }}
    .plot-card {{
        background: white; border-radius: 10px; overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    .plot-card img {{ width: 100%; display: block; }}
    .plot-card .plot-caption {{
        padding: 10px 15px; font-size: 14px; font-weight: 600; color: #555;
    }}
    .rec-card {{
        background: white; padding: 20px; border-radius: 10px; margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    .rec-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
    .rec-priority {{
        color: white; padding: 3px 12px; border-radius: 20px; font-size: 11px;
        font-weight: 700; text-transform: uppercase;
    }}
    .rec-title {{ font-size: 17px; font-weight: 700; color: #333; }}
    .rec-finding, .rec-action, .rec-benefit {{
        margin-top: 8px; font-size: 14px; color: #555;
    }}
    .rec-action {{ background: #f5f5f5; padding: 12px; border-radius: 6px; }}
    .rec-benefit {{ color: #2e7d32; }}
    .footer {{
        text-align: center; padding: 30px; color: #999; font-size: 12px;
    }}
    .phase-legend {{
        display: flex; gap: 20px; margin: 10px 0;
    }}
    .phase-dot {{
        display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px;
    }}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>🔬 LLM 访存行为分析报告</h1>
    <p>
        模型：Qwen3.5-35B-A3B（Q3_K_M 量化，MoE，磁盘约 16GB）<br>
        框架：带 LLM_MEM_TRACE 插桩的 llama.cpp<br>
        环境：WSL2 下 Ubuntu · 纯 CPU 推理<br>
        生成 token：{num_generate} · 阶段：PREFILL + DECODE
    </p>
    <div class="phase-legend">
        <span><span class="phase-dot" style="background:#2196F3;"></span> PREFILL（偏计算/首触加载）</span>
        <span><span class="phase-dot" style="background:#FF9800;"></span> DECODE（偏访存/逐 token）</span>
    </div>
</div>

<div class="test-case">
    <h3>🧪 测试用例设计</h3>
    <p>
        <strong>Prompt：</strong>覆盖 CPU 架构、内存层次、GPU 计算、操作系统内存管理和机器学习推理的综合技术分析，
        用来触发多个知识域上的 MoE expert 路由。<br>
        <strong>生成：</strong>{num_generate} tokens，用于观察 decode 阶段逐 token 访存行为。<br>
        <strong>目的：</strong>同时捕获 prefill 阶段的大范围权重 first-touch，以及 decode 阶段的 KV cache 与重复计算图访存。
    </p>
</div>

<h2 class="section-title">📊 关键指标</h2>
<div class="metrics-grid">
    {metric_cards}
</div>

<h2 class="section-title">📈 可视化分析</h2>
<div class="plot-gallery">
    {plot_gallery}
</div>

<h2 class="section-title">🎯 优化建议</h2>
{rec_cards}

<div class="footer">
    <p>由 LLM Memory Trace Analysis Pipeline 生成 · llama.cpp LLM_MEM_TRACE</p>
    <p>所有时间戳为 CLOCK_MONOTONIC 纳秒 · RSS/VMS 来自 /proc/self/stat</p>
</div>

</div>
</body>
</html>"""

    html_path = os.path.join(out_dir, "analysis_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [OK] HTML report: {html_path}")
    return html_path


# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Memory Trace Analysis")
    parser.add_argument("--trace-dir", required=True, help="Directory containing trace JSONL files")
    parser.add_argument("--output-dir", required=True, help="Directory for output visualizations")
    parser.add_argument("--num-generate", type=int, default=80, help="Number of tokens generated (for report)")
    args = parser.parse_args()

    trace_dir = args.trace_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  LLM Memory Trace Analysis")
    print("=" * 60)
    print()

    # Load data
    print("[1/7] Loading trace data...")
    data = load_all(trace_dir)
    for name, records in data.items():
        print(f"      {name}: {len(records)} events")

    # Collect metrics
    print("\n[2/7] Computing key metrics...")
    metrics = collect_metrics(data)

    process_metrics_path = os.path.join(trace_dir, "process_metrics.json")
    if os.path.exists(process_metrics_path):
        with open(process_metrics_path, "r", encoding="utf-8") as f:
            process_metrics = json.load(f)
        metrics["process_wall_time_s"] = float(process_metrics.get("wall_time_s", 0))
        metrics["process_user_time_s"] = float(process_metrics.get("user_time_s", 0))
        metrics["process_system_time_s"] = float(process_metrics.get("system_time_s", 0))
        metrics["process_max_rss_gb"] = float(process_metrics.get("max_rss_kb", 0)) / (1024**2)
        metrics["process_major_faults"] = int(process_metrics.get("major_faults", 0))
        metrics["process_minor_faults"] = int(process_metrics.get("minor_faults", 0))
        metrics["process_file_inputs"] = int(process_metrics.get("file_inputs", 0))
        metrics["process_file_outputs"] = int(process_metrics.get("file_outputs", 0))
        metrics["process_exit_code"] = int(process_metrics.get("exit_code", 0))
        metrics["total_major_faults"] = metrics["process_major_faults"]
        metrics["total_minor_faults"] = metrics["process_minor_faults"]
        metrics["fault_metric_source"] = "gnu_time_process"
        if metrics["process_max_rss_gb"] > 0:
            metrics["rss_peak_gb_trace"] = metrics.get("rss_peak_gb", 0)
            metrics["rss_peak_gb"] = metrics["process_max_rss_gb"]
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"      {k}: {v:.2f}")
        else:
            print(f"      {k}: {v}")

    # Generate plots
    print("\n[3/7] Generating visualizations...")
    plot_paths = {}

    plot_paths["Memory Timeline"] = plot_memory_timeline(data["memory"], out_dir)
    plot_paths["KV Cache"] = plot_kv_cache(data["kv"], out_dir)

    result = plot_expert_activation(data["expert"], out_dir)
    if result:
        plot_paths["Expert Activation"], sorted_experts, global_counts = result

    result = plot_tensor_access(data["tensor"], out_dir)
    if result:
        plot_paths["Tensor Access"], layer_freq, backend_counts = result

    result = plot_token_latency(data["memory"], out_dir)
    if result:
        plot_paths["Token Latency"], pf_mean, dc_mean = result

    plot_paths["Summary Dashboard"] = plot_summary_dashboard(data, out_dir)

    result = plot_page_residency(data, out_dir)
    if result:
        plot_paths["Page Residency"] = result

    # Data summary for report
    data_summary = {name: len(records) for name, records in data.items()}

    # Generate recommendations
    print("\n[4/7] Generating optimization recommendations...")
    recommendations = generate_recommendations(metrics, data)
    for rec in recommendations:
        print(f"      [{rec['priority']}] {rec['title']}")

    # Generate HTML report
    print("\n[5/7] Building HTML report...")
    html_path = generate_html_report(out_dir, metrics, recommendations, plot_paths, data_summary, args.num_generate)

    # Export metrics as JSON
    print("\n[6/7] Exporting metrics...")
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  [OK] {metrics_path}")
    stage_opportunity_path = os.path.join(out_dir, "stage_scheduling_opportunity.json")
    with open(stage_opportunity_path, "w") as f:
        json.dump(metrics["expert_stage_scheduling_opportunity"], f, indent=2, default=str)
    print(f"  [OK] {stage_opportunity_path}")

    # Write a text summary
    print("\n[7/7] Writing text summary...")
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  LLM Memory Behavior Analysis — Text Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("KEY METRICS\n")
        f.write("-" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")

        f.write("\n\nOPTIMIZATION RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        for rec in recommendations:
            f.write(f"\n  [{rec['priority']}] {rec['title']}\n")
            f.write(f"  Finding: {rec['finding']}\n")
            f.write(f"  Action: {rec['recommendation']}\n")
            f.write(f"  Benefit: {rec['expected_benefit']}\n")

    print(f"  [OK] {summary_path}")

    print("\n" + "=" * 60)
    print(f"  Analysis complete!")
    print(f"  Open {html_path} to view the full report.")
    print("=" * 60)


if __name__ == "__main__":
    main()
