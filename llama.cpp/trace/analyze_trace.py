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
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


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

    return {
        "tensor": tensor_records,
        "kv":     load_jsonl(os.path.join(trace_dir, "kv_trace.jsonl")),
        "expert": load_jsonl(os.path.join(trace_dir, "expert_trace.jsonl")),
        "memory": load_jsonl(os.path.join(trace_dir, "memory_trace.jsonl")),
    }


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
    token_ends = [r for r in memory_records if r.get("event") == "TOKEN_END" and "latency_ns" in r]
    if not token_ends:
        print("  [SKIP] No token latency data")
        return None

    token_ends = add_relative_time(token_ends)

    prefill_tokens = [r for r in token_ends if r.get("phase") == "PREFILL"]
    decode_tokens = [r for r in token_ends if r.get("phase") == "DECODE"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Token Processing Latency Analysis", fontsize=14, fontweight="bold")

    # --- Latency timeline ---
    ax = axes[0]
    df = pd.DataFrame(token_ends)
    latency_ms = df["latency_ns"] / 1e6
    colors = [phase_color(p) for p in df.get("phase", [])]
    ax.scatter(df["t_ms"], latency_ms, c=colors, s=15, alpha=0.6, edgecolors="none")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Token Latency Timeline")
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
    ax.set_title("Token Latency: Prefill vs Decode")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Token latency by position ---
    ax = axes[2]
    pos_lat = defaultdict(list)
    for r in token_ends:
        pos = r.get("pos")
        if pos is not None:
            pos_lat[pos].append(r["latency_ns"] / 1e3)  # us

    if pos_lat:
        positions = sorted(pos_lat.keys())
        avg_lat = [np.mean(pos_lat[p]) for p in positions]
        colors_pos = ["#2196F3" if p < positions[len(positions) // 2] else "#FF9800" for p in positions]
        ax.bar(positions[:50], avg_lat[:50], color=colors_pos[:50], alpha=0.7)
        ax.set_xlabel("Token Position")
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
    token_ends = [r for r in data["memory"] if r.get("event") == "TOKEN_END" and "latency_ns" in r]
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

    # 6. Token latency summary
    ax = axes[1][2]
    prefill_lat = [t["latency_ns"] / 1e3 for t in token_ends if t.get("phase") == "PREFILL"]
    decode_lat = [t["latency_ns"] / 1e3 for t in token_ends if t.get("phase") == "DECODE"]
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
    ax.set_title("Token Latency: Prefill vs Decode")
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

def collect_metrics(data: dict[str, list[dict]]) -> dict:
    """Extract key numeric metrics for the report."""
    metrics = {}

    # Memory
    mem_events = [r for r in data["memory"] if r.get("event") == "MEMORY_STAT" and "rss_bytes" in r]
    if mem_events:
        rss_values = [r["rss_bytes"] for r in mem_events]
        metrics["rss_peak_gb"] = max(rss_values) / (1024**3)
        metrics["rss_avg_gb"] = np.mean(rss_values) / (1024**3)
        metrics["total_minor_faults"] = sum(max(0, r.get("minor_faults_delta", 0)) for r in mem_events)
        metrics["total_major_faults"] = sum(max(0, r.get("major_faults_delta", 0)) for r in mem_events)
        metrics["minor_faults_cumulative_final"] = mem_events[-1].get("minor_faults", 0)
        metrics["major_faults_cumulative_final"] = mem_events[-1].get("major_faults", 0)
        metrics["mmap_count"] = mem_events[-1].get("mmap_count", 0)

        smaps_events = [r for r in mem_events if "pss_bytes" in r]
        if smaps_events:
            metrics["pss_peak_gb"] = max(r.get("pss_bytes", 0) for r in smaps_events) / (1024**3)
            metrics["pss_avg_gb"] = np.mean([r.get("pss_bytes", 0) for r in smaps_events]) / (1024**3)
            metrics["shared_clean_peak_gb"] = max(r.get("shared_clean_bytes", 0) for r in smaps_events) / (1024**3)
            metrics["private_dirty_peak_gb"] = max(r.get("private_dirty_bytes", 0) for r in smaps_events) / (1024**3)
            metrics["swap_peak_mb"] = max(r.get("swap_bytes", 0) for r in smaps_events) / (1024**2)

    # Token latency
    token_ends = [r for r in data["memory"] if r.get("event") == "TOKEN_END" and "latency_ns" in r]
    if token_ends:
        prefill = [t for t in token_ends if t.get("phase") == "PREFILL"]
        decode = [t for t in token_ends if t.get("phase") == "DECODE"]
        metrics["prefill_tokens"] = len(prefill)
        metrics["decode_tokens"] = len(decode)
        metrics["prefill_avg_latency_us"] = np.mean([t["latency_ns"] / 1e3 for t in prefill]) if prefill else 0
        metrics["decode_avg_latency_us"] = np.mean([t["latency_ns"] / 1e3 for t in decode]) if decode else 0

    # KV Cache
    kv_appends = [r for r in data["kv"] if r.get("event") == "KV_APPEND"]
    if kv_appends:
        kv_unique = dedupe_kv_appends(kv_appends)
        metrics["kv_event_bytes_mb"] = sum(r.get("kv_bytes", 0) for r in kv_appends) / (1024**2)
        metrics["kv_total_mb"] = sum(r.get("kv_bytes", 0) for r in kv_unique) / (1024**2)
        metrics["kv_append_events"] = len(kv_appends)
        metrics["kv_append_events_unique"] = len(kv_unique)
        metrics["kv_append_duplicate_events"] = len(kv_appends) - len(kv_unique)
        kv_layers = set(r.get("layer", -1) for r in kv_unique)
        metrics["kv_layers"] = len(kv_layers)

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
        recs.append({
            "priority": "HIGH",
            "title": "KV Cache 内存管理",
            "finding": (
                f"KV cache 在 {metrics.get('kv_layers', 0)} 个层上增长到 {kv_mb:.1f} MB。"
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
