#!/usr/bin/env python3
"""Trace-driven KV-cache memory policy simulator.

This script reads existing LLM_MEM_TRACE output and estimates the memory
footprint of several conservative KV-cache strategies before any runtime
implementation is attempted. It intentionally does not rerun inference or
change model behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/llm_mem_trace_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_WINDOWS = [128, 256, 512, 1024]
DEFAULT_BUDGETS_MB = [128, 256, 512, 768, 1024]
DEFAULT_QUANT_RATIOS = "int8=0.50,int4=0.25"


@dataclass(frozen=True)
class KVChunk:
    chunk_id: int
    step: int
    phase: str
    tokens: int
    bytes: int
    start_token: int
    end_token: int
    events: int


@dataclass
class TraceStats:
    chunks: list[KVChunk]
    total_tokens: int
    total_kv_bytes: int
    bytes_per_token: float
    ctx_len_capacity: int
    append_events: int
    unique_append_events: int
    layers: int
    kinds: int
    phases: str

    @property
    def reserved_ctx_bytes(self) -> float:
        return self.bytes_per_token * self.ctx_len_capacity if self.ctx_len_capacity > 0 else 0.0


@dataclass
class PolicyResult:
    policy: str
    variant: str
    budget_mb: int | None
    window_tokens: int | None
    sink_tokens: int | None
    block_tokens: int | None
    quant_ratio: float
    total_tokens: int
    total_kv_bytes: int
    reserved_ctx_bytes: float
    peak_resident_bytes: float
    final_resident_bytes: float
    retained_tokens_final: float
    evicted_bytes: float
    hint_events_est: int
    quality_risk: str
    notes: str

    @property
    def total_kv_mb(self) -> float:
        return self.total_kv_bytes / (1024**2)

    @property
    def reserved_ctx_mb(self) -> float:
        return self.reserved_ctx_bytes / (1024**2)

    @property
    def peak_resident_mb(self) -> float:
        return self.peak_resident_bytes / (1024**2)

    @property
    def final_resident_mb(self) -> float:
        return self.final_resident_bytes / (1024**2)

    @property
    def evicted_mb(self) -> float:
        return self.evicted_bytes / (1024**2)

    @property
    def peak_vs_full_pct(self) -> float:
        return self.peak_resident_bytes / max(1, self.total_kv_bytes) * 100.0

    @property
    def saved_vs_full_mb(self) -> float:
        return max(0.0, (self.total_kv_bytes - self.peak_resident_bytes) / (1024**2))

    @property
    def saved_vs_reserved_mb(self) -> float:
        return max(0.0, (self.reserved_ctx_bytes - self.peak_resident_bytes) / (1024**2))

    @property
    def retained_token_pct(self) -> float:
        return self.retained_tokens_final / max(1, self.total_tokens) * 100.0


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def iter_jsonl(path: Path, required_text: str = ""):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if required_text and required_text not in line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def kv_append_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("phase"),
        row.get("step"),
        row.get("layer"),
        row.get("kind"),
        row.get("kv_addr"),
        tuple(row.get("token_ids", [])),
        row.get("kv_bytes", 0),
    )


def choose_phase(phases: set[str]) -> str:
    if "DECODE" in phases:
        return "DECODE"
    if "PREFILL" in phases:
        return "PREFILL"
    return sorted(phases)[0] if phases else "UNKNOWN"


def load_kv_trace(trace_dir: Path) -> TraceStats:
    path = trace_dir / "kv_trace.jsonl"
    seen: set[tuple[Any, ...]] = set()
    by_step: dict[int, dict[str, Any]] = {}
    append_events = 0
    layers: set[int] = set()
    kinds: set[str] = set()
    phases_all: set[str] = set()
    ctx_len_capacity = 0

    for row in iter_jsonl(path, '"KV_APPEND"'):
        if row.get("event") != "KV_APPEND":
            continue
        append_events += 1
        key = kv_append_key(row)
        if key in seen:
            continue
        seen.add(key)

        step = safe_int(row.get("step"), 0)
        tokens = safe_int(row.get("n_tokens"), 0)
        if tokens <= 0:
            token_ids = row.get("token_ids", [])
            tokens = len(token_ids) if isinstance(token_ids, list) and token_ids else 1
        kv_bytes = safe_int(row.get("kv_bytes"), 0)
        if kv_bytes <= 0:
            continue

        phase = str(row.get("phase", "UNKNOWN"))
        phases_all.add(phase)
        layer = safe_int(row.get("layer"), -1)
        if layer >= 0:
            layers.add(layer)
        kind = str(row.get("kind", ""))
        if kind:
            kinds.add(kind)
        ctx_len_capacity = max(ctx_len_capacity, safe_int(row.get("ctx_len"), 0))

        item = by_step.setdefault(step, {"bytes": 0, "tokens": 0, "events": 0, "phases": set()})
        item["bytes"] += kv_bytes
        item["tokens"] = max(item["tokens"], tokens)
        item["events"] += 1
        item["phases"].add(phase)

    chunks: list[KVChunk] = []
    token_cursor = 0
    for chunk_id, step in enumerate(sorted(by_step)):
        item = by_step[step]
        tokens = max(1, safe_int(item["tokens"], 1))
        chunk = KVChunk(
            chunk_id=chunk_id,
            step=step,
            phase=choose_phase(item["phases"]),
            tokens=tokens,
            bytes=safe_int(item["bytes"], 0),
            start_token=token_cursor,
            end_token=token_cursor + tokens,
            events=safe_int(item["events"], 0),
        )
        chunks.append(chunk)
        token_cursor += tokens

    total_kv_bytes = sum(chunk.bytes for chunk in chunks)
    bytes_per_token = total_kv_bytes / max(1, token_cursor)
    return TraceStats(
        chunks=chunks,
        total_tokens=token_cursor,
        total_kv_bytes=total_kv_bytes,
        bytes_per_token=bytes_per_token,
        ctx_len_capacity=ctx_len_capacity,
        append_events=append_events,
        unique_append_events=len(seen),
        layers=len(layers),
        kinds=len(kinds),
        phases=",".join(sorted(phases_all)) if phases_all else "UNKNOWN",
    )


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    clean = sorted((max(0, a), max(0, b)) for a, b in intervals if b > a)
    if not clean:
        return []
    merged = [clean[0]]
    for start, end in clean[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def overlap_tokens(chunk: KVChunk, intervals: list[tuple[int, int]]) -> int:
    overlap = 0
    for start, end in intervals:
        overlap += max(0, min(chunk.end_token, end) - max(chunk.start_token, start))
    return min(chunk.tokens, overlap)


def retained_for_intervals(chunks: list[KVChunk], intervals: list[tuple[int, int]]) -> tuple[float, int, set[int]]:
    intervals = merge_intervals(intervals)
    retained_bytes = 0.0
    retained_tokens = 0
    active: set[int] = set()
    for chunk in chunks:
        keep_tokens = overlap_tokens(chunk, intervals)
        if keep_tokens <= 0:
            continue
        retained_tokens += keep_tokens
        retained_bytes += chunk.bytes * keep_tokens / max(1, chunk.tokens)
        active.add(chunk.chunk_id)
    return retained_bytes, retained_tokens, active


def simulate_full(stats: TraceStats) -> PolicyResult:
    return PolicyResult(
        policy="full",
        variant="full_kv",
        budget_mb=None,
        window_tokens=None,
        sink_tokens=None,
        block_tokens=None,
        quant_ratio=1.0,
        total_tokens=stats.total_tokens,
        total_kv_bytes=stats.total_kv_bytes,
        reserved_ctx_bytes=stats.reserved_ctx_bytes,
        peak_resident_bytes=stats.total_kv_bytes,
        final_resident_bytes=stats.total_kv_bytes,
        retained_tokens_final=stats.total_tokens,
        evicted_bytes=0,
        hint_events_est=0,
        quality_risk="none",
        notes="完整保留当前 trace 中追加的 KV；作为语义保持基线。",
    )


def simulate_ctx_reserved(stats: TraceStats) -> PolicyResult:
    reserved = stats.reserved_ctx_bytes if stats.reserved_ctx_bytes > 0 else stats.total_kv_bytes
    return PolicyResult(
        policy="ctx_reserved",
        variant="full_ctx",
        budget_mb=None,
        window_tokens=None,
        sink_tokens=None,
        block_tokens=None,
        quant_ratio=1.0,
        total_tokens=stats.total_tokens,
        total_kv_bytes=stats.total_kv_bytes,
        reserved_ctx_bytes=stats.reserved_ctx_bytes,
        peak_resident_bytes=reserved,
        final_resident_bytes=reserved,
        retained_tokens_final=stats.total_tokens,
        evicted_bytes=0,
        hint_events_est=0,
        quality_risk="none",
        notes="按 ctx_len 全量预留 KV 的物理占用上界估算，用于衡量按需提交收益。",
    )


def simulate_paged(stats: TraceStats, block_tokens: int) -> PolicyResult:
    block_tokens = max(1, block_tokens)
    peak = 0.0
    for chunk in stats.chunks:
        allocated_tokens = math.ceil(chunk.end_token / block_tokens) * block_tokens
        peak = max(peak, allocated_tokens * stats.bytes_per_token)
    final_tokens = math.ceil(stats.total_tokens / block_tokens) * block_tokens if stats.total_tokens else 0
    final_bytes = final_tokens * stats.bytes_per_token
    return PolicyResult(
        policy="paged_blocks",
        variant=f"block{block_tokens}",
        budget_mb=None,
        window_tokens=None,
        sink_tokens=None,
        block_tokens=block_tokens,
        quant_ratio=1.0,
        total_tokens=stats.total_tokens,
        total_kv_bytes=stats.total_kv_bytes,
        reserved_ctx_bytes=stats.reserved_ctx_bytes,
        peak_resident_bytes=peak,
        final_resident_bytes=final_bytes,
        retained_tokens_final=stats.total_tokens,
        evicted_bytes=0,
        hint_events_est=0,
        quality_risk="none",
        notes="PagedAttention/vAttention 风格估算：按需提交固定 token block，保持完整上下文。",
    )


def simulate_quant(stats: TraceStats, name: str, ratio: float) -> PolicyResult:
    ratio = max(0.01, min(1.0, ratio))
    return PolicyResult(
        policy="kv_quant",
        variant=name,
        budget_mb=None,
        window_tokens=None,
        sink_tokens=None,
        block_tokens=None,
        quant_ratio=ratio,
        total_tokens=stats.total_tokens,
        total_kv_bytes=stats.total_kv_bytes,
        reserved_ctx_bytes=stats.reserved_ctx_bytes * ratio,
        peak_resident_bytes=stats.total_kv_bytes * ratio,
        final_resident_bytes=stats.total_kv_bytes * ratio,
        retained_tokens_final=stats.total_tokens,
        evicted_bytes=0,
        hint_events_est=0,
        quality_risk="needs-quality-test",
        notes="KV 量化预算估算；真实采用前需要 perplexity/输出一致性与速度验证。",
    )


def simulate_interval_policy(stats: TraceStats, policy: str, window_tokens: int, sink_tokens: int) -> PolicyResult:
    window_tokens = max(1, window_tokens)
    sink_tokens = max(0, sink_tokens)
    seen_chunks: list[KVChunk] = []
    previous_active: set[int] = set()
    peak_bytes = 0.0
    final_bytes = 0.0
    final_tokens = 0
    evicted_ids: set[int] = set()
    hint_events = 0

    for chunk in stats.chunks:
        seen_chunks.append(chunk)
        total_seen = chunk.end_token
        intervals = [(max(0, total_seen - window_tokens), total_seen)]
        if policy == "sink_recent" and sink_tokens > 0:
            intervals.append((0, min(sink_tokens, total_seen)))
        retained_bytes, retained_tokens, active = retained_for_intervals(seen_chunks, intervals)
        dropped = previous_active - active
        for chunk_id in dropped:
            if chunk_id not in evicted_ids:
                evicted_ids.add(chunk_id)
                hint_events += 1
        previous_active = active
        peak_bytes = max(peak_bytes, retained_bytes)
        final_bytes = retained_bytes
        final_tokens = retained_tokens

    evicted_bytes = max(0.0, stats.total_kv_bytes - final_bytes)
    variant = f"window{window_tokens}"
    if policy == "sink_recent":
        variant = f"sink{sink_tokens}_window{window_tokens}"
    return PolicyResult(
        policy=policy,
        variant=variant,
        budget_mb=None,
        window_tokens=window_tokens,
        sink_tokens=sink_tokens if policy == "sink_recent" else 0,
        block_tokens=None,
        quant_ratio=1.0,
        total_tokens=stats.total_tokens,
        total_kv_bytes=stats.total_kv_bytes,
        reserved_ctx_bytes=stats.reserved_ctx_bytes,
        peak_resident_bytes=peak_bytes,
        final_resident_bytes=final_bytes,
        retained_tokens_final=final_tokens,
        evicted_bytes=evicted_bytes,
        hint_events_est=hint_events,
        quality_risk="semantic-risk",
        notes="窗口保留会丢弃部分历史 KV；只有在模型/attention 机制允许时才可真实采用。",
    )


def simulate_budget_lru(stats: TraceStats, budget_mb: int, sink_tokens: int) -> PolicyResult:
    budget_bytes = max(1, budget_mb) * 1024 * 1024
    sink_tokens = max(0, sink_tokens)
    resident: list[KVChunk] = []
    resident_ids: set[int] = set()
    resident_bytes = 0
    peak_bytes = 0
    evicted_bytes = 0
    hint_events = 0

    def is_sink(chunk: KVChunk) -> bool:
        return sink_tokens > 0 and chunk.start_token < sink_tokens

    for chunk in stats.chunks:
        resident.append(chunk)
        resident_ids.add(chunk.chunk_id)
        resident_bytes += chunk.bytes

        while resident_bytes > budget_bytes and resident:
            victim_index = None
            for idx, item in enumerate(resident):
                if not is_sink(item):
                    victim_index = idx
                    break
            if victim_index is None:
                victim_index = 0
            victim = resident.pop(victim_index)
            resident_ids.discard(victim.chunk_id)
            resident_bytes -= victim.bytes
            evicted_bytes += victim.bytes
            hint_events += 1

        peak_bytes = max(peak_bytes, resident_bytes)

    retained_tokens = sum(chunk.tokens for chunk in resident if chunk.chunk_id in resident_ids)
    return PolicyResult(
        policy="budget_lru",
        variant=f"sink{sink_tokens}_{budget_mb}mb",
        budget_mb=budget_mb,
        window_tokens=None,
        sink_tokens=sink_tokens,
        block_tokens=None,
        quant_ratio=1.0,
        total_tokens=stats.total_tokens,
        total_kv_bytes=stats.total_kv_bytes,
        reserved_ctx_bytes=stats.reserved_ctx_bytes,
        peak_resident_bytes=peak_bytes,
        final_resident_bytes=resident_bytes,
        retained_tokens_final=retained_tokens,
        evicted_bytes=evicted_bytes,
        hint_events_est=hint_events,
        quality_risk="semantic-risk",
        notes="预算 LRU 估算可回收物理内存，但丢弃历史 KV 后需要重算、换入或接受上下文损失。",
    )


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_quant_ratios(value: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid quant ratio entry: {part}")
        name, ratio = part.split("=", 1)
        out.append((name.strip(), float(ratio)))
    return out


def build_results(
    stats: TraceStats,
    policies: set[str],
    windows: list[int],
    budgets_mb: list[int],
    sink_tokens: int,
    block_tokens: int,
    quant_ratios: list[tuple[str, float]],
) -> list[PolicyResult]:
    results: list[PolicyResult] = []
    if "full" in policies:
        results.append(simulate_full(stats))
    if "ctx_reserved" in policies:
        results.append(simulate_ctx_reserved(stats))
    if "paged_blocks" in policies:
        results.append(simulate_paged(stats, block_tokens))
    if "kv_quant" in policies:
        for name, ratio in quant_ratios:
            results.append(simulate_quant(stats, name, ratio))
    if "sliding_window" in policies:
        for window in windows:
            results.append(simulate_interval_policy(stats, "sliding_window", window, 0))
    if "sink_recent" in policies:
        for window in windows:
            results.append(simulate_interval_policy(stats, "sink_recent", window, sink_tokens))
    if "budget_lru" in policies:
        for budget in budgets_mb:
            results.append(simulate_budget_lru(stats, budget, sink_tokens))
    return results


def write_csv(results: list[PolicyResult], out_dir: Path) -> Path:
    path = out_dir / "kv_policy_simulation.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "policy",
                "variant",
                "budget_mb",
                "window_tokens",
                "sink_tokens",
                "block_tokens",
                "quant_ratio",
                "total_tokens_est",
                "total_kv_mb",
                "reserved_ctx_mb",
                "peak_resident_mb",
                "final_resident_mb",
                "peak_vs_full_pct",
                "saved_vs_full_mb",
                "saved_vs_reserved_mb",
                "retained_tokens_final",
                "retained_token_pct",
                "evicted_mb",
                "hint_events_est",
                "quality_risk",
                "notes",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.policy,
                    row.variant,
                    "" if row.budget_mb is None else row.budget_mb,
                    "" if row.window_tokens is None else row.window_tokens,
                    "" if row.sink_tokens is None else row.sink_tokens,
                    "" if row.block_tokens is None else row.block_tokens,
                    f"{row.quant_ratio:.4f}",
                    row.total_tokens,
                    f"{row.total_kv_mb:.4f}",
                    f"{row.reserved_ctx_mb:.4f}",
                    f"{row.peak_resident_mb:.4f}",
                    f"{row.final_resident_mb:.4f}",
                    f"{row.peak_vs_full_pct:.4f}",
                    f"{row.saved_vs_full_mb:.4f}",
                    f"{row.saved_vs_reserved_mb:.4f}",
                    f"{row.retained_tokens_final:.2f}",
                    f"{row.retained_token_pct:.4f}",
                    f"{row.evicted_mb:.4f}",
                    row.hint_events_est,
                    row.quality_risk,
                    row.notes,
                ]
            )
    return path


def plot_results(results: list[PolicyResult], out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if not results:
        return paths

    ordered = sorted(results, key=lambda r: (r.quality_risk != "none", r.peak_resident_mb, r.policy, r.variant))
    labels = [f"{r.policy}\n{r.variant}" for r in ordered]
    colors = [
        "#2E7D32" if r.quality_risk == "none"
        else "#F9A825" if r.quality_risk == "needs-quality-test"
        else "#C62828"
        for r in ordered
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.75), 6))
    ax.bar(labels, [r.peak_resident_mb for r in ordered], color=colors)
    ax.set_title("KV Cache Peak Resident Estimate")
    ax.set_ylabel("MiB")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    path = out_dir / "01_kv_peak_resident.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.75), 6))
    ax.bar(labels, [r.retained_token_pct for r in ordered], color=colors)
    ax.set_title("Final Retained Token Ratio")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    path = out_dir / "02_kv_retained_tokens.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def write_summary(stats: TraceStats, results: list[PolicyResult], out_dir: Path) -> Path:
    path = out_dir / "kv_policy_summary.md"
    safe_rows = [r for r in results if r.quality_risk == "none" and r.policy not in {"full", "ctx_reserved"}]
    baseline_rows = [r for r in results if r.policy == "full"]
    reserved_rows = [r for r in results if r.policy == "ctx_reserved"]
    quality_rows = [r for r in results if r.quality_risk == "needs-quality-test"]
    risky_rows = [r for r in results if r.quality_risk == "semantic-risk"]
    best_safe = min(safe_rows, key=lambda r: r.peak_resident_bytes, default=None)
    best_quality = min(quality_rows, key=lambda r: r.peak_resident_bytes, default=None)
    best_risky = min(risky_rows, key=lambda r: r.peak_resident_bytes, default=None)

    lines = [
        "# KV Cache 策略离线模拟",
        "",
        "本结果来自已有 `kv_trace.jsonl`，用于筛选后续实验方向；脚本不重跑模型，也不改变推理行为。",
        "",
        "## Trace 概况",
        "",
        f"- 唯一 KV append 事件：{stats.unique_append_events:,} / 原始事件 {stats.append_events:,}。",
        f"- 估算追加 token 数：{stats.total_tokens:,}。",
        f"- 当前 trace 实际追加 KV：{stats.total_kv_bytes / (1024**2):.2f} MiB。",
        f"- `ctx_len` 容量估算：{stats.ctx_len_capacity:,} tokens，对应完整预留约 {stats.reserved_ctx_bytes / (1024**2):.2f} MiB。",
        f"- 覆盖层数/类型/阶段：layers={stats.layers}, kinds={stats.kinds}, phases={stats.phases}。",
        "",
        "## 可优先尝试的保守方向",
        "",
    ]
    if baseline_rows:
        baseline = baseline_rows[0]
        lines.append(
            f"- 当前 trace 追加量基线：峰值约 {baseline.peak_resident_mb:.2f} MiB，"
            f"相对完整 `ctx_len` 预留节省约 {baseline.saved_vs_reserved_mb:.2f} MiB。"
        )
    if reserved_rows:
        reserved = reserved_rows[0]
        lines.append(
            f"- 完整 `ctx_len` 预留上界：峰值约 {reserved.peak_resident_mb:.2f} MiB，"
            "用于模拟未按需提交时的 KV 物理占用压力。"
        )
    if best_safe:
        lines.append(
            f"- 语义保持候选：`{best_safe.policy}/{best_safe.variant}`，峰值约 "
            f"{best_safe.peak_resident_mb:.2f} MiB，较完整 `ctx_len` 预留节省约 "
            f"{best_safe.saved_vs_reserved_mb:.2f} MiB。该方向对应按需物理提交/固定块管理，"
            "适合作为后续 runtime 原型的第一优先级。"
        )
    if best_quality:
        lines.append(
            f"- 需要质量验证候选：`{best_quality.policy}/{best_quality.variant}`，峰值约 "
            f"{best_quality.peak_resident_mb:.2f} MiB。该方向需要增加输出一致性、困惑度或任务指标验证。"
        )
    if best_risky:
        lines.append(
            f"- 高风险内存下限：`{best_risky.policy}/{best_risky.variant}`，峰值约 "
            f"{best_risky.peak_resident_mb:.2f} MiB，但最终只保留 "
            f"{best_risky.retained_token_pct:.1f}% token。该类策略不能直接用于普通全上下文注意力。"
        )
    lines.extend(
        [
            "",
            "## 不能直接下结论的方向",
            "",
            "- H2O/Heavy-Hitter 类策略需要 attention score 或 token 重要度埋点；当前 KV trace 只有 append/reuse/bytes，不能可靠判断 heavy hitter。",
            "- 窗口淘汰、sink+recent 和预算 LRU 会改变可见上下文，除非模型结构或推理任务允许，否则只能作为压力测试上界。",
            "- KV 量化可以显著降低占用，但必须补充质量和速度测试，因为反量化开销可能抵消部分吞吐收益。",
            "",
            "## 输出文件",
            "",
            "- `kv_policy_simulation.csv`：完整策略矩阵。",
            "- `01_kv_peak_resident.png`：峰值物理占用估算。",
            "- `02_kv_retained_tokens.png`：最终 token 保留比例。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True, help="Directory containing kv_trace.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--policies",
        default="full,ctx_reserved,paged_blocks,kv_quant,sliding_window,sink_recent,budget_lru",
        help="Comma-separated policies",
    )
    parser.add_argument("--windows-tokens", default=",".join(str(x) for x in DEFAULT_WINDOWS))
    parser.add_argument("--budgets-mb", default=",".join(str(x) for x in DEFAULT_BUDGETS_MB))
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--quant-ratios", default=DEFAULT_QUANT_RATIOS)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.output_dir) if args.output_dir else trace_dir / "kv_policy_simulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_kv_trace(trace_dir)
    if not stats.chunks:
        raise SystemExit(f"no KV_APPEND events found in {trace_dir / 'kv_trace.jsonl'}")

    policies = {part.strip() for part in args.policies.split(",") if part.strip()}
    results = build_results(
        stats=stats,
        policies=policies,
        windows=parse_int_list(args.windows_tokens),
        budgets_mb=parse_int_list(args.budgets_mb),
        sink_tokens=args.sink_tokens,
        block_tokens=args.block_tokens,
        quant_ratios=parse_quant_ratios(args.quant_ratios),
    )

    csv_path = write_csv(results, out_dir)
    plots = plot_results(results, out_dir)
    summary_path = write_summary(stats, results, out_dir)

    print(f"[OK] chunks={len(stats.chunks):,} tokens={stats.total_tokens:,} kv={stats.total_kv_bytes / (1024**2):.2f} MiB")
    print(f"[OK] {csv_path}")
    for plot in plots:
        print(f"[OK] {plot}")
    print(f"[OK] {summary_path}")


if __name__ == "__main__":
    main()
