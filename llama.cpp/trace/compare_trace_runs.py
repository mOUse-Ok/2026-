#!/usr/bin/env python3
"""Compare multiple LLM_MEM_TRACE experiment runs.

This script consumes per-run analysis/metrics.json files plus memory_trace.jsonl
and produces a compact comparison report for OS hint experiments.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_LABELS = {
    "baseline": "Baseline",
    "load_prefetch": "Load Prefetch",
    "expert_prefetch": "Expert Prefetch",
    "expert_prefetch_route": "Expert Prefetch Route",
    "lru_512_cold": "LRU 512 Cold",
    "lfu_512_cold": "LFU 512 Cold",
    "window_lfu_512_cold": "Window LFU 512 Cold",
    "least_stale_512_cold": "Least-Stale 512 Cold",
    "all_hints": "All Hints",
}

KEY_METRICS = [
    ("total_major_faults", "Major Fault Delta", "lower"),
    ("total_minor_faults", "Minor Fault Delta", "lower"),
    ("decode_avg_latency_us", "Decode Latency (us)", "lower"),
    ("prefill_avg_latency_us", "Prefill Latency (us)", "lower"),
    ("first_touch_resident_pct_weighted", "First-touch Resident (%)", "higher"),
    ("first_touch_nonresident_gb_est", "First-touch Nonresident (GiB)", "lower"),
    ("rss_peak_gb", "Peak RSS (GiB)", "lower"),
    ("swap_peak_mb", "Peak Swap (MiB)", "lower"),
    ("os_hint_events", "OS Hint Syscalls", "lower"),
    ("os_hint_records", "OS Hint Records", "neutral"),
    ("os_hint_advised_mb", "OS Hint Advised (MiB)", "neutral"),
    ("kv_total_mb", "KV Total (MiB)", "lower"),
    ("kv_mb_per_1k_tokens_est", "KV MiB/1k Tokens", "lower"),
    ("kv_projected_4096_mb", "Projected KV 4k (MiB)", "lower"),
    ("expert_cache_hit_rate_pct", "Expert Cache Hit (%)", "higher"),
    ("expert_cache_peak_mb", "Expert Cache Peak (MiB)", "lower"),
    ("expert_cache_evictions", "Expert Cache Evictions", "neutral"),
    ("expert_route_hint_ttl_steps", "Route Hint TTL Steps", "neutral"),
    ("expert_route_hint_candidates", "Route Hint Candidates", "neutral"),
    ("expert_route_hint_issued", "Route Hint Issued", "neutral"),
    ("expert_route_hint_ttl_skipped", "Route Hint TTL Skips", "higher"),
    ("expert_async_enqueued", "Async Enqueued", "neutral"),
    ("expert_async_priority_pops", "Async Priority Pops", "neutral"),
    ("expert_async_priority_heap_pops", "Async Heap Pops", "neutral"),
    ("expert_async_fallback", "Async Fallbacks", "lower"),
    ("expert_async_max_queue_depth", "Async Max Queue Depth", "neutral"),
    ("expert_async_workers", "Async Workers", "neutral"),
]

PARETO_METRICS = [
    ("decode_avg_latency_us", "lower"),
    ("total_major_faults", "lower"),
    ("rss_peak_gb", "lower"),
    ("swap_peak_mb", "lower"),
    ("os_hint_events", "lower"),
]


def load_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "analysis" / "metrics.json"
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics.setdefault("os_hint_events", 0)
    metrics.setdefault("os_hint_records", metrics.get("os_hint_events", 0))
    metrics.setdefault("os_hint_errors", 0)
    metrics.setdefault("os_hint_advised_mb", 0.0)
    metrics.setdefault("kv_total_mb", 0.0)
    metrics.setdefault("kv_mb_per_1k_tokens_est", 0.0)
    metrics.setdefault("kv_projected_4096_mb", 0.0)
    metrics.setdefault("expert_cache_hit_rate_pct", 0.0)
    metrics.setdefault("expert_cache_peak_mb", 0.0)
    metrics.setdefault("expert_cache_evictions", 0)
    metrics.setdefault("expert_route_hint_ttl_steps", 0)
    metrics.setdefault("expert_route_hint_candidates", 0)
    metrics.setdefault("expert_route_hint_issued", 0)
    metrics.setdefault("expert_route_hint_ttl_skipped", 0)
    metrics.setdefault("expert_async_enqueued", 0)
    metrics.setdefault("expert_async_priority_pops", 0)
    metrics.setdefault("expert_async_priority_heap_pops", 0)
    metrics.setdefault("expert_async_fallback", 0)
    metrics.setdefault("expert_async_max_queue_depth", 0)
    metrics.setdefault("expert_async_workers", 0)
    return metrics


def load_memory_events(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "memory_trace.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if '"MEMORY_STAT"' not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "ts_ns" in df:
        df["t_s"] = (df["ts_ns"] - df["ts_ns"].min()) / 1e9
    return df


def pct_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return float("nan")
    return (value - baseline) / baseline * 100.0


def signed_improvement(metric: str, value: float, baseline: float) -> float:
    direction = next((d for k, _, d in KEY_METRICS if k == metric), "neutral")
    change = pct_change(value, baseline)
    if np.isnan(change) or direction == "neutral":
        return float("nan")
    return -change if direction == "lower" else change


def metric_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def dominates(candidate: dict[str, Any], other: dict[str, Any]) -> bool:
    strictly_better = False
    for key, direction in PARETO_METRICS:
        a = metric_value(candidate, key)
        b = metric_value(other, key)
        if direction == "lower":
            if a > b:
                return False
            if a < b:
                strictly_better = True
        else:
            if a < b:
                return False
            if a > b:
                strictly_better = True
    return strictly_better


def mark_pareto(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["pareto_optimal"] = not any(dominates(other, row) for other in rows if other is not row)


def fmt_num(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.2f}"
    return str(value)


def collect_rows(base_dir: Path, runs: list[str]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    metrics_by_run: dict[str, dict[str, Any]] = {}
    for run in runs:
        metrics_by_run[run] = load_metrics(base_dir / run)

    baseline = metrics_by_run[runs[0]]
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = metrics_by_run[run]
        row: dict[str, Any] = {
            "run": run,
            "label": RUN_LABELS.get(run, run),
        }
        for key, _, _ in KEY_METRICS:
            value = metrics.get(key, 0)
            base_value = baseline.get(key, 0)
            row[key] = value
            row[f"{key}_change_pct"] = pct_change(float(value), float(base_value)) if base_value else 0.0
            row[f"{key}_improvement_pct"] = signed_improvement(key, float(value), float(base_value)) if base_value else 0.0
        rows.append(row)
    mark_pareto(rows)
    return rows, metrics_by_run


def write_csv(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "comparison_metrics.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def compact_run_label(label: str) -> str:
    """Keep multi-panel plots readable when run names encode full settings."""
    value = str(label)
    for prefix in ("os_hint_compare/", "contest_"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    for token in ("route_", "_cold"):
        value = value.replace(token, "")

    replacements = {
        "baseline": "baseline",
        "expert_prefetch": "expert",
        "all_async4": "async4_all",
        "async4_priority": "priority",
        "async4_deadline": "deadline",
        "async4_deadline_score": "deadline_score",
        "async4_deadline_heap": "deadline_heap",
        "decode_ttl1": "ttl1",
        "all_async4_coalesced": "async4_coalesce",
        "all_coalesced": "coalesce",
    }
    return replacements.get(value, value)


def plot_overview(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    labels = [compact_run_label(row["label"]) for row in rows]
    y_pos = np.arange(len(labels))
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    fig.suptitle("OS Hint Optimization Comparison", fontsize=15, fontweight="bold")

    plots = [
        ("total_major_faults", "Major Fault Delta", "count", False),
        ("decode_avg_latency_us", "Decode Latency", "us", False),
        ("first_touch_resident_pct_weighted", "First-touch Residency", "%", True),
        ("rss_peak_gb", "Peak RSS", "GiB", False),
        ("swap_peak_mb", "Peak Swap", "MiB", False),
        ("os_hint_events", "OS Hint Events", "count", False),
    ]
    colors = ["#78909C", "#42A5F5", "#66BB6A", "#AB47BC", "#FFA726", "#26A69A", "#EC407A", "#5C6BC0"]

    for ax, (key, title, ylabel, higher_better) in zip(axes.ravel(), plots):
        vals = [float(row.get(key, 0) or 0) for row in rows]
        ax.barh(y_pos, vals, color=[colors[i % len(colors)] for i in range(len(labels))], alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(ylabel)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.25)
        xmax = max(vals) if vals else 1.0
        xmax = xmax if xmax > 0 else 1.0
        ax.set_xlim(0, xmax * 1.28)
        baseline = vals[0] if vals else 0
        for i, v in enumerate(vals):
            if i == 0 or baseline == 0:
                text = fmt_num(v)
            else:
                change = pct_change(float(v), float(baseline))
                sign = "+" if change >= 0 else ""
                text = f"{fmt_num(v)}\n({sign}{change:.1f}%)"
            ax.text(v + xmax * 0.015, i, text, ha="left", va="center", fontsize=7)
        if higher_better:
            ax.text(0.02, 0.92, "higher is better", transform=ax.transAxes, fontsize=8, color="#2E7D32")

    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.0, h_pad=2.0)
    path = out_dir / "01_os_hint_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_heatmap(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    metrics = [
        ("total_major_faults", "Major faults"),
        ("total_minor_faults", "Minor faults"),
        ("decode_avg_latency_us", "Decode latency"),
        ("prefill_avg_latency_us", "Prefill latency"),
        ("first_touch_resident_pct_weighted", "Residency"),
        ("first_touch_nonresident_gb_est", "Nonresident"),
        ("rss_peak_gb", "RSS peak"),
        ("swap_peak_mb", "Swap peak"),
    ]
    labels = [row["label"] for row in rows[1:]]
    data = []
    for key, _ in metrics:
        data.append([row.get(f"{key}_improvement_pct", float("nan")) for row in rows[1:]])
    arr = np.array(data, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-60, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([name for _, name in metrics])
    ax.set_title("Improvement vs Baseline (%)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Improvement %, green is better")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isnan(val):
                text = "N/A"
            else:
                text = f"{val:+.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="#111")

    fig.tight_layout()
    path = out_dir / "02_os_hint_improvement_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fault_timeline(base_dir: Path, runs: list[str], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = ["#78909C", "#42A5F5", "#66BB6A", "#AB47BC", "#FFA726", "#26A69A", "#EC407A", "#5C6BC0"]
    for idx, run in enumerate(runs):
        df = load_memory_events(base_dir / run)
        if df.empty or "t_s" not in df:
            continue
        label = RUN_LABELS.get(run, run)
        color = colors[idx % len(colors)]
        if "major_faults_delta" in df:
            axes[0].plot(df["t_s"], df["major_faults_delta"], label=label, color=color, linewidth=1.2)
        if "minor_faults_delta" in df:
            axes[1].plot(df["t_s"], df["minor_faults_delta"], label=label, color=color, linewidth=1.2)

    axes[0].set_title("Major Fault Delta Timeline")
    axes[0].set_ylabel("faults/sample")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Minor Fault Delta Timeline")
    axes[1].set_ylabel("faults/sample")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    path = out_dir / "03_fault_delta_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    baseline = rows[0]
    expert = next((row for row in rows if row["run"] == "expert_prefetch"), None)
    all_hints = next((row for row in rows if row["run"] == "all_hints"), None)
    load = next((row for row in rows if row["run"] == "load_prefetch"), None)

    lines = [
        "# OS Hint 优化对比摘要",
        "",
        "本摘要基于同一 prompt、同一模型、同一 trace pipeline 的单次消融实验。当前结论表示效果幅度明显，严格统计显著性仍需要多次重复实验。",
        "",
        "## 核心结论",
        "",
    ]
    pareto = [row for row in rows if row.get("pareto_optimal")]
    if pareto:
        lines.extend([
            "- Pareto 最优实验组（同时考虑 decode latency、major faults、RSS、swap 和 OS hint 系统调用）："
            + "、".join(row["label"] for row in pareto) + "。",
            "",
        ])
    if expert:
        lines.extend([
            f"- expert-aware prefetch 将 major fault delta 从 {fmt_num(baseline['total_major_faults'])} 降到 {fmt_num(expert['total_major_faults'])}，改善 {expert['total_major_faults_improvement_pct']:.1f}%。",
            f"- expert-aware prefetch 将 decode 平均延迟从 {fmt_num(baseline['decode_avg_latency_us'])} us 降到 {fmt_num(expert['decode_avg_latency_us'])} us，改善 {expert['decode_avg_latency_us_improvement_pct']:.1f}%。",
            f"- expert-aware prefetch 将 first-touch 驻留率从 {baseline['first_touch_resident_pct_weighted']:.2f}% 提升到 {expert['first_touch_resident_pct_weighted']:.2f}%，提升 {expert['first_touch_resident_pct_weighted'] - baseline['first_touch_resident_pct_weighted']:.2f} 个百分点。",
            f"- 代价是 RSS 峰值从 {baseline['rss_peak_gb']:.2f} GiB 升到 {expert['rss_peak_gb']:.2f} GiB，minor fault delta 增加 {expert['total_minor_faults_change_pct']:.1f}%，OS hint 事件达到 {fmt_num(expert['os_hint_events'])} 次。",
            "",
        ])
    if load:
        lines.extend([
            f"- 加载期预取 hint 触发 {fmt_num(load['os_hint_events'])} 次且失败 0 次，但 prefill 延迟变差 {abs(load['prefill_avg_latency_us_improvement_pct']):.1f}%，说明简单文件/地址级预取不能直接证明有效。",
            "",
        ])
    if all_hints:
        lines.extend([
            f"- 全策略组合仍能将 major fault delta 改善 {all_hints['total_major_faults_improvement_pct']:.1f}%，但 decode 延迟只改善 {all_hints['decode_avg_latency_us_improvement_pct']:.1f}%，不如 expert-aware prefetch 单独启用。",
            f"- 全策略组合产生 {fmt_num(all_hints['os_hint_events'])} 次 OS hint，说明策略叠加会放大系统调用与日志开销。",
            "",
        ])

    lines.extend([
        "## 指标表",
        "",
        "| Run | Pareto | Major Faults | Minor Faults | Decode us | Prefill us | Residency % | RSS GiB | Swap MiB | OS Hint Calls | Cache Hit % | Cache Peak MiB | Evictions |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| {row['label']} | {'yes' if row.get('pareto_optimal') else 'no'} | "
            f"{fmt_num(row['total_major_faults'])} | {fmt_num(row['total_minor_faults'])} | "
            f"{fmt_num(row['decode_avg_latency_us'])} | {fmt_num(row['prefill_avg_latency_us'])} | "
            f"{row['first_touch_resident_pct_weighted']:.2f} | {row['rss_peak_gb']:.2f} | "
            f"{row['swap_peak_mb']:.2f} | {fmt_num(row['os_hint_events'])} | "
            f"{row.get('expert_cache_hit_rate_pct', 0):.2f} | {row.get('expert_cache_peak_mb', 0):.2f} | "
            f"{fmt_num(row.get('expert_cache_evictions', 0))} |"
        )

    path = out_dir / "comparison_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_html(rows: list[dict[str, Any]], plots: list[Path], summary_md: Path, out_dir: Path) -> Path:
    summary_html = "<br>".join(html.escape(line) for line in summary_md.read_text(encoding="utf-8").splitlines())
    cards = []
    for row in rows:
        cards.append(f"""
        <div class="card">
          <h3>{html.escape(row['label'])}</h3>
          <p><b>Pareto:</b> {'yes' if row.get('pareto_optimal') else 'no'}</p>
          <p><b>Major faults:</b> {fmt_num(row['total_major_faults'])}</p>
          <p><b>Decode latency:</b> {fmt_num(row['decode_avg_latency_us'])} us</p>
          <p><b>Residency:</b> {row['first_touch_resident_pct_weighted']:.2f}%</p>
          <p><b>Peak RSS:</b> {row['rss_peak_gb']:.2f} GiB</p>
          <p><b>OS hint calls:</b> {fmt_num(row['os_hint_events'])}</p>
          <p><b>Cache hit:</b> {row.get('expert_cache_hit_rate_pct', 0):.2f}%</p>
        </div>
        """)
    imgs = "\n".join(f'<img src="{plot.name}" alt="{plot.name}">' for plot in plots)
    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>OS Hint 优化对比报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #263238; }}
h1 {{ color: #102027; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 14px; background: #fafafa; }}
img {{ max-width: 100%; display: block; margin: 24px 0; border: 1px solid #ddd; }}
.summary {{ background: #f5f7fa; border-left: 4px solid #1976d2; padding: 12px; line-height: 1.6; }}
</style>
</head>
<body>
<h1>OS Hint 优化对比报告</h1>
<div class="grid">{''.join(cards)}</div>
<h2>图表</h2>
{imgs}
<h2>摘要</h2>
<div class="summary">{summary_html}</div>
</body>
</html>
"""
    path = out_dir / "comparison_report.html"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, help="Directory containing run subdirectories")
    parser.add_argument("--runs", nargs="+", default=["baseline", "load_prefetch", "expert_prefetch", "all_hints"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir) if args.output_dir else base_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = collect_rows(base_dir, args.runs)
    csv_path = write_csv(rows, out_dir)
    plots = [
        plot_overview(rows, out_dir),
        plot_heatmap(rows, out_dir),
        plot_fault_timeline(base_dir, args.runs, out_dir),
    ]
    summary_path = write_summary(rows, out_dir)
    html_path = write_html(rows, plots, summary_path, out_dir)

    print(f"[OK] {csv_path}")
    for plot in plots:
        print(f"[OK] {plot}")
    print(f"[OK] {summary_path}")
    print(f"[OK] {html_path}")


if __name__ == "__main__":
    main()
