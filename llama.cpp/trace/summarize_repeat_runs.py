#!/usr/bin/env python3
"""Summarize repeated LLM memory trace runs.

This script reads per-run analysis/metrics.json files and aggregates selected
metrics by experiment group. It is intended for contest-style validation where
single runs are too noisy to justify a final claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_METRICS = [
    "decode_avg_latency_us",
    "prefill_avg_latency_us",
    "total_major_faults",
    "total_minor_faults",
    "rss_peak_gb",
    "rss_avg_gb",
    "swap_peak_mb",
    "os_hint_events",
    "os_hint_advised_mb",
    "expert_async_enqueued",
    "expert_async_fallback",
    "expert_async_max_queue_depth",
    "expert_route_hint_ttl_skipped",
    "kv_total_mb",
    "kv_mb_per_1k_tokens_est",
]


LOWER_IS_BETTER = {
    "decode_avg_latency_us",
    "prefill_avg_latency_us",
    "total_major_faults",
    "total_minor_faults",
    "rss_peak_gb",
    "rss_avg_gb",
    "swap_peak_mb",
    "os_hint_events",
    "os_hint_advised_mb",
    "expert_async_fallback",
}

ZERO_DEFAULT_METRICS = {
    "os_hint_events",
    "os_hint_advised_mb",
    "expert_async_enqueued",
    "expert_async_fallback",
    "expert_async_max_queue_depth",
    "expert_async_priority_pops",
    "expert_async_priority_heap_pops",
    "expert_route_hint_ttl_skipped",
    "expert_route_hint_candidates",
    "expert_route_hint_issued",
}


def parse_group(value: str) -> tuple[str, list[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--group must use name=run1,run2 syntax")
    name, runs_text = value.split("=", 1)
    name = name.strip()
    runs = [run.strip() for run in runs_text.split(",") if run.strip()]
    if not name or not runs:
        raise argparse.ArgumentTypeError("--group requires a non-empty name and at least one run")
    return name, runs


def load_metrics(base_dir: Path, run: str) -> dict[str, Any]:
    path = base_dir / run / "analysis" / "metrics.json"
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    for key in ZERO_DEFAULT_METRICS:
        metrics.setdefault(key, 0)
    metrics["_run"] = run
    return metrics


def numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, bool):
            out.append(float(value))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "cv_pct": float("nan"),
        }
    avg = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    return {
        "n": float(len(values)),
        "mean": avg,
        "std": sd,
        "min": min(values),
        "max": max(values),
        "cv_pct": abs(sd / avg * 100.0) if avg else 0.0,
    }


def format_number(value: float) -> str:
    if math.isnan(value):
        return ""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.2f}"
    return f"{value:,.4f}"


def build_summary_rows(
        groups: list[tuple[str, list[str]]],
        group_records: dict[str, list[dict[str, Any]]],
        metrics: list[str],
        baseline_group: str | None) -> list[dict[str, Any]]:
    baseline_means: dict[str, float] = {}
    if baseline_group and baseline_group in group_records:
        for metric in metrics:
            values = numeric_values(group_records[baseline_group], metric)
            if values:
                baseline_means[metric] = mean(values)

    rows: list[dict[str, Any]] = []
    for group_name, runs in groups:
        records = group_records[group_name]
        for metric in metrics:
            values = numeric_values(records, metric)
            summary = summarize_values(values)
            row: dict[str, Any] = {
                "group": group_name,
                "metric": metric,
                "runs": ";".join(runs),
                "n": int(summary["n"]),
                "mean": summary["mean"],
                "std": summary["std"],
                "min": summary["min"],
                "max": summary["max"],
                "cv_pct": summary["cv_pct"],
            }
            baseline_mean = baseline_means.get(metric)
            if baseline_mean is not None and not math.isnan(summary["mean"]):
                row["baseline_mean"] = baseline_mean
                row["change_pct_vs_baseline"] = (
                    (summary["mean"] - baseline_mean) / baseline_mean * 100.0
                    if baseline_mean else float("nan")
                )
                if metric in LOWER_IS_BETTER:
                    row["improvement_pct_vs_baseline"] = (
                        (baseline_mean - summary["mean"]) / baseline_mean * 100.0
                        if baseline_mean else float("nan")
                    )
                else:
                    row["improvement_pct_vs_baseline"] = (
                        (summary["mean"] - baseline_mean) / baseline_mean * 100.0
                        if baseline_mean else float("nan")
                    )
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group",
        "metric",
        "runs",
        "n",
        "mean",
        "std",
        "min",
        "max",
        "cv_pct",
        "baseline_mean",
        "change_pct_vs_baseline",
        "improvement_pct_vs_baseline",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, rows: list[dict[str, Any]], metrics: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Repeat Run Summary")
    lines.append("")
    lines.append("This summary aggregates per-run `analysis/metrics.json` files.")
    lines.append("")
    for metric in metrics:
        metric_rows = [row for row in rows if row["metric"] == metric]
        if not metric_rows:
            continue
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| Group | N | Mean | Std | Min | Max | CV % | Improvement vs Baseline % |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in metric_rows:
            lines.append(
                "| {group} | {n} | {mean} | {std} | {min} | {max} | {cv} | {imp} |".format(
                    group=row["group"],
                    n=row["n"],
                    mean=format_number(float(row["mean"])),
                    std=format_number(float(row["std"])),
                    min=format_number(float(row["min"])),
                    max=format_number(float(row["max"])),
                    cv=format_number(float(row["cv_pct"])),
                    imp=format_number(float(row.get("improvement_pct_vs_baseline", float("nan")))),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True, help="Base trace output directory")
    parser.add_argument("--group", action="append", type=parse_group, required=True,
                        help="Experiment group in name=run1,run2 syntax")
    parser.add_argument("--metric", action="append", default=None,
                        help="Metric to aggregate; can be passed multiple times")
    parser.add_argument("--baseline-group", default=None,
                        help="Optional group name used for relative improvement columns")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metrics = args.metric if args.metric else DEFAULT_METRICS
    groups: list[tuple[str, list[str]]] = args.group

    group_records: dict[str, list[dict[str, Any]]] = {}
    for group_name, runs in groups:
        group_records[group_name] = [load_metrics(args.base_dir, run) for run in runs]

    rows = build_summary_rows(groups, group_records, metrics, args.baseline_group)
    write_csv(args.output_dir / "repeat_summary.csv", rows)
    write_markdown(args.output_dir / "repeat_summary.md", rows, metrics)
    print(f"[OK] {args.output_dir / 'repeat_summary.csv'}")
    print(f"[OK] {args.output_dir / 'repeat_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
