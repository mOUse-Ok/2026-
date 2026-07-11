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
    "process_wall_time_s",
    "decode_avg_latency_us",
    "decode_p95_latency_us",
    "decode_p99_latency_us",
    "decode_throughput_tokens_per_s",
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
    "process_wall_time_s",
    "decode_avg_latency_us",
    "decode_p95_latency_us",
    "decode_p99_latency_us",
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_runs(base_dir: Path, runs: list[str]) -> dict[str, Any]:
    required = (
        "run_manifest.json",
        "cache_preparation.json",
        "process_metrics.json",
        "summary.json",
        "output.sha256",
        "analysis/metrics.json",
    )
    validations: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    output_hashes: set[str] = set()

    for run in runs:
        run_dir = base_dir / run
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            raise ValueError(f"{run}: missing required artifacts: {', '.join(missing)}")

        manifest = load_json(run_dir / "run_manifest.json")
        cache = load_json(run_dir / "cache_preparation.json")
        process = load_json(run_dir / "process_metrics.json")
        summary = load_json(run_dir / "summary.json")
        metrics = load_json(run_dir / "analysis" / "metrics.json")
        output_hash = (run_dir / "output.sha256").read_text(encoding="ascii").strip()
        if len(output_hash) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in output_hash):
            raise ValueError(f"{run}: invalid output.sha256")
        output_hashes.add(output_hash.lower())

        if manifest.get("git_dirty"):
            raise ValueError(f"{run}: manifest records a dirty repository")
        if int(process.get("exit_code", 1)) != 0:
            raise ValueError(f"{run}: process exit code is not zero")
        if metrics.get("fault_metric_source") != "gnu_time_process":
            raise ValueError(f"{run}: whole-process fault metrics are missing")
        if metrics.get("latency_metric_source") != "step_end":
            raise ValueError(f"{run}: STEP_END latency metrics are missing")
        if int(metrics.get("decode_steps", 0)) <= 0:
            raise ValueError(f"{run}: decode STEP_END samples are missing")

        sinks = summary.get("sinks")
        if not isinstance(sinks, dict) or not sinks:
            raise ValueError(f"{run}: trace sink accounting is missing")
        for sink, counts in sinks.items():
            if not isinstance(counts, dict) or not counts.get("enabled"):
                continue
            if int(counts.get("dropped", 0)) != 0:
                raise ValueError(f"{run}: {sink} trace dropped events")
            if int(counts.get("enqueued", 0)) != int(counts.get("written", 0)):
                raise ValueError(f"{run}: {sink} trace write count mismatch")

        model = manifest.get("model", {})
        prompt = manifest.get("prompt", {})
        binary = manifest.get("binary", {})
        experiment = manifest.get("experiment", {})
        environment = manifest.get("environment", {})
        cgroup = experiment.get("cgroup", {})
        if cache.get("mode") != experiment.get("cache_mode"):
            raise ValueError(f"{run}: cache preparation and manifest disagree")
        identity = {
            "git_commit": manifest.get("git_commit"),
            "model_sha256": model.get("sha256"),
            "model_size_bytes": model.get("size_bytes"),
            "model_mtime_ns": model.get("mtime_ns") if not model.get("sha256") else None,
            "prompt_sha256": prompt.get("sha256"),
            "binary_sha256": binary.get("sha256"),
            "host": manifest.get("host"),
            "trace_profile": experiment.get("trace_profile"),
            "cache_mode": experiment.get("cache_mode"),
            "requested_memory_max": experiment.get("requested_memory_max"),
            "requested_memory_swap_max": experiment.get("requested_memory_swap_max"),
            "actual_cgroup_memory_max": cgroup.get("memory.max"),
            "actual_cgroup_memory_swap_max": cgroup.get("memory.swap.max"),
            "workload": {
                key: environment.get(key)
                for key in (
                    "NUM_TOKENS_PREDICT",
                    "NUM_THREADS",
                    "BATCH_SIZE",
                    "CTX_SIZE",
                    "TEMP",
                    "SEED",
                    "GPU_LAYERS",
                )
            },
        }
        fingerprints.add(json.dumps(identity, sort_keys=True, ensure_ascii=False))
        validations.append({
            "run": run,
            "repeat_index": experiment.get("repeat_index"),
            "order_position": experiment.get("order_position"),
            "output_sha256": output_hash.lower(),
            "identity": identity,
        })

    if len(fingerprints) != 1:
        raise ValueError("run manifests disagree on code, model, binary, host, cache, cgroup, or workload")
    if len(output_hashes) != 1:
        raise ValueError("deterministic output hashes disagree across runs/groups")
    return {
        "valid": True,
        "run_count": len(runs),
        "output_sha256": next(iter(output_hashes)),
        "runs": validations,
    }


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

    all_runs = [run for _, runs in groups for run in runs]
    validation = validate_runs(args.base_dir, all_runs)

    group_records: dict[str, list[dict[str, Any]]] = {}
    for group_name, runs in groups:
        group_records[group_name] = [load_metrics(args.base_dir, run) for run in runs]

    rows = build_summary_rows(groups, group_records, metrics, args.baseline_group)
    write_csv(args.output_dir / "repeat_summary.csv", rows)
    write_markdown(args.output_dir / "repeat_summary.md", rows, metrics)
    (args.output_dir / "experiment_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] {args.output_dir / 'repeat_summary.csv'}")
    print(f"[OK] {args.output_dir / 'repeat_summary.md'}")
    print(f"[OK] {args.output_dir / 'experiment_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
