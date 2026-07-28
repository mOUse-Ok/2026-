#!/usr/bin/env python3
"""Validate and summarize M5A Pressure Shadow engineering/evidence artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


REQUIRED_SAMPLE_FIELDS = (
    "phase",
    "step",
    "memory_current_bytes",
    "memory_high_bytes",
    "memory_max_bytes",
    "swap_current_bytes",
    "swap_max_bytes",
    "memory_events_high",
    "memory_events_high_delta",
    "anon_bytes",
    "file_bytes",
    "workingset_refault_anon",
    "workingset_refault_file",
    "workingset_refault_anon_delta",
    "workingset_refault_file_delta",
    "cgroup_pgfault",
    "cgroup_pgmajfault",
    "pgscan",
    "pgsteal",
    "pgscan_delta",
    "pgsteal_delta",
    "psi_some_avg10",
    "psi_full_avg10",
    "psi_some_total_us",
    "psi_full_total_us",
    "psi_some_delta_us",
    "psi_full_delta_us",
    "process_rss_bytes",
    "process_vms_bytes",
    "process_pss_bytes",
    "process_pss_age_ns",
    "process_minor_faults",
    "process_major_faults",
    "process_minor_faults_delta",
    "process_major_faults_delta",
    "queue_depth",
    "queued_bytes",
    "queue_started",
    "queue_stopping",
    "queue_status",
    "configured_worker_count",
    "worker_count",
    "busy_workers",
    "current_step_issued_bytes",
    "current_step_hint_calls",
    "current_step_advised_bytes",
)
CORE_AVAILABILITY_FIELDS = (
    "memory_current_bytes",
    "memory_max_bytes",
    "swap_current_bytes",
    "memory_events_high",
    "workingset_refault_anon",
    "workingset_refault_file",
    "cgroup_pgmajfault",
    "pgscan",
    "pgsteal",
    "psi_some_total_us",
    "psi_full_total_us",
    "process_major_faults",
)
QUEUE_CORE_FIELDS = (
    "queue_depth",
    "queued_bytes",
    "worker_count",
    "busy_workers",
)
VALID_STATUSES = {
    "available",
    "not_sampled",
    "unavailable",
    "permission_denied",
    "field_missing",
    "parse_error",
    "io_error",
    "unsupported",
    "no_previous_sample",
    "counter_regression",
    "source_changed",
    "source_stale",
    "not_started",
    "stopping",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def percentile(values: Sequence[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "min": min(finite) if finite else None,
        "p25": percentile(finite, 25),
        "median": percentile(finite, 50),
        "mean": statistics.mean(finite) if finite else None,
        "p95": percentile(finite, 95),
        "max": max(finite) if finite else None,
    }


def wilson(successes: int, total: int, z: float = 1.96) -> dict[str, float | None]:
    if total <= 0:
        return {"low": None, "high": None}
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def trace_integrity(summary: dict[str, Any]) -> dict[str, Any]:
    failures = {}
    for name, sink in summary.get("sinks", {}).items():
        if not sink.get("enabled"):
            continue
        if sink.get("enqueued") != sink.get("written") or sink.get("dropped") != 0:
            failures[name] = sink
    return {"passed": not failures, "failures": failures}


def validate_detail_run(run_id: str, run_info: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run_info["run_dir"])
    records = load_jsonl(run_dir / "memory_trace.jsonl")
    samples = [
        record for record in records
        if record.get("event") == "PRESSURE_SHADOW_SAMPLE"
    ]
    runtime_summaries = [
        record for record in records
        if record.get("event") == "PRESSURE_SHADOW_SUMMARY"
    ]
    manifest = load_json(run_dir / "run_manifest.json")
    trace_summary = load_json(run_dir / "summary.json")
    metrics = load_json(run_dir / "analysis" / "metrics.json")
    errors: Counter[str] = Counter()
    status_counts: dict[str, Counter[str]] = {
        field: Counter() for field in REQUIRED_SAMPLE_FIELDS
    }
    timestamps: list[int] = []
    schema_error_count = 0
    causal_timestamp_errors = 0
    run_id_errors = 0
    busy_worker_errors = 0
    sequence_errors = 0
    source_errors = 0
    previous_sequence: int | None = None
    previous_ready: int | None = None
    cgroup_paths: set[str] = set()
    for sample in samples:
        ready_ts = sample.get("sample_ready_ts_ns")
        if isinstance(ready_ts, int):
            timestamps.append(ready_ts)
        else:
            schema_error_count += 1
        if sample.get("run_id") != run_id:
            run_id_errors += 1
        sequence = sample.get("sample_seq")
        start_ts = sample.get("sample_start_ts_ns")
        target_interval = sample.get("target_interval_ns")
        if (
            sample.get("schema_version") != 1
            or not isinstance(sequence, int)
            or not isinstance(start_ts, int)
            or not isinstance(ready_ts, int)
            or start_ts > ready_ts
            or sample.get("ts_ns") != ready_ts
            or not isinstance(target_interval, int)
            or not 10_000_000 <= target_interval <= 50_000_000
            or not isinstance(sample.get("deadline_lateness_ns"), int)
            or not isinstance(sample.get("missed_samples_since_previous"), int)
            or not isinstance(sample.get("sample_wall_time_ns"), int)
            or not isinstance(sample.get("sample_thread_cpu_time_ns"), int)
        ):
            schema_error_count += 1
        if (
            previous_sequence is not None
            and isinstance(sequence, int)
            and sequence != previous_sequence + 1
        ):
            sequence_errors += 1
        if previous_ready is not None and isinstance(ready_ts, int):
            if sample.get("actual_interval_ns") != ready_ts - previous_ready:
                sequence_errors += 1
        elif sample.get("actual_interval_ns") is not None:
            sequence_errors += 1
        if isinstance(sequence, int):
            previous_sequence = sequence
        if isinstance(ready_ts, int):
            previous_ready = ready_ts

        sources = sample.get("sources")
        cgroup_source = (
            sources.get("cgroup_memory")
            if isinstance(sources, dict)
            else None
        )
        if (
            not isinstance(cgroup_source, dict)
            or cgroup_source.get("scope") != "current_process_cgroup"
        ):
            source_errors += 1
        elif isinstance(cgroup_source.get("path"), str):
            cgroup_paths.add(cgroup_source["path"])
        else:
            source_errors += 1
        for field_name in REQUIRED_SAMPLE_FIELDS:
            field = sample.get(field_name)
            if not isinstance(field, dict):
                schema_error_count += 1
                status_counts[field_name]["field_missing"] += 1
                continue
            required_keys = {"value", "status", "read_ts_ns", "error", "run_id"}
            if not required_keys.issubset(field):
                schema_error_count += 1
            if field.get("run_id") != run_id:
                run_id_errors += 1
            status = str(field.get("status"))
            status_counts[field_name][status] += 1
            if status not in VALID_STATUSES:
                schema_error_count += 1
            if status != "available" and field.get("value") is not None:
                schema_error_count += 1
            read_ts = field.get("read_ts_ns")
            if (
                isinstance(ready_ts, int)
                and isinstance(read_ts, int)
                and read_ts > ready_ts
            ):
                causal_timestamp_errors += 1
        workers = sample.get("worker_count", {})
        busy = sample.get("busy_workers", {})
        if (
            workers.get("status") == "available"
            and busy.get("status") == "available"
            and busy.get("value", 0) > workers.get("value", 0)
        ):
            busy_worker_errors += 1

    intervals = [
        following - current
        for current, following in zip(timestamps, timestamps[1:])
        if following > current
    ]
    runtime_summary = runtime_summaries[-1] if len(runtime_summaries) == 1 else {}
    target_interval_ns = int(runtime_summary.get("sample_interval_ms", 25)) * 1_000_000
    duration_ns = (
        runtime_summary.get("stopped_ts_ns", 0)
        - runtime_summary.get("started_ts_ns", 0)
    )
    sampler_cpu_ns = runtime_summary.get("sampler_cpu_cost", {}).get("total_ns")
    pss_cpu_ns = runtime_summary.get("pss_sampler", {}).get("cpu_total_ns")
    if isinstance(sampler_cpu_ns, int) and isinstance(pss_cpu_ns, int):
        sampler_cpu_ns += pss_cpu_ns
    sampler_cpu_duty = (
        sampler_cpu_ns / duration_ns
        if isinstance(sampler_cpu_ns, int) and duration_ns > 0
        else None
    )
    availability = {
        field: {
            "available": counts.get("available", 0),
            "total": sum(counts.values()),
            "rate": (
                counts.get("available", 0) / sum(counts.values())
                if sum(counts.values()) else None
            ),
            "statuses": dict(counts),
        }
        for field, counts in status_counts.items()
    }
    environment = manifest.get("environment", {})
    active_control_off = (
        environment.get("LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER") == "off"
        and environment.get("LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK") == "0"
        and environment.get("LLM_MEM_TRACE_OPT_EXPERT_SLACK") == "0"
        and environment.get("LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE") == "0"
        and environment.get("LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT") == "0"
    )
    cgroup = manifest.get("experiment", {}).get("cgroup", {})
    cgroup_source_matches = (
        runtime_summary.get("cgroup_path") == cgroup.get("source_path")
    )
    cgroup_limited = (
        cgroup.get("memory.max") not in (None, "max")
        and cgroup.get("memory.swap.max") not in (None, "max")
    )
    integrity = trace_integrity(trace_summary)
    reject_cancel_zero = all(
        int(metrics.get(name, -1)) == 0
        for name in (
            "expert_task_rejected",
            "expert_task_cancelled",
            "expert_task_invalid_transitions",
        )
    )
    core_availability_stable = all(
        availability[field]["rate"] is not None
        and availability[field]["rate"] >= 0.99
        for field in CORE_AVAILABILITY_FIELDS
    )
    queue_active_indexes = [
        index for index, sample in enumerate(samples)
        if sample.get("queue_started", {}).get("status") == "available"
        and sample.get("queue_started", {}).get("value") == 1
    ]
    queue_availability_after_start = bool(queue_active_indexes) and all(
        sum(
            samples[index].get(field, {}).get("status") == "available"
            for index in queue_active_indexes
        ) / len(queue_active_indexes) >= 0.99
        for field in QUEUE_CORE_FIELDS
    )
    interval_ok = (
        not intervals
        or max(intervals) <= 2 * target_interval_ns
    )
    checks = {
        "samples_present": bool(samples),
        "single_runtime_summary": len(runtime_summaries) == 1,
        "summary_sample_count_matches": (
            runtime_summary.get("sample_count") == len(samples)
        ),
        "summary_detail_count_matches": (
            runtime_summary.get("detail_events") == len(samples)
        ),
        "sample_run_id_consistent": run_id_errors == 0,
        "sample_sequence_and_interval_valid": sequence_errors == 0,
        "source_scope_and_path_stable": (
            source_errors == 0 and len(cgroup_paths) == 1
        ),
        "observation_schema_valid": schema_error_count == 0,
        "read_timestamps_not_after_ready": causal_timestamp_errors == 0,
        "sample_timestamps_strictly_increasing": (
            len(timestamps) == len(samples)
            and all(a < b for a, b in zip(timestamps, timestamps[1:]))
        ),
        "no_interval_gap_over_2x_target": interval_ok,
        "no_missed_intervals": runtime_summary.get("missed_intervals") == 0,
        "busy_workers_bounded": busy_worker_errors == 0,
        "core_availability_at_least_99pct": core_availability_stable,
        "queue_availability_while_started_at_least_99pct":
            queue_availability_after_start,
        "active_control_off": active_control_off,
        "cgroup_source_matches_manifest": cgroup_source_matches,
        "cgroup_limits_finite": cgroup_limited,
        "no_legacy_active_pressure_events": not any(
            record.get("event") == "EXPERT_PRESSURE" for record in records
        ),
        "reject_cancel_invalid_zero": reject_cancel_zero,
        "trace_zero_drop_and_drained": integrity["passed"],
        "sampler_cpu_duty_below_5pct": (
            sampler_cpu_duty is not None and sampler_cpu_duty < 0.05
        ),
    }
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "passed": all(checks.values()),
        "checks": checks,
        "sample_count": len(samples),
        "runtime_summary": runtime_summary,
        "interval_ns": distribution(intervals),
        "target_interval_ns": target_interval_ns,
        "sampler_cpu_duty": sampler_cpu_duty,
        "availability": availability,
        "schema_error_count": schema_error_count,
        "causal_timestamp_errors": causal_timestamp_errors,
        "run_id_errors": run_id_errors,
        "busy_worker_errors": busy_worker_errors,
        "sequence_errors": sequence_errors,
        "source_errors": source_errors,
        "trace_integrity": integrity,
        "git_dirty": manifest.get("git_dirty"),
        "workers": run_info.get("workers"),
        "memory_max": run_info.get("memory_max"),
        "memory_swap_max": run_info.get("memory_swap_max"),
        "errors": dict(errors),
    }


def paired_overhead(
    off_dirs: Sequence[Path],
    summary_dirs: Sequence[Path],
) -> dict[str, Any]:
    if len(off_dirs) != len(summary_dirs):
        raise ValueError("--overhead-off and --overhead-summary counts must match")
    pairs = []
    ratios: dict[str, list[float]] = {
        "wall_time": [],
        "cpu_time": [],
        "max_rss": [],
        "major_faults": [],
    }
    for off_dir, summary_dir in zip(off_dirs, summary_dirs):
        off = load_json(off_dir / "process_metrics.json")
        summary = load_json(summary_dir / "process_metrics.json")
        off_wall = float(off["wall_time_s"])
        summary_wall = float(summary["wall_time_s"])
        off_cpu = float(off["user_time_s"]) + float(off["system_time_s"])
        summary_cpu = float(summary["user_time_s"]) + float(summary["system_time_s"])
        pair_ratios = {
            "wall_time": summary_wall / off_wall if off_wall > 0 else None,
            "cpu_time": summary_cpu / off_cpu if off_cpu > 0 else None,
            "max_rss": (
                float(summary["max_rss_kb"]) / float(off["max_rss_kb"])
                if float(off["max_rss_kb"]) > 0 else None
            ),
            "major_faults": (
                float(summary["major_faults"]) / float(off["major_faults"])
                if float(off["major_faults"]) > 0 else None
            ),
        }
        for name, ratio in pair_ratios.items():
            if ratio is not None:
                ratios[name].append(ratio)
        pairs.append(
            {
                "off_dir": str(off_dir),
                "summary_dir": str(summary_dir),
                "off": off,
                "summary": summary,
                "summary_over_off_ratio": pair_ratios,
            }
        )
    return {
        "pair_count": len(pairs),
        "pairs": pairs,
        "ratio_distribution": {
            name: distribution(values) for name, values in ratios.items()
        },
    }


def aggregate_outcomes(analysis: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, dict[str, Any]] = {}
    for fold in analysis.get("folds", []):
        selected_name = fold.get("selected_candidate", {}).get("name")
        for run_id, evaluation in fold.get("evaluation", {}).items():
            metrics = evaluation.get("metrics", {}).get("100ms", {})
            for outcome, value in metrics.items():
                key = totals.setdefault(
                    outcome,
                    {
                        "candidate_names": set(),
                        "tp": 0,
                        "fp": 0,
                        "tn": 0,
                        "fn": 0,
                        "eligible": 0,
                        "unavailable": 0,
                        "episode_count": 0,
                        "hit_episode_count": 0,
                        "outcome_event_count": 0,
                        "hit_outcome_event_count": 0,
                        "lead_times_ns": [],
                        "task_opportunity_hits": 0,
                        "run_metrics": {},
                    },
                )
                key["candidate_names"].add(selected_name)
                for name in ("tp", "fp", "tn", "fn", "eligible", "unavailable"):
                    key[name] += int(value.get(name, 0))
                episode = value.get("episode", {})
                key["episode_count"] += int(episode.get("high_episode_count", 0))
                key["hit_episode_count"] += int(episode.get("hit_episode_count", 0))
                key["outcome_event_count"] += int(episode.get("outcome_event_count", 0))
                key["hit_outcome_event_count"] += int(
                    episode.get("hit_outcome_event_count", 0)
                )
                for row in episode.get("episodes", []):
                    if isinstance(row.get("lead_time_ns"), (int, float)):
                        key["lead_times_ns"].append(float(row["lead_time_ns"]))
                    if row.get("task_opportunity_count", 0) > 0:
                        key["task_opportunity_hits"] += 1
                key["run_metrics"][run_id] = value
    result: dict[str, Any] = {}
    for outcome, total in totals.items():
        tp, fp, tn, fn = (total[name] for name in ("tp", "fp", "tn", "fn"))
        ratio = lambda a, b: a / b if b else None
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        result[outcome] = {
            **{name: total[name] for name in (
                "tp", "fp", "tn", "fn", "eligible", "unavailable",
                "episode_count", "hit_episode_count", "outcome_event_count",
                "hit_outcome_event_count", "task_opportunity_hits",
            )},
            "candidate_names": sorted(total["candidate_names"]),
            "precision": precision,
            "precision_wilson_95": wilson(tp, tp + fp),
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall > 0
                else None
            ),
            "specificity": ratio(tn, tn + fp),
            "false_alarm_rate": ratio(fp, fp + tn),
            "episode_precision": ratio(
                total["hit_episode_count"], total["episode_count"]
            ),
            "outcome_event_recall": ratio(
                total["hit_outcome_event_count"], total["outcome_event_count"]
            ),
            "lead_time_ns": distribution(total["lead_times_ns"]),
            "run_metrics": total["run_metrics"],
        }
    return result


def render_report(summary: dict[str, Any]) -> str:
    conclusion = summary["conclusion"]
    lines = [
        "# M5A Pressure Shadow Report",
        "",
        "> Observation and offline analysis only. No active pressure control was implemented.",
        "",
        "## Conclusion",
        "",
        f"- Result: **{conclusion['result']}**",
        f"- Evidence class: `{conclusion['evidence_class']}`",
        f"- Reason: {conclusion['reason']}",
        f"- M5B started: `false`",
        "",
        "## Engineering validation",
        "",
        f"- detail runs: `{summary['validation']['run_count']}`",
        f"- valid detail runs: `{summary['validation']['passed_run_count']}`",
        f"- equivalence pairs: `{summary['equivalence']['pair_count']}`",
        f"- equivalence passed: `{str(summary['equivalence']['passed']).lower()}`",
        f"- overhead pairs: `{summary['overhead']['pair_count']}`",
        "",
        "## 100 ms risk metrics",
        "",
        "| Outcome | Candidate | TP | FP | FN | Precision | Recall | F1 | Events | Lead median ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for outcome, metrics in sorted(summary["outcomes_100ms"].items()):
        def fmt(value: Any) -> str:
            return "-" if value is None else f"{value:.3f}"
        lead = metrics["lead_time_ns"]["median"]
        lines.append(
            f"| `{outcome}` | `{','.join(metrics['candidate_names'])}` | "
            f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
            f"{fmt(metrics['precision'])} | {fmt(metrics['recall'])} | "
            f"{fmt(metrics['f1'])} | {metrics['outcome_event_count']} | "
            f"{'-' if lead is None else f'{lead / 1_000_000:.3f}'} |"
        )
    lines.extend(
        [
            "",
            "## All preregistered three-state candidate families",
            "",
        ]
    )
    for name, description in sorted(
        summary["candidate"].get("candidate_families", {}).items()
    ):
        lines.append(f"- `{name}`: {description}")
    lines.extend(
        [
            "",
            "## Run-separated folds",
            "",
            "| Fold | Status | Training runs | Evaluation runs | Selected candidate |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for fold in summary["candidate"].get("folds", []):
        selected = fold.get("selected_candidate", {}).get("name", "-")
        lines.append(
            f"| `{fold.get('fold_id', '-')}` | `{fold.get('status', '-')}` | "
            f"{len(fold.get('training_run_ids', []))} | "
            f"{len(fold.get('evaluation_run_ids', []))} | `{selected}` |"
        )
    lines.extend(
        [
            "",
            "## Per-run 100 ms risk metrics",
            "",
            "| Run | Workers | memory.max | Outcome | TP | FP | FN | Precision | Recall | F1 | Coverage |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    validations = summary["validation"]["runs"]
    for outcome, metrics in sorted(summary["outcomes_100ms"].items()):
        for run_id, run_metrics in sorted(metrics.get("run_metrics", {}).items()):
            run = validations.get(run_id, {})
            def run_fmt(value: Any) -> str:
                return "-" if value is None else f"{value:.3f}"
            lines.append(
                f"| `{run_id}` | {run.get('workers', '-')} | "
                f"`{run.get('memory_max', '-')}` | `{outcome}` | "
                f"{run_metrics.get('tp', 0)} | {run_metrics.get('fp', 0)} | "
                f"{run_metrics.get('fn', 0)} | "
                f"{run_fmt(run_metrics.get('precision'))} | "
                f"{run_fmt(run_metrics.get('recall'))} | "
                f"{run_fmt(run_metrics.get('f1'))} | "
                f"{run_fmt(run_metrics.get('high_sample_coverage'))} |"
            )
    lines.extend(
        [
            "",
            "## Sampler overhead",
            "",
            f"- summary/off wall ratio median: `{summary['overhead']['ratio_distribution']['wall_time']['median']}`",
            f"- summary/off CPU ratio median: `{summary['overhead']['ratio_distribution']['cpu_time']['median']}`",
            "- Per-run sampler CPU duty, interval distribution, source availability, and trace integrity are in `M5A_validation.json`.",
            "",
            "## Boundaries",
            "",
            "- Runtime states remain unavailable by design; LOW/MEDIUM/HIGH candidates exist only in offline artifacts.",
            "- The legacy active `EXPERT_PRESSURE` controller was required to remain off.",
            "- Outcome association is predictive correlation under strict future windows, not a causal claim.",
            "- This report stops at M5A and does not authorize or implement M5B.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(
    analysis_path: Path,
    candidate_path: Path,
    equivalence_paths: Sequence[Path],
    overhead_off: Sequence[Path],
    overhead_summary: Sequence[Path],
) -> dict[str, Any]:
    analysis = load_json(analysis_path)
    if analysis.get("analysis") == "M5A_pressure_shadow_complete":
        analysis = analysis["offline_analysis"]
    candidate = load_json(candidate_path)
    validations = {
        run_id: validate_detail_run(run_id, run_info)
        for run_id, run_info in analysis.get("runs", {}).items()
    }
    equivalences = [load_json(path) for path in equivalence_paths]
    equivalence_passed = bool(equivalences) and all(
        item.get("passed") for item in equivalences
    )
    overhead = paired_overhead(overhead_off, overhead_summary)
    outcomes = aggregate_outcomes(analysis)
    fold_separated = bool(analysis.get("folds")) and all(
        fold.get("status") == "run_separated" for fold in analysis["folds"]
    )
    all_clean = bool(validations) and all(
        item.get("git_dirty") is False for item in validations.values()
    )
    workers = {
        item.get("workers") for item in validations.values()
        if item.get("workers") is not None
    }
    memory_limits = {
        item.get("memory_max") for item in validations.values()
        if item.get("memory_max") is not None
    }
    formal_ready = (
        all_clean
        and len(validations) >= 32
        and len(overhead_off) >= 16
        and workers.issuperset({2, 4})
        and len(memory_limits) >= 2
        and fold_separated
    )
    predictive_outcomes = [
        name for name, metrics in outcomes.items()
        if metrics.get("precision") is not None
        and metrics["precision"] >= 0.80
        and metrics.get("recall") is not None
        and metrics["recall"] > 0
        and metrics.get("outcome_event_count", 0) >= 3
        and metrics.get("lead_time_ns", {}).get("median") is not None
        and metrics["lead_time_ns"]["median"] > 0
        and metrics.get("task_opportunity_hits", 0) > 0
    ]
    selected_candidate_names = {
        fold.get("selected_candidate", {}).get("name")
        for fold in analysis.get("folds", [])
        if fold.get("selected_candidate", {}).get("name")
    }
    consistent_candidate = len(selected_candidate_names) == 1
    engineering_passed = (
        equivalence_passed
        and bool(validations)
        and all(item["passed"] for item in validations.values())
    )
    if not engineering_passed:
        failures = []
        if not equivalences:
            failures.append("strict off/summary equivalence evidence is missing")
        elif not equivalence_passed:
            failed_checks = sorted({
                name
                for item in equivalences
                for name, passed in item.get("checks", {}).items()
                if not passed
            })
            failures.append(
                "strict off/summary equivalence failed"
                + (
                    f" ({', '.join(failed_checks)})"
                    if failed_checks else ""
                )
            )
        if not validations:
            failures.append("no detail runtime validation is available")
        else:
            failed_runs = sorted(
                run_id for run_id, item in validations.items()
                if not item["passed"]
            )
            if failed_runs:
                failures.append(
                    f"detail runtime validation failed: {', '.join(failed_runs)}"
                )
        conclusion = {
            "result": "当前环境不支持可靠 Pressure Shadow",
            "result_code": "engineering_failed",
            "evidence_class": "engineering",
            "reason": "; ".join(failures),
            "m5b_reference_conditions_met": False,
        }
    elif not formal_ready:
        conclusion = {
            "result": "工程通过，但需要更多正式受限内存 Evidence",
            "result_code": "engineering_complete_needs_formal_evidence",
            "evidence_class": "engineering/informal",
            "reason": "engineering checks passed, but clean N=8 two-limit formal matrix is incomplete",
            "m5b_reference_conditions_met": False,
        }
    elif len(predictive_outcomes) < 2 or not consistent_candidate:
        conclusion = {
            "result": "信号存在但预测能力不足",
            "result_code": "insufficient_evidence",
            "evidence_class": "formal",
            "reason": (
                "fewer than two independent Outcomes met precision, recall, event-count, and positive-lead references"
                if len(predictive_outcomes) < 2
                else "training folds did not select one consistent interpretable candidate"
            ),
            "m5b_reference_conditions_met": False,
        }
    else:
        conclusion = {
            "result": "Pressure Shadow 支持提出 M5B 候选，等待人工批准",
            "result_code": "shadow_candidate_for_human_review",
            "evidence_class": "formal",
            "reason": "at least two independent Outcomes met the preregistered reference checks; human review is still required",
            "m5b_reference_conditions_met": True,
        }
    return {
        "schema_version": 1,
        "analysis": "M5A_pressure_shadow_summary",
        "validation": {
            "run_count": len(validations),
            "passed_run_count": sum(item["passed"] for item in validations.values()),
            "all_passed": bool(validations) and all(
                item["passed"] for item in validations.values()
            ),
            "runs": validations,
        },
        "equivalence": {
            "pair_count": len(equivalences),
            "passed": equivalence_passed,
            "pairs": equivalences,
        },
        "overhead": overhead,
        "outcomes_100ms": outcomes,
        "candidate": candidate,
        "formal_readiness": {
            "ready": formal_ready,
            "all_detail_manifests_clean": all_clean,
            "detail_run_count": len(validations),
            "overhead_pair_count": len(overhead_off),
            "workers": sorted(workers),
            "memory_limits": sorted(memory_limits),
            "folds_run_separated": fold_separated,
            "selected_candidate_names": sorted(selected_candidate_names),
            "consistent_selected_candidate": consistent_candidate,
        },
        "predictive_outcomes_meeting_reference": predictive_outcomes,
        "conclusion": conclusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--equivalence", action="append", default=[], type=Path)
    parser.add_argument("--overhead-off", action="append", default=[], type=Path)
    parser.add_argument("--overhead-summary", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = summarize(
        args.analysis.resolve(),
        args.candidate.resolve(),
        [path.resolve() for path in args.equivalence],
        [path.resolve() for path in args.overhead_off],
        [path.resolve() for path in args.overhead_summary],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = {
        "schema_version": summary["schema_version"],
        "validation": summary["validation"],
        "overhead": summary["overhead"],
        "formal_readiness": summary["formal_readiness"],
    }
    validation_text = json.dumps(
        validation, ensure_ascii=False, indent=2
    ) + "\n"
    for name in ("M5A_validation.json", "M5A_pressure_shadow_validation.json"):
        (args.output_dir / name).write_text(validation_text, encoding="utf-8")
    (args.output_dir / "M5A_pressure_shadow_equivalence.json").write_text(
        json.dumps(summary["equivalence"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "M5A_pressure_shadow_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded_analysis = load_json(args.analysis)
    offline_analysis = (
        loaded_analysis["offline_analysis"]
        if loaded_analysis.get("analysis") == "M5A_pressure_shadow_complete"
        else loaded_analysis
    )
    (args.output_dir / "M5A_pressure_shadow_offline_analysis.json").write_text(
        json.dumps(offline_analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    complete = {
        "schema_version": summary["schema_version"],
        "analysis": "M5A_pressure_shadow_complete",
        "offline_analysis": offline_analysis,
        "candidate": summary["candidate"],
        "validation": summary["validation"],
        "equivalence": summary["equivalence"],
        "overhead": summary["overhead"],
        "outcomes_100ms": summary["outcomes_100ms"],
        "formal_readiness": summary["formal_readiness"],
        "predictive_outcomes_meeting_reference":
            summary["predictive_outcomes_meeting_reference"],
        "conclusion": summary["conclusion"],
    }
    (args.output_dir / "M5A_pressure_shadow_full.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "M5A_pressure_shadow_report.md").write_text(
        render_report(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary["conclusion"], ensure_ascii=False))


if __name__ == "__main__":
    main()
