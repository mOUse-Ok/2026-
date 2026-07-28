#!/usr/bin/env python3
"""Validate M5A Pressure Shadow off/summary observation-only equivalence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


PRESSURE_EVENTS = {"PRESSURE_SHADOW_SAMPLE", "PRESSURE_SHADOW_SUMMARY"}
TASK_METRICS = (
    "expert_task_created",
    "expert_task_admitted",
    "expert_task_rejected",
    "expert_task_enqueued",
    "expert_task_dequeued",
    "expert_task_issued",
    "expert_task_cancelled",
    "expert_task_invalid_transitions",
    "expert_task_duplicate_create_ids",
    "expert_issue_task_count_mismatches",
    "expert_issue_ids_without_syscalls",
    "expert_syscall_issue_ids_without_tasks",
    "os_hint_events",
    "os_hint_errors",
    "os_hint_advised_mb",
)
CONTROL_ENV = (
    "LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER",
    "LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK",
    "LLM_MEM_TRACE_OPT_EXPERT_SLACK",
    "LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE",
    "LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT",
)
CONFIG_ENV = (
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH",
    "LLM_MEM_TRACE_EXPERT_TASK_MODE",
    "TRACE_PROFILE",
)
TASK_FIELDS = (
    "lifecycle_event",
    "state",
    "task_id",
    "step",
    "layer",
    "expert",
    "phase",
    "stage",
    "tensor",
    "nbytes",
    "score",
    "sequence",
    "reason",
    "hint_status",
)
TASK_SCORE_INDEX = TASK_FIELDS.index("score")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_tuple(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(name) for name in TASK_FIELDS)


def task_without_score(task: tuple[Any, ...]) -> tuple[Any, ...]:
    return task[:TASK_SCORE_INDEX] + task[TASK_SCORE_INDEX + 1:]


def create_task_difference_diagnostics(
    off_records: list[dict[str, Any]],
    summary_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Explain strict CREATE differences without weakening equivalence."""
    pairs = []
    for records in (off_records, summary_records):
        pairs.append(
            [
                task_tuple(record)
                for record in records
                if record.get("event") == "EXPERT_TASK"
                and record.get("lifecycle_event") == "CREATE"
            ]
        )
    off_create, summary_create = pairs
    identity_mismatches = 0
    score_mismatches: list[dict[str, Any]] = []
    for index, (off_task, summary_task) in enumerate(
        zip(off_create, summary_create), 1
    ):
        if task_without_score(off_task) != task_without_score(summary_task):
            identity_mismatches += 1
        off_score = off_task[TASK_SCORE_INDEX]
        summary_score = summary_task[TASK_SCORE_INDEX]
        if off_score == summary_score:
            continue
        absolute_delta = (
            abs(float(off_score) - float(summary_score))
            if isinstance(off_score, (int, float))
            and isinstance(summary_score, (int, float))
            else None
        )
        score_mismatches.append(
            {
                "create_index": index,
                "task_id": off_task[TASK_FIELDS.index("task_id")],
                "step": off_task[TASK_FIELDS.index("step")],
                "layer": off_task[TASK_FIELDS.index("layer")],
                "expert": off_task[TASK_FIELDS.index("expert")],
                "off_score": off_score,
                "summary_score": summary_score,
                "absolute_delta": absolute_delta,
            }
        )
    finite_deltas = [
        row["absolute_delta"]
        for row in score_mismatches
        if row["absolute_delta"] is not None
    ]
    return {
        "off_create_count": len(off_create),
        "summary_create_count": len(summary_create),
        "identity_without_score_multiset_equal": (
            Counter(map(task_without_score, off_create))
            == Counter(map(task_without_score, summary_create))
        ),
        "identity_without_score_order_equal": (
            list(map(task_without_score, off_create))
            == list(map(task_without_score, summary_create))
        ),
        "identity_without_score_position_mismatches": identity_mismatches,
        "score_position_mismatch_count": len(score_mismatches),
        "max_absolute_score_delta": max(finite_deltas, default=None),
        "first_score_mismatches": score_mismatches[:12],
        "note": (
            "diagnostic only: route score remains part of strict Task business "
            "identity and these differences are not normalized away"
        ),
    }


def hint_tuple(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        record.get(name)
        for name in (
            "action",
            "trigger",
            "phase",
            "step",
            "tensor",
            "layer",
            "expert",
            "size",
            "advised_bytes",
            "file_offset",
            "result",
            "errno",
        )
    )


def grouped_normalized_order(
    records: list[dict[str, Any]],
    identity: Any,
) -> list[tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...]]]:
    """Normalize concurrency within a phase/step/layer scheduling group.

    Multi-worker return timestamps are not a total order. The observable order
    contract is the first-appearance order of scheduling groups plus the exact
    ordered-input multiset within each group.
    """
    groups: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    for record in records:
        group = (
            record.get("phase"),
            record.get("step"),
            record.get("layer"),
        )
        groups.setdefault(group, []).append(identity(record))
    return [
        (group, tuple(sorted(values, key=repr)))
        for group, values in groups.items()
    ]


def normalized_task_order(records: list[dict[str, Any]]) -> dict[str, Any]:
    task_records = [
        record for record in records if record.get("event") == "EXPERT_TASK"
    ]
    by_event = {
        event: [
            record for record in task_records
            if record.get("lifecycle_event") == event
        ]
        for event in ("CREATE", "ADMIT", "ENQUEUE", "DEQUEUE", "ISSUE")
    }
    return {
        "CREATE": [task_tuple(record) for record in by_event["CREATE"]],
        "ADMIT": [task_tuple(record) for record in by_event["ADMIT"]],
        "ENQUEUE": [task_tuple(record) for record in by_event["ENQUEUE"]],
        "DEQUEUE": grouped_normalized_order(by_event["DEQUEUE"], task_tuple),
        "ISSUE": grouped_normalized_order(by_event["ISSUE"], task_tuple),
    }


def normalized_hint_order(records: list[dict[str, Any]]) -> list[Any]:
    return grouped_normalized_order(
        [record for record in records if record.get("event") == "OS_HINT"],
        hint_tuple,
    )


def existing_event_counts(records: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        str(record.get("event"))
        for record in records
        if record.get("event") not in PRESSURE_EVENTS
    )


def existing_schema(records: list[dict[str, Any]]) -> dict[str, list[list[str]]]:
    schemas: dict[str, set[tuple[str, ...]]] = {}
    for record in records:
        event = record.get("event")
        if event in PRESSURE_EVENTS:
            continue
        schemas.setdefault(str(event), set()).add(tuple(sorted(record)))
    return {
        event: [list(keys) for keys in sorted(keysets)]
        for event, keysets in sorted(schemas.items())
    }


def trace_integrity(summary: dict[str, Any]) -> dict[str, Any]:
    sinks = summary.get("sinks", {})
    failures = {}
    for name, sink in sinks.items():
        if not sink.get("enabled"):
            continue
        enqueued = sink.get("enqueued")
        written = sink.get("written")
        dropped = sink.get("dropped")
        if enqueued != written or dropped != 0:
            failures[name] = {
                "enqueued": enqueued,
                "written": written,
                "dropped": dropped,
            }
    return {"passed": not failures, "failures": failures}


def selected_environment(manifest: dict[str, Any]) -> dict[str, Any]:
    environment = manifest.get("environment", {})
    return {name: environment.get(name) for name in (*CONFIG_ENV, *CONTROL_ENV)}


def validate(off_dir: Path, summary_dir: Path) -> dict[str, Any]:
    off_records = read_jsonl(off_dir / "memory_trace.jsonl")
    summary_records = read_jsonl(summary_dir / "memory_trace.jsonl")
    off_metrics = read_json(off_dir / "analysis" / "metrics.json")
    summary_metrics = read_json(summary_dir / "analysis" / "metrics.json")
    off_manifest = read_json(off_dir / "run_manifest.json")
    summary_manifest = read_json(summary_dir / "run_manifest.json")

    off_tasks = [
        task_tuple(record) for record in off_records
        if record.get("event") == "EXPERT_TASK"
    ]
    summary_tasks = [
        task_tuple(record) for record in summary_records
        if record.get("event") == "EXPERT_TASK"
    ]
    off_hints = [
        hint_tuple(record) for record in off_records
        if record.get("event") == "OS_HINT"
    ]
    summary_hints = [
        hint_tuple(record) for record in summary_records
        if record.get("event") == "OS_HINT"
    ]
    off_task_order = normalized_task_order(off_records)
    summary_task_order = normalized_task_order(summary_records)
    off_hint_order = normalized_hint_order(off_records)
    summary_hint_order = normalized_hint_order(summary_records)
    metric_comparison = {
        name: {
            "off": off_metrics.get(name),
            "summary": summary_metrics.get(name),
            "equal": off_metrics.get(name) == summary_metrics.get(name),
        }
        for name in TASK_METRICS
    }
    off_environment = selected_environment(off_manifest)
    summary_environment = selected_environment(summary_manifest)
    config_comparison = {
        name: {
            "off": off_environment[name],
            "summary": summary_environment[name],
            "equal": off_environment[name] == summary_environment[name],
        }
        for name in (*CONFIG_ENV, *CONTROL_ENV)
    }
    active_control_off = all(
        str(summary_environment.get(name, "")).lower() in {"off", "0"}
        for name in CONTROL_ENV
    )
    off_pressure = [
        record for record in off_records if record.get("event") in PRESSURE_EVENTS
    ]
    summary_pressure_samples = [
        record for record in summary_records
        if record.get("event") == "PRESSURE_SHADOW_SAMPLE"
    ]
    summary_pressure_summaries = [
        record for record in summary_records
        if record.get("event") == "PRESSURE_SHADOW_SUMMARY"
    ]
    off_integrity = trace_integrity(read_json(off_dir / "summary.json"))
    summary_integrity = trace_integrity(read_json(summary_dir / "summary.json"))
    checks = {
        "output_hash_equal": (
            (off_dir / "output.sha256").read_text(encoding="utf-8").strip()
            == (summary_dir / "output.sha256").read_text(encoding="utf-8").strip()
        ),
        "task_multiset_equal": Counter(off_tasks) == Counter(summary_tasks),
        "task_order_equal": off_task_order == summary_task_order,
        "hint_multiset_equal": Counter(off_hints) == Counter(summary_hints),
        "hint_order_equal": off_hint_order == summary_hint_order,
        "task_metrics_equal": all(item["equal"] for item in metric_comparison.values()),
        "existing_event_counts_equal": (
            existing_event_counts(off_records) == existing_event_counts(summary_records)
        ),
        "existing_schema_equal": (
            existing_schema(off_records) == existing_schema(summary_records)
        ),
        "configuration_equal": all(item["equal"] for item in config_comparison.values()),
        "active_control_off": active_control_off,
        "off_has_no_pressure_events": not off_pressure,
        "summary_has_no_detail_samples": not summary_pressure_samples,
        "summary_has_exactly_one_runtime_summary": len(summary_pressure_summaries) == 1,
        "off_trace_integrity": off_integrity["passed"],
        "summary_trace_integrity": summary_integrity["passed"],
        "reject_cancel_zero": all(
            int(summary_metrics.get(name, -1)) == 0
            for name in (
                "expert_task_rejected",
                "expert_task_cancelled",
                "expert_task_invalid_transitions",
            )
        ),
    }
    return {
        "schema_version": 1,
        "analysis": "M5A_pressure_shadow_equivalence",
        "passed": all(checks.values()),
        "checks": checks,
        "off_dir": str(off_dir),
        "summary_dir": str(summary_dir),
        "off_memory_trace_sha256": file_hash(off_dir / "memory_trace.jsonl"),
        "summary_memory_trace_sha256": file_hash(summary_dir / "memory_trace.jsonl"),
        "task_counts": {"off": len(off_tasks), "summary": len(summary_tasks)},
        "hint_counts": {"off": len(off_hints), "summary": len(summary_hints)},
        "order_normalization": (
            "CREATE/ADMIT/ENQUEUE preserve exact sequence; DEQUEUE/ISSUE/Hint "
            "preserve phase-step-layer group order and exact within-group multiset "
            "because multi-worker return timestamps are a partial order"
        ),
        "raw_task_event_order_equal": off_tasks == summary_tasks,
        "raw_hint_return_order_equal": off_hints == summary_hints,
        "create_task_difference_diagnostics":
            create_task_difference_diagnostics(off_records, summary_records),
        "metric_comparison": metric_comparison,
        "config_comparison": config_comparison,
        "off_trace_integrity_detail": off_integrity,
        "summary_trace_integrity_detail": summary_integrity,
        "off_existing_event_counts": dict(existing_event_counts(off_records)),
        "summary_existing_event_counts": dict(existing_event_counts(summary_records)),
        "off_existing_schema": existing_schema(off_records),
        "summary_existing_schema": existing_schema(summary_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.off.resolve(), args.summary.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        failed = [name for name, passed in result["checks"].items() if not passed]
        raise SystemExit(f"Pressure Shadow equivalence failed: {','.join(failed)}")


if __name__ == "__main__":
    main()
