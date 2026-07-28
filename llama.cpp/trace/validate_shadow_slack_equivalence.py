#!/usr/bin/env python3
"""Validate the M4A Shadow off/on single-variable runtime equivalence contract."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


TASK_FIELDS = (
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def hint_multiset(records: list[dict[str, Any]]) -> Counter[tuple[Any, ...]]:
    return Counter(
        (
            record.get("action"),
            record.get("trigger"),
            record.get("phase"),
            record.get("step"),
            record.get("tensor"),
            record.get("layer"),
            record.get("expert"),
            record.get("size"),
            record.get("advised_bytes"),
            record.get("result"),
            record.get("errno"),
        )
        for record in records
        if record.get("event") == "OS_HINT"
    )


def trace_integrity(summary: dict[str, Any]) -> bool:
    return all(
        not sink.get("enabled")
        or (
            int(sink.get("enqueued", -1)) == int(sink.get("written", -2))
            and int(sink.get("dropped", -1)) == 0
        )
        for sink in summary.get("sinks", {}).values()
    )


def validate_detail_integrity(run_dir: Path) -> dict[str, Any]:
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
    terminal = {"REJECTED", "ISSUED", "CANCELLED"}
    states: dict[int, str] = {}
    previous_ts: dict[int, int] = {}
    create_counts: Counter[int] = Counter()
    issue_task_counts: Counter[int] = Counter()
    issue_expected_counts: dict[int, set[int]] = {}
    syscall_issue_ids: set[int] = set()
    event_counts: Counter[str] = Counter()
    invalid_task_id_records = 0
    invalid_issue_id_records = 0
    invalid_transitions = 0
    state_mismatches = 0
    timestamp_regressions = 0
    os_hint_errors = 0
    shadow_records = 0
    shadow_semantic_errors = 0
    shadow_target_alignment_errors = 0

    with (run_dir / "memory_trace.jsonl").open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"memory_trace.jsonl:{line_number}: {error}") from error
            event = record.get("event")
            if event == "OS_HINT":
                if int(record.get("result", 0)) != 0:
                    os_hint_errors += 1
                issue_id = record.get("issue_id")
                if isinstance(issue_id, int) and issue_id > 0:
                    syscall_issue_ids.add(issue_id)
                continue
            if event == "EXPERT_SHADOW_SLACK":
                shadow_records += 1
                shadow_semantic_errors += int(
                    record.get("schema_version") != 2
                    or record.get("semantics") != "logical_first_use"
                    or record.get("physical_load_observed") is not False
                    or record.get("issue_target")
                    != "issue_ts < logical_first_use_ts"
                    or record.get("return_target")
                    != "final_enabled_hint_return_ts < logical_first_use_ts"
                )
                prediction_ts = record.get("prediction_ts_ns")
                first_use_ts = record.get("first_use_ts_ns")
                issue_ts = record.get("issue_ts_ns")
                returned_ts = record.get("returned_ts_ns")
                if all(isinstance(value, int) for value in (
                    prediction_ts, first_use_ts, issue_ts, returned_ts
                )):
                    shadow_target_alignment_errors += int(
                        record.get("actual_issue_slack_ns") != first_use_ts - issue_ts
                        or record.get("actual_return_slack_ns")
                        != first_use_ts - returned_ts
                        or record.get("issue_on_time") is not (issue_ts < first_use_ts)
                        or record.get("return_on_time") is not (returned_ts < first_use_ts)
                    )
                else:
                    shadow_target_alignment_errors += 1
                for prediction in record.get("predictions", []):
                    horizon = prediction.get("predicted_first_use_horizon_ns")
                    queue = prediction.get("predicted_queue_wait_ns")
                    pre_issue = prediction.get("predicted_pre_issue_overhead_ns")
                    syscall = prediction.get("predicted_hint_syscall_service_ns")
                    if not all(isinstance(value, int) for value in (
                        horizon, queue, pre_issue, syscall
                    )):
                        shadow_target_alignment_errors += 1
                        continue
                    expected_issue = horizon - queue - pre_issue
                    shadow_target_alignment_errors += int(
                        prediction.get("predicted_issue_slack_ns") != expected_issue
                        or prediction.get("predicted_return_slack_ns")
                        != expected_issue - syscall
                    )
                continue
            if event != "EXPERT_TASK":
                continue

            task_id = record.get("task_id")
            if not isinstance(task_id, int) or task_id <= 0:
                invalid_task_id_records += 1
                continue
            lifecycle_event = str(record.get("lifecycle_event", "UNKNOWN"))
            event_counts[lifecycle_event] += 1
            if lifecycle_event == "CREATE":
                create_counts[task_id] += 1
            old_state = states.get(task_id)
            next_state = transitions.get(old_state, {}).get(lifecycle_event)
            if next_state is None:
                invalid_transitions += 1
            else:
                states[task_id] = next_state
                state_mismatches += int(record.get("state") != next_state)
            ts_ns = int(record.get("ts_ns", 0))
            if task_id in previous_ts and ts_ns < previous_ts[task_id]:
                timestamp_regressions += 1
            previous_ts[task_id] = max(previous_ts.get(task_id, 0), ts_ns)
            if lifecycle_event == "ISSUE":
                issue_id = record.get("issue_id")
                if not isinstance(issue_id, int) or issue_id <= 0:
                    invalid_issue_id_records += 1
                else:
                    issue_task_counts[issue_id] += 1
                    issue_expected_counts.setdefault(issue_id, set()).add(
                        int(record.get("issue_task_count", 0))
                    )

    task_issue_ids = set(issue_task_counts)
    detail = {
        "task_event_count": sum(event_counts.values()),
        "task_event_counts": dict(sorted(event_counts.items())),
        "unique_task_ids": len(states),
        "duplicate_create_ids": sum(max(0, count - 1) for count in create_counts.values()),
        "invalid_task_id_records": invalid_task_id_records,
        "invalid_transitions": invalid_transitions,
        "state_mismatches": state_mismatches,
        "incomplete_tasks": sum(state not in terminal for state in states.values()),
        "timestamp_regressions": timestamp_regressions,
        "invalid_issue_id_records": invalid_issue_id_records,
        "unique_issue_ids": len(task_issue_ids),
        "issue_task_count_mismatches": sum(
            any(expected != issue_task_counts[issue_id] for expected in expected_values)
            for issue_id, expected_values in issue_expected_counts.items()
        ),
        "linked_syscall_issue_ids": len(syscall_issue_ids),
        "issue_ids_without_syscalls": len(task_issue_ids - syscall_issue_ids),
        "syscall_issue_ids_without_tasks": len(syscall_issue_ids - task_issue_ids),
        "os_hint_errors": os_hint_errors,
        "shadow_records": shadow_records,
        "shadow_semantic_errors": shadow_semantic_errors,
        "shadow_target_alignment_errors": shadow_target_alignment_errors,
    }
    detail["passed"] = all(
        detail[field] == 0
        for field in (
            "duplicate_create_ids",
            "invalid_task_id_records",
            "invalid_transitions",
            "state_mismatches",
            "incomplete_tasks",
            "timestamp_regressions",
            "invalid_issue_id_records",
            "issue_task_count_mismatches",
            "issue_ids_without_syscalls",
            "syscall_issue_ids_without_tasks",
            "os_hint_errors",
            "shadow_semantic_errors",
            "shadow_target_alignment_errors",
        )
    )
    return detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", required=True, type=Path)
    parser.add_argument("--shadow", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--detail", type=Path)
    args = parser.parse_args()

    off_metrics = read_json(args.off / "analysis" / "metrics.json")
    shadow_metrics = read_json(args.shadow / "analysis" / "metrics.json")
    off_memory = load_jsonl(args.off / "memory_trace.jsonl")
    shadow_memory = load_jsonl(args.shadow / "memory_trace.jsonl")
    field_comparison = {
        field: {
            "off": off_metrics.get(field),
            "shadow": shadow_metrics.get(field),
            "equal": off_metrics.get(field) == shadow_metrics.get(field),
        }
        for field in TASK_FIELDS
    }
    off_hash = (args.off / "output.sha256").read_text(encoding="utf-8").strip()
    shadow_hash = (args.shadow / "output.sha256").read_text(encoding="utf-8").strip()
    hints_equal = hint_multiset(off_memory) == hint_multiset(shadow_memory)
    off_integrity = trace_integrity(read_json(args.off / "summary.json"))
    shadow_integrity = trace_integrity(read_json(args.shadow / "summary.json"))
    runtime_summary = shadow_metrics.get("expert_shadow_slack", {}).get("runtime_summary") or {}
    shadow_accounting_ok = (
        int(runtime_summary.get("predicted_tasks", 0))
        == int(runtime_summary.get("finalized_tasks", 0))
        + int(runtime_summary.get("expired_tasks", 0))
        + int(runtime_summary.get("pending_tasks", 0))
    )
    detail_integrity = validate_detail_integrity(args.detail) if args.detail else None
    passed = (
        off_hash == shadow_hash
        and hints_equal
        and off_integrity
        and shadow_integrity
        and shadow_accounting_ok
        and all(value["equal"] for value in field_comparison.values())
        and (detail_integrity is None or detail_integrity["passed"])
    )
    result = {
        "schema_version": 1,
        "passed": passed,
        "output_hash_equal": off_hash == shadow_hash,
        "off_output_sha256": off_hash,
        "shadow_output_sha256": shadow_hash,
        "hint_multiset_equal": hints_equal,
        "off_trace_integrity": off_integrity,
        "shadow_trace_integrity": shadow_integrity,
        "shadow_accounting_ok": shadow_accounting_ok,
        "detail_integrity": detail_integrity,
        "field_comparison": field_comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("Shadow off/on equivalence validation failed")


if __name__ == "__main__":
    main()
