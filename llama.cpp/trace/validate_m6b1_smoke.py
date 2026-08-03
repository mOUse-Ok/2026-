#!/usr/bin/env python3
"""Validate the fixed two-case M6B1 engineering smoke contract."""

from __future__ import annotations

import argparse
from collections import Counter
import heapq
import json
from pathlib import Path
from typing import Any

import audit_router_score_determinism as router_audit


COMMON_ENV = {
    "LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER": "off",
    "LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK": "0",
    "LLM_MEM_TRACE_OPT_EXPERT_SLACK": "0",
    "LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE": "0",
    "LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT": "0",
    "LLM_MEM_TRACE_OPT_EXPERT_SLACK_MODE": "off",
    "LLM_MEM_TRACE_PRESSURE_SHADOW_MODE": "off",
    "LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE": "1",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC": "1",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE": "131072",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY": "1",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP": "0",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH": "1",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_WAIT_US": "0",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_COALESCE": "0",
    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_FALLBACK": "1",
    "LLM_MEM_TRACE_OPT_EXPERT_COALESCE": "0",
    "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK": "0",
    "LLM_MEM_TRACE_OPT_EXPERT_POLICY": "route",
    "LLM_MEM_TRACE_OPT_EXPERT_TTL_STEPS": "0",
    "LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS": "0",
    "LLM_MEM_TRACE_ROUTER_SCORE_DIAGNOSTIC": "1",
    "LLM_MEM_TRACE_ROUTER_TENSOR_SYNC_PROTOCOL": "m6b1.2-v1",
}

TASK_COUNT_FIELDS = (
    "expert_task_created",
    "expert_task_admitted",
    "expert_task_rejected",
    "expert_task_enqueued",
    "expert_task_dequeued",
    "expert_task_issued",
    "expert_task_cancelled",
    "expert_task_terminal",
    "expert_task_in_flight",
    "expert_task_invalid_transitions",
    "expert_task_duplicate_create_ids",
    "expert_task_trace_invalid_transitions",
    "expert_task_trace_incomplete",
    "expert_task_trace_timestamp_regressions",
    "expert_issue_task_count_mismatches",
    "expert_issue_ids_without_syscalls",
    "expert_syscall_issue_ids_without_tasks",
    "os_hint_events",
    "os_hint_errors",
)

STATIC_TASK_FIELDS = (
    "state", "task_id", "step", "layer", "expert", "phase", "stage", "tensor",
    "nbytes", "score", "score_f64_bits", "sequence", "reason", "hint_status",
)
TASK_SCORE_INDICES = {
    1 + STATIC_TASK_FIELDS.index("score"),
    1 + STATIC_TASK_FIELDS.index("score_f64_bits"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def trace_complete(summary: dict[str, Any]) -> bool:
    return all(
        not values.get("enabled")
        or (
            int(values.get("enqueued", -1)) == int(values.get("written", -2))
            and int(values.get("dropped", -1)) == 0
        )
        for values in summary.get("sinks", {}).values()
        if isinstance(values, dict)
    )


def lifecycle_multiset(records: list[dict[str, Any]]) -> Counter[tuple[Any, ...]]:
    return Counter(
        (record.get("lifecycle_event"), *(record.get(field) for field in STATIC_TASK_FIELDS))
        for record in records
        if record.get("event") == "EXPERT_TASK"
        and record.get("lifecycle_event") in {"CREATE", "ADMIT", "ENQUEUE"}
    )


def without_task_score(task: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(
        value for index, value in enumerate(task) if index not in TASK_SCORE_INDICES
    )


def hint_multiset(records: list[dict[str, Any]]) -> Counter[tuple[Any, ...]]:
    fields = (
        "action", "trigger", "phase", "step", "tensor", "layer", "expert",
        "size", "advised_bytes", "result", "errno",
    )


def router_multiset(records: list[dict[str, Any]]) -> Counter[tuple[Any, ...]]:
    return Counter(
        (
            record.get("phase"), record.get("step"), record.get("layer"),
            record.get("token"), record.get("top_k"),
            tuple(record.get("experts", [])), tuple(record.get("score_raw_bits", [])),
            tuple(record.get("score_f32_bits", [])), record.get("observation_sync"),
            record.get("observation_sync_protocol"),
        )
        for record in records if record.get("event") == "EXPERT_ROUTE"
    )


def router_diagnostics_complete(records: list[dict[str, Any]]) -> bool:
    routes = [record for record in records if record.get("event") == "EXPERT_ROUTE"]
    return bool(routes) and all(
        isinstance(record.get("score_raw_bits"), list)
        and isinstance(record.get("score_f32_bits"), list)
        and record.get("observation_sync") == "producer_barrier_hook_release_barrier"
        and record.get("observation_sync_protocol") == "m6b1.2-v1"
        for record in routes
    )
    return Counter(
        tuple(record.get(field) for field in fields)
        for record in records
        if record.get("event") == "OS_HINT"
    )


def lifecycle_integrity(records: list[dict[str, Any]]) -> dict[str, int | bool]:
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
    timestamps: dict[int, int] = {}
    creates: Counter[int] = Counter()
    invalid = state_mismatches = regressions = invalid_ids = 0
    for record in records:
        if record.get("event") != "EXPERT_TASK":
            continue
        task_id = record.get("task_id")
        if not isinstance(task_id, int) or task_id <= 0:
            invalid_ids += 1
            continue
        event = str(record.get("lifecycle_event", ""))
        creates[task_id] += int(event == "CREATE")
        next_state = transitions.get(states.get(task_id), {}).get(event)
        if next_state is None:
            invalid += 1
        else:
            states[task_id] = next_state
            state_mismatches += int(record.get("state") != next_state)
        timestamp = int(record.get("ts_ns", 0))
        regressions += int(task_id in timestamps and timestamp < timestamps[task_id])
        timestamps[task_id] = max(timestamp, timestamps.get(task_id, 0))
    result: dict[str, int | bool] = {
        "unique_tasks": len(states),
        "duplicate_create_ids": sum(max(0, count - 1) for count in creates.values()),
        "invalid_task_ids": invalid_ids,
        "invalid_transitions": invalid,
        "state_mismatches": state_mismatches,
        "timestamp_regressions": regressions,
        "incomplete_tasks": sum(state not in terminal for state in states.values()),
    }
    result["passed"] = all(
        result[field] == 0
        for field in (
            "duplicate_create_ids", "invalid_task_ids", "invalid_transitions",
            "state_mismatches", "timestamp_regressions", "incomplete_tasks",
        )
    )
    return result


def classification(record: dict[str, Any], decision_ns: int, threshold_ns: int, guard_ns: int) -> str:
    enqueued_ns = int(record.get("enqueued_ts_ns", 0))
    if enqueued_ns == 0 or decision_ns < enqueued_ns:
        return "normal"
    deadline_ns = int(record.get("deadline_ts_ns", 0))
    urgent_limit = min((1 << 64) - 1, decision_ns + guard_ns)
    if deadline_ns != 0 and deadline_ns <= urgent_limit:
        return "urgent"
    if decision_ns - enqueued_ns >= threshold_ns:
        return "protected"
    return "normal"


def selection_key(record: dict[str, Any], task_class: str) -> tuple[Any, ...]:
    deadline = int(record.get("deadline_ts_ns", 0))
    legacy = (deadline == 0, deadline, -float(record.get("score", 0.0)), int(record.get("sequence", 0)))
    class_rank = {"urgent": 0, "protected": 1, "normal": 2}[task_class]
    if task_class == "protected":
        return (class_rank, int(record.get("enqueued_ts_ns", 0)), *legacy)
    return (class_rank, *legacy)


def replay_candidate_selections(records: list[dict[str, Any]]) -> dict[str, int | bool]:
    task_records: dict[int, dict[str, Any]] = {}
    dequeued: dict[int, int] = {}
    for record in records:
        if record.get("event") != "EXPERT_TASK" or not isinstance(record.get("task_id"), int):
            continue
        task_id = int(record["task_id"])
        task_records[task_id] = record
        if record.get("lifecycle_event") == "DEQUEUE":
            dequeued[task_id] = int(record.get("dequeued_ts_ns", record.get("ts_ns", 0)))

    selections = [
        record for record in records if record.get("event") == "EXPERT_MAX_WAIT_SELECTION"
    ]
    selections.sort(key=lambda record: (
        int(record.get("decision_ts_ns", 0)),
        dequeued.get(int(record.get("task_id", 0)), 0),
        int(record.get("task_id", 0)),
    ))
    remaining = set(task_records)
    status: dict[int, str] = {}
    class_counts: Counter[str] = Counter()
    class_heaps: dict[str, list[tuple[Any, ...]]] = {
        "urgent": [], "protected": [], "normal": [],
    }
    arrivals = sorted(
        (int(record.get("enqueued_ts_ns", 0)), task_id)
        for task_id, record in task_records.items()
    )
    threshold = int(selections[0].get("threshold_ns", 0)) if selections else 0
    guard = int(selections[0].get("urgent_guard_ns", 0)) if selections else 0
    protection_transitions = sorted(
        (min((1 << 64) - 1, enqueued + threshold), task_id)
        for enqueued, task_id in arrivals
        if enqueued != 0
    )
    urgent_transitions = sorted(
        (max(0, int(record.get("deadline_ts_ns", 0)) - guard), task_id)
        for task_id, record in task_records.items()
        if int(record.get("enqueued_ts_ns", 0)) != 0
        and int(record.get("deadline_ts_ns", 0)) != 0
    )
    arrival_index = protection_index = urgent_index = 0

    def set_status(task_id: int, task_class: str) -> None:
        old = status.get(task_id)
        if old == task_class or old == "removed":
            return
        if old is not None:
            class_counts[old] -= 1
        status[task_id] = task_class
        class_counts[task_class] += 1
        heapq.heappush(
            class_heaps[task_class],
            (*selection_key(task_records[task_id], task_class), task_id),
        )

    def heap_top(task_class: str) -> int | None:
        heap = class_heaps[task_class]
        while heap and status.get(heap[0][-1]) != task_class:
            heapq.heappop(heap)
        return heap[0][-1] if heap else None

    winner_mismatches = candidate_count_mismatches = normal_flag_mismatches = 0
    timestamp_mismatches = missing_tasks = urgent_winner_mismatches = config_mismatches = 0
    for selected in selections:
        task_id = int(selected.get("task_id", 0))
        decision = int(selected.get("decision_ts_ns", 0))
        config_mismatches += int(
            int(selected.get("threshold_ns", -1)) != threshold
            or int(selected.get("urgent_guard_ns", -1)) != guard
        )

        while arrival_index < len(arrivals) and arrivals[arrival_index][0] <= decision:
            _, arrived_id = arrivals[arrival_index]
            set_status(
                arrived_id,
                classification(task_records[arrived_id], decision, threshold, guard),
            )
            arrival_index += 1
        while urgent_index < len(urgent_transitions) and urgent_transitions[urgent_index][0] <= decision:
            _, transitioned_id = urgent_transitions[urgent_index]
            if transitioned_id in status and status.get(transitioned_id) != "removed":
                set_status(transitioned_id, "urgent")
            urgent_index += 1
        while (protection_index < len(protection_transitions)
                and protection_transitions[protection_index][0] <= decision):
            _, transitioned_id = protection_transitions[protection_index]
            if status.get(transitioned_id) == "normal":
                set_status(transitioned_id, "protected")
            protection_index += 1

        if task_id not in task_records or task_id not in remaining or task_id not in status:
            missing_tasks += 1
            continue
        protected_count = class_counts["protected"]
        normal_present = class_counts["normal"] > 0
        candidate_count_mismatches += int(
            int(selected.get("protected_candidate_count", -1)) != protected_count
        )
        normal_flag_mismatches += int(
            selected.get("normal_competitor_present") is not normal_present
        )
        expected_class = next(
            (task_class for task_class in ("urgent", "protected", "normal")
             if class_counts[task_class] > 0),
            "normal",
        )
        expected_id = heap_top(expected_class)
        mismatch = int(expected_id != task_id)
        winner_mismatches += mismatch
        urgent_winner_mismatches += mismatch * int(expected_class == "urgent")
        enqueue = int(task_records[task_id].get("enqueued_ts_ns", 0))
        dequeue = dequeued.get(task_id, 0)
        timestamp_mismatches += int(not (enqueue <= decision <= dequeue))
        selected_class = status[task_id]
        class_counts[selected_class] -= 1
        status[task_id] = "removed"
        remaining.remove(task_id)
    result: dict[str, int | bool] = {
        "selection_events": len(selections),
        "missing_or_duplicate_selected_tasks": missing_tasks,
        "winner_mismatches": winner_mismatches,
        "urgent_winner_mismatches": urgent_winner_mismatches,
        "protected_candidate_count_mismatches": candidate_count_mismatches,
        "normal_competitor_flag_mismatches": normal_flag_mismatches,
        "timestamp_mismatches": timestamp_mismatches,
        "config_mismatches": config_mismatches,
        "unselected_tasks": len(remaining),
    }
    result["passed"] = all(
        result[field] == 0
        for field in (
            "missing_or_duplicate_selected_tasks", "winner_mismatches",
            "urgent_winner_mismatches", "protected_candidate_count_mismatches",
            "normal_competitor_flag_mismatches", "timestamp_mismatches",
            "config_mismatches", "unselected_tasks",
        )
    )
    return result


def environment_check(
    manifest: dict[str, Any], mode: str, workers: int = 2,
) -> dict[str, Any]:
    environment = manifest.get("environment", {})
    mismatches = {
        name: {"expected": expected, "actual": environment.get(name)}
        for name, expected in COMMON_ENV.items()
        if environment.get(name) != expected
    }
    if environment.get("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE") != mode:
        mismatches["LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE"] = {
            "expected": mode,
            "actual": environment.get("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE"),
        }
    if environment.get("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS") != str(workers):
        mismatches["LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS"] = {
            "expected": str(workers),
            "actual": environment.get("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS"),
        }
    threshold = environment.get("LLM_MEM_TRACE_OPT_EXPERT_MAX_WAIT_THRESHOLD_US")
    guard = environment.get("LLM_MEM_TRACE_OPT_EXPERT_URGENT_GUARD_US")
    if mode == "max_wait_protection":
        if threshold != "1":
            mismatches["LLM_MEM_TRACE_OPT_EXPERT_MAX_WAIT_THRESHOLD_US"] = {
                "expected": "1", "actual": threshold,
            }
        if guard != "0":
            mismatches["LLM_MEM_TRACE_OPT_EXPERT_URGENT_GUARD_US"] = {
                "expected": "0", "actual": guard,
            }
    else:
        if threshold is not None or guard is not None:
            mismatches["baseline_max_wait_parameters"] = {
                "expected": "absent", "actual": [threshold, guard],
            }
    return {"passed": not mismatches, "mismatches": mismatches}


def validate(
    baseline_dir: Path, candidate_dir: Path, workers: int = 2,
) -> dict[str, Any]:
    baseline_metrics = read_json(baseline_dir / "analysis" / "metrics.json")
    candidate_metrics = read_json(candidate_dir / "analysis" / "metrics.json")
    baseline_manifest = read_json(baseline_dir / "run_manifest.json")
    candidate_manifest = read_json(candidate_dir / "run_manifest.json")
    baseline_records = read_jsonl(baseline_dir / "memory_trace.jsonl")
    candidate_records = read_jsonl(candidate_dir / "memory_trace.jsonl")
    baseline_routes = read_jsonl(baseline_dir / "expert_trace.jsonl")
    candidate_routes = read_jsonl(candidate_dir / "expert_trace.jsonl")

    metric_comparison = {
        field: {
            "baseline": baseline_metrics.get(field),
            "candidate": candidate_metrics.get(field),
            "equal": baseline_metrics.get(field) == candidate_metrics.get(field),
        }
        for field in TASK_COUNT_FIELDS
    }
    candidate_counts = {
        field: int(candidate_metrics.get(field, -1))
        for field in (
            "expert_task_created", "expert_task_admitted", "expert_task_enqueued",
            "expert_task_dequeued", "expert_task_issued", "expert_task_terminal",
        )
    }
    candidate_task_conservation = len(set(candidate_counts.values())) == 1 and all(
        int(candidate_metrics.get(field, -1)) == 0
        for field in (
            "expert_task_rejected", "expert_task_cancelled", "expert_task_in_flight",
            "expert_task_invalid_transitions", "expert_async_fallback",
            "expert_async_queue_full_fallbacks", "expert_async_start_fail_fallbacks",
        )
    )
    baseline_max_events = sum(
        record.get("event") in {"EXPERT_MAX_WAIT_SELECTION", "EXPERT_MAX_WAIT_SUMMARY"}
        for record in baseline_records
    )
    candidate_summary_events = sum(
        record.get("event") == "EXPERT_MAX_WAIT_SUMMARY" for record in candidate_records
    )
    candidate_mechanism_ok = (
        candidate_summary_events == 1
        and int(candidate_metrics.get("expert_max_wait_protection_selected_count", 0)) > 0
        and int(candidate_metrics.get("expert_max_wait_missing_deadline_count", -1)) == 0
        and int(candidate_metrics.get("expert_max_wait_missing_enqueue_timestamp_count", -1)) == 0
        and int(candidate_metrics.get("expert_max_wait_enqueue_time_regression_count", -1)) == 0
        and int(candidate_metrics.get("expert_max_wait_protection_still_waiting_count", -1)) == 0
        and int(candidate_metrics.get("expert_max_wait_classification_mismatches", -1)) == 0
        and int(candidate_metrics.get("expert_max_wait_timestamp_mismatches", -1)) == 0
        and int(candidate_metrics.get("expert_max_wait_duplicate_selection_task_ids", -1)) == 0
        and int(candidate_metrics.get("expert_max_wait_summary_selection_mismatch", -1)) == 0
    )
    inactive_controller_events = sum(
        record.get("event") in {
            "EXPERT_PRESSURE",
            "EXPERT_SHADOW_SLACK",
            "EXPERT_SHADOW_SLACK_SUMMARY",
            "PRESSURE_SHADOW_SAMPLE",
            "PRESSURE_SHADOW_SUMMARY",
            "EXPERT_PREDICT_SUMMARY",
        }
        for record in baseline_records + candidate_records
    )

    baseline_hash = (baseline_dir / "output.sha256").read_text(encoding="ascii").strip()
    candidate_hash = (candidate_dir / "output.sha256").read_text(encoding="ascii").strip()
    manifest_invariants = {
        "git_commit": baseline_manifest.get("git_commit") == candidate_manifest.get("git_commit"),
        "model": baseline_manifest.get("model") == candidate_manifest.get("model"),
        "prompt_sha256": baseline_manifest.get("prompt", {}).get("sha256")
        == candidate_manifest.get("prompt", {}).get("sha256"),
        "binary_sha256": baseline_manifest.get("binary", {}).get("sha256")
        == candidate_manifest.get("binary", {}).get("sha256"),
    }
    replay = replay_candidate_selections(candidate_records)
    baseline_audit = router_audit.load_run("baseline", baseline_dir)
    candidate_audit = router_audit.load_run("candidate", candidate_dir)
    baseline_same_run_replay = router_audit.replay(baseline_audit, {})
    candidate_same_run_replay = router_audit.replay(candidate_audit, {})
    baseline_tasks = lifecycle_multiset(baseline_records)
    candidate_tasks = lifecycle_multiset(candidate_records)
    baseline_without_score: Counter[tuple[Any, ...]] = Counter()
    candidate_without_score: Counter[tuple[Any, ...]] = Counter()
    for task, count in baseline_tasks.items():
        baseline_without_score[without_task_score(task)] += count
    for task, count in candidate_tasks.items():
        candidate_without_score[without_task_score(task)] += count
    result = {
        "schema_version": 1,
        "baseline": str(baseline_dir),
        "candidate": str(candidate_dir),
        "output_hash_equal": baseline_hash == candidate_hash,
        "baseline_output_sha256": baseline_hash,
        "candidate_output_sha256": candidate_hash,
        "workers": workers,
        "baseline_environment": environment_check(
            baseline_manifest, "deadline_score", workers
        ),
        "candidate_environment": environment_check(
            candidate_manifest, "max_wait_protection", workers
        ),
        "manifest_invariants": manifest_invariants,
        "baseline_trace_complete": trace_complete(read_json(baseline_dir / "summary.json")),
        "candidate_trace_complete": trace_complete(read_json(candidate_dir / "summary.json")),
        "baseline_lifecycle": lifecycle_integrity(baseline_records),
        "candidate_lifecycle": lifecycle_integrity(candidate_records),
        "task_lifecycle_multiset_equal": baseline_tasks == candidate_tasks,
        "task_lifecycle_identity_without_score_equal": (
            baseline_without_score == candidate_without_score
        ),
        "task_lifecycle_baseline_only_records": sum(
            (baseline_tasks - candidate_tasks).values()
        ),
        "task_lifecycle_candidate_only_records": sum(
            (candidate_tasks - baseline_tasks).values()
        ),
        "task_lifecycle_score_note": (
            "route_score remains part of strict Task business identity and is not normalized"
        ),
        "router_diagnostics_complete": (
            router_diagnostics_complete(baseline_routes)
            and router_diagnostics_complete(candidate_routes)
        ),
        "router_bit_multiset_equal": (
            router_multiset(baseline_routes) == router_multiset(candidate_routes)
        ),
        "hint_multiset_equal": hint_multiset(baseline_records) == hint_multiset(candidate_records),
        "task_and_hint_metric_comparison": metric_comparison,
        "candidate_task_conservation": candidate_task_conservation,
        "baseline_max_wait_event_count": baseline_max_events,
        "candidate_max_wait_summary_events": candidate_summary_events,
        "candidate_mechanism_ok": candidate_mechanism_ok,
        "candidate_selection_replay": replay,
        "baseline_same_run_replay": baseline_same_run_replay,
        "candidate_same_run_replay": candidate_same_run_replay,
        "inactive_controller_event_count": inactive_controller_events,
        "protected_selected_count": int(
            candidate_metrics.get("expert_max_wait_protection_selected_count", 0)
        ),
        "urgent_selected_count": int(
            candidate_metrics.get("expert_max_wait_urgent_selected_count", 0)
        ),
    }
    result["passed"] = all((
        result["output_hash_equal"],
        result["baseline_environment"]["passed"],
        result["candidate_environment"]["passed"],
        all(manifest_invariants.values()),
        result["baseline_trace_complete"],
        result["candidate_trace_complete"],
        result["baseline_lifecycle"]["passed"],
        result["candidate_lifecycle"]["passed"],
        result["task_lifecycle_multiset_equal"],
        result["router_diagnostics_complete"],
        result["router_bit_multiset_equal"],
        result["hint_multiset_equal"],
        all(value["equal"] for value in metric_comparison.values()),
        candidate_task_conservation,
        baseline_max_events == 0,
        candidate_mechanism_ok,
        replay["passed"],
        baseline_same_run_replay["passed"],
        candidate_same_run_replay["passed"],
        inactive_controller_events == 0,
    ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    result = validate(args.baseline, args.candidate, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"],
        "protected_selected_count": result["protected_selected_count"],
        "urgent_winner_mismatches": result["candidate_selection_replay"]["urgent_winner_mismatches"],
    }, sort_keys=True))
    if not result["passed"]:
        raise SystemExit("M6B1 smoke validation failed; evidence was retained")


if __name__ == "__main__":
    main()
