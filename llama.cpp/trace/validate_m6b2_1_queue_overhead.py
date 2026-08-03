#!/usr/bin/env python3
"""Validate one fixed-scheduler M6B2.1 feature-off/summary engineering pair."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import audit_router_score_determinism as router_audit
import validate_m6b1_smoke as m6b1


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
    "LLM_MEM_TRACE_AUDIT_NOT_M6B2_CALIBRATION": "1",
    "LLM_MEM_TRACE_AUDIT_UNLIMITED_CGROUP_NOT_BASELINE": "1",
    "LLM_MEM_TRACE_AUDIT_PERFORMANCE_CLAIM": "0",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def hint_multiset(records: list[dict[str, Any]]) -> Counter[tuple[Any, ...]]:
    fields = (
        "action", "trigger", "phase", "step", "tensor", "layer", "expert",
        "size", "advised_bytes", "result", "errno",
    )
    return Counter(
        tuple(record.get(field) for field in fields)
        for record in records
        if record.get("event") == "OS_HINT"
    )


def environment_check(
    manifest: dict[str, Any],
    scheduler: str,
    observation: str,
    workers: int,
) -> dict[str, Any]:
    environment = manifest.get("environment", {})
    expected = dict(COMMON_ENV)
    expected.update({
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE": scheduler,
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS": str(workers),
        "LLM_MEM_TRACE_QUEUE_OVERHEAD_MODE": observation,
    })
    if scheduler == "max_wait_protection":
        expected.update({
            "LLM_MEM_TRACE_OPT_EXPERT_MAX_WAIT_THRESHOLD_US": "1",
            "LLM_MEM_TRACE_OPT_EXPERT_URGENT_GUARD_US": "0",
        })
    mismatches = {
        name: {"expected": value, "actual": environment.get(name)}
        for name, value in expected.items()
        if environment.get(name) != value
    }
    if scheduler != "max_wait_protection":
        for name in (
            "LLM_MEM_TRACE_OPT_EXPERT_MAX_WAIT_THRESHOLD_US",
            "LLM_MEM_TRACE_OPT_EXPERT_URGENT_GUARD_US",
        ):
            if name in environment:
                mismatches[name] = {
                    "expected": "absent",
                    "actual": environment.get(name),
                }
    return {"passed": not mismatches, "mismatches": mismatches}


def queue_summary_check(
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    scheduler: str,
    workers: int,
    observation: str = "summary",
) -> dict[str, Any]:
    summaries = [
        record for record in records
        if record.get("event") == "EXPERT_QUEUE_OVERHEAD_SUMMARY"
    ]
    details = [
        record for record in records
        if record.get("event") == "EXPERT_QUEUE_OVERHEAD_SELECTION"
    ]
    summary = summaries[0] if len(summaries) == 1 else {}
    global_metrics = summary.get("global", {})
    selection_count = int(summary.get("selection_count", -1))
    batch_count = int(summary.get("batch_count", -1))
    expected_priority_pops = int(metrics.get("expert_async_priority_pops", -2))
    expected_dequeues = int(metrics.get("expert_task_dequeued", -3))
    aggregate_counts = {
        name: int(global_metrics.get(name, {}).get("count", -1))
        for name in (
            "mutex_acquire_wait_ns",
            "mutex_hold_ns",
            "queue_scan_ns",
            "queue_scan_candidates",
        )
    }
    aggregate_unavailable = {
        name: int(global_metrics.get(name, {}).get("unavailable_count", -1))
        for name in (
            "mutex_acquire_wait_ns",
            "mutex_hold_ns",
            "queue_scan_ns",
            "queue_scan_candidates",
        )
    }
    candidates_total = global_metrics.get(
        "queue_scan_candidates", {}
    ).get("total")
    detail_checks = {
        "detail_count": (
            len(details) == selection_count
            if observation == "detail" else len(details) == 0
        ),
        "detail_attempt_count": int(
            summary.get("detail_event_count", -1)
        ) == (selection_count if observation == "detail" else 0),
        "detail_schema": all(
            record.get("schema_version") == "m6b2.1-queue-overhead-v1"
            and record.get("semantics") == "direct_queue_selection_measurement"
            and record.get("physical_load_observed") is False
            and record.get("priority_mode") == scheduler
            and record.get("selection_strategy") == "linear_scan"
            and isinstance(record.get("queue_scan_candidates"), int)
            and int(record["queue_scan_candidates"]) >= 1
            and record.get("queue_scan_candidates")
            == record.get("queue_depth_before")
            and record.get("mutex_acquire_wait_ns") is not None
            and record.get("mutex_hold_ns") is not None
            and record.get("queue_scan_ns") is not None
            and record.get("error_flags") == []
            for record in details
        ),
        "detail_raw_boundaries": all(
            int(record.get("lock_wait_start_ts_ns", 1))
            <= int(record.get("lock_acquired_ts_ns", 0))
            <= int(record.get("batch_decision_ts_ns", 0))
            <= int(record.get("lock_release_ts_ns", 0))
            and int(record.get("lock_acquired_ts_ns", 1))
            <= int(record.get("scan_start_ts_ns", 0))
            <= int(record.get("scan_end_ts_ns", 0))
            <= int(record.get("lock_release_ts_ns", 0))
            for record in details
        ),
        "detail_ids_unique": (
            len({record.get("decision_id") for record in details})
            == len(details)
        ),
        "detail_batch_slot": all(
            int(record.get("batch_slot", -1)) == 0 for record in details
        ),
        "detail_analyzer_clean": (
            int(metrics.get(
                "expert_queue_overhead_selection_semantic_violations", -1
            )) == 0
            and int(metrics.get(
                "expert_queue_overhead_duplicate_decision_ids", -1
            )) == 0
            and int(metrics.get(
                "expert_queue_overhead_winner_link_mismatches", -1
            )) == 0
            and int(metrics.get(
                "expert_queue_overhead_summary_detail_mismatch", -1
            )) == 0
        ) if observation == "detail" else True,
    }
    checks = {
        "exactly_one_summary": len(summaries) == 1,
        **detail_checks,
        "schema": summary.get("schema_version") == "m6b2.1-queue-overhead-v1",
        "mode": summary.get("mode") == observation,
        "semantics": (
            summary.get("semantics") == "direct_queue_selection_measurement"
            and summary.get("physical_load_observed") is False
        ),
        "scheduler": summary.get("priority_mode") == scheduler,
        "workers": int(summary.get("workers", -1)) == workers,
        "batch_size": int(summary.get("scheduler_batch", -1)) == 1,
        "selection_nonzero": selection_count > 0,
        "selection_priority_pop_balance": selection_count == expected_priority_pops,
        "selection_dequeue_balance": selection_count == expected_dequeues,
        "batch_selection_balance": batch_count == selection_count,
        "acquire_count": aggregate_counts["mutex_acquire_wait_ns"] == batch_count,
        "hold_count": aggregate_counts["mutex_hold_ns"] == batch_count,
        "scan_count": aggregate_counts["queue_scan_ns"] == selection_count,
        "candidate_sample_count": (
            aggregate_counts["queue_scan_candidates"] == selection_count
        ),
        "all_direct_metrics_available": all(
            value == 0 for value in aggregate_unavailable.values()
        ),
        "candidate_total_sane": (
            isinstance(candidates_total, int)
            and candidates_total >= selection_count
        ),
        "no_clock_regression": int(summary.get("clock_regression_count", -1)) == 0,
        "no_overflow": int(summary.get("overflow_count", -1)) == 0,
        "decision_ids_dense": int(summary.get("next_decision_id", -1)) == selection_count,
        "batch_ids_dense": int(summary.get("next_batch_id", -1)) == batch_count,
        "analyzer_available": metrics.get("expert_queue_overhead_available") is True,
        "analyzer_schema_clean": int(
            metrics.get("expert_queue_overhead_schema_violations", -1)
        ) == 0,
        "analyzer_balance": int(
            metrics.get("expert_queue_overhead_priority_pop_mismatch", -1)
        ) == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "selection_count": selection_count,
        "batch_count": batch_count,
        "candidates_total": candidates_total,
        "aggregate_counts": aggregate_counts,
        "aggregate_unavailable": aggregate_unavailable,
        "clock_read_count": summary.get("clock_read_count"),
        "condition_wait_count": summary.get("condition_wait_count"),
        "condition_reacquire_count": summary.get("condition_reacquire_count"),
        "idle_wait_exit_count": summary.get("idle_wait_exit_count"),
    }


def validate(
    off_dir: Path,
    summary_dir: Path,
    scheduler: str,
    workers: int,
    observation: str = "summary",
) -> dict[str, Any]:
    off_metrics = read_json(off_dir / "analysis" / "metrics.json")
    summary_metrics = read_json(summary_dir / "analysis" / "metrics.json")
    off_manifest = read_json(off_dir / "run_manifest.json")
    summary_manifest = read_json(summary_dir / "run_manifest.json")
    off_records = read_jsonl(off_dir / "memory_trace.jsonl")
    summary_records = read_jsonl(summary_dir / "memory_trace.jsonl")
    off_routes = read_jsonl(off_dir / "expert_trace.jsonl")
    summary_routes = read_jsonl(summary_dir / "expert_trace.jsonl")

    task_metrics = {
        field: {
            "off": off_metrics.get(field),
            "summary": summary_metrics.get(field),
            "equal": off_metrics.get(field) == summary_metrics.get(field),
        }
        for field in TASK_COUNT_FIELDS
    }
    off_tasks = m6b1.lifecycle_multiset(off_records)
    summary_tasks = m6b1.lifecycle_multiset(summary_records)
    off_audit = router_audit.load_run("off", off_dir)
    summary_audit = router_audit.load_run("summary", summary_dir)
    off_replay = router_audit.replay(off_audit, {})
    summary_replay = router_audit.replay(summary_audit, {})
    off_queue_events = sum(
        record.get("event") in {
            "EXPERT_QUEUE_OVERHEAD_SUMMARY",
            "EXPERT_QUEUE_OVERHEAD_SELECTION",
        }
        for record in off_records
    )
    manifest_invariants = {
        "git_commit": off_manifest.get("git_commit") == summary_manifest.get("git_commit"),
        "model": off_manifest.get("model") == summary_manifest.get("model"),
        "prompt_sha256": (
            off_manifest.get("prompt", {}).get("sha256")
            == summary_manifest.get("prompt", {}).get("sha256")
        ),
        "binary_sha256": (
            off_manifest.get("binary", {}).get("sha256")
            == summary_manifest.get("binary", {}).get("sha256")
        ),
    }
    off_hash = (off_dir / "output.sha256").read_text(encoding="ascii").strip()
    summary_hash = (
        summary_dir / "output.sha256"
    ).read_text(encoding="ascii").strip()

    max_wait_replays: dict[str, Any] | None = None
    if scheduler == "max_wait_protection":
        max_wait_replays = {
            "off": m6b1.replay_candidate_selections(off_records),
            "summary": m6b1.replay_candidate_selections(summary_records),
        }

    queue_check = queue_summary_check(
        summary_records, summary_metrics, scheduler, workers, observation
    )
    result = {
        "schema_version": "m6b2.1-engineering-equivalence-v1",
        "classification": "engineering_correctness_only",
        "not_m6b2_calibration": True,
        "unlimited_cgroup_not_a_baseline": True,
        "performance_claim": False,
        "scheduler": scheduler,
        "workers": workers,
        "observation": observation,
        "off": str(off_dir),
        "summary": str(summary_dir),
        "output_hash_equal": off_hash == summary_hash,
        "off_output_sha256": off_hash,
        "summary_output_sha256": summary_hash,
        "off_environment": environment_check(
            off_manifest, scheduler, "off", workers
        ),
        "summary_environment": environment_check(
            summary_manifest, scheduler, observation, workers
        ),
        "manifest_invariants": manifest_invariants,
        "off_trace_complete": m6b1.trace_complete(
            read_json(off_dir / "summary.json")
        ),
        "summary_trace_complete": m6b1.trace_complete(
            read_json(summary_dir / "summary.json")
        ),
        "off_lifecycle": m6b1.lifecycle_integrity(off_records),
        "summary_lifecycle": m6b1.lifecycle_integrity(summary_records),
        "strict_task_lifecycle_multiset_equal": off_tasks == summary_tasks,
        "strict_task_off_only_records": sum((off_tasks - summary_tasks).values()),
        "strict_task_summary_only_records": sum((summary_tasks - off_tasks).values()),
        "hint_logical_multiset_equal": (
            hint_multiset(off_records) == hint_multiset(summary_records)
        ),
        "router_diagnostics_complete": (
            m6b1.router_diagnostics_complete(off_routes)
            and m6b1.router_diagnostics_complete(summary_routes)
        ),
        "router_bit_multiset_equal": (
            m6b1.router_multiset(off_routes)
            == m6b1.router_multiset(summary_routes)
        ),
        "off_same_run_replay": off_replay,
        "summary_same_run_replay": summary_replay,
        "task_and_hint_metric_comparison": task_metrics,
        "off_queue_event_count": off_queue_events,
        "off_analyzer_marks_unavailable": (
            off_metrics.get("expert_queue_overhead_available") is False
            and off_metrics.get("expert_queue_overhead_unavailable_reason")
            == "no_expert_queue_overhead_summary"
        ),
        "summary_queue_observation": queue_check,
        "max_wait_replays": max_wait_replays,
    }
    required = (
        result["output_hash_equal"],
        result["off_environment"]["passed"],
        result["summary_environment"]["passed"],
        all(manifest_invariants.values()),
        result["off_trace_complete"],
        result["summary_trace_complete"],
        result["off_lifecycle"]["passed"],
        result["summary_lifecycle"]["passed"],
        result["strict_task_lifecycle_multiset_equal"],
        result["hint_logical_multiset_equal"],
        result["router_diagnostics_complete"],
        result["router_bit_multiset_equal"],
        off_replay["passed"],
        summary_replay["passed"],
        all(value["equal"] for value in task_metrics.values()),
        off_queue_events == 0,
        result["off_analyzer_marks_unavailable"],
        queue_check["passed"],
        max_wait_replays is None or all(
            replay["passed"] for replay in max_wait_replays.values()
        ),
    )
    result["passed"] = all(required)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument(
        "--scheduler",
        required=True,
        choices=("deadline_score", "max_wait_protection"),
    )
    parser.add_argument("--workers", required=True, type=int, choices=(2, 4))
    parser.add_argument(
        "--observation",
        choices=("summary", "detail"),
        default="summary",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate(
        args.off,
        args.summary,
        args.scheduler,
        args.workers,
        args.observation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "passed": result["passed"],
        "scheduler": result["scheduler"],
        "workers": result["workers"],
        "selection_count": result[
            "summary_queue_observation"
        ]["selection_count"],
    }, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(
            "M6B2.1 engineering equivalence failed; evidence was retained"
        )


if __name__ == "__main__":
    main()
