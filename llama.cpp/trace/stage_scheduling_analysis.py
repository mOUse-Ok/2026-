#!/usr/bin/env python3
"""Offline-only Expert Tensor Stage scheduling opportunity analysis."""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from typing import Any

import numpy as np


STAGES = ("EARLY", "LATE", "UNKNOWN")
PHASES = ("PREFILL", "DECODE", "UNKNOWN")
NO_ISSUED_REASONS = (
    "ttl_duplicate",
    "cache_hit",
    "policy_skip",
    "queue_or_worker_failure",
    "shutdown_pending",
    "other",
)


def _semantic_key(record: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        str(record.get("run_id", "single_run")),
        int(record.get("step", -1)),
        int(record.get("layer", -1)),
        int(record.get("expert", -1)),
        str(record.get("tensor", "")),
    )


def _percentiles(values: list[int]) -> dict[str, float | None]:
    return {
        name: float(np.percentile(values, percentile)) if values else None
        for name, percentile in (("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))
    }


def _reason_counts() -> dict[str, int]:
    return {reason: 0 for reason in NO_ISSUED_REASONS}


def _explicit_ttl_duplicate(record: dict[str, Any]) -> bool:
    markers = {
        "ttl_duplicate",
        "route_ttl_duplicate",
        "duplicate_hint",
        "expert_prefetch_skip_duplicate",
        "expert_route_hint_skip_duplicate",
        "expert_route_ttl_duplicate",
    }
    return any(str(record.get(field, "")).lower() in markers
               for field in ("action", "decision", "reason", "trigger", "outcome"))


def _explicit_cache_hit(record: dict[str, Any]) -> bool:
    return (record.get("cache_hit") is True or
            str(record.get("decision", "")).lower() == "hit" or
            str(record.get("action", "")).lower() == "expert_cache_hit")


def _explicit_queue_or_worker_failure(record: dict[str, Any]) -> bool:
    markers = {
        "queue_full",
        "queue_or_worker_failure",
        "worker_start_failure",
        "start_fail",
        "expert_prefetch_cancel_queue_full",
    }
    return any(str(record.get(field, "")).lower() in markers
               for field in ("action", "decision", "reason", "trigger", "outcome"))


def _explicit_policy_skip(record: dict[str, Any]) -> bool:
    if str(record.get("decision", "")).lower() == "skip":
        return True
    if record.get("event") != "EXPERT_TASK":
        return False
    return (str(record.get("lifecycle_event", "")) in {"REJECT", "CANCEL"} and
            not _explicit_queue_or_worker_failure(record))


def classify_no_issued_task_reasons(memory_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify no_issued_task only when matching trace evidence is explicit."""
    task_records: dict[tuple[str, int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    hint_records: dict[tuple[str, int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    final_task_states: dict[tuple[str, int], str] = {}
    task_keys: dict[tuple[str, int], tuple[str, int, int, int, str]] = {}

    for record in memory_records:
        if record.get("event") == "EXPERT_TASK":
            key = _semantic_key(record)
            task_records[key].append(record)
            task_id = record.get("task_id")
            if isinstance(task_id, int) and task_id > 0:
                identity = (str(record.get("run_id", "single_run")), task_id)
                final_task_states[identity] = str(record.get("state", ""))
                task_keys[identity] = key
        elif record.get("event") in {"OS_HINT", "EXPERT_ROUTE_HINT"}:
            hint_records[_semantic_key(record)].append(record)

    shutdown_pending_keys = {
        task_keys[identity]
        for identity, state in final_task_states.items()
        if state in {"ENQUEUED", "DEQUEUED"} and identity in task_keys
    }

    overall = _reason_counts()
    by_phase = {phase: _reason_counts() for phase in PHASES}
    by_stage = {stage: _reason_counts() for stage in STAGES}
    by_phase_stage = {
        phase: {stage: _reason_counts() for stage in STAGES}
        for phase in PHASES
    }
    observations_seen: set[tuple[Any, ...]] = set()

    for record in memory_records:
        if (record.get("event") != "EXPERT_FIRST_USE" or
                record.get("unmatched_reason") != "no_issued_task"):
            continue
        observation_id = (
            *_semantic_key(record),
            str(record.get("stage", "UNKNOWN")),
            int(record.get("first_use_ts_ns", record.get("ts_ns", 0))),
        )
        if observation_id in observations_seen:
            continue
        observations_seen.add(observation_id)

        key = _semantic_key(record)
        first_use_ts = int(record.get("first_use_ts_ns", record.get("ts_ns", 0)))
        evidence = [
            candidate for candidate in task_records.get(key, []) + hint_records.get(key, [])
            if int(candidate.get("ts_ns", 0)) <= first_use_ts
        ]
        if any(_explicit_ttl_duplicate(candidate) for candidate in evidence):
            reason = "ttl_duplicate"
        elif any(_explicit_cache_hit(candidate) for candidate in evidence):
            reason = "cache_hit"
        elif any(_explicit_queue_or_worker_failure(candidate) for candidate in evidence):
            reason = "queue_or_worker_failure"
        elif key in shutdown_pending_keys:
            reason = "shutdown_pending"
        elif any(_explicit_policy_skip(candidate) for candidate in evidence):
            reason = "policy_skip"
        else:
            reason = "other"

        phase = str(record.get("phase", "UNKNOWN"))
        stage = str(record.get("stage", "UNKNOWN"))
        phase = phase if phase in PHASES else "UNKNOWN"
        stage = stage if stage in STAGES else "UNKNOWN"
        overall[reason] += 1
        by_phase[phase][reason] += 1
        by_stage[stage][reason] += 1
        by_phase_stage[phase][stage][reason] += 1

    return {
        "total_no_issued_task": len(observations_seen),
        "reason_counts": overall,
        "by_phase": by_phase,
        "by_stage": by_stage,
        "by_phase_stage": by_phase_stage,
        "classification_semantics": "explicit_trace_evidence_else_other",
    }


def _new_inversion_accumulator() -> dict[str, Any]:
    return {
        "eligible_late_tasks": 0,
        "inversion_count": 0,
        "blocked_early_tasks": 0,
        "unique_blocked_early_task_ids": set(),
        "inverted_late_bytes": 0,
        "blocked_durations_ns": [],
        "blocked_duration_unknown_count": 0,
    }


def _finalize_inversion_accumulator(accumulator: dict[str, Any]) -> dict[str, Any]:
    eligible = int(accumulator["eligible_late_tasks"])
    inversions = int(accumulator["inversion_count"])
    return {
        "eligible_late_tasks": eligible,
        "inversion_count": inversions,
        "inversion_ratio": inversions / eligible if eligible else 0.0,
        "blocked_early_tasks": int(accumulator["blocked_early_tasks"]),
        "unique_blocked_early_tasks": len(accumulator["unique_blocked_early_task_ids"]),
        "inverted_late_bytes": int(accumulator["inverted_late_bytes"]),
        "early_blocked_by_late_ns": _percentiles(accumulator["blocked_durations_ns"]),
        "blocked_duration_unknown_count": int(accumulator["blocked_duration_unknown_count"]),
    }


def analyze_stage_inversions(memory_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect a LATE dequeue/issue while same-step/layer EARLY work remains queued."""
    tasks: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    for event_index, record in enumerate(memory_records):
        if record.get("event") != "EXPERT_TASK":
            continue
        task_id = record.get("task_id")
        if not isinstance(task_id, int) or task_id <= 0:
            continue
        identity = (str(record.get("run_id", "single_run")), task_id)
        task = tasks[identity]
        task.setdefault("identity", identity)
        task.setdefault("task_id", task_id)
        task.setdefault("run_id", identity[0])
        for field in ("step", "layer", "phase", "stage", "nbytes"):
            if field in record:
                task[field] = record[field]
        lifecycle_event = str(record.get("lifecycle_event", ""))
        event_point = (int(record.get("ts_ns", 0)), event_index)
        if lifecycle_event == "ENQUEUE":
            task["enqueue_point"] = event_point
        elif lifecycle_event == "DEQUEUE":
            task["dequeue_point"] = event_point
        elif lifecycle_event == "ISSUE":
            task["issue_point"] = (int(record.get("issued_ts_ns", record.get("ts_ns", 0))), event_index)

    early_by_group: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    late_tasks: list[dict[str, Any]] = []
    for task in tasks.values():
        if "enqueue_point" not in task:
            continue
        group = (str(task.get("run_id", "single_run")), int(task.get("step", -1)), int(task.get("layer", -1)))
        release_points = [
            task[field] for field in ("dequeue_point", "issue_point") if field in task
        ]
        task["release_point"] = min(release_points) if release_points else None
        if task.get("stage") == "EARLY":
            early_by_group[group].append(task)
        elif task.get("stage") == "LATE" and task["release_point"] is not None:
            late_tasks.append(task)

    overall = _new_inversion_accumulator()
    phase_accumulators: dict[str, dict[str, Any]] = {
        phase: _new_inversion_accumulator() for phase in PHASES
    }
    layer_accumulators: dict[int, dict[str, Any]] = defaultdict(_new_inversion_accumulator)

    def update(accumulator: dict[str, Any], late: dict[str, Any], blocked: list[dict[str, Any]]) -> None:
        accumulator["eligible_late_tasks"] += 1
        if not blocked:
            return
        accumulator["inversion_count"] += 1
        accumulator["blocked_early_tasks"] += len(blocked)
        accumulator["inverted_late_bytes"] += int(late.get("nbytes", 0))
        late_point = late["release_point"]
        for early in blocked:
            accumulator["unique_blocked_early_task_ids"].add(early["identity"])
            release = early.get("release_point")
            if release is None:
                accumulator["blocked_duration_unknown_count"] += 1
            else:
                accumulator["blocked_durations_ns"].append(max(0, int(release[0]) - int(late_point[0])))

    for late in late_tasks:
        group = (str(late.get("run_id", "single_run")), int(late.get("step", -1)), int(late.get("layer", -1)))
        late_point = late["release_point"]
        blocked = [
            early for early in early_by_group.get(group, [])
            if early["enqueue_point"] <= late_point and
            (early.get("release_point") is None or early["release_point"] > late_point)
        ]
        phase = str(late.get("phase", "UNKNOWN"))
        phase = phase if phase in PHASES else "UNKNOWN"
        layer = int(late.get("layer", -1))
        update(overall, late, blocked)
        update(phase_accumulators[phase], late, blocked)
        update(layer_accumulators[layer], late, blocked)

    return {
        **_finalize_inversion_accumulator(overall),
        "by_phase": {
            phase: _finalize_inversion_accumulator(phase_accumulators[phase])
            for phase in PHASES
        },
        "by_layer": {
            str(layer): _finalize_inversion_accumulator(layer_accumulators[layer])
            for layer in sorted(layer_accumulators)
        },
        "event_semantics": "first_of_dequeue_or_issue_per_late_task",
    }


def extract_simulation_tasks(memory_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build fixed-service simulation jobs from real Task and first-use events."""
    first_use_by_key: dict[tuple[str, int, int, int, str], int] = {}
    task_records: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    enqueue_order: dict[tuple[str, int], int] = {}

    for event_index, record in enumerate(memory_records):
        if record.get("event") == "EXPERT_FIRST_USE":
            key = _semantic_key(record)
            timestamp = int(record.get("first_use_ts_ns", record.get("ts_ns", 0)))
            if timestamp > 0:
                first_use_by_key[key] = min(first_use_by_key.get(key, timestamp), timestamp)
        elif record.get("event") == "EXPERT_TASK":
            task_id = record.get("task_id")
            if not isinstance(task_id, int) or task_id <= 0:
                continue
            identity = (str(record.get("run_id", "single_run")), task_id)
            task_records[identity].append((event_index, record))
            if record.get("lifecycle_event") == "ENQUEUE" and identity not in enqueue_order:
                enqueue_order[identity] = len(enqueue_order)

    jobs: list[dict[str, Any]] = []
    quality = Counter()
    for identity, indexed_records in task_records.items():
        quality["task_ids_seen"] += 1
        records = [record for _, record in indexed_records]
        enqueue = next((record for record in records if record.get("lifecycle_event") == "ENQUEUE"), None)
        issue = next((record for record in reversed(records) if record.get("lifecycle_event") == "ISSUE"), None)
        if enqueue is None:
            quality["excluded_without_enqueue"] += 1
            continue
        if issue is None:
            quality["excluded_without_issue"] += 1
            continue
        issued_ts = int(issue.get("issued_ts_ns", 0))
        returned_ts = int(issue.get("returned_ts_ns", issue.get("ts_ns", 0)))
        if issued_ts <= 0 or returned_ts < issued_ts:
            quality["excluded_invalid_service_duration"] += 1
            continue

        sequence_value = issue.get("sequence", enqueue.get("sequence"))
        if isinstance(sequence_value, int) and sequence_value >= 0:
            sequence = sequence_value
            quality["sequence_from_trace"] += 1
        else:
            sequence = enqueue_order[identity]
            quality["sequence_from_enqueue_order_proxy"] += 1
        if "deadline_ts_ns" in issue:
            quality["deadline_from_trace"] += 1
            if int(issue.get("deadline_ts_ns", 0)) > 0:
                quality["nonzero_deadline_tasks"] += 1
            else:
                quality["zero_deadline_tasks"] += 1
        else:
            quality["deadline_missing_default_zero"] += 1
            quality["zero_deadline_tasks"] += 1

        key = _semantic_key(issue)
        jobs.append({
            "identity": identity,
            "task_id": identity[1],
            "run_id": identity[0],
            "arrival_ts_ns": int(enqueue.get("enqueued_ts_ns", enqueue.get("ts_ns", 0))),
            "observed_issue_ts_ns": issued_ts,
            "observed_returned_ts_ns": returned_ts,
            "raw_service_ns": returned_ts - issued_ts,
            "service_ns": returned_ts - issued_ts,
            "issue_id": int(issue.get("issue_id", 0)),
            "issue_task_count": int(issue.get("issue_task_count", 1)),
            "deadline_ts_ns": int(issue.get("deadline_ts_ns", 0)),
            "first_use_ts_ns": first_use_by_key.get(key),
            "phase": str(issue.get("phase", "UNKNOWN")),
            "stage": str(issue.get("stage", "UNKNOWN")),
            "step": int(issue.get("step", -1)),
            "layer": int(issue.get("layer", -1)),
            "score": float(issue.get("score", 0.0)),
            "sequence": sequence,
            "nbytes": int(issue.get("nbytes", 0)),
        })

    issue_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        issue_group_id = job["issue_id"] if job["issue_id"] > 0 else -job["task_id"]
        issue_groups[(job["run_id"], issue_group_id)].append(job)

    coalesced_groups = 0
    for group in issue_groups.values():
        if len(group) <= 1:
            continue
        coalesced_groups += 1
        group.sort(key=lambda job: (job["sequence"], job["task_id"]))
        group_duration = max(job["observed_returned_ts_ns"] for job in group) - min(
            job["observed_issue_ts_ns"] for job in group
        )
        total_bytes = sum(max(1, job["nbytes"]) for job in group)
        allocated = 0
        for index, job in enumerate(group):
            if index == len(group) - 1:
                duration = max(0, group_duration - allocated)
            else:
                duration = int(group_duration * max(1, job["nbytes"]) / total_bytes)
                allocated += duration
            job["service_ns"] = duration

    quality["simulation_tasks"] = len(jobs)
    quality["tasks_with_first_use"] = sum(job["first_use_ts_ns"] is not None for job in jobs)
    quality["coalesced_issue_groups_decomposed"] = coalesced_groups
    for field in (
        "sequence_from_trace",
        "sequence_from_enqueue_order_proxy",
        "deadline_from_trace",
        "deadline_missing_default_zero",
        "nonzero_deadline_tasks",
        "zero_deadline_tasks",
    ):
        quality.setdefault(field, 0)
    if not jobs:
        sequence_semantics = "no_simulation_tasks"
        deadline_semantics = "no_simulation_tasks"
    else:
        sequence_semantics = (
            "trace_sequence" if quality["sequence_from_enqueue_order_proxy"] == 0
            else "trace_sequence_with_enqueue_order_proxy"
        )
        deadline_semantics = (
            "trace_deadline" if quality["deadline_missing_default_zero"] == 0
            else "trace_deadline_with_zero_default"
        )
    return jobs, {
        **dict(quality),
        "arrival_semantics": "enqueued_ts_ns",
        "sequence_semantics": sequence_semantics,
        "deadline_semantics": deadline_semantics,
        "service_semantics": "observed_issue_return_duration; coalesced_groups_split_by_bytes",
    }


def _deadline_score_key(job: dict[str, Any]) -> tuple[Any, ...]:
    deadline = int(job.get("deadline_ts_ns", 0))
    deadline_key = (0, deadline) if deadline > 0 else (1, 0)
    return (*deadline_key, -float(job.get("score", 0.0)), int(job["sequence"]), int(job["task_id"]))


def _stage_deadline_score_key(job: dict[str, Any]) -> tuple[Any, ...]:
    stage_rank = {"EARLY": 0, "LATE": 1, "UNKNOWN": 2}.get(str(job.get("stage")), 2)
    return (
        int(job.get("step", -1)),
        int(job.get("layer", -1)),
        stage_rank,
        -float(job.get("score", 0.0)),
        int(job["sequence"]),
        int(job["task_id"]),
    )


def simulate_task_schedule(
        jobs: list[dict[str, Any]], worker_count: int, policy: str) -> dict[tuple[str, int], int]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if policy not in {"deadline_score", "stage_deadline_score"}:
        raise ValueError(f"unsupported policy: {policy}")
    if not jobs:
        return {}

    key_function = _deadline_score_key if policy == "deadline_score" else _stage_deadline_score_key
    arrivals = sorted(
        jobs,
        key=lambda job: (int(job["arrival_ts_ns"]), int(job["sequence"]), int(job["task_id"])),
    )
    worker_heap = [(int(arrivals[0]["arrival_ts_ns"]), worker_id) for worker_id in range(worker_count)]
    heapq.heapify(worker_heap)
    ready: list[tuple[tuple[Any, ...], str, int, dict[str, Any]]] = []
    issue_times: dict[tuple[str, int], int] = {}
    arrival_index = 0
    current_time = int(arrivals[0]["arrival_ts_ns"])

    while len(issue_times) < len(arrivals):
        worker_free, worker_id = heapq.heappop(worker_heap)
        current_time = max(current_time, worker_free)
        while arrival_index < len(arrivals) and int(arrivals[arrival_index]["arrival_ts_ns"]) <= current_time:
            job = arrivals[arrival_index]
            heapq.heappush(ready, (key_function(job), str(job["run_id"]), int(job["task_id"]), job))
            arrival_index += 1
        if not ready:
            current_time = max(current_time, int(arrivals[arrival_index]["arrival_ts_ns"]))
            while arrival_index < len(arrivals) and int(arrivals[arrival_index]["arrival_ts_ns"]) <= current_time:
                job = arrivals[arrival_index]
                heapq.heappush(ready, (key_function(job), str(job["run_id"]), int(job["task_id"]), job))
                arrival_index += 1

        _, _, _, job = heapq.heappop(ready)
        issue_times[job["identity"]] = current_time
        completion = current_time + max(0, int(job.get("service_ns", 0)))
        heapq.heappush(worker_heap, (completion, worker_id))

    return issue_times


def _on_time_rate(
        jobs: list[dict[str, Any]],
        issue_times: dict[tuple[str, int], int],
        stage: str) -> tuple[float | None, int]:
    eligible = [job for job in jobs if job.get("stage") == stage and job.get("first_use_ts_ns") is not None]
    if not eligible:
        return None, 0
    on_time = sum(issue_times[job["identity"]] < int(job["first_use_ts_ns"]) for job in eligible)
    return on_time / len(eligible), len(eligible)


def compare_stage_schedules(jobs: list[dict[str, Any]], worker_count: int) -> dict[str, Any]:
    baseline = simulate_task_schedule(jobs, worker_count, "deadline_score")
    staged = simulate_task_schedule(jobs, worker_count, "stage_deadline_score")
    early_advancement = [
        baseline[job["identity"]] - staged[job["identity"]]
        for job in jobs if job.get("stage") == "EARLY"
    ]
    late_delay = [
        staged[job["identity"]] - baseline[job["identity"]]
        for job in jobs if job.get("stage") == "LATE"
    ]
    early_before, early_eligible = _on_time_rate(jobs, baseline, "EARLY")
    early_after, _ = _on_time_rate(jobs, staged, "EARLY")
    late_before, late_eligible = _on_time_rate(jobs, baseline, "LATE")
    late_after, _ = _on_time_rate(jobs, staged, "LATE")

    known_first_use = [job for job in jobs if job.get("first_use_ts_ns") is not None]
    issue_after_before = sum(
        baseline[job["identity"]] >= int(job["first_use_ts_ns"])
        for job in known_first_use
    )
    issue_after_after = sum(
        staged[job["identity"]] >= int(job["first_use_ts_ns"])
        for job in known_first_use
    )
    newly_late_late_tasks = sum(
        job.get("stage") == "LATE" and job.get("first_use_ts_ns") is not None and
        baseline[job["identity"]] < int(job["first_use_ts_ns"]) <= staged[job["identity"]]
        for job in jobs
    )
    unchanged = sum(baseline[job["identity"]] == staged[job["identity"]] for job in jobs)

    return {
        "worker_count": worker_count,
        "tasks_simulated": len(jobs),
        "early_issue_advancement_ns": _percentiles(early_advancement),
        "late_issue_delay_ns": _percentiles(late_delay),
        "early_on_time_rate_before": early_before,
        "early_on_time_rate_after": early_after,
        "early_on_time_eligible_tasks": early_eligible,
        "late_on_time_rate_before": late_before,
        "late_on_time_rate_after": late_after,
        "late_on_time_eligible_tasks": late_eligible,
        "issue_after_first_use_before": issue_after_before,
        "issue_after_first_use_after": issue_after_after,
        "newly_late_late_tasks": newly_late_late_tasks,
        "unchanged_task_ratio": unchanged / len(jobs) if jobs else 0.0,
        "on_time_semantics": "hint_issue_timestamp_strictly_before_logical_first_use",
    }


def analyze_stage_scheduling_opportunity(memory_records: list[dict[str, Any]]) -> dict[str, Any]:
    no_issued = classify_no_issued_task_reasons(memory_records)
    inversions = analyze_stage_inversions(memory_records)
    jobs, data_quality = extract_simulation_tasks(memory_records)
    async_summaries = [
        record for record in memory_records
        if record.get("event") == "EXPERT_ASYNC_SUMMARY"
    ]
    if async_summaries:
        summary = async_summaries[-1]
        data_quality["observed_priority_mode"] = str(summary.get("priority_mode", "unknown"))
        data_quality["observed_worker_count"] = int(summary.get("workers", 0))
        data_quality["observed_priority_enabled"] = bool(summary.get("priority_enabled", False))
        data_quality["observed_priority_heap_enabled"] = bool(
            summary.get("priority_heap_enabled", False)
        )
    simulations = {
        str(workers): compare_stage_schedules(jobs, workers)
        for workers in (1, 2, 4)
    }
    opportunity_observed = inversions["inversion_count"] > 0
    late_risk = any(
        simulation["newly_late_late_tasks"] > 0 or
        (simulation["late_issue_delay_ns"]["p95"] is not None and
         simulation["late_issue_delay_ns"]["p95"] > 0)
        for simulation in simulations.values()
    )
    return {
        "no_issued_task_reasons": no_issued,
        "stage_inversion": inversions,
        "simulation": simulations,
        "data_quality": data_quality,
        "conclusion": {
            "scheduling_opportunity_observed": opportunity_observed,
            "potential_issue_time_change_only": True,
            "late_degradation_risk_observed": late_risk,
            "limitations": [
                "on_time means hint issue before logical first-use only",
                "fixed observed service durations do not model page residency",
                "no performance, page-load completion, or major-fault reduction claim",
            ],
        },
    }
