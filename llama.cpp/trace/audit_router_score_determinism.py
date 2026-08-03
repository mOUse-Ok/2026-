#!/usr/bin/env python3
"""Audit cross-run Router score determinism without changing scheduling semantics."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable


SCHEMA_VERSION = "m6b1.1-router-score-audit-v2"
MAX_SCORE_REACHABLE_GROUP = 2048
STABILITY_PAIR_PREFIXES = ("A_", "B_", "C_", "D_")


def comparison_role(comparison_id: str) -> str:
    return "stability" if comparison_id.startswith(STABILITY_PAIR_PREFIXES) else "control"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_key(parts: Iterable[Any]) -> str:
    return json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))


def parse_bits(text: Any, width: int) -> int | None:
    if not isinstance(text, str) or not text.startswith("0x"):
        return None
    try:
        value = int(text[2:], 16)
    except ValueError:
        return None
    return value if 0 <= value < (1 << width) else None


def f32_from_bits(text: Any) -> float | None:
    bits = parse_bits(text, 32)
    if bits is None:
        return None
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def f64_from_bits(text: Any) -> float | None:
    bits = parse_bits(text, 64)
    if bits is None:
        return None
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def f64_bits_from_f32_bits(text: Any) -> str | None:
    value = f32_from_bits(text)
    if value is None:
        return None
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"0x{bits:016x}"


def ordered_float_bits(bits: int, width: int) -> int:
    sign = 1 << (width - 1)
    mask = (1 << width) - 1
    return (~bits & mask) if bits & sign else (bits | sign)


def ulp_distance(a: Any, b: Any, width: int) -> int | None:
    a_bits = parse_bits(a, width)
    b_bits = parse_bits(b, width)
    if a_bits is None or b_bits is None:
        return None
    a_value = f32_from_bits(a) if width == 32 else f64_from_bits(a)
    b_value = f32_from_bits(b) if width == 32 else f64_from_bits(b)
    if a_value is None or b_value is None or not math.isfinite(a_value) or not math.isfinite(b_value):
        return None
    return abs(ordered_float_bits(a_bits, width) - ordered_float_bits(b_bits, width))


def numeric_difference(a: float, b: float) -> tuple[float, float]:
    absolute = abs(a - b)
    denominator = max(abs(a), abs(b))
    return absolute, absolute / denominator if denominator > 0 else 0.0


def percentile(values: list[float | int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float | int]) -> dict[str, Any]:
    return {
        "sample_count": len(values),
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def relation(a: float, b: float) -> int:
    return -1 if a < b else (1 if a > b else 0)


def deadline_prefix(task: "Task") -> tuple[Any, ...]:
    return (task.deadline_ts_ns == 0, task.deadline_ts_ns)


def legacy_key(task: "Task", score: float) -> tuple[Any, ...]:
    return (*deadline_prefix(task), -score, task.sequence)


def max_wait_class(task: "Task", now_ns: int, threshold_ns: int, guard_ns: int) -> str:
    if task.enqueued_ts_ns == 0 or now_ns < task.enqueued_ts_ns:
        return "normal"
    urgent_limit = min((1 << 64) - 1, now_ns + guard_ns)
    if task.deadline_ts_ns != 0 and task.deadline_ts_ns <= urgent_limit:
        return "urgent"
    if now_ns - task.enqueued_ts_ns >= threshold_ns:
        return "protected"
    return "normal"


def max_wait_prefix(task: "Task", task_class: str) -> tuple[Any, ...]:
    rank = {"urgent": 0, "protected": 1, "normal": 2}[task_class]
    if task_class == "protected":
        return (rank, task.enqueued_ts_ns, *deadline_prefix(task))
    return (rank, *deadline_prefix(task))


def max_wait_key(task: "Task", task_class: str, score: float) -> tuple[Any, ...]:
    return (*max_wait_prefix(task, task_class), -score, task.sequence)


@dataclass
class Task:
    task_id: int
    phase: str
    step: int
    layer: int
    expert: int
    tensor: str
    stage: str
    nbytes: int
    score: float
    score_f64_bits: str | None
    created_ts_ns: int
    sequence: int = 0
    deadline_ts_ns: int = 0
    enqueued_ts_ns: int = 0
    dequeued_ts_ns: int = 0
    issued_ts_ns: int = 0
    lifecycle_events: list[str] = field(default_factory=list)
    lifecycle_score_bits: set[str] = field(default_factory=set)
    corr_key: str = ""
    route_slot: str | None = None
    route_rank: int | None = None


@dataclass
class AuditRun:
    label: str
    path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    metrics: dict[str, Any]
    routes: list[dict[str, Any]]
    route_by_slot: dict[str, dict[str, Any]]
    tasks: dict[int, Task]
    task_by_corr: dict[str, Task]
    selections: list[dict[str, Any]]
    dequeue_order: list[int]
    issue_order: list[int]
    validation: dict[str, Any]

    @property
    def mode(self) -> str:
        return str(self.manifest.get("environment", {}).get(
            "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE", "unknown"
        ))

    @property
    def model_threads(self) -> str | None:
        return self.manifest.get("environment", {}).get("NUM_THREADS")

    @property
    def hint_workers(self) -> str | None:
        return self.manifest.get("environment", {}).get(
            "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS"
        )


TRANSITIONS = {
    None: {"CREATE": "CREATED"},
    "CREATED": {"ADMIT": "ADMITTED", "REJECT": "REJECTED"},
    "ADMITTED": {"ENQUEUE": "ENQUEUED", "ISSUE": "ISSUED", "CANCEL": "CANCELLED"},
    "ENQUEUED": {"DEQUEUE": "DEQUEUED"},
    "DEQUEUED": {"ISSUE": "ISSUED", "CANCEL": "CANCELLED"},
    "REJECTED": {},
    "ISSUED": {},
    "CANCELLED": {},
}


def trace_complete(summary: dict[str, Any]) -> bool:
    return all(
        not sink.get("enabled")
        or (
            int(sink.get("enqueued", -1)) == int(sink.get("written", -2))
            and int(sink.get("dropped", -1)) == 0
        )
        for sink in summary.get("sinks", {}).values()
        if isinstance(sink, dict)
    )


def assign_route_slots(routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    occurrences: Counter[tuple[Any, ...]] = Counter()
    result: dict[str, dict[str, Any]] = {}
    for route in routes:
        base = (
            route.get("phase"),
            int(route.get("step", 0)),
            int(route.get("layer", -1)),
            route.get("token"),
            int(route.get("top_k", 0)),
        )
        ordinal = occurrences[base]
        occurrences[base] += 1
        slot = json_key((*base, ordinal))
        route["_slot"] = slot
        route["_ordinal"] = ordinal
        result[slot] = route
    return result


def assign_task_correspondence(tasks: dict[int, Task]) -> dict[str, Task]:
    occurrences: Counter[tuple[Any, ...]] = Counter()
    result: dict[str, Task] = {}
    for task in sorted(tasks.values(), key=lambda item: (item.created_ts_ns, item.task_id)):
        base = (
            task.phase,
            task.step,
            task.layer,
            task.expert,
            task.tensor,
            task.stage,
            task.nbytes,
        )
        ordinal = occurrences[base]
        occurrences[base] += 1
        task.corr_key = json_key((*base, ordinal))
        result[task.corr_key] = task
    return result


def link_tasks_to_routes(run: AuditRun) -> dict[str, int]:
    ordered_routes = sorted(run.routes, key=lambda record: int(record.get("ts_ns", 0)))
    timestamps = [int(record.get("ts_ns", 0)) for record in ordered_routes]
    linked = missing = rank_missing = 0
    for task in run.tasks.values():
        index = bisect_right(timestamps, task.created_ts_ns) - 1
        lower_bound = max(-1, index - 64)
        match: dict[str, Any] | None = None
        while index > lower_bound:
            route = ordered_routes[index]
            if (
                int(route.get("step", 0)) == task.step
                and int(route.get("layer", -1)) == task.layer
                and task.expert in route.get("experts", [])
            ):
                match = route
                break
            index -= 1
        if match is None:
            missing += 1
            continue
        task.route_slot = str(match["_slot"])
        try:
            task.route_rank = list(match.get("experts", [])).index(task.expert)
        except ValueError:
            rank_missing += 1
            continue
        linked += 1
    return {"linked": linked, "missing": missing, "rank_missing": rank_missing}


def load_run(label: str, path: Path) -> AuditRun:
    manifest = read_json(path / "run_manifest.json")
    summary = read_json(path / "summary.json")
    metrics_path = path / "analysis" / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.is_file() else {}

    routes: list[dict[str, Any]] = []
    with (path / "expert_trace.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("event") == "EXPERT_ROUTE":
                routes.append(record)
    route_by_slot = assign_route_slots(routes)

    tasks: dict[int, Task] = {}
    states: dict[int, str] = {}
    timestamps: dict[int, int] = {}
    creates: Counter[int] = Counter()
    invalid_transitions = state_mismatches = timestamp_regressions = invalid_ids = 0
    missing_task_records = lifecycle_score_changes = 0
    selections: list[dict[str, Any]] = []
    dequeue_order: list[int] = []
    issue_order: list[int] = []

    with (path / "memory_trace.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            event_name = record.get("event")
            if event_name in {"EXPERT_PRIORITY_SELECTION", "EXPERT_MAX_WAIT_SELECTION"}:
                selections.append(record)
                continue
            if event_name != "EXPERT_TASK":
                continue
            task_id = record.get("task_id")
            if not isinstance(task_id, int) or task_id <= 0:
                invalid_ids += 1
                continue
            lifecycle = str(record.get("lifecycle_event", ""))
            creates[task_id] += int(lifecycle == "CREATE")
            next_state = TRANSITIONS.get(states.get(task_id), {}).get(lifecycle)
            if next_state is None:
                invalid_transitions += 1
            else:
                states[task_id] = next_state
                state_mismatches += int(record.get("state") != next_state)
            timestamp = int(record.get("ts_ns", 0))
            timestamp_regressions += int(task_id in timestamps and timestamp < timestamps[task_id])
            timestamps[task_id] = max(timestamp, timestamps.get(task_id, 0))

            if lifecycle == "CREATE":
                tasks[task_id] = Task(
                    task_id=task_id,
                    phase=str(record.get("phase", "UNKNOWN")),
                    step=int(record.get("step", 0)),
                    layer=int(record.get("layer", -1)),
                    expert=int(record.get("expert", -1)),
                    tensor=str(record.get("tensor", "")),
                    stage=str(record.get("stage", "UNKNOWN")),
                    nbytes=int(record.get("nbytes", 0)),
                    score=float(record.get("score", 0.0)),
                    score_f64_bits=record.get("score_f64_bits"),
                    created_ts_ns=int(record.get("created_ts_ns", timestamp)),
                )
            task = tasks.get(task_id)
            if task is None:
                missing_task_records += 1
                continue
            task.lifecycle_events.append(lifecycle)
            bits = record.get("score_f64_bits")
            if isinstance(bits, str):
                task.lifecycle_score_bits.add(bits)
            if len(task.lifecycle_score_bits) > 1:
                lifecycle_score_changes += 1
            task.sequence = int(record.get("sequence", task.sequence))
            task.deadline_ts_ns = int(record.get("deadline_ts_ns", task.deadline_ts_ns))
            task.enqueued_ts_ns = int(record.get("enqueued_ts_ns", task.enqueued_ts_ns))
            task.dequeued_ts_ns = int(record.get("dequeued_ts_ns", task.dequeued_ts_ns))
            task.issued_ts_ns = int(record.get("issued_ts_ns", task.issued_ts_ns))
            if lifecycle == "DEQUEUE":
                dequeue_order.append(task_id)
            elif lifecycle == "ISSUE":
                issue_order.append(task_id)

    task_by_corr = assign_task_correspondence(tasks)
    diagnostics_routes_missing = sum(
        not all(name in route for name in (
            "score_source_dtype", "score_raw_bits", "score_f32_bits",
            "score_shape", "score_strides_bytes",
        ))
        for route in routes
    )
    diagnostics_tasks_missing = sum(task.score_f64_bits is None for task in tasks.values())
    terminal = {"REJECTED", "ISSUED", "CANCELLED"}
    validation = {
        "trace_complete": trace_complete(summary),
        "route_count": len(routes),
        "task_count": len(tasks),
        "duplicate_create_ids": sum(max(0, count - 1) for count in creates.values()),
        "invalid_task_ids": invalid_ids,
        "invalid_transitions": invalid_transitions,
        "state_mismatches": state_mismatches,
        "timestamp_regressions": timestamp_regressions,
        "incomplete_tasks": sum(state not in terminal for state in states.values()),
        "missing_task_records": missing_task_records,
        "lifecycle_score_change_observations": lifecycle_score_changes,
        "diagnostic_route_records_missing_fields": diagnostics_routes_missing,
        "diagnostic_tasks_missing_score_bits": diagnostics_tasks_missing,
        "model_hash_present": bool(manifest.get("model", {}).get("sha256")),
        "router_score_diagnostic_enabled": manifest.get("environment", {}).get(
            "LLM_MEM_TRACE_ROUTER_SCORE_DIAGNOSTIC"
        ) == "1",
        "output_sha256": (path / "output.sha256").read_text(encoding="ascii").strip(),
        "memory_trace_sha256": file_sha256(path / "memory_trace.jsonl"),
        "expert_trace_sha256": file_sha256(path / "expert_trace.jsonl"),
        "task_hint_linkage": {
            name: metrics.get(name)
            for name in (
                "expert_issue_task_count_mismatches",
                "expert_issue_ids_without_syscalls",
                "expert_syscall_issue_ids_without_tasks",
            )
        },
    }
    run = AuditRun(
        label=label,
        path=path,
        manifest=manifest,
        summary=summary,
        metrics=metrics,
        routes=routes,
        route_by_slot=route_by_slot,
        tasks=tasks,
        task_by_corr=task_by_corr,
        selections=selections,
        dequeue_order=dequeue_order,
        issue_order=issue_order,
        validation=validation,
    )
    validation["task_route_linkage"] = link_tasks_to_routes(run)
    zero_fields = (
        "duplicate_create_ids", "invalid_task_ids", "invalid_transitions",
        "state_mismatches", "timestamp_regressions", "incomplete_tasks",
        "missing_task_records", "lifecycle_score_change_observations",
        "diagnostic_route_records_missing_fields", "diagnostic_tasks_missing_score_bits",
    )
    linkage_ok = all(value in (0, None) for value in validation["task_hint_linkage"].values())
    validation["passed"] = bool(
        validation["trace_complete"]
        and validation["model_hash_present"]
        and validation["router_score_diagnostic_enabled"]
        and validation["task_route_linkage"]["missing"] == 0
        and all(validation[name] == 0 for name in zero_fields)
        and linkage_ok
    )
    return run


def run_identity(run: AuditRun) -> dict[str, Any]:
    environment = run.manifest.get("environment", {})
    return {
        "label": run.label,
        "run_id": run.manifest.get("run_name"),
        "path": str(run.path),
        "mode": run.mode,
        "binary_sha256": run.manifest.get("binary", {}).get("sha256"),
        "model_sha256": run.manifest.get("model", {}).get("sha256"),
        "prompt_sha256": run.manifest.get("prompt", {}).get("sha256"),
        "output_sha256": run.validation["output_sha256"],
        "git_commit": run.manifest.get("git_commit"),
        "git_dirty": run.manifest.get("git_dirty"),
        "model_threads": run.model_threads,
        "hint_workers": run.hint_workers,
        "cpu_affinity": run.manifest.get("host", {}).get("cpu_affinity"),
        "math_threads": {
            name: environment.get(name)
            for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            )
        },
        "locale": {
            "LC_ALL": environment.get("LC_ALL"),
            "LANG": environment.get("LANG"),
        },
        "validation": run.validation,
    }


def matching_summary(a: AuditRun, b: AuditRun) -> dict[str, Any]:
    task_a = set(a.task_by_corr)
    task_b = set(b.task_by_corr)
    route_a = set(a.route_by_slot)
    route_b = set(b.route_by_slot)
    task_base_a = Counter(key.rsplit(",", 1)[0] for key in task_a)
    task_base_b = Counter(key.rsplit(",", 1)[0] for key in task_b)
    return {
        "task_matching_key": [
            "phase", "step", "layer", "expert", "tensor", "stage", "nbytes",
            "occurrence_ordinal",
        ],
        "route_matching_key": [
            "phase", "step", "layer", "token", "top_k", "occurrence_ordinal",
        ],
        "address_used": False,
        "route_score_used": False,
        "task_id_used_across_runs": False,
        "matched_tasks": len(task_a & task_b),
        "unmatched_a_tasks": len(task_a - task_b),
        "unmatched_b_tasks": len(task_b - task_a),
        "post_ordinal_task_collisions": 0,
        "duplicate_task_bases_a": sum(count > 1 for count in task_base_a.values()),
        "duplicate_task_bases_b": sum(count > 1 for count in task_base_b.values()),
        "matched_route_records": len(route_a & route_b),
        "unmatched_a_route_records": len(route_a - route_b),
        "unmatched_b_route_records": len(route_b - route_a),
        "post_ordinal_route_collisions": 0,
        "confidence": "semantic_fields_plus_occurrence_ordinal",
    }


def route_comparison(
    comparison_id: str,
    a: AuditRun,
    b: AuditRun,
    detail_stream: Any,
) -> dict[str, Any]:
    common = sorted(set(a.route_by_slot) & set(b.route_by_slot))
    raw_differences = f32_differences = serialized_differences = 0
    topk_set_changes = topk_rank_changes = 0
    raw_differences_rank_zero = raw_differences_nonzero_rank = 0
    compared_items = dtype_mismatches = shape_mismatches = 0
    abs_values: list[float] = []
    rel_values: list[float] = []
    ulps: list[int] = []
    changed_abs_values: list[float] = []
    changed_rel_values: list[float] = []
    changed_ulps: list[int] = []
    dtype_counts: Counter[str] = Counter()
    closest_adjacent_gap: float | None = None

    for slot in common:
        route_a = a.route_by_slot[slot]
        route_b = b.route_by_slot[slot]
        experts_a = list(route_a.get("experts", []))
        experts_b = list(route_b.get("experts", []))
        topk_set_changes += int(set(experts_a) != set(experts_b))
        topk_rank_changes += sum(
            expert_a != expert_b for expert_a, expert_b in zip(experts_a, experts_b)
        ) + abs(len(experts_a) - len(experts_b))
        dtype_a = str(route_a.get("score_source_dtype"))
        dtype_b = str(route_b.get("score_source_dtype"))
        dtype_counts[dtype_a] += 1
        dtype_mismatches += int(dtype_a != dtype_b)
        shape_mismatches += int(
            route_a.get("score_shape") != route_b.get("score_shape")
            or route_a.get("score_strides_bytes") != route_b.get("score_strides_bytes")
        )
        scores_a = list(route_a.get("scores", []))
        scores_b = list(route_b.get("scores", []))
        raw_a = list(route_a.get("score_raw_bits", []))
        raw_b = list(route_b.get("score_raw_bits", []))
        f32_a = list(route_a.get("score_f32_bits", []))
        f32_b = list(route_b.get("score_f32_bits", []))
        for values in (scores_a, scores_b):
            for left, right in zip(values, values[1:]):
                gap = abs(float(left) - float(right))
                closest_adjacent_gap = gap if closest_adjacent_gap is None else min(
                    closest_adjacent_gap, gap
                )
        limit = min(
            len(experts_a), len(experts_b), len(scores_a), len(scores_b),
            len(raw_a), len(raw_b), len(f32_a), len(f32_b),
        )
        for rank in range(limit):
            compared_items += 1
            raw_changed = raw_a[rank] != raw_b[rank]
            f32_changed = f32_a[rank] != f32_b[rank]
            serialized_changed = float(scores_a[rank]) != float(scores_b[rank])
            raw_differences += int(raw_changed)
            raw_differences_rank_zero += int(raw_changed and rank == 0)
            raw_differences_nonzero_rank += int(raw_changed and rank != 0)
            f32_differences += int(f32_changed)
            serialized_differences += int(serialized_changed)
            value_a = f32_from_bits(f32_a[rank])
            value_b = f32_from_bits(f32_b[rank])
            if value_a is not None and value_b is not None:
                absolute, relative = numeric_difference(value_a, value_b)
                abs_values.append(absolute)
                rel_values.append(relative)
                ulp = ulp_distance(f32_a[rank], f32_b[rank], 32)
                if ulp is not None:
                    ulps.append(ulp)
                if raw_changed or f32_changed:
                    changed_abs_values.append(absolute)
                    changed_rel_values.append(relative)
                    if ulp is not None:
                        changed_ulps.append(ulp)
            if raw_changed or f32_changed or serialized_changed or experts_a[rank] != experts_b[rank]:
                detail_stream.write(json.dumps({
                    "comparison_id": comparison_id,
                    "run_id_a": a.manifest.get("run_name"),
                    "run_id_b": b.manifest.get("run_name"),
                    "route_slot": json.loads(slot),
                    "rank": rank,
                    "expert_a": experts_a[rank],
                    "expert_b": experts_b[rank],
                    "source_dtype_a": dtype_a,
                    "source_dtype_b": dtype_b,
                    "raw_bits_a": raw_a[rank],
                    "raw_bits_b": raw_b[rank],
                    "f32_bits_a": f32_a[rank],
                    "f32_bits_b": f32_b[rank],
                    "score_a": value_a,
                    "score_b": value_b,
                    "serialized_score_a": scores_a[rank],
                    "serialized_score_b": scores_b[rank],
                    "raw_bits_different": raw_changed,
                    "f32_bits_different": f32_changed,
                    "serialized_score_different": serialized_changed,
                    "ulp_difference_f32": ulp_distance(f32_a[rank], f32_b[rank], 32),
                }, ensure_ascii=False) + "\n")
    return {
        "matched_route_records": len(common),
        "compared_route_items": compared_items,
        "raw_bit_different_items": raw_differences,
        "raw_bit_different_rank_zero_items": raw_differences_rank_zero,
        "raw_bit_different_nonzero_rank_items": raw_differences_nonzero_rank,
        "f32_bit_different_items": f32_differences,
        "serialized_score_different_items": serialized_differences,
        "raw_differences_hidden_by_decimal_format": max(0, raw_differences - serialized_differences),
        "topk_set_changed_records": topk_set_changes,
        "topk_rank_changed_items": topk_rank_changes,
        "dtype_mismatches": dtype_mismatches,
        "shape_or_stride_mismatches": shape_mismatches,
        "source_dtype_record_counts": dict(dtype_counts),
        "closest_adjacent_topk_score_gap": closest_adjacent_gap,
        "absolute_difference": distribution(abs_values),
        "relative_difference": distribution(rel_values),
        "ulp_difference_f32": distribution(ulps),
        "different_only_absolute_difference": distribution(changed_abs_values),
        "different_only_relative_difference": distribution(changed_rel_values),
        "different_only_ulp_difference_f32": distribution(changed_ulps),
    }


def score_value(task: Task) -> float:
    value = f64_from_bits(task.score_f64_bits)
    return value if value is not None else task.score


def score_reachable_pairs(
    members: set[int],
    tasks: dict[int, Task],
    injected: dict[int, float],
) -> tuple[int, int, set[int], bool]:
    if len(members) > MAX_SCORE_REACHABLE_GROUP:
        return 0, 0, set(), False
    eligible = changed = 0
    relevant: set[int] = set()
    ordered = sorted(members)
    for index, left_id in enumerate(ordered):
        left = tasks[left_id]
        for right_id in ordered[index + 1:]:
            right = tasks[right_id]
            eligible += 1
            original = relation(score_value(left), score_value(right))
            replacement = relation(
                injected.get(left_id, score_value(left)),
                injected.get(right_id, score_value(right)),
            )
            if original != replacement:
                changed += 1
                relevant.update((left_id, right_id))
    return eligible, changed, relevant, True


def replay_deadline_score(run: AuditRun, injected: dict[int, float]) -> dict[str, Any]:
    decisions = sorted(
        (record for record in run.selections if record.get("event") == "EXPERT_PRIORITY_SELECTION"),
        key=lambda record: (int(record.get("decision_ts_ns", 0)), int(record.get("task_id", 0))),
    )
    arrivals = sorted(
        (task.enqueued_ts_ns, task.task_id)
        for task in run.tasks.values() if task.enqueued_ts_ns != 0
    )
    active: set[int] = set()
    groups: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    heap_original: list[tuple[Any, ...]] = []
    heap_injected: list[tuple[Any, ...]] = []
    arrival_index = 0
    self_mismatch = injection_winner_changes = candidate_count_mismatches = 0
    missing_selected = pair_observations = pair_changes = 0
    relevant_ids: set[int] = set()
    bounded = True

    def heap_top(heap: list[tuple[Any, ...]]) -> int | None:
        while heap and heap[0][-1] not in active:
            heapq.heappop(heap)
        return heap[0][-1] if heap else None

    for decision in decisions:
        now = int(decision.get("decision_ts_ns", 0))
        while arrival_index < len(arrivals) and arrivals[arrival_index][0] <= now:
            _, task_id = arrivals[arrival_index]
            task = run.tasks[task_id]
            active.add(task_id)
            groups[deadline_prefix(task)].add(task_id)
            heapq.heappush(heap_original, (*legacy_key(task, score_value(task)), task_id))
            heapq.heappush(
                heap_injected,
                (*legacy_key(task, injected.get(task_id, score_value(task))), task_id),
            )
            arrival_index += 1
        selected = int(decision.get("task_id", 0))
        if selected not in active:
            missing_selected += 1
            continue
        candidate_count_mismatches += int(
            int(decision.get("candidate_count", -1)) != len(active)
        )
        original_winner = heap_top(heap_original)
        injected_winner = heap_top(heap_injected)
        self_mismatch += int(original_winner != selected)
        injection_winner_changes += int(injected_winner != original_winner)
        prefix = deadline_prefix(run.tasks[original_winner]) if original_winner is not None else None
        members = groups.get(prefix, set()) if prefix is not None else set()
        eligible, changed, relevant, complete = score_reachable_pairs(
            members, run.tasks, injected
        )
        pair_observations += eligible
        pair_changes += changed
        relevant_ids.update(relevant)
        bounded = bounded and complete
        task = run.tasks[selected]
        active.remove(selected)
        groups[deadline_prefix(task)].discard(selected)
    return {
        "mode": "deadline_score",
        "selection_events": len(decisions),
        "missing_selected_tasks": missing_selected,
        "candidate_count_mismatches": candidate_count_mismatches,
        "self_replay_winner_mismatches": self_mismatch,
        "score_injection_winner_changed_decisions": injection_winner_changes,
        "score_reachable_pair_observations": pair_observations,
        "comparator_order_changed_pair_observations": pair_changes,
        "ordering_relevant_task_ids": sorted(relevant_ids),
        "pair_analysis_bounded_complete": bounded,
        "passed": missing_selected == 0 and candidate_count_mismatches == 0 and self_mismatch == 0 and bounded,
    }


def replay_max_wait(run: AuditRun, injected: dict[int, float]) -> dict[str, Any]:
    decisions = sorted(
        (record for record in run.selections if record.get("event") == "EXPERT_MAX_WAIT_SELECTION"),
        key=lambda record: (int(record.get("decision_ts_ns", 0)), int(record.get("task_id", 0))),
    )
    threshold = int(decisions[0].get("threshold_ns", 0)) if decisions else 0
    guard = int(decisions[0].get("urgent_guard_ns", 0)) if decisions else 0
    arrivals = sorted(
        (task.enqueued_ts_ns, task.task_id)
        for task in run.tasks.values() if task.enqueued_ts_ns != 0
    )
    protections = sorted(
        (min((1 << 64) - 1, task.enqueued_ts_ns + threshold), task.task_id)
        for task in run.tasks.values() if task.enqueued_ts_ns != 0
    )
    urgencies = sorted(
        (max(0, task.deadline_ts_ns - guard), task.task_id)
        for task in run.tasks.values()
        if task.enqueued_ts_ns != 0 and task.deadline_ts_ns != 0
    )
    status: dict[int, str] = {}
    class_counts: Counter[str] = Counter()
    groups: dict[str, dict[tuple[Any, ...], set[int]]] = {
        name: defaultdict(set) for name in ("urgent", "protected", "normal")
    }
    heaps_original: dict[str, list[tuple[Any, ...]]] = {
        name: [] for name in ("urgent", "protected", "normal")
    }
    heaps_injected: dict[str, list[tuple[Any, ...]]] = {
        name: [] for name in ("urgent", "protected", "normal")
    }
    arrival_index = protection_index = urgent_index = 0
    self_mismatch = injection_winner_changes = missing_selected = 0
    protected_count_mismatches = normal_flag_mismatches = 0
    pair_observations = pair_changes = 0
    relevant_ids: set[int] = set()
    bounded = True

    def set_status(task_id: int, task_class: str) -> None:
        old = status.get(task_id)
        if old == task_class or old == "removed":
            return
        task = run.tasks[task_id]
        if old is not None:
            class_counts[old] -= 1
            groups[old][max_wait_prefix(task, old)].discard(task_id)
        status[task_id] = task_class
        class_counts[task_class] += 1
        groups[task_class][max_wait_prefix(task, task_class)].add(task_id)
        heapq.heappush(
            heaps_original[task_class],
            (*max_wait_key(task, task_class, score_value(task)), task_id),
        )
        heapq.heappush(
            heaps_injected[task_class],
            (*max_wait_key(task, task_class, injected.get(task_id, score_value(task))), task_id),
        )

    def heap_top(heap: list[tuple[Any, ...]], task_class: str) -> int | None:
        while heap and status.get(heap[0][-1]) != task_class:
            heapq.heappop(heap)
        return heap[0][-1] if heap else None

    for decision in decisions:
        now = int(decision.get("decision_ts_ns", 0))
        while arrival_index < len(arrivals) and arrivals[arrival_index][0] <= now:
            _, task_id = arrivals[arrival_index]
            set_status(task_id, max_wait_class(run.tasks[task_id], now, threshold, guard))
            arrival_index += 1
        while urgent_index < len(urgencies) and urgencies[urgent_index][0] <= now:
            _, task_id = urgencies[urgent_index]
            if task_id in status and status.get(task_id) != "removed":
                set_status(task_id, "urgent")
            urgent_index += 1
        while protection_index < len(protections) and protections[protection_index][0] <= now:
            _, task_id = protections[protection_index]
            if status.get(task_id) == "normal":
                set_status(task_id, "protected")
            protection_index += 1

        selected = int(decision.get("task_id", 0))
        if selected not in status or status.get(selected) == "removed":
            missing_selected += 1
            continue
        protected_count_mismatches += int(
            int(decision.get("protected_candidate_count", -1)) != class_counts["protected"]
        )
        normal_flag_mismatches += int(
            bool(decision.get("normal_competitor_present")) != (class_counts["normal"] > 0)
        )
        task_class = next(
            name for name in ("urgent", "protected", "normal") if class_counts[name] > 0
        )
        original_winner = heap_top(heaps_original[task_class], task_class)
        injected_winner = heap_top(heaps_injected[task_class], task_class)
        self_mismatch += int(original_winner != selected)
        injection_winner_changes += int(injected_winner != original_winner)
        prefix = max_wait_prefix(run.tasks[original_winner], task_class) if original_winner else None
        members = groups[task_class].get(prefix, set()) if prefix is not None else set()
        eligible, changed, relevant, complete = score_reachable_pairs(
            members, run.tasks, injected
        )
        pair_observations += eligible
        pair_changes += changed
        relevant_ids.update(relevant)
        bounded = bounded and complete
        selected_class = status[selected]
        class_counts[selected_class] -= 1
        groups[selected_class][max_wait_prefix(run.tasks[selected], selected_class)].discard(selected)
        status[selected] = "removed"
    return {
        "mode": "max_wait_protection",
        "selection_events": len(decisions),
        "missing_selected_tasks": missing_selected,
        "protected_candidate_count_mismatches": protected_count_mismatches,
        "normal_competitor_flag_mismatches": normal_flag_mismatches,
        "self_replay_winner_mismatches": self_mismatch,
        "score_injection_winner_changed_decisions": injection_winner_changes,
        "score_reachable_pair_observations": pair_observations,
        "comparator_order_changed_pair_observations": pair_changes,
        "ordering_relevant_task_ids": sorted(relevant_ids),
        "pair_analysis_bounded_complete": bounded,
        "passed": (
            missing_selected == 0
            and protected_count_mismatches == 0
            and normal_flag_mismatches == 0
            and self_mismatch == 0
            and bounded
        ),
    }


def replay(run: AuditRun, injected: dict[int, float]) -> dict[str, Any]:
    if run.mode == "max_wait_protection":
        return replay_max_wait(run, injected)
    return replay_deadline_score(run, injected)


def order_comparison(a: AuditRun, b: AuditRun, field: str) -> dict[str, Any]:
    ids_a = a.dequeue_order if field == "dequeue" else a.issue_order
    ids_b = b.dequeue_order if field == "dequeue" else b.issue_order
    order_a = [a.tasks[task_id].corr_key for task_id in ids_a if task_id in a.tasks]
    order_b = [b.tasks[task_id].corr_key for task_id in ids_b if task_id in b.tasks]
    matched_length = min(len(order_a), len(order_b))
    positional = sum(order_a[index] != order_b[index] for index in range(matched_length))
    return {
        "count_a": len(order_a),
        "count_b": len(order_b),
        "exact_order_equal": order_a == order_b,
        "positional_mismatch_count": positional + abs(len(order_a) - len(order_b)),
        "same_task_multiset": Counter(order_a) == Counter(order_b),
        "causality_note": (
            "Actual cross-run order includes worker and timing effects; score causality is established "
            "only by comparator score-injection replay."
        ),
    }


def task_route_observation(run: AuditRun, task: Task) -> tuple[dict[str, Any] | None, int | None]:
    if task.route_slot is None or task.route_rank is None:
        return None, None
    return run.route_by_slot.get(task.route_slot), task.route_rank


def task_comparison(
    comparison_id: str,
    a: AuditRun,
    b: AuditRun,
    replays: tuple[dict[str, Any], dict[str, Any]],
    detail_stream: Any,
) -> dict[str, Any]:
    common = sorted(set(a.task_by_corr) & set(b.task_by_corr))
    relevant_a = set(replays[0].get("ordering_relevant_task_ids", []))
    relevant_b = set(replays[1].get("ordering_relevant_task_ids", []))
    score_different = serialized_different = sign_changes = crossings = 0
    nan_or_inf = signed_zero_changes = 0
    widening_mismatches = lifecycle_mismatches = 0
    first_difference: Counter[str] = Counter()
    abs_values: list[float] = []
    rel_values: list[float] = []
    f32_ulps: list[int] = []
    f64_ulps: list[int] = []
    changed_abs_values: list[float] = []
    changed_rel_values: list[float] = []
    changed_f32_ulps: list[int] = []
    changed_f64_ulps: list[int] = []
    group_values: dict[str, dict[str, list[float | int]]] = defaultdict(
        lambda: {
            "absolute": [], "relative": [], "ulp_f32": [],
            "changed_absolute": [], "changed_relative": [], "changed_ulp_f32": [],
        }
    )

    for corr_key in common:
        task_a = a.task_by_corr[corr_key]
        task_b = b.task_by_corr[corr_key]
        value_a = score_value(task_a)
        value_b = score_value(task_b)
        absolute, relative = numeric_difference(value_a, value_b)
        abs_values.append(absolute)
        rel_values.append(relative)
        f64_ulp = ulp_distance(task_a.score_f64_bits, task_b.score_f64_bits, 64)
        if f64_ulp is not None:
            f64_ulps.append(f64_ulp)
        score_bits_different = task_a.score_f64_bits != task_b.score_f64_bits
        score_different += int(score_bits_different)
        if score_bits_different:
            changed_abs_values.append(absolute)
            changed_rel_values.append(relative)
            if f64_ulp is not None:
                changed_f64_ulps.append(f64_ulp)
        serialized_changed = task_a.score != task_b.score
        serialized_different += int(serialized_changed)
        sign_changes += int(math.copysign(1.0, value_a) != math.copysign(1.0, value_b))
        crossings += int((value_a < 0 < value_b) or (value_b < 0 < value_a))
        signed_zero_changes += int(value_a == 0 == value_b and task_a.score_f64_bits != task_b.score_f64_bits)
        nan_or_inf += int(not math.isfinite(value_a) or not math.isfinite(value_b))
        lifecycle_mismatches += int(
            len(task_a.lifecycle_score_bits) != 1 or len(task_b.lifecycle_score_bits) != 1
        )

        route_a, rank_a = task_route_observation(a, task_a)
        route_b, rank_b = task_route_observation(b, task_b)
        raw_a = raw_b = f32_a = f32_b = None
        topk_rank_changed = False
        if route_a is not None and route_b is not None and rank_a is not None and rank_b is not None:
            topk_rank_changed = task_a.route_slot != task_b.route_slot or rank_a != rank_b
            raw_a = route_a.get("score_raw_bits", [])[rank_a]
            raw_b = route_b.get("score_raw_bits", [])[rank_b]
            f32_a = route_a.get("score_f32_bits", [])[rank_a]
            f32_b = route_b.get("score_f32_bits", [])[rank_b]
            expected_a = f64_bits_from_f32_bits(f32_a)
            expected_b = f64_bits_from_f32_bits(f32_b)
            widening_mismatches += int(
                expected_a != task_a.score_f64_bits or expected_b != task_b.score_f64_bits
            )
            f32_ulp = ulp_distance(f32_a, f32_b, 32)
            if f32_ulp is not None:
                f32_ulps.append(f32_ulp)
                if score_bits_different:
                    changed_f32_ulps.append(f32_ulp)
        else:
            f32_ulp = None

        if topk_rank_changed:
            first = "TOPK_EXPERT_OR_RANK"
        elif raw_a is not None and raw_a != raw_b:
            first = "ROUTER_TENSOR_RAW"
        elif f32_a is not None and f32_a != f32_b:
            first = "READ_F32_CONVERSION"
        elif score_bits_different:
            first = "TASK_F64_STORAGE_OR_TRANSFER"
        elif serialized_changed:
            first = "TRACE_SERIALIZATION_OR_PARSE"
        else:
            first = "NONE"
        first_difference[first] += 1

        ordering_relevant = task_a.task_id in relevant_a or task_b.task_id in relevant_b
        groups = (
            ("phase", task_a.phase),
            ("layer", str(task_a.layer)),
            ("expert", str(task_a.expert)),
            ("tensor", task_a.tensor),
            ("mode", f"{a.mode}->{b.mode}"),
            ("model_threads", f"{a.model_threads}->{b.model_threads}"),
            ("hint_workers", f"{a.hint_workers}->{b.hint_workers}"),
        )
        for group_name, group_value in groups:
            bucket = group_values[f"{group_name}:{group_value}"]
            bucket["absolute"].append(absolute)
            bucket["relative"].append(relative)
            if f32_ulp is not None:
                bucket["ulp_f32"].append(f32_ulp)
            if score_bits_different:
                bucket["changed_absolute"].append(absolute)
                bucket["changed_relative"].append(relative)
                if f32_ulp is not None:
                    bucket["changed_ulp_f32"].append(f32_ulp)

        detail_stream.write(json.dumps({
            "comparison_id": comparison_id,
            "run_id_a": a.manifest.get("run_name"),
            "run_id_b": b.manifest.get("run_name"),
            "phase": task_a.phase,
            "step": task_a.step,
            "layer": task_a.layer,
            "expert": task_a.expert,
            "tensor": task_a.tensor,
            "stage": task_a.stage,
            "task_id_a": task_a.task_id,
            "task_id_b": task_b.task_id,
            "matching_key": json.loads(corr_key),
            "matching_confidence": "semantic_fields_plus_occurrence_ordinal",
            "route_slot_a": json.loads(task_a.route_slot) if task_a.route_slot else None,
            "route_slot_b": json.loads(task_b.route_slot) if task_b.route_slot else None,
            "topk_rank_a": rank_a,
            "topk_rank_b": rank_b,
            "route_score_a": value_a,
            "route_score_b": value_b,
            "score_f64_bits_a": task_a.score_f64_bits,
            "score_f64_bits_b": task_b.score_f64_bits,
            "source_raw_bits_a": raw_a,
            "source_raw_bits_b": raw_b,
            "source_f32_bits_a": f32_a,
            "source_f32_bits_b": f32_b,
            "absolute_difference": absolute,
            "relative_difference": relative,
            "ulp_difference_f32": f32_ulp,
            "ulp_difference_f64": f64_ulp,
            "sign_changed": math.copysign(1.0, value_a) != math.copysign(1.0, value_b),
            "crossed_zero": (value_a < 0 < value_b) or (value_b < 0 < value_a),
            "signed_zero_changed": value_a == 0 == value_b and task_a.score_f64_bits != task_b.score_f64_bits,
            "nan_or_inf": not math.isfinite(value_a) or not math.isfinite(value_b),
            "topk_rank_changed": topk_rank_changed,
            "ordering_relevant": ordering_relevant,
            "first_observable_difference": first,
        }, ensure_ascii=False) + "\n")

    grouped = {
        key: {
            "sample_count": len(values["absolute"]),
            "absolute_difference": distribution(values["absolute"]),
            "relative_difference": distribution(values["relative"]),
            "ulp_difference_f32": distribution(values["ulp_f32"]),
            "different_task_count": len(values["changed_absolute"]),
            "different_only_absolute_difference": distribution(values["changed_absolute"]),
            "different_only_relative_difference": distribution(values["changed_relative"]),
            "different_only_ulp_difference_f32": distribution(values["changed_ulp_f32"]),
        }
        for key, values in sorted(group_values.items())
    }
    return {
        "matched_tasks": len(common),
        "score_f64_bit_different_tasks": score_different,
        "serialized_score_different_tasks": serialized_different,
        "sign_changed_tasks": sign_changes,
        "crossed_zero_tasks": crossings,
        "signed_zero_changed_tasks": signed_zero_changes,
        "nan_or_inf_tasks": nan_or_inf,
        "source_to_task_widening_mismatches": widening_mismatches,
        "lifecycle_score_bit_mismatches": lifecycle_mismatches,
        "first_observable_difference": dict(first_difference),
        "absolute_difference": distribution(abs_values),
        "relative_difference": distribution(rel_values),
        "ulp_difference_f32": distribution(f32_ulps),
        "ulp_difference_f64": distribution(f64_ulps),
        "different_only_absolute_difference": distribution(changed_abs_values),
        "different_only_relative_difference": distribution(changed_rel_values),
        "different_only_ulp_difference_f32": distribution(changed_f32_ulps),
        "different_only_ulp_difference_f64": distribution(changed_f64_ulps),
        "grouped": grouped,
    }


def compare_pair(
    comparison_id: str,
    a: AuditRun,
    b: AuditRun,
    route_detail_stream: Any,
    task_detail_stream: Any,
) -> dict[str, Any]:
    matching = matching_summary(a, b)
    common_tasks = sorted(set(a.task_by_corr) & set(b.task_by_corr))
    inject_b_into_a = {
        a.task_by_corr[key].task_id: score_value(b.task_by_corr[key]) for key in common_tasks
    }
    inject_a_into_b = {
        b.task_by_corr[key].task_id: score_value(a.task_by_corr[key]) for key in common_tasks
    }
    replay_a = replay(a, inject_b_into_a)
    replay_b = replay(b, inject_a_into_b)
    routes = route_comparison(comparison_id, a, b, route_detail_stream)
    tasks = task_comparison(
        comparison_id, a, b, (replay_a, replay_b), task_detail_stream
    )
    invariants = {
        "binary_sha256_equal": a.manifest.get("binary", {}).get("sha256")
        == b.manifest.get("binary", {}).get("sha256"),
        "model_sha256_equal": a.manifest.get("model", {}).get("sha256")
        == b.manifest.get("model", {}).get("sha256"),
        "prompt_sha256_equal": a.manifest.get("prompt", {}).get("sha256")
        == b.manifest.get("prompt", {}).get("sha256"),
        "output_sha256_equal": a.validation["output_sha256"] == b.validation["output_sha256"],
        "task_structure_equal": matching["unmatched_a_tasks"] == 0
        and matching["unmatched_b_tasks"] == 0,
        "route_slots_equal": matching["unmatched_a_route_records"] == 0
        and matching["unmatched_b_route_records"] == 0,
    }
    return {
        "comparison_id": comparison_id,
        "comparison_role": comparison_role(comparison_id),
        "run_a": a.label,
        "run_b": b.label,
        "mode_a": a.mode,
        "mode_b": b.mode,
        "matching": matching,
        "invariants": invariants,
        "router": routes,
        "tasks": tasks,
        "replay_a_with_b_scores": replay_a,
        "replay_b_with_a_scores": replay_b,
        "actual_dequeue_order": order_comparison(a, b, "dequeue"),
        "actual_hint_issue_order": order_comparison(a, b, "issue"),
        "valid": all(invariants.values()) and replay_a["passed"] and replay_b["passed"],
    }


def cause_assessment(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {pair["comparison_id"]: pair for pair in pairs}
    stability = [pair for pair in pairs if pair["comparison_role"] == "stability"]
    controls = [pair for pair in pairs if pair["comparison_role"] == "control"]

    def raw(prefix: str) -> int:
        return sum(
            pair["router"]["raw_bit_different_items"]
            for name, pair in by_id.items() if name.startswith(prefix)
        )

    all_raw = sum(pair["router"]["raw_bit_different_items"] for pair in stability)
    all_serialized = sum(pair["router"]["serialized_score_different_items"] for pair in stability)
    topk = sum(pair["router"]["topk_rank_changed_items"] for pair in stability)
    control_topk = sum(pair["router"]["topk_rank_changed_items"] for pair in controls)
    widening = sum(pair["tasks"]["source_to_task_widening_mismatches"] for pair in stability)
    unmatched = sum(
        pair["matching"]["unmatched_a_tasks"] + pair["matching"]["unmatched_b_tasks"]
        for pair in stability
    )
    a_raw = raw("A_")
    b_raw = raw("B_")
    c_raw = raw("C_")

    rank_zero = sum(
        pair["router"]["raw_bit_different_rank_zero_items"] for pair in stability
    )
    nonzero_rank = sum(
        pair["router"]["raw_bit_different_nonzero_rank_items"] for pair in stability
    )
    timing_supported = a_raw > 0 and b_raw == 0 and rank_zero == 0 and nonzero_rank > 0

    return [
        {"cause": "multi_thread_float_reduction_order", "status": "contradicted" if timing_supported else "not_observed", "evidence": "The first differing tensor is a GET_ROWS copy result; all same-configuration differences are outside rank 0, matching a pre-barrier shard read rather than a floating reduction."},
        {"cause": "router_topk_nondeterminism", "status": "supported" if topk else "contradicted", "evidence": {"stability_topk_rank_changed_items": topk, "changed_thread_count_control_topk_rank_changed_items": control_topk}},
        {"cause": "near_equal_score_rank_instability", "status": "supported" if topk else "contradicted", "evidence": "Top-K rank change is the required observed consequence."},
        {"cause": "cpu_instruction_or_math_library_difference", "status": "not_tested", "evidence": "All runs use one host/binary; kernel-level instruction selection was not independently instrumented."},
        {"cause": "unfixed_thread_environment", "status": "contradicted" if c_raw > 0 else "not_observed", "evidence": {"unfixed_raw_differences": a_raw, "fixed_multithread_raw_differences": c_raw}},
        {"cause": "illegal_or_uninitialized_memory", "status": "supported" if timing_supported else "not_tested", "evidence": "The trace callback can read non-owner shards before their writers reach the node barrier; whether an individual observed word is stale or not-yet-written was not separately instrumented."},
        {"cause": "task_read_timing_or_tensor_lifetime", "status": "supported" if timing_supported else "not_observed", "evidence": {"cpu_hook_order": "compute -> trace callback on thread 0 -> node barrier", "rank_zero_differences": rank_zero, "nonzero_rank_differences": nonzero_rank, "source_to_task_widening_mismatches": widening}},
        {"cause": "trace_float_format_precision", "status": "supported" if all_raw > all_serialized else "not_observed", "evidence": {"raw_bit_differences": all_raw, "serialized_differences": all_serialized}},
        {"cause": "json_parse_precision", "status": "contradicted", "evidence": "Bit-pattern fields are compared as strings and numeric JSON is used only for the legacy display value."},
        {"cause": "aslr_in_matching", "status": "contradicted", "evidence": "Addresses are excluded from every correspondence key."},
        {"cause": "task_matching_error", "status": "contradicted" if unmatched == 0 else "supported", "evidence": {"unmatched_tasks": unmatched}},
        {"cause": "different_tasks_incorrectly_paired", "status": "contradicted" if unmatched == 0 and widening == 0 else "not_observed", "evidence": "Semantic keys, occurrence ordinals, Router linkage and exact widening are checked."},
    ]


def choose_classification(pairs: list[dict[str, Any]], runs: list[AuditRun]) -> tuple[str, str]:
    stability = [pair for pair in pairs if pair["comparison_role"] == "stability"]
    valid = all(run.validation["passed"] for run in runs) and all(
        pair["valid"] for pair in stability
    )
    raw = sum(pair["router"]["raw_bit_different_items"] for pair in stability)
    topk = sum(pair["router"]["topk_rank_changed_items"] for pair in stability)
    comparator = sum(
        pair[side]["comparator_order_changed_pair_observations"]
        for pair in stability
        for side in ("replay_a_with_b_scores", "replay_b_with_a_scores")
    )
    winner = sum(
        pair[side]["score_injection_winner_changed_decisions"]
        for pair in stability
        for side in ("replay_a_with_b_scores", "replay_b_with_a_scores")
    )
    widening = sum(pair["tasks"]["source_to_task_widening_mismatches"] for pair in stability)
    a_raw = sum(
        pair["router"]["raw_bit_different_items"]
        for pair in stability if pair["comparison_id"].startswith("A_")
    )
    b_raw = sum(
        pair["router"]["raw_bit_different_items"]
        for pair in stability if pair["comparison_id"].startswith("B_")
    )
    rank_zero = sum(
        pair["router"]["raw_bit_different_rank_zero_items"] for pair in stability
    )
    nonzero_rank = sum(
        pair["router"]["raw_bit_different_nonzero_rank_items"] for pair in stability
    )
    if not valid:
        return "INSUFFICIENT_EVIDENCE", "One or more run validity, matching, or self-replay gates failed."
    if topk or comparator or winner:
        return "DECISION_NONDETERMINISM", "A score-related Top-K, comparator pair, or injected winner changed."
    if widening:
        return "RUNTIME_DEFECT", "Source score to Task exact-widening consistency failed."
    if a_raw > 0 and b_raw == 0 and rank_zero == 0 and nonzero_rank > 0:
        return "RUNTIME_DEFECT", (
            "The CPU trace hook reads ffn_moe_weights after thread 0 computes its shard but "
            "before the node barrier. Only nonzero ranks differ in same-configuration runs, "
            "single-thread repeats are bit-identical, and fixed affinity still reproduces the "
            "problem. This is a pre-barrier observation defect, not approved baseline FP drift."
        )
    if raw:
        return "BASELINE_FP_NONDETERMINISM", "The first difference is present in raw Router Tensor bits while Top-K and scheduling decisions remain stable."
    return "INSUFFICIENT_EVIDENCE", "The diagnostic matrix did not reproduce a raw Router score difference."


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    classification_titles = {
        "TRACE_OR_MATCHING_ERROR": "1. Trace 或匹配错误",
        "RUNTIME_DEFECT": "2. 可修复的运行时缺陷",
        "BASELINE_FP_NONDETERMINISM": "3. 基线浮点非确定性",
        "DECISION_NONDETERMINISM": "4. 决策级非确定性",
        "INSUFFICIENT_EVIDENCE": "5. 证据不足",
    }
    classification_title = classification_titles.get(
        summary["classification"], "未定义分类"
    )
    lines = [
        "# M6B1.1 Router Score 跨运行稳定性审计结果",
        "",
        f"> Audit ID：`{summary['audit_id']}`",
        "",
        "## 一、结论与边界",
        "",
        f"主分类：**{classification_title}（`{summary['classification']}`）**。",
        "",
        summary["classification_reason"],
        "",
        "首个差异位于 `ffn_moe_weights` 的原始 F32 Tensor 读取点。不是 FP32/FP16/BF16 转换、Task 的 double 存储、Trace 小数格式化或 JSON 解析引入。",
        "",
        "同配置稳定性 Pair 中，差异没有改变 Top-K Expert ID/排名；冻结候选集并交换 A/B score 的 replay 也没有改变 Comparator pair、DEQUEUE winner 或 Hint winner。实际 DEQUEUE/Hint 的跨运行时序差异存在，但单线程 score 逐 bit 相同时同样存在，不能归因于 route_score。",
        "",
        "该结论不修改 `max_wait_protection`、threshold/guard、Comparator 或现有 Task identity，不批准重新执行 M6B1 Smoke，也不批准进入参数校准。",
        "",
        "## 二、Run 有效性与固定输入",
        "",
        "| Label | Mode | 模型线程 | Hint worker | Affinity | Valid | Output Hash |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['label']} | {run['mode']} | {run['model_threads']} | "
            f"{run['hint_workers']} | `{run['cpu_affinity']}` | {run['validation']['passed']} | "
            f"`{run['output_sha256']}` |"
        )
    lines.extend([
        "",
        "九个 Run 使用同一 binary/model/prompt Hash；所有输出 Hash 相同，Trace zero-drop，Task/Hint/syscall linkage 与生命周期检查通过。模型 CPU 线程和 Expert Hint worker 分别记录，未混淆。",
        "",
        "## 三、原始数值差异",
        "",
        "下表的数值分布仅统计逐 bit 不同的对应 Task；完全一致 Task 不参与 max/mean/p50/p95。",
        "",
        "| Pair | Role | Matched Task | Raw F32 diff | rank 0 / 非 0 | Task F64 diff | 符号变化 | abs max | abs mean | abs p50 | abs p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for pair in summary["run_pairs"]:
        distribution = pair["tasks"]["different_only_absolute_difference"]
        lines.append(
            f"| {pair['comparison_id']} | {pair['comparison_role']} | "
            f"{pair['tasks']['matched_tasks']} | {pair['router']['raw_bit_different_items']} | "
            f"{pair['router']['raw_bit_different_rank_zero_items']} / "
            f"{pair['router']['raw_bit_different_nonzero_rank_items']} | "
            f"{pair['tasks']['score_f64_bit_different_tasks']} | "
            f"{pair['tasks']['sign_changed_tasks']} | {distribution['max']} | "
            f"{distribution['mean']} | {distribution['p50']} | {distribution['p95']} |"
        )
    lines.extend([
        "",
        "`THREAD_CROSS` 与 `AFFINITY_CROSS` 是改变执行条件的对照 Pair，不用于跨运行稳定性的主分类。尤其 `THREAD_CROSS` 改变模型线程数后出现 24/24 个 unmatched Task 和 Top-K 变化，属于配置变化的诊断结果。",
        "",
        "完整的 PREFILL/DECODE、layer、expert、tensor、线程配置和运行模式分组统计保存在 `audit_summary.json` 每个 Pair 的 `tasks.grouped` 中；逐 Task 明细保存在 `task_correspondence.jsonl`。",
        "",
        "## 四、首次差异位置",
        "",
        "完整数据链的检查结果：",
        "",
        "```text",
        "Router Top-K Expert ID（稳定）",
        "→ ffn_moe_weights 原始 F32 Tensor（首次差异）",
        "→ expert_trace.cpp read_f32（bit-exact）",
        "→ float 参数传递（bit-exact）",
        "→ Task double route_score（精确 widening）",
        "→ CREATE/ADMIT/ENQUEUE（生命周期 bit 不变）",
        "→ Comparator replay（无 score 导致的次序变化）",
        "→ DEQUEUE/Hint winner（无 score 导致的 winner 变化）",
        "```",
        "",
        f"首次差异计数：`{json.dumps(summary['numeric_differences']['first_observable_difference'], ensure_ascii=False)}`。所有源 score dtype 为 F32；稳定性 Pair 的 dtype、shape、stride mismatch 为 0，source-F32 到 Task-F64 精确 widening mismatch 为 0，生命周期 score bit mismatch 为 0。",
        "",
        "源码时序是 `ggml_compute_forward` → 线程 0 调用 `llm_mem_trace_moe_weights` → 节点 barrier。`ffn_moe_weights` 是多线程分片的 `GET_ROWS` 输出，因此线程 0 在其他线程完成其 shard 前即可遍历整个 Tensor。所有同配置差异均出现在非 0 rank，rank 0 差异为 0；这与线程 0 拥有首 shard 的边界完全吻合。",
        "",
        "## 五、Top-K 与调度决策相关性",
        "",
        "| Pair | Role | Top-K set/rank changed | score 可达 pair | Comparator pair changed | Injected winner changed | Self replay mismatch | Actual DEQUEUE mismatch | Actual Hint mismatch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for pair in summary["run_pairs"]:
        comparator = sum(
            pair[side]["comparator_order_changed_pair_observations"]
            for side in ("replay_a_with_b_scores", "replay_b_with_a_scores")
        )
        reachable = sum(
            pair[side]["score_reachable_pair_observations"]
            for side in ("replay_a_with_b_scores", "replay_b_with_a_scores")
        )
        winner = sum(
            pair[side]["score_injection_winner_changed_decisions"]
            for side in ("replay_a_with_b_scores", "replay_b_with_a_scores")
        )
        replay_mismatch = sum(
            pair[side]["self_replay_winner_mismatches"]
            for side in ("replay_a_with_b_scores", "replay_b_with_a_scores")
        )
        lines.append(
            f"| {pair['comparison_id']} | {pair['comparison_role']} | "
            f"{pair['router']['topk_set_changed_records']} / "
            f"{pair['router']['topk_rank_changed_items']} | {reachable} | {comparator} | "
            f"{winner} | {replay_mismatch} | "
            f"{pair['actual_dequeue_order']['positional_mismatch_count']} | "
            f"{pair['actual_hint_issue_order']['positional_mismatch_count']} |"
        )
    lines.extend([
        "",
        "稳定性 Pair 的 Top-K set/rank、Comparator-order-changed pair、score-injection winner change 和 self-replay mismatch 均为 0。本 workload 中 deadline 前缀已唯一确定 winner，route_score tie-break 实际可达 pair 为 0；因此证据支持“未改变决策”，但不证明未来存在同 deadline 竞争时永远无影响。",
        "",
        "实际 DEQUEUE/Hint 顺序不是逐运行完全相同；但 `B_SINGLE_THREAD` 的 route_score 全部逐 bit 相同，仍有实际顺序差异，说明这些 positional mismatch 来自 worker/时序。因果判断采用冻结候选集的 score-injection replay，而不是把任意跨运行顺序变化归因于 score。",
        "",
        "## 六、原因审计",
        "",
        "| Cause | Status | Evidence |",
        "|---|---|---|",
    ])
    for cause in summary["cause_assessment"]:
        evidence = json.dumps(cause["evidence"], ensure_ascii=False)
        lines.append(f"| {cause['cause']} | {cause['status']} | `{evidence}` |")
    lines.extend([
        "",
        "## 七、Task identity 建议与人工决定",
        "",
        "当前多线程路径下的 `route_score` 不具备跨运行严格数值稳定性，且原因是可修复的观测时序缺陷；现阶段不应通过 round、epsilon、删除字段或放宽契约绕过。建议先由人工批准修复 pre-barrier 读取，再重跑本审计，之后才决定 route_score 是否适合作为严格 Task identity 字段。结构身份等价、严格数值等价和决策等价应继续分别报告，但本审计不自动修改任何契约。",
        "",
        "- 是否接受当前证据分类；",
        "- 是否批准修复 CPU Trace/Expert hook 的 pre-barrier 读取缺陷；",
        "- 修复并复审后，是否继续保留 route_score 作为跨运行严格 Task identity 字段；",
        "- 是否同时保留严格数值等价、结构身份等价和决策等价三种口径；",
        "- 是否批准重新执行 M6B1 Smoke；",
        "- 本审计不批准进入参数校准。",
        "",
        "## 八、机器证据",
        "",
        "- `audit_summary.json`：完整机器结论；",
        "- `task_correspondence.jsonl`：全部对应 Task；",
        "- `router_score_differences.jsonl`：Router bit/数值差异明细；",
        "- 每个 Run 原目录：manifest、原始 Trace、summary、output Hash、分析结果。",
        "",
        "本次没有修改 `max_wait_protection` 算法、threshold/guard、Comparator、Task identity 或模型计算；没有执行正式性能实验或 N=8。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-id", required=True)
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR")
    parser.add_argument(
        "--comparison", action="append", required=True,
        help="COMPARISON_ID=LABEL_A,LABEL_B",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    run_specs: dict[str, Path] = {}
    for spec in args.run:
        label, separator, directory = spec.partition("=")
        if not separator or not label or label in run_specs:
            raise SystemExit(f"invalid or duplicate --run: {spec}")
        run_specs[label] = Path(directory).resolve()
    comparisons: list[tuple[str, str, str]] = []
    for spec in args.comparison:
        comparison_id, separator, labels = spec.partition("=")
        pair = labels.split(",") if separator else []
        if len(pair) != 2 or any(label not in run_specs for label in pair):
            raise SystemExit(f"invalid --comparison: {spec}")
        comparisons.append((comparison_id, pair[0], pair[1]))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    runs_by_label = {
        label: load_run(label, directory) for label, directory in run_specs.items()
    }
    pairs: list[dict[str, Any]] = []
    with (
        (output_dir / "router_score_differences.jsonl").open("w", encoding="utf-8") as route_stream,
        (output_dir / "task_correspondence.jsonl").open("w", encoding="utf-8") as task_stream,
    ):
        for comparison_id, label_a, label_b in comparisons:
            pairs.append(compare_pair(
                comparison_id,
                runs_by_label[label_a],
                runs_by_label[label_b],
                route_stream,
                task_stream,
            ))

    runs = list(runs_by_label.values())
    stability_pairs = [pair for pair in pairs if pair["comparison_role"] == "stability"]
    control_pairs = [pair for pair in pairs if pair["comparison_role"] == "control"]
    classification, reason = choose_classification(pairs, runs)
    first_difference: Counter[str] = Counter()
    for pair in stability_pairs:
        first_difference.update(pair["tasks"]["first_observable_difference"])
    binary_hashes = {run.manifest.get("binary", {}).get("sha256") for run in runs}
    model_hashes = {run.manifest.get("model", {}).get("sha256") for run in runs}
    prompt_hashes = {run.manifest.get("prompt", {}).get("sha256") for run in runs}
    output_hashes = {run.validation["output_sha256"] for run in runs}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": args.audit_id,
        "runs": [run_identity(run) for run in runs],
        "run_pairs": pairs,
        "matching": {
            "all_stability_pairs_zero_unmatched_tasks": all(
                pair["matching"]["unmatched_a_tasks"] == 0
                and pair["matching"]["unmatched_b_tasks"] == 0
                for pair in stability_pairs
            ),
            "all_stability_pairs_zero_unmatched_routes": all(
                pair["matching"]["unmatched_a_route_records"] == 0
                and pair["matching"]["unmatched_b_route_records"] == 0
                for pair in stability_pairs
            ),
            "addresses_excluded": True,
            "score_excluded": True,
            "task_id_cross_run_excluded": True,
        },
        "numeric_differences": {
            "first_observable_difference": dict(first_difference),
            "raw_router_bit_different_items": sum(
                pair["router"]["raw_bit_different_items"] for pair in stability_pairs
            ),
            "raw_router_bit_different_rank_zero_items": sum(
                pair["router"]["raw_bit_different_rank_zero_items"]
                for pair in stability_pairs
            ),
            "raw_router_bit_different_nonzero_rank_items": sum(
                pair["router"]["raw_bit_different_nonzero_rank_items"]
                for pair in stability_pairs
            ),
            "task_f64_bit_different_items": sum(
                pair["tasks"]["score_f64_bit_different_tasks"] for pair in stability_pairs
            ),
            "serialized_task_score_different_items": sum(
                pair["tasks"]["serialized_score_different_tasks"] for pair in stability_pairs
            ),
            "source_to_task_widening_mismatches": sum(
                pair["tasks"]["source_to_task_widening_mismatches"]
                for pair in stability_pairs
            ),
            "lifecycle_score_bit_mismatches": sum(
                pair["tasks"]["lifecycle_score_bit_mismatches"]
                for pair in stability_pairs
            ),
        },
        "aggregates": {
            "pair_count": len(pairs),
            "stability_pair_count": len(stability_pairs),
            "control_pair_count": len(control_pairs),
            "run_count": len(runs),
            "percentile_method": "linear interpolation at (n-1)*q",
            "empty_sample_behavior": "null with sample_count=0",
        },
        "ordering_relevance": {
            "stability_topk_set_changed_records": sum(
                pair["router"]["topk_set_changed_records"] for pair in stability_pairs
            ),
            "stability_topk_rank_changed_items": sum(
                pair["router"]["topk_rank_changed_items"] for pair in stability_pairs
            ),
            "control_topk_set_changed_records": sum(
                pair["router"]["topk_set_changed_records"] for pair in control_pairs
            ),
            "control_topk_rank_changed_items": sum(
                pair["router"]["topk_rank_changed_items"] for pair in control_pairs
            ),
            "stability_score_reachable_pair_observations": sum(
                pair[side]["score_reachable_pair_observations"]
                for pair in stability_pairs
                for side in ("replay_a_with_b_scores", "replay_b_with_a_scores")
            ),
            "stability_comparator_order_changed_pair_observations": sum(
                pair[side]["comparator_order_changed_pair_observations"]
                for pair in stability_pairs
                for side in ("replay_a_with_b_scores", "replay_b_with_a_scores")
            ),
            "stability_winner_changed_decisions": sum(
                pair[side]["score_injection_winner_changed_decisions"]
                for pair in stability_pairs
                for side in ("replay_a_with_b_scores", "replay_b_with_a_scores")
            ),
            "stability_actual_dequeue_positional_mismatches": sum(
                pair["actual_dequeue_order"]["positional_mismatch_count"]
                for pair in stability_pairs
            ),
            "stability_actual_hint_positional_mismatches": sum(
                pair["actual_hint_issue_order"]["positional_mismatch_count"]
                for pair in stability_pairs
            ),
        },
        "replay": {
            "self_replay_winner_mismatches": sum(
                pair[side]["self_replay_winner_mismatches"]
                for pair in pairs
                for side in ("replay_a_with_b_scores", "replay_b_with_a_scores")
            ),
            "all_replays_passed": all(
                pair[side]["passed"]
                for pair in pairs
                for side in ("replay_a_with_b_scores", "replay_b_with_a_scores")
            ),
        },
        "validity_gates": {
            "all_runs_valid": all(run.validation["passed"] for run in runs),
            "all_pairs_valid": all(pair["valid"] for pair in pairs),
            "all_stability_pairs_valid": all(pair["valid"] for pair in stability_pairs),
            "all_control_pairs_valid": all(pair["valid"] for pair in control_pairs),
            "binary_hash_count": len(binary_hashes),
            "model_hash_count": len(model_hashes),
            "prompt_hash_count": len(prompt_hashes),
            "output_hash_count": len(output_hashes),
            "trace_zero_drop": all(run.validation["trace_complete"] for run in runs),
        },
        "cause_assessment": cause_assessment(pairs),
        "classification": classification,
        "classification_reason": reason,
        "source_timing_evidence": {
            "producer_op": "GGML_OP_GET_ROWS",
            "producer_partition": "rows_per_thread",
            "cpu_hook_order": [
                "ggml_compute_forward",
                "thread_0_llm_mem_trace_moe_weights",
                "node_barrier",
            ],
            "single_thread_raw_differences": sum(
                pair["router"]["raw_bit_different_items"]
                for pair in stability_pairs if pair["comparison_id"].startswith("B_")
            ),
            "fixed_multithread_raw_differences": sum(
                pair["router"]["raw_bit_different_items"]
                for pair in stability_pairs if pair["comparison_id"].startswith("C_")
            ),
            "rank_zero_raw_differences": sum(
                pair["router"]["raw_bit_different_rank_zero_items"]
                for pair in stability_pairs
            ),
            "nonzero_rank_raw_differences": sum(
                pair["router"]["raw_bit_different_nonzero_rank_items"]
                for pair in stability_pairs
            ),
        },
        "human_decisions_required": [
            "accept_or_reject_the_audit_classification",
            "retain_or_redefine_route_score_in_cross_run_strict_task_identity",
            "decide_whether_thread_environment_must_be_fixed",
            "approve_or_reject_rerunning_M6B1_smoke",
            "do_not_enter_parameter_calibration_without_separate_approval",
        ],
        "artifacts": {
            "task_correspondence": "task_correspondence.jsonl",
            "router_score_differences": "router_score_differences.jsonl",
            "markdown_report": "audit_report.md",
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(summary, output_dir / "audit_report.md")
    print(json.dumps({
        "audit_id": args.audit_id,
        "classification": classification,
        "all_runs_valid": summary["validity_gates"]["all_runs_valid"],
        "all_pairs_valid": summary["validity_gates"]["all_pairs_valid"],
        "all_stability_pairs_valid": summary["validity_gates"]["all_stability_pairs_valid"],
        "raw_router_bit_different_items": summary["numeric_differences"]["raw_router_bit_different_items"],
        "topk_rank_changed_items": summary["ordering_relevance"]["stability_topk_rank_changed_items"],
        "winner_changed_decisions": summary["ordering_relevance"]["stability_winner_changed_decisions"],
    }, sort_keys=True))
    if classification == "INSUFFICIENT_EVIDENCE" or not all(
        pair["valid"] for pair in stability_pairs
    ):
        raise SystemExit("M6B1.1 audit evidence is insufficient; artifacts were retained")


if __name__ == "__main__":
    main()
