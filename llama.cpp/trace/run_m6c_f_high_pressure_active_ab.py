#!/usr/bin/env python3
"""Execute M6C-F high-pressure Reserved-Service Active A/B.

Only generation length and prefetch worker count vary. The runtime policy is
the already-frozen S1-D representative; no parameter search is implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import run_m6c_active_ab as base


ROOT = base.ROOT
LLAMA_ROOT = base.LLAMA_ROOT
TRACE_ROOT = base.TRACE_ROOT
PIPELINE = base.PIPELINE
BINARY = base.BINARY
MODEL = base.MODEL
OUTPUT_ROOT = TRACE_ROOT / "m6c_f_high_pressure_active_ab_20260805_v1"
RAW_ROOT = OUTPUT_ROOT / "raw_runs"
SMOKE_ROOT = OUTPUT_ROOT / "smoke"
MEMORY_MAX = base.MEMORY_MAX
SWAP_MAX = base.SWAP_MAX
ORDER_SEED = "m6c_f_20260805_high_pressure_fixed_s1d"
EXPECTED_MODEL_SHA256 = base.EXPECTED_MODEL_SHA256
EXPECTED_HELPER_SHA256 = base.EXPECTED_HELPER_SHA256
EXPECTED_PROMPT_SHA256 = base.EXPECTED_PROMPT_SHA256

SCENARIOS: dict[str, dict[str, int | str]] = {
    "P1": {"generated_tokens": 64, "workers": 1, "meaning": "medium_length_service_limited"},
    "P2": {"generated_tokens": 256, "workers": 1, "meaning": "long_decode_strong_queue_pressure"},
    "P3": {"generated_tokens": 256, "workers": 2, "meaning": "long_decode_normal_prefetch_concurrency"},
}

BASE_ORDER = ((1, "B0"), (1, "B1"), (2, "B1"), (2, "B0"), (3, "B0"), (3, "B1"))
CONFIRM_ORDER = ((4, "B1"), (4, "B0"), (5, "B0"), (5, "B1"))
LIFECYCLE_BITS = {"CREATE": 1, "ADMIT": 2, "ENQUEUE": 4, "DEQUEUE": 8, "ISSUE": 16}
ALL_LIFECYCLE_BITS = sum(LIFECYCLE_BITS.values())


GateError = base.GateError


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    return count


def file_record(path: Path, root: Path, parse: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": base.sha256_file(path),
    }
    if parse and path.suffix == ".json":
        base.read_json(path)
        record["parse_status"] = "JSON_OK"
    elif parse and path.suffix == ".jsonl":
        lines = sum(1 for _ in base.event_records(path))
        record["parse_status"] = "JSONL_OK"
        record["line_count"] = lines
    return record


def common_environment(
        run_id: str,
        scenario: str,
        configuration: str,
        repeat: int,
        order: int,
        output_base: Path) -> dict[str, str]:
    settings = SCENARIOS[scenario]
    workers = int(settings["workers"])
    env = base.common_environment(run_id, workers, configuration, repeat, order)
    env.update({
        "TRACE_BASE_DIR": str(output_base),
        "TRACE_OUT_DIR": str(output_base / run_id),
        "NUM_TOKENS_PREDICT": str(settings["generated_tokens"]),
        "ORDER_MODE": "scenario_blocked_interleaved_preregistered",
        "ORDER_SEED": ORDER_SEED,
        "LLM_MEM_TRACE_AUDIT_CASE": run_id,
        "LLM_MEM_TRACE_AUDIT_SLOT_ID": f"{scenario}_r{repeat}_{configuration}",
        "LLM_MEM_TRACE_AUDIT_SCENARIO": scenario,
        "LLM_MEM_TRACE_AUDIT_GENERATED_TOKENS": str(settings["generated_tokens"]),
    })
    return env


def build_plan() -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    order = 1
    for scenario in ("P1", "P2", "P3"):
        settings = SCENARIOS[scenario]
        for repeat, configuration in BASE_ORDER:
            run_id = f"m6c_f_20260805_v1_p{order:02d}_{scenario}_r{repeat}_{configuration}"
            plan.append({
                "order": order,
                "run_id": run_id,
                "scenario": scenario,
                "configuration": configuration,
                "repeat": repeat,
                "workers": settings["workers"],
                "generated_tokens": settings["generated_tokens"],
                "scope_unit": f"m6c-f-v1-p{order:02d}-{scenario.lower()}-r{repeat}-{configuration.lower()}",
            })
            order += 1
    return plan


def build_confirmation_plan(scenario: str, start_order: int) -> list[dict[str, Any]]:
    settings = SCENARIOS[scenario]
    result: list[dict[str, Any]] = []
    order = start_order
    for repeat, configuration in CONFIRM_ORDER:
        run_id = f"m6c_f_20260805_v1_p{order:02d}_{scenario}_r{repeat}_{configuration}_confirm"
        result.append({
            "order": order,
            "run_id": run_id,
            "scenario": scenario,
            "configuration": configuration,
            "repeat": repeat,
            "workers": settings["workers"],
            "generated_tokens": settings["generated_tokens"],
            "scope_unit": f"m6c-f-v1-p{order:02d}-{scenario.lower()}-r{repeat}-{configuration.lower()}-confirm",
            "confirmation": True,
        })
        order += 1
    return result


def task_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["task_id"],
        record["step"],
        record["layer"],
        record["expert"],
        record["phase"],
        record["stage"],
        record["tensor"],
        record["nbytes"],
        record["route_score_f64_bits"],
        record["sequence"],
    )


def wait_distribution(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    tail_count = max(1, math.ceil(len(ordered) * 0.01)) if ordered else 0
    return {
        **base.distribution_ns(values),
        "worst_1pct_mean_ns": statistics.fmean(ordered[-tail_count:]) if tail_count else None,
        "ge_25ms_count": sum(value >= 25_000_000 for value in values),
        "ge_50ms_count": sum(value >= 50_000_000 for value in values),
        "ge_100ms_count": sum(value >= 100_000_000 for value in values),
    }


def validate_and_measure_run(
        run_dir: Path,
        scenario: str,
        configuration: str) -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    settings = SCENARIOS[scenario]
    workers = int(settings["workers"])
    required = (
        "memory_trace.jsonl", "expert_trace.jsonl", "summary.json",
        "process_metrics.json", "run_manifest.json", "output.sha256",
        "inference_output.txt", "cache_preparation.json", "analysis/metrics.json",
        "m6c_f_scope.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise GateError(f"{run_dir.name}: missing artifacts {missing}")
    trace_summary = base.read_json(run_dir / "summary.json")
    trace_drop = sum(int(sink.get("dropped", 0)) for sink in trace_summary["sinks"].values())
    if trace_drop != 0:
        raise GateError(f"{run_dir.name}: Trace drop is nonzero")
    process = base.read_json(run_dir / "process_metrics.json")
    if int(process.get("exit_code", -1)) != 0:
        raise GateError(f"{run_dir.name}: inference exit code is nonzero")
    output_sha = base.read_text(run_dir / "output.sha256")
    if output_sha != base.sha256_file(run_dir / "inference_output.txt"):
        raise GateError(f"{run_dir.name}: output artifact Hash mismatch")
    manifest = base.read_json(run_dir / "run_manifest.json")
    environment = manifest["environment"]
    expected_env = {
        "LLM_MEM_TRACE_OPT_EXPERT_RESERVED_SERVICE_ACTIVE": "1" if configuration == "B1" else "0",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE": "deadline_score",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS": str(workers),
        "NUM_TOKENS_PREDICT": str(settings["generated_tokens"]),
        "NUM_THREADS": "8", "BATCH_SIZE": "512", "CTX_SIZE": "2048",
        "GPU_LAYERS": "0", "TEMP": "0.0", "SEED": "1234",
    }
    mismatches = {name: (environment.get(name), value) for name, value in expected_env.items() if environment.get(name) != value}
    if mismatches:
        raise GateError(f"{run_dir.name}: frozen environment mismatch {mismatches}")
    if manifest["host"].get("cpu_affinity") != list(range(8)):
        raise GateError(f"{run_dir.name}: CPU affinity is not 0-7")

    lifecycle_mask: dict[int, int] = {}
    lifecycle_duplicate_count = 0
    tasks: dict[int, dict[str, Any]] = {}
    queue_waits: list[int] = []
    hint_service: list[int] = []
    step_latency: dict[str, list[int]] = {"PREFILL": [], "DECODE": []}
    async_summaries: list[dict[str, Any]] = []
    reserved_summaries: list[dict[str, Any]] = []
    reserved_selections: list[dict[str, Any]] = []
    queue_summaries: list[dict[str, Any]] = []
    reject_cancel = 0
    for record in base.event_records(run_dir / "memory_trace.jsonl"):
        event = record.get("event")
        if event == "EXPERT_TASK":
            task_id = int(record.get("task_id", 0))
            lifecycle = str(record.get("lifecycle_event", ""))
            if lifecycle in LIFECYCLE_BITS:
                bit = LIFECYCLE_BITS[lifecycle]
                before = lifecycle_mask.get(task_id, 0)
                if before & bit:
                    lifecycle_duplicate_count += 1
                lifecycle_mask[task_id] = before | bit
            if lifecycle == "DEQUEUE":
                wait_ns = int(record["queue_wait_ns"])
                queue_waits.append(wait_ns)
                tasks[task_id] = {
                    "task_id": task_id,
                    "step": int(record["step"]),
                    "layer": int(record["layer"]),
                    "expert": int(record["expert"]),
                    "phase": str(record["phase"]),
                    "stage": str(record["stage"]),
                    "tensor": str(record["tensor"]),
                    "nbytes": int(record["nbytes"]),
                    "route_score_f64_bits": str(record["score_f64_bits"]),
                    "sequence": int(record["sequence"]),
                    "deadline_ts_ns": int(record["deadline_ts_ns"]),
                    "enqueued_ts_ns": int(record["enqueued_ts_ns"]),
                    "dequeued_ts_ns": int(record["dequeued_ts_ns"]),
                    "queue_wait_ns": wait_ns,
                }
            elif lifecycle == "ISSUE" and record.get("returned_ts_ns") is not None:
                hint_service.append(int(record["returned_ts_ns"]) - int(record["issued_ts_ns"]))
            elif lifecycle in {"REJECT", "CANCEL"}:
                reject_cancel += 1
        elif event == "STEP_END" and record.get("phase") in step_latency:
            step_latency[str(record["phase"])].append(int(record["latency_ns"]))
        elif event == "EXPERT_ASYNC_SUMMARY":
            async_summaries.append(record)
        elif event == "EXPERT_RESERVED_SERVICE_SUMMARY":
            reserved_summaries.append(record)
        elif event == "EXPERT_RESERVED_SERVICE_SELECTION":
            reserved_selections.append(record)
        elif event == "EXPERT_QUEUE_OVERHEAD_SUMMARY":
            queue_summaries.append(record)

    bad_lifecycle = sum(mask != ALL_LIFECYCLE_BITS for mask in lifecycle_mask.values())
    if not tasks or bad_lifecycle or lifecycle_duplicate_count or reject_cancel:
        raise GateError(
            f"{run_dir.name}: lifecycle Gate failed tasks={len(tasks)} bad={bad_lifecycle} "
            f"duplicate={lifecycle_duplicate_count} reject_cancel={reject_cancel}"
        )
    if set(tasks) != set(lifecycle_mask):
        raise GateError(f"{run_dir.name}: Task conservation mismatch")
    if len(async_summaries) != 1 or len(queue_summaries) != 1:
        raise GateError(f"{run_dir.name}: async/queue summary count mismatch")
    async_summary = async_summaries[0]
    queue_summary = queue_summaries[0]
    if int(async_summary.get("final_queue_depth", -1)) != 0 or int(async_summary.get("final_queued_bytes", -1)) != 0:
        raise GateError(f"{run_dir.name}: final queue is not empty")
    if bool(async_summary.get("reserved_service_active")) != (configuration == "B1"):
        raise GateError(f"{run_dir.name}: feature flag runtime state mismatch")

    default_reserved = {
        "reserved_trigger_count": 0, "reserved_due_count": 0,
        "reserved_selected_count": 0, "active_winner_changed_count": 0,
        "reserved_same_as_legacy_head_count": 0, "hard_urgent_override_count": 0,
        "hard_urgent_safety_violation": 0, "stale_handle_count": 0,
        "duplicate_erase_count": 0, "generation_mismatch_count": 0,
        "full_store_scan_count": 0, "invariant_error_count": 0,
        "store_index_registry_bytes_conserved": None, "final_queue_empty": True,
    }
    winner_changes: list[dict[str, Any]] = []
    divergence_count = 0
    if configuration == "B1":
        if len(reserved_summaries) != 1:
            raise GateError(f"{run_dir.name}: Reserved summary count mismatch")
        reserved = reserved_summaries[0]
        frozen = {
            "reserved_numerator": 1, "reserved_denominator": 8,
            "eligibility_age_ns": 41000000, "hard_urgent_guard_ns": 0,
            "eligibility_rule": "AGE_GATED_ALL", "reserved_winner": "oldest_eligible",
            "debt_policy": "single_pending_latch", "reset_policy": "reset_when_no_eligible",
        }
        if any(reserved.get(name) != value for name, value in frozen.items()):
            raise GateError(f"{run_dir.name}: frozen Active configuration mismatch")
        zero_fields = (
            "hard_urgent_safety_violation", "stale_handle_count", "duplicate_erase_count",
            "generation_mismatch_count", "full_store_scan_count", "invariant_error_count",
        )
        if any(int(reserved.get(name, -1)) != 0 for name in zero_fields):
            raise GateError(f"{run_dir.name}: Active safety/invariant failure")
        if not reserved.get("store_index_registry_bytes_conserved") or not reserved.get("final_queue_empty"):
            raise GateError(f"{run_dir.name}: Active conservation/final-empty failure")
        selection_count = int(reserved.get("selection_count", -1))
        if selection_count != len(reserved_selections) or selection_count != len(tasks):
            raise GateError(f"{run_dir.name}: Active selection count mismatch")
        reserved_selections.sort(key=lambda record: int(record["decision_id"]))
        if [int(record["decision_id"]) for record in reserved_selections] != list(range(selection_count)):
            raise GateError(f"{run_dir.name}: Active decision IDs are not dense and unique")
        slot_by_task: dict[int, int] = {}
        for record in reserved_selections:
            selected_id = int(record["selected_task_id"])
            if selected_id in slot_by_task:
                raise GateError(f"{run_dir.name}: Task selected more than once")
            slot_by_task[selected_id] = int(record["decision_id"])
            legacy = record["legacy_head"]
            aging = record["aging_head"]
            if int(legacy["task_id"]) != int(aging["task_id"]):
                divergence_count += 1
            if record.get("active_winner_changed_vs_legacy"):
                selected = record["selected"]
                decision_ts = int(record["decision_ts_ns"])
                winner_changes.append({
                    "decision_id": int(record["decision_id"]),
                    "worker_id": int(record["worker_id"]),
                    "selected_source": record["winner_source"],
                    "s1_winner": selected,
                    "counterfactual_s0_legacy_winner": legacy,
                    "s1_current_wait_ns": decision_ts - int(selected["enqueued_ts_ns"]),
                    "legacy_current_wait_ns": decision_ts - int(legacy["enqueued_ts_ns"]),
                })
        for change in winner_changes:
            selected_id = int(change["s1_winner"]["task_id"])
            legacy_id = int(change["counterfactual_s0_legacy_winner"]["task_id"])
            if selected_id not in slot_by_task or legacy_id not in slot_by_task:
                raise GateError(f"{run_dir.name}: changed winner final slot unavailable")
            change["s1_winner_final_selection_slot"] = slot_by_task[selected_id]
            change["legacy_winner_final_selection_slot"] = slot_by_task[legacy_id]
            change["s1_winner_final_queue_wait_ns"] = tasks[selected_id]["queue_wait_ns"]
            change["legacy_winner_final_queue_wait_ns"] = tasks[legacy_id]["queue_wait_ns"]
        changed = sum(bool(record.get("active_winner_changed_vs_legacy")) for record in reserved_selections)
        same = sum(bool(record.get("reserved_same_as_legacy_head")) for record in reserved_selections)
        if changed != int(reserved["active_winner_changed_count"]) or same != int(reserved["reserved_same_as_legacy_head_count"]):
            raise GateError(f"{run_dir.name}: Active summary/Detail winner mismatch")
    else:
        if reserved_summaries or reserved_selections:
            raise GateError(f"{run_dir.name}: feature-off emitted Reserved events")
        reserved = default_reserved

    scope = base.read_json(run_dir / "m6c_f_scope.json")
    before, after = scope["before"], scope["after"]
    if before["path"] != after["path"] or before["inode"] != after["inode"]:
        raise GateError(f"{run_dir.name}: cgroup changed during Run")
    if before["memory.max"] != MEMORY_MAX or before["memory.swap.max"] != SWAP_MAX:
        raise GateError(f"{run_dir.name}: cgroup limits mismatch")
    events_delta = scope["delta"]["memory.events"]
    if any(int(events_delta.get(name, 0)) for name in ("oom", "oom_kill", "oom_group_kill")):
        raise GateError(f"{run_dir.name}: OOM event observed")

    metrics = base.read_json(run_dir / "analysis" / "metrics.json")
    queue_global = queue_summary["global"]
    generated_tokens_actual = min(
        int(settings["generated_tokens"]), len(step_latency["DECODE"]) + 1
    )
    reserved_names = (
        "reserved_trigger_count", "reserved_due_count", "reserved_selected_count",
        "active_winner_changed_count", "reserved_same_as_legacy_head_count",
        "hard_urgent_override_count", "hard_urgent_safety_violation",
        "stale_handle_count", "duplicate_erase_count", "generation_mismatch_count",
        "full_store_scan_count", "invariant_error_count",
        "store_index_registry_bytes_conserved", "final_queue_empty",
        "insert_count", "erase_count", "selection_count",
        "legacy_heap_sift_count", "aging_heap_sift_count",
        "enqueue_index_op_mean_ns", "enqueue_index_op_max_ns",
        "dequeue_index_op_mean_ns", "dequeue_index_op_max_ns",
    )
    result = {
        "schema_version": "m6c-f-run-v1",
        "run_id": run_dir.name,
        "scenario": scenario,
        "configuration": configuration,
        "workers": workers,
        "generated_tokens_requested": int(settings["generated_tokens"]),
        "generated_tokens_actual": generated_tokens_actual,
        "generated_token_semantics": "trace_decode_step_count_plus_terminal_sample",
        "valid": True,
        "output_sha256": output_sha,
        "task_count": len(tasks),
        "trace_drop_count": trace_drop,
        "wall_time_s": float(process["wall_time_s"]),
        "prefill_latency_total_ms": sum(step_latency["PREFILL"]) / 1e6,
        "prefill_latency_mean_ms": statistics.fmean(step_latency["PREFILL"]) / 1e6,
        "decode_latency_total_ms": sum(step_latency["DECODE"]) / 1e6,
        "decode_latency_mean_ms": statistics.fmean(step_latency["DECODE"]) / 1e6,
        "decode_throughput_tokens_per_s": float(metrics["decode_throughput_tokens_per_s"]),
        "major_faults": int(process["major_faults"]),
        "minor_faults": int(process["minor_faults"]),
        "rss_peak_kb": int(process["max_rss_kb"]),
        "cgroup_memory_peak_bytes": int(after["memory.peak"]),
        "swap_peak_bytes": int(after["memory.swap.peak"]),
        "psi_some_total_delta_us": int(scope["delta"]["memory.pressure.total"].get("some", 0)),
        "psi_full_total_delta_us": int(scope["delta"]["memory.pressure.total"].get("full", 0)),
        "queue_wait": wait_distribution(queue_waits),
        "hint_service": base.distribution_ns(hint_service),
        "max_queue_depth": int(async_summary["max_queue_depth"]),
        "legacy_aging_head_divergence_count": divergence_count,
        "legacy_aging_head_divergence_rate": divergence_count / len(tasks),
        "active_winner_changed_rate": int(reserved.get("active_winner_changed_count", 0)) / len(tasks),
        "enqueue_queue_op": {
            "mean_ns": int(async_summary["enqueue_queue_op_mean_ns"]),
            "max_ns": int(async_summary["enqueue_queue_op_max_ns"]),
        },
        "lock_acquire_wait": queue_global["mutex_acquire_wait_ns"],
        "lock_hold": queue_global["mutex_hold_ns"],
        "queue_selection": queue_global["queue_scan_ns"],
        "scanned_candidates": queue_global["queue_scan_candidates"],
        "reserved": {name: reserved.get(name) for name in reserved_names},
        "winner_change_detail_count": len(winner_changes),
        "correctness": {
            "lifecycle_duplicate_count": lifecycle_duplicate_count,
            "lifecycle_incomplete_count": bad_lifecycle,
            "reject_cancel_count": reject_cancel,
            "trace_drop_count": trace_drop,
            "final_queue_empty": True,
            "cgroup_effective": True,
        },
        "cgroup": {
            "path": before["path"], "inode": before["inode"],
            "memory.max": before["memory.max"], "memory.swap.max": before["memory.swap.max"],
            "memory.events_delta": events_delta,
        },
    }
    return result, tasks, winner_changes


def finalize_run_artifacts(
        run_dir: Path,
        result: dict[str, Any],
        tasks: dict[int, dict[str, Any]],
        winner_changes: list[dict[str, Any]]) -> dict[str, Any]:
    task_path = run_dir / "m6c_f_task_waits.jsonl"
    change_path = run_dir / "m6c_f_winner_changes.jsonl"
    task_lines = write_jsonl(task_path, (tasks[task_id] for task_id in sorted(tasks)))
    change_lines = write_jsonl(change_path, winner_changes)
    if task_lines != result["task_count"] or change_lines != result["winner_change_detail_count"]:
        raise GateError(f"{run_dir.name}: derived JSONL line count mismatch")
    # Reopen after close before recording authority metadata.
    result["derived_artifacts"] = {
        "task_waits": file_record(task_path, run_dir),
        "winner_changes": file_record(change_path, run_dir),
    }
    result_path = run_dir / "m6c_f_run_result.json"
    base.write_json(result_path, result)
    reopened = base.read_json(result_path)
    if reopened != result:
        raise GateError(f"{run_dir.name}: closed run result failed deterministic reopen")
    return result


def read_task_waits(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in base.event_records(path):
        task_id = int(record["task_id"])
        if task_id in result:
            raise GateError(f"duplicate task_id in {path}")
        result[task_id] = record
    return result


def analyze_pair(b0: dict[str, Any], b1: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    if b0["scenario"] != b1["scenario"]:
        raise GateError("cross-scenario pair is invalid")
    b0_tasks = read_task_waits(raw_root / b0["run_id"] / "m6c_f_task_waits.jsonl")
    b1_tasks = read_task_waits(raw_root / b1["run_id"] / "m6c_f_task_waits.jsonl")
    if set(b0_tasks) != set(b1_tasks):
        raise GateError(f"pair {b0['run_id']} / {b1['run_id']}: Task ID set mismatch")
    deltas: dict[int, int] = {}
    identity_mismatch = 0
    for task_id in b0_tasks:
        if task_identity(b0_tasks[task_id]) != task_identity(b1_tasks[task_id]):
            identity_mismatch += 1
        deltas[task_id] = int(b1_tasks[task_id]["queue_wait_ns"]) - int(b0_tasks[task_id]["queue_wait_ns"])
    if identity_mismatch:
        raise GateError(f"pair {b0['run_id']} / {b1['run_id']}: {identity_mismatch} immutable Task mismatches")
    winner_impacts: list[dict[str, Any]] = []
    for change in base.event_records(raw_root / b1["run_id"] / "m6c_f_winner_changes.jsonl"):
        selected_id = int(change["s1_winner"]["task_id"])
        legacy_id = int(change["counterfactual_s0_legacy_winner"]["task_id"])
        change = dict(change)
        change["s1_winner_B1_minus_B0_wait_ns"] = deltas[selected_id]
        change["legacy_winner_B1_minus_B0_wait_ns"] = deltas[legacy_id]
        winner_impacts.append(change)
    delta_values = list(deltas.values())
    physical_fields = (
        "wall_time_s", "prefill_latency_total_ms", "decode_latency_total_ms",
        "decode_latency_mean_ms", "decode_throughput_tokens_per_s",
        "major_faults", "minor_faults", "rss_peak_kb",
        "cgroup_memory_peak_bytes", "swap_peak_bytes", "psi_some_total_delta_us",
        "psi_full_total_delta_us",
    )
    distribution_fields = ("mean", "p95_bucket_upper_bound", "p99_bucket_upper_bound", "max")
    implementation_fields = (
        "lock_acquire_wait", "lock_hold", "queue_selection", "scanned_candidates",
    )
    relative_physical = {
        name: ((b1[name] - b0[name]) / b0[name]) if b0[name] else None
        for name in physical_fields
    }
    return {
        "schema_version": "m6c-f-pair-v1",
        "scenario": b0["scenario"],
        "repeat": int(b0["run_id"].split("_r", 1)[1].split("_", 1)[0]),
        "B0_run_id": b0["run_id"],
        "B1_run_id": b1["run_id"],
        "output_comparable": b0["output_sha256"] == b1["output_sha256"],
        "generated_tokens_comparable": b0["generated_tokens_actual"] == b1["generated_tokens_actual"],
        "physical_B1_minus_B0": {name: b1[name] - b0[name] for name in physical_fields},
        "physical_relative_change": relative_physical,
        "hint_service_B1_minus_B0": {
            name: b1["hint_service"][name] - b0["hint_service"][name]
            for name in ("p50_ns", "p95_ns", "p99_ns", "max_ns")
        },
        "implementation_B1_minus_B0": {
            section: {
                name: b1[section][name] - b0[section][name]
                for name in distribution_fields
            }
            for section in implementation_fields
        },
        "implementation_relative_change": {
            section: {
                name: ((b1[section][name] - b0[section][name]) / b0[section][name])
                if b0[section][name] else None
                for name in distribution_fields
            }
            for section in implementation_fields
        },
        "queue_wait_B1_minus_B0": {
            name: b1["queue_wait"][name] - b0["queue_wait"][name]
            for name in ("p50_ns", "p95_ns", "p99_ns", "max_ns", "worst_1pct_mean_ns")
        },
        "threshold_count_B1_minus_B0": {
            name: b1["queue_wait"][name] - b0["queue_wait"][name]
            for name in ("ge_25ms_count", "ge_50ms_count", "ge_100ms_count")
        },
        "task_wait_delta": {
            "task_count": len(delta_values),
            "improved_task_count": sum(value < 0 for value in delta_values),
            "worsened_task_count": sum(value > 0 for value in delta_values),
            "unchanged_task_count": sum(value == 0 for value in delta_values),
            "nonzero_task_rate": sum(value != 0 for value in delta_values) / len(delta_values),
            "max_single_task_improvement_ns": max((-value for value in delta_values if value < 0), default=0),
            "max_single_task_worsening_ns": max((value for value in delta_values if value > 0), default=0),
            "sum_wait_delta_ns": sum(delta_values),
        },
        "winner_change_count": b1["reserved"]["active_winner_changed_count"],
        "winner_change_task_impacts": winner_impacts,
    }


def scenario_analysis(
        scenario: str,
        runs: list[dict[str, Any]],
        pairs: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_runs = [run for run in runs if run["scenario"] == scenario]
    b1_runs = [run for run in scenario_runs if run["configuration"] == "B1"]
    scenario_pairs = [pair for pair in pairs if pair["scenario"] == scenario]
    majority = len(scenario_pairs) // 2 + 1
    median_pair: dict[str, Any] = {}
    paths = {
        "wall_time_s": lambda pair: pair["physical_B1_minus_B0"]["wall_time_s"],
        "decode_total_ms": lambda pair: pair["physical_B1_minus_B0"]["decode_latency_total_ms"],
        "decode_mean_ms": lambda pair: pair["physical_B1_minus_B0"]["decode_latency_mean_ms"],
        "major_faults": lambda pair: pair["physical_B1_minus_B0"]["major_faults"],
        "rss_peak_kb": lambda pair: pair["physical_B1_minus_B0"]["rss_peak_kb"],
        "swap_peak_bytes": lambda pair: pair["physical_B1_minus_B0"]["swap_peak_bytes"],
        "psi_full_us": lambda pair: pair["physical_B1_minus_B0"]["psi_full_total_delta_us"],
        "queue_wait_p95_ns": lambda pair: pair["queue_wait_B1_minus_B0"]["p95_ns"],
        "queue_wait_p99_ns": lambda pair: pair["queue_wait_B1_minus_B0"]["p99_ns"],
        "queue_wait_max_ns": lambda pair: pair["queue_wait_B1_minus_B0"]["max_ns"],
        "worst_1pct_mean_ns": lambda pair: pair["queue_wait_B1_minus_B0"]["worst_1pct_mean_ns"],
        "sum_wait_delta_ns": lambda pair: pair["task_wait_delta"]["sum_wait_delta_ns"],
    }
    for name, getter in paths.items():
        median_pair[name] = statistics.median(getter(pair) for pair in scenario_pairs)
    tail_direction_count = sum(
        pair["queue_wait_B1_minus_B0"]["p99_ns"] < 0 and
        pair["queue_wait_B1_minus_B0"]["worst_1pct_mean_ns"] < 0
        for pair in scenario_pairs
    )
    physical_improvement_by_metric = {
        "wall_time_s": sum(pair["physical_B1_minus_B0"]["wall_time_s"] < 0 for pair in scenario_pairs),
        "decode_latency_total_ms": sum(
            pair["physical_B1_minus_B0"]["decode_latency_total_ms"] < 0 for pair in scenario_pairs
        ),
        "decode_throughput_tokens_per_s": sum(
            pair["physical_B1_minus_B0"]["decode_throughput_tokens_per_s"] > 0
            for pair in scenario_pairs
        ),
        "major_faults": sum(pair["physical_B1_minus_B0"]["major_faults"] < 0 for pair in scenario_pairs),
    }
    physical_direction_count = max(physical_improvement_by_metric.values())
    winner_change_run_count = sum(
        int(run["reserved"]["active_winner_changed_count"]) > 0 for run in b1_runs
    )
    return {
        "scenario": scenario,
        "settings": SCENARIOS[scenario],
        "n_per_configuration": len(b1_runs),
        "majority_pair_count": majority,
        "B1_winner_change_counts": [int(run["reserved"]["active_winner_changed_count"]) for run in b1_runs],
        "B1_winner_change_rates": [run["active_winner_changed_rate"] for run in b1_runs],
        "B1_legacy_aging_divergence_counts": [run["legacy_aging_head_divergence_count"] for run in b1_runs],
        "B1_reserved_selected_counts": [int(run["reserved"]["reserved_selected_count"]) for run in b1_runs],
        "all_B1_no_winner_change": all(int(run["reserved"]["active_winner_changed_count"]) == 0 for run in b1_runs),
        "stable_nonzero_winner_change": all(int(run["reserved"]["active_winner_changed_count"]) > 0 for run in b1_runs),
        "winner_change_run_count": winner_change_run_count,
        "winner_change_in_majority_of_runs": winner_change_run_count >= majority,
        "tail_improvement_pair_count": tail_direction_count,
        "physical_improvement_pair_count": physical_direction_count,
        "physical_improvement_pair_count_by_metric": physical_improvement_by_metric,
        "median_B1_minus_B0": median_pair,
        "pair_results": scenario_pairs,
    }


def confirmation_choice(analyses: list[dict[str, Any]]) -> tuple[str | None, str]:
    eligible = [
        analysis for analysis in analyses
        if analysis["stable_nonzero_winner_change"] and
        (analysis["tail_improvement_pair_count"] >= analysis["majority_pair_count"] or
         analysis["physical_improvement_pair_count"] >= analysis["majority_pair_count"])
    ]
    if not eligible:
        return None, "no scenario has stable nonzero winner change plus a majority-direction mechanism/physical signal"
    eligible.sort(key=lambda analysis: (
        -statistics.median(analysis["B1_winner_change_rates"]),
        analysis["scenario"],
    ))
    return eligible[0]["scenario"], "deterministic highest median winner-change rate among eligible scenarios"


def final_decision(analyses: list[dict[str, Any]], runs: list[dict[str, Any]]) -> tuple[str, bool, list[str]]:
    b1_runs = [run for run in runs if run["configuration"] == "B1"]
    safety_fields = (
        "hard_urgent_safety_violation", "stale_handle_count", "duplicate_erase_count",
        "generation_mismatch_count", "full_store_scan_count", "invariant_error_count",
    )
    if any(int(run["reserved"].get(name) or 0) for run in b1_runs for name in safety_fields):
        return "HIGH_PRESSURE_RESERVED_ROUTE_REGRESSION", False, ["safety or invariant counter is nonzero"]

    run_by_id = {run["run_id"]: run for run in runs}

    def relative_values(analysis: dict[str, Any], section: str, metric: str) -> list[float]:
        values = [pair[section][metric] for pair in analysis["pair_results"]]
        return [float(value) for value in values if value is not None]

    regression_reasons: list[str] = []
    for analysis in analyses:
        majority = analysis["majority_pair_count"]
        # "Clear" is preregistered here as the same metric worsening in a
        # majority of pairs with a >=10% median relative change. Histogram
        # implementation overhead uses a stricter 50% threshold.
        for metric in (
                "wall_time_s", "decode_latency_total_ms", "major_faults", "rss_peak_kb",
                "cgroup_memory_peak_bytes", "swap_peak_bytes", "psi_some_total_delta_us",
                "psi_full_total_delta_us"):
            values = relative_values(analysis, "physical_relative_change", metric)
            if values and sum(value > 0 for value in values) >= majority and statistics.median(values) >= 0.10:
                regression_reasons.append(f"{analysis['scenario']} {metric} has majority >=10% median regression")
        for metric in ("p99_ns", "worst_1pct_mean_ns"):
            relatives = []
            for pair in analysis["pair_results"]:
                b0 = run_by_id[pair["B0_run_id"]]["queue_wait"][metric]
                delta = pair["queue_wait_B1_minus_B0"][metric]
                if b0:
                    relatives.append(delta / b0)
            if relatives and sum(value > 0 for value in relatives) >= majority and statistics.median(relatives) >= 0.10:
                regression_reasons.append(f"{analysis['scenario']} queue {metric} has majority >=10% median regression")
        for section in ("lock_acquire_wait", "lock_hold", "queue_selection"):
            values = [
                pair["implementation_relative_change"][section]["p99_bucket_upper_bound"]
                for pair in analysis["pair_results"]
            ]
            values = [float(value) for value in values if value is not None]
            if values and sum(value > 0 for value in values) >= majority and statistics.median(values) >= 0.50:
                regression_reasons.append(f"{analysis['scenario']} {section} p99 bucket has majority >=50% regression")
    if regression_reasons:
        return "HIGH_PRESSURE_RESERVED_ROUTE_REGRESSION", False, regression_reasons
    if all(analysis["all_B1_no_winner_change"] for analysis in analyses):
        return "HIGH_PRESSURE_RESERVED_ROUTE_STILL_NO_OP", False, [
            "all B1 Runs in P1/P2/P3 have active_winner_changed_count=0"
        ]
    fairness = any(
        analysis["winner_change_in_majority_of_runs"] and
        analysis["tail_improvement_pair_count"] >= analysis["majority_pair_count"]
        for analysis in analyses
    )
    physical = any(
        analysis["winner_change_in_majority_of_runs"] and
        analysis["physical_improvement_pair_count"] >= analysis["majority_pair_count"]
        for analysis in analyses
    )
    strong = False
    for analysis in analyses:
        pairs = analysis["pair_results"]
        majority = analysis["majority_pair_count"]
        for metric in ("wall_time_s", "decode_latency_total_ms", "major_faults"):
            improvements = []
            for pair in pairs:
                b0_id = pair["B0_run_id"]
                b0 = run_by_id[b0_id]
                delta = pair["physical_B1_minus_B0"][metric]
                improvements.append((-delta / b0[metric]) if b0[metric] else 0.0)
            if sum(value > 0 for value in improvements) >= majority and statistics.median(improvements) >= 0.10:
                strong = True
        throughput = [
            pair["physical_relative_change"]["decode_throughput_tokens_per_s"]
            for pair in pairs
        ]
        throughput = [float(value) for value in throughput if value is not None]
        if throughput and sum(value > 0 for value in throughput) >= majority and statistics.median(throughput) >= 0.10:
            strong = True
        for metric in ("p99_ns", "worst_1pct_mean_ns"):
            improvements = []
            for pair in pairs:
                b0 = run_by_id[pair["B0_run_id"]]["queue_wait"][metric]
                delta = pair["queue_wait_B1_minus_B0"][metric]
                improvements.append((-delta / b0) if b0 else 0.0)
            if sum(value > 0 for value in improvements) >= majority and statistics.median(improvements) >= 0.10:
                strong = True
    if fairness and physical:
        return "HIGH_PRESSURE_RESERVED_ROUTE_HAS_PROMISING_SIGNAL", strong, ["winner change, fairness and physical directions agree in a majority of pairs"]
    if fairness:
        return "FAIRNESS_SIGNAL_WITHOUT_PHYSICAL_PERFORMANCE_SIGNAL", strong, ["winner change and fairness signal exist without stable physical direction"]
    return "REORDERING_OCCURRED_WITHOUT_FAIRNESS_BENEFIT", strong, ["winner changes occur without stable queue-tail benefit"]


def scope_child(args: argparse.Namespace) -> int:
    output_base = Path(args.output_base).resolve()
    run_dir = output_base / args.run_id
    before = base.cgroup_snapshot()
    if before["memory.max"] != MEMORY_MAX or before["memory.swap.max"] != SWAP_MAX:
        raise GateError("effective delegated cgroup limits are not 7 GiB / 2 GiB")
    env = os.environ.copy()
    env.update(common_environment(
        args.run_id, args.scenario, args.configuration, args.repeat, args.order, output_base
    ))
    started = base.utc_now()
    completed = subprocess.run(
        ["bash", str(PIPELINE)], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    after = base.cgroup_snapshot()
    run_dir.mkdir(parents=True, exist_ok=True)
    base.write_text(run_dir / "pipeline.log", completed.stdout)
    base.write_json(run_dir / "m6c_f_scope.json", {
        "schema_version": "m6c-f-scope-v1",
        "run_id": args.run_id,
        "started_at_utc": started,
        "ended_at_utc": base.utc_now(),
        "pipeline_exit_code": completed.returncode,
        "before": before,
        "after": after,
        "delta": base.snapshot_delta(before, after),
    })
    if completed.returncode:
        raise GateError(f"pipeline failed with exit code {completed.returncode}")
    return 0


def run_in_scope(output_base: Path, entry: dict[str, Any]) -> None:
    command = [
        "systemd-run", "--user", "--scope", "--collect", "--quiet",
        f"--unit={entry['scope_unit']}",
        "-p", f"MemoryMax={MEMORY_MAX}", "-p", f"MemorySwapMax={SWAP_MAX}",
        "taskset", "-c", "0-7", sys.executable, str(Path(__file__).resolve()),
        "--scope-child", "--output-base", str(output_base),
        "--run-id", entry["run_id"], "--scenario", entry["scenario"],
        "--configuration", entry["configuration"], "--repeat", str(entry["repeat"]),
        "--order", str(entry["order"]),
    ]
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base.write_text(OUTPUT_ROOT / f"{entry['run_id']}_launcher.log", completed.stdout)
    if completed.returncode:
        raise GateError(f"{entry['run_id']}: delegated scope launcher failed ({completed.returncode})")


def execute_entry(output_base: Path, entry: dict[str, Any]) -> dict[str, Any]:
    print(f"START {entry['order']:02d} {entry['scenario']} {entry['configuration']} r{entry['repeat']}", flush=True)
    run_in_scope(output_base, entry)
    run_dir = output_base / entry["run_id"]
    result, tasks, changes = validate_and_measure_run(
        run_dir, entry["scenario"], entry["configuration"]
    )
    finalize_run_artifacts(run_dir, result, tasks, changes)
    # Independent deterministic reopen/parse outside the delegated scope.
    second, second_tasks, second_changes = validate_and_measure_run(
        run_dir, entry["scenario"], entry["configuration"]
    )
    if second != {key: value for key, value in result.items() if key != "derived_artifacts"}:
        raise GateError(f"{entry['run_id']}: deterministic raw Trace reparse mismatch")
    if second_tasks != tasks or second_changes != changes:
        raise GateError(f"{entry['run_id']}: deterministic Task/winner reparse mismatch")
    print(
        f"PASS  {entry['order']:02d} tasks={result['task_count']} "
        f"head_div={result['legacy_aging_head_divergence_count']} "
        f"winner_change={result['reserved']['active_winner_changed_count']}",
        flush=True,
    )
    return result


def make_pairs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (run["scenario"], int(run["run_id"].split("_r", 1)[1].split("_", 1)[0]), run["configuration"]): run
        for run in runs
    }
    result: list[dict[str, Any]] = []
    for scenario in ("P1", "P2", "P3"):
        repeats = sorted({repeat for (name, repeat, _configuration) in indexed if name == scenario})
        for repeat in repeats:
            b0 = indexed[(scenario, repeat, "B0")]
            b1 = indexed[(scenario, repeat, "B1")]
            pair = analyze_pair(b0, b1, RAW_ROOT)
            if not pair["output_comparable"] or not pair["generated_tokens_comparable"]:
                raise GateError(f"{scenario} repeat {repeat}: B0/B1 output or token count is not comparable")
            result.append(pair)
    return result


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M6C-F High-Pressure Reserved-Service Active A/B Report", "",
        "> 物理系统已重新执行；本实验不是参数搜索，也不是正式 N=8。", "",
        f"最终枚举：`{report['final_enum']}`", "",
        f"追加 N=5：`{report['confirmation']['executed']}`；{report['confirmation']['reason']}", "",
        "## Raw Runs", "",
        "| Run | Scenario | cfg | tasks | actual tokens | wall s | decode total ms | majflt | RSS MiB | swap MiB | qwait p95/p99/max ms | head divergence | winner change |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for run in report["raw_runs"]:
        wait = run["queue_wait"]
        lines.append(
            f"| {run['run_id']} | {run['scenario']} | {run['configuration']} | {run['task_count']} | "
            f"{run['generated_tokens_actual']} | {run['wall_time_s']:.2f} | {run['decode_latency_total_ms']:.1f} | "
            f"{run['major_faults']} | {run['rss_peak_kb']/1024:.1f} | {run['swap_peak_bytes']/1048576:.2f} | "
            f"{wait['p95_ns']/1e6:.3f}/{wait['p99_ns']/1e6:.3f}/{wait['max_ns']/1e6:.3f} | "
            f"{run['legacy_aging_head_divergence_count']} | {run['reserved']['active_winner_changed_count']} |"
        )
    lines.extend(["", "## Scenario Conclusions", ""])
    for analysis in report["scenario_analysis"]:
        lines.append(
            f"- {analysis['scenario']}: winner changes={analysis['B1_winner_change_counts']}, "
            f"head divergence={analysis['B1_legacy_aging_divergence_counts']}, "
            f"tail-majority={analysis['tail_improvement_pair_count']}/{len(analysis['pair_results'])}, "
            f"physical-majority={analysis['physical_improvement_pair_count']}/{len(analysis['pair_results'])}."
        )
    lines.extend(["", "## Safety", "", f"```json\n{json.dumps(report['safety'], indent=2, sort_keys=True)}\n```", ""])
    return "\n".join(lines)


def build_artifact_index(root: Path) -> dict[str, Any]:
    excluded = {"artifact_index.json", "artifact_validation.json", "authority.json", "authority.sha256"}
    files = [path for path in sorted(root.rglob("*")) if path.is_file() and path.name not in excluded]
    records = []
    for index, path in enumerate(files, 1):
        parse = path.suffix in {".json", ".jsonl"}
        records.append(file_record(path, root, parse=parse))
        if index % 25 == 0:
            print(f"INDEX {index}/{len(files)}", flush=True)
    return {"schema_version": "m6c-f-artifact-index-v1", "file_count": len(records), "files": records}


def validate_artifacts(root: Path) -> int:
    index = base.read_json(root / "artifact_index.json")
    errors: list[str] = []
    for number, record in enumerate(index["files"], 1):
        path = root / record["path"]
        if not path.is_file():
            errors.append(f"missing {record['path']}")
            continue
        if path.stat().st_size != record["size_bytes"]:
            errors.append(f"size mismatch {record['path']}")
        if base.sha256_file(path) != record["sha256"]:
            errors.append(f"SHA mismatch {record['path']}")
        if record.get("parse_status") == "JSON_OK":
            try:
                base.read_json(path)
            except Exception as exc:  # noqa: BLE001 - validator must capture all parse failures
                errors.append(f"JSON parse failure {record['path']}: {exc}")
        elif record.get("parse_status") == "JSONL_OK":
            try:
                count = sum(1 for _ in base.event_records(path))
                if count != record["line_count"]:
                    errors.append(f"line count mismatch {record['path']}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"JSONL parse failure {record['path']}: {exc}")
        if number % 25 == 0:
            print(f"VALIDATE {number}/{index['file_count']}", flush=True)
    validation = {
        "schema_version": "m6c-f-artifact-validation-v1",
        "validated_at_utc": base.utc_now(),
        "file_count": index["file_count"],
        "passed": not errors,
        "errors": errors,
    }
    base.write_json(root / "artifact_validation.json", validation)
    return 0 if not errors else 2


def execute(root: Path) -> int:
    if root.exists():
        raise GateError(f"output directory already exists: {root}")
    if base.sha256_file(MODEL) != EXPECTED_MODEL_SHA256:
        raise GateError("model SHA mismatch")
    helper = LLAMA_ROOT / "trace" / "prepare_model_cache.py"
    if base.sha256_file(helper) != EXPECTED_HELPER_SHA256:
        raise GateError("cold-cache helper SHA mismatch")
    reference = base.read_json(base.REFERENCE_MANIFEST)
    if reference["frozen"]["prompt_sha256"] != EXPECTED_PROMPT_SHA256:
        raise GateError("A.3 prompt authority mismatch")
    root.mkdir(parents=True)
    RAW_ROOT.mkdir()
    SMOKE_ROOT.mkdir()
    plan = build_plan()
    smoke = {
        "order": 0, "run_id": "m6c_f_20260805_v1_smoke_P2_B1",
        "scenario": "P2", "configuration": "B1", "repeat": 0,
        "workers": 1, "generated_tokens": 256,
        "scope_unit": "m6c-f-v1-smoke-p2-b1",
    }
    protocol = {
        "schema_version": "m6c-f-protocol-v1",
        "created_at_utc": base.utc_now(),
        "stage": "M6C-F High-Pressure Reserved-Service Active A/B",
        "formal_n8": False,
        "parameter_search": False,
        "binary_path": str(BINARY),
        "binary_sha256": base.sha256_file(BINARY),
        "runner_sha256": base.sha256_file(Path(__file__).resolve()),
        "pipeline_sha256": base.sha256_file(PIPELINE),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "prompt_sha256": EXPECTED_PROMPT_SHA256,
        "cgroup": {"memory.max": MEMORY_MAX, "memory.swap.max": SWAP_MAX, "new_scope_per_run": True},
        "frozen_common": {
            "context_size": 2048, "model_threads": 8, "gpu_layers": 0,
            "model_batch_size": 512, "temperature": 0.0, "seed": 1234,
            "cpu_affinity": "0-7", "cache": "cold", "trace": "A.3 custom detail",
        },
        "B0": {"priority_mode": "deadline_score", "reserved_active": False},
        "B1": {
            "priority_mode": "deadline_score", "reserved_active": True,
            "R": 1, "D": 8, "eligibility_age_ns": 41000000,
            "hard_urgent_guard_ns": 0, "eligibility_rule": "AGE_GATED_ALL",
            "reserved_winner": "oldest_eligible", "debt_policy": "single_pending_latch",
            "reset_policy": "reset_when_no_eligible",
        },
        "scenarios": SCENARIOS,
        "smoke": smoke,
        "matrix": plan,
        "confirmation_rule": {
            "eligible": "all three B1 winner-change counts nonzero and at least 2/3 pairs have tail or physical directional improvement",
            "maximum_scenarios": 1,
            "selection": "highest median winner-change rate; scenario ID lexical tie-break",
            "additional_runs": list(CONFIRM_ORDER),
        },
        "classification_rules": {
            "majority": "floor(N/2)+1; recomputed as 3 if the one allowed scenario is extended to N=5",
            "tail_improvement_pair": "both queue-wait p99 and worst-1-percent mean B1-B0 are negative",
            "physical_improvement": "a single named metric (wall, decode total, throughput, or Major Fault) improves in a majority of pairs",
            "stable_winner_change_for_confirmation": "active_winner_changed_count is nonzero in every N=3 B1 Run",
            "winner_change_for_final_signal": "active_winner_changed_count is nonzero in a majority of B1 Runs",
            "clear_physical_or_resource_regression": "same metric worsens in a majority of pairs and median relative change is at least 10 percent",
            "clear_queue_tail_regression": "p99 or worst-1-percent mean worsens in a majority of pairs and median relative change is at least 10 percent",
            "clear_implementation_regression": "lock acquire, lock hold, or queue-selection p99 histogram upper bound worsens in a majority of pairs and median relative change is at least 50 percent",
            "strong_effect": "same queue-tail or important physical metric improves in a majority of pairs with median relative improvement at least 10 percent",
        },
    }
    base.write_json(root / "protocol.json", protocol)
    print("SMOKE P2 B1", flush=True)
    run_in_scope(SMOKE_ROOT, smoke)
    smoke_result, smoke_tasks, smoke_changes = validate_and_measure_run(
        SMOKE_ROOT / smoke["run_id"], "P2", "B1"
    )
    finalize_run_artifacts(SMOKE_ROOT / smoke["run_id"], smoke_result, smoke_tasks, smoke_changes)
    base.write_json(root / "smoke_result.json", smoke_result)
    print(
        f"SMOKE PASS tasks={smoke_result['task_count']} "
        f"winner_change={smoke_result['reserved']['active_winner_changed_count']}", flush=True
    )

    runs: list[dict[str, Any]] = []
    for entry in plan:
        runs.append(execute_entry(RAW_ROOT, entry))
    pairs = make_pairs(runs)
    analyses = [scenario_analysis(scenario, runs, pairs) for scenario in ("P1", "P2", "P3")]
    chosen, choice_reason = confirmation_choice(analyses)
    confirmation_plan: list[dict[str, Any]] = []
    if chosen is not None:
        confirmation_plan = build_confirmation_plan(chosen, len(plan) + 1)
        print(f"CONFIRM {chosen} with four additional Runs", flush=True)
        for entry in confirmation_plan:
            runs.append(execute_entry(RAW_ROOT, entry))
        pairs = make_pairs(runs)
        analyses = [scenario_analysis(scenario, runs, pairs) for scenario in ("P1", "P2", "P3")]

    base.write_json(root / "paired_results.json", {
        "schema_version": "m6c-f-paired-results-v1", "pairs": pairs
    })
    base.write_json(root / "mechanism_analysis.json", {
        "schema_version": "m6c-f-mechanism-analysis-v1", "scenarios": analyses,
        "confirmation": {"executed": chosen is not None, "scenario": chosen, "reason": choice_reason},
    })
    final_enum, strong, reasons = final_decision(analyses, runs)
    safety = {
        "all_runs_valid": all(run["valid"] for run in runs),
        "trace_drop_total": sum(run["trace_drop_count"] for run in runs),
        "hard_urgent_safety_violation_total": sum(int(run["reserved"].get("hard_urgent_safety_violation") or 0) for run in runs),
        "stale_handle_total": sum(int(run["reserved"].get("stale_handle_count") or 0) for run in runs),
        "full_store_scan_total": sum(int(run["reserved"].get("full_store_scan_count") or 0) for run in runs),
        "invariant_error_total": sum(int(run["reserved"].get("invariant_error_count") or 0) for run in runs),
        "all_final_queues_empty": all(run["correctness"]["final_queue_empty"] for run in runs),
    }
    report = {
        "schema_version": "m6c-f-final-report-v1",
        "created_at_utc": base.utc_now(),
        "physical_system_reexecuted": True,
        "performance_claim": False,
        "formal_n8": False,
        "binary_sha256": protocol["binary_sha256"],
        "raw_run_count": len(runs),
        "raw_runs": runs,
        "paired_results": pairs,
        "scenario_analysis": analyses,
        "confirmation": {"executed": chosen is not None, "scenario": chosen, "reason": choice_reason},
        "safety": safety,
        "strong_effect_ge_10_percent": strong,
        "final_enum": final_enum,
        "decision_reasons": reasons,
    }
    base.write_json(root / "final_report.json", report)
    base.write_text(root / "final_report.md", report_markdown(report))
    index = build_artifact_index(root)
    base.write_json(root / "artifact_index.json", index)
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--validate-artifacts", "--output-root", str(root)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(completed.stdout, end="", flush=True)
    if completed.returncode:
        raise GateError("independent artifact validator failed")
    validation = base.read_json(root / "artifact_validation.json")
    authority = {
        "schema_version": "m6c-f-authority-v1",
        "final_enum": final_enum,
        "strong_effect_ge_10_percent": strong,
        "binary_sha256": protocol["binary_sha256"],
        "report_sha256": base.sha256_file(root / "final_report.json"),
        "artifact_index_sha256": base.sha256_file(root / "artifact_index.json"),
        "artifact_validation_sha256": base.sha256_file(root / "artifact_validation.json"),
        "artifact_validation_passed": validation["passed"],
    }
    base.write_json(root / "authority.json", authority)
    base.write_text(root / "authority.sha256", base.sha256_file(root / "authority.json") + "\n")
    print(json.dumps(authority, indent=2, sort_keys=True), flush=True)
    print(f"authority_sha256={base.read_text(root / 'authority.sha256')}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--scope-child", action="store_true")
    parser.add_argument("--validate-artifacts", action="store_true")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--output-base")
    parser.add_argument("--run-id")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS))
    parser.add_argument("--configuration", choices=("B0", "B1"))
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--order", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.scope_child:
            if not args.output_base or not args.run_id or not args.scenario or not args.configuration:
                raise GateError("scope-child arguments are incomplete")
            return scope_child(args)
        if args.validate_artifacts:
            return validate_artifacts(Path(args.output_root).resolve())
        if args.execute:
            return execute(Path(args.output_root).resolve())
        parser.error("choose --execute, --scope-child, or --validate-artifacts")
    except GateError as exc:
        print(f"M6C-F Gate failure: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
