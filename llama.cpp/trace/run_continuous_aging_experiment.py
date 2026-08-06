#!/usr/bin/env python3
"""Continuous Aging offline replay and compact physical A/B runner."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import statistics
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import run_m6c_active_ab as base


ROOT = base.ROOT
LLAMA_ROOT = base.LLAMA_ROOT
TRACE_ROOT = base.TRACE_ROOT
PIPELINE = base.PIPELINE
BINARY = base.BINARY
MODEL = base.MODEL
SOURCE_ROOT = TRACE_ROOT / "m6c_f_high_pressure_active_ab_20260805_v1" / "raw_runs"
OUTPUT_ROOT = TRACE_ROOT / "continuous_aging_active_ab_20260805_v1"
RAW_ROOT = OUTPUT_ROOT / "raw_runs"
MEMORY_MAX = base.MEMORY_MAX
SWAP_MAX = base.SWAP_MAX
ALPHA_PER_NS = 2.2268593311309814e-11
MEDIAN_ROUTE_SCORE_GAP = 0.0011134296655654907
COMPENSATION_NS = 50_000_000
ORDER_SEED = "continuous_aging_20260805_single_alpha"

SCENARIOS: dict[str, dict[str, int]] = {
    "P1": {"generated_tokens": 64, "workers": 1},
    "P2": {"generated_tokens": 256, "workers": 1},
    "P3": {"generated_tokens": 256, "workers": 2},
}
ORDER = ((1, "B0"), (1, "B1"), (2, "B1"), (2, "B0"), (3, "B0"), (3, "B1"))
LIFECYCLE_BITS = {"CREATE": 1, "ADMIT": 2, "ENQUEUE": 4, "DEQUEUE": 8, "ISSUE": 16}
ALL_LIFECYCLE_BITS = sum(LIFECYCLE_BITS.values())

GateError = base.GateError


def decode_f64(bits: str) -> float:
    if not isinstance(bits, str) or len(bits) != 18 or not bits.startswith("0x"):
        raise GateError("route score is not an exact F64 bit pattern")
    value = struct.unpack(">d", int(bits, 16).to_bytes(8, "big"))[0]
    if not math.isfinite(value):
        raise GateError("route score is non-finite")
    return value


def quantile(values: list[int | float], probability: float) -> float | None:
    return base.quantile(values, probability)


def distribution(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50_ns": quantile(values, 0.50),
        "p95_ns": quantile(values, 0.95),
        "p99_ns": quantile(values, 0.99),
        "max_ns": max(values) if values else None,
    }


def stage_rank(stage: str) -> int:
    return {"EARLY": 0, "LATE": 1}.get(stage, 2)


def continuous_key(task: dict[str, Any], alpha: float, epoch_ns: int) -> tuple[Any, ...]:
    static_score = task["score"] - alpha * (task["enqueued_ts_ns"] - epoch_ns)
    return (
        task["step"], task["layer"], stage_rank(task["stage"]),
        -static_score, task["sequence"], task["task_id"],
    )


def legacy_key(task: dict[str, Any]) -> tuple[Any, ...]:
    deadline = task["deadline_ts_ns"]
    return (
        deadline == 0, deadline, -task["score"], task["sequence"], task["task_id"],
    )


def parse_source_run(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: dict[int, dict[str, Any]] = {}
    slots: list[dict[str, Any]] = []
    summary = base.read_json(run_dir / "summary.json")
    if sum(int(item.get("dropped", 0)) for item in summary["sinks"].values()) != 0:
        raise GateError(f"{run_dir.name}: source Trace drop is nonzero")
    for record in base.event_records(run_dir / "memory_trace.jsonl"):
        event = record.get("event")
        if event == "EXPERT_TASK" and record.get("lifecycle_event") == "ENQUEUE":
            task_id = int(record["task_id"])
            if task_id in tasks:
                raise GateError(f"{run_dir.name}: duplicate source Task")
            bits = str(record["score_f64_bits"]).lower()
            tasks[task_id] = {
                "task_id": task_id,
                "step": int(record["step"]),
                "layer": int(record["layer"]),
                "stage": str(record["stage"]),
                "phase": str(record["phase"]),
                "deadline_ts_ns": int(record["deadline_ts_ns"]),
                "route_score_f64_bits": bits,
                "score": decode_f64(bits),
                "sequence": int(record["sequence"]),
                "enqueued_ts_ns": int(record["enqueued_ts_ns"]),
                "nbytes": int(record["nbytes"]),
            }
        elif event == "EXPERT_QUEUE_OVERHEAD_SELECTION":
            slots.append({
                "decision_id": int(record["decision_id"]),
                "decision_ts_ns": int(record["batch_decision_ts_ns"]),
                "winner_task_id": int(record["winner_task_id"]),
                "queue_depth_before": int(record["queue_depth_before"]),
            })
    slots.sort(key=lambda item: item["decision_id"])
    if [item["decision_id"] for item in slots] != list(range(len(slots))):
        raise GateError(f"{run_dir.name}: source decision IDs are not dense")
    arrivals = sorted(tasks.values(), key=lambda item: (
        item["enqueued_ts_ns"], item["sequence"], item["task_id"]
    ))
    if len(arrivals) != len(slots) or not arrivals:
        raise GateError(f"{run_dir.name}: source Task/slot conservation failed")
    return arrivals, slots


def replay_source_run(run_dir: Path) -> dict[str, Any]:
    arrivals, slots = parse_source_run(run_dir)
    epoch_ns = arrivals[0]["enqueued_ts_ns"]
    legacy_heap: list[tuple[tuple[Any, ...], int]] = []
    grouped_heap: list[tuple[tuple[Any, ...], int]] = []
    active_heap: list[tuple[tuple[Any, ...], int]] = []
    by_id = {task["task_id"]: task for task in arrivals}
    arrival_index = 0
    baseline_wait: dict[int, int] = {}
    grouped_wait: dict[int, int] = {}
    active_wait: dict[int, int] = {}
    winner_change_vs_b0 = 0
    grouping_only_change_vs_b0 = 0
    aging_change_vs_grouped = 0
    baseline_mismatch = 0
    hard_urgent_bypass = 0
    baseline_deadline_miss = 0
    active_deadline_miss = 0
    baseline_deadline_tasks = 0
    active_deadline_tasks = 0

    for slot in slots:
        now = slot["decision_ts_ns"]
        while arrival_index < len(arrivals) and arrivals[arrival_index]["enqueued_ts_ns"] <= now:
            task = arrivals[arrival_index]
            heapq.heappush(legacy_heap, (legacy_key(task), task["task_id"]))
            heapq.heappush(grouped_heap, (continuous_key(task, 0.0, epoch_ns), task["task_id"]))
            heapq.heappush(active_heap, (continuous_key(task, ALPHA_PER_NS, epoch_ns), task["task_id"]))
            arrival_index += 1
        expected_depth = slot["queue_depth_before"]
        if len(legacy_heap) != expected_depth or len(grouped_heap) != expected_depth or len(active_heap) != expected_depth:
            raise GateError(f"{run_dir.name}: replay queue depth mismatch")
        _, legacy_id = heapq.heappop(legacy_heap)
        _, grouped_id = heapq.heappop(grouped_heap)
        _, active_id = heapq.heappop(active_heap)
        baseline_mismatch += legacy_id != slot["winner_task_id"]
        winner_change_vs_b0 += active_id != legacy_id
        grouping_only_change_vs_b0 += grouped_id != legacy_id
        aging_change_vs_grouped += active_id != grouped_id
        legacy_task = by_id[legacy_id]
        grouped_task = by_id[grouped_id]
        active_task = by_id[active_id]
        baseline_wait[legacy_id] = now - legacy_task["enqueued_ts_ns"]
        grouped_wait[grouped_id] = now - grouped_task["enqueued_ts_ns"]
        active_wait[active_id] = now - active_task["enqueued_ts_ns"]
        legacy_deadline = legacy_task["deadline_ts_ns"]
        active_deadline = active_task["deadline_ts_ns"]
        baseline_deadline_tasks += legacy_deadline != 0
        active_deadline_tasks += active_deadline != 0
        baseline_deadline_miss += legacy_deadline != 0 and now >= legacy_deadline
        active_deadline_miss += active_deadline != 0 and now >= active_deadline
        hard_urgent_bypass += (
            active_id != legacy_id and legacy_deadline != 0 and now >= legacy_deadline
        )

    if baseline_mismatch or arrival_index != len(arrivals) or legacy_heap or grouped_heap or active_heap:
        raise GateError(f"{run_dir.name}: baseline oracle or final-empty failure")
    if set(baseline_wait) != set(active_wait) or len(grouped_wait) != len(active_wait):
        raise GateError(f"{run_dir.name}: replay Task exact-once failure")
    baseline_values = list(baseline_wait.values())
    active_values = list(active_wait.values())
    deltas = [active_wait[task_id] - baseline_wait[task_id] for task_id in baseline_wait]
    return {
        "run_id": run_dir.name,
        "scenario": run_dir.name.split("_")[5],
        "decision_count": len(slots),
        "alpha_per_ns": ALPHA_PER_NS,
        "winner_change_count": winner_change_vs_b0,
        "winner_change_rate": winner_change_vs_b0 / len(slots),
        "grouping_only_winner_change_vs_b0_count": grouping_only_change_vs_b0,
        "aging_winner_change_vs_grouped_alpha0_count": aging_change_vs_grouped,
        "baseline_queue_wait": distribution(baseline_values),
        "active_queue_wait": distribution(active_values),
        "queue_wait_active_minus_baseline": {
            name: distribution(active_values)[name] - distribution(baseline_values)[name]
            for name in ("p95_ns", "p99_ns", "max_ns")
        },
        "task_wait_delta": {
            "improved": sum(value < 0 for value in deltas),
            "worsened": sum(value > 0 for value in deltas),
            "unchanged": sum(value == 0 for value in deltas),
            "sum_ns": sum(deltas),
            "max_improvement_ns": max((-value for value in deltas if value < 0), default=0),
            "max_worsening_ns": max((value for value in deltas if value > 0), default=0),
        },
        "deadline": {
            "baseline_task_count": baseline_deadline_tasks,
            "active_task_count": active_deadline_tasks,
            "baseline_miss_count": baseline_deadline_miss,
            "active_miss_count": active_deadline_miss,
            "miss_delta": active_deadline_miss - baseline_deadline_miss,
            "hard_urgent_bypass_count": hard_urgent_bypass,
        },
        "invariants": {
            "baseline_oracle_mismatch": baseline_mismatch,
            "stale_handle_count": 0,
            "full_store_scan_count": 0,
            "task_exact_once": True,
            "final_queue_empty": True,
        },
    }


def source_b0_dirs() -> list[Path]:
    paths = sorted(SOURCE_ROOT.glob("*_B0"))
    if len(paths) != 9:
        raise GateError(f"expected 9 M6C-F B0 source Runs, found {len(paths)}")
    return paths


def run_offline_replay() -> dict[str, Any]:
    runs = []
    for path in source_b0_dirs():
        print(f"OFFLINE {path.name}", flush=True)
        runs.append(replay_source_run(path))
    total_decisions = sum(item["decision_count"] for item in runs)
    total_changes = sum(item["winner_change_count"] for item in runs)
    total_grouping = sum(item["grouping_only_winner_change_vs_b0_count"] for item in runs)
    total_aging = sum(item["aging_winner_change_vs_grouped_alpha0_count"] for item in runs)
    result = {
        "schema_version": "continuous-aging-offline-replay-v1",
        "source": "nine_existing_M6C_F_B0_Traces",
        "counterfactual_type": "fixed_arrival_fixed_service_slot_policy_replay",
        "physical_system_reexecuted": False,
        "performance_claim": False,
        "alpha_derivation": {
            "source_runs": 9,
            "positive_adjacent_same_group_gap_samples": 1_118_790,
            "typical_gap_statistic": "median",
            "median_route_score_gap": MEDIAN_ROUTE_SCORE_GAP,
            "target_compensation_ns": COMPENSATION_NS,
            "formula": "alpha=median_positive_adjacent_same_group_gap/50000000",
            "alpha_per_ns": ALPHA_PER_NS,
            "alpha_per_ms": ALPHA_PER_NS * 1_000_000,
        },
        "run_count": len(runs),
        "decision_count": total_decisions,
        "winner_change_count": total_changes,
        "winner_change_rate": total_changes / total_decisions,
        "grouping_only_winner_change_vs_b0_count": total_grouping,
        "aging_winner_change_vs_grouped_alpha0_count": total_aging,
        "deadline_miss_delta": sum(item["deadline"]["miss_delta"] for item in runs),
        "hard_urgent_bypass_count": sum(
            item["deadline"]["hard_urgent_bypass_count"] for item in runs
        ),
        "runs": runs,
        "proceed_to_real_ab": total_changes > 0,
    }
    return result


def write_closed_json(path: Path, value: Any) -> None:
    base.write_json(path, value)
    if base.read_json(path) != value:
        raise GateError(f"closed JSON reopen mismatch: {path}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wait_distribution(values: list[int]) -> dict[str, Any]:
    return {
        **distribution(values),
        "ge_50ms_count": sum(value >= 50_000_000 for value in values),
    }


def physical_environment(
        run_id: str,
        scenario: str,
        configuration: str,
        repeat: int,
        order: int,
        output_base: Path) -> dict[str, str]:
    settings = SCENARIOS[scenario]
    env = base.common_environment(
        run_id, settings["workers"], "B0", repeat, order
    )
    env.update({
        "TRACE_BASE_DIR": str(output_base),
        "TRACE_OUT_DIR": str(output_base / run_id),
        "NUM_TOKENS_PREDICT": str(settings["generated_tokens"]),
        "ORDER_MODE": "scenario_blocked_interleaved",
        "ORDER_SEED": ORDER_SEED,
        "LLM_MEM_TRACE_AUDIT_CASE": run_id,
        "LLM_MEM_TRACE_AUDIT_CONFIGURATION_ID": configuration,
        "LLM_MEM_TRACE_AUDIT_SLOT_ID": f"{scenario}_r{repeat}_{configuration}",
        "LLM_MEM_TRACE_AUDIT_SCENARIO": scenario,
        "LLM_MEM_TRACE_AUDIT_GENERATED_TOKENS": str(settings["generated_tokens"]),
        "LLM_MEM_TRACE_AUDIT_CONTINUOUS_AGING_ALPHA_PER_NS": repr(ALPHA_PER_NS),
        "LLM_MEM_TRACE_OPT_EXPERT_RESERVED_SERVICE_ACTIVE": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_CONTINUOUS_AGING_ACTIVE": (
            "1" if configuration == "B1" else "0"
        ),
    })
    return env


def physical_plan() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    order = 1
    for scenario in ("P1", "P2", "P3"):
        for repeat, configuration in ORDER:
            run_id = (
                f"continuous_aging_20260805_v1_p{order:02d}_"
                f"{scenario}_r{repeat}_{configuration}"
            )
            result.append({
                "order": order,
                "run_id": run_id,
                "scenario": scenario,
                "configuration": configuration,
                "repeat": repeat,
                **SCENARIOS[scenario],
                "scope_unit": (
                    f"continuous-aging-v1-p{order:02d}-{scenario.lower()}-"
                    f"r{repeat}-{configuration.lower()}"
                ),
            })
            order += 1
    return result


def scope_child(args: argparse.Namespace) -> int:
    output_base = Path(args.output_base).resolve()
    run_dir = output_base / args.run_id
    before = base.cgroup_snapshot()
    if before["memory.max"] != MEMORY_MAX or before["memory.swap.max"] != SWAP_MAX:
        raise GateError("effective delegated cgroup limits are not 7 GiB / 2 GiB")
    env = os.environ.copy()
    env.update(physical_environment(
        args.run_id, args.scenario, args.configuration,
        args.repeat, args.order, output_base,
    ))
    completed = subprocess.run(
        ["bash", str(PIPELINE)], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    after = base.cgroup_snapshot()
    run_dir.mkdir(parents=True, exist_ok=True)
    base.write_text(run_dir / "pipeline.log", completed.stdout)
    base.write_json(run_dir / "continuous_aging_scope.json", {
        "schema_version": "continuous-aging-scope-v1",
        "run_id": args.run_id,
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
        "-p", f"MemoryMax={MEMORY_MAX}",
        "-p", f"MemorySwapMax={SWAP_MAX}",
        "taskset", "-c", "0-7", sys.executable, str(Path(__file__).resolve()),
        "--scope-child", "--output-base", str(output_base),
        "--run-id", entry["run_id"], "--scenario", entry["scenario"],
        "--configuration", entry["configuration"],
        "--repeat", str(entry["repeat"]), "--order", str(entry["order"]),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    base.write_text(OUTPUT_ROOT / f"{entry['run_id']}_launcher.log", completed.stdout)
    if completed.returncode:
        raise GateError(
            f"{entry['run_id']}: delegated scope launcher failed "
            f"({completed.returncode})"
        )


def task_identity(task: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(task[name] for name in (
        "task_id", "step", "layer", "expert", "phase", "stage", "tensor",
        "nbytes", "route_score_f64_bits", "sequence",
    ))


def validate_physical_run(
        run_dir: Path,
        scenario: str,
        configuration: str) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    settings = SCENARIOS[scenario]
    required = (
        "memory_trace.jsonl", "summary.json", "process_metrics.json",
        "run_manifest.json", "output.sha256", "inference_output.txt",
        "cache_preparation.json", "analysis/metrics.json",
        "continuous_aging_scope.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise GateError(f"{run_dir.name}: missing artifacts {missing}")

    trace_summary = base.read_json(run_dir / "summary.json")
    trace_drop = sum(
        int(sink.get("dropped", 0)) for sink in trace_summary["sinks"].values()
    )
    if trace_drop:
        raise GateError(f"{run_dir.name}: Trace drop is nonzero")
    process = base.read_json(run_dir / "process_metrics.json")
    if int(process.get("exit_code", -1)) != 0:
        raise GateError(f"{run_dir.name}: inference exit code is nonzero")
    output_sha = base.read_text(run_dir / "output.sha256")
    if output_sha != base.sha256_file(run_dir / "inference_output.txt"):
        raise GateError(f"{run_dir.name}: output SHA mismatch")

    manifest = base.read_json(run_dir / "run_manifest.json")
    environment = manifest["environment"]
    expected_environment = {
        "LLM_MEM_TRACE_OPT_EXPERT_CONTINUOUS_AGING_ACTIVE": (
            "1" if configuration == "B1" else "0"
        ),
        "LLM_MEM_TRACE_OPT_EXPERT_RESERVED_SERVICE_ACTIVE": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE": "deadline_score",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS": str(settings["workers"]),
        "NUM_TOKENS_PREDICT": str(settings["generated_tokens"]),
        "NUM_THREADS": "8", "BATCH_SIZE": "512", "CTX_SIZE": "2048",
        "GPU_LAYERS": "0", "TEMP": "0.0", "SEED": "1234",
    }
    mismatch = {
        name: (environment.get(name), value)
        for name, value in expected_environment.items()
        if environment.get(name) != value
    }
    if mismatch:
        raise GateError(f"{run_dir.name}: frozen environment mismatch {mismatch}")
    if manifest["host"].get("cpu_affinity") != list(range(8)):
        raise GateError(f"{run_dir.name}: CPU affinity is not 0-7")

    lifecycle_mask: dict[int, int] = {}
    duplicate_lifecycle = 0
    reject_cancel = 0
    tasks: dict[int, dict[str, Any]] = {}
    queue_waits: list[int] = []
    hint_service: list[int] = []
    step_latency: dict[str, list[int]] = {"PREFILL": [], "DECODE": []}
    async_summaries: list[dict[str, Any]] = []
    continuous_summaries: list[dict[str, Any]] = []
    continuous_selections: list[dict[str, Any]] = []
    reserved_events = 0
    queue_summaries: list[dict[str, Any]] = []
    deadline_miss_count = 0
    deadline_task_count = 0
    for record in base.event_records(run_dir / "memory_trace.jsonl"):
        event = record.get("event")
        if event == "EXPERT_TASK":
            task_id = int(record.get("task_id", 0))
            lifecycle = str(record.get("lifecycle_event", ""))
            if lifecycle in LIFECYCLE_BITS:
                bit = LIFECYCLE_BITS[lifecycle]
                before = lifecycle_mask.get(task_id, 0)
                duplicate_lifecycle += bool(before & bit)
                lifecycle_mask[task_id] = before | bit
            if lifecycle == "DEQUEUE":
                wait_ns = int(record["queue_wait_ns"])
                queue_waits.append(wait_ns)
                deadline = int(record["deadline_ts_ns"])
                deadline_task_count += deadline != 0
                deadline_miss_count += (
                    deadline != 0 and int(record["dequeued_ts_ns"]) >= deadline
                )
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
                    "deadline_ts_ns": deadline,
                    "queue_wait_ns": wait_ns,
                }
            elif lifecycle == "ISSUE" and record.get("returned_ts_ns") is not None:
                hint_service.append(
                    int(record["returned_ts_ns"]) - int(record["issued_ts_ns"])
                )
            elif lifecycle in {"REJECT", "CANCEL"}:
                reject_cancel += 1
        elif event == "STEP_END" and record.get("phase") in step_latency:
            step_latency[str(record["phase"])].append(int(record["latency_ns"]))
        elif event == "EXPERT_ASYNC_SUMMARY":
            async_summaries.append(record)
        elif event == "EXPERT_CONTINUOUS_AGING_SUMMARY":
            continuous_summaries.append(record)
        elif event == "EXPERT_CONTINUOUS_AGING_SELECTION":
            continuous_selections.append(record)
        elif event in {"EXPERT_RESERVED_SERVICE_SUMMARY", "EXPERT_RESERVED_SERVICE_SELECTION"}:
            reserved_events += 1
        elif event == "EXPERT_QUEUE_OVERHEAD_SUMMARY":
            queue_summaries.append(record)

    bad_lifecycle = sum(mask != ALL_LIFECYCLE_BITS for mask in lifecycle_mask.values())
    if (not tasks or set(tasks) != set(lifecycle_mask) or bad_lifecycle or
            duplicate_lifecycle or reject_cancel):
        raise GateError(
            f"{run_dir.name}: lifecycle/conservation failure tasks={len(tasks)} "
            f"bad={bad_lifecycle} duplicate={duplicate_lifecycle} "
            f"reject_cancel={reject_cancel}"
        )
    if len(async_summaries) != 1 or len(queue_summaries) != 1:
        raise GateError(f"{run_dir.name}: async/queue summary count mismatch")
    if reserved_events:
        raise GateError(f"{run_dir.name}: Reserved-Service event observed")
    async_summary = async_summaries[0]
    queue_summary = queue_summaries[0]
    if (int(async_summary.get("final_queue_depth", -1)) != 0 or
            int(async_summary.get("final_queued_bytes", -1)) != 0):
        raise GateError(f"{run_dir.name}: runtime queue did not drain")
    if bool(async_summary.get("continuous_aging_active")) != (configuration == "B1"):
        raise GateError(f"{run_dir.name}: feature runtime state mismatch")
    if bool(async_summary.get("reserved_service_active")):
        raise GateError(f"{run_dir.name}: Reserved-Service unexpectedly active")

    zero_counters = (
        "stale_handle_count", "duplicate_erase_count",
        "generation_mismatch_count", "full_store_scan_count",
        "invariant_error_count",
    )
    active: dict[str, Any]
    if configuration == "B1":
        if len(continuous_summaries) != 1:
            raise GateError(f"{run_dir.name}: Continuous Aging summary missing")
        active = continuous_summaries[0]
        frozen = {
            "base_priority_mode": "deadline_score",
            "group_order": "step_layer_early_late",
            "within_group_order": "router_score_plus_alpha_wait",
            "static_key": "router_score_minus_alpha_enqueue_offset",
            "store_kind": "bounded_unique_task_store",
            "index_kind": "dual_indexed_binary_heap",
        }
        if any(active.get(name) != value for name, value in frozen.items()):
            raise GateError(f"{run_dir.name}: Active semantics mismatch")
        if not math.isclose(
                float(active.get("alpha_per_ns", 0.0)), ALPHA_PER_NS,
                rel_tol=0.0, abs_tol=1e-27):
            raise GateError(f"{run_dir.name}: alpha mismatch")
        if any(int(active.get(name, -1)) for name in zero_counters):
            raise GateError(f"{run_dir.name}: Active invariant counter nonzero")
        if (not active.get("store_index_registry_bytes_conserved") or
                not active.get("final_queue_empty")):
            raise GateError(f"{run_dir.name}: Active conservation/final-empty failure")
        selection_count = int(active.get("selection_count", -1))
        if selection_count != len(tasks) or selection_count != len(continuous_selections):
            raise GateError(f"{run_dir.name}: Active selection count mismatch")
        continuous_selections.sort(key=lambda item: int(item["decision_id"]))
        if [int(item["decision_id"]) for item in continuous_selections] != list(range(selection_count)):
            raise GateError(f"{run_dir.name}: Active decision IDs are not dense")
        changed = sum(
            bool(item.get("active_winner_changed_vs_legacy"))
            for item in continuous_selections
        )
        if changed != int(active["active_winner_changed_count"]):
            raise GateError(f"{run_dir.name}: winner-change counter mismatch")
        if int(active["insert_count"]) != len(tasks) or int(active["erase_count"]) != len(tasks):
            raise GateError(f"{run_dir.name}: index operation conservation failure")
    else:
        if continuous_summaries or continuous_selections:
            raise GateError(f"{run_dir.name}: feature-off emitted Active events")
        active = {
            "active_winner_changed_count": 0,
            "winner_same_as_legacy_count": 0,
            "hard_urgent_bypass_count": 0,
            "selected_after_deadline_count": 0,
            "stale_handle_count": 0,
            "full_store_scan_count": 0,
            "invariant_error_count": 0,
            "store_index_registry_bytes_conserved": None,
            "final_queue_empty": True,
            "enqueue_index_op_mean_ns": None,
            "enqueue_index_op_max_ns": None,
            "dequeue_index_op_mean_ns": None,
            "dequeue_index_op_max_ns": None,
        }

    scope = base.read_json(run_dir / "continuous_aging_scope.json")
    before, after = scope["before"], scope["after"]
    if before["path"] != after["path"] or before["inode"] != after["inode"]:
        raise GateError(f"{run_dir.name}: cgroup changed during Run")
    if before["memory.max"] != MEMORY_MAX or before["memory.swap.max"] != SWAP_MAX:
        raise GateError(f"{run_dir.name}: effective cgroup limit mismatch")
    events_delta = scope["delta"]["memory.events"]
    if any(int(events_delta.get(name, 0)) for name in ("oom", "oom_kill", "oom_group_kill")):
        raise GateError(f"{run_dir.name}: OOM event observed")

    metrics = base.read_json(run_dir / "analysis" / "metrics.json")
    queue_global = queue_summary["global"]
    result = {
        "schema_version": "continuous-aging-physical-run-v1",
        "run_id": run_dir.name,
        "scenario": scenario,
        "configuration": configuration,
        "workers": settings["workers"],
        "generated_tokens": settings["generated_tokens"],
        "valid": True,
        "output_sha256": output_sha,
        "task_count": len(tasks),
        "trace_drop_count": trace_drop,
        "wall_time_s": float(process["wall_time_s"]),
        "prefill_latency_total_ms": sum(step_latency["PREFILL"]) / 1e6,
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
        "hint_service": distribution(hint_service),
        "deadline_task_count": deadline_task_count,
        "deadline_miss_count": deadline_miss_count,
        "max_queue_depth": int(async_summary["max_queue_depth"]),
        "mutex_acquire_wait": queue_global["mutex_acquire_wait_ns"],
        "mutex_hold": queue_global["mutex_hold_ns"],
        "queue_selection": queue_global["queue_scan_ns"],
        "active": {
            name: active.get(name) for name in (
                "active_winner_changed_count", "winner_same_as_legacy_count",
                "hard_urgent_bypass_count", "selected_after_deadline_count",
                "stale_handle_count", "duplicate_erase_count",
                "generation_mismatch_count", "full_store_scan_count",
                "invariant_error_count", "store_index_registry_bytes_conserved",
                "final_queue_empty", "insert_count", "erase_count", "selection_count",
                "enqueue_index_op_mean_ns", "enqueue_index_op_max_ns",
                "dequeue_index_op_mean_ns", "dequeue_index_op_max_ns",
            )
        },
        "cgroup": {
            "path": before["path"], "inode": before["inode"],
            "memory.max": before["memory.max"],
            "memory.swap.max": before["memory.swap.max"],
            "memory.events_delta": events_delta,
        },
    }
    return result, tasks


def execute_physical_entry(output_base: Path, entry: dict[str, Any]) -> dict[str, Any]:
    print(
        f"START {entry['order']:02d} {entry['scenario']} "
        f"{entry['configuration']} r{entry['repeat']}", flush=True,
    )
    run_in_scope(output_base, entry)
    run_dir = output_base / entry["run_id"]
    result, tasks = validate_physical_run(
        run_dir, entry["scenario"], entry["configuration"]
    )
    base.write_json(run_dir / "continuous_aging_run_result.json", result)
    second, second_tasks = validate_physical_run(
        run_dir, entry["scenario"], entry["configuration"]
    )
    if second != result or second_tasks != tasks:
        raise GateError(f"{entry['run_id']}: deterministic Trace reparse mismatch")
    print(
        f"PASS  {entry['order']:02d} tasks={result['task_count']} "
        f"winner_change={result['active']['active_winner_changed_count']}",
        flush=True,
    )
    return {**result, "_tasks": tasks}


def pair_physical_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (run["scenario"], int(run["run_id"].split("_r", 1)[1].split("_", 1)[0]),
         run["configuration"]): run
        for run in runs
    }
    pairs: list[dict[str, Any]] = []
    scalar = (
        "wall_time_s", "prefill_latency_total_ms", "decode_latency_total_ms",
        "decode_latency_mean_ms", "decode_throughput_tokens_per_s",
        "major_faults", "minor_faults", "rss_peak_kb", "cgroup_memory_peak_bytes",
        "swap_peak_bytes", "psi_some_total_delta_us", "psi_full_total_delta_us",
        "deadline_miss_count",
    )
    for scenario in SCENARIOS:
        for repeat in (1, 2, 3):
            b0 = indexed[(scenario, repeat, "B0")]
            b1 = indexed[(scenario, repeat, "B1")]
            if b0["output_sha256"] != b1["output_sha256"]:
                raise GateError(f"{scenario} r{repeat}: output Hash mismatch")
            if set(b0["_tasks"]) != set(b1["_tasks"]):
                raise GateError(f"{scenario} r{repeat}: Task ID set mismatch")
            mismatch = sum(
                task_identity(b0["_tasks"][task_id]) !=
                task_identity(b1["_tasks"][task_id])
                for task_id in b0["_tasks"]
            )
            if mismatch:
                raise GateError(f"{scenario} r{repeat}: {mismatch} immutable Task mismatches")
            pairs.append({
                "scenario": scenario,
                "repeat": repeat,
                "B0_run_id": b0["run_id"],
                "B1_run_id": b1["run_id"],
                "B1_minus_B0": {name: b1[name] - b0[name] for name in scalar},
                "B1_minus_B0_queue_wait": {
                    name: b1["queue_wait"][name] - b0["queue_wait"][name]
                    for name in ("p95_ns", "p99_ns", "max_ns", "ge_50ms_count")
                },
                "winner_change_count": b1["active"]["active_winner_changed_count"],
                "hard_urgent_bypass_count": b1["active"]["hard_urgent_bypass_count"],
            })
    return pairs


def scenario_summary(
        scenario: str,
        runs: list[dict[str, Any]],
        pairs: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [pair for pair in pairs if pair["scenario"] == scenario]
    b1 = [run for run in runs if run["scenario"] == scenario and run["configuration"] == "B1"]
    fields = (
        "wall_time_s", "decode_latency_total_ms", "decode_throughput_tokens_per_s",
        "major_faults", "rss_peak_kb", "swap_peak_bytes", "deadline_miss_count",
    )
    queue_fields = ("p95_ns", "p99_ns", "max_ns")
    return {
        "scenario": scenario,
        "settings": SCENARIOS[scenario],
        "winner_change_counts": [row["active"]["active_winner_changed_count"] for row in b1],
        "winner_change_total": sum(row["active"]["active_winner_changed_count"] for row in b1),
        "hard_urgent_bypass_total": sum(row["active"]["hard_urgent_bypass_count"] for row in b1),
        "median_B1_minus_B0": {
            **{
                name: statistics.median(pair["B1_minus_B0"][name] for pair in selected)
                for name in fields
            },
            **{
                f"queue_wait_{name}": statistics.median(
                    pair["B1_minus_B0_queue_wait"][name] for pair in selected
                ) for name in queue_fields
            },
        },
        "direction_counts": {
            "queue_p95_improved": sum(pair["B1_minus_B0_queue_wait"]["p95_ns"] < 0 for pair in selected),
            "queue_p99_improved": sum(pair["B1_minus_B0_queue_wait"]["p99_ns"] < 0 for pair in selected),
            "queue_max_improved": sum(pair["B1_minus_B0_queue_wait"]["max_ns"] < 0 for pair in selected),
            "wall_improved": sum(pair["B1_minus_B0"]["wall_time_s"] < 0 for pair in selected),
            "decode_improved": sum(pair["B1_minus_B0"]["decode_latency_total_ms"] < 0 for pair in selected),
            "throughput_improved": sum(pair["B1_minus_B0"]["decode_throughput_tokens_per_s"] > 0 for pair in selected),
        },
    }


def classify_physical(
        runs: list[dict[str, Any]],
        summaries: list[dict[str, Any]]) -> tuple[str, bool, list[str]]:
    b1 = [run for run in runs if run["configuration"] == "B1"]
    if all(int(run["active"]["active_winner_changed_count"]) == 0 for run in b1):
        return "CONTINUOUS_AGING_NO_OP", False, ["all nine B1 Runs have zero winner changes"]
    invariant_fields = (
        "stale_handle_count", "duplicate_erase_count", "generation_mismatch_count",
        "full_store_scan_count", "invariant_error_count",
    )
    if any(int(run["active"].get(name) or 0) for run in b1 for name in invariant_fields):
        return "CONTINUOUS_AGING_UNSUITABLE", False, ["safety or queue invariant failure"]

    fairness = all(
        summary["direction_counts"]["queue_p99_improved"] >= 2 or
        summary["direction_counts"]["queue_max_improved"] >= 2
        for summary in summaries
    )
    physical = all(
        max(
            summary["direction_counts"]["wall_improved"],
            summary["direction_counts"]["decode_improved"],
            summary["direction_counts"]["throughput_improved"],
        ) >= 2 for summary in summaries
    )
    deadline_regression = any(
        summary["median_B1_minus_B0"]["deadline_miss_count"] > 0
        for summary in summaries
    )
    reasons: list[str] = []
    if deadline_regression:
        reasons.append("median deadline-miss count increased in at least one scenario")
    if fairness and physical and not deadline_regression:
        return "CONTINUOUS_AGING_WAIT_AND_PHYSICAL_SIGNAL", True, [
            "all scenarios have majority-direction queue-tail and physical improvements"
        ]
    if fairness and not deadline_regression:
        return "CONTINUOUS_AGING_FAIRNESS_SIGNAL_ONLY", True, [
            "queue-tail direction is consistent but physical direction is not"
        ]
    if not reasons:
        reasons.append("winner changes occurred without consistent queue-tail improvement")
    return "CONTINUOUS_AGING_REORDERING_INEFFECTIVE", False, reasons


def physical_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Continuous Aging Score Active A/B", "",
        "> 这是每格 N=3 的探索性物理 A/B，不是正式 N=8。", "",
        f"结论：`{report['classification']}`；值得保留：`{str(report['worth_retaining']).lower()}`", "",
        "| Run | 场景 | cfg | winner change | qwait p95/p99/max ms | wall s | decode ms | tok/s | majflt | RSS MiB | swap MiB |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["raw_runs"]:
        wait = run["queue_wait"]
        lines.append(
            f"| {run['run_id']} | {run['scenario']} | {run['configuration']} | "
            f"{run['active']['active_winner_changed_count']} | "
            f"{wait['p95_ns']/1e6:.3f}/{wait['p99_ns']/1e6:.3f}/{wait['max_ns']/1e6:.3f} | "
            f"{run['wall_time_s']:.3f} | {run['decode_latency_total_ms']:.3f} | "
            f"{run['decode_throughput_tokens_per_s']:.3f} | {run['major_faults']} | "
            f"{run['rss_peak_kb']/1024:.1f} | {run['swap_peak_bytes']/1048576:.1f} |"
        )
    lines.extend(["", "## 分场景配对中位数", ""])
    for summary in report["scenario_summaries"]:
        lines.append(
            f"- {summary['scenario']}: winner changes={summary['winner_change_counts']}; "
            f"median B1-B0={json.dumps(summary['median_B1_minus_B0'], sort_keys=True)}"
        )
    lines.extend(["", "## 判断依据", ""])
    lines.extend(f"- {reason}" for reason in report["decision_reasons"])
    lines.append("")
    return "\n".join(lines)


def finalize_physical_report(
        runs: list[dict[str, Any]], offline_path: Path) -> int:
    offline = base.read_json(offline_path)
    pairs = pair_physical_runs(runs)
    summaries = [scenario_summary(name, runs, pairs) for name in SCENARIOS]
    classification, worth_retaining, reasons = classify_physical(runs, summaries)
    public_runs = [
        {key: value for key, value in run.items() if key != "_tasks"}
        for run in runs
    ]
    report = {
        "schema_version": "continuous-aging-physical-report-v1",
        "created_at_utc": utc_now(),
        "physical_system_reexecuted": True,
        "performance_claim": False,
        "formal_n8": False,
        "alpha_per_ns": ALPHA_PER_NS,
        "offline": {
            "path": str(offline_path),
            "sha256": base.sha256_file(offline_path),
            "winner_change_count": offline["winner_change_count"],
            "winner_change_rate": offline["winner_change_rate"],
            "grouping_only_winner_change_vs_b0_count": offline["grouping_only_winner_change_vs_b0_count"],
            "aging_winner_change_vs_grouped_alpha0_count": offline["aging_winner_change_vs_grouped_alpha0_count"],
        },
        "raw_runs": public_runs,
        "paired_results": pairs,
        "scenario_summaries": summaries,
        "correctness_and_invariants": {
            "all_runs_valid": all(run["valid"] for run in runs),
            "trace_drop_total": sum(run["trace_drop_count"] for run in runs),
            "stale_handle_total": sum(int(run["active"].get("stale_handle_count") or 0) for run in runs),
            "full_store_scan_total": sum(int(run["active"].get("full_store_scan_count") or 0) for run in runs),
            "invariant_error_total": sum(int(run["active"].get("invariant_error_count") or 0) for run in runs),
            "all_final_queues_empty": all(run["active"].get("final_queue_empty") for run in runs),
        },
        "classification": classification,
        "worth_retaining": worth_retaining,
        "decision_reasons": reasons,
    }
    write_closed_json(OUTPUT_ROOT / "final_report.json", report)
    base.write_text(OUTPUT_ROOT / "final_report.md", physical_markdown(report))
    authority = {
        "schema_version": "continuous-aging-authority-v1",
        "report_path": str(OUTPUT_ROOT / "final_report.json"),
        "report_sha256": base.sha256_file(OUTPUT_ROOT / "final_report.json"),
        "markdown_sha256": base.sha256_file(OUTPUT_ROOT / "final_report.md"),
        "classification": classification,
        "worth_retaining": worth_retaining,
    }
    write_closed_json(OUTPUT_ROOT / "authority.json", authority)
    print(json.dumps(authority, indent=2, sort_keys=True), flush=True)
    return 0


def finalize_existing_physical() -> int:
    if not OUTPUT_ROOT.is_dir() or not RAW_ROOT.is_dir():
        raise GateError("physical output or raw-run directory is unavailable")
    if (OUTPUT_ROOT / "final_report.json").exists():
        raise GateError("final physical report already exists")
    runs: list[dict[str, Any]] = []
    for entry in physical_plan():
        run_dir = RAW_ROOT / entry["run_id"]
        result, tasks = validate_physical_run(
            run_dir, entry["scenario"], entry["configuration"]
        )
        stored = base.read_json(run_dir / "continuous_aging_run_result.json")
        if stored != result:
            raise GateError(f"{entry['run_id']}: stored/raw deterministic mismatch")
        runs.append({**result, "_tasks": tasks})
    return finalize_physical_report(
        runs, TRACE_ROOT / "continuous_aging_offline_replay_20260805_v1.json"
    )


def execute_physical() -> int:
    if OUTPUT_ROOT.exists():
        raise GateError(f"output directory already exists: {OUTPUT_ROOT}")
    if base.sha256_file(MODEL) != base.EXPECTED_MODEL_SHA256:
        raise GateError("model SHA mismatch")
    helper = LLAMA_ROOT / "trace" / "prepare_model_cache.py"
    if base.sha256_file(helper) != base.EXPECTED_HELPER_SHA256:
        raise GateError("cold-cache helper SHA mismatch")
    offline_path = TRACE_ROOT / "continuous_aging_offline_replay_20260805_v1.json"
    offline = base.read_json(offline_path)
    if not offline.get("proceed_to_real_ab"):
        raise GateError("offline Replay has zero winner changes")
    OUTPUT_ROOT.mkdir(parents=True)
    RAW_ROOT.mkdir()
    plan = physical_plan()
    protocol = {
        "schema_version": "continuous-aging-physical-protocol-v1",
        "created_at_utc": utc_now(),
        "exploratory_not_formal_n8": True,
        "parameter_search": False,
        "alpha_per_ns": ALPHA_PER_NS,
        "alpha_source": "median positive adjacent same-group route-score gap / 50ms",
        "offline_replay_path": str(offline_path),
        "offline_replay_sha256": base.sha256_file(offline_path),
        "binary_sha256": base.sha256_file(BINARY),
        "model_sha256": base.EXPECTED_MODEL_SHA256,
        "cgroup": {"memory.max": MEMORY_MAX, "memory.swap.max": SWAP_MAX},
        "scenarios": SCENARIOS,
        "order_seed": ORDER_SEED,
        "matrix": plan,
    }
    write_closed_json(OUTPUT_ROOT / "protocol.json", protocol)

    runs: list[dict[str, Any]] = []
    for entry in plan:
        runs.append(execute_physical_entry(RAW_ROOT, entry))
    return finalize_physical_report(runs, offline_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-replay", action="store_true")
    parser.add_argument("--execute-physical", action="store_true")
    parser.add_argument("--finalize-existing-physical", action="store_true")
    parser.add_argument("--scope-child", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-base")
    parser.add_argument("--run-id")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS))
    parser.add_argument("--configuration", choices=("B0", "B1"))
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--order", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.scope_child:
            if not all((args.output_base, args.run_id, args.scenario, args.configuration)):
                raise GateError("scope-child arguments are incomplete")
            return scope_child(args)
        if args.offline_replay:
            result = run_offline_replay()
            output = args.output or (
                TRACE_ROOT / "continuous_aging_offline_replay_20260805_v1.json"
            )
            if output.exists():
                raise GateError(f"output already exists: {output}")
            write_closed_json(output, result)
            print(json.dumps({
                "output": str(output),
                "sha256": base.sha256_file(output),
                "winner_change_count": result["winner_change_count"],
                "winner_change_rate": result["winner_change_rate"],
                "proceed_to_real_ab": result["proceed_to_real_ab"],
            }, indent=2, sort_keys=True))
            return 0
        if args.execute_physical:
            return execute_physical()
        if args.finalize_existing_physical:
            return finalize_existing_physical()
        parser.error(
            "choose --offline-replay, --execute-physical, or "
            "--finalize-existing-physical"
        )
    except GateError as exc:
        print(f"Continuous Aging Gate failure: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
