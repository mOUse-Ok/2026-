#!/usr/bin/env python3
"""Run and independently validate the minimal M6C Reserved-Service Active A/B.

This is deliberately limited to the frozen S1-D representative and the A.3
four-token workload. It is an engineering smoke/calibration, never formal N=8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
LLAMA_ROOT = ROOT / "llama.cpp"
TRACE_ROOT = LLAMA_ROOT / "trace_output"
DEFAULT_OUTPUT = TRACE_ROOT / "m6c_active_ab_20260805_v3"
SMOKE_EVIDENCE = (
    TRACE_ROOT
    / "m6c_active_ab_20260805_v2"
    / "m6c_active_ab_20260805_v2_smoke_w2_B1"
)
PIPELINE = LLAMA_ROOT / "trace" / "run_trace_pipeline.sh"
BINARY = LLAMA_ROOT / "build" / "bin" / "llama-cli"
MODEL = ROOT / "models" / "Qwen3.5-35B-A3B-Q3_K_M.gguf"
REFERENCE_MANIFEST = (
    TRACE_ROOT
    / "m6b2a3_directed_20260804_n5_v3_p01_w2_r1_B0_a1"
    / "m6b2a3_manifest.json"
)
EXPECTED_MODEL_SHA256 = "5607c8fcc8b04ada7d1a1152b9a5b6c1e67e6768232c16f6b03d9719d5ab1b2d"
EXPECTED_PROMPT_SHA256 = "59f51358b13d0600feaf78e0cccfb71c9f25bdce3259ddae301e6c3217897e4f"
EXPECTED_OUTPUT_SHA256 = "e720f3885685e2fb1f094f2b8801fa66e7b30367960f1d3100b19647476f3a0f"
EXPECTED_CANONICAL_OUTPUT_SHA256 = "9f4e2c4794bd84973d75e8c6b783a683b1e3ce6c9482c1ce2601064ad82b0692"
EXPECTED_HELPER_SHA256 = "46010daec4af945297bbaaa5f263472a6353e16425b9fcef27f163bdf50b1cdc"
MEMORY_MAX = "7516192768"
SWAP_MAX = "2147483648"
ORDER_SEED = "m6c_active_ab_20260805_minimal_s1d"


class GateError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_output_sha256(path: Path) -> str:
    # llama-cli embeds the source build ID in stdout. Runtime implementation
    # necessarily changes that banner, so correctness freezes every other byte.
    text = path.read_text(encoding="utf-8")
    canonical = "".join(
        line for line in text.splitlines(keepends=True)
        if not line.startswith("build      :")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def quantile(values: list[int | float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution_ns(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50_ns": quantile(values, 0.50),
        "p95_ns": quantile(values, 0.95),
        "p99_ns": quantile(values, 0.99),
        "max_ns": max(values) if values else None,
    }


def parse_key_values(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return result


def pressure_totals(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        for part in parts[1:]:
            if part.startswith("total="):
                result[parts[0]] = int(part.split("=", 1)[1])
    return result


def current_cgroup_path() -> Path:
    for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            path = Path("/sys/fs/cgroup") / parts[2].lstrip("/")
            if parts[2] == "/init.scope":
                raise GateError("delegated scope is /init.scope")
            return path
    raise GateError("unified cgroup path is unavailable")


def cgroup_snapshot() -> dict[str, Any]:
    path = current_cgroup_path()
    names = (
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory.swap.current",
        "memory.swap.peak",
        "memory.swap.max",
        "memory.events",
        "memory.events.local",
        "memory.pressure",
        "memory.stat",
    )
    values: dict[str, Any] = {"path": str(path), "inode": path.stat().st_ino}
    for name in names:
        candidate = path / name
        values[name] = candidate.read_text(encoding="utf-8").strip() if candidate.exists() else None
    return values


def snapshot_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("memory.current", "memory.peak", "memory.swap.current", "memory.swap.peak"):
        try:
            result[name] = int(after[name]) - int(before[name])
        except (KeyError, TypeError, ValueError):
            result[name] = None
    for name in ("memory.events", "memory.events.local", "memory.stat"):
        lhs = parse_key_values(before.get(name) or "")
        rhs = parse_key_values(after.get(name) or "")
        result[name] = {key: rhs.get(key, 0) - lhs.get(key, 0) for key in sorted(set(lhs) | set(rhs))}
    lhs_pressure = pressure_totals(before.get("memory.pressure") or "")
    rhs_pressure = pressure_totals(after.get("memory.pressure") or "")
    result["memory.pressure.total"] = {
        key: rhs_pressure.get(key, 0) - lhs_pressure.get(key, 0)
        for key in sorted(set(lhs_pressure) | set(rhs_pressure))
    }
    return result


def common_environment(run_id: str, workers: int, config: str, repeat: int, order: int) -> dict[str, str]:
    return {
        "ALLOW_DIRTY_REPO": "1",
        "TRACE_PROFILE": "custom",
        "CACHE_MODE": "cold",
        "NUM_TOKENS_PREDICT": "4",
        "NUM_THREADS": "8",
        "BATCH_SIZE": "512",
        "CTX_SIZE": "2048",
        "TEMP": "0.0",
        "SEED": "1234",
        "GPU_LAYERS": "0",
        "MODEL_FILE": str(MODEL),
        "MODEL_SHA256": EXPECTED_MODEL_SHA256,
        "LLAMA_CLI": str(BINARY),
        "TRACE_BASE_DIR": str(DEFAULT_OUTPUT),
        "RUN_NAME": run_id,
        "TRACE_OUT_DIR": str(DEFAULT_OUTPUT / run_id),
        "REPEAT_INDEX": str(repeat),
        "ORDER_POSITION": f"{order:02d}",
        "ORDER_MODE": "interleaved_preregistered",
        "ORDER_SEED": ORDER_SEED,
        "MEMORY_MAX": MEMORY_MAX,
        "MEMORY_SWAP_MAX": SWAP_MAX,
        "LLM_MEM_TRACE_AUDIT_CASE": run_id,
        "LLM_MEM_TRACE_AUDIT_CONFIGURATION_ID": config,
        "LLM_MEM_TRACE_AUDIT_CPU_AFFINITY": "0-7",
        "LLM_MEM_TRACE_AUDIT_FORMAL_N8": "0",
        "LLM_MEM_TRACE_AUDIT_HINT_WORKERS": str(workers),
        "LLM_MEM_TRACE_AUDIT_MODEL_THREADS": "8",
        "LLM_MEM_TRACE_AUDIT_PERFORMANCE_CLAIM": "0",
        "LLM_MEM_TRACE_AUDIT_SLOT_ID": f"w{workers}_r{repeat}_{config}",
        "LLM_MEM_TRACE_ALLOW_DROP": "0",
        "LLM_MEM_TRACE_TENSOR": "0",
        "LLM_MEM_TRACE_KV": "0",
        "LLM_MEM_TRACE_EXPERT": "1",
        "LLM_MEM_TRACE_MEMORY": "1",
        "LLM_MEM_TRACE_RESIDENCY": "0",
        "LLM_MEM_TRACE_SMAPS": "0",
        "LLM_MEM_TRACE_QUEUE_LIMIT": "524288",
        "LLM_MEM_TRACE_EXPERT_TASK_MODE": "detail",
        "LLM_MEM_TRACE_OS_HINTS": "1",
        "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH": "1",
        "LLM_MEM_TRACE_OPT_EXPERT_POLICY": "route",
        "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_TTL_STEPS": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_COALESCE": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC": "1",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE": "131072",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS": str(workers),
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY": "1",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE": "deadline_score",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH": "1",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_WAIT_US": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_COALESCE": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_FALLBACK": "1",
        "LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE": "1",
        "LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER": "off",
        "LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_SLACK": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT": "0",
        "LLM_MEM_TRACE_OPT_EXPERT_SLACK_MODE": "off",
        "LLM_MEM_TRACE_PRESSURE_SHADOW_MODE": "off",
        "LLM_MEM_TRACE_QUEUE_OVERHEAD_MODE": "detail",
        "LLM_MEM_TRACE_ROUTER_SCORE_DIAGNOSTIC": "1",
        "LLM_MEM_TRACE_ROUTER_TENSOR_SYNC_PROTOCOL": "m6b1.2-v1",
        "LLM_MEM_TRACE_OPT_EXPERT_RESERVED_SERVICE_ACTIVE": "1" if config == "B1" else "0",
        "OMP_NUM_THREADS": "8",
        "OMP_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "MKL_DYNAMIC": "FALSE",
        "BLIS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }


def event_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(record, dict):
                yield record


def validate_and_measure_run(run_dir: Path, config: str, workers: int) -> dict[str, Any]:
    required = (
        "memory_trace.jsonl",
        "expert_trace.jsonl",
        "summary.json",
        "process_metrics.json",
        "run_manifest.json",
        "output.sha256",
        "inference_output.txt",
        "cache_preparation.json",
        "analysis/metrics.json",
        "m6c_active_scope.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise GateError(f"{run_dir.name}: missing artifacts: {missing}")

    trace_summary = read_json(run_dir / "summary.json")
    if any(int(sink.get("dropped", 0)) != 0 for sink in trace_summary["sinks"].values()):
        raise GateError(f"{run_dir.name}: Trace drop is nonzero")
    process = read_json(run_dir / "process_metrics.json")
    if int(process.get("exit_code", -1)) != 0:
        raise GateError(f"{run_dir.name}: inference exit code is nonzero")
    output_sha = read_text(run_dir / "output.sha256")
    if output_sha != sha256_file(run_dir / "inference_output.txt"):
        raise GateError(f"{run_dir.name}: raw output artifact Hash mismatch")
    canonical_output_sha = canonical_output_sha256(run_dir / "inference_output.txt")
    if canonical_output_sha != EXPECTED_CANONICAL_OUTPUT_SHA256:
        raise GateError(f"{run_dir.name}: canonical generated output Hash mismatch")
    manifest = read_json(run_dir / "run_manifest.json")
    expected_active = "1" if config == "B1" else "0"
    if manifest["environment"].get("LLM_MEM_TRACE_OPT_EXPERT_RESERVED_SERVICE_ACTIVE") != expected_active:
        raise GateError(f"{run_dir.name}: feature flag manifest mismatch")
    if manifest["environment"].get("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE") != "deadline_score":
        raise GateError(f"{run_dir.name}: baseline priority mode changed")
    if manifest["host"].get("cpu_affinity") != list(range(8)):
        raise GateError(f"{run_dir.name}: CPU affinity is not 0-7")

    task_events: dict[int, list[str]] = {}
    queue_waits: list[int] = []
    hint_service: list[int] = []
    step_latency: dict[str, list[int]] = {"PREFILL": [], "DECODE": []}
    async_summaries: list[dict[str, Any]] = []
    reserved_summaries: list[dict[str, Any]] = []
    reserved_selections: list[dict[str, Any]] = []
    queue_summaries: list[dict[str, Any]] = []
    reject_cancel_count = 0
    for record in event_records(run_dir / "memory_trace.jsonl"):
        event = record.get("event")
        if event == "EXPERT_TASK":
            task_id = int(record.get("task_id", 0))
            lifecycle = str(record.get("lifecycle_event", ""))
            task_events.setdefault(task_id, []).append(lifecycle)
            if lifecycle == "DEQUEUE" and record.get("queue_wait_ns") is not None:
                queue_waits.append(int(record["queue_wait_ns"]))
            if lifecycle == "ISSUE" and record.get("returned_ts_ns") is not None:
                hint_service.append(int(record["returned_ts_ns"]) - int(record["issued_ts_ns"]))
            if lifecycle in {"REJECT", "CANCEL"}:
                reject_cancel_count += 1
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

    if not task_events or reject_cancel_count:
        raise GateError(f"{run_dir.name}: empty Task stream or Reject/Cancel observed")
    required_lifecycle = ("CREATE", "ADMIT", "ENQUEUE", "DEQUEUE", "ISSUE")
    bad_tasks = 0
    for events in task_events.values():
        if any(events.count(name) != 1 for name in required_lifecycle):
            bad_tasks += 1
    if bad_tasks:
        raise GateError(f"{run_dir.name}: {bad_tasks} Tasks fail exact-once lifecycle")
    if len(async_summaries) != 1 or len(queue_summaries) != 1:
        raise GateError(f"{run_dir.name}: expected exactly one async/queue summary")
    async_summary = async_summaries[0]
    queue_summary = queue_summaries[0]
    if int(async_summary.get("final_queue_depth", -1)) != 0 or int(async_summary.get("final_queued_bytes", -1)) != 0:
        raise GateError(f"{run_dir.name}: final queue is not empty")
    if bool(async_summary.get("reserved_service_active")) != (config == "B1"):
        raise GateError(f"{run_dir.name}: runtime feature-off/feature-on mismatch")

    reserved: dict[str, Any]
    if config == "B1":
        if len(reserved_summaries) != 1:
            raise GateError(f"{run_dir.name}: missing Reserved-Service summary")
        reserved = reserved_summaries[0]
        exact = {
            "reserved_numerator": 1,
            "reserved_denominator": 8,
            "eligibility_age_ns": 41000000,
            "hard_urgent_guard_ns": 0,
            "eligibility_rule": "AGE_GATED_ALL",
            "debt_policy": "single_pending_latch",
            "reset_policy": "reset_when_no_eligible",
            "reserved_winner": "oldest_eligible",
        }
        for name, value in exact.items():
            if reserved.get(name) != value:
                raise GateError(f"{run_dir.name}: frozen setting {name} mismatch")
        zero_fields = (
            "hard_urgent_safety_violation",
            "stale_handle_count",
            "duplicate_erase_count",
            "generation_mismatch_count",
            "full_store_scan_count",
            "invariant_error_count",
        )
        if any(int(reserved.get(name, -1)) != 0 for name in zero_fields):
            raise GateError(f"{run_dir.name}: Active safety/invariant failure")
        if not reserved.get("store_index_registry_bytes_conserved") or not reserved.get("final_queue_empty"):
            raise GateError(f"{run_dir.name}: Active conservation failure")
        selection_count = int(reserved.get("selection_count", -1))
        if selection_count != len(reserved_selections):
            raise GateError(f"{run_dir.name}: Active selection Detail count mismatch")
        decision_ids = sorted(int(record["decision_id"]) for record in reserved_selections)
        if decision_ids != list(range(selection_count)):
            raise GateError(f"{run_dir.name}: Active decision IDs are not unique and dense")
        changed = sum(bool(record.get("active_winner_changed_vs_legacy")) for record in reserved_selections)
        same = sum(bool(record.get("reserved_same_as_legacy_head")) for record in reserved_selections)
        if changed != int(reserved["active_winner_changed_count"]) or same != int(reserved["reserved_same_as_legacy_head_count"]):
            raise GateError(f"{run_dir.name}: Active winner counter mismatch")
    else:
        if reserved_summaries or reserved_selections:
            raise GateError(f"{run_dir.name}: feature-off emitted Active events")
        reserved = {
            "reserved_trigger_count": 0,
            "reserved_due_count": 0,
            "reserved_selected_count": 0,
            "active_winner_changed_count": 0,
            "reserved_same_as_legacy_head_count": 0,
            "hard_urgent_override_count": 0,
            "hard_urgent_safety_violation": 0,
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

    scope = read_json(run_dir / "m6c_active_scope.json")
    before = scope["before"]
    after = scope["after"]
    if before["path"] != after["path"] or before["inode"] != after["inode"]:
        raise GateError(f"{run_dir.name}: scope changed during Run")
    if before["memory.max"] != MEMORY_MAX or before["memory.swap.max"] != SWAP_MAX:
        raise GateError(f"{run_dir.name}: cgroup limits mismatch")
    events_delta = scope["delta"]["memory.events"]
    if any(int(events_delta.get(name, 0)) != 0 for name in ("oom", "oom_kill", "oom_group_kill")):
        raise GateError(f"{run_dir.name}: OOM event observed")

    metrics = read_json(run_dir / "analysis" / "metrics.json")
    queue_global = queue_summary["global"]
    queue_wait = distribution_ns(queue_waits)
    hint = distribution_ns(hint_service)
    result = {
        "schema_version": "m6c-active-ab-run-v1",
        "run_id": run_dir.name,
        "configuration": config,
        "workers": workers,
        "valid": True,
        "output_sha256": output_sha,
        "canonical_output_sha256": canonical_output_sha,
        "task_count": len(task_events),
        "trace_drop_count": sum(int(sink.get("dropped", 0)) for sink in trace_summary["sinks"].values()),
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
        "queue_wait": queue_wait,
        "hint_service": hint,
        "max_queue_depth": int(async_summary["max_queue_depth"]),
        "enqueue_queue_op_mean_ns": int(async_summary["enqueue_queue_op_mean_ns"]),
        "enqueue_queue_op_max_ns": int(async_summary["enqueue_queue_op_max_ns"]),
        "lock_hold": queue_global["mutex_hold_ns"],
        "lock_acquire_wait": queue_global["mutex_acquire_wait_ns"],
        "queue_selection_or_scan": queue_global["queue_scan_ns"],
        "scanned_candidates": queue_global["queue_scan_candidates"],
        "reserved": {name: reserved.get(name) for name in (
            "reserved_trigger_count",
            "reserved_due_count",
            "reserved_selected_count",
            "active_winner_changed_count",
            "reserved_same_as_legacy_head_count",
            "hard_urgent_override_count",
            "hard_urgent_safety_violation",
            "stale_handle_count",
            "full_store_scan_count",
            "invariant_error_count",
            "store_index_registry_bytes_conserved",
            "final_queue_empty",
            "insert_count",
            "erase_count",
            "selection_count",
            "legacy_heap_sift_count",
            "aging_heap_sift_count",
            "enqueue_index_op_mean_ns",
            "enqueue_index_op_max_ns",
            "dequeue_index_op_mean_ns",
            "dequeue_index_op_max_ns",
        )},
        "cgroup": {
            "path": before["path"],
            "memory.max": before["memory.max"],
            "memory.swap.max": before["memory.swap.max"],
            "memory.events_delta": events_delta,
        },
        "artifact_sha256": {
            name: sha256_file(run_dir / name)
            for name in (
                "memory_trace.jsonl",
                "expert_trace.jsonl",
                "summary.json",
                "process_metrics.json",
                "run_manifest.json",
                "output.sha256",
                "m6c_active_scope.json",
            )
        },
    }
    return result


def scope_child(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / args.run_id
    before = cgroup_snapshot()
    if before["memory.max"] != MEMORY_MAX or before["memory.swap.max"] != SWAP_MAX:
        raise GateError("effective delegated cgroup limits do not match 7 GiB / 2 GiB")
    env = os.environ.copy()
    env.update(common_environment(args.run_id, args.workers, args.configuration, args.repeat, args.order))
    env["TRACE_BASE_DIR"] = str(output_root)
    env["TRACE_OUT_DIR"] = str(run_dir)
    started = utc_now()
    completed = subprocess.run(
        ["bash", str(PIPELINE)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pipeline_log = completed.stdout
    # The pipeline has closed llama-cli and every Trace artifact at this point.
    after = cgroup_snapshot()
    scope = {
        "schema_version": "m6c-active-ab-scope-v1",
        "run_id": args.run_id,
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "pipeline_exit_code": completed.returncode,
        "before": before,
        "after": after,
        "delta": snapshot_delta(before, after),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_text(run_dir / "pipeline.log", pipeline_log)
    write_json(run_dir / "m6c_active_scope.json", scope)
    if completed.returncode != 0:
        raise GateError(f"pipeline failed with exit code {completed.returncode}")
    result = validate_and_measure_run(run_dir, args.configuration, args.workers)
    write_json(run_dir / "run_result.json", result)
    return 0


def matrix_plan() -> list[dict[str, Any]]:
    # Blocked alternating order, frozen before the first inference.
    pairs = {
        2: (("B0", "B1"), ("B1", "B0"), ("B0", "B1")),
        4: (("B1", "B0"), ("B0", "B1"), ("B1", "B0")),
    }
    plan: list[dict[str, Any]] = []
    order = 1
    for workers in (2, 4):
        for repeat, configurations in enumerate(pairs[workers], 1):
            for config in configurations:
                run_id = f"m6c_active_ab_20260805_v3_p{order:02d}_w{workers}_r{repeat}_{config}"
                plan.append({
                    "order": order,
                    "run_id": run_id,
                    "workers": workers,
                    "repeat": repeat,
                    "configuration": config,
                    "scope_unit": f"m6c-act-v3-p{order:02d}-w{workers}-r{repeat}-{config.lower()}",
                })
                order += 1
    return plan


def run_in_scope(output_root: Path, entry: dict[str, Any]) -> None:
    command = [
        "systemd-run", "--user", "--scope", "--collect", "--quiet",
        f"--unit={entry['scope_unit']}",
        "-p", f"MemoryMax={MEMORY_MAX}",
        "-p", f"MemorySwapMax={SWAP_MAX}",
        "taskset", "-c", "0-7",
        sys.executable, str(Path(__file__).resolve()),
        "--scope-child",
        "--output-root", str(output_root),
        "--run-id", entry["run_id"],
        "--configuration", entry["configuration"],
        "--workers", str(entry["workers"]),
        "--repeat", str(entry["repeat"]),
        "--order", str(entry["order"]),
    ]
    log_path = output_root / f"{entry['run_id']}_launcher.log"
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    write_text(log_path, completed.stdout)
    if completed.returncode != 0:
        raise GateError(f"{entry['run_id']}: delegated scope launcher failed ({completed.returncode})")


def median_cell(results: list[dict[str, Any]], workers: int, config: str) -> dict[str, Any]:
    selected = [row for row in results if row["workers"] == workers and row["configuration"] == config]
    scalar_paths = {
        "wall_time_s": lambda row: row["wall_time_s"],
        "prefill_latency_total_ms": lambda row: row["prefill_latency_total_ms"],
        "decode_latency_mean_ms": lambda row: row["decode_latency_mean_ms"],
        "throughput": lambda row: row["decode_throughput_tokens_per_s"],
        "major_faults": lambda row: row["major_faults"],
        "minor_faults": lambda row: row["minor_faults"],
        "rss_peak_kb": lambda row: row["rss_peak_kb"],
        "swap_peak_bytes": lambda row: row["swap_peak_bytes"],
        "psi_full_delta": lambda row: row["psi_full_total_delta_us"],
        "queue_wait_p95_ns": lambda row: row["queue_wait"]["p95_ns"],
        "queue_wait_p99_ns": lambda row: row["queue_wait"]["p99_ns"],
        "queue_wait_max_ns": lambda row: row["queue_wait"]["max_ns"],
        "lock_hold_mean_ns": lambda row: row["lock_hold"]["mean"],
    }
    return {
        "workers": workers,
        "configuration": config,
        "n": len(selected),
        "median": {name: statistics.median(fn(row) for row in selected) for name, fn in scalar_paths.items()},
        "run_ids": [row["run_id"] for row in selected],
    }


def pair_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["workers"], int(row["run_id"].split("_r", 1)[1].split("_", 1)[0]), row["configuration"]): row for row in results}
    output: list[dict[str, Any]] = []
    for workers in (2, 4):
        for repeat in (1, 2, 3):
            b0 = indexed[(workers, repeat, "B0")]
            b1 = indexed[(workers, repeat, "B1")]
            output.append({
                "workers": workers,
                "repeat": repeat,
                "B0_run_id": b0["run_id"],
                "B1_run_id": b1["run_id"],
                "B1_minus_B0": {
                    "wall_time_s": b1["wall_time_s"] - b0["wall_time_s"],
                    "prefill_latency_total_ms": b1["prefill_latency_total_ms"] - b0["prefill_latency_total_ms"],
                    "decode_latency_mean_ms": b1["decode_latency_mean_ms"] - b0["decode_latency_mean_ms"],
                    "major_faults": b1["major_faults"] - b0["major_faults"],
                    "minor_faults": b1["minor_faults"] - b0["minor_faults"],
                    "rss_peak_kb": b1["rss_peak_kb"] - b0["rss_peak_kb"],
                    "swap_peak_bytes": b1["swap_peak_bytes"] - b0["swap_peak_bytes"],
                    "queue_wait_p95_ns": b1["queue_wait"]["p95_ns"] - b0["queue_wait"]["p95_ns"],
                    "queue_wait_p99_ns": b1["queue_wait"]["p99_ns"] - b0["queue_wait"]["p99_ns"],
                    "queue_wait_max_ns": b1["queue_wait"]["max_ns"] - b0["queue_wait"]["max_ns"],
                },
            })
    return output


def decide_enum(results: list[dict[str, Any]], cells: list[dict[str, Any]]) -> tuple[str, list[str]]:
    b1 = [row for row in results if row["configuration"] == "B1"]
    safety_fields = ("hard_urgent_safety_violation", "stale_handle_count", "full_store_scan_count", "invariant_error_count")
    if any(int(row["reserved"].get(name) or 0) != 0 for row in b1 for name in safety_fields):
        return "ACTIVE_RESERVED_ROUTE_REGRESSION", ["Active safety or invariant counter is nonzero"]
    changed = sum(int(row["reserved"]["active_winner_changed_count"]) for row in b1)
    if changed == 0:
        return "ACTIVE_RESERVED_ROUTE_NO_OP", ["active_winner_changed_count is zero in every B1 Run"]
    cell = {(row["workers"], row["configuration"]): row["median"] for row in cells}
    tail_improves = all(
        cell[(workers, "B1")][metric] < cell[(workers, "B0")][metric]
        for workers in (2, 4)
        for metric in ("queue_wait_p95_ns", "queue_wait_p99_ns", "queue_wait_max_ns")
    )
    latency_ok = all(
        cell[(workers, "B1")][metric] <= cell[(workers, "B0")][metric] * 1.05
        for workers in (2, 4)
        for metric in ("wall_time_s", "prefill_latency_total_ms", "decode_latency_mean_ms")
    )
    system_ok = all(
        cell[(workers, "B1")][metric] <= max(1.0, cell[(workers, "B0")][metric]) * 1.20
        for workers in (2, 4)
        for metric in ("major_faults", "rss_peak_kb", "swap_peak_bytes")
    )
    if tail_improves and latency_ok and system_ok:
        return "ACTIVE_RESERVED_ROUTE_HAS_SIGNAL", ["winner changed and both worker strata satisfy the preregistered directional screen"]
    obvious_regression = any(
        cell[(workers, "B1")][metric] > cell[(workers, "B0")][metric] * 1.20
        for workers in (2, 4)
        for metric in ("wall_time_s", "prefill_latency_total_ms", "decode_latency_mean_ms", "rss_peak_kb")
    )
    if obvious_regression:
        return "ACTIVE_RESERVED_ROUTE_REGRESSION", ["winner changed but a primary physical metric regressed by more than 20%"]
    return "ACTIVE_RESERVED_ROUTE_NO_BENEFIT", ["winner changed without consistent queue-tail improvement across workers=2/4"]


def make_report(output_root: Path, results: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    cells = [median_cell(results, workers, config) for workers in (2, 4) for config in ("B0", "B1")]
    pairs = pair_results(results)
    final_enum, reasons = decide_enum(results, cells)
    return {
        "schema_version": "m6c-active-ab-report-v1",
        "stage": "minimal_real_active_ab_not_formal_n8",
        "created_at_utc": utc_now(),
        "physical_system_reexecuted": True,
        "performance_claim": False,
        "formal_n8": False,
        "protocol_sha256": sha256_file(output_root / "preregistration.json"),
        "binary_sha256": protocol["binary_sha256"],
        "source_tree_dirty": True,
        "run_count": len(results),
        "raw_runs": results,
        "paired_results": pairs,
        "cell_medians": cells,
        "active_winner_changed_total": sum(
            int(row["reserved"]["active_winner_changed_count"])
            for row in results if row["configuration"] == "B1"
        ),
        "correctness_safety_invariants": {
            "all_runs_valid": all(row["valid"] for row in results),
            "output_hash_consistent": len({row["output_sha256"] for row in results}) == 1,
            "trace_drop_total": sum(row["trace_drop_count"] for row in results),
            "hard_urgent_safety_violation_total": sum(
                int(row["reserved"].get("hard_urgent_safety_violation") or 0) for row in results
            ),
            "stale_handle_total": sum(int(row["reserved"].get("stale_handle_count") or 0) for row in results),
            "full_store_scan_total": sum(int(row["reserved"].get("full_store_scan_count") or 0) for row in results),
            "invariant_error_total": sum(int(row["reserved"].get("invariant_error_count") or 0) for row in results),
            "all_final_queues_empty": all(row["reserved"].get("final_queue_empty") for row in results),
        },
        "final_enum": final_enum,
        "decision_reasons": reasons,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# M6C Reserved-Service Minimal Active A/B Report",
        "",
        "> 这是 4-token、每格 N=3 的工程验证，不是正式 N=8，不形成正式性能结论。",
        "",
        f"最终枚举：`{report['final_enum']}`",
        "",
        f"B1 `active_winner_changed_count` 合计：{report['active_winner_changed_total']}",
        "",
        "## 12 Run 原始值",
        "",
        "| Run | w | cfg | wall s | prefill ms | decode mean ms | majflt | minflt | RSS KiB | swap peak B | qwait p95 ms | p99 ms | max ms | trigger/due/selected/changed |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["raw_runs"]:
        reserved = row["reserved"]
        lines.append(
            f"| {row['run_id']} | {row['workers']} | {row['configuration']} | "
            f"{row['wall_time_s']:.2f} | {row['prefill_latency_total_ms']:.3f} | "
            f"{row['decode_latency_mean_ms']:.3f} | {row['major_faults']} | {row['minor_faults']} | "
            f"{row['rss_peak_kb']} | {row['swap_peak_bytes']} | "
            f"{row['queue_wait']['p95_ns']/1e6:.3f} | {row['queue_wait']['p99_ns']/1e6:.3f} | "
            f"{row['queue_wait']['max_ns']/1e6:.3f} | "
            f"{reserved.get('reserved_trigger_count')}/{reserved.get('reserved_due_count')}/"
            f"{reserved.get('reserved_selected_count')}/{reserved.get('active_winner_changed_count')} |"
        )
    lines.extend(["", "## workers × configuration 中位数", ""])
    for cell in report["cell_medians"]:
        lines.append(f"- workers={cell['workers']} {cell['configuration']}: `{json.dumps(cell['median'], sort_keys=True)}`")
    lines.extend(["", "## 正确性与安全", "", f"```json\n{json.dumps(report['correctness_safety_invariants'], indent=2, sort_keys=True)}\n```", ""])
    return "\n".join(lines)


def artifact_index(output_root: Path, run_ids: list[str]) -> dict[str, Any]:
    paths = [
        output_root / "preregistration.json",
        output_root / "smoke_result.json",
        output_root / "m6c_active_ab_report.json",
        output_root / "m6c_active_ab_report.md",
    ]
    for run_id in run_ids:
        run = output_root / run_id
        paths.extend(run / name for name in (
            "run_result.json",
            "m6c_active_scope.json",
            "run_manifest.json",
            "process_metrics.json",
            "summary.json",
            "output.sha256",
            "memory_trace.jsonl",
            "expert_trace.jsonl",
        ))
    return {
        "schema_version": "m6c-active-ab-artifact-index-v1",
        "files": [
            {
                "path": str(path.relative_to(output_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }


def validate_artifact_index(output_root: Path, index: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    for record in index["files"]:
        path = output_root / record["path"]
        if not path.is_file():
            errors.append(f"missing: {record['path']}")
            continue
        if path.stat().st_size != record["size_bytes"]:
            errors.append(f"size mismatch: {record['path']}")
        if sha256_file(path) != record["sha256"]:
            errors.append(f"SHA mismatch: {record['path']}")
    return {
        "schema_version": "m6c-active-ab-artifact-validation-v1",
        "validated_at_utc": utc_now(),
        "file_count": len(index["files"]),
        "passed": not errors,
        "errors": errors,
    }


def execute(output_root: Path) -> int:
    if output_root.exists():
        raise GateError(f"output directory already exists: {output_root}")
    if sha256_file(MODEL) != EXPECTED_MODEL_SHA256:
        raise GateError("model SHA mismatch")
    helper = LLAMA_ROOT / "trace" / "prepare_model_cache.py"
    if sha256_file(helper) != EXPECTED_HELPER_SHA256:
        raise GateError("cold-cache helper SHA mismatch")
    reference = read_json(REFERENCE_MANIFEST)
    if reference["frozen"]["prompt_sha256"] != EXPECTED_PROMPT_SHA256:
        raise GateError("A.3 prompt authority mismatch")
    output_root.mkdir(parents=True)
    plan = matrix_plan()
    smoke = {
        "order": 0,
        "run_id": SMOKE_EVIDENCE.name,
        "workers": 2,
        "repeat": 0,
        "configuration": "B1",
        "scope_unit": None,
        "evidence_path": str(SMOKE_EVIDENCE),
        "reuse_reason": "physical smoke already completed; v2 failed only because raw stdout includes the expected new build banner",
    }
    protocol = {
        "schema_version": "m6c-active-ab-preregistration-v1",
        "created_at_utc": utc_now(),
        "purpose": "minimal_physical_active_ab_not_formal_n8",
        "formal_n8": False,
        "performance_claim": False,
        "binary_path": str(BINARY),
        "binary_sha256": sha256_file(BINARY),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "pipeline_sha256": sha256_file(PIPELINE),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "prompt_sha256": EXPECTED_PROMPT_SHA256,
        "a3_raw_output_sha256_reference": EXPECTED_OUTPUT_SHA256,
        "canonical_output_sha256_expected": EXPECTED_CANONICAL_OUTPUT_SHA256,
        "cgroup": {"memory.max": MEMORY_MAX, "memory.swap.max": SWAP_MAX, "new_scope_per_run": True},
        "cpu_affinity": "0-7",
        "workload": {"tokens": 4, "threads": 8, "gpu_layers": 0, "batch": 512, "context": 2048, "seed": 1234, "temperature": 0.0},
        "B0": {"priority_mode": "deadline_score", "reserved_service_active": False},
        "B1": {
            "priority_mode": "deadline_score",
            "reserved_service_active": True,
            "R": 1,
            "D": 8,
            "eligibility_age_ns": 41000000,
            "hard_urgent_guard_ns": 0,
            "eligibility_rule": "AGE_GATED_ALL",
            "debt_policy": "single_pending_latch",
            "reset_policy": "reset_when_no_eligible",
            "reserved_winner": "oldest_eligible",
        },
        "smoke": smoke,
        "matrix": plan,
        "decision_screen": {
            "no_op": "all B1 active_winner_changed_count == 0",
            "signal": "winner change nonzero; p95/p99/max tail lower at workers 2 and 4; primary latency <=5% worse; Fault/RSS/Swap <=20% worse",
            "regression": "safety/invariant error or a primary physical median >20% worse",
        },
    }
    write_json(output_root / "preregistration.json", protocol)

    smoke_result = validate_and_measure_run(SMOKE_EVIDENCE, "B1", 2)
    smoke_result["reused_from_preserved_v2_evidence"] = True
    smoke_result["source_evidence_path"] = str(SMOKE_EVIDENCE)
    write_json(output_root / "smoke_result.json", smoke_result)
    # Smoke is a hard Gate; no matrix starts unless all validation above passes.

    results: list[dict[str, Any]] = []
    for entry in plan:
        run_in_scope(output_root, entry)
        result = validate_and_measure_run(output_root / entry["run_id"], entry["configuration"], entry["workers"])
        expected = read_json(output_root / entry["run_id"] / "run_result.json")
        if result != expected:
            raise GateError(f"{entry['run_id']}: independent deterministic artifact parse mismatch")
        results.append(result)

    report = make_report(output_root, results, protocol)
    write_json(output_root / "m6c_active_ab_report.json", report)
    write_text(output_root / "m6c_active_ab_report.md", markdown_report(report))
    all_run_ids = [entry["run_id"] for entry in plan]
    index = artifact_index(output_root, all_run_ids)
    write_json(output_root / "artifact_index.json", index)
    validation = validate_artifact_index(output_root, read_json(output_root / "artifact_index.json"))
    write_json(output_root / "artifact_validation.json", validation)
    if not validation["passed"]:
        raise GateError("independent artifact validation failed")
    authority = {
        "schema_version": "m6c-active-ab-authority-v1",
        "final_enum": report["final_enum"],
        "report_sha256": sha256_file(output_root / "m6c_active_ab_report.json"),
        "artifact_index_sha256": sha256_file(output_root / "artifact_index.json"),
        "artifact_validation_sha256": sha256_file(output_root / "artifact_validation.json"),
        "artifact_validation_passed": True,
    }
    write_json(output_root / "authority.json", authority)
    write_text(output_root / "authority.sha256", sha256_file(output_root / "authority.json") + "\n")
    print(json.dumps(authority, indent=2, sort_keys=True))
    print(f"authority_sha256={read_text(output_root / 'authority.sha256')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--scope-child", action="store_true")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id")
    parser.add_argument("--configuration", choices=("B0", "B1"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--order", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.scope_child:
            if not args.run_id or args.configuration is None or args.workers not in (2, 4):
                raise GateError("scope child arguments are incomplete")
            return scope_child(args)
        if args.execute:
            return execute(Path(args.output_root).resolve())
        parser.error("choose --execute or --scope-child")
    except GateError as exc:
        print(f"M6C Active A/B Gate failure: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
