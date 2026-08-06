"""A.3 Evidence hash validation, reconstructability audit, and S0 replay."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, TextIO

from .model import ModelInvariantError, OfflineQueue, TaskInput, TaskSpec, parse_f64_bits


COUNTERFACTUAL_DECLARATION = {
    "counterfactual_type": "fixed_arrival_fixed_service_slot_policy_replay",
    "physical_system_reexecuted": False,
    "performance_claim": False,
}


class ReconstructabilityStatus(str, Enum):
    RECONSTRUCTABLE = "RECONSTRUCTABLE"
    ORDER_UNAVAILABLE = "ORDER_UNAVAILABLE"
    MISSING_ROUTE_SCORE_BITS = "MISSING_ROUTE_SCORE_BITS"
    TASK_CONSERVATION_FAILURE = "TASK_CONSERVATION_FAILURE"
    SLOT_STREAM_UNAVAILABLE = "SLOT_STREAM_UNAVAILABLE"
    HASH_MISMATCH = "HASH_MISMATCH"
    UNSUPPORTED_EVENT = "UNSUPPORTED_EVENT"


@dataclass(frozen=True)
class ServiceSlot:
    decision_id: int
    batch_id: int
    batch_slot: int
    worker_id: int
    decision_ts_ns: int
    winner_task_id: int
    queue_depth_before: int
    priority_mode: str
    diagnostic_event: str
    diagnostic_decision_ts_ns: int
    diagnostic_candidate_count: int | None


@dataclass
class EvidenceRun:
    run_id: str
    evidence_dir: Path
    configuration_id: str
    workers: int
    repeat_index: int
    priority_mode: str
    capacity: int
    tasks: list[TaskSpec]
    slots: list[ServiceSlot]
    lifecycle_counts: dict[str, int]
    terminal_states: dict[int, str]
    prior_validation: dict[str, Any]
    unavailable_fields: list[str] = field(default_factory=list)


@dataclass
class ReconstructabilityResult:
    run_id: str
    status: ReconstructabilityStatus
    reasons: list[str]
    checks: dict[str, bool]
    counts: dict[str, int]
    unavailable_fields: list[str]
    run: EvidenceRun | None = None

    def machine_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "reasons": self.reasons,
            "checks": self.checks,
            "counts": self.counts,
            "unavailable_fields": self.unavailable_fields,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_evidence_index(repo_root: Path, evidence_index_path: Path) -> dict[str, Any]:
    index = load_json(evidence_index_path)
    artifacts = index.get("source_artifacts")
    if not isinstance(artifacts, list):
        return {
            "passed": False,
            "checked": 0,
            "mismatches": [{"path": str(evidence_index_path), "reason": "source_artifacts_missing"}],
        }
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for record in artifacts:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            mismatches.append({"path": None, "reason": "malformed_index_record"})
            continue
        relative = Path(record["path"])
        path = repo_root / relative
        if not path.is_file():
            mismatches.append({"path": record["path"], "reason": "missing"})
            continue
        actual_size = path.stat().st_size
        expected_size = record.get("size_bytes")
        if not isinstance(expected_size, int) or actual_size != expected_size:
            mismatches.append({
                "path": record["path"],
                "reason": "size_mismatch",
                "expected": expected_size,
                "actual": actual_size,
            })
            continue
        actual_hash = sha256_file(path)
        expected_hash = record.get("sha256")
        if actual_hash != expected_hash:
            mismatches.append({
                "path": record["path"],
                "reason": "sha256_mismatch",
                "expected": expected_hash,
                "actual": actual_hash,
            })
            continue
        checked += 1
    return {
        "schema_version": index.get("schema_version"),
        "passed": not mismatches and checked == len(artifacts),
        "checked": checked,
        "indexed": len(artifacts),
        "mismatches": mismatches,
        "source_run_membership": index.get("source_run_membership"),
        "evidence_index_path": str(evidence_index_path),
        "evidence_index_sha256": sha256_file(evidence_index_path),
    }


TRANSITIONS: dict[str | None, dict[str, str]] = {
    None: {"CREATE": "CREATED"},
    "CREATED": {"ADMIT": "ADMITTED", "REJECT": "REJECTED"},
    "ADMITTED": {"ENQUEUE": "ENQUEUED", "ISSUE": "ISSUED", "CANCEL": "CANCELLED"},
    "ENQUEUED": {"DEQUEUE": "DEQUEUED"},
    "DEQUEUED": {"ISSUE": "ISSUED", "CANCEL": "CANCELLED"},
    "REJECTED": {},
    "ISSUED": {},
    "CANCELLED": {},
}


def _trace_complete(summary: dict[str, Any]) -> bool:
    sinks = summary.get("sinks")
    if not isinstance(sinks, dict) or not sinks:
        return False
    for value in sinks.values():
        if not isinstance(value, dict):
            return False
        if value.get("enabled"):
            if int(value.get("enqueued", -1)) != int(value.get("written", -2)):
                return False
            if int(value.get("dropped", -1)) != 0:
                return False
    return True


def _required_int(record: dict[str, Any], field_name: str) -> int:
    value = record.get(field_name)
    if not isinstance(value, int):
        raise ValueError(f"missing integer field {field_name}")
    return value


def _status_for_failures(failures: Counter[str]) -> ReconstructabilityStatus:
    precedence = (
        ("hash", ReconstructabilityStatus.HASH_MISMATCH),
        ("route_bits", ReconstructabilityStatus.MISSING_ROUTE_SCORE_BITS),
        ("unsupported", ReconstructabilityStatus.UNSUPPORTED_EVENT),
        ("slot", ReconstructabilityStatus.SLOT_STREAM_UNAVAILABLE),
        ("order", ReconstructabilityStatus.ORDER_UNAVAILABLE),
        ("conservation", ReconstructabilityStatus.TASK_CONSERVATION_FAILURE),
    )
    for name, status in precedence:
        if failures[name]:
            return status
    return ReconstructabilityStatus.RECONSTRUCTABLE


def audit_run(
    run_entry: dict[str, Any],
    *,
    global_hash_passed: bool,
) -> ReconstructabilityResult:
    run_id = str(run_entry.get("run_id", ""))
    evidence_dir = Path(str(run_entry.get("evidence_dir", "")))
    slot_meta = run_entry.get("slot") if isinstance(run_entry.get("slot"), dict) else {}
    failures: Counter[str] = Counter()
    reasons: list[str] = []
    checks: dict[str, bool] = {}
    unavailable_fields = [
        "queue_op_id (A.3 predates M6C queue mutation IDs)",
        "M6C eligibility age (no formal parameter selected)",
        "M6C hard-urgent guard (no formal parameter selected)",
    ]

    def fail(kind: str, reason: str) -> None:
        failures[kind] += 1
        reasons.append(reason)

    if not global_hash_passed:
        fail("hash", "global Evidence index/hash validation failed")
    required_files = (
        "run_manifest.json",
        "summary.json",
        "memory_trace.jsonl",
        "m6b2a3_validation.json",
    )
    for name in required_files:
        if not (evidence_dir / name).is_file():
            fail("hash", f"required indexed source missing: {name}")
    if failures["hash"]:
        return ReconstructabilityResult(
            run_id,
            _status_for_failures(failures),
            reasons,
            {"evidence_hash": False},
            {},
            unavailable_fields,
        )

    manifest = load_json(evidence_dir / "run_manifest.json")
    summary = load_json(evidence_dir / "summary.json")
    prior_validation = load_json(evidence_dir / "m6b2a3_validation.json")
    manifest_run_id = manifest.get("run_name")
    checks["run_id"] = manifest_run_id == run_id == evidence_dir.name
    if not checks["run_id"]:
        fail("conservation", "Run ID differs across index, manifest, and directory")

    environment = manifest.get("environment") if isinstance(manifest.get("environment"), dict) else {}
    priority_mode = str(environment.get("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE", ""))
    checks["priority_mode_supported"] = priority_mode in {"deadline_score", "max_wait_protection"}
    if not checks["priority_mode_supported"]:
        fail("unsupported", f"unsupported source priority mode: {priority_mode!r}")
    try:
        capacity = int(environment.get("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE", ""))
    except (TypeError, ValueError):
        capacity = 0
    if capacity <= 0:
        fail("unsupported", "bounded queue capacity is unavailable")
    checks["zero_drop"] = _trace_complete(summary)
    if not checks["zero_drop"]:
        fail("conservation", "Trace sink zero-drop Gate failed")
    checks["prior_validation"] = bool(prior_validation.get("passed"))
    checks["prior_winner_replay"] = bool(
        prior_validation.get("selection_winner_replay", {}).get("passed")
    )
    if not checks["prior_validation"] or not checks["prior_winner_replay"]:
        fail("conservation", "frozen A.3 validation or current-policy winner replay failed")

    states: dict[int, str] = {}
    lifecycle_counts: Counter[str] = Counter()
    creates: Counter[int] = Counter()
    enqueues: dict[int, TaskSpec] = {}
    score_bits_seen: dict[int, set[str]] = {}
    sequences: dict[int, int] = {}
    overhead_records: list[dict[str, Any]] = []
    priority_records: dict[int, dict[str, Any]] = {}
    max_wait_records: dict[int, dict[str, Any]] = {}
    unsupported_queued_cancel = 0
    timestamp_regressions = 0
    last_task_ts: dict[int, int] = {}
    invalid_transitions = 0
    state_mismatches = 0

    with (evidence_dir / "memory_trace.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            event_name = record.get("event")
            if event_name == "EXPERT_QUEUE_OVERHEAD_SELECTION":
                overhead_records.append(record)
                continue
            if event_name == "EXPERT_PRIORITY_SELECTION":
                task_id = record.get("task_id")
                if isinstance(task_id, int):
                    if task_id in priority_records:
                        fail("slot", f"duplicate priority diagnostic for task {task_id}")
                    priority_records[task_id] = record
                continue
            if event_name == "EXPERT_MAX_WAIT_SELECTION":
                task_id = record.get("task_id")
                if isinstance(task_id, int):
                    if task_id in max_wait_records:
                        fail("slot", f"duplicate max-wait diagnostic for task {task_id}")
                    max_wait_records[task_id] = record
                continue
            if event_name != "EXPERT_TASK":
                continue
            task_id = record.get("task_id")
            if not isinstance(task_id, int) or task_id <= 0:
                fail("conservation", "invalid Task ID in lifecycle")
                continue
            lifecycle = str(record.get("lifecycle_event", ""))
            lifecycle_counts[lifecycle] += 1
            creates[task_id] += int(lifecycle == "CREATE")
            previous = states.get(task_id)
            next_state = TRANSITIONS.get(previous, {}).get(lifecycle)
            if next_state is None:
                invalid_transitions += 1
                if previous == "ENQUEUED" and lifecycle == "CANCEL":
                    unsupported_queued_cancel += 1
            else:
                states[task_id] = next_state
                state_mismatches += int(record.get("state") != next_state)
            timestamp = record.get("ts_ns")
            if not isinstance(timestamp, int):
                fail("conservation", f"Task {task_id} missing lifecycle timestamp")
                timestamp = 0
            if task_id in last_task_ts and timestamp < last_task_ts[task_id]:
                timestamp_regressions += 1
            last_task_ts[task_id] = max(timestamp, last_task_ts.get(task_id, 0))
            bits = record.get("score_f64_bits")
            if isinstance(bits, str):
                score_bits_seen.setdefault(task_id, set()).add(bits.lower())
            if lifecycle == "ENQUEUE":
                if task_id in enqueues:
                    fail("conservation", f"duplicate ENQUEUE for Task {task_id}")
                    continue
                if not isinstance(bits, str):
                    fail("route_bits", f"Task {task_id} has no route-score F64 bits")
                    continue
                try:
                    parse_f64_bits(bits)
                    spec = TaskSpec(
                        run_id=run_id,
                        task_id=task_id,
                        deadline_ts_ns=_required_int(record, "deadline_ts_ns"),
                        route_score_f64_bits=bits.lower(),
                        sequence=_required_int(record, "sequence"),
                        enqueued_ts_ns=_required_int(record, "enqueued_ts_ns"),
                        nbytes=_required_int(record, "nbytes"),
                        stage=str(record.get("stage")) if record.get("stage") is not None else None,
                        phase=str(record.get("phase")) if record.get("phase") is not None else None,
                        layer=int(record["layer"]) if isinstance(record.get("layer"), int) else None,
                    )
                except (ValueError, ModelInvariantError) as exc:
                    fail("route_bits" if "route score" in str(exc) else "conservation", str(exc))
                    continue
                if spec.enqueued_ts_ns <= 0 or spec.enqueued_ts_ns != timestamp:
                    fail("order", f"Task {task_id} lacks canonical ENQUEUE timestamp equality")
                if spec.sequence in sequences:
                    fail("order", f"duplicate sequence {spec.sequence}")
                sequences[spec.sequence] = task_id
                enqueues[task_id] = spec

    if unsupported_queued_cancel:
        fail("unsupported", "A queued CANCEL lacks an authoritative queue mutation order")
    terminal = {"REJECTED", "ISSUED", "CANCELLED"}
    duplicate_creates = sum(max(0, count - 1) for count in creates.values())
    incomplete = sum(state not in terminal for state in states.values())
    if duplicate_creates or invalid_transitions or state_mismatches or timestamp_regressions or incomplete:
        fail("conservation", "lifecycle state machine or exact-once terminal Gate failed")
    if any(len(bits) != 1 for bits in score_bits_seen.values()):
        fail("route_bits", "route-score F64 bits changed within a Task lifecycle")

    expected_diagnostic = priority_records if priority_mode == "deadline_score" else max_wait_records
    slots: list[ServiceSlot] = []
    decision_ids: set[int] = set()
    actual_winners: set[int] = set()
    for record in overhead_records:
        try:
            decision_id = _required_int(record, "decision_id")
            batch_id = _required_int(record, "batch_id")
            batch_slot = _required_int(record, "batch_slot")
            worker_id = _required_int(record, "worker_id")
            decision_ts_ns = _required_int(record, "batch_decision_ts_ns")
            winner_task_id = _required_int(record, "winner_task_id")
            queue_depth_before = _required_int(record, "queue_depth_before")
        except ValueError as exc:
            fail("slot", str(exc))
            continue
        diagnostic = expected_diagnostic.get(winner_task_id)
        if diagnostic is None:
            fail("slot", f"winner {winner_task_id} lacks current-policy diagnostic event")
            continue
        diagnostic_event = str(diagnostic.get("event", ""))
        diagnostic_time = diagnostic.get("decision_ts_ns")
        if not isinstance(diagnostic_time, int):
            fail("slot", f"winner {winner_task_id} lacks diagnostic decision time")
            continue
        if decision_id in decision_ids:
            fail("slot", f"duplicate decision_id {decision_id}")
        decision_ids.add(decision_id)
        if winner_task_id in actual_winners:
            fail("conservation", f"Task {winner_task_id} selected twice")
        actual_winners.add(winner_task_id)
        candidate_count = diagnostic.get("candidate_count")
        if candidate_count is not None and not isinstance(candidate_count, int):
            fail("slot", f"invalid candidate count for Task {winner_task_id}")
            candidate_count = None
        slots.append(ServiceSlot(
            decision_id,
            batch_id,
            batch_slot,
            worker_id,
            decision_ts_ns,
            winner_task_id,
            queue_depth_before,
            priority_mode,
            diagnostic_event,
            diagnostic_time,
            candidate_count,
        ))

    slots.sort(key=lambda item: item.decision_id)
    if not slots or [slot.decision_id for slot in slots] != list(range(len(slots))):
        fail("slot", "decision IDs are missing, duplicated, or not dense from zero")
    if any(slots[index].decision_ts_ns > slots[index + 1].decision_ts_ns for index in range(max(0, len(slots) - 1))):
        fail("order", "decision time regressed despite explicit decision linearization")

    enqueue_times = {task.enqueued_ts_ns for task in enqueues.values()}
    decision_times = {slot.decision_ts_ns for slot in slots}
    if enqueue_times.intersection(decision_times):
        fail("order", "an ENQUEUE and decision share a timestamp without queue_op_id")

    arrivals = sorted(enqueues.values(), key=lambda task: (task.enqueued_ts_ns, task.sequence))
    arrival_index = 0
    actual_live: set[int] = set()
    for index, service_slot in enumerate(slots):
        while arrival_index < len(arrivals) and arrivals[arrival_index].enqueued_ts_ns <= service_slot.decision_ts_ns:
            actual_live.add(arrivals[arrival_index].task_id)
            arrival_index += 1
        if len(actual_live) != service_slot.queue_depth_before:
            fail("order", f"queue depth cannot be reconstructed at decision {service_slot.decision_id}")
            break
        if service_slot.winner_task_id not in actual_live:
            fail("order", f"actual winner not live at decision {service_slot.decision_id}")
            break
        if service_slot.diagnostic_candidate_count is not None and service_slot.diagnostic_candidate_count != len(actual_live):
            fail("order", f"diagnostic candidate count mismatch at decision {service_slot.decision_id}")
            break
        actual_live.remove(service_slot.winner_task_id)

    enqueue_count = len(enqueues)
    selected_count = len(slots)
    checks["task_conservation"] = (
        enqueue_count == selected_count
        and not actual_live
        and arrival_index == len(arrivals)
        and set(enqueues) == actual_winners
        and lifecycle_counts["ENQUEUE"] == lifecycle_counts["DEQUEUE"]
        and lifecycle_counts["ENQUEUE"] == lifecycle_counts["ISSUE"] + lifecycle_counts["CANCEL"]
    )
    if not checks["task_conservation"]:
        fail("conservation", "Task/slot/dequeue/terminal conservation failed")
    checks["route_score_bits"] = not failures["route_bits"] and len(enqueues) == enqueue_count
    checks["decision_linearization"] = not failures["slot"] and not failures["order"]

    status = _status_for_failures(failures)
    counts = {
        "created": lifecycle_counts["CREATE"],
        "rejected": lifecycle_counts["REJECT"],
        "enqueued": enqueue_count,
        "dequeued": lifecycle_counts["DEQUEUE"],
        "issued": lifecycle_counts["ISSUE"],
        "cancelled": lifecycle_counts["CANCEL"],
        "service_slots": selected_count,
        "priority_diagnostics": len(priority_records),
        "max_wait_diagnostics": len(max_wait_records),
    }
    run: EvidenceRun | None = None
    if status == ReconstructabilityStatus.RECONSTRUCTABLE:
        run = EvidenceRun(
            run_id=run_id,
            evidence_dir=evidence_dir,
            configuration_id=str(slot_meta.get("configuration_id", "")),
            workers=int(slot_meta.get("workers", 0)),
            repeat_index=int(slot_meta.get("repeat_index", 0)),
            priority_mode=priority_mode,
            capacity=capacity,
            tasks=arrivals,
            slots=slots,
            lifecycle_counts=dict(lifecycle_counts),
            terminal_states=states,
            prior_validation=prior_validation,
            unavailable_fields=unavailable_fields,
        )
    return ReconstructabilityResult(
        run_id,
        status,
        reasons,
        checks,
        counts,
        unavailable_fields,
        run,
    )


def _legacy_reference_key(task: TaskSpec) -> tuple[bool, int, float, int]:
    score = parse_f64_bits(task.route_score_f64_bits)
    return (
        task.deadline_ts_ns == 0,
        task.deadline_ts_ns,
        -score,
        task.sequence,
    )


def replay_s0(run: EvidenceRun, decision_stream: TextIO | None = None) -> dict[str, Any]:
    queue = OfflineQueue(run.capacity)
    reference_heap: list[tuple[tuple[bool, int, float, int], int]] = []
    arrivals = run.tasks
    arrival_index = 0
    oracle_mismatches = 0
    runtime_mismatches = 0
    first_mismatch: dict[str, Any] | None = None
    selected_ids: set[int] = set()

    for service_slot in run.slots:
        while arrival_index < len(arrivals) and arrivals[arrival_index].enqueued_ts_ns <= service_slot.decision_ts_ns:
            spec = arrivals[arrival_index]
            queue.enqueue(spec)
            heapq.heappush(reference_heap, (_legacy_reference_key(spec), spec.task_id))
            arrival_index += 1
        if queue.store.live_count != service_slot.queue_depth_before:
            raise ModelInvariantError(
                f"replay queue depth mismatch at decision {service_slot.decision_id}"
            )
        if not reference_heap:
            raise ModelInvariantError("reference deadline_score heap is empty")
        _, expected_task_id = heapq.heappop(reference_heap)
        record = queue.select(
            decision_ts_ns=service_slot.decision_ts_ns,
            batch_id=service_slot.batch_id,
            batch_slot=service_slot.batch_slot,
            worker_id=service_slot.worker_id,
        )
        selected_task_id = int(record["selected_task"]["task_id"])
        oracle_mismatch = selected_task_id != expected_task_id
        runtime_comparison_available = run.priority_mode == "deadline_score"
        runtime_mismatch = (
            runtime_comparison_available
            and selected_task_id != service_slot.winner_task_id
        )
        oracle_mismatches += int(oracle_mismatch)
        runtime_mismatches += int(runtime_mismatch)
        if (oracle_mismatch or runtime_mismatch) and first_mismatch is None:
            first_mismatch = {
                "decision_id": service_slot.decision_id,
                "selected_task_id": selected_task_id,
                "oracle_task_id": expected_task_id,
                "runtime_task_id": service_slot.winner_task_id,
                "runtime_comparison_available": runtime_comparison_available,
            }
        if selected_task_id in selected_ids:
            raise ModelInvariantError("S0 selected a Task more than once")
        selected_ids.add(selected_task_id)
        queue.complete_selected(selected_task_id)

        replay_queue_op_id = record.pop("queue_op_id")
        record.update({
            "queue_op_id": None,
            "queue_op_id_availability_reason": "A.3 Evidence predates M6C queue mutation IDs",
            "replay_queue_op_id": replay_queue_op_id,
            "source_decision_id": service_slot.decision_id,
            "source_priority_mode": run.priority_mode,
            "source_actual_winner_task_id": service_slot.winner_task_id,
            "deadline_score_oracle_winner_task_id": expected_task_id,
            "s0_oracle_winner_mismatch": oracle_mismatch,
            "runtime_deadline_score_comparison_available": runtime_comparison_available,
            "runtime_deadline_score_winner_mismatch": runtime_mismatch if runtime_comparison_available else None,
            "oldest_eligible": {
                "available": False,
                "reason": "S0 has no approved eligibility-age parameter",
            },
            **COUNTERFACTUAL_DECLARATION,
        })
        if decision_stream is not None:
            decision_stream.write(json.dumps(record, sort_keys=True) + "\n")
        if oracle_mismatch or runtime_mismatch:
            break

    remaining_arrivals = len(arrivals) - arrival_index
    audit = queue.audit_invariants()
    stale = (
        queue.legacy_index.stale_handle_count + queue.aging_index.stale_handle_count
    )
    final_queue_empty = queue.store.live_count == 0 and not reference_heap
    selected_exact_once = len(selected_ids) == len(run.tasks)
    passed = (
        oracle_mismatches == 0
        and runtime_mismatches == 0
        and remaining_arrivals == 0
        and final_queue_empty
        and selected_exact_once
        and bool(audit["passed"])
        and stale == 0
        and queue.full_store_scan_count == 0
    )
    return {
        "run_id": run.run_id,
        "configuration_id": run.configuration_id,
        "workers": run.workers,
        "repeat_index": run.repeat_index,
        "source_priority_mode": run.priority_mode,
        "service_slots": len(run.slots),
        "s0_decisions_executed": queue.next_decision_id,
        "s0_oracle_winner_mismatches": oracle_mismatches,
        "runtime_deadline_score_comparison_available": run.priority_mode == "deadline_score",
        "runtime_deadline_score_winner_mismatches": runtime_mismatches if run.priority_mode == "deadline_score" else None,
        "first_mismatch": first_mismatch,
        "remaining_arrivals": remaining_arrivals,
        "final_queue_empty": final_queue_empty,
        "selected_exact_once": selected_exact_once,
        "full_store_scan_count": queue.full_store_scan_count,
        "stale_handle_count": stale,
        "operation_counters": queue.counters(),
        "invariants": audit,
        "passed": passed,
        "oracle_authority": (
            "independent heap using the frozen C++ deadline_score key and authoritative F64 bits; "
            "B0 additionally compares each decision to the observed runtime winner"
        ),
        **COUNTERFACTUAL_DECLARATION,
    }


def load_run_index(path: Path) -> list[dict[str, Any]]:
    value = load_json(path)
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) != 30:
        raise ValueError("A.3 explicit COMPLETE run index must contain exactly 30 Runs")
    if value.get("completed_valid_runs") != 30 or value.get("planned_valid_runs") != 30:
        raise ValueError("A.3 run index is not the complete 30-Run set")
    result: list[dict[str, Any]] = []
    for record in runs:
        if not isinstance(record, dict) or record.get("valid") is not True:
            raise ValueError("A.3 run index contains a non-valid source Run")
        result.append(record)
    return result
