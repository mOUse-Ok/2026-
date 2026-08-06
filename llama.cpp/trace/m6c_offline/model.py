"""Bounded Task store, eager dual indexed heaps, and shared S0/S1 transactions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import struct
from typing import Callable, Iterable

from .policy import (
    FixturePolicyConfig,
    PolicyState,
    PolicyTransition,
    UINT64_MAX,
    decide_reserved_service,
    saturating_add_u64,
)


class ModelInvariantError(RuntimeError):
    """Fail-closed error for store, handle, index, bytes, or lifecycle violations."""


@dataclass(frozen=True, order=True)
class Handle:
    slot_id: int
    generation: int


@dataclass(frozen=True)
class TaskSpec:
    run_id: str
    task_id: int
    deadline_ts_ns: int
    route_score_f64_bits: str
    sequence: int
    enqueued_ts_ns: int
    nbytes: int
    stage: str | None = None
    phase: str | None = None
    layer: int | None = None


@dataclass(frozen=True)
class TaskInput:
    run_id: str
    task_id: int
    slot_id: int
    generation: int
    deadline_ts_ns: int
    route_score_f64_bits: str
    sequence: int
    enqueued_ts_ns: int
    eligible_ts_ns: int | None
    nbytes: int
    stage: str | None = None
    phase: str | None = None
    layer: int | None = None

    @property
    def handle(self) -> Handle:
        return Handle(self.slot_id, self.generation)


def parse_f64_bits(bits: str) -> float:
    if not isinstance(bits, str) or len(bits) != 18 or not bits.startswith("0x"):
        raise ModelInvariantError("route score must be an exact 0x-prefixed 64-bit pattern")
    try:
        raw = int(bits[2:], 16)
    except ValueError as exc:
        raise ModelInvariantError("route score bit pattern is not hexadecimal") from exc
    value = struct.unpack(">d", raw.to_bytes(8, "big"))[0]
    if not math.isfinite(value):
        raise ModelInvariantError("non-finite route score is not a valid frozen queue key")
    return value


def normalize_f64_bits(bits: str) -> str:
    parse_f64_bits(bits)
    return f"0x{int(bits[2:], 16):016x}"


def legacy_higher(left: TaskInput, right: TaskInput) -> bool:
    """Exact Python transcription of compare_deadline_score from C++."""

    if left.deadline_ts_ns != right.deadline_ts_ns:
        if left.deadline_ts_ns == 0:
            return False
        if right.deadline_ts_ns == 0:
            return True
        return left.deadline_ts_ns < right.deadline_ts_ns
    left_score = parse_f64_bits(left.route_score_f64_bits)
    right_score = parse_f64_bits(right.route_score_f64_bits)
    if left_score != right_score:
        return left_score > right_score
    return left.sequence < right.sequence


def aging_higher(left: TaskInput, right: TaskInput) -> bool:
    return (
        left.enqueued_ts_ns,
        left.sequence,
        left.task_id,
    ) < (
        right.enqueued_ts_ns,
        right.sequence,
        right.task_id,
    )


class TaskStore:
    """One bounded owner for every live Task entity."""

    def __init__(self, capacity: int, *, generation_limit: int = UINT64_MAX) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.generation_limit = generation_limit
        self._slots: list[TaskInput | None] = [None] * capacity
        self._generations: list[int] = [0] * capacity
        self._free: list[int] = list(range(capacity - 1, -1, -1))
        self.registry: dict[int, Handle] = {}
        self.selected: dict[int, TaskInput] = {}
        self.terminal: dict[int, str] = {}
        self.queued_bytes = 0
        self.lookup_count = 0
        self.generation_mismatch_count = 0
        self.duplicate_remove_count = 0

    @property
    def live_count(self) -> int:
        return len(self.registry)

    def resolve(self, handle: Handle) -> TaskInput:
        self.lookup_count += 1
        if handle.slot_id < 0 or handle.slot_id >= self.capacity:
            self.generation_mismatch_count += 1
            raise ModelInvariantError("slot_id outside bounded Task store")
        task = self._slots[handle.slot_id]
        if task is None or task.generation != handle.generation:
            self.generation_mismatch_count += 1
            raise ModelInvariantError("stale handle or generation mismatch")
        return task

    def allocate(self, spec: TaskSpec, eligible_ts_ns: int | None) -> TaskInput:
        if spec.task_id <= 0 or spec.task_id in self.registry or spec.task_id in self.selected or spec.task_id in self.terminal:
            raise ModelInvariantError("Task ID must be unique")
        if spec.sequence < 0 or spec.deadline_ts_ns < 0 or spec.enqueued_ts_ns <= 0:
            raise ModelInvariantError("invalid immutable Task key")
        if spec.nbytes < 0:
            raise ModelInvariantError("negative Task bytes")
        bits = normalize_f64_bits(spec.route_score_f64_bits)
        if not self._free:
            raise ModelInvariantError("bounded Task store capacity exhausted")
        slot_id = self._free.pop()
        generation = self._generations[slot_id]
        if generation >= self.generation_limit:
            self._free.append(slot_id)
            raise ModelInvariantError("generation near-overflow fail-closed")
        generation += 1
        self._generations[slot_id] = generation
        task = TaskInput(
            run_id=spec.run_id,
            task_id=spec.task_id,
            slot_id=slot_id,
            generation=generation,
            deadline_ts_ns=spec.deadline_ts_ns,
            route_score_f64_bits=bits,
            sequence=spec.sequence,
            enqueued_ts_ns=spec.enqueued_ts_ns,
            eligible_ts_ns=eligible_ts_ns,
            nbytes=spec.nbytes,
            stage=spec.stage,
            phase=spec.phase,
            layer=spec.layer,
        )
        self._slots[slot_id] = task
        self.registry[task.task_id] = task.handle
        self.queued_bytes += task.nbytes
        return task

    def rollback_allocate(self, handle: Handle) -> None:
        task = self.resolve(handle)
        if self.registry.get(task.task_id) != handle:
            raise ModelInvariantError("rollback registry mismatch")
        del self.registry[task.task_id]
        self.queued_bytes -= task.nbytes
        self._slots[handle.slot_id] = None
        self._free.append(handle.slot_id)

    def remove_live(self, handle: Handle, terminal: str) -> TaskInput:
        task = self.resolve(handle)
        if self.registry.get(task.task_id) != handle:
            self.duplicate_remove_count += 1
            raise ModelInvariantError("duplicate removal or registry mismatch")
        del self.registry[task.task_id]
        self.queued_bytes -= task.nbytes
        if self.queued_bytes < 0:
            raise ModelInvariantError("queued bytes underflow")
        self._slots[handle.slot_id] = None
        self._free.append(handle.slot_id)
        if terminal == "SELECTED_OFFLINE":
            self.selected[task.task_id] = task
        else:
            if task.task_id in self.terminal:
                raise ModelInvariantError("duplicate Task terminal")
            self.terminal[task.task_id] = terminal
        return task

    def complete_selected(self, task_id: int, terminal: str = "SELECTED_OFFLINE") -> None:
        if task_id not in self.selected:
            raise ModelInvariantError("Task was not selected or already terminal")
        if task_id in self.terminal:
            raise ModelInvariantError("duplicate Task terminal")
        del self.selected[task_id]
        self.terminal[task_id] = terminal

    def reject(self, task_id: int) -> None:
        if task_id <= 0 or task_id in self.registry or task_id in self.selected or task_id in self.terminal:
            raise ModelInvariantError("duplicate reject or Task identity")
        self.terminal[task_id] = "REJECTED"

    def audit(self) -> dict[str, int | bool]:
        live = [task for task in self._slots if task is not None]
        bytes_sum = sum(task.nbytes for task in live)
        return {
            "live_slots": len(live),
            "registry_size": len(self.registry),
            "queued_bytes": self.queued_bytes,
            "recomputed_queued_bytes": bytes_sum,
            "generation_mismatch_count": self.generation_mismatch_count,
            "duplicate_remove_count": self.duplicate_remove_count,
            "passed": (
                len(live) == len(self.registry)
                and bytes_sum == self.queued_bytes
                and all(self.registry.get(task.task_id) == task.handle for task in live)
            ),
        }


class IndexedHeap:
    """Binary indexed heap with eager O(log n) erase and no stale entries."""

    def __init__(
        self,
        name: str,
        store: TaskStore,
        higher: Callable[[TaskInput, TaskInput], bool],
    ) -> None:
        self.name = name
        self.store = store
        self.higher = higher
        self.heap: list[Handle] = []
        self.positions: dict[Handle, int] = {}
        self.head_ops = 0
        self.insert_ops = 0
        self.erase_ops = 0
        self.compare_count = 0
        self.swap_count = 0
        self.stale_handle_count = 0

    def __len__(self) -> int:
        return len(self.heap)

    def _is_higher(self, left: Handle, right: Handle) -> bool:
        self.compare_count += 1
        try:
            return self.higher(self.store.resolve(left), self.store.resolve(right))
        except ModelInvariantError:
            self.stale_handle_count += 1
            raise

    def _swap(self, left: int, right: int) -> None:
        self.swap_count += 1
        self.heap[left], self.heap[right] = self.heap[right], self.heap[left]
        self.positions[self.heap[left]] = left
        self.positions[self.heap[right]] = right

    def _sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if not self._is_higher(self.heap[index], self.heap[parent]):
                break
            self._swap(index, parent)
            index = parent

    def _sift_down(self, index: int) -> None:
        size = len(self.heap)
        while True:
            left = 2 * index + 1
            if left >= size:
                return
            right = left + 1
            best = left
            if right < size and self._is_higher(self.heap[right], self.heap[left]):
                best = right
            if not self._is_higher(self.heap[best], self.heap[index]):
                return
            self._swap(index, best)
            index = best

    def insert(self, handle: Handle) -> None:
        if handle in self.positions:
            raise ModelInvariantError(f"duplicate handle in {self.name}")
        self.store.resolve(handle)
        self.insert_ops += 1
        self.positions[handle] = len(self.heap)
        self.heap.append(handle)
        self._sift_up(len(self.heap) - 1)

    def head(self) -> Handle | None:
        self.head_ops += 1
        if not self.heap:
            return None
        handle = self.heap[0]
        self.store.resolve(handle)
        return handle

    def erase(self, handle: Handle) -> None:
        index = self.positions.pop(handle, None)
        if index is None:
            raise ModelInvariantError(f"duplicate erase or missing handle in {self.name}")
        self.erase_ops += 1
        last = self.heap.pop()
        if index == len(self.heap):
            return
        self.heap[index] = last
        self.positions[last] = index
        parent = (index - 1) // 2
        if index > 0 and self._is_higher(self.heap[index], self.heap[parent]):
            self._sift_up(index)
        else:
            self._sift_down(index)

    def audit(self) -> dict[str, int | bool]:
        valid = len(self.heap) == len(self.positions)
        for index, handle in enumerate(self.heap):
            valid = valid and self.positions.get(handle) == index
            self.store.resolve(handle)
            left = 2 * index + 1
            right = left + 1
            if left < len(self.heap):
                valid = valid and not self.higher(
                    self.store.resolve(self.heap[left]), self.store.resolve(handle)
                )
            if right < len(self.heap):
                valid = valid and not self.higher(
                    self.store.resolve(self.heap[right]), self.store.resolve(handle)
                )
        return {
            "name": self.name,
            "size": len(self.heap),
            "position_size": len(self.positions),
            "stale_handle_count": self.stale_handle_count,
            "passed": bool(valid),
        }

    def counters(self) -> dict[str, int]:
        return {
            f"{self.name}_head_ops": self.head_ops,
            f"{self.name}_insert_ops": self.insert_ops,
            f"{self.name}_erase_ops": self.erase_ops,
            f"{self.name}_compare_count": self.compare_count,
            f"{self.name}_swap_count": self.swap_count,
            f"{self.name}_stale_handle_count": self.stale_handle_count,
        }


def _task_snapshot(task: TaskInput | None, now_ns: int | None = None) -> dict[str, object]:
    if task is None:
        return {"available": False}
    result: dict[str, object] = {
        "available": True,
        "task_id": task.task_id,
        "slot_id": task.slot_id,
        "generation": task.generation,
        "deadline_ts_ns": task.deadline_ts_ns,
        "route_score_f64_bits": task.route_score_f64_bits,
        "sequence": task.sequence,
        "enqueued_ts_ns": task.enqueued_ts_ns,
        "eligible_ts_ns": task.eligible_ts_ns,
    }
    result["waiting_ns"] = None if now_ns is None else max(0, now_ns - task.enqueued_ts_ns)
    return result


class OfflineQueue:
    """Shared structural implementation for S0 and synthetic-only S1."""

    def __init__(
        self,
        capacity: int,
        *,
        policy_config: FixturePolicyConfig | None = None,
        generation_limit: int = UINT64_MAX,
    ) -> None:
        self.store = TaskStore(capacity, generation_limit=generation_limit)
        self.legacy_index = IndexedHeap("legacy", self.store, legacy_higher)
        self.aging_index = IndexedHeap("aging", self.store, aging_higher)
        self.policy_config = policy_config
        self.policy_state = PolicyState()
        self.next_decision_id = 0
        self.next_queue_op_id = 0
        self.full_store_scan_count = 0
        self.invariant_errors: list[str] = []
        self.structural_operations: list[tuple[str, int]] = []

    @property
    def mode(self) -> str:
        return "s1_fixture" if self.policy_config is not None else "s0"

    def _eligible_time(self, spec: TaskSpec) -> int | None:
        if self.policy_config is None:
            return None
        return saturating_add_u64(
            spec.enqueued_ts_ns,
            self.policy_config.minimum_eligibility_age_ns,
        )

    def enqueue(self, spec: TaskSpec, *, fail_at: str | None = None) -> TaskInput:
        if fail_at == "before_store":
            raise MemoryError("injected allocation failure")
        task = self.store.allocate(spec, self._eligible_time(spec))
        legacy_inserted = aging_inserted = False
        try:
            if fail_at == "after_store":
                raise MemoryError("injected post-store failure")
            self.legacy_index.insert(task.handle)
            legacy_inserted = True
            if fail_at == "after_legacy":
                raise MemoryError("injected Legacy Index insertion failure")
            self.aging_index.insert(task.handle)
            aging_inserted = True
            if fail_at == "after_aging":
                raise MemoryError("injected Aging Index insertion failure")
        except Exception:
            if aging_inserted:
                self.aging_index.erase(task.handle)
            if legacy_inserted:
                self.legacy_index.erase(task.handle)
            self.store.rollback_allocate(task.handle)
            raise
        self.next_queue_op_id += 1
        self.structural_operations.append(("enqueue", task.task_id))
        self.assert_constant_invariants()
        return task

    def cancel(self, task_id: int, reason: str = "CANCELLED") -> TaskInput:
        handle = self.store.registry.get(task_id)
        if handle is None:
            raise ModelInvariantError("queued cancel requires a live registry handle")
        self.legacy_index.erase(handle)
        self.aging_index.erase(handle)
        task = self.store.remove_live(handle, reason)
        self.next_queue_op_id += 1
        self.structural_operations.append(("cancel", task.task_id))
        self._normalize_state_when_empty()
        self.assert_constant_invariants()
        return task

    def reject(self, task_id: int) -> None:
        self.store.reject(task_id)
        self.structural_operations.append(("reject", task_id))

    def _head_task(self, index: IndexedHeap) -> TaskInput | None:
        handle = index.head()
        return None if handle is None else self.store.resolve(handle)

    def _normalize_state_when_empty(self) -> None:
        if self.store.live_count == 0:
            self.policy_state = PolicyState()

    def counters(self) -> dict[str, int]:
        result = {
            "task_store_lookup_count": self.store.lookup_count,
            "full_store_scan_count": self.full_store_scan_count,
            "generation_mismatch_count": self.store.generation_mismatch_count,
            "duplicate_remove_count": self.store.duplicate_remove_count,
        }
        result.update(self.legacy_index.counters())
        result.update(self.aging_index.counters())
        return result

    def _counter_delta(self, before: dict[str, int]) -> dict[str, int]:
        after = self.counters()
        return {name: after[name] - before.get(name, 0) for name in after}

    def select(
        self,
        *,
        decision_ts_ns: int,
        batch_id: int,
        batch_slot: int,
        worker_id: int,
        stopping: bool = False,
    ) -> dict[str, object]:
        if self.store.live_count == 0:
            raise ModelInvariantError("cannot select from an empty queue")
        if not (0 <= decision_ts_ns <= UINT64_MAX):
            raise ModelInvariantError("decision time outside uint64")
        counters_before = self.counters()
        live_before = self.store.live_count
        bytes_before = self.store.queued_bytes
        legacy = self._head_task(self.legacy_index)
        oldest = self._head_task(self.aging_index)
        if legacy is None or oldest is None:
            raise ModelInvariantError("both indexes must have a head for every live Task")

        hard_present = False
        eligible_present = False
        oldest_eligible: TaskInput | None = None
        transition: PolicyTransition | None = None
        if self.policy_config is not None:
            hard_limit = saturating_add_u64(
                decision_ts_ns, self.policy_config.hard_urgent_guard_ns
            )
            hard_present = legacy.deadline_ts_ns != 0 and legacy.deadline_ts_ns <= hard_limit
            eligible_present = (
                oldest.eligible_ts_ns is not None
                and decision_ts_ns >= oldest.eligible_ts_ns
            )
            oldest_eligible = oldest if eligible_present else None
            transition = decide_reserved_service(
                self.policy_state,
                self.policy_config,
                waiting_eligible_present=eligible_present,
                hard_urgent_present=hard_present,
                queue_size=live_before,
                stopping=stopping,
            )
            if transition.selected_source == "reserved":
                if oldest_eligible is None:
                    raise ModelInvariantError("reserved source without oldest eligible Task")
                winner = oldest_eligible
            else:
                winner = legacy
            if hard_present and winner.handle != legacy.handle:
                raise ModelInvariantError("hard-urgent safety violation")
            self.policy_state = transition.state_after
        else:
            winner = legacy

        decision_id = self.next_decision_id
        queue_op_id = self.next_queue_op_id
        self.legacy_index.erase(winner.handle)
        self.aging_index.erase(winner.handle)
        selected = self.store.remove_live(winner.handle, "SELECTED_OFFLINE")
        self.next_decision_id += 1
        self.next_queue_op_id += 1
        self.structural_operations.append(("select", selected.task_id))

        credit_reset_after_remove = False
        if self.policy_config is not None and not stopping:
            next_oldest = self._head_task(self.aging_index)
            next_eligible = (
                next_oldest is not None
                and next_oldest.eligible_ts_ns is not None
                and decision_ts_ns >= next_oldest.eligible_ts_ns
            )
            if not next_eligible and self.policy_state != PolicyState():
                self.policy_state = PolicyState()
                credit_reset_after_remove = True
        self._normalize_state_when_empty()
        self.assert_constant_invariants()
        delta = self._counter_delta(counters_before)

        if transition is None:
            source = "legacy"
            reason = "S0_LEGACY_HEAD"
            credit_before = credit_accrued = credit_after = 0
            debt_before = debt_after = debt_created = debt_repaid = False
            reserved_due = False
            credit_reset = False
        else:
            source = transition.selected_source
            reason = transition.override_reason
            credit_before = transition.credit_before
            credit_accrued = transition.credit_accrued
            credit_after = self.policy_state.credit
            debt_before = transition.debt_before
            debt_after = self.policy_state.pending_debt
            debt_created = transition.debt_created
            debt_repaid = transition.debt_repaid
            reserved_due = transition.reserved_due
            credit_reset = transition.credit_reset or credit_reset_after_remove

        return {
            "queue_op_id": queue_op_id,
            "decision_id": decision_id,
            "batch_id": batch_id,
            "batch_slot": batch_slot,
            "worker_id": worker_id,
            "decision_ts_ns": decision_ts_ns,
            "mode": self.mode,
            "selected_task": _task_snapshot(selected, decision_ts_ns),
            "selected_source": source,
            "legacy_head": _task_snapshot(legacy, decision_ts_ns),
            "aging_head": _task_snapshot(oldest, decision_ts_ns),
            "oldest_eligible": _task_snapshot(oldest_eligible, decision_ts_ns),
            "hard_urgent_present": hard_present,
            "reserved_due": reserved_due,
            "override_reason": reason,
            "credit_before": credit_before,
            "credit_accrued": credit_accrued,
            "credit_after": credit_after,
            "debt_before": debt_before,
            "debt_after": debt_after,
            "debt_created": debt_created,
            "debt_repaid": debt_repaid,
            "credit_reset": credit_reset,
            "task_store_live_before": live_before,
            "task_store_live_after": self.store.live_count,
            "legacy_index_size_before": live_before,
            "legacy_index_size_after": len(self.legacy_index),
            "aging_index_size_before": live_before,
            "aging_index_size_after": len(self.aging_index),
            "queued_bytes_before": bytes_before,
            "queued_bytes_after": self.store.queued_bytes,
            "index_operation_counts": delta,
            "stale_entries": (
                self.legacy_index.stale_handle_count + self.aging_index.stale_handle_count
            ),
            "full_store_scan_count": self.full_store_scan_count,
            "selection_complexity_class": "O_1_HEAD_PLUS_O_LOG_N_ERASE",
            "invariant_errors": list(self.invariant_errors),
        }

    def complete_selected(self, task_id: int) -> None:
        self.store.complete_selected(task_id)

    def drain(self, decision_times: Iterable[int], *, abort: bool = False) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for slot, now_ns in enumerate(decision_times):
            if self.store.live_count == 0:
                break
            record = self.select(
                decision_ts_ns=now_ns,
                batch_id=slot,
                batch_slot=0,
                worker_id=0,
                stopping=True,
            )
            task_id = int(record["selected_task"]["task_id"])
            self.store.complete_selected(
                task_id, "ABORTED_OFFLINE" if abort else "SELECTED_OFFLINE"
            )
            records.append(record)
        if self.store.live_count != 0:
            raise ModelInvariantError("shutdown fixture did not drain the queue")
        return records

    def assert_constant_invariants(self) -> None:
        live = self.store.live_count
        if len(self.legacy_index) != live or len(self.aging_index) != live:
            self.invariant_errors.append("store_index_size_mismatch")
            raise ModelInvariantError("store/index/registry sizes differ")
        if self.full_store_scan_count != 0:
            self.invariant_errors.append("full_store_scan_nonzero")
            raise ModelInvariantError("selection full-store scan is forbidden")
        if self.legacy_index.stale_handle_count or self.aging_index.stale_handle_count:
            self.invariant_errors.append("stale_handle_nonzero")
            raise ModelInvariantError("stale handle is fail-closed")

    def audit_invariants(self) -> dict[str, object]:
        store = self.store.audit()
        legacy = self.legacy_index.audit()
        aging = self.aging_index.audit()
        passed = (
            bool(store["passed"])
            and bool(legacy["passed"])
            and bool(aging["passed"])
            and self.store.live_count == len(self.legacy_index) == len(self.aging_index)
            and self.full_store_scan_count == 0
        )
        return {
            "passed": passed,
            "store": store,
            "legacy_index": legacy,
            "aging_index": aging,
            "full_store_scan_count": self.full_store_scan_count,
            "policy_state": asdict(self.policy_state),
            "invariant_errors": list(self.invariant_errors),
        }


def clone_spec(task: TaskInput) -> TaskSpec:
    return TaskSpec(
        run_id=task.run_id,
        task_id=task.task_id,
        deadline_ts_ns=task.deadline_ts_ns,
        route_score_f64_bits=task.route_score_f64_bits,
        sequence=task.sequence,
        enqueued_ts_ns=task.enqueued_ts_ns,
        nbytes=task.nbytes,
        stage=task.stage,
        phase=task.phase,
        layer=task.layer,
    )
