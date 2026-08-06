from __future__ import annotations

import json
import math
import random
import struct
import unittest

from m6c_offline.model import (
    Handle,
    ModelInvariantError,
    OfflineQueue,
    TaskSpec,
    legacy_higher,
)
from m6c_offline.policy import (
    FixturePolicyConfig,
    PolicyConfigurationError,
    PolicyState,
    UINT64_MAX,
    decide_reserved_service,
    saturating_add_u64,
)


def bits(value: float) -> str:
    return "0x" + struct.pack(">d", value).hex()


def fixture(r: int = 1, d: int = 4, age: int = 10, guard: int = 0) -> FixturePolicyConfig:
    result = FixturePolicyConfig(r, d, age, guard)
    assert result.test_fixture_only is True
    assert result.experimental_parameter is False
    return result


def spec(
    task_id: int,
    *,
    enqueued: int = 10,
    deadline: int = 0,
    score: float = 0.0,
    nbytes: int = 100,
) -> TaskSpec:
    return TaskSpec(
        run_id="synthetic",
        task_id=task_id,
        deadline_ts_ns=deadline,
        route_score_f64_bits=bits(score),
        sequence=task_id - 1,
        enqueued_ts_ns=enqueued,
        nbytes=nbytes,
        stage="LATE",
        phase="PREFILL",
        layer=0,
    )


def enqueue_many(queue: OfflineQueue, tasks: list[TaskSpec]) -> None:
    for task in tasks:
        queue.enqueue(task)


class TestM6COfflinePolicy(unittest.TestCase):
    def test_01_no_eligible_resets_credit_and_debt(self) -> None:
        transition = decide_reserved_service(
            PolicyState(1, False), fixture(2, 4, age=10),
            waiting_eligible_present=False,
            hard_urgent_present=False,
            queue_size=2,
        )
        self.assertEqual((transition.credit_after, transition.debt_after), (0, False))
        self.assertEqual(transition.override_reason, "NO_WAITING_ELIGIBLE")

    def test_02_credit_accumulates_once_per_slot(self) -> None:
        transition = decide_reserved_service(
            PolicyState(), fixture(),
            waiting_eligible_present=True,
            hard_urgent_present=False,
            queue_size=3,
        )
        self.assertEqual(transition.credit_after, 1)
        self.assertEqual(transition.selected_source, "legacy")

    def test_03_fresh_reserved_due_consumes_denominator(self) -> None:
        transition = decide_reserved_service(
            PolicyState(3, False), fixture(),
            waiting_eligible_present=True,
            hard_urgent_present=False,
            queue_size=3,
        )
        self.assertEqual(transition.selected_source, "reserved")
        self.assertEqual(transition.credit_after, 0)
        self.assertEqual(transition.override_reason, "RESERVED_CREDIT_USED")

    def test_05_hard_urgent_overrides_fresh_reserved_due(self) -> None:
        transition = decide_reserved_service(
            PolicyState(3, False), fixture(),
            waiting_eligible_present=True,
            hard_urgent_present=True,
            queue_size=3,
        )
        self.assertEqual(transition.selected_source, "hard_urgent")
        self.assertTrue(transition.debt_created)
        self.assertTrue(transition.debt_after)

    def test_06_override_creates_only_boolean_debt(self) -> None:
        first = decide_reserved_service(
            PolicyState(3, False), fixture(),
            waiting_eligible_present=True,
            hard_urgent_present=True,
            queue_size=4,
        )
        second = decide_reserved_service(
            first.state_after, fixture(),
            waiting_eligible_present=True,
            hard_urgent_present=True,
            queue_size=3,
        )
        self.assertTrue(second.debt_after)
        self.assertFalse(second.debt_created)

    def test_07_pending_debt_pauses_credit(self) -> None:
        transition = decide_reserved_service(
            PolicyState(0, True), fixture(),
            waiting_eligible_present=True,
            hard_urgent_present=True,
            queue_size=3,
        )
        self.assertEqual(transition.credit_accrued, 0)
        self.assertEqual(transition.credit_after, 0)

    def test_08_first_safe_slot_repays_once(self) -> None:
        transition = decide_reserved_service(
            PolicyState(0, True), fixture(),
            waiting_eligible_present=True,
            hard_urgent_present=False,
            queue_size=3,
        )
        self.assertEqual(transition.selected_source, "reserved")
        self.assertTrue(transition.debt_repaid)
        self.assertFalse(transition.debt_after)

    def test_09_repayment_does_not_immediately_create_new_reserved_slot(self) -> None:
        repaid = decide_reserved_service(
            PolicyState(0, True), fixture(),
            waiting_eligible_present=True,
            hard_urgent_present=False,
            queue_size=3,
        )
        next_slot = decide_reserved_service(
            repaid.state_after, fixture(),
            waiting_eligible_present=True,
            hard_urgent_present=False,
            queue_size=2,
        )
        self.assertEqual(next_slot.selected_source, "legacy")
        self.assertEqual(next_slot.credit_after, 1)

    def test_10_two_r_less_than_or_equal_d_boundary(self) -> None:
        self.assertEqual(fixture(2, 4).reserved_denominator, 4)
        with self.assertRaises(PolicyConfigurationError):
            fixture(3, 5)
        with self.assertRaises(PolicyConfigurationError):
            FixturePolicyConfig(1, 4, 10, 0, test_fixture_only=False)
        with self.assertRaises(PolicyConfigurationError):
            FixturePolicyConfig(1, 4, 10, 0, experimental_parameter=True)

    def test_12_guard_boundary_and_saturating_add(self) -> None:
        self.assertEqual(saturating_add_u64(UINT64_MAX - 2, 9), UINT64_MAX)
        queue = OfflineQueue(4, policy_config=fixture(1, 4, age=1, guard=5))
        enqueue_many(queue, [spec(1, enqueued=1, deadline=105), spec(2, enqueued=2)])
        record = queue.select(decision_ts_ns=100, batch_id=0, batch_slot=0, worker_id=0)
        self.assertTrue(record["hard_urgent_present"])
        self.assertEqual(record["selected_task"]["task_id"], 1)


class TestM6COfflineStructure(unittest.TestCase):
    def test_04_reserved_winner_is_oldest_eligible(self) -> None:
        queue = OfflineQueue(8, policy_config=fixture(age=1))
        enqueue_many(queue, [
            spec(1, enqueued=10, score=0.1),
            spec(2, enqueued=20, score=0.9),
            spec(3, enqueued=30, score=0.8),
        ])
        queue.policy_state = PolicyState(3, False)
        record = queue.select(decision_ts_ns=100, batch_id=0, batch_slot=0, worker_id=1)
        self.assertEqual(record["selected_source"], "reserved")
        self.assertEqual(record["selected_task"]["task_id"], 1)

    def test_11_deadline_zero_is_preserved_and_nonurgent(self) -> None:
        queue = OfflineQueue(4, policy_config=fixture(age=1, guard=100))
        enqueue_many(queue, [spec(1, deadline=0, score=100), spec(2, deadline=500, score=0)])
        record = queue.select(decision_ts_ns=100, batch_id=0, batch_slot=0, worker_id=0)
        self.assertEqual(record["selected_task"]["task_id"], 2)
        self.assertFalse(record["hard_urgent_present"])

    def test_13_queue_only_one_task(self) -> None:
        queue = OfflineQueue(2, policy_config=fixture(age=1))
        queue.enqueue(spec(1, enqueued=1))
        record = queue.select(decision_ts_ns=20, batch_id=0, batch_slot=0, worker_id=0)
        self.assertEqual(record["selected_task"]["task_id"], 1)
        self.assertIn(record["override_reason"], {"QUEUE_ONLY_ONE_CLASS", "RESERVED_CREDIT_USED"})

    def test_14_batch_updates_credit_per_slot_with_one_frozen_time(self) -> None:
        queue = OfflineQueue(8, policy_config=fixture(age=1))
        enqueue_many(queue, [spec(i, enqueued=i) for i in range(1, 7)])
        records = [
            queue.select(decision_ts_ns=100, batch_id=7, batch_slot=slot, worker_id=0)
            for slot in range(4)
        ]
        self.assertEqual([record["credit_before"] for record in records], [0, 1, 2, 3])
        self.assertEqual([record["decision_ts_ns"] for record in records], [100] * 4)
        self.assertEqual(records[-1]["selected_source"], "reserved")

    def test_15_worker_id_never_changes_winner_or_tie_break(self) -> None:
        winners = []
        for worker in (0, 99):
            queue = OfflineQueue(4)
            enqueue_many(queue, [spec(1, score=0.1), spec(2, score=0.9)])
            record = queue.select(decision_ts_ns=100, batch_id=0, batch_slot=0, worker_id=worker)
            winners.append(record["selected_task"]["task_id"])
            self.assertEqual(record["decision_id"], 0)
        self.assertEqual(winners, [2, 2])

    def test_16_cancel_eagerly_erases_both_indexes(self) -> None:
        queue = OfflineQueue(4)
        enqueue_many(queue, [spec(1), spec(2)])
        queue.cancel(1)
        self.assertEqual(queue.store.live_count, 1)
        self.assertEqual(len(queue.legacy_index), 1)
        self.assertEqual(len(queue.aging_index), 1)
        self.assertTrue(queue.audit_invariants()["passed"])

    def test_17_reject_never_enters_store_or_indexes(self) -> None:
        queue = OfflineQueue(2)
        queue.reject(9)
        self.assertEqual(queue.store.live_count, 0)
        self.assertEqual(queue.store.terminal[9], "REJECTED")

    def test_18_clean_shutdown_drain_uses_shared_selection_transaction(self) -> None:
        queue = OfflineQueue(4, policy_config=fixture(age=1))
        enqueue_many(queue, [spec(1), spec(2)])
        records = queue.drain([100, 100])
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["override_reason"] == "SHUTDOWN_DRAIN" for record in records))
        self.assertEqual(queue.store.live_count, 0)

    def test_19_explicit_abort_fixture_is_not_runtime_shutdown_event(self) -> None:
        queue = OfflineQueue(2)
        queue.enqueue(spec(1))
        queue.drain([100], abort=True)
        self.assertEqual(queue.store.terminal[1], "ABORTED_OFFLINE")
        self.assertNotEqual(queue.store.terminal[1], "SHUTDOWN_DRAINED")

    def test_20_capacity_full_is_fail_closed(self) -> None:
        queue = OfflineQueue(1)
        queue.enqueue(spec(1))
        with self.assertRaises(ModelInvariantError):
            queue.enqueue(spec(2))
        self.assertTrue(queue.audit_invariants()["passed"])

    def test_21_enqueue_transaction_rollback_after_store(self) -> None:
        queue = OfflineQueue(2)
        with self.assertRaises(MemoryError):
            queue.enqueue(spec(1), fail_at="after_store")
        self.assertEqual(queue.store.live_count, 0)
        self.assertEqual(queue.store.queued_bytes, 0)
        self.assertTrue(queue.audit_invariants()["passed"])

    def test_22_index_insertion_failure_injection_rolls_back(self) -> None:
        for point in ("before_store", "after_legacy", "after_aging"):
            queue = OfflineQueue(2)
            with self.assertRaises(MemoryError):
                queue.enqueue(spec(1), fail_at=point)
            self.assertEqual(queue.store.live_count, 0)
            self.assertEqual(len(queue.legacy_index), 0)
            self.assertEqual(len(queue.aging_index), 0)

    def test_23_duplicate_erase_is_fail_closed(self) -> None:
        queue = OfflineQueue(2)
        task = queue.enqueue(spec(1))
        queue.cancel(1)
        with self.assertRaises(ModelInvariantError):
            queue.legacy_index.erase(task.handle)

    def test_24_generation_mismatch_is_fail_closed(self) -> None:
        queue = OfflineQueue(2)
        task = queue.enqueue(spec(1))
        with self.assertRaises(ModelInvariantError):
            queue.store.resolve(Handle(task.slot_id, task.generation + 1))

    def test_25_generation_near_overflow_stops_reuse(self) -> None:
        queue = OfflineQueue(1, generation_limit=1)
        queue.enqueue(spec(1))
        queue.cancel(1)
        with self.assertRaises(ModelInvariantError):
            queue.enqueue(spec(2))

    def test_26_queued_bytes_conservation(self) -> None:
        queue = OfflineQueue(4)
        enqueue_many(queue, [spec(1, nbytes=10), spec(2, nbytes=30)])
        self.assertEqual(queue.store.queued_bytes, 40)
        queue.cancel(1)
        self.assertEqual(queue.store.queued_bytes, 30)
        self.assertTrue(queue.store.audit()["passed"])

    def test_27_task_terminal_is_exactly_once(self) -> None:
        queue = OfflineQueue(2)
        queue.enqueue(spec(1))
        record = queue.select(decision_ts_ns=100, batch_id=0, batch_slot=0, worker_id=0)
        queue.complete_selected(record["selected_task"]["task_id"])
        with self.assertRaises(ModelInvariantError):
            queue.complete_selected(1)

    def test_28_s0_s1_share_identical_structural_operations(self) -> None:
        s0 = OfflineQueue(4)
        s1 = OfflineQueue(4, policy_config=fixture(age=1000))
        tasks = [spec(1, score=0.2), spec(2, score=0.8)]
        enqueue_many(s0, tasks)
        enqueue_many(s1, tasks)
        for queue in (s0, s1):
            record = queue.select(decision_ts_ns=100, batch_id=0, batch_slot=0, worker_id=0)
            queue.complete_selected(record["selected_task"]["task_id"])
        self.assertEqual(s0.structural_operations, s1.structural_operations)

    def test_29_hard_urgent_safety(self) -> None:
        queue = OfflineQueue(8, policy_config=fixture(age=1, guard=1))
        enqueue_many(queue, [
            spec(1, enqueued=1, deadline=0, score=100),
            spec(2, enqueued=2, deadline=100, score=0),
            spec(3, enqueued=3, deadline=101, score=100),
        ])
        queue.policy_state = PolicyState(3, False)
        record = queue.select(decision_ts_ns=99, batch_id=0, batch_slot=0, worker_id=0)
        self.assertTrue(record["hard_urgent_present"])
        self.assertEqual(record["selected_task"]["task_id"], 2)

    def test_30_sustained_hard_urgent_overload_keeps_one_debt(self) -> None:
        queue = OfflineQueue(16, policy_config=fixture(age=1, guard=1000))
        tasks = [spec(i, enqueued=i, deadline=100 + i) for i in range(1, 10)]
        tasks.append(spec(10, enqueued=1, deadline=0))
        enqueue_many(queue, tasks)
        queue.policy_state = PolicyState(3, False)
        debt_values = []
        credit_values = []
        for slot in range(5):
            record = queue.select(decision_ts_ns=100, batch_id=slot, batch_slot=0, worker_id=slot % 2)
            debt_values.append(record["debt_after"])
            credit_values.append(record["credit_after"])
        self.assertTrue(all(debt_values))
        self.assertEqual(len(set(credit_values)), 1)

    def test_31_longest_reserved_streak_is_one_under_frozen_fixture(self) -> None:
        queue = OfflineQueue(64, policy_config=fixture(age=1))
        enqueue_many(queue, [spec(i, enqueued=i) for i in range(1, 41)])
        streak = longest = 0
        for slot in range(30):
            record = queue.select(decision_ts_ns=100, batch_id=slot, batch_slot=0, worker_id=0)
            if record["selected_source"] == "reserved":
                streak += 1
                longest = max(longest, streak)
            else:
                streak = 0
        self.assertLessEqual(longest, 1)

    def test_32_stale_handle_counter_remains_zero(self) -> None:
        queue = OfflineQueue(8)
        enqueue_many(queue, [spec(i) for i in range(1, 7)])
        for _ in range(3):
            record = queue.select(decision_ts_ns=100, batch_id=0, batch_slot=0, worker_id=0)
            queue.complete_selected(record["selected_task"]["task_id"])
        audit = queue.audit_invariants()
        self.assertEqual(audit["legacy_index"]["stale_handle_count"], 0)
        self.assertEqual(audit["aging_index"]["stale_handle_count"], 0)

    def test_33_selection_full_store_scan_counter_remains_zero(self) -> None:
        queue = OfflineQueue(8)
        enqueue_many(queue, [spec(i) for i in range(1, 7)])
        for _ in range(6):
            record = queue.select(decision_ts_ns=100, batch_id=0, batch_slot=0, worker_id=0)
            queue.complete_selected(record["selected_task"]["task_id"])
        self.assertEqual(queue.full_store_scan_count, 0)

    def test_34_repeated_identical_input_is_byte_deterministic(self) -> None:
        outputs = []
        for _ in range(2):
            queue = OfflineQueue(8, policy_config=fixture(age=1))
            enqueue_many(queue, [spec(i, enqueued=i, score=i / 10) for i in range(1, 7)])
            records = []
            for slot in range(4):
                records.append(queue.select(
                    decision_ts_ns=100,
                    batch_id=2,
                    batch_slot=slot,
                    worker_id=slot % 2,
                ))
            outputs.append(json.dumps(records, sort_keys=True, separators=(",", ":")))
        self.assertEqual(outputs[0], outputs[1])

    def test_cpp_golden_deadline_score_ordering(self) -> None:
        queue = OfflineQueue(8)
        enqueue_many(queue, [
            spec(1, deadline=0, score=100),
            spec(2, deadline=200, score=0.1),
            spec(3, deadline=100, score=-1),
            spec(4, deadline=100, score=0.5),
            spec(5, deadline=100, score=0.5),
        ])
        order = []
        while queue.store.live_count:
            record = queue.select(decision_ts_ns=1000, batch_id=0, batch_slot=0, worker_id=0)
            task_id = record["selected_task"]["task_id"]
            order.append(task_id)
            queue.complete_selected(task_id)
        self.assertEqual(order, [4, 5, 3, 2, 1])

    def test_authoritative_bits_preserve_signed_zero_comparator_tie(self) -> None:
        left = spec(1, score=0.0)
        right = TaskSpec(**{**spec(2, score=-0.0).__dict__, "sequence": 1})
        queue = OfflineQueue(2)
        task_left = queue.enqueue(left)
        task_right = queue.enqueue(right)
        self.assertTrue(legacy_higher(task_left, task_right))

    def test_deterministic_random_property_loop_matches_reference(self) -> None:
        for seed in range(20):
            rng = random.Random(seed)
            queue = OfflineQueue(128)
            live: dict[int, TaskSpec] = {}
            for task_id in range(1, 80):
                task = spec(
                    task_id,
                    enqueued=task_id,
                    deadline=0 if rng.randrange(4) == 0 else rng.randrange(1, 50),
                    score=rng.uniform(-2.0, 2.0),
                    nbytes=rng.randrange(0, 1000),
                )
                queue.enqueue(task)
                live[task_id] = task
                if task_id % 5 == 0:
                    expected = min(
                        live.values(),
                        key=lambda item: (
                            item.deadline_ts_ns == 0,
                            item.deadline_ts_ns,
                            -struct.unpack(">d", bytes.fromhex(item.route_score_f64_bits[2:]))[0],
                            item.sequence,
                        ),
                    )
                    record = queue.select(
                        decision_ts_ns=1000,
                        batch_id=task_id,
                        batch_slot=0,
                        worker_id=rng.randrange(4),
                    )
                    actual = record["selected_task"]["task_id"]
                    self.assertEqual(actual, expected.task_id)
                    queue.complete_selected(actual)
                    del live[actual]
            self.assertTrue(queue.audit_invariants()["passed"])
            self.assertEqual(queue.full_store_scan_count, 0)


if __name__ == "__main__":
    unittest.main()
