import struct
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest


TRACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACE_DIR))

import audit_router_score_determinism as audit


def f64_bits(value):
    return f"0x{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


class RouterScoreDeterminismAuditTest(unittest.TestCase):
    def task(self, task_id, score, sequence):
        return audit.Task(
            task_id=task_id,
            phase="DECODE",
            step=2,
            layer=1,
            expert=task_id,
            tensor=f"tensor-{task_id}",
            stage="EARLY",
            nbytes=4096,
            score=score,
            score_f64_bits=f64_bits(score),
            created_ts_ns=1,
            sequence=sequence,
            deadline_ts_ns=100,
            enqueued_ts_ns=10,
            dequeued_ts_ns=50 + sequence,
            issued_ts_ns=60 + sequence,
        )

    def make_run(self):
        first = self.task(1, 0.1, 1)
        second = self.task(2, 0.2, 2)
        tasks = {1: first, 2: second}
        return audit.AuditRun(
            label="run",
            path=Path("."),
            manifest={
                "run_name": "run",
                "environment": {
                    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE": "deadline_score",
                    "NUM_THREADS": "8",
                    "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS": "2",
                },
            },
            summary={},
            metrics={},
            routes=[],
            route_by_slot={},
            tasks=tasks,
            task_by_corr={},
            selections=[
                {
                    "event": "EXPERT_PRIORITY_SELECTION",
                    "task_id": 2,
                    "decision_ts_ns": 50,
                    "candidate_count": 2,
                },
                {
                    "event": "EXPERT_PRIORITY_SELECTION",
                    "task_id": 1,
                    "decision_ts_ns": 60,
                    "candidate_count": 1,
                },
            ],
            dequeue_order=[2, 1],
            issue_order=[2, 1],
            validation={},
        )

    def test_f32_ulp_and_exact_widening(self):
        self.assertEqual(audit.ulp_distance("0x3f800000", "0x3f800001", 32), 1)
        self.assertEqual(
            audit.f64_bits_from_f32_bits("0x3f800000"),
            "0x3ff0000000000000",
        )

    def test_deadline_replay_detects_score_order_change(self):
        run = self.make_run()
        replay = audit.replay_deadline_score(run, {1: 0.3, 2: 0.05})
        self.assertTrue(replay["passed"], replay)
        self.assertEqual(replay["self_replay_winner_mismatches"], 0)
        self.assertEqual(replay["score_injection_winner_changed_decisions"], 1)
        self.assertEqual(replay["comparator_order_changed_pair_observations"], 1)

    def test_task_correspondence_excludes_score_address_and_task_id(self):
        left = self.task(1, 0.1, 1)
        right = self.task(99, 0.9, 8)
        right.expert = left.expert
        right.tensor = left.tensor
        left.created_ts_ns = 10
        right.created_ts_ns = 20
        left_key = next(iter(audit.assign_task_correspondence({1: left})))
        right_key = next(iter(audit.assign_task_correspondence({99: right})))
        self.assertEqual(left_key, right_key)

    def test_large_score_reachable_group_fails_bounded_gate(self):
        members = set(range(audit.MAX_SCORE_REACHABLE_GROUP + 1))
        eligible, changed, relevant, complete = audit.score_reachable_pairs(
            members, {}, {}
        )
        self.assertEqual((eligible, changed, relevant), (0, 0, set()))
        self.assertFalse(complete)

    def test_comparison_roles_separate_stability_from_controls(self):
        self.assertEqual(audit.comparison_role("A_OFF_OFF_12"), "stability")
        self.assertEqual(audit.comparison_role("D_MAX_WAIT_SELF"), "stability")
        self.assertEqual(audit.comparison_role("THREAD_CROSS"), "control")

    def test_pre_barrier_pattern_is_runtime_defect(self):
        replay = {
            "comparator_order_changed_pair_observations": 0,
            "score_injection_winner_changed_decisions": 0,
        }
        base = {
            "comparison_role": "stability",
            "valid": True,
            "router": {
                "raw_bit_different_items": 1,
                "raw_bit_different_rank_zero_items": 0,
                "raw_bit_different_nonzero_rank_items": 1,
                "topk_rank_changed_items": 0,
            },
            "tasks": {"source_to_task_widening_mismatches": 0},
            "replay_a_with_b_scores": replay,
            "replay_b_with_a_scores": replay,
        }
        pair_a = dict(base, comparison_id="A_OFF_OFF_12")
        pair_b = {
            **base,
            "comparison_id": "B_SINGLE_THREAD",
            "router": {
                **base["router"],
                "raw_bit_different_items": 0,
                "raw_bit_different_nonzero_rank_items": 0,
            },
        }
        classification, _ = audit.choose_classification(
            [pair_a, pair_b], [SimpleNamespace(validation={"passed": True})]
        )
        self.assertEqual(classification, "RUNTIME_DEFECT")


if __name__ == "__main__":
    unittest.main()
