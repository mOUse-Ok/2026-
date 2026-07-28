import sys
import unittest
from pathlib import Path

TRACE_DIR = Path(__file__).resolve().parents[1]
if str(TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(TRACE_DIR))

import validate_pressure_shadow_equivalence as equivalence


def task(task_id, lifecycle, layer, score=0.5):
    return {
        "event": "EXPERT_TASK",
        "lifecycle_event": lifecycle,
        "state": {
            "CREATE": "CREATED",
            "ADMIT": "ADMITTED",
            "ENQUEUE": "ENQUEUED",
            "DEQUEUE": "DEQUEUED",
            "ISSUE": "ISSUED",
        }[lifecycle],
        "task_id": task_id,
        "step": 7,
        "layer": layer,
        "expert": task_id % 4,
        "phase": "DECODE",
        "stage": "EARLY",
        "tensor": f"blk.{layer}.ffn_gate_exps.weight",
        "nbytes": 4096,
        "score": score,
        "sequence": task_id,
    }


class PressureShadowEquivalenceTests(unittest.TestCase):
    def test_route_score_is_part_of_task_business_identity(self):
        left = equivalence.task_tuple(task(1, "CREATE", 2, score=0.4))
        right = equivalence.task_tuple(task(1, "CREATE", 2, score=0.5))
        self.assertNotEqual(left, right)

    def test_score_diagnostics_do_not_normalize_strict_difference(self):
        left = [task(1, "CREATE", 2, score=0.4)]
        right = [task(1, "CREATE", 2, score=0.5)]
        diagnostics = equivalence.create_task_difference_diagnostics(left, right)
        self.assertTrue(diagnostics["identity_without_score_order_equal"])
        self.assertEqual(diagnostics["score_position_mismatch_count"], 1)
        self.assertAlmostEqual(diagnostics["max_absolute_score_delta"], 0.1)
        self.assertNotEqual(
            equivalence.task_tuple(left[0]),
            equivalence.task_tuple(right[0]),
        )

    def test_concurrent_worker_returns_normalize_within_group(self):
        left = [task(1, "ISSUE", 2), task(2, "ISSUE", 2)]
        right = list(reversed(left))
        self.assertEqual(
            equivalence.grouped_normalized_order(
                left, equivalence.task_tuple
            ),
            equivalence.grouped_normalized_order(
                right, equivalence.task_tuple
            ),
        )

    def test_scheduling_group_order_is_preserved(self):
        left = [task(1, "DEQUEUE", 2), task(2, "DEQUEUE", 3)]
        right = list(reversed(left))
        self.assertNotEqual(
            equivalence.grouped_normalized_order(
                left, equivalence.task_tuple
            ),
            equivalence.grouped_normalized_order(
                right, equivalence.task_tuple
            ),
        )

    def test_pressure_events_do_not_change_existing_event_schema(self):
        baseline = [{"event": "STEP_BEGIN", "step": 1, "ts_ns": 10}]
        observed = baseline + [
            {
                "event": "PRESSURE_SHADOW_SAMPLE",
                "schema_version": 1,
                "sample_seq": 1,
            }
        ]
        self.assertEqual(
            equivalence.existing_event_counts(baseline),
            equivalence.existing_event_counts(observed),
        )
        self.assertEqual(
            equivalence.existing_schema(baseline),
            equivalence.existing_schema(observed),
        )


if __name__ == "__main__":
    unittest.main()
