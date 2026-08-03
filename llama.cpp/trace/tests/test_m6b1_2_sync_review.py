import copy
import sys
from pathlib import Path
import unittest


TRACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACE_DIR))

import validate_m6b1_2_sync_review as validator


class M6B12SyncReviewTest(unittest.TestCase):
    def pair(self):
        replay = {
            "passed": True,
            "self_replay_winner_mismatches": 0,
            "score_injection_winner_changed_decisions": 0,
        }
        return {
            "valid": True,
            "matching": {
                "unmatched_a_tasks": 0, "unmatched_b_tasks": 0,
                "unmatched_a_route_records": 0, "unmatched_b_route_records": 0,
            },
            "router": {
                "raw_bit_different_items": 0,
                "f32_bit_different_items": 0,
                "topk_set_changed_records": 0,
                "topk_rank_changed_items": 0,
            },
            "tasks": {
                "score_f64_bit_different_tasks": 0,
                "serialized_score_different_tasks": 0,
                "source_to_task_widening_mismatches": 0,
                "lifecycle_score_bit_mismatches": 0,
            },
            "replay_a_with_b_scores": replay,
            "replay_b_with_a_scores": replay,
        }

    def test_zero_mismatch_pair_passes(self):
        self.assertTrue(validator.pair_gate(self.pair())["passed"])

    def test_one_raw_bit_mismatch_fails(self):
        pair = copy.deepcopy(self.pair())
        pair["router"]["raw_bit_different_items"] = 1
        result = validator.pair_gate(pair)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["raw_f32_bits_equal"])


if __name__ == "__main__":
    unittest.main()
