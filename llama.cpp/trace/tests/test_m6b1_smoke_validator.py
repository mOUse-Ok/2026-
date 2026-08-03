import copy
import sys
from pathlib import Path
import unittest


TRACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACE_DIR))

import validate_m6b1_smoke as validator


class M6B1SmokeValidatorTest(unittest.TestCase):
    @staticmethod
    def task(task_id, enqueued, dequeued, deadline, score, sequence):
        return {
            "event": "EXPERT_TASK",
            "lifecycle_event": "DEQUEUE",
            "task_id": task_id,
            "enqueued_ts_ns": enqueued,
            "dequeued_ts_ns": dequeued,
            "deadline_ts_ns": deadline,
            "score": score,
            "sequence": sequence,
        }

    @staticmethod
    def selection(task_id, decision, protected, normal):
        return {
            "event": "EXPERT_MAX_WAIT_SELECTION",
            "task_id": task_id,
            "decision_ts_ns": decision,
            "threshold_ns": 100,
            "urgent_guard_ns": 50,
            "protected_candidate_count": protected,
            "normal_competitor_present": normal,
        }

    def records(self):
        return [
            self.task(1, 100, 151, 200, 0.1, 1),
            self.task(2, 10, 161, 1000, 0.1, 2),
            self.task(3, 100, 171, 900, 0.9, 3),
            self.selection(1, 150, 1, True),
            self.selection(2, 160, 1, True),
            self.selection(3, 170, 0, True),
        ]

    def test_replays_urgent_protected_normal_with_one_pass_heaps(self):
        replay = validator.replay_candidate_selections(self.records())
        self.assertTrue(replay["passed"], replay)
        self.assertEqual(replay["selection_events"], 3)

    def test_detects_urgent_winner_bypass(self):
        records = copy.deepcopy(self.records())
        records[3]["task_id"] = 2
        replay = validator.replay_candidate_selections(records)
        self.assertFalse(replay["passed"])
        self.assertGreater(replay["urgent_winner_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
