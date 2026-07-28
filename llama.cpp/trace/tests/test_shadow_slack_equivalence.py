import json
import sys
import tempfile
import unittest
from pathlib import Path


TRACE_DIR = Path(__file__).resolve().parents[1]
if str(TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(TRACE_DIR))

from validate_shadow_slack_equivalence import validate_detail_integrity  # noqa: E402


class ShadowSlackEquivalenceTests(unittest.TestCase):
    def test_full_detail_lifecycle_and_linkage_validation(self):
        records = []
        states = (
            ("CREATE", "CREATED"),
            ("ADMIT", "ADMITTED"),
            ("ENQUEUE", "ENQUEUED"),
            ("DEQUEUE", "DEQUEUED"),
            ("ISSUE", "ISSUED"),
        )
        for index, (event, state) in enumerate(states, 1):
            record = {
                "event": "EXPERT_TASK",
                "task_id": 1,
                "lifecycle_event": event,
                "state": state,
                "ts_ns": index,
            }
            if event == "ISSUE":
                record.update({"issue_id": 7, "issue_task_count": 1})
            records.append(record)
        records.extend((
            {"event": "OS_HINT", "issue_id": 7, "result": 0},
            {
                "event": "EXPERT_SHADOW_SLACK",
                "schema_version": 2,
                "semantics": "logical_first_use",
                "physical_load_observed": False,
                "issue_target": "issue_ts < logical_first_use_ts",
                "return_target": "final_enabled_hint_return_ts < logical_first_use_ts",
                "prediction_ts_ns": 1,
                "issue_ts_ns": 3,
                "returned_ts_ns": 4,
                "first_use_ts_ns": 5,
                "actual_issue_slack_ns": 2,
                "actual_return_slack_ns": 1,
                "issue_on_time": True,
                "return_on_time": True,
                "predictions": [{
                    "predicted_first_use_horizon_ns": 10,
                    "predicted_queue_wait_ns": 2,
                    "predicted_pre_issue_overhead_ns": 3,
                    "predicted_hint_syscall_service_ns": 1,
                    "predicted_issue_slack_ns": 5,
                    "predicted_return_slack_ns": 4,
                }],
            },
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory_trace.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = validate_detail_integrity(Path(directory))
        self.assertTrue(result["passed"])
        self.assertEqual(result["unique_task_ids"], 1)
        self.assertEqual(result["unique_issue_ids"], 1)
        self.assertEqual(result["issue_ids_without_syscalls"], 0)

    def test_missing_syscall_and_invalid_transition_fail(self):
        records = [
            {
                "event": "EXPERT_TASK",
                "task_id": 1,
                "lifecycle_event": "CREATE",
                "state": "CREATED",
                "ts_ns": 1,
            },
            {
                "event": "EXPERT_TASK",
                "task_id": 1,
                "lifecycle_event": "ISSUE",
                "state": "ISSUED",
                "issue_id": 8,
                "issue_task_count": 1,
                "ts_ns": 2,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory_trace.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = validate_detail_integrity(Path(directory))
        self.assertFalse(result["passed"])
        self.assertEqual(result["invalid_transitions"], 1)
        self.assertEqual(result["issue_ids_without_syscalls"], 1)


if __name__ == "__main__":
    unittest.main()
