#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TRACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACE_DIR))

import validate_m6b2_1_queue_overhead as validator  # noqa: E402


class M6B21QueueOverheadValidatorTest(unittest.TestCase):
    @staticmethod
    def aggregate(count: int, total: int) -> dict:
        return {
            "count": count,
            "total": total,
            "unavailable_count": 0,
        }

    def test_summary_balance(self) -> None:
        records = [{
            "event": "EXPERT_QUEUE_OVERHEAD_SUMMARY",
            "schema_version": "m6b2.1-queue-overhead-v1",
            "mode": "summary",
            "semantics": "direct_queue_selection_measurement",
            "physical_load_observed": False,
            "priority_mode": "deadline_score",
            "workers": 2,
            "scheduler_batch": 1,
            "selection_count": 3,
            "batch_count": 3,
            "clock_regression_count": 0,
            "overflow_count": 0,
            "detail_event_count": 0,
            "next_decision_id": 3,
            "next_batch_id": 3,
            "clock_read_count": 160,
            "condition_wait_count": 2,
            "condition_reacquire_count": 2,
            "idle_wait_exit_count": 2,
            "global": {
                "mutex_acquire_wait_ns": self.aggregate(3, 6),
                "mutex_hold_ns": self.aggregate(3, 30),
                "queue_scan_ns": self.aggregate(3, 12),
                "queue_scan_candidates": self.aggregate(3, 9),
            },
        }]
        metrics = {
            "expert_async_priority_pops": 3,
            "expert_task_dequeued": 3,
            "expert_queue_overhead_available": True,
            "expert_queue_overhead_schema_violations": 0,
            "expert_queue_overhead_priority_pop_mismatch": 0,
        }
        result = validator.queue_summary_check(
            records, metrics, "deadline_score", 2
        )
        self.assertTrue(result["passed"], result)

    def test_off_environment_and_engineering_markers(self) -> None:
        environment = dict(validator.COMMON_ENV)
        environment.update({
            "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE": "deadline_score",
            "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS": "4",
            "LLM_MEM_TRACE_QUEUE_OVERHEAD_MODE": "off",
        })
        result = validator.environment_check(
            {"environment": environment},
            "deadline_score",
            "off",
            4,
        )
        self.assertTrue(result["passed"], result)

    def test_invalid_summary_is_rejected(self) -> None:
        result = validator.queue_summary_check(
            [],
            {
                "expert_async_priority_pops": 0,
                "expert_task_dequeued": 0,
            },
            "max_wait_protection",
            2,
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
