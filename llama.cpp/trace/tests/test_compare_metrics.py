#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TRACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACE_DIR))

from compare_trace_runs import dominates, validate_official_metrics  # noqa: E402


class CompareMetricsTest(unittest.TestCase):
    def test_missing_metric_cannot_dominate_complete_row(self) -> None:
        incomplete = {
            "decode_p95_latency_us": 1,
            "process_wall_time_s": 1,
        }
        complete = {
            "decode_p95_latency_us": 2,
            "process_wall_time_s": 2,
            "total_major_faults": 2,
            "rss_peak_gb": 2,
            "swap_peak_mb": 2,
            "os_hint_events": 2,
        }
        self.assertFalse(dominates(incomplete, complete))

    def test_official_mode_requires_sources(self) -> None:
        metrics = {
            "decode_p95_latency_us": 1,
            "process_wall_time_s": 1,
            "total_major_faults": 1,
            "rss_peak_gb": 1,
            "swap_peak_mb": 1,
            "os_hint_events": 1,
        }
        with self.assertRaisesRegex(ValueError, "GNU time"):
            validate_official_metrics("run", metrics)

    def test_official_mode_rejects_prefill_only_smoke(self) -> None:
        metrics = {
            "decode_p95_latency_us": 0,
            "process_wall_time_s": 1,
            "total_major_faults": 1,
            "rss_peak_gb": 1,
            "swap_peak_mb": 1,
            "os_hint_events": 0,
            "fault_metric_source": "gnu_time_process",
            "latency_metric_source": "step_end",
            "decode_steps": 0,
        }
        with self.assertRaisesRegex(ValueError, "decode step"):
            validate_official_metrics("run", metrics)


if __name__ == "__main__":
    unittest.main()
