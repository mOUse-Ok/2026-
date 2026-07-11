#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TRACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACE_DIR))

from trace_metrics import collect_core_metrics, inference_latency_records  # noqa: E402


class AnalysisMetricsTest(unittest.TestCase):
    def test_step_latency_and_fault_window_are_authoritative(self) -> None:
        memory = [
            {"event": "TRACE_START", "ts_ns": 100},
            {"event": "MEMORY_STAT", "ts_ns": 110, "rss_bytes": 1024, "minor_faults": 100, "major_faults": 10},
            {"event": "TOKEN_END", "ts_ns": 200, "phase": "PREFILL", "latency_ns": 99_000_000},
            {"event": "STEP_END", "ts_ns": 2_000_100, "phase": "PREFILL", "step": 1, "n_tokens": 4, "latency_ns": 2_000_000},
            {"event": "MEMORY_STAT", "ts_ns": 2_000_200, "rss_bytes": 2048, "minor_faults": 140, "major_faults": 13},
            {"event": "STEP_END", "ts_ns": 5_000_100, "phase": "DECODE", "step": 2, "n_tokens": 1, "latency_ns": 3_000_000},
            {"event": "MEMORY_STAT", "ts_ns": 5_000_200, "rss_bytes": 1536, "minor_faults": 175, "major_faults": 17},
        ]
        latency, source = inference_latency_records(memory)
        self.assertEqual(source, "step_end")
        self.assertEqual(len(latency), 2)

        metrics = collect_core_metrics(memory)
        self.assertEqual(metrics["trace_window_minor_faults"], 75)
        self.assertEqual(metrics["trace_window_major_faults"], 7)
        self.assertEqual(metrics["fault_metric_source"], "trace_window")
        self.assertEqual(metrics["latency_metric_source"], "step_end")
        self.assertEqual(metrics["prefill_tokens"], 4)
        self.assertEqual(metrics["decode_tokens"], 1)
        self.assertEqual(metrics["prefill_avg_latency_us"], 2000.0)
        self.assertEqual(metrics["decode_p95_latency_us"], 3000.0)
        self.assertAlmostEqual(metrics["decode_throughput_tokens_per_s"], 1 / 0.003)

    def test_legacy_token_latency_remains_an_explicit_fallback(self) -> None:
        records = [{"event": "TOKEN_END", "phase": "DECODE", "latency_ns": 42_000}]
        latency, source = inference_latency_records(records)
        self.assertEqual(source, "token_end_legacy")
        self.assertEqual(latency, records)


if __name__ == "__main__":
    unittest.main()
