#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TRACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACE_DIR))

from trace_metrics import collect_core_metrics, inference_latency_records  # noqa: E402
from analyze_trace import collect_metrics  # noqa: E402


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

    def test_feedback_slack_and_prediction_metrics(self) -> None:
        memory = [
            {
                "event": "OS_HINT",
                "action": "expert_madvise_willneed_predicted",
                "decision": "prefetch",
                "result": 0,
                "advised_bytes": 4096,
                "route_confidence": 0.75,
                "value_ratio": 2.0,
                "slack_ns": 800_000,
                "predicted": True,
            },
            {
                "event": "OS_HINT",
                "action": "expert_prefetch_cancel_expired",
                "decision": "skip",
                "route_confidence": 0.2,
                "value_ratio": 0.4,
                "slack_ns": 0,
                "predicted": True,
            },
            {
                "event": "EXPERT_ASYNC_SUMMARY",
                "enqueued": 5,
                "issued": 3,
                "max_queued_bytes": 8 * 1024**2,
                "cancelled_expired": 1,
                "cancelled_pressure": 1,
                "cancelled_value": 0,
                "coalesced_syscalls_saved": 2,
                "batch_size": 8,
            },
            {
                "event": "EXPERT_PRESSURE",
                "level": "high",
                "memory_ratio_pct": 91.0,
                "psi_some_avg10": 2.5,
                "psi_full_avg10": 0.1,
                "refault_delta": 128,
                "prefetch_budget_bytes": 256 * 1024**2,
            },
            {
                "event": "EXPERT_PREDICT_SUMMARY",
                "observed_routes": 20,
                "learned_transitions": 10,
                "transition_buckets": 4,
                "prediction_sets": 5,
                "prediction_candidates": 10,
                "evaluated_sets": 4,
                "evaluated_candidates": 8,
                "prediction_hits": 6,
                "prediction_set_hits": 4,
                "actual_experts_evaluated": 32,
                "unevaluated_sets": 1,
                "capacity_skips": 0,
            },
        ]
        metrics = collect_metrics({"memory": memory, "tensor": [], "kv": [], "expert": []})
        self.assertEqual(metrics["expert_predicted_records"], 2)
        self.assertEqual(metrics["expert_controller_cancelled_total"], 1)
        self.assertEqual(metrics["expert_async_coalesced_syscalls_saved"], 2)
        self.assertEqual(metrics["expert_pressure_high_or_critical_samples"], 1)
        self.assertEqual(metrics["expert_pressure_refault_delta_total"], 128)
        self.assertEqual(metrics["expert_prediction_candidates"], 10)
        self.assertEqual(metrics["expert_prediction_precision_pct"], 75.0)
        self.assertEqual(metrics["expert_prediction_recall_pct"], 18.75)


if __name__ == "__main__":
    unittest.main()
