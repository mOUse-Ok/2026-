#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TRACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACE_DIR))

from trace_metrics import collect_core_metrics, inference_latency_records  # noqa: E402
from analyze_trace import collect_expert_stage_pairing, collect_metrics  # noqa: E402


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

    def test_expert_task_lifecycle_metrics(self) -> None:
        def task_event(
                task_id: int,
                lifecycle_event: str,
                state: str,
                ts_ns: int,
                *,
                created: int,
                enqueued: int = 0,
                dequeued: int = 0,
                issued: int = 0,
                returned: int = 0,
                reason: str | None = None) -> dict:
            record = {
                "event": "EXPERT_TASK",
                "ts_ns": ts_ns,
                "lifecycle_event": lifecycle_event,
                "state": state,
                "task_id": task_id,
                "step": 7,
                "layer": 3,
                "expert": 11,
                "phase": "DECODE",
                "stage": "LATE",
                "tensor": "blk.3.ffn_down_exps.weight",
                "addr": "0x1000",
                "nbytes": 4096,
                "score": 0.75,
                "created_ts_ns": created,
                "enqueued_ts_ns": enqueued,
                "dequeued_ts_ns": dequeued,
                "issued_ts_ns": issued,
                "queue_wait_ns": dequeued - enqueued if enqueued and dequeued >= enqueued else None,
            }
            if returned:
                record["returned_ts_ns"] = returned
            if lifecycle_event == "ISSUE":
                record["hint_status"] = "returned"
                record["issue_id"] = 17
                record["issue_task_count"] = 1
            if reason:
                record["reason"] = reason
            return record

        memory = [
            task_event(1, "CREATE", "CREATED", 100, created=100),
            task_event(1, "ADMIT", "ADMITTED", 110, created=100),
            task_event(1, "ENQUEUE", "ENQUEUED", 120, created=100, enqueued=120),
            task_event(1, "DEQUEUE", "DEQUEUED", 170, created=100, enqueued=120, dequeued=170),
            task_event(
                1, "ISSUE", "ISSUED", 210,
                created=100, enqueued=120, dequeued=170, issued=180, returned=210,
            ),
            task_event(2, "CREATE", "CREATED", 300, created=300),
            task_event(2, "REJECT", "REJECTED", 310, created=300, reason="pressure_budget"),
            task_event(3, "CREATE", "CREATED", 400, created=400),
            task_event(3, "ADMIT", "ADMITTED", 410, created=400),
            task_event(3, "ENQUEUE", "ENQUEUED", 420, created=400, enqueued=420),
            task_event(3, "DEQUEUE", "DEQUEUED", 450, created=400, enqueued=420, dequeued=450),
            task_event(
                3, "CANCEL", "CANCELLED", 460,
                created=400, enqueued=420, dequeued=450, reason="deadline_missed",
            ),
            {
                "event": "EXPERT_TASK_SUMMARY",
                "trace_mode": "detail",
                "detail_events_enabled": True,
                "created": 3,
                "admitted": 2,
                "rejected": 1,
                "enqueued": 2,
                "dequeued": 2,
                "issued": 1,
                "cancelled": 1,
                "terminal": 3,
                "in_flight": 0,
                "invalid_transitions": 0,
                "rejected_pressure": 1,
                "cancelled_expired": 1,
                "issue_groups": 1,
                "coalesced_issue_groups": 0,
                "same_stage_issue_groups": 1,
                "cross_stage_issue_groups": 0,
                "early_task_count": 0,
                "late_task_count": 3,
                "unknown_task_count": 0,
                "queue_wait_ns_by_stage": {
                    "EARLY": {"count": 0, "total_ns": 0, "min_ns": 0, "max_ns": 0},
                    "LATE": {"count": 2, "total_ns": 80, "min_ns": 30, "max_ns": 50},
                    "UNKNOWN": {"count": 0, "total_ns": 0, "min_ns": 0, "max_ns": 0},
                },
            },
            {
                "event": "OS_HINT",
                "action": "expert_madvise_willneed",
                "decision": "prefetch",
                "issue_id": 17,
                "advised_bytes": 4096,
                "result": 0,
            },
            {
                "event": "EXPERT_FIRST_USE",
                "semantics": "logical_first_use",
                "physical_load_observed": False,
                "matched": True,
                "match_count": 1,
                "match_index": 0,
                "ambiguous_match": False,
                "task_id": 1,
                "issue_id": 17,
                "step": 7,
                "layer": 3,
                "expert": 11,
                "phase": "DECODE",
                "stage": "LATE",
                "tensor": "blk.3.ffn_down_exps.weight",
                "addr": "0x1000",
                "nbytes": 4096,
                "issued_ts_ns": 180,
                "first_use_ts_ns": 260,
                "create_to_first_use_ns": 160,
                "issue_to_first_use_ns": 80,
                "queue_wait_ns": 50,
            },
            {
                "event": "EXPERT_FIRST_USE_SUMMARY",
                "semantics": "logical_first_use",
                "physical_load_observed": False,
                "eligible_tasks": 1,
                "logical_first_uses": 1,
                "matched_tasks": 1,
                "unmatched_tasks": 0,
                "unmatched_first_uses": 0,
                "ambiguous_matches": 0,
                "duplicate_first_use_ignored": 0,
                "matcher_peak_live_tasks": 1,
                "matcher_expired_tasks": 0,
                "late_issued_tasks": 0,
                "pending_issued_tasks": 0,
                "ignored_old_uses": 0,
                "create_to_first_use_ns_by_stage": {
                    "EARLY": {"count": 0, "total_ns": 0, "min_ns": 0, "max_ns": 0},
                    "LATE": {"count": 1, "total_ns": 160, "min_ns": 160, "max_ns": 160},
                    "UNKNOWN": {"count": 0, "total_ns": 0, "min_ns": 0, "max_ns": 0},
                },
            },
        ]
        metrics = collect_metrics({"memory": memory, "tensor": [], "kv": [], "expert": []})
        self.assertEqual(metrics["expert_task_created"], 3)
        self.assertEqual(metrics["expert_task_terminal"], 3)
        self.assertEqual(metrics["expert_task_unique_ids"], 3)
        self.assertEqual(metrics["expert_task_trace_issue"], 1)
        self.assertEqual(metrics["expert_task_trace_cancel"], 1)
        self.assertEqual(metrics["expert_task_reason_pressure_budget"], 1)
        self.assertEqual(metrics["expert_task_trace_invalid_transitions"], 0)
        self.assertEqual(metrics["expert_task_trace_incomplete"], 0)
        self.assertEqual(metrics["expert_task_hint_returned_records"], 1)
        self.assertEqual(metrics["expert_issue_unique_ids"], 1)
        self.assertEqual(metrics["expert_issue_linked_syscalls"], 1)
        self.assertEqual(metrics["expert_issue_ids_without_syscalls"], 0)
        self.assertEqual(metrics["expert_syscall_issue_ids_without_tasks"], 0)
        self.assertEqual(metrics["expert_issue_task_count_mismatches"], 0)
        self.assertEqual(metrics["expert_first_use_matched_tasks"], 1)
        self.assertEqual(metrics["expert_first_use_summary_semantic_violations"], 0)
        self.assertEqual(metrics["expert_first_use_semantic_violations"], 0)
        self.assertEqual(metrics["expert_first_use_task_link_mismatches"], 0)
        self.assertEqual(metrics["expert_first_use_task_match_rate_pct"], 100.0)
        self.assertAlmostEqual(metrics["expert_first_use_issue_to_first_use_p50_us"], 0.08)
        self.assertAlmostEqual(metrics["expert_task_queue_wait_p50_us"], 0.04)
        self.assertAlmostEqual(metrics["expert_task_create_to_issue_p50_us"], 0.08)
        self.assertAlmostEqual(metrics["expert_task_hint_return_p50_us"], 0.03)

    def test_expert_task_summary_without_detail_events(self) -> None:
        memory = [{
            "event": "EXPERT_TASK_SUMMARY",
            "trace_mode": "summary",
            "detail_events_enabled": False,
            "created": 10,
            "admitted": 8,
            "rejected": 2,
            "enqueued": 6,
            "dequeued": 6,
            "issued": 7,
            "cancelled": 1,
            "terminal": 10,
            "in_flight": 0,
            "invalid_transitions": 0,
            "issue_groups": 7,
            "coalesced_issue_groups": 2,
            "same_stage_issue_groups": 6,
            "cross_stage_issue_groups": 1,
            "early_task_count": 4,
            "late_task_count": 5,
            "unknown_task_count": 1,
            "queue_wait_ns_by_stage": {
                "EARLY": {"count": 2, "total_ns": 40, "min_ns": 10, "max_ns": 30},
                "LATE": {"count": 1, "total_ns": 50, "min_ns": 50, "max_ns": 50},
                "UNKNOWN": {"count": 0, "total_ns": 0, "min_ns": 0, "max_ns": 0},
            },
        }]
        metrics = collect_metrics({"memory": memory, "tensor": [], "kv": [], "expert": []})
        self.assertEqual(metrics["expert_task_detail_events_enabled"], 0)
        self.assertEqual(metrics["expert_task_created"], 10)
        self.assertEqual(metrics["expert_task_issued"], 7)
        self.assertEqual(metrics["expert_task_trace_mode"], "summary")
        self.assertEqual(metrics["expert_task_issue_groups"], 7)
        self.assertEqual(metrics["expert_task_coalesced_issue_groups"], 2)
        self.assertEqual(metrics["expert_task_cross_stage_issue_groups"], 1)
        self.assertEqual(metrics["expert_task_early_task_count"], 4)
        self.assertEqual(metrics["expert_task_queue_wait_ns_by_stage"]["LATE"]["total_ns"], 50)
        self.assertEqual(metrics["expert_controller_cancelled_total"], 3)
        self.assertNotIn("expert_task_detail_records", metrics)

    def test_coalesced_issue_group_is_one_to_many(self) -> None:
        def issued(task_id: int) -> dict:
            return {
                "event": "EXPERT_TASK",
                "lifecycle_event": "ISSUE",
                "state": "ISSUED",
                "task_id": task_id,
                "issue_id": 91,
                "issue_task_count": 2,
                "step": 4,
                "layer": 2,
                "expert": task_id,
                "phase": "PREFILL",
                "stage": "LATE",
                "tensor": "blk.2.ffn_down_exps.weight",
                "addr": f"0x{task_id * 4096:x}",
                "nbytes": 4096,
                "score": 0.5,
                "ts_ns": 200,
                "created_ts_ns": 100,
                "enqueued_ts_ns": 120,
                "dequeued_ts_ns": 150,
                "issued_ts_ns": 180,
                "returned_ts_ns": 200,
                "queue_wait_ns": 30,
            }

        memory = [
            issued(1),
            issued(2),
            {
                "event": "OS_HINT",
                "action": "expert_madvise_willneed_batch",
                "issue_id": 91,
                "advised_bytes": 8192,
                "result": 0,
            },
        ]
        metrics = collect_metrics({"memory": memory, "tensor": [], "kv": [], "expert": []})
        self.assertEqual(metrics["expert_issue_unique_ids"], 1)
        self.assertEqual(metrics["expert_issue_coalesced_groups"], 1)
        self.assertEqual(metrics["expert_issue_coalesced_tasks"], 2)
        self.assertEqual(metrics["expert_issue_max_tasks_per_group"], 2)
        self.assertEqual(metrics["expert_issue_task_count_mismatches"], 0)
        self.assertEqual(metrics["expert_issue_ids_with_syscalls"], 1)

    def test_stage_pairing_uses_trace_stage_and_collapses_one_to_many_records(self) -> None:
        def first_use(
                run_id: str,
                step: int,
                layer: int,
                expert: int,
                tensor: str,
                stage: str,
                phase: str,
                ts_ns: int,
                task_id: int) -> dict:
            return {
                "event": "EXPERT_FIRST_USE",
                "run_id": run_id,
                "step": step,
                "layer": layer,
                "expert": expert,
                "tensor": tensor,
                "stage": stage,
                "phase": phase,
                "first_use_ts_ns": ts_ns,
                "task_id": task_id,
                "matched": True,
            }

        records = [
            first_use("r1", 1, 2, 3, "blk.2.ffn_down_exps.weight", "EARLY", "PREFILL", 100, 1),
            first_use("r1", 1, 2, 3, "blk.2.ffn_down_exps.weight", "EARLY", "PREFILL", 100, 2),
            first_use("r1", 1, 2, 3, "blk.2.ffn_down_exps.weight", "LATE", "PREFILL", 160, 3),
            first_use("r1", 2, 2, 4, "arbitrary.tensor", "EARLY", "DECODE", 300, 4),
            first_use("r1", 2, 2, 4, "also.arbitrary", "LATE", "DECODE", 250, 5),
            first_use("r1", 3, 5, 6, "early.only", "EARLY", "DECODE", 400, 6),
            first_use("r1", 4, 5, 7, "unknown.only", "UNKNOWN", "DECODE", 500, 7),
        ]
        pairing = collect_expert_stage_pairing(records)
        self.assertEqual(pairing["stage_source"], "trace_field")
        self.assertEqual(pairing["paired_experts"], 2)
        self.assertEqual(pairing["late_after_early_count"], 1)
        self.assertEqual(pairing["late_before_early_count"], 1)
        self.assertEqual(pairing["equal_timestamp_count"], 0)
        self.assertEqual(pairing["late_after_early_ratio"], 0.5)
        self.assertEqual(pairing["delta_ns"]["p25"], -22.5)
        self.assertEqual(pairing["by_phase"]["PREFILL"]["paired_experts"], 1)
        self.assertEqual(pairing["by_phase"]["DECODE"]["paired_experts"], 1)
        self.assertEqual(pairing["by_layer"]["2"]["paired_experts"], 2)
        self.assertEqual(pairing["unmatched_reasons"]["missing_late"], 1)
        self.assertEqual(pairing["unmatched_reasons"]["unknown_stage_only"], 1)
        self.assertEqual(pairing["duplicate_match_records_collapsed"], 1)

    def test_m1_summary_aggregates_duplicate_task_semantics(self) -> None:
        memory = [{
            "event": "EXPERT_FIRST_USE_SUMMARY",
            "semantics": "logical_first_use",
            "physical_load_observed": False,
            "eligible_tasks": 5,
            "logical_first_uses": 3,
            "matched_tasks": 3,
            "unmatched_tasks": 2,
            "unmatched_first_uses": 1,
            "ambiguous_matches": 1,
            "duplicate_first_use_ignored": 2,
            "matcher_peak_live_tasks": 4,
            "matcher_expired_tasks": 1,
            "late_issued_tasks": 1,
            "pending_issued_tasks": 0,
            "ignored_old_uses": 0,
            "create_to_first_use_ns_by_stage": {
                "EARLY": {"count": 2, "total_ns": 300, "min_ns": 100, "max_ns": 200},
                "LATE": {"count": 1, "total_ns": 250, "min_ns": 250, "max_ns": 250},
                "UNKNOWN": {"count": 0, "total_ns": 0, "min_ns": 0, "max_ns": 0},
            },
        }]
        metrics = collect_metrics({"memory": memory, "tensor": [], "kv": [], "expert": []})
        self.assertEqual(metrics["expert_first_use_eligible_tasks"], 5)
        self.assertEqual(metrics["expert_first_use_matched_tasks"], 3)
        self.assertEqual(metrics["expert_first_use_unmatched_tasks"], 2)
        self.assertEqual(metrics["expert_first_use_ambiguous_matches"], 1)
        self.assertEqual(metrics["expert_first_use_duplicate_first_use_ignored"], 2)
        self.assertEqual(metrics["expert_first_use_matcher_peak_live_tasks"], 4)
        self.assertEqual(metrics["expert_first_use_matcher_expired_tasks"], 1)
        self.assertEqual(metrics["expert_first_use_create_to_first_use_ns_by_stage"]["EARLY"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
