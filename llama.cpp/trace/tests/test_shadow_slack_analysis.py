import sys
import unittest
from pathlib import Path


TRACE_DIR = Path(__file__).resolve().parents[1]
if str(TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(TRACE_DIR))

from shadow_slack_analysis import (  # noqa: E402
    analyze_shadow_slack,
    calibration_bucket,
)


GROUPINGS = ("phase_layer", "phase_stage", "phase_layer_stage")
ESTIMATORS = ("ewma", "median", "p25")
QUEUES = ("queue_depth_worker_ewma", "queued_bytes_issue_throughput")
CALIBRATIONS = ("raw", "residual_quantile")


def predictions(
    prediction_ts,
    actual_queue,
    *,
    warmup=False,
    fallback="exact",
    force_late=False,
):
    records = []
    horizon_error = {
        "phase_layer": 100,
        "phase_stage": 50,
        "phase_layer_stage": 25,
    }
    for grouping in GROUPINGS:
        for estimator in ESTIMATORS:
            for queue in QUEUES:
                for calibration in CALIBRATIONS:
                    queue_error = 10 if queue == QUEUES[0] else -20
                    predicted_horizon = 1000 + horizon_error[grouping]
                    predicted_queue = actual_queue + queue_error
                    predicted_pre_issue = 2000 if force_late else 55
                    predicted_syscall = 15
                    issue_slack = (
                        predicted_horizon - predicted_queue - predicted_pre_issue
                    )
                    records.append({
                        "predicted_first_use_ts_ns": prediction_ts + predicted_horizon,
                        "predicted_first_use_horizon_ns": predicted_horizon,
                        "raw_predicted_first_use_horizon_ns": predicted_horizon,
                        "residual_adjustment_ns": 0,
                        "predicted_queue_wait_ns": predicted_queue,
                        "predicted_pre_issue_overhead_ns": predicted_pre_issue,
                        "predicted_hint_syscall_service_ns": predicted_syscall,
                        "predicted_worker_occupied_ns": 70,
                        "predicted_issue_slack_ns": issue_slack,
                        "predicted_return_slack_ns": issue_slack - predicted_syscall,
                        "estimator_sample_count": 8,
                        "residual_sample_count": 8 if calibration != "raw" else 0,
                        "estimator_warmup": warmup,
                        "residual_warmup": warmup if calibration != "raw" else False,
                        "queue_warmup": warmup,
                        "pre_issue_warmup": warmup,
                        "syscall_service_warmup": warmup,
                        "worker_warmup": warmup,
                        "deadline_model": f"{grouping}_{estimator}",
                        "queue_model": queue,
                        "calibration_model": calibration,
                        "fallback_level": fallback,
                        "residual_fallback_level": (
                            fallback if calibration != "raw" else "exact"
                        ),
                        "queue_fallback_level": fallback,
                        "pre_issue_fallback_level": fallback,
                        "syscall_service_fallback_level": fallback,
                        "worker_fallback_level": fallback,
                        "prediction_available": True,
                        "issue_prediction_available": True,
                        "return_prediction_available": True,
                        "clipped_low": False,
                        "clipped_high": False,
                    })
    return records


def shadow_record(
    task_id,
    *,
    issue_ts,
    first_use_ts,
    phase="DECODE",
    stage="EARLY",
    layer=3,
    workers=4,
    warmup=False,
    fallback="exact",
    force_late=False,
):
    prediction_ts = first_use_ts - 1000
    enqueue_ts = prediction_ts
    dequeue_ts = enqueue_ts + 100
    returned_ts = issue_ts + 10
    actual_queue = dequeue_ts - enqueue_ts
    actual_pre = issue_ts - dequeue_ts
    actual_syscall = returned_ts - issue_ts
    actual_occupied = returned_ts - dequeue_ts
    issue_slack = first_use_ts - issue_ts
    return_slack = first_use_ts - returned_ts
    return {
        "event": "EXPERT_SHADOW_SLACK",
        "schema_version": 2,
        "run_id": "r1",
        "semantics": "logical_first_use",
        "physical_load_observed": False,
        "issue_target": "issue_ts < logical_first_use_ts",
        "return_target": "final_enabled_hint_return_ts < logical_first_use_ts",
        "task_id": task_id,
        "issue_id": task_id,
        "issue_task_count": 1,
        "prediction_ts_ns": prediction_ts,
        "enqueued_ts_ns": enqueue_ts,
        "dequeued_ts_ns": dequeue_ts,
        "issue_ts_ns": issue_ts,
        "returned_ts_ns": returned_ts,
        "first_use_ts_ns": first_use_ts,
        "phase": phase,
        "stage": stage,
        "layer": layer,
        "actual_first_use_horizon_ns": 1000,
        "actual_queue_wait_ns": actual_queue,
        "actual_pre_issue_overhead_ns": actual_pre,
        "actual_hint_syscall_service_ns": actual_syscall,
        "actual_worker_occupied_ns": actual_occupied,
        "actual_issue_slack_ns": issue_slack,
        "actual_return_slack_ns": return_slack,
        "issue_on_time": issue_slack > 0,
        "return_on_time": return_slack > 0,
        "active_workers": workers,
        "nbytes": 1024,
        "issued_nbytes": 1024,
        "coalesced": False,
        "predictions": predictions(
            prediction_ts,
            actual_queue,
            warmup=warmup,
            fallback=fallback,
            force_late=force_late,
        ),
    }


class ShadowSlackAnalysisTests(unittest.TestCase):
    def test_dual_targets_oracle_queue_and_stage_comparison(self):
        records = [
            shadow_record(1, issue_ts=200, first_use_ts=1000, stage="EARLY", layer=3),
            shadow_record(2, issue_ts=2100, first_use_ts=2000, stage="LATE", layer=4),
            shadow_record(
                3, issue_ts=3200, first_use_ts=3000, stage="LATE", layer=5,
                force_late=True,
            ),
        ]
        result = analyze_shadow_slack(records)

        self.assertEqual(result["source"], "detail")
        self.assertEqual(result["valid_unique_tasks"], 3)
        self.assertEqual(result["candidate_count"], 36)
        self.assertEqual(result["semantic_violations"], 0)
        key = "phase_layer_ewma|queue_depth_worker_ewma|raw"
        first_use = result["candidates"][key]["first_use"]["operational"]
        issue = result["candidates"][key]["issue"]["operational"]
        returned = result["candidates"][key]["return"]["operational"]
        self.assertEqual(first_use["count"], 3)
        self.assertEqual(first_use["mae_ns"], 100)
        self.assertEqual(issue["true_positive"], 1)
        self.assertEqual(issue["false_positive"], 1)
        self.assertEqual(issue["true_negative"], 1)
        self.assertEqual(issue["predicted_late_precision"], 1.0)
        self.assertEqual(returned["true_negative"], 1)
        self.assertIn("DECODE|workers=4", result["candidates"][key]["issue"]["by_phase_workers"])
        self.assertEqual(
            result["candidates"][key]["issue"]
            ["mature_by_phase_workers"]["DECODE|workers=4"]["count"],
            3,
        )

        queue_a = result["queue_models"][QUEUES[0]]
        queue_b = result["queue_models"][QUEUES[1]]
        self.assertEqual(queue_a["all_available"]["count"], 3)
        self.assertEqual(queue_a["all_available"]["mae_ns"], 10)
        self.assertEqual(queue_b["all_available"]["mae_ns"], 20)
        self.assertEqual(result["queue_paired_common_count"], 3)

        oracle_issue = result["oracle_attribution"][key]["issue"]
        oracle_return = result["oracle_attribution"][key]["return"]
        self.assertEqual(len(oracle_issue), 8)
        self.assertEqual(len(oracle_return), 16)
        self.assertEqual(
            oracle_issue["actual_first_use+actual_queue+actual_pre_issue"]["overall"]["mae_ns"],
            0,
        )
        self.assertEqual(
            oracle_return[
                "actual_first_use+actual_queue+actual_pre_issue+actual_syscall_service"
            ]["overall"]["mae_ns"],
            0,
        )
        self.assertEqual(
            oracle_issue["actual_first_use+actual_queue+actual_pre_issue"]
            ["by_phase"]["DECODE"]["mae_ns"],
            0,
        )
        self.assertEqual(
            oracle_issue["actual_first_use+actual_queue+actual_pre_issue"]
            ["by_stage"]["EARLY"]["count"],
            1,
        )
        self.assertEqual(
            oracle_issue["actual_first_use+actual_queue+actual_pre_issue"]
            ["by_active_workers"]["4"]["count"],
            3,
        )
        comparison = result["stage_comparisons"][
            "phase_layer_stage_ewma_vs_phase_layer_ewma"
            "|queue_depth_worker_ewma|raw"
        ]
        self.assertEqual(comparison["common_coverage_count"], 3)
        self.assertEqual(comparison["first_use"]["mae_delta_ns"], -75)

    def test_warmup_mature_fallback_and_causality(self):
        warmup = shadow_record(
            4, issue_ts=200, first_use_ts=1000,
            warmup=True, fallback="static_default",
        )
        result = analyze_shadow_slack([warmup])
        key = "phase_layer_ewma|queue_depth_worker_ewma|raw"
        issue = result["candidates"][key]["issue"]
        self.assertEqual(issue["operational"]["warmup_count"], 1)
        self.assertEqual(issue["operational"]["fallback_count"], 1)
        self.assertEqual(issue["mature_exact"]["count"], 0)
        self.assertEqual(issue["fallback_only"]["count"], 1)
        self.assertEqual(
            issue["mature_by_phase_workers"]["DECODE|workers=4"]["count"],
            0,
        )
        self.assertEqual(
            issue["fallback_by_phase_workers"]["DECODE|workers=4"]["count"],
            1,
        )

        oracle = result["oracle_attribution"][key]["issue"]
        predicted = oracle[
            "predicted_first_use+predicted_queue+predicted_pre_issue"
        ]["overall"]
        full = oracle[
            "actual_first_use+actual_queue+actual_pre_issue"
        ]["overall"]
        self.assertEqual(predicted["fallback_count"], 1)
        self.assertEqual(predicted["mature_exact_count"], 0)
        self.assertEqual(full["fallback_count"], 0)
        self.assertEqual(full["mature_exact_count"], 1)

        invalid = dict(warmup)
        invalid["task_id"] = 5
        invalid["dequeued_ts_ns"] = invalid["enqueued_ts_ns"] - 1
        result = analyze_shadow_slack([invalid])
        self.assertEqual(result["timestamp_regressions"], 1)
        self.assertEqual(result["candidate_count"], 0)

    def test_coalesced_component_models_are_not_double_counted(self):
        first = shadow_record(6, issue_ts=200, first_use_ts=1000)
        second = shadow_record(7, issue_ts=200, first_use_ts=1000)
        first.update({"issue_id": 50, "issue_task_count": 2, "coalesced": True})
        second.update({"issue_id": 50, "issue_task_count": 2, "coalesced": True})
        records = [
            first,
            second,
            {"event": "OS_HINT", "run_id": "r1", "issue_id": 50, "result": 0},
        ]
        result = analyze_shadow_slack(records)
        self.assertEqual(result["duration_models"]["pre_issue_overhead"]["all_available"]["count"], 2)
        self.assertEqual(result["duration_models"]["hint_syscall_service"]["unique_issue_groups"], 1)
        self.assertEqual(result["duration_models"]["worker_occupied"]["unique_issue_groups"], 1)
        self.assertEqual(result["multi_syscall_audit"]["coalesced_issue_groups"], 1)

    def test_old_trace_and_summary_only_are_compatible(self):
        old = analyze_shadow_slack([{"event": "EXPERT_TASK", "task_id": 1}])
        self.assertEqual(old["source"], "none")
        self.assertEqual(old["candidates"], {})

        summary = analyze_shadow_slack([{
            "event": "EXPERT_SHADOW_SLACK_SUMMARY",
            "mode": "shadow",
            "finalized_tasks": 7,
        }])
        self.assertEqual(summary["source"], "summary")
        self.assertEqual(summary["runtime_summary"]["finalized_tasks"], 7)

    def test_calibration_boundaries_are_unique(self):
        cases = (
            (-5_000_001, 0), (-5_000_000, 1), (-2_000_000, 2),
            (-1_000_000, 3), (-500_000, 4), (0, 4), (1, 5),
            (500_000, 5), (500_001, 6), (1_000_000, 6),
            (1_000_001, 7), (2_000_000, 7), (2_000_001, 8),
            (5_000_000, 8), (5_000_001, 9),
        )
        for value, expected in cases:
            self.assertEqual(calibration_bucket(value), expected)


if __name__ == "__main__":
    unittest.main()
