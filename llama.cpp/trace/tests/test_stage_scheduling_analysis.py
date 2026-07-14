#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TRACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACE_DIR))

from stage_scheduling_analysis import (  # noqa: E402
    analyze_stage_inversions,
    classify_no_issued_task_reasons,
    compare_stage_schedules,
    extract_simulation_tasks,
    simulate_task_schedule,
)


def task_event(
        task_id: int,
        lifecycle_event: str,
        ts_ns: int,
        stage: str,
        *,
        layer: int = 3,
        step: int = 7,
        expert: int | None = None,
        state: str | None = None,
        reason: str | None = None) -> dict:
    record = {
        "event": "EXPERT_TASK",
        "run_id": "run",
        "task_id": task_id,
        "lifecycle_event": lifecycle_event,
        "state": state or {"ENQUEUE": "ENQUEUED", "DEQUEUE": "DEQUEUED", "ISSUE": "ISSUED"}.get(
            lifecycle_event, lifecycle_event + "ED"
        ),
        "ts_ns": ts_ns,
        "step": step,
        "layer": layer,
        "expert": task_id if expert is None else expert,
        "phase": "PREFILL",
        "stage": stage,
        "tensor": f"blk.{layer}.{'ffn_down_exps.weight' if stage == 'LATE' else 'ffn_up_exps.weight'}",
        "nbytes": 100,
        "score": float(task_id),
    }
    if reason:
        record["reason"] = reason
    return record


def simulation_job(
        task_id: int,
        stage: str,
        score: float,
        *,
        layer: int = 0,
        arrival: int = 0,
        service: int = 10,
        first_use: int = 100,
        sequence: int | None = None) -> dict:
    return {
        "identity": ("run", task_id),
        "task_id": task_id,
        "run_id": "run",
        "arrival_ts_ns": arrival,
        "service_ns": service,
        "deadline_ts_ns": 0,
        "first_use_ts_ns": first_use,
        "phase": "PREFILL",
        "stage": stage,
        "step": 1,
        "layer": layer,
        "score": score,
        "sequence": task_id if sequence is None else sequence,
        "nbytes": 100,
    }


class StageSchedulingAnalysisTest(unittest.TestCase):
    def test_no_issued_reason_classification_requires_explicit_evidence(self) -> None:
        records: list[dict] = []
        cases = [
            (1, "ttl_duplicate", {"event": "OS_HINT", "decision": "ttl_duplicate"}),
            (2, "cache_hit", {"event": "OS_HINT", "cache_hit": True}),
            (3, "policy_skip", {
                "event": "EXPERT_TASK", "lifecycle_event": "REJECT", "state": "REJECTED",
                "reason": "pressure_budget", "task_id": 103,
            }),
            (4, "queue_or_worker_failure", {
                "event": "EXPERT_TASK", "lifecycle_event": "CANCEL", "state": "CANCELLED",
                "reason": "queue_full", "task_id": 104,
            }),
            (5, "shutdown_pending", {
                "event": "EXPERT_TASK", "lifecycle_event": "ENQUEUE", "state": "ENQUEUED",
                "task_id": 105,
            }),
            (6, "other", None),
        ]
        for expert, expected, evidence in cases:
            common = {
                "run_id": "run",
                "step": 1,
                "layer": 2,
                "expert": expert,
                "tensor": "blk.2.ffn_up_exps.weight",
                "phase": "DECODE",
                "stage": "EARLY",
            }
            if evidence:
                records.append({**common, **evidence, "ts_ns": 10})
            records.append({
                **common,
                "event": "EXPERT_FIRST_USE",
                "ts_ns": 20,
                "first_use_ts_ns": 20,
                "unmatched_reason": "no_issued_task",
            })

        result = classify_no_issued_task_reasons(records)
        self.assertEqual(result["total_no_issued_task"], 6)
        for _, expected, _ in cases:
            self.assertEqual(result["reason_counts"][expected], 1)
            self.assertEqual(result["by_phase_stage"]["DECODE"]["EARLY"][expected], 1)
        self.assertEqual(result["classification_semantics"], "explicit_trace_evidence_else_other")

    def test_no_inversion(self) -> None:
        records = [
            task_event(1, "ENQUEUE", 1, "EARLY", expert=1),
            task_event(2, "ENQUEUE", 2, "LATE", expert=2),
            task_event(1, "DEQUEUE", 3, "EARLY", expert=1),
            task_event(2, "DEQUEUE", 4, "LATE", expert=2),
        ]
        result = analyze_stage_inversions(records)
        self.assertEqual(result["eligible_late_tasks"], 1)
        self.assertEqual(result["inversion_count"], 0)
        self.assertEqual(result["inversion_ratio"], 0.0)

    def test_all_late_tasks_are_inverted(self) -> None:
        records = [
            task_event(1, "ENQUEUE", 1, "EARLY", expert=1),
            task_event(2, "ENQUEUE", 1, "EARLY", expert=2),
            task_event(3, "ENQUEUE", 1, "LATE", expert=3),
            task_event(4, "ENQUEUE", 1, "LATE", expert=4),
            task_event(3, "DEQUEUE", 2, "LATE", expert=3),
            task_event(4, "DEQUEUE", 3, "LATE", expert=4),
            task_event(1, "DEQUEUE", 5, "EARLY", expert=1),
            task_event(2, "DEQUEUE", 6, "EARLY", expert=2),
        ]
        result = analyze_stage_inversions(records)
        self.assertEqual(result["eligible_late_tasks"], 2)
        self.assertEqual(result["inversion_count"], 2)
        self.assertEqual(result["inversion_ratio"], 1.0)
        self.assertEqual(result["blocked_early_tasks"], 4)
        self.assertEqual(result["unique_blocked_early_tasks"], 2)
        self.assertEqual(result["inverted_late_bytes"], 200)
        self.assertAlmostEqual(result["early_blocked_by_late_ns"]["p95"], 3.85)

    def test_equal_timestamp_inversion_uses_event_order(self) -> None:
        late_first = [
            task_event(1, "ENQUEUE", 1, "EARLY", expert=1),
            task_event(2, "ENQUEUE", 1, "LATE", expert=2),
            task_event(2, "DEQUEUE", 10, "LATE", expert=2),
            task_event(1, "DEQUEUE", 10, "EARLY", expert=1),
        ]
        early_first = [
            task_event(1, "ENQUEUE", 1, "EARLY", expert=1),
            task_event(2, "ENQUEUE", 1, "LATE", expert=2),
            task_event(1, "DEQUEUE", 10, "EARLY", expert=1),
            task_event(2, "DEQUEUE", 10, "LATE", expert=2),
        ]
        self.assertEqual(analyze_stage_inversions(late_first)["inversion_count"], 1)
        self.assertEqual(analyze_stage_inversions(early_first)["inversion_count"], 0)

    def test_multi_worker_simulation_and_equal_arrivals(self) -> None:
        jobs = [
            simulation_job(1, "LATE", 10.0, service=20),
            simulation_job(2, "EARLY", 1.0, service=20),
            simulation_job(3, "EARLY", 0.5, service=20),
        ]
        one = compare_stage_schedules(jobs, 1)
        two = compare_stage_schedules(jobs, 2)
        four = compare_stage_schedules(jobs, 4)
        self.assertEqual(one["early_issue_advancement_ns"]["p95"], 20.0)
        self.assertEqual(one["late_issue_delay_ns"]["p95"], 40.0)
        self.assertGreater(two["unchanged_task_ratio"], one["unchanged_task_ratio"])
        self.assertEqual(four["unchanged_task_ratio"], 1.0)

    def test_stage_policy_does_not_let_next_layer_early_pass_current_layer_late(self) -> None:
        jobs = [
            simulation_job(1, "LATE", 0.1, layer=0, sequence=1),
            simulation_job(2, "EARLY", 100.0, layer=1, sequence=2),
        ]
        issue = simulate_task_schedule(jobs, 1, "stage_deadline_score")
        self.assertEqual(issue[("run", 1)], 0)
        self.assertEqual(issue[("run", 2)], 10)

    def test_unknown_uses_legacy_deadline_score_fallback(self) -> None:
        known = simulation_job(1, "EARLY", 0.9, sequence=1)
        known["deadline_ts_ns"] = 300
        unknown = simulation_job(2, "UNKNOWN", 0.1, sequence=2)
        unknown["deadline_ts_ns"] = 100
        issue = simulate_task_schedule([known, unknown], 1, "stage_deadline_score")
        self.assertEqual(issue[("run", 2)], 0)
        self.assertEqual(issue[("run", 1)], 10)

    def test_late_starvation_risk_is_reported(self) -> None:
        jobs = [simulation_job(1, "LATE", 100.0, first_use=15, sequence=1)] + [
            simulation_job(task_id, "EARLY", 1.0, first_use=100, sequence=task_id)
            for task_id in range(2, 6)
        ]
        result = compare_stage_schedules(jobs, 1)
        self.assertEqual(result["newly_late_late_tasks"], 1)
        self.assertEqual(result["late_on_time_rate_before"], 1.0)
        self.assertEqual(result["late_on_time_rate_after"], 0.0)
        self.assertEqual(result["late_issue_delay_ns"]["p95"], 40.0)

    def test_extract_simulation_tasks_uses_observed_fields(self) -> None:
        enqueue = task_event(1, "ENQUEUE", 100, "EARLY", expert=9)
        enqueue.update({"enqueued_ts_ns": 90, "sequence": 7, "deadline_ts_ns": 500})
        issue = task_event(1, "ISSUE", 140, "EARLY", expert=9)
        issue.update({
            "sequence": 7,
            "deadline_ts_ns": 500,
            "issued_ts_ns": 120,
            "returned_ts_ns": 140,
            "issue_id": 11,
        })
        first_use = {
            "event": "EXPERT_FIRST_USE",
            "run_id": "run",
            "step": 7,
            "layer": 3,
            "expert": 9,
            "tensor": "blk.3.ffn_up_exps.weight",
            "stage": "EARLY",
            "phase": "PREFILL",
            "first_use_ts_ns": 200,
        }
        jobs, quality = extract_simulation_tasks([enqueue, issue, first_use])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["arrival_ts_ns"], 90)
        self.assertEqual(jobs[0]["sequence"], 7)
        self.assertEqual(jobs[0]["deadline_ts_ns"], 500)
        self.assertEqual(jobs[0]["service_ns"], 20)
        self.assertEqual(jobs[0]["first_use_ts_ns"], 200)
        self.assertEqual(quality["sequence_semantics"], "trace_sequence")


if __name__ == "__main__":
    unittest.main()
