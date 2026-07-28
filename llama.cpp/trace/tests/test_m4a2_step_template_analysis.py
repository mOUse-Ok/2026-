import sys
import unittest
from pathlib import Path

import numpy as np


TRACE_DIR = Path(__file__).resolve().parents[1]
if str(TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(TRACE_DIR))

from m4a2_step_template_analysis import (  # noqa: E402
    BASELINE_KEY,
    STATE_AMBIGUOUS,
    STATE_FALLBACK,
    STATE_MATURE,
    STATE_UNAVAILABLE,
    CandidateDefinition,
    RunEvidence,
    StepInfo,
    TaskRecord,
    _classification_summary,
    _paired_comparison,
    build_evaluation_dataset,
    candidate_definitions,
    prepare_run_evidence,
    replay_candidate,
    summarize_candidate,
)


def task(
    run_id,
    workers,
    step,
    task_id,
    *,
    stage="EARLY",
    tensor="blk.0.ffn_gate_exps.weight",
    prediction=200,
    first_use=600,
    layer=0,
    expert=0,
):
    begin = step * 1_000_000
    prediction_ts = begin + prediction
    first_use_ts = begin + first_use
    return TaskRecord(
        run_id=run_id,
        workers=workers,
        line_number=task_id,
        task_id=task_id,
        issue_id=task_id,
        step=step,
        layer=layer,
        expert=expert,
        stage=stage,
        tensor=tensor,
        prediction_ts_ns=prediction_ts,
        enqueued_ts_ns=prediction_ts,
        dequeued_ts_ns=prediction_ts + 10,
        issue_ts_ns=prediction_ts + 20,
        returned_ts_ns=prediction_ts + 30,
        first_use_ts_ns=first_use_ts,
        baseline_h_ns=first_use_ts - prediction_ts + 100,
        predicted_q_ns=10,
        predicted_p_ns=10,
        predicted_s_ns=10,
        baseline_h_mature=True,
        q_mature=True,
        p_mature=True,
        s_mature=True,
    )


def synthetic_run(
    run_id="r2",
    workers=2,
    step_count=10,
    *,
    duplicate_tasks=True,
    stable_late_step=None,
):
    steps = {}
    task_id = 1
    for step_number in range(1, step_count + 1):
        begin = step_number * 1_000_000
        stable = begin + (300 if step_number == stable_late_step else 100)
        step = StepInfo(
            step=step_number,
            step_begin_events=[begin],
            step_end_events=[begin + 900_000],
            stable_events=[stable],
        )
        step.tasks.append(task(
            run_id, workers, step_number, task_id,
            prediction=200, first_use=600 + step_number * 10, expert=0,
        ))
        task_id += 1
        if duplicate_tasks:
            step.tasks.append(task(
                run_id, workers, step_number, task_id,
                prediction=210, first_use=600 + step_number * 10, expert=1,
            ))
            task_id += 1
        step.tasks.append(task(
            run_id, workers, step_number, task_id,
            stage="LATE", tensor="blk.0.ffn_down_exps.weight",
            prediction=400, first_use=800 + step_number * 10, expert=0,
        ))
        task_id += 1
        steps[step_number] = step
    run = RunEvidence(
        run_dir=Path("."),
        run_id=run_id,
        workers=workers,
        manifest={},
        steps=steps,
        all_shadow_count=sum(len(step.tasks) for step in steps.values()),
        phase_counts={"DECODE": sum(len(step.tasks) for step in steps.values())},
        audit={},
        input_identity_before={},
    )
    prepare_run_evidence(run)
    return run


def definition(model, anchor="step_begin", semantic="tensor"):
    return next(
        item for item in candidate_definitions()
        if item.model == model
        and item.anchor_kind == anchor
        and item.semantic == semantic
    )


class M4A2StepTemplateTests(unittest.TestCase):
    def test_current_step_actual_does_not_predict_current_step(self):
        first = synthetic_run()
        second = synthetic_run()
        for item in second.steps[5].tasks:
            item.first_use_ts_ns += 50_000
        data_a = build_evaluation_dataset([first], "tensor")
        data_b = build_evaluation_dataset([second], "tensor")
        candidate = definition("previous_step")
        predicted_a, states_a, _ = replay_candidate(candidate, data_a, [first])
        predicted_b, states_b, _ = replay_candidate(candidate, data_b, [second])
        step5_a = data_a.step_numbers == 5
        step5_b = data_b.step_numbers == 5
        self.assertTrue(np.array_equal(predicted_a[step5_a], predicted_b[step5_b]))
        self.assertTrue(np.all(states_a[step5_a] == STATE_MATURE))
        self.assertTrue(np.all(states_b[step5_b] == STATE_MATURE))
        step6_a = data_a.step_numbers == 6
        step6_b = data_b.step_numbers == 6
        self.assertFalse(np.array_equal(predicted_a[step6_a], predicted_b[step6_b]))

    def test_rolling_warmup_then_mature_and_conservative_accounting(self):
        run = synthetic_run()
        dataset = build_evaluation_dataset([run], "tensor")
        predicted, states, accounting = replay_candidate(
            definition("rolling_median_w64"), dataset, [run]
        )
        self.assertTrue(np.all(states[dataset.step_numbers <= 8] == STATE_FALLBACK))
        self.assertTrue(np.all(states[dataset.step_numbers >= 9] == STATE_MATURE))
        self.assertEqual(
            accounting["eligible"],
            accounting["fallback"] + accounting["mature_exact"]
            + accounting["ambiguous"] + accounting["unavailable"],
        )
        summary = summarize_candidate(predicted, states, dataset)
        self.assertEqual(summary["coverage_accounting"]["fallback"], 24)
        self.assertEqual(summary["coverage_accounting"]["mature_exact"], 6)
        self.assertIn(
            "layer=0|EARLY|workers=2",
            summary["first_use"]["by_layer_stage_workers"],
        )

    def test_run_and_worker_histories_are_isolated(self):
        first = synthetic_run("w2", 2, step_count=2, duplicate_tasks=False)
        second = synthetic_run("w4", 4, step_count=2, duplicate_tasks=False)
        dataset = build_evaluation_dataset([first, second], "tensor")
        _, states, _ = replay_candidate(
            definition("previous_step"), dataset, [first, second]
        )
        for run_index in (0, 1):
            first_step = (dataset.run_indices == run_index) & (dataset.step_numbers == 1)
            second_step = (dataset.run_indices == run_index) & (dataset.step_numbers == 2)
            self.assertTrue(np.all(states[first_step] == STATE_FALLBACK))
            self.assertTrue(np.all(states[second_step] == STATE_MATURE))

    def test_anchor_later_than_prediction_is_unavailable(self):
        run = synthetic_run(step_count=2, stable_late_step=2)
        dataset = build_evaluation_dataset([run], "tensor")
        _, states, accounting = replay_candidate(
            definition("previous_step", anchor="first_stable_event"), dataset, [run]
        )
        self.assertTrue(np.all(states[dataset.step_numbers == 2] == STATE_UNAVAILABLE))
        self.assertGreater(accounting["fallback_reasons"]["invalid_current_step_prefix"], 0)

    def test_ordinal_ties_are_deterministic(self):
        run = synthetic_run(step_count=1, duplicate_tasks=False)
        extra = task("r2", 2, 1, 99, prediction=200, first_use=610, expert=3)
        run.steps[1].tasks.append(extra)
        prepare_run_evidence(run)
        tied = [item for item in run.steps[1].tasks if item.prediction_ts_ns == 1_000_200]
        self.assertEqual([item.task_id for item in tied], sorted(item.task_id for item in tied))
        self.assertEqual(run.audit["ordinal_timestamp_ties"], 1)

    def test_one_to_many_updates_once_and_preserves_task_samples(self):
        run = synthetic_run(step_count=2, duplicate_tasks=True)
        dataset = build_evaluation_dataset([run], "tensor")
        _, states, accounting = replay_candidate(
            definition("previous_step"), dataset, [run]
        )
        self.assertEqual(len(dataset.samples), 6)
        self.assertEqual(int(np.count_nonzero(dataset.per_tensor_representative)), 4)
        self.assertEqual(accounting["history_updates"], 4)
        self.assertEqual(int(np.count_nonzero(states == STATE_MATURE)), 3)

    def test_ambiguous_tensor_group_is_not_mature(self):
        run = synthetic_run(step_count=1)
        run.steps[1].tasks[1].first_use_ts_ns += 1
        dataset = build_evaluation_dataset([run], "tensor")
        _, states, _ = replay_candidate(
            definition("previous_step"), dataset, [run]
        )
        gate_mask = np.asarray([
            item.tensor == "blk.0.ffn_gate_exps.weight" for item in dataset.samples
        ])
        self.assertTrue(np.all(states[gate_mask] == STATE_AMBIGUOUS))

    def test_issue_return_formulas_and_zero_slack_is_late(self):
        run = synthetic_run(step_count=2, duplicate_tasks=False)
        dataset = build_evaluation_dataset([run], "tensor")
        predicted, states, _ = replay_candidate(
            definition("previous_step"), dataset, [run]
        )
        summary = summarize_candidate(predicted, states, dataset)
        self.assertEqual(summary["issue"]["prediction_formula"], "H - Q - P")
        self.assertEqual(summary["return"]["prediction_formula"], "H - Q - P - S")
        zero = _classification_summary(
            np.asarray([0], dtype=np.int64),
            np.asarray([-1], dtype=np.int64),
            np.asarray([True]),
            1,
        )
        self.assertEqual(zero["true_negative"], 1)
        self.assertEqual(zero["predicted_late_precision"], 1.0)

    def test_paired_baseline_uses_identical_mask(self):
        run = synthetic_run(step_count=3, duplicate_tasks=False)
        dataset = build_evaluation_dataset([run], "tensor")
        predicted, states, _ = replay_candidate(
            definition("previous_step"), dataset, [run]
        )
        comparison = _paired_comparison(predicted, states, dataset)
        worker = comparison["by_active_workers"]["2"]
        self.assertTrue(worker["paired_mask_count_equal"])
        self.assertEqual(worker["first_use"]["candidate"]["count"], worker["paired_count"])
        self.assertEqual(worker["first_use"]["baseline"]["count"], worker["paired_count"])

    def test_candidate_metadata_excludes_future_information(self):
        candidate = definition("scaled_median_w64")
        metadata = candidate.metadata()
        self.assertTrue(metadata["online_eligible"])
        self.assertFalse(metadata["uses_current_step_actual"])
        self.assertFalse(metadata["uses_future_information"])
        self.assertEqual(BASELINE_KEY, "phase_stage_median|queue_depth_worker_ewma|raw")


if __name__ == "__main__":
    unittest.main()
