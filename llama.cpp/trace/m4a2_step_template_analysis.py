#!/usr/bin/env python3
"""Offline-only M4A.2 Decode Step template feasibility analysis.

The module consumes existing M4A.1 Detail traces.  It never changes runtime
state: all online-eligible candidates are replayed prequentially, and a Step is
used for template updates only after every prediction for that Step is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


BASELINE_DEADLINE_MODEL = "phase_stage_median"
BASELINE_QUEUE_MODEL = "queue_depth_worker_ewma"
BASELINE_CALIBRATION_MODEL = "raw"
BASELINE_KEY = (
    f"{BASELINE_DEADLINE_MODEL}|{BASELINE_QUEUE_MODEL}|"
    f"{BASELINE_CALIBRATION_MODEL}"
)
WINDOW = 64
MIN_MATURE_STEPS = 8
EWMA_ALPHA = 0.2
NEGATIVE_THRESHOLDS_NS = (0, -500_000, -1_000_000, -2_000_000, -5_000_000)
CALIBRATION_LABELS = (
    "< -5 ms",
    "[-5 ms, -2 ms)",
    "[-2 ms, -1 ms)",
    "[-1 ms, -0.5 ms)",
    "[-0.5 ms, 0]",
    "(0, 0.5 ms]",
    "(0.5 ms, 1 ms]",
    "(1 ms, 2 ms]",
    "(2 ms, 5 ms]",
    "> 5 ms",
)
ANCHORS = ("step_begin", "first_prediction", "first_stable_event")
STABLE_EVENT_WHITELIST = ("LAYER_BEGIN",)

STATE_UNAVAILABLE = 0
STATE_FALLBACK = 1
STATE_MATURE = 2
STATE_AMBIGUOUS = 3
STATE_NAMES = {
    STATE_UNAVAILABLE: "unavailable",
    STATE_FALLBACK: "fallback",
    STATE_MATURE: "mature_exact",
    STATE_AMBIGUOUS: "ambiguous",
}

STAGE_CODES = {"UNKNOWN": 0, "EARLY": 1, "LATE": 2}
BAND_CODES = {"UNKNOWN": 0, "FRONT": 1, "MIDDLE": 2, "BACK": 3}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _percentile(values: Sequence[int | float] | np.ndarray, q: float) -> float | None:
    array = np.asarray(values)
    if array.size == 0:
        return None
    return float(np.percentile(array, q, method="linear"))


def _value_stats(values: Sequence[int | float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean_ns": None,
            "min_ns": None,
            "median_ns": None,
            "p75_ns": None,
            "p95_ns": None,
            "max_ns": None,
        }
    return {
        "count": int(array.size),
        "mean_ns": float(np.mean(array)),
        "min_ns": float(np.min(array)),
        "median_ns": _percentile(array, 50),
        "p75_ns": _percentile(array, 75),
        "p95_ns": _percentile(array, 95),
        "max_ns": float(np.max(array)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, *, include_sha256: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = _sha256(path)
    return result


@dataclass(slots=True)
class TaskRecord:
    run_id: str
    workers: int
    line_number: int
    task_id: int
    issue_id: int
    step: int
    layer: int
    expert: int
    stage: str
    tensor: str
    prediction_ts_ns: int
    enqueued_ts_ns: int
    dequeued_ts_ns: int
    issue_ts_ns: int
    returned_ts_ns: int
    first_use_ts_ns: int
    baseline_h_ns: int
    predicted_q_ns: int
    predicted_p_ns: int
    predicted_s_ns: int
    baseline_h_mature: bool
    q_mature: bool
    p_mature: bool
    s_mature: bool
    ordinal: int = -1
    diagnostic_band: str = "UNKNOWN"
    online_band: str = "UNKNOWN"
    created_task_count: int = 0
    observed_layer_count: int = 0
    previous_step_duration_ns: int | None = None
    early_late_pair_interval_ns: int | None = None

    @property
    def actual_h_ns(self) -> int:
        return self.first_use_ts_ns - self.prediction_ts_ns

    @property
    def actual_issue_slack_ns(self) -> int:
        return self.first_use_ts_ns - self.issue_ts_ns

    @property
    def actual_return_slack_ns(self) -> int:
        return self.first_use_ts_ns - self.returned_ts_ns


@dataclass(slots=True)
class StepInfo:
    step: int
    tasks: list[TaskRecord] = field(default_factory=list)
    step_begin_events: list[int] = field(default_factory=list)
    step_end_events: list[int] = field(default_factory=list)
    stable_events: list[int] = field(default_factory=list)
    mismatched_phase_events: int = 0

    def anchor(self, kind: str) -> int | None:
        if kind == "step_begin":
            return self.step_begin_events[0] if len(self.step_begin_events) == 1 else None
        if kind == "first_prediction":
            return min((task.prediction_ts_ns for task in self.tasks), default=None)
        if kind == "first_stable_event":
            return min(self.stable_events, default=None)
        raise ValueError(f"unknown anchor kind: {kind}")

    @property
    def duration_ns(self) -> int | None:
        if len(self.step_begin_events) != 1 or len(self.step_end_events) != 1:
            return None
        duration = self.step_end_events[0] - self.step_begin_events[0]
        return duration if duration >= 0 else None


@dataclass(slots=True)
class RunEvidence:
    run_dir: Path
    run_id: str
    workers: int
    manifest: dict[str, Any]
    steps: dict[int, StepInfo]
    all_shadow_count: int
    phase_counts: dict[str, int]
    audit: dict[str, Any]
    input_identity_before: dict[str, Any]

    @property
    def decode_tasks(self) -> list[TaskRecord]:
        return [task for step in self.sorted_steps() for task in step.tasks]

    def sorted_steps(self) -> list[StepInfo]:
        return sorted(
            (step for step in self.steps.values() if step.tasks),
            key=lambda step: (
                step.anchor("step_begin")
                if step.anchor("step_begin") is not None
                else min(task.prediction_ts_ns for task in step.tasks),
                step.step,
            ),
        )


@dataclass(frozen=True, slots=True)
class CandidateDefinition:
    semantic: str
    anchor_kind: str
    model: str
    estimator: str
    window: int
    min_samples: int

    @property
    def candidate_id(self) -> str:
        return f"{self.semantic}|{self.anchor_kind}|{self.model}"

    def metadata(self) -> dict[str, Any]:
        template_key = (
            "(layer,stage,tensor)"
            if self.semantic == "tensor"
            else "(layer,stage), earliest logical first-use"
        )
        return {
            "online_eligible": True,
            "uses_current_step_actual": False,
            "uses_future_information": False,
            "workers_scope": "separate_per_run_and_worker",
            "anchor_kind": self.anchor_kind,
            "template_key": template_key,
            "semantic": self.semantic,
            "model": self.model,
            "estimator": self.estimator,
            "window": self.window,
            "min_samples": self.min_samples,
            "fallback_policy": (
                "frozen M4A.1 baseline; scaled model falls back to raw median "
                "when only scale/shift is unavailable"
            ),
            "mature_rule": (
                "prediction uses only completed prior Decode Steps and meets "
                "the candidate-specific minimum history"
            ),
        }


def candidate_definitions() -> list[CandidateDefinition]:
    models = (
        ("previous_step", "previous", 1, 1),
        ("rolling_ewma_w64", "ewma", WINDOW, MIN_MATURE_STEPS),
        ("rolling_median_w64", "median", WINDOW, MIN_MATURE_STEPS),
        ("rolling_p25_w64", "p25", WINDOW, MIN_MATURE_STEPS),
        ("scaled_median_w64", "scaled_median", WINDOW, MIN_MATURE_STEPS),
    )
    return [
        CandidateDefinition(semantic, anchor, model, estimator, window, minimum)
        for semantic in ("tensor", "earliest_stage")
        for anchor in ANCHORS
        for model, estimator, window, minimum in models
    ]


def _find_baseline_prediction(record: dict[str, Any]) -> dict[str, Any] | None:
    for prediction in record.get("predictions", []):
        if (
            prediction.get("deadline_model") == BASELINE_DEADLINE_MODEL
            and prediction.get("queue_model") == BASELINE_QUEUE_MODEL
            and prediction.get("calibration_model") == BASELINE_CALIBRATION_MODEL
        ):
            return prediction
    return None


def _component_is_mature(
    prediction: dict[str, Any], warmup_field: str, fallback_field: str
) -> bool:
    return (
        prediction.get(warmup_field) is False
        and prediction.get(fallback_field) == "exact"
    )


def load_run_evidence(run_dir: Path) -> RunEvidence:
    """Read one existing Detail run without modifying it."""
    run_dir = run_dir.resolve()
    memory_path = run_dir / "memory_trace.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest.get("run_name") or run_dir.name)
    workers = int(
        manifest.get("environment", {}).get(
            "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS", 0
        )
    )
    before = {
        "memory_trace": _file_identity(memory_path),
        "run_manifest": _file_identity(manifest_path, include_sha256=True),
    }
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        before["summary"] = _file_identity(summary_path, include_sha256=True)

    steps: dict[int, StepInfo] = {}
    all_shadow_count = 0
    phase_counts: defaultdict[str, int] = defaultdict(int)
    seen_task_ids: set[int] = set()
    audit: defaultdict[str, int] = defaultdict(int)
    required_task_fields = (
        "task_id", "step", "layer", "expert", "stage", "tensor",
        "prediction_ts_ns", "enqueued_ts_ns", "dequeued_ts_ns",
        "issue_ts_ns", "returned_ts_ns", "first_use_ts_ns",
    )
    file_order_tasks: list[tuple[int, int]] = []

    with memory_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not any(
                marker in line
                for marker in (
                    '"event":"STEP_BEGIN"',
                    '"event":"STEP_END"',
                    '"event":"LAYER_BEGIN"',
                    '"event":"EXPERT_SHADOW_SLACK"',
                )
            ):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                audit["relevant_json_parse_errors"] += 1
                continue
            event = record.get("event")
            step_value = record.get("step")
            if not isinstance(step_value, int):
                audit["events_missing_step"] += 1
                continue
            step = steps.setdefault(step_value, StepInfo(step=step_value))
            if event in ("STEP_BEGIN", "STEP_END", "LAYER_BEGIN"):
                ts = record.get("ts_ns")
                if not isinstance(ts, int):
                    audit["anchor_events_missing_timestamp"] += 1
                    continue
                phase = record.get("phase")
                if phase != "DECODE":
                    if step.tasks:
                        step.mismatched_phase_events += 1
                    continue
                if event == "STEP_BEGIN":
                    step.step_begin_events.append(ts)
                elif event == "STEP_END":
                    step.step_end_events.append(ts)
                elif event in STABLE_EVENT_WHITELIST:
                    step.stable_events.append(ts)
                continue

            if event != "EXPERT_SHADOW_SLACK":
                continue
            all_shadow_count += 1
            phase = str(record.get("phase", "UNKNOWN"))
            phase_counts[phase] += 1
            if phase != "DECODE":
                continue
            if not all(field in record for field in required_task_fields):
                audit["decode_records_missing_required_fields"] += 1
                continue
            task_id = record.get("task_id")
            if not isinstance(task_id, int) or task_id <= 0:
                audit["invalid_task_ids"] += 1
                continue
            if task_id in seen_task_ids:
                audit["duplicate_task_ids"] += 1
                continue
            seen_task_ids.add(task_id)
            baseline = _find_baseline_prediction(record)
            if baseline is None:
                audit["missing_frozen_baseline_prediction"] += 1
                continue
            numeric_fields = (
                "prediction_ts_ns", "enqueued_ts_ns", "dequeued_ts_ns",
                "issue_ts_ns", "returned_ts_ns", "first_use_ts_ns",
            )
            if not all(_is_number(record.get(field)) for field in numeric_fields):
                audit["invalid_task_timestamps"] += 1
                continue
            prediction_ts, enqueue_ts, dequeue_ts, issue_ts, returned_ts, first_use = (
                int(record[field]) for field in numeric_fields
            )
            if prediction_ts != enqueue_ts:
                audit["prediction_enqueue_mismatches"] += 1
            if not enqueue_ts <= dequeue_ts <= issue_ts <= returned_ts:
                audit["worker_timestamp_regressions"] += 1
                continue
            if first_use < prediction_ts:
                audit["first_use_causality_errors"] += 1
                continue
            if (
                record.get("schema_version") != 2
                or record.get("semantics") != "logical_first_use"
                or record.get("physical_load_observed") is not False
                or record.get("issue_target") != "issue_ts < logical_first_use_ts"
                or record.get("return_target")
                != "final_enabled_hint_return_ts < logical_first_use_ts"
            ):
                audit["semantic_alignment_errors"] += 1
            baseline_values = (
                "predicted_first_use_horizon_ns", "predicted_queue_wait_ns",
                "predicted_pre_issue_overhead_ns",
                "predicted_hint_syscall_service_ns",
            )
            if not all(_is_number(baseline.get(field)) for field in baseline_values):
                audit["invalid_frozen_baseline_values"] += 1
                continue
            task = TaskRecord(
                run_id=run_id,
                workers=workers,
                line_number=line_number,
                task_id=task_id,
                issue_id=int(record.get("issue_id", -1)),
                step=step_value,
                layer=int(record.get("layer", -1)),
                expert=int(record.get("expert", -1)),
                stage=str(record.get("stage", "UNKNOWN")),
                tensor=str(record.get("tensor", "UNKNOWN")),
                prediction_ts_ns=prediction_ts,
                enqueued_ts_ns=enqueue_ts,
                dequeued_ts_ns=dequeue_ts,
                issue_ts_ns=issue_ts,
                returned_ts_ns=returned_ts,
                first_use_ts_ns=first_use,
                baseline_h_ns=int(baseline["predicted_first_use_horizon_ns"]),
                predicted_q_ns=int(baseline["predicted_queue_wait_ns"]),
                predicted_p_ns=int(baseline["predicted_pre_issue_overhead_ns"]),
                predicted_s_ns=int(baseline["predicted_hint_syscall_service_ns"]),
                baseline_h_mature=_component_is_mature(
                    baseline, "estimator_warmup", "fallback_level"
                ),
                q_mature=_component_is_mature(
                    baseline, "queue_warmup", "queue_fallback_level"
                ),
                p_mature=_component_is_mature(
                    baseline, "pre_issue_warmup", "pre_issue_fallback_level"
                ),
                s_mature=_component_is_mature(
                    baseline,
                    "syscall_service_warmup",
                    "syscall_service_fallback_level",
                ),
            )
            expected_issue = task.baseline_h_ns - task.predicted_q_ns - task.predicted_p_ns
            expected_return = expected_issue - task.predicted_s_ns
            if int(baseline.get("predicted_issue_slack_ns", expected_issue)) != expected_issue:
                audit["frozen_issue_formula_errors"] += 1
            if int(baseline.get("predicted_return_slack_ns", expected_return)) != expected_return:
                audit["frozen_return_formula_errors"] += 1
            step.tasks.append(task)
            file_order_tasks.append((prediction_ts, task_id))

    for previous, current in zip(file_order_tasks, file_order_tasks[1:]):
        if current[0] < previous[0]:
            audit["file_order_prediction_regressions"] += 1
        if current[1] < previous[1]:
            audit["file_order_task_id_regressions"] += 1

    evidence = RunEvidence(
        run_dir=run_dir,
        run_id=run_id,
        workers=workers,
        manifest=manifest,
        steps=steps,
        all_shadow_count=all_shadow_count,
        phase_counts=dict(phase_counts),
        audit=dict(audit),
        input_identity_before=before,
    )
    prepare_run_evidence(evidence)
    return evidence


def prepare_run_evidence(run: RunEvidence) -> None:
    """Reconstruct causal ordinals and diagnostic/online Step bands."""
    prior_task_counts: list[int] = []
    previous_duration: int | None = None
    tie_count = 0
    sorted_task_id_regressions = 0
    for step in run.sorted_steps():
        step.tasks.sort(key=lambda task: (task.prediction_ts_ns, task.task_id))
        total = len(step.tasks)
        timestamps = [task.prediction_ts_ns for task in step.tasks]
        timestamp_counts: defaultdict[int, int] = defaultdict(int)
        for ts in timestamps:
            timestamp_counts[ts] += 1
        tie_count += sum(count - 1 for count in timestamp_counts.values() if count > 1)
        for previous, current in zip(step.tasks, step.tasks[1:]):
            if (
                current.prediction_ts_ns > previous.prediction_ts_ns
                and current.task_id < previous.task_id
            ):
                sorted_task_id_regressions += 1

        online_total = (
            int(statistics.median(prior_task_counts)) if prior_task_counts else None
        )
        online_first = max(1, online_total // 3) if online_total else None
        online_second = max(2, (2 * online_total) // 3) if online_total else None
        unique_layers: set[int] = set()
        cursor = 0
        while cursor < total:
            end = cursor + 1
            while end < total and timestamps[end] == timestamps[cursor]:
                end += 1
            for task in step.tasks[cursor:end]:
                unique_layers.add(task.layer)
            causal_created = end
            causal_layers = len(unique_layers)
            for ordinal in range(cursor, end):
                task = step.tasks[ordinal]
                task.ordinal = ordinal
                task.created_task_count = causal_created
                task.observed_layer_count = causal_layers
                task.previous_step_duration_ns = previous_duration
                third = min(2, (3 * ordinal) // max(1, total))
                task.diagnostic_band = ("FRONT", "MIDDLE", "BACK")[third]
                if online_first is None or online_second is None:
                    task.online_band = "UNKNOWN"
                elif ordinal < online_first:
                    task.online_band = "FRONT"
                elif ordinal < online_second:
                    task.online_band = "MIDDLE"
                else:
                    task.online_band = "BACK"
            cursor = end

        by_layer_stage: defaultdict[tuple[int, str], list[int]] = defaultdict(list)
        for task in step.tasks:
            by_layer_stage[(task.layer, task.stage)].append(task.first_use_ts_ns)
        for task in step.tasks:
            early = by_layer_stage.get((task.layer, "EARLY"), [])
            late = by_layer_stage.get((task.layer, "LATE"), [])
            if early and late:
                task.early_late_pair_interval_ns = min(late) - min(early)

        prior_task_counts.append(total)
        duration = step.duration_ns
        previous_duration = duration if duration is not None else previous_duration

    run.audit["ordinal_timestamp_ties"] = tie_count
    run.audit["sorted_task_id_regressions"] = sorted_task_id_regressions
    run.audit["decode_step_count"] = len(run.sorted_steps())
    run.audit["decode_task_count"] = sum(len(step.tasks) for step in run.sorted_steps())


@dataclass(slots=True)
class EvaluationUnit:
    key: tuple[Any, ...]
    tasks: list[TaskRecord]
    representative: TaskRecord
    first_use_ts_ns: int
    ambiguous: bool


@dataclass(slots=True)
class EvaluationStep:
    run: RunEvidence
    step: StepInfo
    units: list[EvaluationUnit]


@dataclass(slots=True)
class EvaluationDataset:
    semantic: str
    steps: list[EvaluationStep]
    samples: list[TaskRecord]
    sample_unit_ids: np.ndarray
    per_tensor_representative: np.ndarray
    run_indices: np.ndarray
    workers: np.ndarray
    stage_codes: np.ndarray
    layers: np.ndarray
    step_numbers: np.ndarray
    ordinals: np.ndarray
    diagnostic_bands: np.ndarray
    online_bands: np.ndarray
    actual_h: np.ndarray
    actual_issue_slack: np.ndarray
    actual_return_slack: np.ndarray
    baseline_h: np.ndarray
    predicted_q: np.ndarray
    predicted_p: np.ndarray
    predicted_s: np.ndarray
    baseline_h_mature: np.ndarray
    q_mature: np.ndarray
    p_mature: np.ndarray
    s_mature: np.ndarray
    prediction_offsets_by_anchor: dict[str, np.ndarray]
    previous_step_duration: np.ndarray
    created_task_count: np.ndarray
    observed_layer_count: np.ndarray
    early_late_pair_interval: np.ndarray
    group_indices: dict[str, dict[str, np.ndarray]]
    step_sample_ranges: list[tuple[int, int]]
    multi_record_audit: dict[str, Any]


def _build_units(step: StepInfo, semantic: str) -> list[EvaluationUnit]:
    if semantic == "tensor":
        grouped: defaultdict[tuple[Any, ...], list[TaskRecord]] = defaultdict(list)
        for task in step.tasks:
            grouped[(task.layer, task.stage, task.tensor)].append(task)
        result = []
        for key, tasks in sorted(grouped.items(), key=lambda item: item[0]):
            tasks = sorted(tasks, key=lambda task: (task.prediction_ts_ns, task.task_id))
            first_uses = {task.first_use_ts_ns for task in tasks}
            result.append(EvaluationUnit(
                key=key,
                tasks=tasks,
                representative=min(tasks, key=lambda task: task.task_id),
                first_use_ts_ns=min(first_uses),
                ambiguous=len(first_uses) != 1,
            ))
        return result

    if semantic == "earliest_stage":
        grouped = defaultdict(list)
        for task in step.tasks:
            grouped[(task.layer, task.stage)].append(task)
        result = []
        for key, tasks in sorted(grouped.items(), key=lambda item: item[0]):
            earliest = min(task.first_use_ts_ns for task in tasks)
            matching = [task for task in tasks if task.first_use_ts_ns == earliest]
            representative = min(matching, key=lambda task: task.task_id)
            result.append(EvaluationUnit(
                key=key,
                tasks=[representative],
                representative=representative,
                first_use_ts_ns=earliest,
                ambiguous=False,
            ))
        return result

    raise ValueError(f"unknown semantic: {semantic}")


def _build_group_indices(
    samples: list[TaskRecord], run_indices: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    groups: dict[str, defaultdict[str, list[int]]] = {
        name: defaultdict(list)
        for name in (
            "by_active_workers", "by_stage", "by_stage_workers",
            "by_layer_workers", "by_layer_stage_workers", "by_step", "by_ordinal",
            "by_diagnostic_band", "by_online_band",
        )
    }
    for index, task in enumerate(samples):
        groups["by_active_workers"][str(task.workers)].append(index)
        groups["by_stage"][task.stage].append(index)
        groups["by_stage_workers"][
            f"{task.stage}|workers={task.workers}"
        ].append(index)
        groups["by_layer_workers"][
            f"layer={task.layer}|workers={task.workers}"
        ].append(index)
        groups["by_layer_stage_workers"][
            f"layer={task.layer}|{task.stage}|workers={task.workers}"
        ].append(index)
        groups["by_step"][f"{task.run_id}|step={task.step}"].append(index)
        groups["by_ordinal"][str(task.ordinal)].append(index)
        groups["by_diagnostic_band"][
            f"{task.diagnostic_band}|workers={task.workers}"
        ].append(index)
        groups["by_online_band"][
            f"{task.online_band}|workers={task.workers}"
        ].append(index)
    return {
        dimension: {
            key: np.asarray(indices, dtype=np.int64)
            for key, indices in sorted(values.items())
        }
        for dimension, values in groups.items()
    }


def build_evaluation_dataset(
    runs: Sequence[RunEvidence], semantic: str
) -> EvaluationDataset:
    steps: list[EvaluationStep] = []
    samples: list[TaskRecord] = []
    unit_ids: list[int] = []
    representatives: list[bool] = []
    run_indices: list[int] = []
    ranges: list[tuple[int, int]] = []
    audit_groups: list[dict[str, Any]] = []
    global_unit_id = 0

    for run_index, run in enumerate(runs):
        for step in run.sorted_steps():
            units = _build_units(step, semantic)
            evaluation_step = EvaluationStep(run=run, step=step, units=units)
            steps.append(evaluation_step)
            start = len(samples)
            for unit in units:
                global_unit_id += 1
                if semantic == "tensor":
                    task_count = len(unit.tasks)
                    distinct_experts = len({task.expert for task in unit.tasks})
                    distinct_tensors = len({task.tensor for task in unit.tasks})
                    first_uses = {task.first_use_ts_ns for task in unit.tasks}
                    audit_groups.append({
                        "run_id": run.run_id,
                        "step": step.step,
                        "layer": unit.representative.layer,
                        "stage": unit.representative.stage,
                        "tensor": unit.representative.tensor,
                        "task_count": task_count,
                        "distinct_expert_count": distinct_experts,
                        "distinct_tensor_count": distinct_tensors,
                        "distinct_first_use_ts_count": len(first_uses),
                        "earliest_first_use_ts": min(first_uses),
                        "latest_first_use_ts": max(first_uses),
                        "first_use_span_ns": max(first_uses) - min(first_uses),
                        "ambiguous": unit.ambiguous,
                    })
                for task in unit.tasks:
                    samples.append(task)
                    unit_ids.append(global_unit_id)
                    representatives.append(task is unit.representative)
                    run_indices.append(run_index)
            ranges.append((start, len(samples)))

    run_index_array = np.asarray(run_indices, dtype=np.int16)
    anchor_offsets = {}
    for anchor in ANCHORS:
        values = []
        for task, run_index in zip(samples, run_indices):
            step = runs[run_index].steps[task.step]
            timestamp = step.anchor(anchor)
            values.append(
                task.prediction_ts_ns - timestamp if timestamp is not None else -1
            )
        anchor_offsets[anchor] = np.asarray(values, dtype=np.int64)

    missing_int = np.iinfo(np.int64).min
    dataset = EvaluationDataset(
        semantic=semantic,
        steps=steps,
        samples=samples,
        sample_unit_ids=np.asarray(unit_ids, dtype=np.int64),
        per_tensor_representative=np.asarray(representatives, dtype=bool),
        run_indices=run_index_array,
        workers=np.asarray([task.workers for task in samples], dtype=np.int16),
        stage_codes=np.asarray(
            [STAGE_CODES.get(task.stage, 0) for task in samples], dtype=np.int8
        ),
        layers=np.asarray([task.layer for task in samples], dtype=np.int16),
        step_numbers=np.asarray([task.step for task in samples], dtype=np.int32),
        ordinals=np.asarray([task.ordinal for task in samples], dtype=np.int32),
        diagnostic_bands=np.asarray(
            [BAND_CODES.get(task.diagnostic_band, 0) for task in samples],
            dtype=np.int8,
        ),
        online_bands=np.asarray(
            [BAND_CODES.get(task.online_band, 0) for task in samples], dtype=np.int8
        ),
        actual_h=np.asarray([task.actual_h_ns for task in samples], dtype=np.int64),
        actual_issue_slack=np.asarray(
            [task.actual_issue_slack_ns for task in samples], dtype=np.int64
        ),
        actual_return_slack=np.asarray(
            [task.actual_return_slack_ns for task in samples], dtype=np.int64
        ),
        baseline_h=np.asarray([task.baseline_h_ns for task in samples], dtype=np.int64),
        predicted_q=np.asarray([task.predicted_q_ns for task in samples], dtype=np.int64),
        predicted_p=np.asarray([task.predicted_p_ns for task in samples], dtype=np.int64),
        predicted_s=np.asarray([task.predicted_s_ns for task in samples], dtype=np.int64),
        baseline_h_mature=np.asarray(
            [task.baseline_h_mature for task in samples], dtype=bool
        ),
        q_mature=np.asarray([task.q_mature for task in samples], dtype=bool),
        p_mature=np.asarray([task.p_mature for task in samples], dtype=bool),
        s_mature=np.asarray([task.s_mature for task in samples], dtype=bool),
        prediction_offsets_by_anchor=anchor_offsets,
        previous_step_duration=np.asarray([
            task.previous_step_duration_ns
            if task.previous_step_duration_ns is not None else missing_int
            for task in samples
        ], dtype=np.int64),
        created_task_count=np.asarray(
            [task.created_task_count for task in samples], dtype=np.int32
        ),
        observed_layer_count=np.asarray(
            [task.observed_layer_count for task in samples], dtype=np.int16
        ),
        early_late_pair_interval=np.asarray([
            task.early_late_pair_interval_ns
            if task.early_late_pair_interval_ns is not None else missing_int
            for task in samples
        ], dtype=np.int64),
        group_indices={},
        step_sample_ranges=ranges,
        multi_record_audit={},
    )
    dataset.group_indices = _build_group_indices(samples, run_index_array)
    if semantic == "tensor":
        spans = np.asarray([row["first_use_span_ns"] for row in audit_groups])
        dataset.multi_record_audit = {
            "semantic": "per_tensor_group_then_one_to_many_task_expansion",
            "group_count": len(audit_groups),
            "task_count": len(samples),
            "multi_task_group_count": sum(
                int(row["task_count"] > 1) for row in audit_groups
            ),
            "ambiguous_group_count": sum(int(row["ambiguous"]) for row in audit_groups),
            "ambiguous_task_count": sum(
                row["task_count"] for row in audit_groups if row["ambiguous"]
            ),
            "task_count_stats": _value_stats([
                row["task_count"] for row in audit_groups
            ]),
            "distinct_first_use_count_stats": _value_stats([
                row["distinct_first_use_ts_count"] for row in audit_groups
            ]),
            "first_use_span_stats": _value_stats(spans),
            "by_stage": {
                stage: {
                    "group_count": sum(int(row["stage"] == stage) for row in audit_groups),
                    "task_count": sum(
                        row["task_count"] for row in audit_groups
                        if row["stage"] == stage
                    ),
                    "ambiguous_group_count": sum(
                        int(row["stage"] == stage and row["ambiguous"])
                        for row in audit_groups
                    ),
                }
                for stage in ("EARLY", "LATE", "UNKNOWN")
            },
            "groups": audit_groups,
        }
    else:
        dataset.multi_record_audit = {
            "semantic": "earliest_first_use_per_(run,step,layer,stage)",
            "group_count": len(samples),
            "task_count": len(samples),
            "ambiguous_group_count": 0,
        }
    return dataset


def _estimate_history(history: deque[int], estimator: str) -> float:
    values = np.asarray(history, dtype=np.float64)
    if estimator == "median" or estimator == "scaled_median":
        return float(np.median(values))
    if estimator == "p25":
        return float(np.percentile(values, 25, method="linear"))
    if estimator == "ewma":
        result = float(values[0])
        for value in values[1:]:
            result = EWMA_ALPHA * float(value) + (1.0 - EWMA_ALPHA) * result
        return result
    raise ValueError(f"unsupported estimator: {estimator}")


def replay_candidate(
    definition: CandidateDefinition,
    dataset: EvaluationDataset,
    runs: Sequence[RunEvidence],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return predicted horizons and mutually exclusive sample states."""
    if definition.semantic != dataset.semantic:
        raise ValueError("candidate semantic and dataset semantic differ")
    predicted_h = dataset.baseline_h.copy()
    states = np.full(len(dataset.samples), STATE_UNAVAILABLE, dtype=np.uint8)
    run_state: dict[int, dict[str, Any]] = {}
    for run_index in range(len(runs)):
        run_state[run_index] = {
            "previous": {},
            "histories": defaultdict(lambda: deque(maxlen=definition.window)),
            "delay_history": deque(maxlen=definition.window),
        }
    history_updates = 0
    fallback_reasons: defaultdict[str, int] = defaultdict(int)
    scale_values: list[float] = []
    shift_values: list[float] = []
    run_index_by_identity = {id(run): index for index, run in enumerate(runs)}

    for evaluation_step, (start, end) in zip(dataset.steps, dataset.step_sample_ranges):
        run_index = run_index_by_identity[id(evaluation_step.run)]
        state = run_state[run_index]
        anchor_ts = evaluation_step.step.anchor(definition.anchor_kind)
        if anchor_ts is None:
            fallback_reasons["anchor_unavailable"] += end - start
            states[start:end] = STATE_UNAVAILABLE
            if definition.estimator == "previous":
                state["previous"] = {}
            continue
        first_prediction = evaluation_step.step.anchor("first_prediction")
        if first_prediction is None or first_prediction < anchor_ts:
            fallback_reasons["invalid_current_step_prefix"] += end - start
            states[start:end] = STATE_UNAVAILABLE
            if definition.estimator == "previous":
                state["previous"] = {}
            continue
        current_delay = first_prediction - anchor_ts

        cursor = start
        observations: dict[tuple[Any, ...], int] = {}
        for unit in evaluation_step.units:
            unit_count = len(unit.tasks)
            unit_slice = slice(cursor, cursor + unit_count)
            if unit.ambiguous:
                states[unit_slice] = STATE_AMBIGUOUS
                fallback_reasons["ambiguous_first_use"] += unit_count
                cursor += unit_count
                continue

            predicted_offset: float | None = None
            sample_state = STATE_FALLBACK
            reason = "insufficient_history"
            if definition.estimator == "previous":
                if unit.key in state["previous"]:
                    predicted_offset = float(state["previous"][unit.key])
                    sample_state = STATE_MATURE
                    reason = ""
            else:
                history = state["histories"].get(unit.key)
                if history is not None and len(history) >= definition.min_samples:
                    predicted_offset = _estimate_history(history, definition.estimator)
                    sample_state = STATE_MATURE
                    reason = ""
                    if definition.estimator == "scaled_median":
                        delays = state["delay_history"]
                        historical_delay = (
                            float(np.median(np.asarray(delays, dtype=np.float64)))
                            if len(delays) >= definition.min_samples else None
                        )
                        if historical_delay is None or historical_delay <= 0:
                            sample_state = STATE_FALLBACK
                            reason = "insufficient_current_step_prefix"
                        else:
                            scale = min(2.0, max(0.5, current_delay / historical_delay))
                            shift = current_delay - historical_delay * scale
                            predicted_offset = predicted_offset * scale + shift
                            scale_values.append(scale)
                            shift_values.append(shift)

            if predicted_offset is None:
                states[unit_slice] = STATE_FALLBACK
                fallback_reasons[reason] += unit_count
            else:
                predicted_first_use = int(round(anchor_ts + predicted_offset))
                for sample_index in range(cursor, cursor + unit_count):
                    task = dataset.samples[sample_index]
                    predicted_h[sample_index] = (
                        predicted_first_use - task.prediction_ts_ns
                    )
                states[unit_slice] = sample_state
                if sample_state == STATE_FALLBACK:
                    fallback_reasons[reason] += unit_count
            observations[unit.key] = unit.first_use_ts_ns - anchor_ts
            cursor += unit_count

        if cursor != end:
            raise AssertionError("Step sample range does not match evaluation units")
        # This is the only update point: every current-Step prediction is frozen.
        if definition.estimator == "previous":
            state["previous"] = observations
        else:
            histories = state["histories"]
            for key, value in observations.items():
                histories[key].append(value)
            state["delay_history"].append(current_delay)
        history_updates += len(observations)

    accounting = {name: int(np.count_nonzero(states == code)) for code, name in STATE_NAMES.items()}
    accounting.update({
        "eligible": int(states.size),
        "predicted": int(np.count_nonzero(
            (states == STATE_FALLBACK) | (states == STATE_MATURE)
        )),
        "warmup": int(np.count_nonzero(states == STATE_FALLBACK)),
        "oracle_only": 0,
        "history_updates": history_updates,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "scale_stats": _value_stats(scale_values),
        "shift_stats": _value_stats(shift_values),
    })
    if (
        accounting["unavailable"] + accounting["fallback"]
        + accounting["mature_exact"] + accounting["ambiguous"]
        != accounting["eligible"]
    ):
        raise AssertionError("candidate state accounting is not conservative")
    return predicted_h, states, accounting


def _error_summary(
    signed_errors: np.ndarray,
    evaluated_mask: np.ndarray,
    eligible_count: int,
) -> dict[str, Any]:
    values = signed_errors[evaluated_mask]
    absolute = np.abs(values)
    count = int(values.size)
    return {
        "eligible": int(eligible_count),
        "count": count,
        "coverage": _ratio(count, eligible_count),
        "unavailable_count": max(0, int(eligible_count) - count),
        "mae_ns": float(np.mean(absolute)) if count else None,
        "median_absolute_error_ns": _percentile(absolute, 50),
        "p75_absolute_error_ns": _percentile(absolute, 75),
        "p95_absolute_error_ns": _percentile(absolute, 95),
        "signed_error_mean_ns": float(np.mean(values)) if count else None,
        "signed_error_p25_ns": _percentile(values, 25),
        "signed_error_median_ns": _percentile(values, 50),
        "signed_error_p75_ns": _percentile(values, 75),
        "signed_error_p95_ns": _percentile(values, 95),
    }


def _calibration_masks(slacks: np.ndarray) -> tuple[np.ndarray, ...]:
    return (
        slacks < -5_000_000,
        (slacks >= -5_000_000) & (slacks < -2_000_000),
        (slacks >= -2_000_000) & (slacks < -1_000_000),
        (slacks >= -1_000_000) & (slacks < -500_000),
        (slacks >= -500_000) & (slacks <= 0),
        (slacks > 0) & (slacks <= 500_000),
        (slacks > 500_000) & (slacks <= 1_000_000),
        (slacks > 1_000_000) & (slacks <= 2_000_000),
        (slacks > 2_000_000) & (slacks <= 5_000_000),
        slacks > 5_000_000,
    )


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (centre - margin) / denominator, (centre + margin) / denominator


def _classification_summary(
    predicted_slack: np.ndarray,
    actual_slack: np.ndarray,
    evaluated_mask: np.ndarray,
    eligible_count: int,
) -> dict[str, Any]:
    predicted = predicted_slack[evaluated_mask]
    actual = actual_slack[evaluated_mask]
    count = int(predicted.size)
    predicted_on_time = predicted > 0
    actual_on_time = actual > 0
    predicted_late = ~predicted_on_time
    actual_late = ~actual_on_time
    tp = int(np.count_nonzero(predicted_on_time & actual_on_time))
    tn = int(np.count_nonzero(predicted_late & actual_late))
    fp = int(np.count_nonzero(predicted_on_time & actual_late))
    fn = int(np.count_nonzero(predicted_late & actual_on_time))
    late_precision = _ratio(tn, tn + fn)
    late_recall = _ratio(tn, tn + fp)
    result = _error_summary(predicted_slack - actual_slack, evaluated_mask, eligible_count)
    result.update({
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "on_time_precision": _ratio(tp, tp + fp),
        "on_time_recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "predicted_late_precision": late_precision,
        "predicted_late_recall": late_recall,
        "predicted_late_f1": (
            2.0 * late_precision * late_recall / (late_precision + late_recall)
            if late_precision is not None and late_recall is not None
            and late_precision + late_recall else None
        ),
        "false_reject_candidate_count": fn,
        "false_reject_candidate_rate": _ratio(fn, count),
        "late_prevalence": _ratio(tn + fp, count),
        "threshold_coverage": _ratio(tn + fn, eligible_count),
    })

    calibration: dict[str, Any] = {}
    nonempty_rates: list[float] = []
    for label, mask in zip(CALIBRATION_LABELS, _calibration_masks(predicted)):
        total = int(np.count_nonzero(mask))
        on_time = int(np.count_nonzero(actual_on_time & mask))
        rate = _ratio(on_time, total)
        lower, upper = _wilson_interval(on_time, total)
        if rate is not None:
            nonempty_rates.append(rate)
        calibration[label] = {
            "lower_bound_ns": {
                "< -5 ms": None,
                "[-5 ms, -2 ms)": -5_000_000,
                "[-2 ms, -1 ms)": -2_000_000,
                "[-1 ms, -0.5 ms)": -1_000_000,
                "[-0.5 ms, 0]": -500_000,
                "(0, 0.5 ms]": 0,
                "(0.5 ms, 1 ms]": 500_000,
                "(1 ms, 2 ms]": 1_000_000,
                "(2 ms, 5 ms]": 2_000_000,
                "> 5 ms": 5_000_000,
            }[label],
            "count": total,
            "actual_on_time_count": on_time,
            "actual_late_count": total - on_time,
            "actual_on_time_rate": rate,
            "actual_on_time_wilson_95_low": lower,
            "actual_on_time_wilson_95_high": upper,
            "insufficient": total < 30,
        }
    result["calibration"] = calibration
    result["calibration_monotonicity"] = {
        "nonempty_bucket_count": len(nonempty_rates),
        "adjacent_decrease_count": sum(
            int(current < previous)
            for previous, current in zip(nonempty_rates, nonempty_rates[1:])
        ),
        "nondecreasing": all(
            current >= previous
            for previous, current in zip(nonempty_rates, nonempty_rates[1:])
        ),
    }

    threshold_rows = []
    actual_late_total = int(np.count_nonzero(actual_late))
    for threshold in NEGATIVE_THRESHOLDS_NS:
        selected = predicted <= threshold
        selected_count = int(np.count_nonzero(selected))
        true_late = int(np.count_nonzero(selected & actual_late))
        false_reject = selected_count - true_late
        precision = _ratio(true_late, selected_count)
        recall = _ratio(true_late, actual_late_total)
        threshold_rows.append({
            "threshold_ns": threshold,
            "predicted_late_count": selected_count,
            "actual_late_and_predicted_late": true_late,
            "false_reject_candidate_count": false_reject,
            "false_reject_candidate_rate": _ratio(false_reject, count),
            "predicted_late_precision": precision,
            "predicted_late_recall": recall,
            "predicted_late_f1": (
                2.0 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else None
            ),
            "threshold_coverage_of_evaluated": _ratio(selected_count, count),
            "threshold_coverage_of_eligible": _ratio(selected_count, eligible_count),
        })
    precisions = [
        row["predicted_late_precision"] for row in threshold_rows
        if row["predicted_late_precision"] is not None
    ]
    result["thresholds"] = threshold_rows
    result["negative_threshold_precision_monotonicity"] = {
        "comparable_threshold_count": len(precisions),
        "decrease_count": sum(
            int(current < previous)
            for previous, current in zip(precisions, precisions[1:])
        ),
        "nondecreasing_with_stricter_threshold": all(
            current >= previous
            for previous, current in zip(precisions, precisions[1:])
        ),
    }
    return result


def _dimension_summaries(
    values: np.ndarray,
    actual_values: np.ndarray | None,
    mask: np.ndarray,
    base_mask: np.ndarray,
    dataset: EvaluationDataset,
    dimensions: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    classification = actual_values is not None
    for dimension in dimensions:
        rows = {}
        for key, indices in dataset.group_indices[dimension].items():
            eligible = int(np.count_nonzero(base_mask[indices]))
            selected = indices[base_mask[indices]]
            local_mask = mask[selected]
            if classification:
                rows[key] = _classification_summary(
                    values[selected], actual_values[selected], local_mask, eligible
                )
            else:
                rows[key] = _error_summary(values[selected], local_mask, eligible)
        result[dimension] = rows
    return result


FIRST_USE_DIMENSIONS = (
    "by_active_workers", "by_stage", "by_stage_workers", "by_layer_workers",
    "by_layer_stage_workers", "by_step", "by_ordinal",
    "by_diagnostic_band", "by_online_band",
)
TARGET_DIMENSIONS = (
    "by_active_workers", "by_stage", "by_stage_workers", "by_layer_workers",
    "by_layer_stage_workers", "by_diagnostic_band", "by_online_band",
)
SENSITIVITY_DIMENSIONS = ("by_active_workers", "by_stage", "by_stage_workers")


def summarize_candidate(
    predicted_h: np.ndarray,
    states: np.ndarray,
    dataset: EvaluationDataset,
    *,
    base_mask: np.ndarray | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    if base_mask is None:
        base_mask = np.ones(len(dataset.samples), dtype=bool)
    predicted = base_mask & (
        (states == STATE_FALLBACK) | (states == STATE_MATURE)
    )
    mature_h = base_mask & (states == STATE_MATURE)
    fallback_h = base_mask & (states == STATE_FALLBACK)
    eligible = int(np.count_nonzero(base_mask))
    first_error = predicted_h - dataset.actual_h
    first_dimensions = SENSITIVITY_DIMENSIONS if compact else FIRST_USE_DIMENSIONS
    first_use = {
        "operational": _error_summary(first_error, predicted, eligible),
        "fallback_only": _error_summary(first_error, fallback_h, eligible),
        "mature_exact": _error_summary(first_error, mature_h, eligible),
        **_dimension_summaries(
            first_error, None, mature_h, base_mask, dataset, first_dimensions
        ),
    }

    predicted_issue = predicted_h - dataset.predicted_q - dataset.predicted_p
    predicted_return = predicted_issue - dataset.predicted_s
    issue_mature = mature_h & dataset.q_mature & dataset.p_mature
    return_mature = issue_mature & dataset.s_mature
    issue_fallback = predicted & ~issue_mature
    return_fallback = predicted & ~return_mature
    target_dimensions = SENSITIVITY_DIMENSIONS if compact else TARGET_DIMENSIONS

    issue = {
        "prediction_formula": "H - Q - P",
        "actual_label": "issue_ts < logical_first_use_ts",
        "operational": _classification_summary(
            predicted_issue, dataset.actual_issue_slack, predicted, eligible
        ),
        "fallback_only": _classification_summary(
            predicted_issue, dataset.actual_issue_slack, issue_fallback, eligible
        ),
        "mature_exact": _classification_summary(
            predicted_issue, dataset.actual_issue_slack, issue_mature, eligible
        ),
        **_dimension_summaries(
            predicted_issue, dataset.actual_issue_slack, issue_mature,
            base_mask, dataset, target_dimensions,
        ),
    }
    returned = {
        "prediction_formula": "H - Q - P - S",
        "actual_label": (
            "final_enabled_hint_return_ts < logical_first_use_ts"
        ),
        "operational": _classification_summary(
            predicted_return, dataset.actual_return_slack, predicted, eligible
        ),
        "fallback_only": _classification_summary(
            predicted_return, dataset.actual_return_slack, return_fallback, eligible
        ),
        "mature_exact": _classification_summary(
            predicted_return, dataset.actual_return_slack, return_mature, eligible
        ),
        **_dimension_summaries(
            predicted_return, dataset.actual_return_slack, return_mature,
            base_mask, dataset, target_dimensions,
        ),
    }
    return {
        "coverage_accounting": {
            "eligible": eligible,
            "predicted": int(np.count_nonzero(predicted)),
            "warmup": int(np.count_nonzero(fallback_h)),
            "fallback": int(np.count_nonzero(fallback_h)),
            "ambiguous": int(np.count_nonzero(base_mask & (states == STATE_AMBIGUOUS))),
            "unavailable": int(np.count_nonzero(base_mask & (states == STATE_UNAVAILABLE))),
            "mature_exact": int(np.count_nonzero(mature_h)),
            "issue_mature_exact": int(np.count_nonzero(issue_mature)),
            "return_mature_exact": int(np.count_nonzero(return_mature)),
            "oracle_only": 0,
        },
        "first_use": first_use,
        "issue": issue,
        "return": returned,
    }


def _baseline_states(dataset: EvaluationDataset) -> np.ndarray:
    return np.where(
        dataset.baseline_h_mature, STATE_MATURE, STATE_FALLBACK
    ).astype(np.uint8)


def _paired_comparison(
    candidate_h: np.ndarray,
    candidate_states: np.ndarray,
    dataset: EvaluationDataset,
) -> dict[str, Any]:
    baseline_states = _baseline_states(dataset)
    result: dict[str, Any] = {"by_active_workers": {}}
    for workers in sorted(set(int(value) for value in dataset.workers)):
        worker_mask = dataset.workers == workers
        first_pair = (
            worker_mask & (candidate_states == STATE_MATURE)
            & (baseline_states == STATE_MATURE)
        )
        eligible = int(np.count_nonzero(worker_mask))
        candidate_first = _error_summary(
            candidate_h - dataset.actual_h, first_pair, eligible
        )
        baseline_first = _error_summary(
            dataset.baseline_h - dataset.actual_h, first_pair, eligible
        )
        mae_improvement = (
            (baseline_first["mae_ns"] - candidate_first["mae_ns"])
            / baseline_first["mae_ns"]
            if baseline_first["mae_ns"] not in (None, 0)
            and candidate_first["mae_ns"] is not None else None
        )
        p95_improvement = (
            (
                baseline_first["p95_absolute_error_ns"]
                - candidate_first["p95_absolute_error_ns"]
            ) / baseline_first["p95_absolute_error_ns"]
            if baseline_first["p95_absolute_error_ns"] not in (None, 0)
            and candidate_first["p95_absolute_error_ns"] is not None else None
        )

        worker_result: dict[str, Any] = {
            "paired_count": int(np.count_nonzero(first_pair)),
            "paired_mask_count_equal": True,
            "first_use": {
                "candidate": candidate_first,
                "baseline": baseline_first,
                "mae_delta_ns": (
                    candidate_first["mae_ns"] - baseline_first["mae_ns"]
                    if candidate_first["mae_ns"] is not None
                    and baseline_first["mae_ns"] is not None else None
                ),
                "p95_absolute_error_delta_ns": (
                    candidate_first["p95_absolute_error_ns"]
                    - baseline_first["p95_absolute_error_ns"]
                    if candidate_first["p95_absolute_error_ns"] is not None
                    and baseline_first["p95_absolute_error_ns"] is not None else None
                ),
                "relative_mae_improvement": mae_improvement,
                "relative_p95_improvement": p95_improvement,
            },
        }
        for target, actual, extra_mature, subtract_s in (
            (
                "issue", dataset.actual_issue_slack,
                dataset.q_mature & dataset.p_mature, False,
            ),
            (
                "return", dataset.actual_return_slack,
                dataset.q_mature & dataset.p_mature & dataset.s_mature, True,
            ),
        ):
            pair = first_pair & extra_mature
            candidate_slack = candidate_h - dataset.predicted_q - dataset.predicted_p
            baseline_slack = dataset.baseline_h - dataset.predicted_q - dataset.predicted_p
            if subtract_s:
                candidate_slack = candidate_slack - dataset.predicted_s
                baseline_slack = baseline_slack - dataset.predicted_s
            candidate_target = _classification_summary(
                candidate_slack, actual, pair, eligible
            )
            baseline_target = _classification_summary(
                baseline_slack, actual, pair, eligible
            )
            worker_result[target] = {
                "paired_count": int(np.count_nonzero(pair)),
                "candidate": candidate_target,
                "baseline": baseline_target,
                "predicted_late_precision_delta": (
                    candidate_target["predicted_late_precision"]
                    - baseline_target["predicted_late_precision"]
                    if candidate_target["predicted_late_precision"] is not None
                    and baseline_target["predicted_late_precision"] is not None
                    else None
                ),
            }
        result["by_active_workers"][str(workers)] = worker_result
    return result


def build_anchor_audit(runs: Sequence[RunEvidence]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stable_event_whitelist": list(STABLE_EVENT_WHITELIST),
        "stable_event_priority": list(STABLE_EVENT_WHITELIST),
        "anchors": {},
        "pairwise_anchor_differences": {},
        "cross_run_repeat_stability_available": False,
        "cross_run_repeat_stability_reason": (
            "Only one Detail Evidence run is available for each worker count"
        ),
    }
    for anchor in ANCHORS:
        by_workers: dict[str, Any] = {}
        for workers in sorted({run.workers for run in runs}):
            worker_runs = [run for run in runs if run.workers == workers]
            offsets: list[int] = []
            step_medians: list[float] = []
            step_total = step_available = task_total = task_available = 0
            missing = duplicate = late = regression = mismatch = 0
            for run in worker_runs:
                for step in run.sorted_steps():
                    step_total += 1
                    task_total += len(step.tasks)
                    timestamp = step.anchor(anchor)
                    if anchor == "step_begin":
                        duplicate += max(0, len(step.step_begin_events) - 1)
                        missing += int(len(step.step_begin_events) == 0)
                    elif anchor == "first_stable_event":
                        missing += int(not step.stable_events)
                    elif anchor == "first_prediction":
                        missing += int(not step.tasks)
                    mismatch += step.mismatched_phase_events
                    if timestamp is None:
                        continue
                    step_available += 1
                    local = []
                    for task in step.tasks:
                        if timestamp > task.prediction_ts_ns:
                            late += 1
                            continue
                        value = task.prediction_ts_ns - timestamp
                        if value < 0:
                            regression += 1
                            continue
                        offsets.append(value)
                        local.append(value)
                        task_available += 1
                    if local:
                        step_medians.append(float(np.median(local)))
            by_workers[str(workers)] = {
                "run_count": len(worker_runs),
                "step_total": step_total,
                "step_available": step_available,
                "step_coverage": _ratio(step_available, step_total),
                "task_total": task_total,
                "task_available": task_available,
                "task_coverage": _ratio(task_available, task_total),
                "missing_anchor_count": missing,
                "duplicate_anchor_count": duplicate,
                "phase_step_mismatch_count": mismatch,
                "anchor_later_than_prediction_count": late,
                "timestamp_regression_count": regression,
                "prediction_offset_stats": _value_stats(offsets),
                "step_median_prediction_offset_stats": _value_stats(step_medians),
            }
        result["anchors"][anchor] = {
            "availability": by_workers,
            "causal_legal": all(
                row["anchor_later_than_prediction_count"] == 0
                and row["timestamp_regression_count"] == 0
                for row in by_workers.values()
            ),
            "online_implementable": True,
            "online_state": {
                "step_begin": "existing causal STEP_BEGIN event",
                "first_prediction": "retain first enqueue timestamp per Decode Step",
                "first_stable_event": "retain first predeclared LAYER_BEGIN per Decode Step",
            }[anchor],
            "by_active_workers": by_workers,
        }

    for left_index, left in enumerate(ANCHORS):
        for right in ANCHORS[left_index + 1:]:
            key = f"{left}_minus_{right}"
            rows = {}
            for workers in sorted({run.workers for run in runs}):
                values = []
                for run in (item for item in runs if item.workers == workers):
                    for step in run.sorted_steps():
                        left_ts = step.anchor(left)
                        right_ts = step.anchor(right)
                        if left_ts is not None and right_ts is not None:
                            values.append(left_ts - right_ts)
                rows[str(workers)] = _value_stats(values)
            result["pairwise_anchor_differences"][key] = rows
    return result


def build_ordinal_audit(runs: Sequence[RunEvidence]) -> dict[str, Any]:
    by_workers = {}
    for workers in sorted({run.workers for run in runs}):
        selected = [run for run in runs if run.workers == workers]
        by_workers[str(workers)] = {
            "run_count": len(selected),
            "step_count": sum(run.audit.get("decode_step_count", 0) for run in selected),
            "task_count": sum(run.audit.get("decode_task_count", 0) for run in selected),
            "timestamp_tie_count": sum(
                run.audit.get("ordinal_timestamp_ties", 0) for run in selected
            ),
            "sorted_task_id_regression_count": sum(
                run.audit.get("sorted_task_id_regressions", 0) for run in selected
            ),
            "file_order_prediction_regression_count": sum(
                run.audit.get("file_order_prediction_regressions", 0)
                for run in selected
            ),
            "file_order_task_id_regression_count": sum(
                run.audit.get("file_order_task_id_regressions", 0)
                for run in selected
            ),
            "missing_ordinal_source_count": sum(
                run.audit.get("invalid_task_timestamps", 0) for run in selected
            ),
        }
    return {
        "reconstruction_key": "(prediction_ts_ns, task_id)",
        "task_id_role": "deterministic tie-breaker only",
        "file_line_order_used_as_ordinal": False,
        "diagnostic_step_bands_use_future_final_count": True,
        "diagnostic_step_bands_online_eligible": False,
        "online_step_bands_source": "median task count of completed prior Steps",
        "by_active_workers": by_workers,
        "reconstructable": all(
            row["missing_ordinal_source_count"] == 0 for row in by_workers.values()
        ),
    }


def build_stage_group_audit(runs: Sequence[RunEvidence]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for step in run.sorted_steps():
            grouped: defaultdict[tuple[int, str], list[TaskRecord]] = defaultdict(list)
            for task in step.tasks:
                grouped[(task.layer, task.stage)].append(task)
            for (layer, stage), tasks in sorted(grouped.items()):
                first_uses = {task.first_use_ts_ns for task in tasks}
                rows.append({
                    "run_id": run.run_id,
                    "workers": run.workers,
                    "step": step.step,
                    "layer": layer,
                    "stage": stage,
                    "task_count": len(tasks),
                    "distinct_expert_count": len({task.expert for task in tasks}),
                    "distinct_tensor_count": len({task.tensor for task in tasks}),
                    "distinct_first_use_ts_count": len(first_uses),
                    "earliest_first_use_ts": min(first_uses),
                    "latest_first_use_ts": max(first_uses),
                    "first_use_span_ns": max(first_uses) - min(first_uses),
                })
    return {
        "stage_group_count": len(rows),
        "by_active_workers": {
            str(workers): {
                "group_count": sum(int(row["workers"] == workers) for row in rows),
                "task_count": sum(
                    row["task_count"] for row in rows if row["workers"] == workers
                ),
                "multi_first_use_group_count": sum(
                    int(row["workers"] == workers and row["distinct_first_use_ts_count"] > 1)
                    for row in rows
                ),
                "first_use_span_stats": _value_stats([
                    row["first_use_span_ns"] for row in rows
                    if row["workers"] == workers
                ]),
            }
            for workers in sorted({run.workers for run in runs})
        },
        "groups": rows,
    }


def _pearson(feature: np.ndarray, target: np.ndarray) -> float | None:
    if feature.size < 2 or np.std(feature) == 0 or np.std(target) == 0:
        return None
    return float(np.corrcoef(feature.astype(np.float64), target.astype(np.float64))[0, 1])


def _direction(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value > 0.02:
        return "positive"
    if value < -0.02:
        return "negative"
    return "near_zero"


def build_feature_audit(
    definition: CandidateDefinition,
    predicted_h: np.ndarray,
    states: np.ndarray,
    dataset: EvaluationDataset,
) -> dict[str, Any]:
    missing = np.iinfo(np.int64).min
    first_prediction_delay = (
        dataset.prediction_offsets_by_anchor[definition.anchor_kind]
        - dataset.prediction_offsets_by_anchor["first_prediction"]
    )
    numeric_features = {
        "layer": (dataset.layers.astype(np.int64), True),
        "prediction_relative_time": (
            dataset.prediction_offsets_by_anchor[definition.anchor_kind], True
        ),
        "task_ordinal": (dataset.ordinals.astype(np.int64), True),
        "created_task_count": (dataset.created_task_count.astype(np.int64), True),
        "observed_layer_count": (
            dataset.observed_layer_count.astype(np.int64), True
        ),
        "previous_step_duration": (dataset.previous_step_duration, True),
        "current_step_early_anchor_delay": (first_prediction_delay, True),
        "early_late_pair_interval": (dataset.early_late_pair_interval, False),
    }
    mature = states == STATE_MATURE
    signed_error = predicted_h - dataset.actual_h
    absolute_error = np.abs(signed_error)
    result: dict[str, Any] = {}
    for name, (feature, online_eligible) in numeric_features.items():
        available = feature != missing
        rows = {}
        directions = []
        for workers in sorted(set(int(value) for value in dataset.workers)):
            mask = mature & available & (dataset.workers == workers)
            signed = _pearson(feature[mask], signed_error[mask])
            absolute = _pearson(feature[mask], absolute_error[mask])
            direction = _direction(absolute)
            directions.append(direction)
            rows[str(workers)] = {
                "count": int(np.count_nonzero(mask)),
                "missing_count": int(np.count_nonzero(
                    mature & ~available & (dataset.workers == workers)
                )),
                "missing_rate": _ratio(
                    int(np.count_nonzero(mature & ~available & (dataset.workers == workers))),
                    int(np.count_nonzero(mature & (dataset.workers == workers))),
                ),
                "pearson_signed_error": signed,
                "pearson_absolute_error": absolute,
                "absolute_error_direction": direction,
            }
        result[name] = {
            "online_eligible": online_eligible,
            "by_active_workers": rows,
            "workers_direction_consistent": (
                len(set(directions)) == 1 and directions[0] != "unavailable"
            ),
        }

    for name, values, labels in (
        ("stage", dataset.stage_codes, {0: "UNKNOWN", 1: "EARLY", 2: "LATE"}),
        ("workers", dataset.workers, None),
    ):
        groups = {}
        for code in sorted(set(int(value) for value in values)):
            mask = mature & (values == code)
            groups[labels.get(code, str(code)) if labels else str(code)] = {
                "count": int(np.count_nonzero(mask)),
                "signed_error_mean_ns": (
                    float(np.mean(signed_error[mask])) if np.any(mask) else None
                ),
                "mae_ns": float(np.mean(absolute_error[mask])) if np.any(mask) else None,
            }
        result[name] = {
            "online_eligible": True,
            "type": "categorical_grouped_effect",
            "groups": groups,
        }
    return {
        "candidate": definition.candidate_id,
        "target": "logical first-use signed and absolute error",
        "features": result,
        "causal_note": (
            "Correlation is explanatory only; only fields marked online_eligible "
            "may be considered by a future Shadow implementation"
        ),
    }


def _critical_evidence_errors(runs: Sequence[RunEvidence]) -> int:
    fields = (
        "relevant_json_parse_errors", "decode_records_missing_required_fields",
        "invalid_task_ids", "duplicate_task_ids",
        "missing_frozen_baseline_prediction", "invalid_task_timestamps",
        "worker_timestamp_regressions", "first_use_causality_errors",
        "semantic_alignment_errors", "invalid_frozen_baseline_values",
        "frozen_issue_formula_errors", "frozen_return_formula_errors",
    )
    return sum(run.audit.get(field, 0) for run in runs for field in fields)


def build_evidence_integrity(
    runs: Sequence[RunEvidence], m4a1: dict[str, Any], m4a1_path: Path
) -> dict[str, Any]:
    run_rows = []
    inputs_unchanged = True
    for run in runs:
        memory_path = run.run_dir / "memory_trace.jsonl"
        manifest_path = run.run_dir / "run_manifest.json"
        current = {
            "memory_trace": _file_identity(memory_path),
            "run_manifest": _file_identity(manifest_path, include_sha256=True),
        }
        summary_path = run.run_dir / "summary.json"
        summary = None
        trace_integrity = None
        if summary_path.exists():
            current["summary"] = _file_identity(summary_path, include_sha256=True)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            sinks = summary.get("sinks", {})
            trace_integrity = all(
                not sink.get("enabled")
                or (
                    sink.get("enqueued") == sink.get("written")
                    and sink.get("dropped") == 0
                )
                for sink in sinks.values()
            )
        unchanged = current == run.input_identity_before
        inputs_unchanged &= unchanged
        run_rows.append({
            "run_id": run.run_id,
            "run_dir": str(run.run_dir),
            "active_workers": run.workers,
            "manifest": {
                "git_commit": run.manifest.get("git_commit"),
                "git_dirty": run.manifest.get("git_dirty"),
                "model": run.manifest.get("model"),
                "prompt": run.manifest.get("prompt"),
                "binary": run.manifest.get("binary"),
                "seed": run.manifest.get("environment", {}).get("SEED"),
                "trace_profile": run.manifest.get("experiment", {}).get("trace_profile"),
                "shadow_mode": run.manifest.get("environment", {}).get(
                    "LLM_MEM_TRACE_OPT_EXPERT_SLACK_MODE"
                ),
                "cgroup": run.manifest.get("experiment", {}).get("cgroup"),
            },
            "all_shadow_count": run.all_shadow_count,
            "phase_counts": run.phase_counts,
            "decode_finalized_count": run.audit.get("decode_task_count", 0),
            "audit": run.audit,
            "trace_integrity": trace_integrity,
            "sinks": summary.get("sinks", {}) if summary else None,
            "input_identity_before": run.input_identity_before,
            "input_identity_after": current,
            "input_unchanged": unchanged,
        })
    equivalence = m4a1.get("equivalence", [])
    output_hashes = sorted({
        row.get("shadow_output_sha256") for row in equivalence
        if row.get("shadow_output_sha256")
    })
    return {
        "m4a1_machine_report": {
            **_file_identity(m4a1_path, include_sha256=True),
            "schema_version": m4a1.get("schema_version"),
        },
        "runs": run_rows,
        "critical_evidence_error_count": _critical_evidence_errors(runs),
        "all_inputs_unchanged": inputs_unchanged,
        "all_trace_integrity_passed": all(
            row["trace_integrity"] is True for row in run_rows
        ),
        "m4a1_equivalence_all_passed": bool(equivalence) and all(
            row.get("passed") is True for row in equivalence
        ),
        "output_hashes": output_hashes,
        "output_hash_consistent": len(output_hashes) == 1,
        "hint_multiset_equal": bool(equivalence) and all(
            row.get("hint_multiset_equal") is True for row in equivalence
        ),
        "same_binary": len({
            run.manifest.get("binary", {}).get("sha256") for run in runs
        }) == 1,
        "same_model": len({
            run.manifest.get("model", {}).get("path") for run in runs
        }) == 1,
        "same_prompt_hash": len({
            run.manifest.get("prompt", {}).get("sha256") for run in runs
        }) == 1,
        "same_seed": len({
            run.manifest.get("environment", {}).get("SEED") for run in runs
        }) == 1,
        "delegated_cgroup": all(
            run.manifest.get("experiment", {}).get("cgroup", {}).get("memory.max")
            not in (None, "max") for run in runs
        ),
        "performance_claim_allowed": False,
    }


def _candidate_worker_gate(comparison: dict[str, Any]) -> dict[str, Any]:
    worker_rows = comparison["by_active_workers"]
    workers_present = all(str(worker) in worker_rows for worker in (2, 4))
    mae_reference = workers_present and all(
        (worker_rows[str(worker)]["first_use"]["relative_mae_improvement"] or -1.0)
        >= 0.30 for worker in (2, 4)
    )
    p95_reference = workers_present and all(
        (worker_rows[str(worker)]["first_use"]["relative_p95_improvement"] or -1.0)
        >= 0.20 for worker in (2, 4)
    )
    p95_non_degradation = workers_present and all(
        (worker_rows[str(worker)]["first_use"]["relative_p95_improvement"] or -1.0)
        >= 0.0 for worker in (2, 4)
    )
    positive_mae = workers_present and all(
        (worker_rows[str(worker)]["first_use"]["relative_mae_improvement"] or -1.0)
        > 0.0 for worker in (2, 4)
    )
    late_precision = workers_present and all(
        worker_rows[str(worker)][target]["predicted_late_precision_delta"] is not None
        and worker_rows[str(worker)][target]["predicted_late_precision_delta"] >= 0.05
        for worker in (2, 4) for target in ("issue", "return")
    )
    coverage = workers_present and all(
        (
            worker_rows[str(worker)][target]["candidate"]["thresholds"][0]
            ["threshold_coverage_of_eligible"] or 0.0
        ) >= 0.01
        for worker in (2, 4) for target in ("issue", "return")
    )
    calibration_nonworse = workers_present and all(
        worker_rows[str(worker)][target]["candidate"]["calibration_monotonicity"]
        ["adjacent_decrease_count"]
        <= worker_rows[str(worker)][target]["baseline"]["calibration_monotonicity"]
        ["adjacent_decrease_count"]
        for worker in (2, 4) for target in ("issue", "return")
    )
    calibration_strict_improvement = workers_present and any(
        worker_rows[str(worker)][target]["candidate"]["calibration_monotonicity"]
        ["adjacent_decrease_count"]
        < worker_rows[str(worker)][target]["baseline"]["calibration_monotonicity"]
        ["adjacent_decrease_count"]
        for worker in (2, 4) for target in ("issue", "return")
    )
    return {
        "workers_2_and_4_present": workers_present,
        "mae_improvement_at_least_30_percent_both_workers": mae_reference,
        "p95_improvement_at_least_20_percent_both_workers": p95_reference,
        "p95_non_degradation_both_workers": p95_non_degradation,
        "positive_mae_improvement_both_workers": positive_mae,
        "predicted_late_precision_delta_at_least_5pp_both_targets_workers": late_precision,
        "zero_threshold_coverage_at_least_1_percent_both_targets_workers": coverage,
        "calibration_reversals_nonworse_all_targets_workers": calibration_nonworse,
        "calibration_has_strict_improvement": calibration_strict_improvement,
        "all_reference_conditions": all((
            mae_reference, p95_reference, late_precision, coverage,
            calibration_nonworse, calibration_strict_improvement,
        )),
    }


def _select_best_candidate(
    definitions: Sequence[CandidateDefinition],
    comparisons: dict[str, Any],
) -> tuple[CandidateDefinition, dict[str, Any]]:
    eligible = []
    for definition in definitions:
        if definition.semantic != "tensor":
            continue
        comparison = comparisons[definition.candidate_id]
        workers = comparison.get("by_active_workers", {})
        if not all(str(worker) in workers for worker in (2, 4)):
            continue
        improvements = [
            workers[str(worker)]["first_use"]["relative_mae_improvement"]
            for worker in (2, 4)
        ]
        if any(value is None for value in improvements):
            continue
        p95 = [
            workers[str(worker)]["first_use"]["relative_p95_improvement"]
            for worker in (2, 4)
        ]
        score = (
            min(improvements),
            min(value if value is not None else -math.inf for value in p95),
            sum(improvements),
        )
        eligible.append((score, definition))
    if not eligible:
        raise RuntimeError("no tensor Step template candidate has paired workers=2/4 evidence")
    eligible.sort(key=lambda item: item[0], reverse=True)
    definition = eligible[0][1]
    return definition, {
        "selection_rule": (
            "maximize the worse-worker paired mature Decode relative MAE improvement; "
            "then worse-worker p95 improvement; then summed MAE improvement"
        ),
        "thresholds_not_used_for_candidate_selection": True,
        "selected_score": {
            "worst_worker_relative_mae_improvement": eligible[0][0][0],
            "worst_worker_relative_p95_improvement": eligible[0][0][1],
            "sum_relative_mae_improvement": eligible[0][0][2],
        },
        "ranked_candidates": [
            {
                "candidate": candidate.candidate_id,
                "worst_worker_relative_mae_improvement": score[0],
                "worst_worker_relative_p95_improvement": score[1],
                "sum_relative_mae_improvement": score[2],
            }
            for score, candidate in eligible
        ],
    }


def _threshold_worker_consistency(selected_result: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for target in ("issue", "return"):
        workers = selected_result[target]["by_active_workers"]
        rows = []
        for index, threshold in enumerate(NEGATIVE_THRESHOLDS_NS):
            w2 = workers["2"]["thresholds"][index]
            w4 = workers["4"]["thresholds"][index]
            rows.append({
                "threshold_ns": threshold,
                "workers=2": w2,
                "workers=4": w4,
                "precision_available_both_workers": (
                    w2["predicted_late_precision"] is not None
                    and w4["predicted_late_precision"] is not None
                ),
                "nonzero_coverage_both_workers": (
                    (w2["threshold_coverage_of_eligible"] or 0.0) > 0
                    and (w4["threshold_coverage_of_eligible"] or 0.0) > 0
                ),
                "precision_direction_consistent_with_stricter_threshold": (
                    True if index == 0 else (
                        (
                            w2["predicted_late_precision"]
                            >= workers["2"]["thresholds"][index - 1]
                            ["predicted_late_precision"]
                        ) == (
                            w4["predicted_late_precision"]
                            >= workers["4"]["thresholds"][index - 1]
                            ["predicted_late_precision"]
                        )
                        if w2["predicted_late_precision"] is not None
                        and w4["predicted_late_precision"] is not None
                        and workers["2"]["thresholds"][index - 1]
                        ["predicted_late_precision"] is not None
                        and workers["4"]["thresholds"][index - 1]
                        ["predicted_late_precision"] is not None
                        else None
                    )
                ),
            })
        result[target] = rows
    return result


def analyze_m4a2(
    runs: Sequence[RunEvidence],
    m4a1: dict[str, Any],
    m4a1_path: Path,
) -> dict[str, Any]:
    if sorted(run.workers for run in runs) != [2, 4]:
        raise ValueError("M4A.2 requires exactly one workers=2 and one workers=4 Detail run")
    datasets = {
        semantic: build_evaluation_dataset(runs, semantic)
        for semantic in ("tensor", "earliest_stage")
    }
    definitions = candidate_definitions()
    all_results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    prediction_vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    baseline_results = {}
    for semantic, dataset in datasets.items():
        states = _baseline_states(dataset)
        baseline_results[semantic] = summarize_candidate(
            dataset.baseline_h, states, dataset
        )

    for definition in definitions:
        dataset = datasets[definition.semantic]
        predicted_h, states, accounting = replay_candidate(
            definition, dataset, runs
        )
        result = {
            "metadata": definition.metadata(),
            "state_accounting": accounting,
            **summarize_candidate(predicted_h, states, dataset),
        }
        if definition.semantic == "tensor":
            result["semantic_sensitivity"] = {
                "per_tensor_representative": summarize_candidate(
                    predicted_h,
                    states,
                    dataset,
                    base_mask=dataset.per_tensor_representative,
                    compact=True,
                ),
                "task_level_one_to_many": {
                    "first_use": result["first_use"]["mature_exact"],
                    "issue": result["issue"]["mature_exact"],
                    "return": result["return"]["mature_exact"],
                },
            }
            prediction_vectors[definition.candidate_id] = (predicted_h, states)
        all_results[definition.candidate_id] = result
        comparisons[definition.candidate_id] = _paired_comparison(
            predicted_h, states, dataset
        )

    selected, selection = _select_best_candidate(definitions, comparisons)
    selected_h, selected_states = prediction_vectors[selected.candidate_id]
    selected_result = all_results[selected.candidate_id]
    selected_comparison = comparisons[selected.candidate_id]
    gate = _candidate_worker_gate(selected_comparison)

    evidence_integrity = build_evidence_integrity(runs, m4a1, m4a1_path)
    anchor_audit = build_anchor_audit(runs)
    ordinal_audit = build_ordinal_audit(runs)
    tensor_dataset = datasets["tensor"]
    multi_record_semantics = {
        "primary": tensor_dataset.multi_record_audit,
        "earliest_stage": datasets["earliest_stage"].multi_record_audit,
        "stage_group_audit": build_stage_group_audit(runs),
        "confusion_matrices_mixed_across_semantics": False,
    }

    oracle_state = np.full(len(tensor_dataset.samples), STATE_MATURE, dtype=np.uint8)
    full_oracle = {}
    for anchor in ANCHORS:
        full_oracle[anchor] = {
            "metadata": {
                "offline_diagnostic_only": True,
                "online_eligible": False,
                "uses_current_step_actual": True,
                "uses_future_information": True,
                "anchor_kind": anchor,
                "template_key": "task-level actual logical first-use",
                "estimator": "full_oracle",
                "window": 0,
                "workers_scope": "diagnostic_combined",
                "fallback_policy": "none",
                "mature_rule": "not applicable; offline diagnostic only",
            },
            **summarize_candidate(
                tensor_dataset.actual_h, oracle_state, tensor_dataset, compact=True
            ),
        }
        full_oracle[anchor]["coverage_accounting"]["oracle_only"] = len(
            tensor_dataset.samples
        )

    critical = evidence_integrity["critical_evidence_error_count"]
    anchor_usable = all(
        anchor_audit["anchors"][anchor]["causal_legal"]
        for anchor in ANCHORS
    )
    engineering_pass = all((
        critical == 0,
        evidence_integrity["all_inputs_unchanged"],
        evidence_integrity["all_trace_integrity_passed"],
        evidence_integrity["m4a1_equivalence_all_passed"],
        evidence_integrity["output_hash_consistent"],
        ordinal_audit["reconstructable"],
        anchor_usable,
        len(all_results) == len(definitions),
    ))
    if not engineering_pass and (
        critical > 0 or not ordinal_audit["reconstructable"] or not anchor_usable
    ):
        conclusion = "现有 Trace 字段不足，需最小 Shadow-only 补充后再判断"
    elif gate["all_reference_conditions"]:
        conclusion = "Step Template 明显有效，可以提出下一阶段 Shadow 实现方案"
    elif (
        gate["positive_mae_improvement_both_workers"]
        and gate["p95_non_degradation_both_workers"]
    ):
        conclusion = "Step Template 有有限改善，但不足以继续"
    else:
        conclusion = "Step Template 无效，建议停止 Slack 路线"
    continue_slack = conclusion.startswith("Step Template 明显有效")
    stop_recommendation = None if continue_slack else (
        "停止继续改进 Slack first-use predictor，\n"
        "不进入 M4B，\n"
        "后续转向 M5A Pressure Shadow。"
    )

    all_model_results = m4a1.get("all_model_results", {})
    result: dict[str, Any] = {
        "schema_version": 1,
        "milestone": "M4A.2 Decode Step-aligned Execution Template feasibility",
        "scope": {
            "shadow_only": True,
            "offline_only": True,
            "existing_detail_evidence_only": True,
            "new_inference_runs": 0,
            "runtime_changed": False,
            "active_control_changed": False,
            "performance_claim": False,
        },
        "input_artifacts": {
            "detail_run_dirs": [str(run.run_dir) for run in runs],
            "m4a1_machine_report": str(m4a1_path.resolve()),
        },
        "frozen_baseline": {
            "candidate": BASELINE_KEY,
            "selection_rule": (
                "predeclared from M4A.1: minimize the worse workers=2/4 mature "
                "Decode first-use MAE"
            ),
            "reference": {
                "workers=2": {
                    "mature_count": 75_816,
                    "mae_ns": 6_300_417.175398333,
                    "p95_absolute_error_ns": 6_695_875.25,
                },
                "workers=4": {
                    "mature_count": 75_816,
                    "mae_ns": 1_713_672.206222961,
                    "p95_absolute_error_ns": 4_458_978.0,
                },
            },
            "recomputed_on_m4a2_semantics": baseline_results,
            "m4a1_primary_candidate": (
                "phase_layer_stage_p25|queue_depth_worker_ewma|residual_quantile"
            ),
            "m4a1_all_36_model_results": all_model_results,
            "m4a1_model_count": len(all_model_results),
        },
        "prequential_policy": {
            "order": "predict entire current Step; freeze; then update once per unit",
            "current_step_actual_used_for_current_prediction": False,
            "state_shared_across_runs": False,
            "state_shared_across_workers": False,
            "window": WINDOW,
            "minimum_mature_steps": MIN_MATURE_STEPS,
            "ewma_alpha": EWMA_ALPHA,
            "scale_clip": [0.5, 2.0],
            "quantile_method": "numpy linear",
        },
        "anchor_audit": anchor_audit,
        "ordinal_audit": ordinal_audit,
        "multi_record_semantics": multi_record_semantics,
        "candidate_definitions": {
            definition.candidate_id: definition.metadata()
            for definition in definitions
        },
        "all_candidate_results": all_results,
        "full_oracle_template": full_oracle,
        "paired_baseline_comparisons": comparisons,
        "selection": {
            **selection,
            "selected_candidate": selected.candidate_id,
            "reference_gate": gate,
        },
        "issue_slack": {
            "selected_candidate": selected.candidate_id,
            "prediction_formula": "H - Q - P",
            "actual_label": "issue_ts < logical_first_use_ts",
            "components_frozen_from_m4a1": ["Q", "P"],
            "result": selected_result["issue"],
            "paired_baseline": selected_comparison["by_active_workers"],
        },
        "return_slack": {
            "selected_candidate": selected.candidate_id,
            "prediction_formula": "H - Q - P - S",
            "actual_label": (
                "final_enabled_hint_return_ts < logical_first_use_ts"
            ),
            "components_frozen_from_m4a1": ["Q", "P", "S"],
            "result": selected_result["return"],
            "paired_baseline": selected_comparison["by_active_workers"],
        },
        "thresholds": {
            "fixed_thresholds_ns": list(NEGATIVE_THRESHOLDS_NS),
            "classification_rule": "predicted_late = predicted_slack_ns <= threshold_ns",
            "zero_is_late": True,
            "per_run_posthoc_selection": False,
            "selected_candidate_issue": selected_result["issue"]["mature_exact"]["thresholds"],
            "selected_candidate_return": selected_result["return"]["mature_exact"]["thresholds"],
            "worker_consistency": _threshold_worker_consistency(selected_result),
        },
        "calibration": {
            "bucket_labels": list(CALIBRATION_LABELS),
            "selected_candidate_issue": selected_result["issue"]["mature_exact"],
            "selected_candidate_return": selected_result["return"]["mature_exact"],
        },
        "feature_audit": build_feature_audit(
            selected, selected_h, selected_states, tensor_dataset
        ),
        "coverage_accounting": {
            "tensor_task_level_eligible": len(tensor_dataset.samples),
            "tensor_group_representatives": int(np.count_nonzero(
                tensor_dataset.per_tensor_representative
            )),
            "earliest_stage_eligible": len(datasets["earliest_stage"].samples),
            "selected_candidate": selected_result["coverage_accounting"],
            "all_candidates_conservative": all(
                item["coverage_accounting"]["eligible"]
                == item["coverage_accounting"]["predicted"]
                + item["coverage_accounting"]["ambiguous"]
                + item["coverage_accounting"]["unavailable"]
                for item in all_results.values()
            ),
        },
        "evidence_integrity": evidence_integrity,
        "limitations": [
            "Only one existing Detail Evidence run is available per worker count; independent cross-run repeat stability cannot be estimated",
            "Detail Trace is observation evidence and is not a formal performance benchmark",
            "No delegated cgroup was available; formal isolation is unavailable",
            "Full-oracle and diagnostic Step terciles use future information and are not online candidates",
            "ISSUE and RETURN are not page-in completion or physical residency",
        ],
        "validation": {
            "recorded_separately_after_report_generation": True,
            "required_record": "M4A2_validation.json",
            "required_checks": [
                "M4A.2 targeted unit tests",
                "all llama.cpp/trace/tests Python regression tests",
                "python3 -m py_compile llama.cpp/trace/*.py",
                "strict machine JSON schema/accounting audit",
                "git diff --check",
            ],
        },
        "acceptance": {
            "engineering": "通过" if engineering_pass else "不通过",
            "model": conclusion,
            "reference_gate": gate,
            "continue_slack_route": continue_slack,
            "stop_recommendation": stop_recommendation,
            "enter_m4b": False,
            "enter_m5a": False,
            "active_control_changed": False,
            "human_approval_required": True,
        },
        "artifacts": {},
    }
    return result


def _ms(value: float | int | None) -> str:
    return "-" if value is None else f"{float(value) / 1e6:.3f}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.2f}%"


def _ratio_text(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M4A.2 Decode Step-aligned Execution Template 可行性报告",
        "",
        "> Shadow-only、offline-only。本文只报告 logical first-use 与 Hint ISSUE/RETURN 的时间关系；ISSUE/RETURN 均不表示 page-in complete 或页面驻留。",
        "",
        "## 1. 范围与 Evidence",
        "",
        "本次只读取既有 workers=2/4 M4A.1 Detail Evidence，没有运行新推理，没有修改运行时预测、Comparator、Admission、Task 或 Hint。",
        "",
        "| Run | workers | Shadow Task | PREFILL | DECODE | Trace integrity | Input unchanged |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    evidence = result["evidence_integrity"]
    for run in evidence["runs"]:
        lines.append(
            f"| `{run['run_id']}` | {run['active_workers']} | "
            f"{run['all_shadow_count']} | {run['phase_counts'].get('PREFILL', 0)} | "
            f"{run['phase_counts'].get('DECODE', 0)} | "
            f"{'PASS' if run['trace_integrity'] else 'FAIL'} | "
            f"{'yes' if run['input_unchanged'] else 'no'} |"
        )
    lines.extend((
        "",
        f"- M4A.1 equivalence：{'PASS' if evidence['m4a1_equivalence_all_passed'] else 'FAIL'}；output hash 一致：{'yes' if evidence['output_hash_consistent'] else 'no'}。",
        f"- Hint multiset 等价：{'yes' if evidence['hint_multiset_equal'] else 'no'}；关键 Evidence 错误数：{evidence['critical_evidence_error_count']}。",
        f"- 同一 binary/model/Prompt hash/Seed：{'yes' if all((evidence['same_binary'], evidence['same_model'], evidence['same_prompt_hash'], evidence['same_seed'])) else 'no'}。",
        f"- delegated cgroup：{'available' if evidence['delegated_cgroup'] else 'unavailable'}；因此不宣称正式隔离或性能收益。",
        "",
        "## 2. Anchor 与 ordinal 审计",
        "",
        "第三种稳定 Anchor 的预声明白名单为 `LAYER_BEGIN`。所有 Anchor 均在预测前或与最早 prediction 同时可见。",
        "",
        "| Anchor | workers | Step coverage | Task coverage | Missing | Duplicate | Later than prediction | Causal |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ))
    for anchor, value in result["anchor_audit"]["anchors"].items():
        for workers, row in value["by_active_workers"].items():
            lines.append(
                f"| `{anchor}` | {workers} | {_pct(row['step_coverage'])} | "
                f"{_pct(row['task_coverage'])} | {row['missing_anchor_count']} | "
                f"{row['duplicate_anchor_count']} | "
                f"{row['anchor_later_than_prediction_count']} | "
                f"{'yes' if value['causal_legal'] else 'no'} |"
            )
    ordinal = result["ordinal_audit"]
    lines.extend((
        "",
        "Ordinal 使用 `(prediction_ts_ns, task_id)` 重建；`task_id` 仅用于 timestamp tie 的确定性打破，文件行序不作为 enqueue 顺序。",
        "",
        "| workers | Steps | Tasks | Timestamp ties | Sorted task-id regressions | File-order prediction regressions |",
        "|---:|---:|---:|---:|---:|---:|",
    ))
    for workers, row in ordinal["by_active_workers"].items():
        lines.append(
            f"| {workers} | {row['step_count']} | {row['task_count']} | "
            f"{row['timestamp_tie_count']} | {row['sorted_task_id_regression_count']} | "
            f"{row['file_order_prediction_regression_count']} |"
        )
    lines.extend((
        "",
        "完成后按最终 Task 数三等分的 Step 前/中/后段仅用于诊断；在线分段只使用已完成历史 Step 的 Task 数边界。当前每个 worker 只有一个 Detail Run，不能估计独立跨 Run 重复稳定性。",
        "",
        "## 3. 多记录与预测目标语义",
        "",
    ))
    primary = result["multi_record_semantics"]["primary"]
    stage_audit = result["multi_record_semantics"]["stage_group_audit"]
    lines.extend((
        f"主语义为按 `(run, step, layer, stage, tensor)` 规范化模板观测、组内只更新一次，再一对多展开到 {primary['task_count']} 个 Task。Tensor group={primary['group_count']}，多 Task group={primary['multi_task_group_count']}，ambiguous group={primary['ambiguous_group_count']}，ambiguous Task={primary['ambiguous_task_count']}。",
        "",
        "`earliest-stage` 使用每个 `(run, step, layer, stage)` 最早 logical first-use 对应 Task，单独输出，不与 task-level confusion matrix 混合。",
        "",
        "| workers | Stage groups | Tasks | Multi-first-use groups | First-use span p95 ms |",
        "|---:|---:|---:|---:|---:|",
    ))
    for workers, row in stage_audit["by_active_workers"].items():
        lines.append(
            f"| {workers} | {row['group_count']} | {row['task_count']} | "
            f"{row['multi_first_use_group_count']} | "
            f"{_ms(row['first_use_span_stats']['p95_ns'])} |"
        )

    frozen = result["frozen_baseline"]
    lines.extend((
        "",
        "## 4. 冻结 baseline 与全部候选",
        "",
        f"冻结 M4A.1 baseline：`{frozen['candidate']}`。M4A.1 的 36 个完整候选结果均保存在机器 JSON；本阶段没有按 Step Template 结果更换 baseline。",
        "",
        "相对改善均在 candidate 与 baseline 完全相同的 paired mature Task 上计算。候选选择规则不使用阈值结果。",
        "",
        "| Candidate | w2 paired N | w2 MAE ms | w2 MAE Δ | w2 p95 Δ | w4 paired N | w4 MAE ms | w4 MAE Δ | w4 p95 Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    comparisons = result["paired_baseline_comparisons"]
    for candidate in sorted(result["all_candidate_results"]):
        comparison = comparisons[candidate]["by_active_workers"]
        w2 = comparison.get("2", {})
        w4 = comparison.get("4", {})
        w2_first = w2.get("first_use", {})
        w4_first = w4.get("first_use", {})
        lines.append(
            f"| `{candidate}` | {w2.get('paired_count', 0)} | "
            f"{_ms(w2_first.get('candidate', {}).get('mae_ns'))} | "
            f"{_pct(w2_first.get('relative_mae_improvement'))} | "
            f"{_pct(w2_first.get('relative_p95_improvement'))} | "
            f"{w4.get('paired_count', 0)} | "
            f"{_ms(w4_first.get('candidate', {}).get('mae_ns'))} | "
            f"{_pct(w4_first.get('relative_mae_improvement'))} | "
            f"{_pct(w4_first.get('relative_p95_improvement'))} |"
        )

    selected = result["selection"]["selected_candidate"]
    selected_result = result["all_candidate_results"][selected]
    selected_pair = comparisons[selected]["by_active_workers"]
    lines.extend((
        "",
        "## 5. 最佳 Step Template first-use 结果",
        "",
        f"按预声明 worst-worker paired MAE 规则选出的诊断最佳候选为 `{selected}`。该选择只是 M4A.2 可行性比较，不是运行时阈值或 Active 策略。",
        "",
        "| workers | Paired N | Baseline MAE ms | Candidate MAE ms | MAE improvement | Baseline p95 ms | Candidate p95 ms | p95 improvement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for workers in (2, 4):
        row = selected_pair[str(workers)]
        first = row["first_use"]
        lines.append(
            f"| {workers} | {row['paired_count']} | "
            f"{_ms(first['baseline']['mae_ns'])} | {_ms(first['candidate']['mae_ns'])} | "
            f"{_pct(first['relative_mae_improvement'])} | "
            f"{_ms(first['baseline']['p95_absolute_error_ns'])} | "
            f"{_ms(first['candidate']['p95_absolute_error_ns'])} | "
            f"{_pct(first['relative_p95_improvement'])} |"
        )
    coverage = selected_result["coverage_accounting"]
    lines.extend((
        "",
        f"Task-level coverage：eligible={coverage['eligible']}，mature={coverage['mature_exact']}，fallback={coverage['fallback']}，ambiguous={coverage['ambiguous']}，unavailable={coverage['unavailable']}。",
        "",
        "### EARLY/LATE 与 Layer 分层",
        "",
        "| Scope | N | MAE ms | median abs ms | p95 abs ms | signed mean ms |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    first = selected_result["first_use"]
    for scope, row in first["by_stage_workers"].items():
        lines.append(
            f"| `{scope}` | {row['count']} | {_ms(row['mae_ns'])} | "
            f"{_ms(row['median_absolute_error_ns'])} | "
            f"{_ms(row['p95_absolute_error_ns'])} | "
            f"{_ms(row['signed_error_mean_ns'])} |"
        )
    lines.extend((
        "",
        "完整每 Layer、Step、ordinal、诊断/在线 Step 区段结果保存在机器 JSON 的 `all_candidate_results.<candidate>.first_use`。",
        "其中 `by_layer_stage_workers` 明确分离每个 Layer 的 EARLY/LATE，避免把不同 Stage 隐式混合。",
        "",
        "## 6. Issue/Return Slack 分类",
        "",
        "只替换 H；Q/P/S 固定使用 M4A.1 Queue A 同一 Task 的预测。Issue=`H-Q-P`，Return=`H-Q-P-S`，两套实际标签和 confusion matrix 完全独立。",
        "",
        "| Target | workers | Paired N | Baseline late precision | Candidate late precision | Δ | Candidate late recall | F1 | Zero-threshold coverage | Calibration reversals |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for target in ("issue", "return"):
        for workers in (2, 4):
            row = selected_pair[str(workers)][target]
            candidate = row["candidate"]
            zero = candidate["thresholds"][0]
            lines.append(
                f"| {target.upper()} | {workers} | {row['paired_count']} | "
                f"{_ratio_text(row['baseline']['predicted_late_precision'])} | "
                f"{_ratio_text(candidate['predicted_late_precision'])} | "
                f"{_ratio_text(row['predicted_late_precision_delta'])} | "
                f"{_ratio_text(candidate['predicted_late_recall'])} | "
                f"{_ratio_text(candidate['predicted_late_f1'])} | "
                f"{_pct(zero['threshold_coverage_of_eligible'])} | "
                f"{candidate['calibration_monotonicity']['adjacent_decrease_count']} |"
            )

    lines.extend((
        "",
        "### Conservative negative thresholds",
        "",
        "| Target | workers | Threshold ms | Predicted-late N | Precision | Recall | Coverage eligible | False-reject rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for target in ("issue", "return"):
        target_result = selected_result[target]
        for workers in (2, 4):
            worker = target_result["by_active_workers"][str(workers)]
            for row in worker["thresholds"]:
                lines.append(
                    f"| {target.upper()} | {workers} | "
                    f"{row['threshold_ns'] / 1e6:.1f} | "
                    f"{row['predicted_late_count']} | "
                    f"{_ratio_text(row['predicted_late_precision'])} | "
                    f"{_ratio_text(row['predicted_late_recall'])} | "
                    f"{_pct(row['threshold_coverage_of_eligible'])} | "
                    f"{_pct(row['false_reject_candidate_rate'])} |"
                )
    lines.extend((
        "",
        "`slack == 0` 始终归入 predicted late。各固定 calibration 桶的 N、actual on-time rate、Wilson 95% 区间、insufficient 标记及 EARLY/LATE/Layer 分层均保存在机器 JSON。",
        "",
        "## 7. Full-oracle 与特征审计",
        "",
        "Full-oracle first-use MAE 为 0，只作为 Step 结构诊断上界；它使用当前 Step 真实结果，明确标记 `online_eligible=false`，不进入候选选择。",
        "",
        "特征审计覆盖 layer、stage、prediction 相对时间、ordinal、已创建 Task 数、已观察 Layer 数、workers、前一 Step 时长、当前 Step early Anchor 及 EARLY/LATE 配对间隔。配对间隔使用未来信息，只作离线解释。完整相关方向、缺失率和 worker 一致性保存在机器 JSON。",
        "",
        "## 8. 验收与结论",
        "",
    ))
    gate = result["acceptance"]["reference_gate"]
    for key, value in gate.items():
        lines.append(f"- `{key}`：{value}")
    lines.extend((
        "",
        f"- 工程验收：**{result['acceptance']['engineering']}**。",
        f"- 模型结论：**{result['acceptance']['model']}**。",
    ))
    if result["acceptance"]["stop_recommendation"]:
        lines.extend((
            "",
            "```text",
            result["acceptance"]["stop_recommendation"],
            "```",
        ))
    lines.extend((
        "",
        "无论模型结论如何，本阶段 `enter_m4b=false`、`enter_m5a=false`、`active_control_changed=false`，后续步骤必须等待人工批准。",
        "",
        "## 9. 局限与产物",
        "",
    ))
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend((
        "",
        f"- 完整机器 JSON：`{result['artifacts'].get('full_machine_json', '-')}`",
        f"- 人类可读报告：`{result['artifacts'].get('human_markdown', '-')}`",
        f"- 验证记录：`{result['artifacts'].get('validation_record', '-')}`",
        "",
    ))
    return "\n".join(lines)


def write_outputs(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "M4A2_decode_step_template_full.json"
    markdown_path = output_dir / "M4A2_decode_step_template_report.md"
    validation_path = output_dir / "M4A2_validation.json"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing M4A.2 result in {output_dir}"
        )
    result["artifacts"] = {
        "full_machine_json": str(json_path.resolve()),
        "human_markdown": str(markdown_path.resolve()),
        "validation_record": str(validation_path.resolve()),
        "detail_run_dirs": result["input_artifacts"]["detail_run_dirs"],
    }
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline-only M4A.2 Step Template feasibility analysis"
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--m4a1-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.run_dir) != 2:
        raise SystemExit("M4A.2 requires two --run-dir values: workers=2 and workers=4")
    m4a1_path = args.m4a1_report.resolve()
    m4a1 = json.loads(m4a1_path.read_text(encoding="utf-8"))
    runs = [load_run_evidence(path) for path in args.run_dir]
    result = analyze_m4a2(runs, m4a1, m4a1_path)
    json_path, markdown_path = write_outputs(result, args.output_dir.resolve())
    print(json.dumps({
        "engineering": result["acceptance"]["engineering"],
        "model": result["acceptance"]["model"],
        "selected_candidate": result["selection"]["selected_candidate"],
        "full_machine_json": str(json_path),
        "human_markdown": str(markdown_path),
        "enter_m4b": False,
        "enter_m5a": False,
        "active_control_changed": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
