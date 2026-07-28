#!/usr/bin/env python3
"""Offline M4A.1 analysis for observation-only Expert Shadow Slack records.

Issue and Return are intentionally separate prediction targets.  Oracle
substitution is diagnostic-only and never feeds the online predictors.
"""

from __future__ import annotations

from collections import defaultdict
from array import array
from statistics import mean
from typing import Any, Callable

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by minimal environments
    np = None


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
NEGATIVE_THRESHOLDS_NS = (-500_000, -1_000_000, -2_000_000, -5_000_000)


def percentile(values: list[int | float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def calibration_bucket(slack_ns: int) -> int:
    if slack_ns < -5_000_000:
        return 0
    if slack_ns < -2_000_000:
        return 1
    if slack_ns < -1_000_000:
        return 2
    if slack_ns < -500_000:
        return 3
    if slack_ns <= 0:
        return 4
    if slack_ns <= 500_000:
        return 5
    if slack_ns <= 1_000_000:
        return 6
    if slack_ns <= 2_000_000:
        return 7
    if slack_ns <= 5_000_000:
        return 8
    return 9


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _error_stats(signed_errors: list[int]) -> dict[str, Any]:
    absolute = [abs(value) for value in signed_errors]
    return {
        "count": len(signed_errors),
        "mae_ns": mean(absolute) if absolute else None,
        "median_absolute_error_ns": percentile(absolute, 50),
        "p75_absolute_error_ns": percentile(absolute, 75),
        "p95_absolute_error_ns": percentile(absolute, 95),
        "signed_error_mean_ns": mean(signed_errors) if signed_errors else None,
        "signed_error_p25_ns": percentile(signed_errors, 25),
        "signed_error_median_ns": percentile(signed_errors, 50),
        "signed_error_p75_ns": percentile(signed_errors, 75),
        "signed_error_p95_ns": percentile(signed_errors, 95),
    }


def _value_stats(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean_ns": mean(values) if values else None,
        "min_ns": min(values) if values else None,
        "median_ns": percentile(values, 50),
        "p75_ns": percentile(values, 75),
        "p95_ns": percentile(values, 95),
        "max_ns": max(values) if values else None,
    }


def _new_error_bucket() -> dict[str, Any]:
    return {
        "signed_errors": array("q"),
        "warmup": 0,
        "fallback": 0,
        "mature_exact": 0,
        "clipped": 0,
        "warmup_flags": bytearray(),
        "fallback_flags": bytearray(),
        "clipped_flags": bytearray(),
    }


def _new_target_bucket() -> dict[str, Any]:
    result = _new_error_bucket()
    result.update({
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "predicted_slacks": array("q"),
        "actual_on_time": bytearray(),
        "calibration_total": [0] * len(CALIBRATION_LABELS),
        "calibration_on_time": [0] * len(CALIBRATION_LABELS),
    })
    return result


def _add_error(
    bucket: dict[str, Any],
    signed_error_ns: int,
    warmup: bool,
    fallback: bool,
    clipped: bool,
) -> None:
    bucket["signed_errors"].append(signed_error_ns)
    bucket["warmup"] += int(warmup)
    bucket["fallback"] += int(fallback)
    bucket["mature_exact"] += int(not warmup and not fallback)
    bucket["clipped"] += int(clipped)
    bucket["warmup_flags"].append(warmup)
    bucket["fallback_flags"].append(fallback)
    bucket["clipped_flags"].append(clipped)


def _add_target(
    bucket: dict[str, Any],
    signed_error_ns: int,
    predicted_slack_ns: int,
    actual_on_time: bool,
    warmup: bool,
    fallback: bool,
    clipped: bool,
) -> None:
    _add_error(bucket, signed_error_ns, warmup, fallback, clipped)
    predicted_on_time = predicted_slack_ns > 0
    if predicted_on_time and actual_on_time:
        bucket["tp"] += 1
    elif not predicted_on_time and not actual_on_time:
        bucket["tn"] += 1
    elif predicted_on_time:
        bucket["fp"] += 1
    else:
        bucket["fn"] += 1
    bucket["predicted_slacks"].append(predicted_slack_ns)
    bucket["actual_on_time"].append(actual_on_time)
    index = calibration_bucket(predicted_slack_ns)
    bucket["calibration_total"][index] += 1
    bucket["calibration_on_time"][index] += int(actual_on_time)


def _summarize_error(bucket: dict[str, Any], eligible: int) -> dict[str, Any]:
    result = _error_stats(bucket["signed_errors"])
    count = int(result["count"])
    result.update({
        "eligible": eligible,
        "coverage": _ratio(count, eligible),
        "unavailable_count": max(0, eligible - count),
        "warmup_count": int(bucket["warmup"]),
        "warmup_rate": _ratio(int(bucket["warmup"]), count),
        "fallback_count": int(bucket["fallback"]),
        "fallback_rate": _ratio(int(bucket["fallback"]), count),
        "mature_exact_count": int(bucket["mature_exact"]),
        "clipped_count": int(bucket["clipped"]),
    })
    return result


def _threshold_metrics(
    slacks: list[int], actual_on_time: list[bool], threshold_ns: int
) -> dict[str, Any]:
    selected = [
        index for index, slack in enumerate(slacks)
        if slack <= threshold_ns
    ]
    actual_late_total = sum(int(not value) for value in actual_on_time)
    true_late = sum(int(not actual_on_time[index]) for index in selected)
    false_reject = len(selected) - true_late
    precision = _ratio(true_late, len(selected))
    recall = _ratio(true_late, actual_late_total)
    return {
        "threshold_ns": threshold_ns,
        "predicted_late_count": len(selected),
        "actual_late_and_predicted_late": true_late,
        "false_reject_candidate_count": false_reject,
        "predicted_late_precision": precision,
        "predicted_late_recall": recall,
        "predicted_late_f1": (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        ),
        "threshold_coverage": _ratio(len(selected), len(slacks)),
    }


def _summarize_target(bucket: dict[str, Any], eligible: int) -> dict[str, Any]:
    result = _summarize_error(bucket, eligible)
    tp = int(bucket["tp"])
    tn = int(bucket["tn"])
    fp = int(bucket["fp"])
    fn = int(bucket["fn"])
    on_time_recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    predicted_late_precision = _ratio(tn, tn + fn)
    predicted_late_recall = _ratio(tn, tn + fp)
    result.update({
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "on_time_precision": _ratio(tp, tp + fp),
        "on_time_recall": on_time_recall,
        "specificity": specificity,
        "balanced_accuracy": (
            (on_time_recall + specificity) / 2
            if on_time_recall is not None and specificity is not None
            else None
        ),
        "predicted_late_precision": predicted_late_precision,
        "predicted_late_recall": predicted_late_recall,
        "predicted_late_f1": (
            2 * predicted_late_precision * predicted_late_recall
            / (predicted_late_precision + predicted_late_recall)
            if predicted_late_precision is not None
            and predicted_late_recall is not None
            and predicted_late_precision + predicted_late_recall
            else None
        ),
        "false_reject_candidate_count": fn,
        "false_reject_candidate_rate": _ratio(fn, int(result["count"])),
        "late_prevalence": _ratio(tn + fp, int(result["count"])),
        "threshold_coverage": _ratio(tn + fn, eligible),
    })

    rates: list[float | None] = []
    calibration: dict[str, Any] = {}
    for index, label in enumerate(CALIBRATION_LABELS):
        total = int(bucket["calibration_total"][index])
        on_time = int(bucket["calibration_on_time"][index])
        rate = _ratio(on_time, total)
        rates.append(rate)
        calibration[label] = {
            "count": total,
            "actual_on_time_count": on_time,
            "actual_late_count": total - on_time,
            "actual_on_time_rate": rate,
            "predicted_late_precision": (
                _ratio(total - on_time, total) if index <= 4 else None
            ),
        }
    nonempty_rates = [rate for rate in rates if rate is not None]
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
    threshold_rows = [
        _threshold_metrics(
            bucket["predicted_slacks"], bucket["actual_on_time"], threshold
        )
        for threshold in NEGATIVE_THRESHOLDS_NS
    ]
    precisions = [
        row["predicted_late_precision"] for row in threshold_rows
        if row["predicted_late_precision"] is not None
    ]
    result["negative_thresholds"] = threshold_rows
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


def _summarize_target_numpy(
    signed_errors: Any,
    predicted_slacks: Any,
    actual_on_time: Any,
    warmup: Any,
    fallback: Any,
    mature_exact: Any,
    clipped: Any,
    eligible: int,
) -> dict[str, Any]:
    """Vectorized equivalent of ``_summarize_target`` for large Oracle strata."""
    count = int(signed_errors.size)
    if count == 0:
        return _summarize_target(_new_target_bucket(), eligible)

    absolute = np.abs(signed_errors)

    def quantile(values: Any, q: float) -> float:
        return float(np.percentile(values, q, method="linear"))

    result = {
        "count": count,
        "mae_ns": float(np.mean(absolute)),
        "median_absolute_error_ns": quantile(absolute, 50),
        "p75_absolute_error_ns": quantile(absolute, 75),
        "p95_absolute_error_ns": quantile(absolute, 95),
        "signed_error_mean_ns": float(np.mean(signed_errors)),
        "signed_error_p25_ns": quantile(signed_errors, 25),
        "signed_error_median_ns": quantile(signed_errors, 50),
        "signed_error_p75_ns": quantile(signed_errors, 75),
        "signed_error_p95_ns": quantile(signed_errors, 95),
        "eligible": int(eligible),
        "coverage": _ratio(count, int(eligible)),
        "unavailable_count": max(0, int(eligible) - count),
        "warmup_count": int(np.count_nonzero(warmup)),
        "warmup_rate": _ratio(int(np.count_nonzero(warmup)), count),
        "fallback_count": int(np.count_nonzero(fallback)),
        "fallback_rate": _ratio(int(np.count_nonzero(fallback)), count),
        "mature_exact_count": int(np.count_nonzero(mature_exact)),
        "clipped_count": int(np.count_nonzero(clipped)),
    }

    predicted_on_time = predicted_slacks > 0
    actual = actual_on_time.astype(bool, copy=False)
    actual_late = np.logical_not(actual)
    predicted_late = np.logical_not(predicted_on_time)
    tp = int(np.count_nonzero(predicted_on_time & actual))
    tn = int(np.count_nonzero(predicted_late & actual_late))
    fp = int(np.count_nonzero(predicted_on_time & actual_late))
    fn = int(np.count_nonzero(predicted_late & actual))
    on_time_recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    late_precision = _ratio(tn, tn + fn)
    late_recall = _ratio(tn, tn + fp)
    result.update({
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "on_time_precision": _ratio(tp, tp + fp),
        "on_time_recall": on_time_recall,
        "specificity": specificity,
        "balanced_accuracy": (
            (on_time_recall + specificity) / 2
            if on_time_recall is not None and specificity is not None
            else None
        ),
        "predicted_late_precision": late_precision,
        "predicted_late_recall": late_recall,
        "predicted_late_f1": (
            2 * late_precision * late_recall / (late_precision + late_recall)
            if late_precision is not None and late_recall is not None
            and late_precision + late_recall
            else None
        ),
        "false_reject_candidate_count": fn,
        "false_reject_candidate_rate": _ratio(fn, count),
        "late_prevalence": _ratio(tn + fp, count),
        "threshold_coverage": _ratio(tn + fn, int(eligible)),
    })

    calibration_masks = (
        predicted_slacks < -5_000_000,
        (predicted_slacks >= -5_000_000) & (predicted_slacks < -2_000_000),
        (predicted_slacks >= -2_000_000) & (predicted_slacks < -1_000_000),
        (predicted_slacks >= -1_000_000) & (predicted_slacks < -500_000),
        (predicted_slacks >= -500_000) & (predicted_slacks <= 0),
        (predicted_slacks > 0) & (predicted_slacks <= 500_000),
        (predicted_slacks > 500_000) & (predicted_slacks <= 1_000_000),
        (predicted_slacks > 1_000_000) & (predicted_slacks <= 2_000_000),
        (predicted_slacks > 2_000_000) & (predicted_slacks <= 5_000_000),
        predicted_slacks > 5_000_000,
    )
    rates: list[float | None] = []
    calibration: dict[str, Any] = {}
    for index, (label, mask) in enumerate(zip(CALIBRATION_LABELS, calibration_masks)):
        total = int(np.count_nonzero(mask))
        on_time = int(np.count_nonzero(actual & mask))
        rate = _ratio(on_time, total)
        rates.append(rate)
        calibration[label] = {
            "count": total,
            "actual_on_time_count": on_time,
            "actual_late_count": total - on_time,
            "actual_on_time_rate": rate,
            "predicted_late_precision": (
                _ratio(total - on_time, total) if index <= 4 else None
            ),
        }
    nonempty_rates = [rate for rate in rates if rate is not None]
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

    actual_late_total = int(np.count_nonzero(actual_late))
    threshold_rows = []
    for threshold in NEGATIVE_THRESHOLDS_NS:
        selected = predicted_slacks <= threshold
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
            "predicted_late_precision": precision,
            "predicted_late_recall": recall,
            "predicted_late_f1": (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None
                and precision + recall
                else None
            ),
            "threshold_coverage": _ratio(selected_count, count),
        })
    precisions = [
        row["predicted_late_precision"] for row in threshold_rows
        if row["predicted_late_precision"] is not None
    ]
    result["negative_thresholds"] = threshold_rows
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


def _summarize_error_numpy(
    signed_errors: Any,
    warmup: Any,
    fallback: Any,
    mature_exact: Any,
    clipped: Any,
    eligible: int,
) -> dict[str, Any]:
    count = int(signed_errors.size)
    if count == 0:
        return _summarize_error(_new_error_bucket(), eligible)
    absolute = np.abs(signed_errors)

    def quantile(values: Any, q: float) -> float:
        return float(np.percentile(values, q, method="linear"))

    return {
        "count": count,
        "mae_ns": float(np.mean(absolute)),
        "median_absolute_error_ns": quantile(absolute, 50),
        "p75_absolute_error_ns": quantile(absolute, 75),
        "p95_absolute_error_ns": quantile(absolute, 95),
        "signed_error_mean_ns": float(np.mean(signed_errors)),
        "signed_error_p25_ns": quantile(signed_errors, 25),
        "signed_error_median_ns": quantile(signed_errors, 50),
        "signed_error_p75_ns": quantile(signed_errors, 75),
        "signed_error_p95_ns": quantile(signed_errors, 95),
        "eligible": int(eligible),
        "coverage": _ratio(count, int(eligible)),
        "unavailable_count": max(0, int(eligible) - count),
        "warmup_count": int(np.count_nonzero(warmup)),
        "warmup_rate": _ratio(int(np.count_nonzero(warmup)), count),
        "fallback_count": int(np.count_nonzero(fallback)),
        "fallback_rate": _ratio(int(np.count_nonzero(fallback)), count),
        "mature_exact_count": int(np.count_nonzero(mature_exact)),
        "clipped_count": int(np.count_nonzero(clipped)),
    }


def _summarize_bucket_subset(
    bucket: dict[str, Any], eligible: int, subset: str
) -> dict[str, Any]:
    count = len(bucket["signed_errors"])
    if np is not None:
        signed_errors = np.frombuffer(bucket["signed_errors"], dtype=np.int64)
        warmup = np.frombuffer(bucket["warmup_flags"], dtype=np.uint8).astype(bool)
        fallback = np.frombuffer(
            bucket["fallback_flags"], dtype=np.uint8
        ).astype(bool)
        clipped = np.frombuffer(bucket["clipped_flags"], dtype=np.uint8).astype(bool)
        mature_exact = np.logical_not(warmup | fallback)
        mask = fallback if subset == "fallback" else mature_exact
        if "predicted_slacks" in bucket:
            predicted_slacks = np.frombuffer(
                bucket["predicted_slacks"], dtype=np.int64
            )
            actual_on_time = np.frombuffer(
                bucket["actual_on_time"], dtype=np.uint8
            )
            return _summarize_target_numpy(
                signed_errors[mask], predicted_slacks[mask], actual_on_time[mask],
                warmup[mask], fallback[mask], mature_exact[mask], clipped[mask],
                eligible,
            )
        return _summarize_error_numpy(
            signed_errors[mask], warmup[mask], fallback[mask],
            mature_exact[mask], clipped[mask], eligible,
        )

    target = "predicted_slacks" in bucket
    selected = _new_target_bucket() if target else _new_error_bucket()
    for index in range(count):
        warmup = bool(bucket["warmup_flags"][index])
        fallback = bool(bucket["fallback_flags"][index])
        include = fallback if subset == "fallback" else not warmup and not fallback
        if not include:
            continue
        if target:
            _add_target(
                selected,
                bucket["signed_errors"][index],
                bucket["predicted_slacks"][index],
                bool(bucket["actual_on_time"][index]),
                warmup,
                fallback,
                bool(bucket["clipped_flags"][index]),
            )
        else:
            _add_error(
                selected,
                bucket["signed_errors"][index],
                warmup,
                fallback,
                bool(bucket["clipped_flags"][index]),
            )
    return (
        _summarize_target(selected, eligible)
        if target else _summarize_error(selected, eligible)
    )


DIMENSIONS = (
    "by_phase",
    "by_stage",
    "by_active_workers",
    "by_phase_workers",
    "by_stage_workers",
)


def _new_container(bucket_factory: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    return {
        "bucket_factory": bucket_factory,
        "eligible": 0,
        "unavailable": 0,
        "operational": bucket_factory(),
        "mature": bucket_factory(),
        "fallback_only": bucket_factory(),
        "dimensions": {
            dimension: defaultdict(bucket_factory) for dimension in DIMENSIONS
        },
        "eligible_dimensions": {
            dimension: defaultdict(int) for dimension in DIMENSIONS
        },
    }


def _dimension_keys(phase: str, stage: str, layer: int, workers: int) -> dict[str, str]:
    return {
        "by_phase": phase,
        "by_stage": stage,
        "by_active_workers": str(workers),
        "by_phase_workers": f"{phase}|workers={workers}",
        "by_stage_workers": f"{stage}|workers={workers}",
    }


def _record_eligible(container: dict[str, Any], dimensions: dict[str, str]) -> None:
    container["eligible"] += 1
    for dimension, key in dimensions.items():
        container["eligible_dimensions"][dimension][key] += 1


def _add_container_sample(
    container: dict[str, Any],
    dimensions: dict[str, str],
    add: Callable[..., None],
    args: tuple[Any, ...],
    mature: bool,
    fallback: bool,
) -> None:
    add(container["operational"], *args)
    for dimension, key in dimensions.items():
        add(container["dimensions"][dimension][key], *args)
    if fallback:
        add(container["fallback_only"], *args)
    if mature:
        add(container["mature"], *args)


def _summarize_container(
    container: dict[str, Any], summarize: Callable[[dict[str, Any], int], dict[str, Any]]
) -> dict[str, Any]:
    result = {
        "operational": summarize(container["operational"], container["eligible"]),
        "mature_exact": summarize(container["mature"], container["eligible"]),
        "fallback_only": summarize(container["fallback_only"], container["eligible"]),
        "eligible": container["eligible"],
        "unavailable": container["unavailable"],
    }
    for dimension in DIMENSIONS:
        result[dimension] = {
            key: summarize(
                container["dimensions"][dimension][key],
                container["eligible_dimensions"][dimension][key],
            )
            for key in sorted(container["eligible_dimensions"][dimension])
        }
        result[f"mature_{dimension}"] = {
            key: _summarize_bucket_subset(
                container["dimensions"][dimension][key],
                container["eligible_dimensions"][dimension][key],
                "mature",
            )
            for key in sorted(container["eligible_dimensions"][dimension])
        }
        result[f"fallback_{dimension}"] = {
            key: _summarize_bucket_subset(
                container["dimensions"][dimension][key],
                container["eligible_dimensions"][dimension][key],
                "fallback",
            )
            for key in sorted(container["eligible_dimensions"][dimension])
        }
    return result


def _duration_summary(samples: list[dict[str, Any]], eligible: int) -> dict[str, Any]:
    result = _error_stats([int(sample["error_ns"]) for sample in samples])
    warmup = sum(int(sample.get("warmup") is True) for sample in samples)
    fallback = sum(int(sample.get("fallback") is True) for sample in samples)
    result.update({
        "eligible": eligible,
        "coverage": _ratio(len(samples), eligible),
        "unavailable_count": max(0, eligible - len(samples)),
        "warmup_count": warmup,
        "warmup_rate": _ratio(warmup, len(samples)),
        "fallback_count": fallback,
        "fallback_rate": _ratio(fallback, len(samples)),
        "predicted_value_ns": _value_stats([
            int(sample["predicted_ns"]) for sample in samples
        ]),
        "actual_value_ns": _value_stats([
            int(sample["actual_ns"]) for sample in samples
        ]),
    })
    return result


def _stratified_duration(
    samples: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get(field, "UNKNOWN"))].append(sample)
    return {
        key: _duration_summary(values, len(values))
        for key, values in sorted(grouped.items())
    }


def _worker_size_bucket(nbytes: int) -> str:
    for limit, label in (
        (64 * 1024, "le_64k"),
        (256 * 1024, "le_256k"),
        (1024 * 1024, "le_1m"),
        (4 * 1024 * 1024, "le_4m"),
        (16 * 1024 * 1024, "le_16m"),
    ):
        if nbytes <= limit:
            return label
    return "gt_16m"


def _duration_report(samples: list[dict[str, Any]], eligible: int) -> dict[str, Any]:
    return {
        "all_available": _duration_summary(samples, eligible),
        "mature_exact": _duration_summary([
            sample for sample in samples
            if not sample["warmup"] and not sample["fallback"]
        ], eligible),
        "by_phase": _stratified_duration(samples, "phase"),
        "by_stage": _stratified_duration(samples, "stage"),
        "by_active_workers": _stratified_duration(samples, "active_workers"),
        "by_size_bucket": _stratified_duration(samples, "size_bucket"),
    }


def _oracle_names(return_target: bool) -> list[tuple[str, tuple[bool, bool, bool, bool]]]:
    result: list[tuple[str, tuple[bool, bool, bool, bool]]] = []
    for actual_h in (False, True):
        for actual_q in (False, True):
            for actual_p in (False, True):
                prefix = "+".join((
                    "actual_first_use" if actual_h else "predicted_first_use",
                    "actual_queue" if actual_q else "predicted_queue",
                    "actual_pre_issue" if actual_p else "predicted_pre_issue",
                ))
                if return_target:
                    for actual_s in (False, True):
                        result.append((
                            prefix + "+" + (
                                "actual_syscall_service" if actual_s
                                else "predicted_syscall_service"
                            ),
                            (actual_h, actual_q, actual_p, actual_s),
                        ))
                else:
                    result.append((prefix, (actual_h, actual_q, actual_p, False)))
    return result


def _new_oracle_container() -> dict[str, Any]:
    return {
        "overall": _new_target_bucket(),
        "phase_codes": bytearray(),
        "stage_codes": bytearray(),
        "active_workers": array("I"),
    }


PHASE_CODES = {"UNKNOWN": 0, "PREFILL": 1, "DECODE": 2}
STAGE_CODES = {"UNKNOWN": 0, "EARLY": 1, "LATE": 2}
PHASE_NAMES = {value: key for key, value in PHASE_CODES.items()}
STAGE_NAMES = {value: key for key, value in STAGE_CODES.items()}


def _add_oracle_sample(
    container: dict[str, Any], args: tuple[Any, ...],
    phase: str, stage: str, workers: int,
) -> None:
    _add_target(container["overall"], *args)
    container["phase_codes"].append(PHASE_CODES.get(phase, 0))
    container["stage_codes"].append(STAGE_CODES.get(stage, 0))
    container["active_workers"].append(workers)


def _summarize_oracle_container(
    container: dict[str, Any], candidate_target: dict[str, Any]
) -> dict[str, Any]:
    """Expand one compact Oracle combination into exact stratified summaries."""
    overall = container["overall"]
    count = len(overall["signed_errors"])
    metadata = (
        container["phase_codes"], container["stage_codes"],
        container["active_workers"], overall["warmup_flags"],
        overall["fallback_flags"], overall["clipped_flags"],
    )
    if any(len(values) != count for values in metadata):
        raise ValueError("Oracle compact metadata length mismatch")

    if np is not None:
        signed_errors = np.frombuffer(overall["signed_errors"], dtype=np.int64)
        predicted_slacks = np.frombuffer(overall["predicted_slacks"], dtype=np.int64)
        actual_on_time = np.frombuffer(overall["actual_on_time"], dtype=np.uint8)
        phase_codes = np.frombuffer(container["phase_codes"], dtype=np.uint8)
        stage_codes = np.frombuffer(container["stage_codes"], dtype=np.uint8)
        active_workers = np.frombuffer(container["active_workers"], dtype=np.uint32)
        warmup = np.frombuffer(overall["warmup_flags"], dtype=np.uint8).astype(bool)
        fallback = np.frombuffer(
            overall["fallback_flags"], dtype=np.uint8
        ).astype(bool)
        clipped = np.frombuffer(overall["clipped_flags"], dtype=np.uint8).astype(bool)
        mature_exact = np.logical_not(warmup | fallback)

        def summarize(mask: Any, eligible: int) -> dict[str, Any]:
            return _summarize_target_numpy(
                signed_errors[mask], predicted_slacks[mask], actual_on_time[mask],
                warmup[mask], fallback[mask], mature_exact[mask], clipped[mask],
                eligible,
            )

        result = {
            "overall": summarize(slice(None), candidate_target["eligible"]),
            "by_phase": {},
            "by_stage": {},
            "by_active_workers": {},
        }
        for code in np.unique(phase_codes):
            key = PHASE_NAMES[int(code)]
            result["by_phase"][key] = summarize(
                phase_codes == code,
                candidate_target["eligible_dimensions"]["by_phase"][key],
            )
        for code in np.unique(stage_codes):
            key = STAGE_NAMES[int(code)]
            result["by_stage"][key] = summarize(
                stage_codes == code,
                candidate_target["eligible_dimensions"]["by_stage"][key],
            )
        for workers in np.unique(active_workers):
            key = str(int(workers))
            result["by_active_workers"][key] = summarize(
                active_workers == workers,
                candidate_target["eligible_dimensions"]
                ["by_active_workers"][key],
            )
        return result

    dimensions: dict[str, defaultdict[str, dict[str, Any]]] = {
        "by_phase": defaultdict(_new_target_bucket),
        "by_stage": defaultdict(_new_target_bucket),
        "by_active_workers": defaultdict(_new_target_bucket),
    }
    for index in range(count):
        args = (
            overall["signed_errors"][index],
            overall["predicted_slacks"][index],
            bool(overall["actual_on_time"][index]),
            bool(overall["warmup_flags"][index]),
            bool(overall["fallback_flags"][index]),
            bool(overall["clipped_flags"][index]),
        )
        _add_target(
            dimensions["by_phase"][PHASE_NAMES[container["phase_codes"][index]]],
            *args,
        )
        _add_target(
            dimensions["by_stage"][STAGE_NAMES[container["stage_codes"][index]]],
            *args,
        )
        _add_target(
            dimensions["by_active_workers"][str(container["active_workers"][index])],
            *args,
        )

    result = {
        "overall": _summarize_target(overall, candidate_target["eligible"]),
    }
    for dimension, buckets in dimensions.items():
        result[dimension] = {
            key: _summarize_target(
                bucket,
                candidate_target["eligible_dimensions"][dimension][key],
            )
            for key, bucket in sorted(buckets.items())
        }
    return result


def _paired_stage_comparisons(
    task_predictions: dict[tuple[str, int], dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for estimator in ("ewma", "median", "p25"):
        for conditioned in ("phase_stage", "phase_layer_stage"):
            for queue in (
                "queue_depth_worker_ewma", "queued_bytes_issue_throughput"
            ):
                for calibration in ("raw", "residual_quantile"):
                    baseline_key = f"phase_layer_{estimator}|{queue}|{calibration}"
                    conditioned_key = f"{conditioned}_{estimator}|{queue}|{calibration}"
                    first_base = _new_error_bucket()
                    first_stage = _new_error_bucket()
                    target_buckets = {
                        target: (_new_target_bucket(), _new_target_bucket())
                        for target in ("issue", "return")
                    }
                    paired = 0
                    for predictions in task_predictions.values():
                        baseline = predictions.get(baseline_key)
                        stage = predictions.get(conditioned_key)
                        if not baseline or not stage:
                            continue
                        paired += 1
                        _add_error(first_base, baseline["first_use_error"], False, False, False)
                        _add_error(first_stage, stage["first_use_error"], False, False, False)
                        for target in ("issue", "return"):
                            for sample, bucket in zip(
                                (baseline[target], stage[target]), target_buckets[target]
                            ):
                                _add_target(
                                    bucket,
                                    sample["error"], sample["slack"],
                                    sample["actual_on_time"], False, False, False,
                                )
                    first_base_summary = _summarize_error(first_base, paired)
                    first_stage_summary = _summarize_error(first_stage, paired)
                    key = (
                        f"{conditioned}_{estimator}_vs_phase_layer_{estimator}"
                        f"|{queue}|{calibration}"
                    )
                    comparisons[key] = {
                        "paired_count": paired,
                        "first_use": {
                            "baseline": first_base_summary,
                            "stage_conditioned": first_stage_summary,
                            "mae_delta_ns": (
                                first_stage_summary["mae_ns"]
                                - first_base_summary["mae_ns"]
                                if paired else None
                            ),
                        },
                    }
                    for target in ("issue", "return"):
                        base_summary = _summarize_target(target_buckets[target][0], paired)
                        stage_summary = _summarize_target(target_buckets[target][1], paired)
                        comparisons[key][target] = {
                            "baseline": base_summary,
                            "stage_conditioned": stage_summary,
                            "predicted_late_precision_delta": (
                                stage_summary["predicted_late_precision"]
                                - base_summary["predicted_late_precision"]
                                if stage_summary["predicted_late_precision"] is not None
                                and base_summary["predicted_late_precision"] is not None
                                else None
                            ),
                        }
    return comparisons


def _stage_comparisons_from_summaries(
    candidates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for estimator in ("ewma", "median", "p25"):
        for conditioned in ("phase_stage", "phase_layer_stage"):
            for queue in (
                "queue_depth_worker_ewma", "queued_bytes_issue_throughput"
            ):
                for calibration in ("raw", "residual_quantile"):
                    baseline_key = f"phase_layer_{estimator}|{queue}|{calibration}"
                    conditioned_key = f"{conditioned}_{estimator}|{queue}|{calibration}"
                    baseline = candidates.get(baseline_key)
                    stage = candidates.get(conditioned_key)
                    if not baseline or not stage:
                        continue
                    base_first = baseline["first_use"]["operational"]
                    stage_first = stage["first_use"]["operational"]
                    key = (
                        f"{conditioned}_{estimator}_vs_phase_layer_{estimator}"
                        f"|{queue}|{calibration}"
                    )
                    comparisons[key] = {
                        "common_coverage_count": min(
                            base_first["count"], stage_first["count"]
                        ),
                        "first_use": {
                            "baseline": base_first,
                            "stage_conditioned": stage_first,
                            "mae_delta_ns": (
                                stage_first["mae_ns"] - base_first["mae_ns"]
                                if base_first["mae_ns"] is not None
                                and stage_first["mae_ns"] is not None
                                else None
                            ),
                            "p95_absolute_error_delta_ns": (
                                stage_first["p95_absolute_error_ns"]
                                - base_first["p95_absolute_error_ns"]
                                if base_first["p95_absolute_error_ns"] is not None
                                and stage_first["p95_absolute_error_ns"] is not None
                                else None
                            ),
                        },
                    }
                    for target in ("issue", "return"):
                        base_target = baseline[target]["operational"]
                        stage_target = stage[target]["operational"]
                        comparisons[key][target] = {
                            "baseline": base_target,
                            "stage_conditioned": stage_target,
                            "predicted_late_precision_delta": (
                                stage_target["predicted_late_precision"]
                                - base_target["predicted_late_precision"]
                                if stage_target["predicted_late_precision"] is not None
                                and base_target["predicted_late_precision"] is not None
                                else None
                            ),
                        }
    return comparisons


def analyze_shadow_slack(memory_records: list[dict]) -> dict[str, Any]:
    # ``memory_records`` may be a re-iterable JSONL stream.  Keeping both
    # passes streaming avoids retaining multi-gigabyte Detail records.
    detail_count = 0
    summaries: list[dict] = []
    hint_results: dict[tuple[str, int], list[int]] = defaultdict(list)
    for record in memory_records:
        if record.get("event") == "EXPERT_SHADOW_SLACK":
            detail_count += 1
        elif record.get("event") == "EXPERT_SHADOW_SLACK_SUMMARY":
            summaries.append(record)
        elif record.get("event") == "OS_HINT" and isinstance(record.get("issue_id"), int):
            if isinstance(record.get("result"), int):
                hint_results[(
                    str(record.get("run_id", "unknown")), int(record["issue_id"])
                )].append(int(record["result"]))
    result: dict[str, Any] = {
        "schema_version": 2,
        "source": "detail" if detail_count else "summary" if summaries else "none",
        "detail_records": detail_count,
        "runtime_summary_events": len(summaries),
        "runtime_summary": summaries[-1] if summaries else None,
        "prediction_targets": {
            "issue": {
                "prediction": "first_use_horizon - queue_wait - pre_issue_overhead",
                "actual_label": "issue_ts < logical_first_use_ts",
            },
            "return": {
                "prediction": (
                    "first_use_horizon - queue_wait - pre_issue_overhead "
                    "- hint_syscall_service"
                ),
                "actual_label": (
                    "final_enabled_hint_return_ts < logical_first_use_ts"
                ),
            },
        },
        "semantic_violations": 0,
        "invalid_records": 0,
        "invalid_predictions": 0,
        "duplicate_task_records": 0,
        "causality_errors": 0,
        "timestamp_regressions": 0,
        "missing_time_components": defaultdict(int),
        "candidates": {},
        "oracle_attribution": {},
        "queue_models": {},
        "duration_models": {},
        "time_decomposition": {},
        "stage_comparisons": {},
    }
    if not detail_count:
        result["missing_time_components"] = {}
        return result

    candidates: dict[str, dict[str, Any]] = {}
    oracle: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: {
            "issue": defaultdict(_new_oracle_container),
            "return": defaultdict(_new_oracle_container),
        }
    )
    seen_tasks: set[tuple[str, int]] = set()
    valid_tasks: set[tuple[str, int]] = set()
    queue_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    queue_by_task: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    pre_issue_samples: list[dict[str, Any]] = []
    syscall_by_issue: dict[tuple[str, int], dict[str, Any]] = {}
    occupied_by_issue: dict[tuple[str, int], dict[str, Any]] = {}
    component_values: dict[str, list[int]] = defaultdict(list)
    issue_task_counts: defaultdict[tuple[str, int], int] = defaultdict(int)

    required_top = {
        "task_id", "prediction_ts_ns", "enqueued_ts_ns", "dequeued_ts_ns",
        "issue_ts_ns", "returned_ts_ns", "first_use_ts_ns", "phase", "stage",
        "layer", "predictions",
    }
    required_prediction = {
        "predicted_first_use_ts_ns", "predicted_first_use_horizon_ns",
        "predicted_queue_wait_ns", "predicted_pre_issue_overhead_ns",
        "predicted_hint_syscall_service_ns", "predicted_worker_occupied_ns",
        "predicted_issue_slack_ns", "predicted_return_slack_ns",
        "deadline_model", "queue_model", "calibration_model",
        "estimator_warmup", "queue_warmup", "pre_issue_warmup",
        "syscall_service_warmup", "fallback_level",
    }

    for record in memory_records:
        if record.get("event") != "EXPERT_SHADOW_SLACK":
            continue
        if (
            record.get("schema_version") != 2
            or record.get("semantics") != "logical_first_use"
            or record.get("physical_load_observed") is not False
            or record.get("issue_target") != "issue_ts < logical_first_use_ts"
            or record.get("return_target")
            != "final_enabled_hint_return_ts < logical_first_use_ts"
        ):
            result["semantic_violations"] += 1
        if not required_top.issubset(record):
            result["invalid_records"] += 1
            continue
        task_id = record.get("task_id")
        if not isinstance(task_id, int) or task_id <= 0:
            result["invalid_records"] += 1
            continue
        run_id = str(record.get("run_id", "unknown"))
        identity = (run_id, task_id)
        if identity in seen_tasks:
            result["duplicate_task_records"] += 1
            continue
        seen_tasks.add(identity)
        names = (
            "prediction_ts_ns", "enqueued_ts_ns", "dequeued_ts_ns",
            "issue_ts_ns", "returned_ts_ns", "first_use_ts_ns",
        )
        if not all(_is_number(record.get(name)) for name in names):
            result["invalid_records"] += 1
            continue
        prediction_ts, enqueue_ts, dequeue_ts, issue_ts, return_ts, first_use_ts = (
            int(record[name]) for name in names
        )
        if prediction_ts != enqueue_ts:
            result["causality_errors"] += 1
        if not enqueue_ts <= dequeue_ts <= issue_ts <= return_ts:
            result["timestamp_regressions"] += 1
            continue
        if prediction_ts > first_use_ts or record.get("causality_error") is True:
            result["causality_errors"] += 1
            continue

        actual_h = first_use_ts - prediction_ts
        actual_q = dequeue_ts - enqueue_ts
        actual_p = issue_ts - dequeue_ts
        actual_s = return_ts - issue_ts
        actual_o = return_ts - dequeue_ts
        actual_issue_slack = first_use_ts - issue_ts
        actual_return_slack = first_use_ts - return_ts
        expected_values = {
            "actual_first_use_horizon_ns": actual_h,
            "actual_queue_wait_ns": actual_q,
            "actual_pre_issue_overhead_ns": actual_p,
            "actual_hint_syscall_service_ns": actual_s,
            "actual_worker_occupied_ns": actual_o,
            "actual_issue_slack_ns": actual_issue_slack,
            "actual_return_slack_ns": actual_return_slack,
        }
        for name, expected in expected_values.items():
            if not _is_number(record.get(name)):
                result["missing_time_components"][name] += 1
            elif int(record[name]) != expected:
                result["semantic_violations"] += 1
        if record.get("issue_on_time") is not (actual_issue_slack > 0):
            result["semantic_violations"] += 1
        if record.get("return_on_time") is not (actual_return_slack > 0):
            result["semantic_violations"] += 1
        if actual_o != actual_p + actual_s:
            result["semantic_violations"] += 1

        valid_tasks.add(identity)
        phase = str(record.get("phase", "UNKNOWN"))
        stage = str(record.get("stage", "UNKNOWN"))
        layer = int(record.get("layer", -1))
        workers = int(record.get("active_workers", 0))
        dimensions = _dimension_keys(phase, stage, layer, workers)
        for name, value in (
            ("actual_first_use_horizon", actual_h),
            ("actual_queue_wait", actual_q),
            ("actual_pre_issue_overhead", actual_p),
            ("actual_hint_syscall_service", actual_s),
            ("actual_worker_occupied", actual_o),
            ("actual_issue_slack", actual_issue_slack),
            ("actual_return_slack", actual_return_slack),
        ):
            component_values[name].append(value)

        issue_id = record.get("issue_id")
        issue_identity = (
            (run_id, int(issue_id))
            if isinstance(issue_id, int) and issue_id > 0
            else (run_id, -task_id)
        )
        issue_task_counts[issue_identity] += 1
        seen_queue_models: set[str] = set()
        component_prediction_recorded = False

        for prediction in record.get("predictions", []):
            if not isinstance(prediction, dict) or not required_prediction.issubset(prediction):
                result["invalid_predictions"] += 1
                continue
            values = (
                prediction.get("predicted_first_use_ts_ns"),
                prediction.get("predicted_first_use_horizon_ns"),
                prediction.get("predicted_queue_wait_ns"),
                prediction.get("predicted_pre_issue_overhead_ns"),
                prediction.get("predicted_hint_syscall_service_ns"),
                prediction.get("predicted_worker_occupied_ns"),
                prediction.get("predicted_issue_slack_ns"),
                prediction.get("predicted_return_slack_ns"),
            )
            if not all(_is_number(value) for value in values):
                result["invalid_predictions"] += 1
                continue
            (
                predicted_first_use, predicted_h, predicted_q, predicted_p,
                predicted_s, predicted_o, predicted_issue, predicted_return,
            ) = (int(value) for value in values)
            expected_issue = predicted_h - predicted_q - predicted_p
            expected_return = expected_issue - predicted_s
            if predicted_issue != expected_issue or predicted_return != expected_return:
                result["semantic_violations"] += 1
                continue
            if predicted_first_use != prediction_ts + predicted_h:
                result["semantic_violations"] += 1
                continue

            deadline_model = str(prediction["deadline_model"])
            queue_model = str(prediction["queue_model"])
            calibration_model = str(prediction["calibration_model"])
            candidate_key = f"{deadline_model}|{queue_model}|{calibration_model}"
            if candidate_key not in candidates:
                candidates[candidate_key] = {
                    "metadata": {
                        "deadline_model": deadline_model,
                        "queue_model": queue_model,
                        "calibration_model": calibration_model,
                    },
                    "first_use": _new_container(_new_error_bucket),
                    "issue": _new_container(_new_target_bucket),
                    "return": _new_container(_new_target_bucket),
                }
            candidate = candidates[candidate_key]
            for target in ("first_use", "issue", "return"):
                _record_eligible(candidate[target], dimensions)

            clipped = bool(prediction.get("clipped_low") or prediction.get("clipped_high"))
            residual_model = calibration_model == "residual_quantile"
            first_warmup = bool(prediction.get("estimator_warmup")) or (
                residual_model and bool(prediction.get("residual_warmup"))
            )
            first_fallback = str(prediction.get("fallback_level")) != "exact" or (
                residual_model
                and str(prediction.get("residual_fallback_level", "static_default")) != "exact"
            )
            queue_warmup = bool(prediction.get("queue_warmup"))
            pre_warmup = bool(prediction.get("pre_issue_warmup"))
            syscall_warmup = bool(prediction.get("syscall_service_warmup"))
            queue_fallback = str(prediction.get("queue_fallback_level", "static_default")) != "exact"
            pre_fallback = str(prediction.get("pre_issue_fallback_level", "static_default")) != "exact"
            syscall_fallback = str(
                prediction.get("syscall_service_fallback_level", "static_default")
            ) != "exact"

            first_error = predicted_first_use - first_use_ts
            _add_container_sample(
                candidate["first_use"], dimensions, _add_error,
                (first_error, first_warmup, first_fallback, clipped),
                not first_warmup and not first_fallback, first_fallback,
            )
            target_samples: dict[str, dict[str, Any]] = {}
            for target, predicted_slack, actual_slack, extra_warmup, extra_fallback in (
                ("issue", predicted_issue, actual_issue_slack,
                 queue_warmup or pre_warmup, queue_fallback or pre_fallback),
                ("return", predicted_return, actual_return_slack,
                 queue_warmup or pre_warmup or syscall_warmup,
                 queue_fallback or pre_fallback or syscall_fallback),
            ):
                available = prediction.get(f"{target}_prediction_available",
                                           prediction.get("prediction_available")) is not False
                if not available:
                    candidate[target]["unavailable"] += 1
                    continue
                warmup = first_warmup or extra_warmup
                fallback = first_fallback or extra_fallback
                actual_on_time = actual_slack > 0
                error = predicted_slack - actual_slack
                args = (
                    error, predicted_slack, actual_on_time,
                    warmup, fallback, clipped,
                )
                _add_container_sample(
                    candidate[target], dimensions, _add_target, args,
                    not warmup and not fallback, fallback,
                )
                target_samples[target] = {
                    "error": error,
                    "slack": predicted_slack,
                    "actual_on_time": actual_on_time,
                }

            oracle_targets = (
                (
                    ("issue", predicted_issue, actual_issue_slack, False),
                    ("return", predicted_return, actual_return_slack, True),
                )
                if calibration_model == "raw"
                and queue_model == "queue_depth_worker_ewma"
                else ()
            )
            for target, predicted_target, actual_target, return_target in oracle_targets:
                available = prediction.get(f"{target}_prediction_available",
                                           prediction.get("prediction_available")) is not False
                if not available:
                    continue
                for name, flags in _oracle_names(return_target):
                    use_h, use_q, use_p, use_s = flags
                    slack = (actual_h if use_h else predicted_h)
                    slack -= actual_q if use_q else predicted_q
                    slack -= actual_p if use_p else predicted_p
                    if return_target:
                        slack -= actual_s if use_s else predicted_s
                    oracle_warmup = (
                        (not use_h and first_warmup)
                        or (not use_q and queue_warmup)
                        or (not use_p and pre_warmup)
                        or (return_target and not use_s and syscall_warmup)
                    )
                    oracle_fallback = (
                        (not use_h and first_fallback)
                        or (not use_q and queue_fallback)
                        or (not use_p and pre_fallback)
                        or (return_target and not use_s and syscall_fallback)
                    )
                    oracle_value = oracle[candidate_key][target][name]
                    oracle_args = (
                        slack - actual_target, slack, actual_target > 0,
                        oracle_warmup, oracle_fallback,
                        clipped and not use_h,
                    )
                    _add_oracle_sample(
                        oracle_value, oracle_args, phase, stage, workers,
                    )

            if queue_model not in seen_queue_models:
                queue_sample = {
                    "error_ns": predicted_q - actual_q,
                    "predicted_ns": predicted_q,
                    "actual_ns": actual_q,
                    "warmup": queue_warmup,
                    "fallback": queue_fallback,
                    "phase": phase,
                    "stage": stage,
                    "active_workers": workers,
                    "size_bucket": _worker_size_bucket(int(record.get("nbytes", 0))),
                }
                queue_all[queue_model].append(queue_sample)
                queue_by_task[identity][queue_model] = queue_sample
                seen_queue_models.add(queue_model)

            if not component_prediction_recorded:
                size_bucket = _worker_size_bucket(int(record.get("nbytes", 0)))
                base = {
                    "phase": phase,
                    "stage": stage,
                    "active_workers": workers,
                    "size_bucket": size_bucket,
                }
                pre_issue_samples.append({
                    **base,
                    "error_ns": predicted_p - actual_p,
                    "predicted_ns": predicted_p,
                    "actual_ns": actual_p,
                    "warmup": pre_warmup,
                    "fallback": pre_fallback,
                })
                if issue_identity not in syscall_by_issue:
                    issued_size = int(record.get("issued_nbytes", record.get("nbytes", 0)))
                    issue_base = {
                        **base,
                        "size_bucket": _worker_size_bucket(issued_size),
                        "coalesced": str(record.get("coalesced") is True).lower(),
                    }
                    syscall_by_issue[issue_identity] = {
                        **issue_base,
                        "error_ns": predicted_s - actual_s,
                        "predicted_ns": predicted_s,
                        "actual_ns": actual_s,
                        "warmup": syscall_warmup,
                        "fallback": syscall_fallback,
                    }
                    occupied_by_issue[issue_identity] = {
                        **issue_base,
                        "error_ns": predicted_o - actual_o,
                        "predicted_ns": predicted_o,
                        "actual_ns": actual_o,
                        "warmup": bool(prediction.get("worker_warmup")),
                        "fallback": str(
                            prediction.get("worker_fallback_level", "static_default")
                        ) != "exact",
                    }
                component_prediction_recorded = True

    result["valid_unique_tasks"] = len(valid_tasks)
    result["candidate_count"] = len(candidates)
    result["candidates"] = {
        key: {
            "metadata": candidate["metadata"],
            "first_use": _summarize_container(candidate["first_use"], _summarize_error),
            "issue": _summarize_container(candidate["issue"], _summarize_target),
            "return": _summarize_container(candidate["return"], _summarize_target),
        }
        for key, candidate in sorted(candidates.items())
    }
    result["oracle_attribution"] = {
        key: {
            target: {
                name: _summarize_oracle_container(
                    container, candidates[key][target]
                )
                for name, container in sorted(oracle[key][target].items())
            }
            for target in ("issue", "return")
        }
        for key in sorted(oracle)
    }

    queue_models = sorted(queue_all)
    paired_identities = {
        identity for identity, models in queue_by_task.items()
        if queue_models and all(model in models for model in queue_models)
    }
    result["queue_models"] = {
        model: {
            **_duration_report(queue_all[model], len(valid_tasks)),
            "paired_common": _duration_summary([
                queue_by_task[identity][model] for identity in sorted(paired_identities)
            ], len(paired_identities)),
        }
        for model in queue_models
    }
    result["queue_paired_common_count"] = len(paired_identities)
    syscall_samples = list(syscall_by_issue.values())
    occupied_samples = list(occupied_by_issue.values())
    result["duration_models"] = {
        "pre_issue_overhead": _duration_report(pre_issue_samples, len(valid_tasks)),
        "hint_syscall_service": {
            **_duration_report(syscall_samples, len(syscall_samples)),
            "unique_issue_groups": len(syscall_samples),
            "by_coalesced": _stratified_duration(syscall_samples, "coalesced"),
        },
        "worker_occupied": {
            **_duration_report(occupied_samples, len(occupied_samples)),
            "unique_issue_groups": len(occupied_samples),
            "by_coalesced": _stratified_duration(occupied_samples, "coalesced"),
        },
    }
    result["time_decomposition"] = {
        name: _value_stats(values) for name, values in sorted(component_values.items())
    }
    result["multi_syscall_audit"] = {
        "issue_groups_with_trace_results": len(hint_results),
        "issue_groups_with_multiple_syscalls": sum(
            int(len(values) > 1) for values in hint_results.values()
        ),
        "syscall_failures": sum(
            int(value != 0) for values in hint_results.values() for value in values
        ),
        "coalesced_issue_groups": sum(
            int(count > 1) for count in issue_task_counts.values()
        ),
    }
    result["stage_comparisons"] = _stage_comparisons_from_summaries(
        result["candidates"]
    )
    result["missing_time_components"] = dict(result["missing_time_components"])
    return result


__all__ = [
    "CALIBRATION_LABELS",
    "analyze_shadow_slack",
    "calibration_bucket",
    "percentile",
]
