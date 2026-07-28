#!/usr/bin/env python3
"""Offline M5A Pressure Shadow labeling and interpretable state analysis.

The runtime only records observations. This module assigns strictly-future
Outcome labels and LOW/MEDIUM/HIGH candidates with run-separated thresholds.
It never writes decisions back to the runtime trace.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


WINDOWS_NS = {
    "50ms": 50_000_000,
    "100ms": 100_000_000,
    "250ms": 250_000_000,
}
COUNTER_OUTCOMES = {
    "process_major_fault_burst": "process_major_faults",
    "cgroup_major_fault_burst": "cgroup_pgmajfault",
    "swap_growth": "swap_current_bytes",
    "psi_some_stall": "psi_some_total_us",
    "psi_full_stall": "psi_full_total_us",
    "refault_anon_burst": "workingset_refault_anon",
    "refault_file_burst": "workingset_refault_file",
    "memory_events_high_growth": "memory_events_high",
}
GAUGE_OUTCOMES = {
    "queue_depth_growth": "queue_depth",
    "queue_bytes_growth": "queued_bytes",
}
FIXED_MINIMUMS = {
    "process_major_fault_burst": 1.0,
    "cgroup_major_fault_burst": 1.0,
    "swap_growth": 1.0,
    "psi_some_stall": 1.0,
    "psi_full_stall": 1.0,
    "refault_anon_burst": 1.0,
    "refault_file_burst": 1.0,
    "refault_sum_burst": 1.0,
    "memory_events_high_growth": 1.0,
    "queue_depth_growth": 1.0,
    "queue_bytes_growth": 1.0,
}
TRAINING_QUANTILE_OUTCOMES = {
    "process_major_fault_burst",
    "cgroup_major_fault_burst",
    "psi_some_stall",
    "psi_full_stall",
    "refault_anon_burst",
    "refault_file_burst",
    "refault_sum_burst",
    "queue_depth_growth",
    "queue_bytes_growth",
}
PRIMARY_OUTCOMES = (
    "process_major_fault_burst",
    "cgroup_major_fault_burst",
    "swap_growth",
    "psi_some_stall",
    "psi_full_stall",
    "refault_sum_burst",
    "memory_events_high_growth",
    "decode_long_tail",
    "queue_depth_growth",
    "queue_bytes_growth",
)


def percentile(values: Sequence[float], q: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * q / 100.0
    lower = int(math.floor(position))
    upper = min(lower + 1, len(finite) - 1)
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "min": min(finite) if finite else None,
        "p25": percentile(finite, 25),
        "median": percentile(finite, 50),
        "mean": statistics.mean(finite) if finite else None,
        "p95": percentile(finite, 95),
        "max": max(finite) if finite else None,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if isinstance(value, dict):
                records.append(value)
    return records


def observation(record: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = record.get(field_name)
    if not isinstance(value, dict):
        return {
            "value": None,
            "status": "field_missing",
            "read_ts_ns": None,
            "error": "observation object missing",
        }
    status = value.get("status")
    raw = value.get("value")
    timestamp = value.get("read_ts_ns")
    if status != "available":
        raw = None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raw = None
        if status == "available":
            status = "parse_error"
    return {
        "value": raw,
        "status": status or "field_missing",
        "read_ts_ns": timestamp if isinstance(timestamp, int) else None,
        "error": value.get("error"),
    }


def phase_value(sample: dict[str, Any]) -> str:
    field = sample.get("phase")
    if isinstance(field, dict) and field.get("status") == "available":
        return str(field.get("value", "UNKNOWN"))
    return "UNKNOWN"


@dataclass
class DecodeStep:
    step: int
    begin_ts_ns: int
    end_ts_ns: int
    latency_ns: int


@dataclass
class RunData:
    run_id: str
    run_dir: Path
    samples: list[dict[str, Any]]
    decode_steps: list[DecodeStep]
    task_opportunities: list[int]
    workers: int | None
    memory_max: str | None
    memory_swap_max: str | None
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def stratum(self) -> str:
        return f"workers={self.workers}|memory_max={self.memory_max}"


def load_run(run_dir: Path) -> RunData:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    records = read_jsonl(run_dir / "memory_trace.jsonl")
    samples = [
        record for record in records
        if record.get("event") == "PRESSURE_SHADOW_SAMPLE"
    ]
    run_id = str(
        (samples[0].get("run_id") if samples else None)
        or manifest.get("run_name")
        or run_dir.name
    )
    for sample in samples:
        sample.setdefault("run_id", run_id)
    samples.sort(key=lambda item: int(item.get("sample_ready_ts_ns", 0)))

    begins: dict[tuple[str, int], dict[str, Any]] = {}
    decode_steps: list[DecodeStep] = []
    opportunities: dict[tuple[str, int], int] = {}
    for record in records:
        event = record.get("event")
        phase = str(record.get("phase", "UNKNOWN"))
        step = record.get("step")
        if event == "STEP_BEGIN" and isinstance(step, int):
            begins[(phase, step)] = record
        elif event == "STEP_END" and phase == "DECODE" and isinstance(step, int):
            begin = begins.get((phase, step))
            begin_ts = begin.get("ts_ns") if begin else None
            end_ts = record.get("ts_ns")
            latency = record.get("latency_ns")
            if all(isinstance(value, int) for value in (begin_ts, end_ts, latency)):
                decode_steps.append(DecodeStep(step, begin_ts, end_ts, latency))
        elif event == "EXPERT_TASK" and record.get("lifecycle_event") in {
            "CREATE", "ADMIT", "ENQUEUE"
        }:
            timestamp = record.get("ts_ns")
            if isinstance(timestamp, int):
                task_id = record.get("task_id")
                identity = (
                    ("task", task_id)
                    if isinstance(task_id, int)
                    else (str(record.get("lifecycle_event")), timestamp)
                )
                previous = opportunities.get(identity)
                if previous is None or timestamp < previous:
                    opportunities[identity] = timestamp
    decode_steps.sort(key=lambda item: item.begin_ts_ns)

    environment = manifest.get("environment", {})
    workers_raw = environment.get("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS")
    try:
        workers = int(workers_raw) if workers_raw is not None else None
    except (TypeError, ValueError):
        workers = None
    experiment = manifest.get("experiment", {})
    cgroup = experiment.get("cgroup", {})
    memory_max = experiment.get("requested_memory_max") or cgroup.get("memory.max")
    memory_swap_max = experiment.get("requested_memory_swap_max") or cgroup.get("memory.swap.max")
    return RunData(
        run_id=run_id,
        run_dir=run_dir,
        samples=samples,
        decode_steps=decode_steps,
        task_opportunities=sorted(opportunities.values()),
        workers=workers,
        memory_max=memory_max,
        memory_swap_max=memory_swap_max,
        manifest=manifest,
    )


def validate_sample_contract(run: RunData) -> None:
    manifest_run_id = run.manifest.get("run_name")
    if not isinstance(manifest_run_id, str) or not manifest_run_id:
        raise ValueError(f"{run.run_dir}: manifest run_name missing")
    if run.run_id != manifest_run_id or run.run_id == "missing_run_id":
        raise ValueError(
            f"{run.run_dir}: trace run_id {run.run_id!r} does not match "
            f"manifest run_name {manifest_run_id!r}"
        )
    if not run.samples:
        raise ValueError(f"{run.run_dir}: no PRESSURE_SHADOW_SAMPLE events")

    previous_seq: int | None = None
    previous_ready: int | None = None
    cgroup_paths: set[str] = set()
    for index, sample_value in enumerate(run.samples):
        prefix = f"{run.run_dir}:sample[{index}]"
        if sample_value.get("run_id") != run.run_id:
            raise ValueError(f"{prefix}: run_id mismatch")
        sequence = sample_value.get("sample_seq")
        ready = sample_value.get("sample_ready_ts_ns")
        start = sample_value.get("sample_start_ts_ns")
        if not all(isinstance(value, int) for value in (sequence, ready, start)):
            raise ValueError(f"{prefix}: sample identity/time fields missing")
        if sample_value.get("ts_ns") != ready or start > ready:
            raise ValueError(f"{prefix}: non-causal sample timestamps")
        if previous_seq is not None and sequence != previous_seq + 1:
            raise ValueError(f"{prefix}: sample_seq is not contiguous")
        if previous_ready is not None and ready <= previous_ready:
            raise ValueError(f"{prefix}: sample_ready_ts_ns is not increasing")
        previous_seq = sequence
        previous_ready = ready

        sources = sample_value.get("sources")
        if not isinstance(sources, dict):
            raise ValueError(f"{prefix}: sources object missing")
        cgroup_source = sources.get("cgroup_memory")
        if not isinstance(cgroup_source, dict):
            raise ValueError(f"{prefix}: cgroup_memory source missing")
        if cgroup_source.get("scope") != "current_process_cgroup":
            raise ValueError(f"{prefix}: invalid cgroup source scope")
        cgroup_path = cgroup_source.get("path")
        if isinstance(cgroup_path, str):
            cgroup_paths.add(cgroup_path)

        for field_name, field_value in sample_value.items():
            if not isinstance(field_value, dict) or "status" not in field_value:
                continue
            if field_value.get("run_id") != run.run_id:
                raise ValueError(f"{prefix}:{field_name}: field run_id mismatch")
            status = field_value.get("status")
            if status != "available" and field_value.get("value") is not None:
                raise ValueError(
                    f"{prefix}:{field_name}: unavailable field carries a value"
                )
            read_ts = field_value.get("read_ts_ns")
            if isinstance(read_ts, int) and read_ts > ready:
                raise ValueError(
                    f"{prefix}:{field_name}: read timestamp is after sample ready"
                )
    if len(cgroup_paths) != 1:
        raise ValueError(
            f"{run.run_dir}: cgroup source path must be one stable non-null path"
        )


def next_decode_step(steps: Sequence[DecodeStep], ready_ts_ns: int) -> DecodeStep | None:
    begin_times = [step.begin_ts_ns for step in steps]
    index = bisect.bisect_right(begin_times, ready_ts_ns)
    return steps[index] if index < len(steps) else None


def counter_future(
    samples: Sequence[dict[str, Any]],
    sample_index: int,
    field_name: str,
    window_ns: int,
) -> dict[str, Any]:
    state_ts = samples[sample_index].get("sample_ready_ts_ns")
    if not isinstance(state_ts, int):
        return {"eligible": False, "reason": "sample_ready_ts_missing"}
    deadline = state_ts + window_ns
    successful: list[tuple[int, float]] = []
    for future in samples[sample_index + 1:]:
        field = observation(future, field_name)
        read_ts = field["read_ts_ns"]
        if isinstance(read_ts, int) and read_ts > deadline:
            break
        if (
            field["status"] == "available"
            and isinstance(read_ts, int)
            and read_ts > state_ts
            and isinstance(field["value"], (int, float))
        ):
            successful.append((read_ts, float(field["value"])))
    if not successful:
        return {"eligible": False, "reason": "no_successful_baseline_after_state"}
    baseline_ts, baseline = successful[0]
    later = successful[1:]
    if not later:
        return {
            "eligible": False,
            "reason": "no_successful_read_after_baseline_in_window",
            "baseline_ts_ns": baseline_ts,
            "baseline_value": baseline,
        }
    maximum_delta = 0.0
    onset_candidates: list[tuple[int, float]] = []
    for timestamp, value in later:
        if value < baseline:
            return {
                "eligible": False,
                "reason": "counter_reset",
                "baseline_ts_ns": baseline_ts,
                "reset_ts_ns": timestamp,
            }
        delta = value - baseline
        maximum_delta = max(maximum_delta, delta)
        onset_candidates.append((timestamp, delta))
    return {
        "eligible": True,
        "reason": None,
        "baseline_ts_ns": baseline_ts,
        "baseline_value": baseline,
        "magnitude": maximum_delta,
        "timeline": onset_candidates,
    }


def refault_sum_future(
    samples: Sequence[dict[str, Any]],
    sample_index: int,
    window_ns: int,
) -> dict[str, Any]:
    synthetic: list[dict[str, Any]] = []
    for sample in samples:
        anon = observation(sample, "workingset_refault_anon")
        file = observation(sample, "workingset_refault_file")
        combined = {
            "status": "unavailable",
            "value": None,
            "read_ts_ns": None,
            "error": "anon_or_file_unavailable",
        }
        if (
            anon["status"] == "available"
            and file["status"] == "available"
            and anon["read_ts_ns"] == file["read_ts_ns"]
        ):
            combined = {
                "status": "available",
                "value": anon["value"] + file["value"],
                "read_ts_ns": anon["read_ts_ns"],
                "error": None,
            }
        copy = dict(sample)
        copy["_refault_sum"] = combined
        synthetic.append(copy)
    return counter_future(synthetic, sample_index, "_refault_sum", window_ns)


def gauge_future(
    samples: Sequence[dict[str, Any]],
    sample_index: int,
    field_name: str,
    window_ns: int,
) -> dict[str, Any]:
    current = observation(samples[sample_index], field_name)
    state_ts = samples[sample_index].get("sample_ready_ts_ns")
    if current["status"] != "available" or not isinstance(state_ts, int):
        return {"eligible": False, "reason": "current_gauge_unavailable"}
    future_values: list[tuple[int, float]] = []
    deadline = state_ts + window_ns
    for future in samples[sample_index + 1:]:
        item = observation(future, field_name)
        read_ts = item["read_ts_ns"]
        if isinstance(read_ts, int) and read_ts > deadline:
            break
        if (
            item["status"] == "available"
            and isinstance(read_ts, int)
            and read_ts > state_ts
        ):
            future_values.append((read_ts, float(item["value"])))
    if not future_values:
        return {"eligible": False, "reason": "no_future_gauge_snapshot"}
    timeline = [
        (timestamp, max(0.0, value - float(current["value"])))
        for timestamp, value in future_values
    ]
    return {
        "eligible": True,
        "reason": None,
        "baseline_ts_ns": current["read_ts_ns"],
        "baseline_value": current["value"],
        "magnitude": max(delta for _, delta in timeline),
        "timeline": timeline,
    }


def raw_outcomes(run: RunData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(run.samples):
        by_window: dict[str, dict[str, Any]] = {}
        for window_name, window_ns in WINDOWS_NS.items():
            outcomes: dict[str, Any] = {}
            for outcome_name, field_name in COUNTER_OUTCOMES.items():
                outcomes[outcome_name] = counter_future(
                    run.samples, index, field_name, window_ns
                )
            outcomes["refault_sum_burst"] = refault_sum_future(
                run.samples, index, window_ns
            )
            for outcome_name, field_name in GAUGE_OUTCOMES.items():
                outcomes[outcome_name] = gauge_future(
                    run.samples, index, field_name, window_ns
                )
            by_window[window_name] = outcomes
        ready_ts = sample.get("sample_ready_ts_ns")
        decode = (
            next_decode_step(run.decode_steps, ready_ts)
            if isinstance(ready_ts, int)
            else None
        )
        rows.append(
            {
                "run_id": run.run_id,
                "sample_index": index,
                "sample_ready_ts_ns": ready_ts,
                "phase": phase_value(sample),
                "windows": by_window,
                "next_decode": {
                    "eligible": decode is not None,
                    "reason": None if decode is not None else "no_complete_future_decode_step",
                    "step": decode.step if decode else None,
                    "begin_ts_ns": decode.begin_ts_ns if decode else None,
                    "onset_ts_ns": decode.begin_ts_ns if decode else None,
                    "latency_ns": decode.latency_ns if decode else None,
                },
            }
        )
    return rows


def grouped_folds(run_ids: Sequence[str]) -> list[dict[str, Any]]:
    unique = sorted(set(run_ids))
    if len(unique) < 2:
        return [
            {
                "fold_id": "exploratory_all",
                "training_run_ids": unique,
                "evaluation_run_ids": unique,
                "status": "exploratory_no_run_separation",
            }
        ]
    return [
        {
            "fold_id": f"leave_out_{evaluation}",
            "training_run_ids": [run_id for run_id in unique if run_id != evaluation],
            "evaluation_run_ids": [evaluation],
            "status": "run_separated",
        }
        for evaluation in unique
    ]


def training_outcome_thresholds(
    raw_by_run: dict[str, list[dict[str, Any]]],
    training_run_ids: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    thresholds: dict[str, dict[str, dict[str, Any]]] = {}
    for window_name in WINDOWS_NS:
        thresholds[window_name] = {}
        for outcome_name in (*COUNTER_OUTCOMES, "refault_sum_burst", *GAUGE_OUTCOMES):
            magnitudes: list[float] = []
            for run_id in training_run_ids:
                for row in raw_by_run.get(run_id, []):
                    outcome = row["windows"][window_name][outcome_name]
                    if outcome.get("eligible") and isinstance(outcome.get("magnitude"), (int, float)):
                        if outcome["magnitude"] > 0:
                            magnitudes.append(float(outcome["magnitude"]))
            fixed = FIXED_MINIMUMS[outcome_name]
            training_quantile = (
                percentile(magnitudes, 75)
                if outcome_name in TRAINING_QUANTILE_OUTCOMES
                else None
            )
            threshold = max(fixed, training_quantile or fixed)
            thresholds[window_name][outcome_name] = {
                "value": threshold,
                "source": (
                    "max(preregistered_absolute_minimum,training_positive_p75)"
                    if outcome_name in TRAINING_QUANTILE_OUTCOMES
                    else "preregistered_positive_growth"
                ),
                "absolute_minimum": fixed,
                "training_positive_p75": training_quantile,
                "training_positive_count": len(magnitudes),
                "training_run_ids": list(training_run_ids),
            }
    return thresholds


def decode_thresholds(
    runs: dict[str, RunData],
    training_run_ids: Sequence[str],
) -> dict[str, Any]:
    by_stratum: dict[str, list[float]] = {}
    global_latencies: list[float] = []
    for run_id in training_run_ids:
        run = runs[run_id]
        latencies = [float(step.latency_ns) for step in run.decode_steps]
        by_stratum.setdefault(run.stratum, []).extend(latencies)
        global_latencies.extend(latencies)
    global_p95 = percentile(global_latencies, 95)
    return {
        "by_stratum": {
            stratum: {
                "value_ns": percentile(values, 95),
                "sample_count": len(values),
                "source": "training_same_configuration_p95",
            }
            for stratum, values in sorted(by_stratum.items())
        },
        "global_fallback": {
            "value_ns": global_p95,
            "sample_count": len(global_latencies),
            "source": "training_global_p95_fallback",
        },
        "training_run_ids": list(training_run_ids),
    }


def past_counter_delta(
    samples: Sequence[dict[str, Any]],
    index: int,
    field_name: str,
) -> dict[str, Any]:
    current = observation(samples[index], field_name)
    if current["status"] != "available":
        return {"value": None, "status": current["status"], "reason": current["error"]}
    for prior in reversed(samples[:index]):
        previous = observation(prior, field_name)
        if previous["status"] != "available":
            continue
        if current["value"] < previous["value"]:
            return {"value": None, "status": "counter_reset", "reason": "counter_regression"}
        return {
            "value": current["value"] - previous["value"],
            "status": "available",
            "reason": None,
        }
    return {"value": None, "status": "not_sampled", "reason": "no_past_baseline"}


def sample_features(run: RunData) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for index, sample in enumerate(run.samples):
        current = observation(sample, "memory_current_bytes")
        high = observation(sample, "memory_high_bytes")
        maximum = observation(sample, "memory_max_bytes")
        ratios: list[float] = []
        sources: list[str] = []
        for name, limit in (("memory.high", high), ("memory.max", maximum)):
            if (
                current["status"] == "available"
                and limit["status"] == "available"
                and isinstance(limit["value"], (int, float))
                and limit["value"] > 0
            ):
                ratios.append(float(current["value"]) / float(limit["value"]))
                sources.append(name)
        memory_ratio = max(ratios) if ratios else None

        refault_anon = observation(sample, "workingset_refault_anon_delta")
        refault_file = observation(sample, "workingset_refault_file_delta")
        refault_sum = (
            refault_anon["value"] + refault_file["value"]
            if refault_anon["status"] == refault_file["status"] == "available"
            else None
        )
        stall_values = {
            "psi_some_delta_us": observation(sample, "psi_some_delta_us"),
            "psi_full_delta_us": observation(sample, "psi_full_delta_us"),
            "refault_sum_delta": {
                "value": refault_sum,
                "status": "available" if refault_sum is not None else "unavailable",
            },
            "pgscan_delta": observation(sample, "pgscan_delta"),
            "pgsteal_delta": observation(sample, "pgsteal_delta"),
        }
        depth = observation(sample, "queue_depth")
        queued_bytes = observation(sample, "queued_bytes")
        workers = observation(sample, "worker_count")
        depth_per_worker = None
        if depth["status"] == "available" and workers["status"] == "available":
            denominator = max(1.0, float(workers["value"]))
            depth_per_worker = float(depth["value"]) / denominator

        features.append(
            {
                "run_id": run.run_id,
                "sample_index": index,
                "sample_ready_ts_ns": sample.get("sample_ready_ts_ns"),
                "phase": phase_value(sample),
                "memory_ratio": memory_ratio,
                "memory_sources": sources,
                "memory_status": "available" if ratios else "unavailable",
                "stall_values": stall_values,
                "stall_status": (
                    "available"
                    if all(
                        item.get("status") == "available"
                        for item in stall_values.values()
                    )
                    else "unavailable"
                ),
                "depth_per_worker": depth_per_worker,
                "queued_bytes": (
                    float(queued_bytes["value"])
                    if queued_bytes["status"] == "available"
                    else None
                ),
                "queue_status": (
                    "available"
                    if depth_per_worker is not None and queued_bytes["status"] == "available"
                    else "unavailable"
                ),
            }
        )
    return features


def _training_feature_values(
    features_by_run: dict[str, list[dict[str, Any]]],
    training_run_ids: Sequence[str],
    extractor: Any,
) -> list[float]:
    values: list[float] = []
    for run_id in training_run_ids:
        for feature in features_by_run.get(run_id, []):
            value = extractor(feature)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
    return values


def training_state_thresholds(
    features_by_run: dict[str, list[dict[str, Any]]],
    training_run_ids: Sequence[str],
) -> dict[str, Any]:
    stall_names = (
        "psi_some_delta_us",
        "psi_full_delta_us",
        "refault_sum_delta",
        "pgscan_delta",
        "pgsteal_delta",
    )
    stall: dict[str, Any] = {}
    for name in stall_names:
        values = _training_feature_values(
            features_by_run,
            training_run_ids,
            lambda feature, field=name: feature["stall_values"][field].get("value"),
        )
        positive = [value for value in values if value > 0]
        stall[name] = {
            "medium": max(1.0, percentile(positive, 75) or 1.0),
            "high": max(1.0, percentile(positive, 95) or 1.0),
            "training_available_count": len(values),
            "training_positive_count": len(positive),
            "source": "training_positive_p75_p95_with_absolute_minimum_1",
        }
    depth_values = _training_feature_values(
        features_by_run, training_run_ids, lambda feature: feature["depth_per_worker"]
    )
    byte_values = _training_feature_values(
        features_by_run, training_run_ids, lambda feature: feature["queued_bytes"]
    )
    return {
        "memory_ratio": {
            "medium": 0.75,
            "high": 0.90,
            "source": "preregistered_fixed_fraction_of_effective_cgroup_limit",
        },
        "stall": stall,
        "queue": {
            "depth_per_worker": {
                "medium": max(1.0, percentile([v for v in depth_values if v > 0], 75) or 1.0),
                "high": max(1.0, percentile([v for v in depth_values if v > 0], 95) or 1.0),
            },
            "queued_bytes": {
                "medium": max(1.0, percentile([v for v in byte_values if v > 0], 75) or 1.0),
                "high": max(1.0, percentile([v for v in byte_values if v > 0], 95) or 1.0),
            },
            "source": "training_positive_p75_p95_with_absolute_minimum_1",
        },
        "training_run_ids": list(training_run_ids),
    }


def ordinal(value: float, medium: float, high: float) -> int:
    if value >= high:
        return 2
    if value >= medium:
        return 1
    return 0


def candidate_states(
    features: Sequence[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    output = {name: [] for name in ("A_memory_only", "B_memory_and_risk", "C_two_of_three", "D_linear")}
    for feature in features:
        dimensions: dict[str, int | None] = {"memory": None, "stall": None, "queue": None}
        if feature["memory_status"] == "available":
            rule = thresholds["memory_ratio"]
            dimensions["memory"] = ordinal(feature["memory_ratio"], rule["medium"], rule["high"])
        if feature["stall_status"] == "available":
            levels = [
                ordinal(
                    float(feature["stall_values"][name]["value"]),
                    thresholds["stall"][name]["medium"],
                    thresholds["stall"][name]["high"],
                )
                for name in thresholds["stall"]
            ]
            dimensions["stall"] = max(levels)
        if feature["queue_status"] == "available":
            depth_rule = thresholds["queue"]["depth_per_worker"]
            bytes_rule = thresholds["queue"]["queued_bytes"]
            dimensions["queue"] = max(
                ordinal(feature["depth_per_worker"], depth_rule["medium"], depth_rule["high"]),
                ordinal(feature["queued_bytes"], bytes_rule["medium"], bytes_rule["high"]),
            )

        def record(name: str, level: int | None, reason: str | None, score: float | None = None) -> None:
            output[name].append(
                {
                    "run_id": feature["run_id"],
                    "sample_index": feature["sample_index"],
                    "sample_ready_ts_ns": feature["sample_ready_ts_ns"],
                    "phase": feature["phase"],
                    "state": (
                        ("LOW", "MEDIUM", "HIGH")[level]
                        if isinstance(level, int)
                        else "UNAVAILABLE"
                    ),
                    "state_unavailable_reason": reason,
                    "dimension_levels": dimensions,
                    "score": score,
                }
            )

        memory = dimensions["memory"]
        record(
            "A_memory_only",
            memory,
            None if memory is not None else "memory_headroom_unavailable",
        )
        if all(dimensions[name] is not None for name in ("memory", "stall", "queue")):
            stall = int(dimensions["stall"])
            queue = int(dimensions["queue"])
            memory_level = int(memory)
            level_b = (
                2 if memory_level == 2 and (stall == 2 or queue == 2)
                else 1 if max(memory_level, stall, queue) >= 1
                else 0
            )
            high_votes = sum(value == 2 for value in (memory_level, stall, queue))
            level_c = (
                2 if high_votes >= 2
                else 1 if max(memory_level, stall, queue) >= 1
                else 0
            )
            linear_score = (0.40 * memory_level + 0.35 * stall + 0.25 * queue) / 2.0
            level_d = 2 if linear_score >= 0.75 else 1 if linear_score >= 0.35 else 0
            record("B_memory_and_risk", level_b, None)
            record("C_two_of_three", level_c, None)
            record("D_linear", level_d, None, linear_score)
        else:
            missing = ",".join(
                name for name, value in dimensions.items() if value is None
            )
            for name in ("B_memory_and_risk", "C_two_of_three", "D_linear"):
                record(name, None, f"core_dimension_unavailable:{missing}")
    return output


def label_outcome(raw: dict[str, Any], threshold: float) -> dict[str, Any]:
    if not raw.get("eligible"):
        return {
            **raw,
            "positive": None,
            "threshold": threshold,
            "onset_ts_ns": None,
        }
    magnitude = float(raw.get("magnitude", 0.0))
    onset = None
    if magnitude >= threshold:
        for timestamp, delta in raw.get("timeline", []):
            if delta >= threshold:
                onset = timestamp
                break
    return {
        **raw,
        "positive": magnitude >= threshold,
        "threshold": threshold,
        "onset_ts_ns": onset,
    }


def apply_fold_labels(
    run: RunData,
    raw_rows: Sequence[dict[str, Any]],
    outcome_thresholds: dict[str, Any],
    decode_rules: dict[str, Any],
) -> list[dict[str, Any]]:
    stratum_rule = decode_rules["by_stratum"].get(run.stratum)
    decode_threshold = (
        stratum_rule.get("value_ns")
        if stratum_rule and stratum_rule.get("value_ns") is not None
        else decode_rules["global_fallback"].get("value_ns")
    )
    decode_source = (
        "training_same_configuration_p95"
        if stratum_rule and stratum_rule.get("value_ns") is not None
        else "training_global_p95_fallback"
    )
    labeled: list[dict[str, Any]] = []
    for row in raw_rows:
        windows: dict[str, Any] = {}
        for window_name, outcomes in row["windows"].items():
            windows[window_name] = {
                name: label_outcome(
                    raw,
                    outcome_thresholds[window_name][name]["value"],
                )
                for name, raw in outcomes.items()
            }
        decode = dict(row["next_decode"])
        if not decode.get("eligible") or decode_threshold is None:
            decode.update(
                {
                    "positive": None,
                    "threshold_ns": decode_threshold,
                    "threshold_source": decode_source,
                    "reason": decode.get("reason") or "training_decode_p95_unavailable",
                }
            )
        else:
            decode.update(
                {
                    "positive": decode["latency_ns"] > decode_threshold,
                    "threshold_ns": decode_threshold,
                    "threshold_source": decode_source,
                }
            )
        labeled.append({**row, "windows": windows, "next_decode": decode})
    return labeled


def confusion(states: Sequence[dict[str, Any]], actual: Sequence[bool | None]) -> dict[str, Any]:
    tp = fp = tn = fn = unavailable = 0
    high = 0
    for state, label in zip(states, actual):
        if state["state"] == "UNAVAILABLE" or label is None:
            unavailable += 1
            continue
        predicted = state["state"] == "HIGH"
        high += int(predicted)
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1
    eligible = tp + fp + tn + fn
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "eligible": eligible,
        "unavailable": unavailable,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        ),
        "specificity": ratio(tn, tn + fp),
        "false_alarm_rate": ratio(fp, fp + tn),
        "high_sample_coverage": ratio(high, eligible),
        "outcome_prevalence": ratio(tp + fn, eligible),
    }


def unavailable_reason_counts(
    states: Sequence[dict[str, Any]],
    details: Sequence[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    state_reasons: Counter[str] = Counter()
    outcome_reasons: Counter[str] = Counter()
    for state, detail in zip(states, details):
        if state.get("state") == "UNAVAILABLE":
            state_reasons[
                str(state.get("state_unavailable_reason") or "unspecified")
            ] += 1
        if detail.get("positive") is None:
            outcome_reasons[str(detail.get("reason") or "unspecified")] += 1
    return {
        "state": dict(sorted(state_reasons.items())),
        "outcome": dict(sorted(outcome_reasons.items())),
    }


def state_durations(
    states: Sequence[dict[str, Any]],
    max_gap_ns: int,
) -> dict[str, Any]:
    durations = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNAVAILABLE": 0}
    transitions = 0
    for current, following in zip(states, states[1:]):
        start = current.get("sample_ready_ts_ns")
        end = following.get("sample_ready_ts_ns")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            continue
        gap = end - start
        if gap > max_gap_ns:
            durations["UNAVAILABLE"] += gap
            continue
        state = current["state"]
        durations[state] += gap
        if following["state"] != state:
            transitions += 1
    total = sum(durations.values())
    return {
        "duration_ns": durations,
        "transition_count": transitions,
        "high_time_coverage": durations["HIGH"] / total if total else None,
        "integrated_duration_ns": total,
    }


def merge_high_episodes(
    states: Sequence[dict[str, Any]],
    max_gap_ns: int,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous_ts: int | None = None
    for state in states:
        timestamp = state.get("sample_ready_ts_ns")
        is_high = state.get("state") == "HIGH" and isinstance(timestamp, int)
        adjacent = (
            is_high
            and current is not None
            and previous_ts is not None
            and timestamp - previous_ts <= max_gap_ns
        )
        if is_high and (current is None or not adjacent):
            if current is not None:
                episodes.append(current)
            current = {
                "start_ts_ns": timestamp,
                "end_ts_ns": timestamp,
                "sample_count": 1,
            }
        elif is_high and current is not None:
            current["end_ts_ns"] = timestamp
            current["sample_count"] += 1
        elif current is not None:
            episodes.append(current)
            current = None
        previous_ts = timestamp if isinstance(timestamp, int) else previous_ts
    if current is not None:
        episodes.append(current)
    for episode in episodes:
        episode["duration_ns"] = episode["end_ts_ns"] - episode["start_ts_ns"]
    return episodes


def episode_metrics(
    states: Sequence[dict[str, Any]],
    labels: Sequence[dict[str, Any]],
    opportunities: Sequence[int],
    window_ns: int,
    max_gap_ns: int,
) -> dict[str, Any]:
    episodes = merge_high_episodes(states, max_gap_ns)
    outcome_events = sorted(
        {
            int(label["onset_ts_ns"])
            for label in labels
            if label.get("positive") and isinstance(label.get("onset_ts_ns"), int)
        }
    )
    hit_events: set[int] = set()
    credited_events: set[int] = set()
    leads: list[float] = []
    episode_rows: list[dict[str, Any]] = []
    for episode in episodes:
        start = episode["start_ts_ns"]
        horizon_end = episode["end_ts_ns"] + window_ns
        event = next(
            (
                timestamp for timestamp in outcome_events
                if start < timestamp <= horizon_end and timestamp not in credited_events
            ),
            None,
        )
        opportunity_slice = [
            timestamp for timestamp in opportunities
            if start < timestamp < event
        ] if event is not None else []
        row = dict(episode)
        row["outcome_onset_ts_ns"] = event
        row["hit"] = event is not None
        row["lead_time_ns"] = event - start if event is not None else None
        row["first_task_opportunity_ts_ns"] = opportunity_slice[0] if opportunity_slice else None
        row["task_opportunity_count"] = len(opportunity_slice)
        row["high_to_first_opportunity_ns"] = (
            opportunity_slice[0] - start if opportunity_slice else None
        )
        row["first_opportunity_to_outcome_ns"] = (
            event - opportunity_slice[0]
            if event is not None and opportunity_slice
            else None
        )
        if event is not None:
            credited_events.add(event)
            hit_events.add(event)
            leads.append(float(event - start))
        episode_rows.append(row)
    span_ns = (
        states[-1]["sample_ready_ts_ns"] - states[0]["sample_ready_ts_ns"]
        if len(states) >= 2
        and isinstance(states[0].get("sample_ready_ts_ns"), int)
        and isinstance(states[-1].get("sample_ready_ts_ns"), int)
        else 0
    )
    false_episodes = sum(not row["hit"] for row in episode_rows)
    return {
        "dedupe_rule": "one outcome onset timestamp; multiple HIGH episodes credit the earliest matching episode only",
        "high_episode_count": len(episodes),
        "hit_episode_count": sum(row["hit"] for row in episode_rows),
        "episode_precision": (
            sum(row["hit"] for row in episode_rows) / len(episodes)
            if episodes else None
        ),
        "outcome_event_count": len(outcome_events),
        "evidence_status": (
            "sufficient" if len(outcome_events) >= 3 else "insufficient"
        ),
        "hit_outcome_event_count": len(hit_events),
        "outcome_event_recall": (
            len(hit_events) / len(outcome_events) if outcome_events else None
        ),
        "false_high_episodes_per_minute": (
            false_episodes / (span_ns / 60_000_000_000)
            if span_ns > 0 else None
        ),
        "lead_time_ns": distribution(leads),
        "episode_duration_ns": distribution(
            [row["duration_ns"] for row in episode_rows]
        ),
        "episodes": episode_rows,
    }


def select_candidate(
    candidate_names: Sequence[str],
    training_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def key(name: str) -> tuple[Any, ...]:
        metrics = training_metrics[name]["selection_summary"]
        qualifying = metrics[
            "qualifying_independent_outcomes_precision_0_80_nonzero_recall"
        ]
        nonzero = metrics["independent_outcomes_with_nonzero_recall"]
        worst_stratum_f1 = metrics.get("worst_run_outcome_f1")
        f1 = metrics.get("mean_independent_outcome_f1")
        precision = metrics.get("mean_independent_outcome_precision")
        coverage = metrics.get("mean_high_sample_coverage")
        return (
            qualifying,
            nonzero,
            worst_stratum_f1 if worst_stratum_f1 is not None else -1.0,
            f1 if f1 is not None else -1.0,
            precision if precision is not None else -1.0,
            coverage if coverage is not None else -1.0,
            -candidate_names.index(name),
        )
    selected = max(candidate_names, key=key)
    return {
        "name": selected,
        "selection_source": "training_only",
        "priority": [
            "count_independent_outcomes_with_precision_at_least_0.80_and_nonzero_recall",
            "count_independent_outcomes_with_nonzero_recall",
            "worst_complete_run_outcome_F1",
            "mean_independent_outcome_F1",
            "mean_independent_outcome_precision",
            "mean_coverage",
            "simpler_preregistered_order",
        ],
        "training_metrics": training_metrics[selected],
    }


def outcome_label_sequence(
    labeled_rows: Sequence[dict[str, Any]],
    outcome_name: str,
    window_name: str,
) -> list[bool | None]:
    if outcome_name == "decode_long_tail":
        return [row["next_decode"].get("positive") for row in labeled_rows]
    return [
        row["windows"][window_name][outcome_name].get("positive")
        for row in labeled_rows
    ]


def outcome_detail_sequence(
    labeled_rows: Sequence[dict[str, Any]],
    outcome_name: str,
    window_name: str,
) -> list[dict[str, Any]]:
    if outcome_name == "decode_long_tail":
        return [row["next_decode"] for row in labeled_rows]
    return [row["windows"][window_name][outcome_name] for row in labeled_rows]


def training_candidate_metrics(
    candidate: str,
    states_by_run: dict[str, dict[str, list[dict[str, Any]]]],
    labels_by_run: dict[str, list[dict[str, Any]]],
    training_run_ids: Sequence[str],
) -> dict[str, Any]:
    per_outcome: dict[str, Any] = {}
    stratum_f1_values: list[float] = []
    qualifying_outcomes = 0
    nonzero_recall_outcomes = 0
    independent_f1_values: list[float] = []
    independent_precision_values: list[float] = []
    coverage_values: list[float] = []

    for window_name in WINDOWS_NS:
        per_outcome[window_name] = {}
        for outcome_name in PRIMARY_OUTCOMES:
            if outcome_name == "decode_long_tail" and window_name != "100ms":
                continue
            all_states: list[dict[str, Any]] = []
            all_actual: list[bool | None] = []
            by_run: dict[str, Any] = {}
            for run_id in training_run_ids:
                states = states_by_run[run_id][candidate]
                actual = outcome_label_sequence(
                    labels_by_run[run_id], outcome_name, window_name
                )
                metrics = confusion(states, actual)
                by_run[run_id] = metrics
                all_states.extend(states)
                all_actual.extend(actual)
                if metrics["f1"] is not None:
                    stratum_f1_values.append(float(metrics["f1"]))
            aggregate = confusion(all_states, all_actual)
            aggregate["by_run"] = by_run
            per_outcome[window_name][outcome_name] = aggregate
            if window_name == "100ms":
                precision = aggregate["precision"]
                recall = aggregate["recall"]
                if precision is not None:
                    independent_precision_values.append(float(precision))
                if recall is not None and recall > 0:
                    nonzero_recall_outcomes += 1
                if (
                    precision is not None
                    and precision >= 0.80
                    and recall is not None
                    and recall > 0
                ):
                    qualifying_outcomes += 1
                if aggregate["f1"] is not None:
                    independent_f1_values.append(float(aggregate["f1"]))
                if aggregate["high_sample_coverage"] is not None:
                    coverage_values.append(
                        float(aggregate["high_sample_coverage"])
                    )

    return {
        "selection_source": "training_only_independent_outcomes",
        "per_outcome": per_outcome,
        "selection_summary": {
            "qualifying_independent_outcomes_precision_0_80_nonzero_recall":
                qualifying_outcomes,
            "independent_outcomes_with_nonzero_recall":
                nonzero_recall_outcomes,
            "worst_run_outcome_f1": (
                min(stratum_f1_values) if stratum_f1_values else None
            ),
            "mean_independent_outcome_f1": (
                statistics.mean(independent_f1_values)
                if independent_f1_values else None
            ),
            "mean_independent_outcome_precision": (
                statistics.mean(independent_precision_values)
                if independent_precision_values else None
            ),
            "mean_high_sample_coverage": (
                statistics.mean(coverage_values) if coverage_values else None
            ),
        },
    }


def run_candidate_metrics(
    run: RunData,
    selected_states: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    target_interval_ns: int,
) -> dict[str, Any]:
    run_metrics: dict[str, Any] = {}
    durations = state_durations(selected_states, 2 * target_interval_ns)
    for window_name in WINDOWS_NS:
        run_metrics[window_name] = {}
        for outcome_name in PRIMARY_OUTCOMES:
            if outcome_name == "decode_long_tail" and window_name != "100ms":
                continue
            actual = outcome_label_sequence(rows, outcome_name, window_name)
            details = outcome_detail_sequence(rows, outcome_name, window_name)
            metrics = confusion(selected_states, actual)
            metrics["unavailable_reasons"] = unavailable_reason_counts(
                selected_states, details
            )
            metrics["duration"] = durations
            metrics["episode"] = episode_metrics(
                selected_states,
                details,
                run.task_opportunities,
                WINDOWS_NS[window_name],
                2 * target_interval_ns,
            )
            metrics["by_phase"] = {}
            for phase in ("PREFILL", "DECODE"):
                indexes = [
                    index for index, state in enumerate(selected_states)
                    if state.get("phase") == phase
                ]
                phase_states = [selected_states[index] for index in indexes]
                phase_actual = [actual[index] for index in indexes]
                phase_details = [details[index] for index in indexes]
                metrics["by_phase"][phase] = {
                    **confusion(phase_states, phase_actual),
                    "unavailable_reasons": unavailable_reason_counts(
                        phase_states, phase_details
                    ),
                    "duration": state_durations(
                        phase_states, 2 * target_interval_ns
                    ),
                    "episode": episode_metrics(
                        phase_states,
                        phase_details,
                        run.task_opportunities,
                        WINDOWS_NS[window_name],
                        2 * target_interval_ns,
                    ),
                }
            run_metrics[window_name][outcome_name] = metrics

    decode_actual = outcome_label_sequence(
        rows, "decode_long_tail", "100ms"
    )
    decode_details = outcome_detail_sequence(
        rows, "decode_long_tail", "100ms"
    )
    run_metrics["next_decode"] = {
        **confusion(selected_states, decode_actual),
        "unavailable_reasons": unavailable_reason_counts(
            selected_states, decode_details
        ),
        "duration": durations,
        "episode": episode_metrics(
            selected_states,
            decode_details,
            run.task_opportunities,
            WINDOWS_NS["250ms"],
            2 * target_interval_ns,
        ),
    }
    return run_metrics


def analyze(run_dirs: Sequence[Path], strict: bool = False) -> dict[str, Any]:
    runs = {run.run_id: run for run in (load_run(path) for path in run_dirs)}
    if len(runs) != len(run_dirs):
        raise ValueError("run_id values must be unique")
    if strict:
        for run in runs.values():
            validate_sample_contract(run)
    raw_by_run = {run_id: raw_outcomes(run) for run_id, run in runs.items()}
    features_by_run = {
        run_id: sample_features(run) for run_id, run in runs.items()
    }
    folds = grouped_folds(list(runs))
    fold_results: list[dict[str, Any]] = []
    labeled_output: list[dict[str, Any]] = []

    for fold in folds:
        training_ids = fold["training_run_ids"]
        evaluation_ids = fold["evaluation_run_ids"]
        outcome_thresholds = training_outcome_thresholds(raw_by_run, training_ids)
        decode_rules = decode_thresholds(runs, training_ids)
        state_thresholds = training_state_thresholds(features_by_run, training_ids)
        states_by_run: dict[str, dict[str, list[dict[str, Any]]]] = {}
        labels_by_run: dict[str, list[dict[str, Any]]] = {}
        for run_id in set(training_ids) | set(evaluation_ids):
            states_by_run[run_id] = candidate_states(
                features_by_run[run_id], state_thresholds
            )
            labels_by_run[run_id] = apply_fold_labels(
                runs[run_id],
                raw_by_run[run_id],
                outcome_thresholds,
                decode_rules,
            )

        candidate_names = list(next(iter(states_by_run.values())).keys()) if states_by_run else []
        training_metrics: dict[str, dict[str, Any]] = {
            candidate: training_candidate_metrics(
                candidate,
                states_by_run,
                labels_by_run,
                training_ids,
            )
            for candidate in candidate_names
        }
        selected = select_candidate(candidate_names, training_metrics) if candidate_names else {
            "name": None,
            "selection_source": "unavailable",
        }

        evaluation_metrics: dict[str, Any] = {}
        for run_id in evaluation_ids:
            run = runs[run_id]
            rows = labels_by_run[run_id]
            target_interval_ns = 25_000_000
            summary_events = [
                record for record in read_jsonl(run.run_dir / "memory_trace.jsonl")
                if record.get("event") == "PRESSURE_SHADOW_SUMMARY"
            ]
            if summary_events and isinstance(summary_events[-1].get("sample_interval_ms"), int):
                target_interval_ns = summary_events[-1]["sample_interval_ms"] * 1_000_000
            all_candidate_metrics = {
                candidate: run_candidate_metrics(
                    run,
                    states_by_run[run_id][candidate],
                    rows,
                    target_interval_ns,
                )
                for candidate in candidate_names
            }
            run_metrics = (
                all_candidate_metrics[selected["name"]]
                if selected["name"] else {}
            )
            evaluation_metrics[run_id] = {
                "workers": run.workers,
                "memory_max": run.memory_max,
                "memory_swap_max": run.memory_swap_max,
                "selected_candidate": selected["name"],
                "metrics": run_metrics,
                "all_candidate_metrics": all_candidate_metrics,
            }
            for row, feature in zip(rows, features_by_run[run_id]):
                labeled_output.append(
                    {
                        "schema_version": 1,
                        "fold_id": fold["fold_id"],
                        "split_status": fold["status"],
                        "split": "evaluation",
                        "training_run_ids": training_ids,
                        "evaluation_run_ids": evaluation_ids,
                        "run_id": run_id,
                        "workers": run.workers,
                        "memory_max": run.memory_max,
                        "memory_swap_max": run.memory_swap_max,
                        "sample_index": row["sample_index"],
                        "sample_ready_ts_ns": row["sample_ready_ts_ns"],
                        "phase": row["phase"],
                        "availability": {
                            "memory": feature["memory_status"],
                            "stall_reclaim": feature["stall_status"],
                            "queue": feature["queue_status"],
                        },
                        "features": feature,
                        "outcomes": {
                            "windows": row["windows"],
                            "next_decode": row["next_decode"],
                        },
                        "states": {
                            name: states_by_run[run_id][name][row["sample_index"]]
                            for name in candidate_names
                        },
                        "selected_candidate": selected["name"],
                    }
                )

        fold_results.append(
            {
                **fold,
                "outcome_thresholds": outcome_thresholds,
                "decode_thresholds": decode_rules,
                "state_thresholds": state_thresholds,
                "candidate_training_metrics": training_metrics,
                "selected_candidate": selected,
                "evaluation": evaluation_metrics,
            }
        )

    return {
        "schema_version": 1,
        "analysis": "M5A_pressure_shadow_offline",
        "strict_sample_contract": strict,
        "causal_contract": {
            "state_time": "PRESSURE_SHADOW_SAMPLE.sample_ready_ts_ns",
            "counter_baseline": "first successful read_ts_ns strictly greater than state time",
            "windows": "(t,t+window]",
            "next_decode": "first complete DECODE STEP_BEGIN.ts_ns strictly greater than state time",
            "evaluation_threshold_leakage": "forbidden",
        },
        "run_count": len(runs),
        "runs": {
            run_id: {
                "run_dir": str(run.run_dir),
                "sample_count": len(run.samples),
                "decode_step_count": len(run.decode_steps),
                "task_opportunity_count": len(run.task_opportunities),
                "workers": run.workers,
                "memory_max": run.memory_max,
                "memory_swap_max": run.memory_swap_max,
            }
            for run_id, run in sorted(runs.items())
        },
        "folds": fold_results,
        "labeled_samples": labeled_output,
    }


def candidate_artifact(full: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": full["schema_version"],
        "analysis": full["analysis"],
        "candidate_families": {
            "A_memory_only": "memory headroom ordinal",
            "B_memory_and_risk": "HIGH iff memory high AND (stall high OR queue high)",
            "C_two_of_three": "HIGH iff at least two dimensions high",
            "D_linear": "0.40 memory + 0.35 stall + 0.25 queue; fixed 0.35/0.75 cuts",
        },
        "unavailable_policy": "any required core dimension unavailable propagates UNAVAILABLE; never LOW",
        "folds": [
            {
                key: fold[key]
                for key in (
                    "fold_id",
                    "status",
                    "training_run_ids",
                    "evaluation_run_ids",
                    "state_thresholds",
                    "candidate_training_metrics",
                    "selected_candidate",
                    "evaluation",
                )
            }
            for fold in full["folds"]
        ],
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "M5A_pressure_shadow_full.json"
    labeled_path = output_dir / "M5A_pressure_samples_labeled.jsonl"
    candidate_path = output_dir / "M5A_pressure_state_candidates.json"
    full_without_samples = dict(result)
    labeled_samples = full_without_samples.pop("labeled_samples")
    full_path.write_text(
        json.dumps(full_without_samples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with labeled_path.open("w", encoding="utf-8") as stream:
        for sample in labeled_samples:
            stream.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    candidate_path.write_text(
        json.dumps(candidate_artifact(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = analyze(
        [path.resolve() for path in args.run_dir],
        strict=args.strict,
    )
    write_outputs(result, args.output_dir.resolve())
    print(
        json.dumps(
            {
                "run_count": result["run_count"],
                "fold_count": len(result["folds"]),
                "labeled_sample_count": len(result["labeled_samples"]),
                "output_dir": str(args.output_dir.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
