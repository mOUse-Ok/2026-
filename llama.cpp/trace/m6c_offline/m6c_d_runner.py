#!/usr/bin/env python3
"""Preregistered M6C-D calibration, holdout, and robustness replay runner.

The default CLI is the formal offline entry point.  It accepts only a closed,
ready preregistration and never executes inference or reads physical-system
outcomes.  ``--synthetic-self-audit`` is a deliberately separate path used to
exercise the complete stage machine without opening A.3 Evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import statistics
import struct
from typing import Any, Callable, Iterable, TextIO

from .artifacts import DecisionDigestSink, write_finalized_jsonl
from .evidence import (
    COUNTERFACTUAL_DECLARATION,
    EvidenceRun,
    ReconstructabilityStatus,
    ServiceSlot,
    audit_run,
    load_run_index,
    sha256_file,
    validate_evidence_index,
)
from .model import ModelInvariantError, OfflineQueue, TaskSpec, parse_f64_bits
from .policy import PolicyConfigurationError, UINT64_MAX
from .preregistration import CALIBRATION, HOLDOUT, ROBUSTNESS


FINAL_READY = "PREREGISTRATION_READY_FOR_S1_REPLAY"
RUNNER_SCHEMA = "m6c-d-reserved-service-replay-v1"
EXPECTED_CANDIDATES = ("S1-A", "S1-B", "S1-C", "S1-D")
EXPECTED_TIE_BREAK = [
    "worse-worker p95 median delta",
    "worse-worker p99 median delta",
    "worse-worker lateness p95 median delta",
    "lower reserved share",
    "higher eligibility age",
    "ASCII candidate ID",
]
COUNTERFACTUAL = dict(COUNTERFACTUAL_DECLARATION)


class M6CDGateError(RuntimeError):
    """A frozen input, correctness, safety, or stage gate failed closed."""


@dataclass(frozen=True)
class PreregisteredPolicyConfig:
    """Duck-typed policy config whose values must come from preregistration."""

    candidate_id: str
    reserved_numerator: int
    reserved_denominator: int
    minimum_eligibility_age_ns: int
    hard_urgent_guard_ns: int
    preregistration_sha256: str
    test_fixture_only: bool
    experimental_parameter: bool

    def __post_init__(self) -> None:
        r, d = self.reserved_numerator, self.reserved_denominator
        if self.candidate_id not in EXPECTED_CANDIDATES:
            raise PolicyConfigurationError("candidate is outside the frozen allowlist")
        if not isinstance(r, int) or not isinstance(d, int):
            raise PolicyConfigurationError("R and D must be integers")
        if not (0 < r < d and 2 * r <= d):
            raise PolicyConfigurationError("candidate must satisfy 0 < R < D and 2R <= D")
        if self.minimum_eligibility_age_ns <= 0:
            raise PolicyConfigurationError("AGE_GATED_ALL requires positive age")
        if not 0 <= self.hard_urgent_guard_ns <= UINT64_MAX:
            raise PolicyConfigurationError("guard is outside uint64")
        if len(self.preregistration_sha256) != 64:
            raise PolicyConfigurationError("missing preregistration SHA256")
        if self.test_fixture_only == self.experimental_parameter:
            raise PolicyConfigurationError(
                "synthetic and formal policy markers must be mutually exclusive"
            )


def _canonical_line(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _write_closed_text(path: Path, text: str, *, parse_json: bool = False) -> dict[str, Any]:
    """Write exclusively, close, then reopen for final size/SHA/parse metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("x", encoding="utf-8")
    try:
        stream.write(text)
        stream.flush()
    finally:
        stream.close()
    raw = path.read_bytes()
    if parse_json:
        json.loads(raw)
    return {
        "path": str(path),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "closed_before_hash": True,
        "json_parseable": bool(parse_json),
    }


def _write_closed_json(path: Path, value: Any) -> dict[str, Any]:
    return _write_closed_text(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        parse_json=True,
    )


def _nearest_rank(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _median(values: Iterable[int | float | None]) -> int | float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def _longest_streak(values: Iterable[str], expected: str) -> int:
    longest = current = 0
    for value in values:
        if value == expected:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _summary(values: list[int]) -> dict[str, int | None]:
    return {
        "count": len(values),
        "p50": _nearest_rank(values, 0.50),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "max": max(values) if values else None,
    }


def _worst_1pct_mean(values: list[int]) -> float | None:
    if not values:
        return None
    count = max(1, math.ceil(0.01 * len(values)))
    return statistics.fmean(sorted(values, reverse=True)[:count])


def _candidate_map(preregistration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = preregistration.get("candidate_allowlist")
    if not isinstance(values, list):
        raise M6CDGateError("candidate_allowlist missing")
    result = {
        item.get("candidate_id"): item
        for item in values
        if isinstance(item, dict) and item.get("candidate_id") != "S0"
    }
    if tuple(result) != EXPECTED_CANDIDATES:
        raise M6CDGateError("candidate allowlist or ordering differs from frozen v1")
    expected = {
        "S1-A": (1, 16, 28_000_000, 0),
        "S1-B": (1, 16, 41_000_000, 0),
        "S1-C": (1, 8, 28_000_000, 0),
        "S1-D": (1, 8, 41_000_000, 0),
    }
    actual = {
        key: (value.get("R"), value.get("D"), value.get("eligibility_age_ns"), value.get("guard_ns"))
        for key, value in result.items()
    }
    if actual != expected:
        raise M6CDGateError("S1-A/B/C/D parameters differ from frozen v1")
    return result


def validate_frozen_contract(preregistration: dict[str, Any], *, require_ready: bool) -> None:
    """Reject any change to the v1 split, candidates, metrics, budgets, or gates."""

    if require_ready and preregistration.get("final_enum") != FINAL_READY:
        raise M6CDGateError("formal M6C-D requires a ready preregistration")
    _candidate_map(preregistration)
    split = preregistration.get("run_split", {})
    roles = {
        "calibration": CALIBRATION,
        "holdout": HOLDOUT,
        "robustness": ROBUSTNESS,
    }
    all_ids: list[str] = []
    for role, expected_rows in roles.items():
        records = split.get(role)
        if not isinstance(records, list) or len(records) != len(expected_rows):
            raise M6CDGateError(f"frozen {role} Run count changed")
        for record, expected in zip(records, expected_rows):
            run_id = record.get("run_id") if isinstance(record, dict) else None
            if not isinstance(run_id, str) or not run_id:
                raise M6CDGateError(f"invalid {role} Run ID")
            expected_run_id, workers, repeat, configuration = expected
            if record != {
                "configuration_id": configuration,
                "expected_service_slots": 29_262,
                "repeat_index": repeat,
                "run_id": expected_run_id,
                "workers": workers,
            }:
                raise M6CDGateError(f"frozen {role} Run identity or metadata changed")
            all_ids.append(run_id)
    if len(set(all_ids)) != 30:
        raise M6CDGateError("frozen Run split is not disjoint")
    if preregistration.get("metric_definitions") != {
        "cross_run_median": "midpoint for even N; middle value for odd N",
        "lateness": "selected_ts_ns-deadline_ts_ns for selected-after-deadline Tasks",
        "longest_task_list_size": 100,
        "paired_delta": "S1-S0",
        "queue_wait": "selection_decision_ts_ns-enqueued_ts_ns",
        "raw_deadline_inversion": "live earlier nonzero deadline exists while selected deadline is later or zero",
        "selected_after_deadline": "selected_ts_ns>=deadline_ts_ns for all nonzero deadlines",
        "task_percentile": "nearest-rank",
        "worst_1pct_mean": "mean of largest max(1,ceil(0.01*N)) waits",
    }:
        raise M6CDGateError("metric definitions differ from frozen v1")
    if preregistration.get("fairness_budgets") != [
        {"absolute_ns": 2_000_000, "metric": "worker_median_queue_wait_p99", "relative": 0.1},
        {"absolute_ns": 10_000_000, "metric": "per_run_queue_wait_p99", "relative": 0.25},
        {"absolute_ns": 5_000_000, "metric": "per_run_worst_1pct_mean", "relative": 0.2},
        {"absolute_ns": 10_000_000, "metric": "per_run_max_queue_wait", "relative": 0.25},
    ]:
        raise M6CDGateError("fairness budgets differ from frozen v1")
    if preregistration.get("urgency_budgets") != [
        {"absolute_rate": 0.0005, "metric": "selected_after_deadline_rate", "relative": 0.05},
        {"absolute_rate": 0.0005, "metric": "raw_deadline_inversion_rate", "relative": 0.05},
        {"absolute_ns": 1_000_000, "metric": "lateness_p95", "relative": 0.05},
        {"absolute_ns": 2_000_000, "metric": "lateness_p99", "relative": 0.1},
        {"absolute_ns": 10_000_000, "metric": "lateness_max", "relative": 0.25},
    ]:
        raise M6CDGateError("urgency budgets differ from frozen v1")
    calibration_gate = preregistration.get("calibration_gate", {})
    if calibration_gate != {
        "correctness_and_safety": "all zero/PASS",
        "intervention": "nonzero, <=R/D+max(0.005,1/N), longest reserved streak<=1",
        "maximum_finalists": 2,
        "p95_run_support": "2/2 Runs improve in each worker stratum",
        "p95_worker_medians": "workers=2 and workers=4 both <0ns",
        "p99_and_tail": "within frozen fairness budgets",
        "tie_break": EXPECTED_TIE_BREAK,
        "urgency": "every Run within frozen urgency budgets",
    }:
        raise M6CDGateError("calibration finalist rule differs from frozen v1")
    if preregistration.get("holdout_gate") != {
        "correctness_and_safety": "all zero/PASS",
        "determinism_and_artifacts": "30/30-equivalent applicable Gate PASS",
        "p95_run_support": "at least 2/3 Runs improve in each worker stratum",
        "p95_worker_medians": "workers=2 and workers=4 both <0ns",
        "tail_and_urgency": "every frozen budget applies",
        "unique_recommendation": "same frozen tie-break if two finalists pass",
    }:
        raise M6CDGateError("holdout recommendation rule differs from frozen v1")
    if preregistration.get("material_worse_definition") != {
        "input_units": "raw ns or raw rate",
        "operator": "delta_abs>abs_budget AND (S0>0 ? delta_abs/S0>rel_budget : delta_abs>0)",
    }:
        raise M6CDGateError("material-worse operator differs from frozen v1")
    robustness = preregistration.get("robustness_boundary", {})
    if robustness != {
        "can_select_or_change_parameters": False,
        "candidate_count": 1,
        "primary_n30_claim": False,
        "runs": 20,
        "starts_after_unique_b0_holdout_recommendation": True,
    }:
        raise M6CDGateError("robustness boundary differs from frozen v1")


def _policy_config(
    candidate: dict[str, Any],
    preregistration_sha256: str,
    *,
    synthetic: bool,
) -> PreregisteredPolicyConfig:
    return PreregisteredPolicyConfig(
        candidate_id=str(candidate["candidate_id"]),
        reserved_numerator=int(candidate["R"]),
        reserved_denominator=int(candidate["D"]),
        minimum_eligibility_age_ns=int(candidate["eligibility_age_ns"]),
        hard_urgent_guard_ns=int(candidate["guard_ns"]),
        preregistration_sha256=preregistration_sha256,
        test_fixture_only=synthetic,
        experimental_parameter=not synthetic,
    )


def _safety_pass(result: dict[str, Any]) -> tuple[bool, list[str]]:
    safety = result.get("safety", {})
    blockers = []
    for name in (
        "hard_urgent_violation",
        "reserved_state_machine_violation",
        "stale_handle_count",
        "full_store_scan_count",
        "task_conservation_error",
        "queued_bytes_error",
        "store_index_registry_error",
        "duplicate_remove_count",
        "generation_mismatch_count",
        "s0_oracle_winner_mismatches",
        "b0_runtime_winner_mismatches",
        "nondeterministic_rerun",
    ):
        value = safety.get(name)
        if value not in (0, None):
            blockers.append(name)
    if safety.get("invariants_passed") is not True:
        blockers.append("invariants_passed")
    return not blockers, blockers


def _run_once(
    run: EvidenceRun,
    policy_config: PreregisteredPolicyConfig | None,
    *,
    decision_stream: TextIO | DecisionDigestSink | None = None,
) -> dict[str, Any]:
    """Replay one Run in dense decision order with one isolated queue state."""

    queue = OfflineQueue(run.capacity, policy_config=policy_config)  # type: ignore[arg-type]
    arrivals = list(run.tasks)
    if arrivals != sorted(arrivals, key=lambda item: (item.enqueued_ts_ns, item.sequence, item.task_id)):
        raise M6CDGateError(f"{run.run_id}: arrivals are not canonically ordered")
    arrival_index = 0
    selected_ids: set[int] = set()
    waits: dict[int, int] = {}
    task_rows: dict[int, dict[str, Any]] = {}
    selected_sources: list[str] = []
    eligibility_to_selection: list[int] = []
    eligible_excluded = 0
    late_values: list[int] = []
    deadline_tasks = 0
    selected_after_deadline = 0
    raw_inversions = 0
    hard_urgent_violations = 0
    state_violations = 0
    b0_runtime_mismatches = 0
    s0_oracle_mismatches = 0
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    reference_heap: list[tuple[tuple[bool, int, float, int], int]] = []

    for expected_decision_id, service_slot in enumerate(run.slots):
        if service_slot.decision_id != expected_decision_id:
            raise M6CDGateError(f"{run.run_id}: decision IDs are not dense and ordered")
        while (
            arrival_index < len(arrivals)
            and arrivals[arrival_index].enqueued_ts_ns <= service_slot.decision_ts_ns
        ):
            arriving = arrivals[arrival_index]
            queue.enqueue(arriving)
            if policy_config is None:
                heapq.heappush(reference_heap, (
                    (
                        arriving.deadline_ts_ns == 0,
                        arriving.deadline_ts_ns,
                        -parse_f64_bits(arriving.route_score_f64_bits),
                        arriving.sequence,
                    ),
                    arriving.task_id,
                ))
            arrival_index += 1
        if queue.store.live_count != service_slot.queue_depth_before:
            raise M6CDGateError(
                f"{run.run_id}: queue depth mismatch at decision {service_slot.decision_id}"
            )
        record = queue.select(
            decision_ts_ns=service_slot.decision_ts_ns,
            batch_id=service_slot.batch_id,
            batch_slot=service_slot.batch_slot,
            worker_id=service_slot.worker_id,
        )
        if policy_config is not None:
            record["mode"] = "s1_synthetic_fixture" if policy_config.test_fixture_only else "s1_preregistered"
            record["candidate_id"] = policy_config.candidate_id
            record["test_fixture_only"] = policy_config.test_fixture_only
            record["experimental_parameter"] = policy_config.experimental_parameter
        selected = record["selected_task"]
        selected_id = int(selected["task_id"])
        if selected_id in selected_ids:
            raise ModelInvariantError("Task selected more than once")
        selected_ids.add(selected_id)
        wait = service_slot.decision_ts_ns - int(selected["enqueued_ts_ns"])
        if wait < 0:
            raise ModelInvariantError("selection precedes enqueue")
        waits[selected_id] = wait
        source = str(record["selected_source"])
        selected_sources.append(source)
        counts[f"source_{source}"] += 1
        counts["waiting_eligible"] += int(bool(record["oldest_eligible"].get("available")))
        counts["no_eligible"] += int(not bool(record["oldest_eligible"].get("available")))
        counts["debt_pending"] += int(bool(record["debt_before"]))
        counts["debt_created"] += int(bool(record["debt_created"]))
        counts["debt_repaid"] += int(bool(record["debt_repaid"]))
        counts["hard_urgent_override"] += int(record["override_reason"] == "HARD_URGENT_OVERRIDE")
        if record["hard_urgent_present"] and selected_id != int(record["legacy_head"]["task_id"]):
            hard_urgent_violations += 1
        if not isinstance(record["debt_before"], bool) or not isinstance(record["debt_after"], bool):
            state_violations += 1
        if policy_config is not None and not 0 <= int(record["credit_after"]) < policy_config.reserved_denominator:
            state_violations += 1

        deadline = int(selected["deadline_ts_ns"])
        late = deadline != 0 and service_slot.decision_ts_ns >= deadline
        deadline_tasks += int(deadline != 0)
        selected_after_deadline += int(late)
        lateness = service_slot.decision_ts_ns - deadline if late else None
        if lateness is not None:
            late_values.append(lateness)
        legacy_deadline = int(record["legacy_head"]["deadline_ts_ns"])
        inversion = (
            legacy_deadline != 0
            and selected_id != int(record["legacy_head"]["task_id"])
            and (deadline == 0 or legacy_deadline < deadline)
        )
        raw_inversions += int(inversion)
        eligible_ts = selected.get("eligible_ts_ns")
        if eligible_ts is not None and service_slot.decision_ts_ns >= int(eligible_ts):
            eligibility_to_selection.append(service_slot.decision_ts_ns - int(eligible_ts))
        else:
            eligible_excluded += 1
        task_rows[selected_id] = {
            "task_id": selected_id,
            "sequence": int(selected["sequence"]),
            "enqueued_ts_ns": int(selected["enqueued_ts_ns"]),
            "selected_ts_ns": service_slot.decision_ts_ns,
            "queue_wait_ns": wait,
            "deadline_ts_ns": deadline,
            "selected_after_deadline": late,
            "lateness_ns": lateness,
            "selected_source": source,
        }
        if policy_config is None:
            if not reference_heap:
                raise ModelInvariantError("independent deadline_score oracle is empty")
            _, expected = heapq.heappop(reference_heap)
            s0_oracle_mismatches += int(selected_id != expected)
            if run.priority_mode == "deadline_score":
                b0_runtime_mismatches += int(selected_id != service_slot.winner_task_id)
        queue.complete_selected(selected_id)

        trace_record = {
            **record,
            "source_decision_id": service_slot.decision_id,
            "source_actual_winner_task_id": service_slot.winner_task_id,
            "raw_deadline_inversion": inversion,
            **COUNTERFACTUAL,
        }
        line = _canonical_line(trace_record)
        digest.update(line.encode("utf-8"))
        if decision_stream is not None:
            decision_stream.write(line)

    audit = queue.audit_invariants()
    stale = queue.legacy_index.stale_handle_count + queue.aging_index.stale_handle_count
    conservation_error = int(
        arrival_index != len(arrivals)
        or len(selected_ids) != len(arrivals)
        or queue.store.live_count != 0
        or queue.store.selected
        or len(queue.store.terminal) != len(arrivals)
    )
    waits_list = list(waits.values())
    longest = sorted(
        task_rows.values(),
        key=lambda item: (-item["queue_wait_ns"], item["sequence"], item["task_id"]),
    )[:100]
    deadline_rate = selected_after_deadline / deadline_tasks if deadline_tasks else 0.0
    inversion_rate = raw_inversions / len(run.slots) if run.slots else None
    result = {
        "run_id": run.run_id,
        "configuration_id": run.configuration_id,
        "workers": run.workers,
        "repeat_index": run.repeat_index,
        "candidate_id": "S0" if policy_config is None else policy_config.candidate_id,
        "parameters": None if policy_config is None else {
            "R": policy_config.reserved_numerator,
            "D": policy_config.reserved_denominator,
            "eligibility_age_ns": policy_config.minimum_eligibility_age_ns,
            "guard_ns": policy_config.hard_urgent_guard_ns,
            "test_fixture_only": policy_config.test_fixture_only,
            "experimental_parameter": policy_config.experimental_parameter,
        },
        "decision_count": len(run.slots),
        "decision_order_dense": True,
        "decision_digest_sha256": digest.hexdigest(),
        "queue_wait": {**_summary(waits_list), "worst_1pct_mean": _worst_1pct_mean(waits_list)},
        "task_metrics": [task_rows[task_id] for task_id in sorted(task_rows)],
        "longest_waiting_tasks": longest,
        "eligibility_to_selection": {
            **_summary(eligibility_to_selection),
            "excluded_count": eligible_excluded,
        },
        "urgency": {
            "nonzero_deadline_count": deadline_tasks,
            "selected_after_deadline_count": selected_after_deadline,
            "selected_after_deadline_rate": deadline_rate,
            "lateness": _summary(late_values),
            "raw_deadline_inversion_count": raw_inversions,
            "raw_deadline_inversion_rate": inversion_rate,
        },
        "intervention": {
            "reserved_selection_count": counts["source_reserved"],
            "reserved_selection_share": counts["source_reserved"] / len(run.slots) if run.slots else None,
            "hard_urgent_selection_count": counts["source_hard_urgent"],
            "legacy_selection_count": counts["source_legacy"],
            "waiting_eligible_count": counts["waiting_eligible"],
            "no_eligible_count": counts["no_eligible"],
            "debt_pending_count": counts["debt_pending"],
            "debt_created_count": counts["debt_created"],
            "debt_repaid_count": counts["debt_repaid"],
            "hard_urgent_override_count": counts["hard_urgent_override"],
            "longest_reserved_streak": _longest_streak(selected_sources, "reserved"),
            "longest_legacy_streak": _longest_streak(selected_sources, "legacy"),
        },
        "safety": {
            "hard_urgent_violation": hard_urgent_violations,
            "reserved_state_machine_violation": state_violations,
            "stale_handle_count": stale,
            "full_store_scan_count": queue.full_store_scan_count,
            "task_conservation_error": conservation_error,
            "queued_bytes_error": int(queue.store.queued_bytes != 0),
            "store_index_registry_error": int(not bool(audit["passed"])),
            "duplicate_remove_count": queue.store.duplicate_remove_count,
            "generation_mismatch_count": queue.store.generation_mismatch_count,
            "s0_oracle_winner_mismatches": s0_oracle_mismatches,
            "b0_runtime_winner_mismatches": b0_runtime_mismatches,
            "nondeterministic_rerun": 0,
            "invariants_passed": bool(audit["passed"]),
        },
        "operation_counters": queue.counters(),
        "final_policy_state": {
            "credit": queue.policy_state.credit,
            "pending_debt": queue.policy_state.pending_debt,
        },
        **COUNTERFACTUAL,
    }
    passed, blockers = _safety_pass(result)
    result["safety_passed"] = passed
    result["blocking_reasons"] = blockers
    return result


def replay_case(
    run: EvidenceRun,
    policy_config: PreregisteredPolicyConfig | None,
) -> dict[str, Any]:
    """Replay twice and fail the candidate if the deterministic digest differs."""

    first = _run_once(run, policy_config)
    second = _run_once(run, policy_config)
    deterministic = (
        first["decision_digest_sha256"] == second["decision_digest_sha256"]
        and first["task_metrics"] == second["task_metrics"]
        and first["operation_counters"] == second["operation_counters"]
        and first["final_policy_state"] == second["final_policy_state"]
    )
    first["deterministic_rerun"] = {
        "passed": deterministic,
        "first_digest_sha256": first["decision_digest_sha256"],
        "second_digest_sha256": second["decision_digest_sha256"],
    }
    first["safety"]["nondeterministic_rerun"] = int(not deterministic)
    passed, blockers = _safety_pass(first)
    first["safety_passed"] = passed
    first["blocking_reasons"] = blockers
    return first


def _delta(new: int | float | None, baseline: int | float | None) -> int | float | None:
    return None if new is None or baseline is None else new - baseline


def pair_results(s0: dict[str, Any], s1: dict[str, Any]) -> dict[str, Any]:
    s0_tasks = {item["task_id"]: item for item in s0["task_metrics"]}
    s1_tasks = {item["task_id"]: item for item in s1["task_metrics"]}
    if set(s0_tasks) != set(s1_tasks):
        s1["safety"]["task_conservation_error"] += 1
        passed, blockers = _safety_pass(s1)
        s1["safety_passed"] = passed
        s1["blocking_reasons"] = blockers
        paired_task_deltas: list[dict[str, Any]] = []
    else:
        paired_task_deltas = [
            {
                "task_id": task_id,
                "queue_wait_delta_ns": s1_tasks[task_id]["queue_wait_ns"] - s0_tasks[task_id]["queue_wait_ns"],
            }
            for task_id in sorted(s0_tasks)
        ]
    changes = Counter(
        "improved" if item["queue_wait_delta_ns"] < 0 else "worsened" if item["queue_wait_delta_ns"] > 0 else "equal"
        for item in paired_task_deltas
    )
    result = deepcopy(s1)
    result["paired_against_s0"] = {
        "queue_wait_p50_delta_ns": _delta(s1["queue_wait"]["p50"], s0["queue_wait"]["p50"]),
        "queue_wait_p95_delta_ns": _delta(s1["queue_wait"]["p95"], s0["queue_wait"]["p95"]),
        "queue_wait_p99_delta_ns": _delta(s1["queue_wait"]["p99"], s0["queue_wait"]["p99"]),
        "queue_wait_max_delta_ns": _delta(s1["queue_wait"]["max"], s0["queue_wait"]["max"]),
        "worst_1pct_mean_delta_ns": _delta(
            s1["queue_wait"]["worst_1pct_mean"], s0["queue_wait"]["worst_1pct_mean"]
        ),
        "selected_after_deadline_rate_delta": _delta(
            s1["urgency"]["selected_after_deadline_rate"],
            s0["urgency"]["selected_after_deadline_rate"],
        ),
        "raw_deadline_inversion_rate_delta": _delta(
            s1["urgency"]["raw_deadline_inversion_rate"],
            s0["urgency"]["raw_deadline_inversion_rate"],
        ),
        "lateness_p95_delta_ns": _delta(
            s1["urgency"]["lateness"]["p95"], s0["urgency"]["lateness"]["p95"]
        ),
        "lateness_p99_delta_ns": _delta(
            s1["urgency"]["lateness"]["p99"], s0["urgency"]["lateness"]["p99"]
        ),
        "lateness_max_delta_ns": _delta(
            s1["urgency"]["lateness"]["max"], s0["urgency"]["lateness"]["max"]
        ),
        "task_change_counts": dict(changes),
        "task_deltas": paired_task_deltas,
    }
    result["s0_metrics"] = {
        "queue_wait": s0["queue_wait"],
        "urgency": s0["urgency"],
        "decision_digest_sha256": s0["decision_digest_sha256"],
    }
    return result


def _material_worse(new: float | int | None, old: float | int | None, absolute: float, relative: float) -> bool:
    if new is None and old is None:
        return False
    if new is None:
        return False
    if old is None:
        return True
    delta = new - old
    return delta > absolute and ((old > 0 and delta / old > relative) or (old == 0 and delta > 0))


def _candidate_aggregate(
    candidate_id: str,
    paired_runs: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    by_worker: dict[str, Any] = {}
    for workers in (2, 4):
        runs = [item for item in paired_runs if item["workers"] == workers]
        by_worker[str(workers)] = {
            "N": len(runs),
            "p95_delta_median_ns": _median(item["paired_against_s0"]["queue_wait_p95_delta_ns"] for item in runs),
            "p99_delta_median_ns": _median(item["paired_against_s0"]["queue_wait_p99_delta_ns"] for item in runs),
            "lateness_p95_delta_median_ns": _median(item["paired_against_s0"]["lateness_p95_delta_ns"] for item in runs),
            "reserved_share_median": _median(item["intervention"]["reserved_selection_share"] for item in runs),
            "p95_improving_runs": sum(item["paired_against_s0"]["queue_wait_p95_delta_ns"] < 0 for item in runs),
            "raw_run_ids": [item["run_id"] for item in runs],
        }
    return {
        "candidate_id": candidate_id,
        "parameters": deepcopy(candidate),
        "workers": by_worker,
    }


def _tie_value(value: int | float | None) -> float:
    return math.inf if value is None else float(value)


def _tie_key(aggregate: dict[str, Any]) -> tuple[float, float, float, float, int, str]:
    workers = aggregate["workers"]
    candidate = aggregate["parameters"]
    return (
        max(_tie_value(workers["2"]["p95_delta_median_ns"]), _tie_value(workers["4"]["p95_delta_median_ns"])),
        max(_tie_value(workers["2"]["p99_delta_median_ns"]), _tie_value(workers["4"]["p99_delta_median_ns"])),
        max(_tie_value(workers["2"]["lateness_p95_delta_median_ns"]), _tie_value(workers["4"]["lateness_p95_delta_median_ns"])),
        max(_tie_value(workers["2"]["reserved_share_median"]), _tie_value(workers["4"]["reserved_share_median"])),
        -int(candidate["eligibility_age_ns"]),
        str(aggregate["candidate_id"]),
    )


def _budget_gate(paired_runs: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for item in paired_runs:
        run_id = item["run_id"]
        s0q, s1q = item["s0_metrics"]["queue_wait"], item["queue_wait"]
        for field, absolute, relative in (
            ("p99", 10_000_000, 0.25),
            ("worst_1pct_mean", 5_000_000, 0.20),
            ("max", 10_000_000, 0.25),
        ):
            if _material_worse(s1q[field], s0q[field], absolute, relative):
                failures.append(f"{run_id}:fairness:{field}")
        s0u, s1u = item["s0_metrics"]["urgency"], item["urgency"]
        for field, absolute, relative in (
            ("selected_after_deadline_rate", 0.0005, 0.05),
            ("raw_deadline_inversion_rate", 0.0005, 0.05),
        ):
            if _material_worse(s1u[field], s0u[field], absolute, relative):
                failures.append(f"{run_id}:urgency:{field}")
        for field, absolute, relative in (
            ("p95", 1_000_000, 0.05),
            ("p99", 2_000_000, 0.10),
            ("max", 10_000_000, 0.25),
        ):
            if _material_worse(s1u["lateness"][field], s0u["lateness"][field], absolute, relative):
                failures.append(f"{run_id}:urgency:lateness_{field}")
    for workers in (2, 4):
        subset = [item for item in paired_runs if item["workers"] == workers]
        s0_median = _median(item["s0_metrics"]["queue_wait"]["p99"] for item in subset)
        s1_median = _median(item["queue_wait"]["p99"] for item in subset)
        if _material_worse(s1_median, s0_median, 2_000_000, 0.10):
            failures.append(f"workers={workers}:fairness:p99_median")
    return failures


def evaluate_candidate_gate(
    aggregate: dict[str, Any],
    paired_runs: list[dict[str, Any]],
    *,
    stage: str,
) -> dict[str, Any]:
    failures: list[str] = []
    for item in paired_runs:
        passed, blockers = _safety_pass(item)
        if not passed:
            failures.extend(f"{item['run_id']}:safety:{name}" for name in blockers)
        total = item["decision_count"]
        intervention = item["intervention"]
        share_limit = item["parameters"]["R"] / item["parameters"]["D"] + max(0.005, 1 / total)
        if intervention["reserved_selection_count"] <= 0:
            failures.append(f"{item['run_id']}:intervention:zero_reserved")
        if intervention["longest_reserved_streak"] > 1:
            failures.append(f"{item['run_id']}:intervention:reserved_streak")
        if intervention["reserved_selection_share"] > share_limit:
            failures.append(f"{item['run_id']}:intervention:share_too_wide")
    failures.extend(_budget_gate(paired_runs))
    needed = 2 if stage == "calibration" else 3
    required_improvements = 2
    for workers in (2, 4):
        summary = aggregate["workers"][str(workers)]
        if summary["N"] != needed:
            failures.append(f"workers={workers}:unexpected_N")
        if summary["p95_delta_median_ns"] is None or summary["p95_delta_median_ns"] >= 0:
            failures.append(f"workers={workers}:p95_median_not_improved")
        if summary["p95_improving_runs"] < required_improvements:
            failures.append(f"workers={workers}:p95_run_support")
    return {
        "passed": not failures,
        "failures": failures,
        "stage": stage,
        "aggregate": aggregate,
    }


def select_finalists(
    candidate_results: dict[str, list[dict[str, Any]]],
    candidates: dict[str, dict[str, Any]],
    *,
    stage: str,
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for candidate_id, runs in candidate_results.items():
        aggregate = _candidate_aggregate(candidate_id, runs, candidates[candidate_id])
        gates[candidate_id] = evaluate_candidate_gate(aggregate, runs, stage=stage)
    if stage == "calibration":
        pairs = (
            ("S1-C", "S1-A", "medium_share_lower_than_low_share"),
            ("S1-D", "S1-B", "medium_share_lower_than_low_share"),
            ("S1-A", "S1-B", "moderate_age_lower_than_sparse_age"),
            ("S1-C", "S1-D", "moderate_age_lower_than_sparse_age"),
        )
        for workers in (2, 4):
            for expected_higher, reference, reason in pairs:
                higher = gates[expected_higher]["aggregate"]["workers"][str(workers)]["reserved_share_median"]
                lower = gates[reference]["aggregate"]["workers"][str(workers)]["reserved_share_median"]
                decisions = candidate_results[expected_higher][0]["decision_count"]
                tolerance = max(0.005, 1 / decisions)
                if higher is None or lower is None or higher + tolerance < lower:
                    gates[expected_higher]["passed"] = False
                    gates[expected_higher]["failures"].append(f"workers={workers}:direction:{reason}")
    passing = [gate["aggregate"] for gate in gates.values() if gate["passed"]]
    ordered = sorted(passing, key=_tie_key)
    selected = [item["candidate_id"] for item in ordered[:2]] if stage == "calibration" else (
        [ordered[0]["candidate_id"]] if ordered else []
    )
    return {
        "stage": stage,
        "candidate_gates": gates,
        "passing_candidates_in_frozen_order": [item["candidate_id"] for item in ordered],
        "selected_candidates": selected,
        "maximum_finalists": 2 if stage == "calibration" else 1,
        "tie_break": EXPECTED_TIE_BREAK,
    }


class EvidenceRunProvider:
    """Lazily opens only the current preregistered role from A.3 Evidence."""

    def __init__(self, repo_root: Path, preregistration: dict[str, Any]) -> None:
        self.repo_root = repo_root.resolve()
        source = preregistration["source_evidence_index"]
        self.evidence_index_path = self.repo_root / source["path"]
        if sha256_file(self.evidence_index_path) != source["sha256"]:
            raise M6CDGateError("A.3 Evidence index SHA mismatch")
        validation = validate_evidence_index(self.repo_root, self.evidence_index_path)
        if not validation.get("passed"):
            raise M6CDGateError("A.3 Evidence artifact Hash validation failed")
        run_index_path = self.repo_root / "llama.cpp/trace_output/m6b2a3_directed_20260804_n5_v3_run_index.json"
        self.entries = {item["run_id"]: item for item in load_run_index(run_index_path)}
        self.cache: dict[str, EvidenceRun] = {}
        self.opened_roles: Counter[str] = Counter()

    def load(self, role: str, record: dict[str, Any]) -> EvidenceRun:
        run_id = record["run_id"]
        self.opened_roles[role] += 1
        if run_id not in self.cache:
            result = audit_run(self.entries[run_id], global_hash_passed=True)
            if result.status != ReconstructabilityStatus.RECONSTRUCTABLE or result.run is None:
                raise M6CDGateError(f"{run_id}: {result.status.value}")
            self.cache[run_id] = result.run
        return self.cache[run_id]


RunLoader = Callable[[str, dict[str, Any]], EvidenceRun]


def _stage_streams(
    output_dir: Path,
    stage_dir: Path,
    runs: list[EvidenceRun],
    candidates: list[str],
    candidate_map: dict[str, dict[str, Any]],
    preregistration_sha256: str,
    *,
    synthetic: bool,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    stream_dir = stage_dir / "decision_streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        for candidate_id in candidates:
            config = _policy_config(candidate_map[candidate_id], preregistration_sha256, synthetic=synthetic)
            path = stream_dir / f"{run.run_id}__{candidate_id}.jsonl"

            def write(stream: TextIO, current_run: EvidenceRun = run, current=config) -> dict[str, Any]:
                return _run_once(current_run, current, decision_stream=stream)

            result, metadata = write_finalized_jsonl(
                path,
                write,
                expected_line_count=len(run.slots),
            )
            if result["decision_digest_sha256"] != metadata["sha256"]:
                raise M6CDGateError("post-close decision stream SHA differs from replay digest")
            metadata["path"] = str(path.relative_to(output_dir))
            metadata["run_id"] = run.run_id
            metadata["candidate_id"] = candidate_id
            artifacts.append(metadata)
    return artifacts


def _finalize_stage(
    output_dir: Path,
    stage_dir: Path,
    results: dict[str, Any],
    selection: dict[str, Any],
    streams: list[dict[str, Any]],
) -> dict[str, Any]:
    results_meta = _write_closed_json(stage_dir / "results.json", results)
    selection_name = {
        "calibration": "calibration_selection.json",
        "holdout": "holdout_selection.json",
        "robustness": "robustness_conclusion.json",
    }[selection["stage"]]
    selection_meta = _write_closed_json(stage_dir / selection_name, selection)
    for meta in (results_meta, selection_meta):
        meta["path"] = str(Path(meta["path"]).relative_to(output_dir))
    manifest = {
        "schema_version": "m6c-d-stage-artifact-manifest-v1",
        "stage": selection["stage"],
        "artifacts": [results_meta, selection_meta, *streams],
        "hashes_computed_after_close": True,
        "durable_fsync_requested": False,
        "durability_claim": "close-and-reopen integrity only",
    }
    manifest_meta = _write_closed_json(stage_dir / "artifact_manifest.json", manifest)
    manifest_meta["path"] = str(Path(manifest_meta["path"]).relative_to(output_dir))
    from .m6c_d_validator import validate_stage_manifest

    validation = validate_stage_manifest(output_dir, stage_dir / "artifact_manifest.json")
    if not validation["passed"]:
        raise M6CDGateError(f"{selection['stage']} artifact validation failed")
    validation_meta = _write_closed_json(stage_dir / "independent_validation.json", validation)
    validation_meta["path"] = str(Path(validation_meta["path"]).relative_to(output_dir))
    return {
        "manifest": manifest_meta,
        "validation": validation_meta,
        "selection": selection_meta,
        "stream_count": len(streams),
    }


def execute_pipeline(
    preregistration: dict[str, Any],
    preregistration_sha256: str,
    output_dir: Path,
    run_loader: RunLoader,
    *,
    synthetic: bool,
) -> dict[str, Any]:
    """Execute the three frozen stages; later roles are loaded only after their gate."""

    validate_frozen_contract(preregistration, require_ready=not synthetic)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates = _candidate_map(preregistration)
    split = preregistration["run_split"]
    stage_records: dict[str, Any] = {}

    calibration_runs = [run_loader("calibration", record) for record in split["calibration"]]
    s0_calibration = {run.run_id: replay_case(run, None) for run in calibration_runs}
    for run_id, result in s0_calibration.items():
        if not result["safety_passed"]:
            raise M6CDGateError(f"{run_id}: S0 correctness/safety Gate failed")
    calibration_results: dict[str, list[dict[str, Any]]] = {candidate_id: [] for candidate_id in candidates}
    for run in calibration_runs:
        for candidate_id, candidate in candidates.items():
            result = replay_case(
                run,
                _policy_config(candidate, preregistration_sha256, synthetic=synthetic),
            )
            calibration_results[candidate_id].append(pair_results(s0_calibration[run.run_id], result))
    calibration_selection = select_finalists(calibration_results, candidates, stage="calibration")
    finalists = calibration_selection["selected_candidates"]
    calibration_streams = _stage_streams(
        output_dir,
        output_dir / "01_calibration",
        calibration_runs,
        finalists,
        candidates,
        preregistration_sha256,
        synthetic=synthetic,
    )
    stage_records["calibration"] = _finalize_stage(
        output_dir,
        output_dir / "01_calibration",
        {
            "s0": s0_calibration,
            "candidates": calibration_results,
            "test_fixture_only": synthetic,
            **COUNTERFACTUAL,
        },
        calibration_selection,
        calibration_streams,
    )
    calibration_manifest_sha = sha256_file(output_dir / "01_calibration/artifact_manifest.json")

    if not finalists:
        report = {
            "schema_version": RUNNER_SCHEMA,
            "final_enum": "M6C_ROUTE_STOP_RECOMMENDED",
            "finalists": [],
            "unique_recommendation": None,
            "robustness_executed": False,
            "holdout_runs_loaded": 0,
            "robustness_runs_loaded": 0,
            "test_fixture_only": synthetic,
            **COUNTERFACTUAL,
        }
        return _finalize_root(output_dir, report, stage_records)

    if sha256_file(output_dir / "01_calibration/artifact_manifest.json") != calibration_manifest_sha:
        raise M6CDGateError("calibration artifacts changed before holdout")
    holdout_runs = [run_loader("holdout", record) for record in split["holdout"]]
    s0_holdout = {run.run_id: replay_case(run, None) for run in holdout_runs}
    for run_id, result in s0_holdout.items():
        if not result["safety_passed"]:
            raise M6CDGateError(f"{run_id}: S0 correctness/safety Gate failed")
    holdout_results: dict[str, list[dict[str, Any]]] = {candidate_id: [] for candidate_id in finalists}
    for run in holdout_runs:
        for candidate_id in finalists:
            result = replay_case(
                run,
                _policy_config(candidates[candidate_id], preregistration_sha256, synthetic=synthetic),
            )
            holdout_results[candidate_id].append(pair_results(s0_holdout[run.run_id], result))
    holdout_selection = select_finalists(holdout_results, candidates, stage="holdout")
    recommendation = holdout_selection["selected_candidates"]
    unique_recommendation = recommendation[0] if len(recommendation) == 1 else None
    holdout_streams = _stage_streams(
        output_dir,
        output_dir / "02_holdout",
        holdout_runs,
        finalists,
        candidates,
        preregistration_sha256,
        synthetic=synthetic,
    )
    stage_records["holdout"] = _finalize_stage(
        output_dir,
        output_dir / "02_holdout",
        {"s0": s0_holdout, "candidates": holdout_results, "test_fixture_only": synthetic, **COUNTERFACTUAL},
        holdout_selection,
        holdout_streams,
    )
    holdout_manifest_sha = sha256_file(output_dir / "02_holdout/artifact_manifest.json")

    robustness_count = 0
    robustness_safety_passed = True
    if unique_recommendation is not None:
        if sha256_file(output_dir / "01_calibration/artifact_manifest.json") != calibration_manifest_sha:
            raise M6CDGateError("calibration artifacts changed before robustness")
        if sha256_file(output_dir / "02_holdout/artifact_manifest.json") != holdout_manifest_sha:
            raise M6CDGateError("holdout artifacts changed before robustness")
        robustness_runs = [run_loader("robustness", record) for record in split["robustness"]]
        robustness_count = len(robustness_runs)
        s0_robustness = {run.run_id: replay_case(run, None) for run in robustness_runs}
        for run_id, result in s0_robustness.items():
            if not result["safety_passed"]:
                raise M6CDGateError(f"{run_id}: S0 correctness/safety Gate failed")
        robustness_results = []
        for run in robustness_runs:
            result = replay_case(
                run,
                _policy_config(candidates[unique_recommendation], preregistration_sha256, synthetic=synthetic),
            )
            robustness_results.append(pair_results(s0_robustness[run.run_id], result))
        robustness_safety_passed = all(item["safety_passed"] for item in robustness_results)
        robustness_selection = {
            "stage": "robustness",
            "candidate_id": unique_recommendation,
            "run_count": robustness_count,
            "all_safety_passed": robustness_safety_passed,
            "can_select_or_change_parameters": False,
            "primary_n30_claim": False,
        }
        robustness_streams = _stage_streams(
            output_dir,
            output_dir / "03_robustness_appendix",
            robustness_runs,
            [unique_recommendation],
            candidates,
            preregistration_sha256,
            synthetic=synthetic,
        )
        stage_records["robustness"] = _finalize_stage(
            output_dir,
            output_dir / "03_robustness_appendix",
            {
                "s0": s0_robustness,
                "candidate": robustness_results,
                "test_fixture_only": synthetic,
                **COUNTERFACTUAL,
            },
            robustness_selection,
            robustness_streams,
        )

    report = {
        "schema_version": RUNNER_SCHEMA,
        "final_enum": (
            "SYNTHETIC_THREE_STAGE_PASS" if synthetic and robustness_safety_passed
            else "M6C_D_REPLAY_COMPLETE" if robustness_safety_passed
            else "M6C_D_CORRECTNESS_OR_SAFETY_FAILURE"
        ),
        "finalists": finalists,
        "unique_recommendation": unique_recommendation,
        "robustness_executed": unique_recommendation is not None,
        "calibration_runs_loaded": len(calibration_runs),
        "holdout_runs_loaded": len(holdout_runs),
        "robustness_runs_loaded": robustness_count,
        "test_fixture_only": synthetic,
        "experimental_parameter": not synthetic,
        "formal_s1_evidence_replay_executed": not synthetic,
        **COUNTERFACTUAL,
    }
    return _finalize_root(output_dir, report, stage_records)


def _finalize_root(output_dir: Path, report: dict[str, Any], stages: dict[str, Any]) -> dict[str, Any]:
    report_meta = _write_closed_json(output_dir / "m6c_d_report.json", report)
    report_meta["path"] = str(Path(report_meta["path"]).relative_to(output_dir))
    root_manifest = {
        "schema_version": "m6c-d-root-artifact-manifest-v1",
        "report": report_meta,
        "stages": stages,
        "hashes_computed_after_close": True,
    }
    _write_closed_json(output_dir / "artifact_manifest.json", root_manifest)
    from .m6c_d_validator import validate_output

    validation = validate_output(output_dir)
    if not validation["passed"]:
        raise M6CDGateError("root artifact validation failed")
    _write_closed_json(output_dir / "independent_validation.json", validation)
    report["artifact_validation_passed"] = True
    report["output_dir"] = str(output_dir)
    return report


def _bits(value: float) -> str:
    return "0x" + struct.pack(">d", value).hex()


def build_synthetic_runs(preregistration: dict[str, Any], *, decisions: int = 64) -> dict[str, EvidenceRun]:
    """Create a complete in-memory 30-Run fixture without reading A.3 files."""

    if decisions < 32:
        raise ValueError("synthetic fixture requires at least 32 decisions")
    result: dict[str, EvidenceRun] = {}
    for role in ("calibration", "holdout", "robustness"):
        for entry in preregistration["run_split"][role]:
            run_id = entry["run_id"]
            workers = int(entry["workers"])
            tasks = [
                TaskSpec(
                    run_id=run_id,
                    task_id=index + 1,
                    deadline_ts_ns=1_000_000_000,
                    route_score_f64_bits=_bits(float(index + 1)),
                    sequence=index,
                    enqueued_ts_ns=(index + 1) * 1_000_000,
                    nbytes=4096,
                    stage="LATE",
                    phase="PREFILL",
                    layer=0,
                )
                for index in range(decisions)
            ]
            slots = [
                ServiceSlot(
                    decision_id=index,
                    batch_id=index,
                    batch_slot=0,
                    worker_id=index % workers,
                    decision_ts_ns=(100 + index) * 1_000_000,
                    winner_task_id=decisions - index,
                    queue_depth_before=decisions - index,
                    priority_mode="deadline_score",
                    diagnostic_event="SYNTHETIC",
                    diagnostic_decision_ts_ns=(100 + index) * 1_000_000,
                    diagnostic_candidate_count=decisions - index,
                )
                for index in range(decisions)
            ]
            result[run_id] = EvidenceRun(
                run_id=run_id,
                evidence_dir=Path("synthetic://m6c-d"),
                configuration_id=entry["configuration_id"],
                workers=workers,
                repeat_index=int(entry["repeat_index"]),
                priority_mode="deadline_score",
                capacity=decisions + 1,
                tasks=tasks,
                slots=slots,
                lifecycle_counts={"CREATE": decisions, "ENQUEUE": decisions, "DEQUEUE": decisions},
                terminal_states={task.task_id: "ISSUED" for task in tasks},
                prior_validation={"synthetic": True},
            )
    return result


def _load_closed_preregistration(path: Path, *, require_detached_sha: bool) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    digest = sha256_file(path)
    if require_detached_sha:
        detached_path = path.with_suffix(".sha256")
        fields = detached_path.read_text(encoding="utf-8").strip().split()
        if len(fields) != 2 or fields[0] != digest or fields[1] != path.name:
            raise M6CDGateError("preregistration detached SHA mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    return value, digest


def execute_formal(repo_root: Path, preregistration_path: Path, output_dir: Path) -> dict[str, Any]:
    preregistration, digest = _load_closed_preregistration(preregistration_path, require_detached_sha=True)
    validate_frozen_contract(preregistration, require_ready=True)
    expected_source = preregistration.get("expected_m6c_d_command", {}).get("module_source_sha256")
    if expected_source != sha256_file(Path(__file__)):
        raise M6CDGateError("runner source SHA differs from ready preregistration")
    provider = EvidenceRunProvider(repo_root, preregistration)
    return execute_pipeline(
        preregistration,
        digest,
        output_dir,
        provider.load,
        synthetic=False,
    )


def execute_synthetic_self_audit(preregistration_path: Path, output_dir: Path) -> dict[str, Any]:
    preregistration, digest = _load_closed_preregistration(preregistration_path, require_detached_sha=True)
    validate_frozen_contract(preregistration, require_ready=False)
    runs = build_synthetic_runs(preregistration)
    loaded: Counter[str] = Counter()

    def loader(role: str, record: dict[str, Any]) -> EvidenceRun:
        loaded[role] += 1
        return runs[record["run_id"]]

    report = execute_pipeline(preregistration, digest, output_dir, loader, synthetic=True)
    report["input_access_audit"] = {
        "a3_evidence_opened": False,
        "a3_holdout_evidence_opened": False,
        "a3_robustness_evidence_opened": False,
        "synthetic_runs_loaded_by_role": dict(loaded),
    }
    return report


def create_preregistration_v2(
    v1_dir: Path,
    v2_dir: Path,
    *,
    test_audit: dict[str, Any],
    synthetic_audit: dict[str, Any],
) -> dict[str, Any]:
    """Promote v1 by resolving only its M6C-D command gate."""

    v1_path = v1_dir.resolve() / "m6c_c_preregistration.json"
    v1, v1_sha = _load_closed_preregistration(v1_path, require_detached_sha=True)
    validate_frozen_contract(v1, require_ready=False)
    if v1.get("final_enum") != "METRIC_OR_GATE_UNRESOLVED":
        raise M6CDGateError("v1 is not the expected command-gate-only historical record")
    if not test_audit.get("passed") or not synthetic_audit.get("passed"):
        raise M6CDGateError("runner test/audit must pass before v2")
    runner_sha = sha256_file(Path(__file__))
    v2 = deepcopy(v1)
    output_relative = "llama.cpp/trace_output/m6c_d_reserved_service_calibration_holdout_v2"
    prereg_relative = "llama.cpp/trace_output/m6c_c_preregistration_20260805_v2/m6c_c_preregistration.json"
    command = v2["expected_m6c_d_command"]
    command.update({
        "command": (
            "PYTHONPATH=llama.cpp/trace python3 -m m6c_offline.m6c_d_runner "
            f"--preregistration {prereg_relative} --output-dir {output_relative}"
        ),
        "module_available": True,
        "module_source_sha256": runner_sha,
        "output_dir": output_relative,
        "unresolved_reason": None,
    })
    v2["command_gate_resolved"] = True
    v2["runner_test_and_audit"] = {
        "base_v1_path": str(v1_path),
        "base_v1_sha256": v1_sha,
        "runner_path": str(Path(__file__).resolve()),
        "runner_source_sha256": runner_sha,
        "independent_validator_path": str((Path(__file__).parent / "m6c_d_validator.py").resolve()),
        "independent_validator_sha256": sha256_file(Path(__file__).parent / "m6c_d_validator.py"),
        "runner_test_path": str(
            (Path(__file__).parents[1] / "tests/test_m6c_offline_m6c_d.py").resolve()
        ),
        "runner_test_sha256": sha256_file(
            Path(__file__).parents[1] / "tests/test_m6c_offline_m6c_d.py"
        ),
        "tests": test_audit,
        "synthetic_three_stage": synthetic_audit,
        "a3_s1_replay_executed": False,
    }
    v2["final_enum"] = FINAL_READY
    v2_dir = v2_dir.resolve()
    v2_dir.mkdir(parents=True, exist_ok=False)
    json_meta = _write_closed_json(v2_dir / "m6c_c_preregistration.json", v2)
    markdown = "\n".join([
        "# M6C-C Reserved-Service Replay Preregistration v2",
        "",
        f"- v1 SHA256: `{v1_sha}`",
        f"- runner SHA256: `{runner_sha}`",
        "- command_gate_resolved: `true`",
        f"- final enum: `{FINAL_READY}`",
        "- v1 parameters, Run split, metrics, budgets, Gates, and tie-break are inherited unchanged.",
        "- Synthetic-only three-stage audit passed; no A.3 S1 replay was executed.",
        "",
    ])
    markdown_meta = _write_closed_text(v2_dir / "m6c_c_preregistration.md", markdown)
    sha_meta = _write_closed_text(
        v2_dir / "m6c_c_preregistration.sha256",
        f"{json_meta['sha256']}  m6c_c_preregistration.json\n",
    )
    for meta in (json_meta, markdown_meta, sha_meta):
        meta["path"] = str(Path(meta["path"]).relative_to(v2_dir))
    manifest = {
        "schema_version": "m6c-c-preregistration-artifact-manifest-v2",
        "artifacts": [json_meta, markdown_meta, sha_meta],
        "hashes_computed_after_close": True,
        "durability_claim": "close-and-reopen integrity only",
    }
    _write_closed_json(v2_dir / "artifact_manifest.json", manifest)
    from .m6c_d_validator import validate_preregistration_v2

    validation = validate_preregistration_v2(v1_dir.resolve(), v2_dir, Path(__file__).resolve())
    if not validation["passed"]:
        raise M6CDGateError("independent v2 schema/Hash/frozen inheritance validation failed")
    _write_closed_json(v2_dir / "independent_validation.json", validation)
    return {
        "path": str(v2_dir / "m6c_c_preregistration.json"),
        "sha256": json_meta["sha256"],
        "runner_sha256": runner_sha,
        "final_enum": FINAL_READY,
        "validation_passed": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synthetic-self-audit", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.synthetic_self_audit:
        result = execute_synthetic_self_audit(Path(args.preregistration), Path(args.output_dir))
    else:
        result = execute_formal(Path(args.repo_root), Path(args.preregistration), Path(args.output_dir))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
