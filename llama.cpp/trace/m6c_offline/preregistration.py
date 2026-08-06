#!/usr/bin/env python3
"""Generate M6C-C parameters from frozen S0 inputs without executing S1."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .evidence import COUNTERFACTUAL_DECLARATION, sha256_file


SCHEMA_VERSION = "m6c-c-reserved-service-preregistration-v1"
STAGE = "M6C-C Reserved-Service Replay Preregistration"
ROUNDING_QUANTUM_NS = 1_000_000
EXPECTED_DECISIONS = 29_262
M6C_D_MODULE = "m6c_offline.m6c_d_runner"

AUTHORITY_SHA256 = {
    "llama.cpp/trace_output/m6c_b_offline_20260805_a3_s0_v4/m6c_b_report.md":
        "16dd06777b8c612b58a557e7f80bf2907492a6d57668b22d171c49cdfbe8652a",
    "llama.cpp/trace_output/m6c_b_offline_20260805_a3_s0_v4/m6c_b_report.json":
        "3952ed0fe481245ef5f654047aa41b0b62173423cbc69e2ed7afbeb63dbed43c",
    "llama.cpp/trace_output/m6c_b_offline_20260805_a3_s0_v4/s0_replay_report.json":
        "76824ef118ea29ff800ba43eb93e873694ffa0ba9678b5d1e381c1ed644ee2bf",
    "llama.cpp/trace_output/m6c_b_offline_20260805_a3_s0_v4/artifact_validation_report.json":
        "8fc858a4ad665d43e09196052a808252a5cc1a741bbe7ede23ecccd89eb90699",
    "llama.cpp/trace_output/m6b2a3_directed_20260804_n5_v3_analysis_v2/evidence_index.json":
        "03af8c334ef9f3591b6318c44f0132894349f9b5dd89b89e86a1181e13a5a66e",
}

CALIBRATION = (
    ("m6b2a3_directed_20260804_n5_v3_p01_w2_r1_B0_a1", 2, 1, "B0"),
    ("m6b2a3_directed_20260804_n5_v3_p12_w2_r2_B0_a1", 2, 2, "B0"),
    ("m6b2a3_directed_20260804_n5_v3_p04_w4_r1_B0_a1", 4, 1, "B0"),
    ("m6b2a3_directed_20260804_n5_v3_p07_w4_r2_B0_a1", 4, 2, "B0"),
)
HOLDOUT = (
    ("m6b2a3_directed_20260804_n5_v3_p13_w2_r3_B0_a1", 2, 3, "B0"),
    ("m6b2a3_directed_20260804_n5_v3_p23_w2_r4_B0_a1", 2, 4, "B0"),
    ("m6b2a3_directed_20260804_n5_v3_p26_w2_r5_B0_a1", 2, 5, "B0"),
    ("m6b2a3_directed_20260804_n5_v3_p18_w4_r3_B0_a1", 4, 3, "B0"),
    ("m6b2a3_directed_20260804_n5_v3_p20_w4_r4_B0_a1", 4, 4, "B0"),
    ("m6b2a3_directed_20260804_n5_v3_p30_w4_r5_B0_a1", 4, 5, "B0"),
)
ROBUSTNESS = (
    ("m6b2a3_directed_20260804_n5_v3_p02_w2_r1_C1_a1", 2, 1, "C1"),
    ("m6b2a3_directed_20260804_n5_v3_p03_w2_r1_C2_a1", 2, 1, "C2"),
    ("m6b2a3_directed_20260804_n5_v3_p05_w4_r1_C1_a1", 4, 1, "C1"),
    ("m6b2a3_directed_20260804_n5_v3_p06_w4_r1_C2_a1", 4, 1, "C2"),
    ("m6b2a3_directed_20260804_n5_v3_p08_w4_r2_C1_a1", 4, 2, "C1"),
    ("m6b2a3_directed_20260804_n5_v3_p09_w4_r2_C2_a1", 4, 2, "C2"),
    ("m6b2a3_directed_20260804_n5_v3_p10_w2_r2_C1_a1", 2, 2, "C1"),
    ("m6b2a3_directed_20260804_n5_v3_p11_w2_r2_C2_a1", 2, 2, "C2"),
    ("m6b2a3_directed_20260804_n5_v3_p14_w2_r3_C2_a1", 2, 3, "C2"),
    ("m6b2a3_directed_20260804_n5_v3_p15_w2_r3_C1_a1", 2, 3, "C1"),
    ("m6b2a3_directed_20260804_n5_v3_p16_w4_r3_C1_a1", 4, 3, "C1"),
    ("m6b2a3_directed_20260804_n5_v3_p17_w4_r3_C2_a1", 4, 3, "C2"),
    ("m6b2a3_directed_20260804_n5_v3_p19_w4_r4_C2_a1", 4, 4, "C2"),
    ("m6b2a3_directed_20260804_n5_v3_p21_w4_r4_C1_a1", 4, 4, "C1"),
    ("m6b2a3_directed_20260804_n5_v3_p22_w2_r4_C1_a1", 2, 4, "C1"),
    ("m6b2a3_directed_20260804_n5_v3_p24_w2_r4_C2_a1", 2, 4, "C2"),
    ("m6b2a3_directed_20260804_n5_v3_p25_w2_r5_C2_a1", 2, 5, "C2"),
    ("m6b2a3_directed_20260804_n5_v3_p27_w2_r5_C1_a1", 2, 5, "C1"),
    ("m6b2a3_directed_20260804_n5_v3_p28_w4_r5_C1_a1", 4, 5, "C1"),
    ("m6b2a3_directed_20260804_n5_v3_p29_w4_r5_C2_a1", 4, 5, "C2"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def nearest_rank(values: list[int], percentile: float) -> int:
    if not values or not 0 < percentile <= 1:
        raise ValueError("nearest-rank requires nonempty values and 0 < p <= 1")
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def ceil_quantum(value: int, quantum: int = ROUNDING_QUANTUM_NS) -> int:
    if value <= 0 or quantum <= 0:
        raise ValueError("age and quantum must be positive")
    return ((value + quantum - 1) // quantum) * quantum


def derive_ages(distributions: dict[int, list[int]]) -> tuple[dict[str, Any], dict[str, int]]:
    stats: dict[str, Any] = {}
    for workers in (2, 4):
        values = distributions.get(workers, [])
        if not values or any(value < 0 for value in values):
            raise ValueError(f"invalid workers={workers} S0 wait distribution")
        stats[str(workers)] = {
            "sample_count": len(values),
            "minimum_ns": min(values),
            "maximum_ns": max(values),
            "p75_ns": nearest_rank(values, 0.75),
            "p90_ns": nearest_rank(values, 0.90),
        }
    ages = {
        "AGE_MODERATE": ceil_quantum(max(stats["2"]["p75_ns"], stats["4"]["p75_ns"])),
        "AGE_SPARSE": ceil_quantum(max(stats["2"]["p90_ns"], stats["4"]["p90_ns"])),
    }
    if not 0 < ages["AGE_MODERATE"] < ages["AGE_SPARSE"]:
        raise ValueError("derived ages are nonpositive, equal, or reversed after frozen rounding")
    return stats, ages


def _run_record(item: tuple[str, int, int, str]) -> dict[str, Any]:
    run_id, workers, repeat, configuration = item
    return {
        "run_id": run_id,
        "workers": workers,
        "repeat_index": repeat,
        "configuration_id": configuration,
        "expected_service_slots": EXPECTED_DECISIONS,
    }


def validate_run_split(s0_report: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    runs = s0_report.get("runs")
    if not isinstance(runs, list):
        return {"passed": False, "errors": [{"reason": "runs_missing"}]}
    actual = {run.get("run_id"): run for run in runs if isinstance(run, dict)}
    expected_tuples = CALIBRATION + HOLDOUT + ROBUSTNESS
    expected_ids = {item[0] for item in expected_tuples}
    if len(runs) != 30 or set(actual) != expected_ids:
        errors.append({
            "reason": "run_membership_mismatch",
            "expected_count": 30,
            "actual_count": len(runs),
            "missing": sorted(expected_ids - set(actual)),
            "extra": sorted(set(actual) - expected_ids),
        })
    for run_id, workers, repeat, configuration in expected_tuples:
        run = actual.get(run_id)
        if run is None:
            continue
        expected = {
            "workers": workers,
            "repeat_index": repeat,
            "configuration_id": configuration,
            "service_slots": EXPECTED_DECISIONS,
        }
        for field, value in expected.items():
            if run.get(field) != value:
                errors.append({
                    "reason": "run_field_mismatch",
                    "run_id": run_id,
                    "field": field,
                    "expected": value,
                    "actual": run.get(field),
                })
        if configuration == "B0":
            if run.get("source_priority_mode") != "deadline_score":
                errors.append({"reason": "b0_not_deadline_score", "run_id": run_id})
            if run.get("runtime_deadline_score_winner_mismatches") != 0:
                errors.append({"reason": "b0_runtime_mismatch", "run_id": run_id})
    b0_repeats = Counter(
        (run.get("workers"), run.get("repeat_index"))
        for run in runs if run.get("configuration_id") == "B0"
    )
    expected_b0 = Counter((workers, repeat) for workers in (2, 4) for repeat in range(1, 6))
    if b0_repeats != expected_b0:
        errors.append({"reason": "b0_worker_repeat_strata_invalid"})
    return {
        "passed": not errors,
        "errors": errors,
        "actual_run_count": len(runs),
        "expected_run_count": 30,
        "calibration_count": len(CALIBRATION),
        "holdout_count": len(HOLDOUT),
        "robustness_count": len(ROBUSTNESS),
    }


def validate_authority(repo_root: Path) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    for relative, expected_sha in AUTHORITY_SHA256.items():
        path = repo_root / relative
        actual_sha = sha256_file(path) if path.is_file() else None
        records.append({
            "path": relative,
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "passed": actual_sha == expected_sha,
        })
    validator_path = repo_root / (
        "llama.cpp/trace_output/m6c_b_offline_20260805_a3_s0_v4/"
        "artifact_validation_report.json"
    )
    validator = json.loads(validator_path.read_text(encoding="utf-8"))
    validator_pass = (
        validator.get("passed") is True
        and validator.get("summary", {}).get(
            "decision_stream_sha_report_matches_actual", {}
        ).get("display") == "30/30"
        and validator.get("summary", {}).get(
            "deterministic_rerun_matches", {}
        ).get("display") == "30/30"
    )
    return records, all(record["passed"] for record in records) and validator_pass


def read_s0_waits(repo_root: Path, run: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    relative = run.get("decision_stream")
    if not isinstance(relative, str):
        raise ValueError("calibration Run has no decision stream path")
    path = (repo_root / relative).resolve()
    path.relative_to(repo_root)
    digest = hashlib.sha256()
    waits: list[int] = []
    byte_count = 0
    final_line_complete = True
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            byte_count += len(raw)
            if not raw.endswith(b"\n"):
                final_line_complete = False
            record = json.loads(raw)
            if record.get("decision_id") != line_number - 1:
                raise ValueError(f"decision ID mismatch in {run['run_id']}")
            if (
                record.get("mode") != "s0"
                or record.get("selected_source") != "legacy"
                or record.get("source_priority_mode") != "deadline_score"
                or record.get("reserved_due") is not False
                or record.get("credit_before") != 0
                or record.get("credit_accrued") != 0
                or record.get("credit_after") != 0
                or record.get("debt_before") is not False
                or record.get("debt_after") is not False
            ):
                raise ValueError(f"non-S0 policy state observed in {run['run_id']}")
            selected = record.get("selected_task")
            if not isinstance(selected, dict):
                raise ValueError(f"selected Task missing in {run['run_id']}")
            decision_ts = record.get("decision_ts_ns")
            enqueued_ts = selected.get("enqueued_ts_ns")
            if not isinstance(decision_ts, int) or not isinstance(enqueued_ts, int):
                raise ValueError(f"S0 wait input missing in {run['run_id']}")
            wait = decision_ts - enqueued_ts
            if wait < 0 or selected.get("waiting_ns") != wait:
                raise ValueError(f"S0 wait mismatch in {run['run_id']}")
            waits.append(wait)
    actual_sha = digest.hexdigest()
    checks = {
        "path": relative,
        "size_bytes": byte_count,
        "line_count": len(waits),
        "sha256": actual_sha,
        "reported_size_bytes": run.get("decision_stream_size_bytes"),
        "reported_line_count": run.get("decision_stream_line_count"),
        "reported_sha256": run.get("decision_stream_sha256"),
        "final_line_complete": final_line_complete,
        "mode_s0_only": True,
        "s1_winner_executed": False,
    }
    if not (
        byte_count == path.stat().st_size == run.get("decision_stream_size_bytes")
        and len(waits) == EXPECTED_DECISIONS == run.get("decision_stream_line_count")
        and actual_sha == run.get("decision_stream_sha256")
        and final_line_complete
    ):
        raise ValueError(f"calibration stream artifact mismatch in {run['run_id']}")
    return waits, checks


def _source_inventory(repo_root: Path) -> list[dict[str, Any]]:
    patterns = (
        "llama.cpp/trace/m6c_offline/*.py",
        "llama.cpp/trace/m6c_offline/*.json",
        "llama.cpp/trace/m6c_offline/*.md",
        "llama.cpp/trace/tests/test_m6c_offline*.py",
        "docs/codex/M6C_C_replay_preregistration_task.md",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(repo_root.glob(pattern))
    return [{
        "path": str(path.relative_to(repo_root)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in sorted(paths)]


def _write_closed_text(path: Path, content: str, *, parse_json: bool = False) -> dict[str, Any]:
    stream = path.open("x", encoding="utf-8")
    try:
        stream.write(content)
        stream.flush()
    finally:
        stream.close()
    if not stream.closed:
        raise RuntimeError(f"writer did not close: {path}")
    reopened = path.read_text(encoding="utf-8")
    if parse_json:
        json.loads(reopened)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "closed_before_hash": True,
        "parse_validation": "PASS" if parse_json else "UTF8_PASS",
        "durable_fsync_requested": False,
        "durability_claim": "close-and-reopen integrity only",
    }


def _candidate_allowlist(ages: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"candidate_id": "S0", "mode": "deadline_score_same_structure", "enabled": False},
        {"candidate_id": "S1-A", "mode": "reserved_service", "R": 1, "D": 16,
         "eligibility_age_ns": ages["AGE_MODERATE"], "age_source": "AGE_MODERATE", "guard_ns": 0},
        {"candidate_id": "S1-B", "mode": "reserved_service", "R": 1, "D": 16,
         "eligibility_age_ns": ages["AGE_SPARSE"], "age_source": "AGE_SPARSE", "guard_ns": 0},
        {"candidate_id": "S1-C", "mode": "reserved_service", "R": 1, "D": 8,
         "eligibility_age_ns": ages["AGE_MODERATE"], "age_source": "AGE_MODERATE", "guard_ns": 0},
        {"candidate_id": "S1-D", "mode": "reserved_service", "R": 1, "D": 8,
         "eligibility_age_ns": ages["AGE_SPARSE"], "age_source": "AGE_SPARSE", "guard_ns": 0},
    ]


def _metric_and_gate_contract() -> dict[str, Any]:
    return {
        "metric_definitions": {
            "paired_delta": "S1-S0",
            "queue_wait": "selection_decision_ts_ns-enqueued_ts_ns",
            "task_percentile": "nearest-rank",
            "cross_run_median": "midpoint for even N; middle value for odd N",
            "worst_1pct_mean": "mean of largest max(1,ceil(0.01*N)) waits",
            "selected_after_deadline": "selected_ts_ns>=deadline_ts_ns for all nonzero deadlines",
            "lateness": "selected_ts_ns-deadline_ts_ns for selected-after-deadline Tasks",
            "raw_deadline_inversion": (
                "live earlier nonzero deadline exists while selected deadline is later or zero"
            ),
            "longest_task_list_size": 100,
        },
        "material_worse_definition": {
            "operator": "delta_abs>abs_budget AND (S0>0 ? delta_abs/S0>rel_budget : delta_abs>0)",
            "input_units": "raw ns or raw rate",
        },
        "fairness_budgets": [
            {"metric": "worker_median_queue_wait_p99", "absolute_ns": 2_000_000, "relative": 0.10},
            {"metric": "per_run_queue_wait_p99", "absolute_ns": 10_000_000, "relative": 0.25},
            {"metric": "per_run_worst_1pct_mean", "absolute_ns": 5_000_000, "relative": 0.20},
            {"metric": "per_run_max_queue_wait", "absolute_ns": 10_000_000, "relative": 0.25},
        ],
        "urgency_budgets": [
            {"metric": "selected_after_deadline_rate", "absolute_rate": 0.0005, "relative": 0.05},
            {"metric": "raw_deadline_inversion_rate", "absolute_rate": 0.0005, "relative": 0.05},
            {"metric": "lateness_p95", "absolute_ns": 1_000_000, "relative": 0.05},
            {"metric": "lateness_p99", "absolute_ns": 2_000_000, "relative": 0.10},
            {"metric": "lateness_max", "absolute_ns": 10_000_000, "relative": 0.25},
        ],
        "calibration_gate": {
            "correctness_and_safety": "all zero/PASS",
            "p95_worker_medians": "workers=2 and workers=4 both <0ns",
            "p95_run_support": "2/2 Runs improve in each worker stratum",
            "p99_and_tail": "within frozen fairness budgets",
            "urgency": "every Run within frozen urgency budgets",
            "intervention": "nonzero, <=R/D+max(0.005,1/N), longest reserved streak<=1",
            "maximum_finalists": 2,
            "tie_break": [
                "worse-worker p95 median delta", "worse-worker p99 median delta",
                "worse-worker lateness p95 median delta", "lower reserved share",
                "higher eligibility age", "ASCII candidate ID",
            ],
        },
        "holdout_gate": {
            "correctness_and_safety": "all zero/PASS",
            "p95_worker_medians": "workers=2 and workers=4 both <0ns",
            "p95_run_support": "at least 2/3 Runs improve in each worker stratum",
            "tail_and_urgency": "every frozen budget applies",
            "determinism_and_artifacts": "30/30-equivalent applicable Gate PASS",
            "unique_recommendation": "same frozen tie-break if two finalists pass",
        },
        "robustness_boundary": {
            "runs": 20,
            "candidate_count": 1,
            "starts_after_unique_b0_holdout_recommendation": True,
            "can_select_or_change_parameters": False,
            "primary_n30_claim": False,
        },
    }


def _markdown(report: dict[str, Any], json_sha: str) -> str:
    stats = report["calibration_statistics"]
    ages = report["derived_age_values_ns"]
    return "\n".join([
        "# M6C-C Reserved-Service Replay Preregistration",
        "",
        f"最终枚举：`{report['final_enum']}`",
        "",
        f"Preregistration JSON SHA256：`{json_sha}`",
        "",
        "## S0-only 参数来源",
        "",
        f"- workers=2：N={stats['2']['sample_count']}，p75={stats['2']['p75_ns']}ns，p90={stats['2']['p90_ns']}ns。",
        f"- workers=4：N={stats['4']['sample_count']}，p75={stats['4']['p75_ns']}ns，p90={stats['4']['p90_ns']}ns。",
        f"- `AGE_MODERATE={ages['AGE_MODERATE']}ns`；`AGE_SPARSE={ages['AGE_SPARSE']}ns`。",
        "- 只读取四个 B0 calibration S0 stream；未执行 S1 winner或 Reserved-Service counterfactual replay。",
        "",
        "## 冻结参数与矩阵",
        "",
        "- LOW_SHARE：R=1,D=16；MEDIUM_SHARE：R=1,D=8。",
        "- guard=0；AGE_GATED_ALL；single_pending_latch；reset_when_no_eligible。",
        "- allowlist：S0、S1-A、S1-B、S1-C、S1-D；不得扩张。",
        "",
        "## Run split",
        "",
        "- calibration：4 个 B0 Run；holdout：6 个 B0 Run；robustness：20 个 C1/C2 Run。",
        "- 精确 Run ID、指标、风险预算、晋级、Holdout 和 robustness 规则见机器 JSON。",
        "",
        "## 未解决 Gate",
        "",
        f"- M6C-D module：`{report['expected_m6c_d_command']['module']}`；available={str(report['expected_m6c_d_command']['module_available']).lower()}。",
        "- 任务合同禁止用不存在或未审计的 placeholder CLI 输出 READY；因此当前枚举如实反映该 Gate。",
        "",
        "## 固定声明",
        "",
        "```text",
        "counterfactual_type = fixed_arrival_fixed_service_slot_policy_replay",
        "physical_system_reexecuted = false",
        "performance_claim = false",
        "s1_evidence_parameter_comparison_executed = false",
        "```",
        "",
        "本阶段停止。不得自动执行 M6C-D、Shadow、Active、推理、smoke 或正式 N=8。",
        "",
    ])


def execute(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    authority, authority_passed = validate_authority(repo_root)
    s0_path = repo_root / (
        "llama.cpp/trace_output/m6c_b_offline_20260805_a3_s0_v4/s0_replay_report.json"
    )
    s0_report = json.loads(s0_path.read_text(encoding="utf-8"))
    split_validator = validate_run_split(s0_report)
    run_map = {run["run_id"]: run for run in s0_report.get("runs", [])}
    distributions: dict[int, list[int]] = {2: [], 4: []}
    calibration_inputs: list[dict[str, Any]] = []
    parameter_error: str | None = None
    stats: dict[str, Any] = {}
    ages: dict[str, int] = {}

    if authority_passed and split_validator["passed"]:
        try:
            for run_id, workers, repeat, configuration in CALIBRATION:
                waits, checks = read_s0_waits(repo_root, run_map[run_id])
                distributions[workers].extend(waits)
                checks.update({
                    "run_id": run_id,
                    "workers": workers,
                    "repeat_index": repeat,
                    "configuration_id": configuration,
                })
                calibration_inputs.append(checks)
            stats, ages = derive_ages(distributions)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            parameter_error = str(exc)

    module_available = importlib.util.find_spec(M6C_D_MODULE) is not None
    prereg_relative = str(
        (output_dir / "m6c_c_preregistration.json").relative_to(repo_root)
    )
    m6c_d_output = "llama.cpp/trace_output/m6c_d_reserved_service_calibration_holdout_v1"
    expected_command = (
        f"PYTHONPATH=llama.cpp/trace python3 -m {M6C_D_MODULE} "
        f"--preregistration {prereg_relative} --output-dir {m6c_d_output}"
    )

    if not authority_passed:
        final_enum = "EVIDENCE_INPUT_CHANGED"
    elif not split_validator["passed"]:
        final_enum = "PRIMARY_RUN_SPLIT_INVALID"
    elif parameter_error or not ages:
        final_enum = "PARAMETER_SOURCE_INVALID"
    elif not module_available:
        final_enum = "METRIC_OR_GATE_UNRESOLVED"
    else:
        final_enum = "PREREGISTRATION_READY_FOR_S1_REPLAY"

    contract = _metric_and_gate_contract()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_enum": final_enum,
        "authority_artifacts": authority,
        "source_evidence_index": {
            "path": "llama.cpp/trace_output/m6b2a3_directed_20260804_n5_v3_analysis_v2/evidence_index.json",
            "sha256": AUTHORITY_SHA256[
                "llama.cpp/trace_output/m6b2a3_directed_20260804_n5_v3_analysis_v2/evidence_index.json"
            ],
            "checked": 695,
            "indexed": 695,
            "source_run_membership": "explicit 30-entry COMPLETE run index; no directory glob",
        },
        "source_inventory": _source_inventory(repo_root),
        "run_split": {
            "calibration": [_run_record(item) for item in CALIBRATION],
            "holdout": [_run_record(item) for item in HOLDOUT],
            "robustness": [_run_record(item) for item in ROBUSTNESS],
        },
        "split_validator": split_validator,
        "parameter_sources": {
            "reserved_shares": {
                "LOW_SHARE": {"R": 1, "D": 16, "source": "simple general integer prior"},
                "MEDIUM_SHARE": {"R": 1, "D": 8, "source": "simple general integer prior"},
            },
            "eligibility_age": "only four B0 calibration S0 queue-wait distributions",
            "hard_urgent_guard_ns": 0,
            "debt_policy": "single_pending_latch",
            "reset_policy": "reset_when_no_eligible",
            "eligibility_rule": "AGE_GATED_ALL",
            "derived_from_m6b2_c1_c2": False,
            "derived_from_m6b2_threshold_or_guard": False,
            "derived_from_50ms_violation": False,
            "post_hoc_parameter_change_allowed": False,
            "parameter_error": parameter_error,
        },
        "quantile_definition": {
            "name": "nearest-rank",
            "formula": "sorted_x[ceil(p*N)-1]",
            "percentiles": [0.75, 0.90],
            "pooling": "all Tasks from two calibration B0 Runs within each workers stratum",
            "filters": [],
        },
        "rounding_quantum_ns": ROUNDING_QUANTUM_NS,
        "rounding_rule": "ceil_to_positive_multiple",
        "calibration_statistics": stats,
        "calibration_input_streams": calibration_inputs,
        "derived_age_values_ns": ages,
        "candidate_allowlist": _candidate_allowlist(ages) if ages else [],
        **contract,
        "missing_invalid_rules": [
            "missing/null/unavailable is not zero",
            "empty or degenerate age source is PARAMETER_SOURCE_INVALID",
            "split mismatch is PRIMARY_RUN_SPLIT_INVALID",
            "uncomputable metric or Gate is METRIC_OR_GATE_UNRESOLVED",
            "all abnormal Runs retained; no formal confidence intervals",
            "physical Fault/RSS/Swap/PSI/Hint/latency/throughput are not assigned to S1",
        ],
        "output_artifact_schema": {
            "per_run_candidate": [
                "input path/size/line/SHA", "exact parameters", "decision digest",
                "correctness/safety counters", "credit/debt transitions",
                "store/index/registry/bytes conservation", "fairness/urgency/intervention",
                "deterministic digest", "artifact finalization", "counterfactual declarations",
            ],
            "stage_outputs": [
                "01_calibration", "calibration_selection.json",
                "02_holdout", "holdout_selection.json",
                "03_robustness_appendix", "robustness_conclusion.json",
            ],
        },
        "execution_stage_boundaries": {
            "order": ["01_calibration", "02_holdout", "03_robustness_appendix"],
            "calibration_candidate_count": 4,
            "maximum_finalists": 2,
            "holdout_only_after_calibration_manifest_finalized": True,
            "robustness_only_after_unique_holdout_recommendation": True,
            "later_stage_may_rewrite_prior_stage": False,
        },
        "expected_m6c_d_command": {
            "command": expected_command,
            "module": M6C_D_MODULE,
            "module_available": module_available,
            "module_source_sha256": None,
            "output_dir": m6c_d_output,
            "output_dir_must_not_exist": True,
            "execution_authorized": False,
            "unresolved_reason": None if module_available else "audited M6C-D offline module does not exist",
        },
        "random_seed": "none",
        "input_access_audit": {
            "s0_calibration_streams_read": 4,
            "holdout_streams_read_for_parameter_derivation": 0,
            "robustness_streams_read_for_parameter_derivation": 0,
            "s1_winner_executed": False,
            "reserved_service_counterfactual_replay_executed": False,
            "policy_module_imported": False,
        },
        "formal_parameters_selected": bool(ages),
        "s1_evidence_parameter_comparison_executed": False,
        "shadow_executed": False,
        "active_executed": False,
        "smoke_executed": False,
        "inference_runs": 0,
        "formal_n8_executed": False,
        "forbidden_actions": [
            "run M6C-D without human approval", "modify parameters or candidate allowlist",
            "run Shadow/Active/smoke/inference/N=8", "modify runtime",
            "claim physical performance", "use C1/C2 or holdout to derive parameters",
        ],
        "next_stage_requires_human_approval": True,
        **COUNTERFACTUAL_DECLARATION,
    }

    json_path = output_dir / "m6c_c_preregistration.json"
    json_metadata = _write_closed_text(
        json_path,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        parse_json=True,
    )
    markdown_path = output_dir / "m6c_c_preregistration.md"
    markdown_metadata = _write_closed_text(
        markdown_path,
        _markdown(report, json_metadata["sha256"]),
    )
    sha_path = output_dir / "m6c_c_preregistration.sha256"
    sha_metadata = _write_closed_text(
        sha_path,
        f"{json_metadata['sha256']}  m6c_c_preregistration.json\n",
    )
    artifact_manifest = {
        "schema_version": "m6c-c-preregistration-artifact-manifest-v1",
        "artifacts": [json_metadata, markdown_metadata, sha_metadata],
        "artifact_count": 3,
        "hashes_computed_after_close": True,
        "durable_fsync_requested": False,
        "durability_claim": "close-and-reopen integrity only",
    }
    _write_closed_text(
        output_dir / "artifact_manifest.json",
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        parse_json=True,
    )

    from .preregistration_validator import validate_preregistration

    validation = validate_preregistration(repo_root, output_dir)
    _write_closed_text(
        output_dir / "independent_validation.json",
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        parse_json=True,
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = execute(Path(args.repo_root), Path(args.output_dir))
    print(json.dumps({
        "final_enum": report["final_enum"],
        "AGE_MODERATE": report.get("derived_age_values_ns", {}).get("AGE_MODERATE"),
        "AGE_SPARSE": report.get("derived_age_values_ns", {}).get("AGE_SPARSE"),
        "s1_replay_executed": report["s1_evidence_parameter_comparison_executed"],
        "output_dir": str(Path(args.output_dir).resolve()),
    }, sort_keys=True))
    return 0 if report["final_enum"] == "PREREGISTRATION_READY_FOR_S1_REPLAY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
