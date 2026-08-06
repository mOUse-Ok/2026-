#!/usr/bin/env python3
"""Independent schema, Hash, split, and S0 age validator for M6C-C."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from .evidence import sha256_file


ALLOWED_ENUMS = {
    "PREREGISTRATION_READY_FOR_S1_REPLAY",
    "PARAMETER_SOURCE_INVALID",
    "PRIMARY_RUN_SPLIT_INVALID",
    "METRIC_OR_GATE_UNRESOLVED",
    "EVIDENCE_INPUT_CHANGED",
    "M6C_ROUTE_STOP_RECOMMENDED",
}


def _nearest_rank(values: list[int], p: float) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty S0 wait distribution")
    return ordered[math.ceil(p * len(ordered)) - 1]


def _ceil_ms(value: int) -> int:
    return ((value + 999_999) // 1_000_000) * 1_000_000


def _read_calibration_waits(repo_root: Path, run: dict[str, Any]) -> tuple[list[int], str, int]:
    path = (repo_root / run["decision_stream"]).resolve()
    path.relative_to(repo_root)
    digest = hashlib.sha256()
    waits: list[int] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            record = json.loads(raw)
            if record.get("decision_id") != line_number - 1:
                raise ValueError("decision stream is not dense and ordered")
            if record.get("mode") != "s0" or record.get("selected_source") != "legacy":
                raise ValueError("non-S0 selection observed during age validation")
            if any(record.get(field) != 0 for field in ("credit_before", "credit_accrued", "credit_after")):
                raise ValueError("credit changed in S0 age input")
            if record.get("debt_before") is not False or record.get("debt_after") is not False:
                raise ValueError("debt changed in S0 age input")
            selected = record["selected_task"]
            wait = record["decision_ts_ns"] - selected["enqueued_ts_ns"]
            if wait < 0 or wait != selected.get("waiting_ns"):
                raise ValueError("S0 wait input mismatch")
            waits.append(wait)
    return waits, digest.hexdigest(), path.stat().st_size


def _validate_schema(report: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        import jsonschema
    except ImportError:
        return False, "jsonschema dependency unavailable"
    try:
        jsonschema.validate(report, schema)
    except jsonschema.ValidationError as exc:
        return False, exc.message
    return True, None


def validate_preregistration(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    report_path = output_dir / "m6c_c_preregistration.json"
    manifest_path = output_dir / "artifact_manifest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_path = repo_root / "llama.cpp/trace/m6c_offline/preregistration_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors: list[dict[str, Any]] = []

    schema_passed, schema_error = _validate_schema(report, schema)
    if not schema_passed:
        errors.append({"reason": "schema_validation_failed", "detail": schema_error})

    artifact_records: list[dict[str, Any]] = []
    for item in manifest.get("artifacts", []):
        path = Path(item.get("path", "")).resolve()
        try:
            path.relative_to(output_dir)
        except ValueError:
            errors.append({"reason": "artifact_path_escapes_output", "path": str(path)})
            continue
        actual_size = path.stat().st_size if path.is_file() else None
        actual_sha = sha256_file(path) if path.is_file() else None
        passed = actual_size == item.get("size_bytes") and actual_sha == item.get("sha256")
        if not passed:
            errors.append({"reason": "artifact_hash_or_size_mismatch", "path": str(path)})
        artifact_records.append({
            "path": str(path),
            "reported_size_bytes": item.get("size_bytes"),
            "actual_size_bytes": actual_size,
            "reported_sha256": item.get("sha256"),
            "actual_sha256": actual_sha,
            "passed": passed,
        })
    if len(artifact_records) != 3:
        errors.append({"reason": "artifact_manifest_count_mismatch"})

    detached_path = output_dir / "m6c_c_preregistration.sha256"
    detached = detached_path.read_text(encoding="utf-8").strip().split()
    actual_report_sha = sha256_file(report_path)
    detached_passed = (
        len(detached) == 2
        and detached[0] == actual_report_sha
        and detached[1] == "m6c_c_preregistration.json"
    )
    if not detached_passed:
        errors.append({"reason": "detached_preregistration_sha_mismatch"})

    authority_records: list[dict[str, Any]] = []
    for item in report.get("authority_artifacts", []):
        path = (repo_root / item.get("path", "")).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError:
            errors.append({"reason": "authority_path_escapes_repo"})
            continue
        actual_sha = sha256_file(path) if path.is_file() else None
        passed = (
            actual_sha == item.get("expected_sha256") == item.get("actual_sha256")
            and path.stat().st_size == item.get("size_bytes")
        ) if path.is_file() else False
        if not passed:
            errors.append({"reason": "authority_artifact_changed", "path": item.get("path")})
        authority_records.append({"path": item.get("path"), "actual_sha256": actual_sha, "passed": passed})

    s0_path = repo_root / (
        "llama.cpp/trace_output/m6c_b_offline_20260805_a3_s0_v4/s0_replay_report.json"
    )
    s0 = json.loads(s0_path.read_text(encoding="utf-8"))
    run_map = {item["run_id"]: item for item in s0["runs"]}
    split = report.get("run_split", {})
    calibration = split.get("calibration", [])
    holdout = split.get("holdout", [])
    robustness = split.get("robustness", [])
    all_ids = [item.get("run_id") for role in (calibration, holdout, robustness) for item in role]
    split_passed = (
        len(calibration) == 4
        and len(holdout) == 6
        and len(robustness) == 20
        and len(set(all_ids)) == 30
        and set(all_ids) == set(run_map)
        and all(item.get("configuration_id") == "B0" and item.get("repeat_index") in (1, 2) for item in calibration)
        and all(item.get("configuration_id") == "B0" and item.get("repeat_index") in (3, 4, 5) for item in holdout)
        and all(item.get("configuration_id") in ("C1", "C2") for item in robustness)
        and sorted((item["workers"], item["repeat_index"]) for item in calibration)
            == [(2, 1), (2, 2), (4, 1), (4, 2)]
        and sorted((item["workers"], item["repeat_index"]) for item in holdout)
            == [(2, 3), (2, 4), (2, 5), (4, 3), (4, 4), (4, 5)]
    )
    if not split_passed:
        errors.append({"reason": "independent_split_validation_failed"})

    distributions: dict[int, list[int]] = {2: [], 4: []}
    stream_checks: list[dict[str, Any]] = []
    try:
        for item in calibration:
            source = run_map[item["run_id"]]
            waits, actual_sha, actual_size = _read_calibration_waits(repo_root, source)
            if (
                len(waits) != 29_262
                or actual_sha != source.get("decision_stream_sha256")
                or actual_size != source.get("decision_stream_size_bytes")
            ):
                raise ValueError("calibration stream metadata mismatch")
            distributions[item["workers"]].extend(waits)
            stream_checks.append({
                "run_id": item["run_id"],
                "workers": item["workers"],
                "line_count": len(waits),
                "sha256": actual_sha,
                "size_bytes": actual_size,
                "s0_only": True,
                "passed": True,
            })
        independent_stats = {
            str(workers): {
                "sample_count": len(distributions[workers]),
                "p75_ns": _nearest_rank(distributions[workers], 0.75),
                "p90_ns": _nearest_rank(distributions[workers], 0.90),
            } for workers in (2, 4)
        }
        independent_ages = {
            "AGE_MODERATE": _ceil_ms(max(independent_stats["2"]["p75_ns"], independent_stats["4"]["p75_ns"])),
            "AGE_SPARSE": _ceil_ms(max(independent_stats["2"]["p90_ns"], independent_stats["4"]["p90_ns"])),
        }
        ages_passed = (
            independent_ages == report.get("derived_age_values_ns")
            and all(
                independent_stats[str(workers)][field] == report.get("calibration_statistics", {}).get(str(workers), {}).get(field)
                for workers in (2, 4) for field in ("sample_count", "p75_ns", "p90_ns")
            )
            and 0 < independent_ages["AGE_MODERATE"] < independent_ages["AGE_SPARSE"]
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        independent_stats = {}
        independent_ages = {}
        ages_passed = False
        errors.append({"reason": "independent_age_derivation_failed", "detail": str(exc)})
    if not ages_passed:
        errors.append({"reason": "derived_age_mismatch"})

    candidates = report.get("candidate_allowlist", [])
    candidates_passed = (
        [item.get("candidate_id") for item in candidates] == ["S0", "S1-A", "S1-B", "S1-C", "S1-D"]
        and [(item.get("R"), item.get("D")) for item in candidates[1:]]
            == [(1, 16), (1, 16), (1, 8), (1, 8)]
        and all(item.get("guard_ns") == 0 for item in candidates[1:])
    )
    if not candidates_passed:
        errors.append({"reason": "candidate_allowlist_mismatch"})

    command = report.get("expected_m6c_d_command", {})
    module = command.get("module")
    module_available = isinstance(module, str) and importlib.util.find_spec(module) is not None
    command_gate_resolved = (
        module_available
        and isinstance(command.get("module_source_sha256"), str)
        and len(command["module_source_sha256"]) == 64
        and "<" not in command.get("command", "")
        and ">" not in command.get("command", "")
    )
    expected_enum = (
        "PREREGISTRATION_READY_FOR_S1_REPLAY"
        if command_gate_resolved else "METRIC_OR_GATE_UNRESOLVED"
    )
    final_enum_passed = report.get("final_enum") == expected_enum
    if not final_enum_passed:
        errors.append({"reason": "final_enum_does_not_match_command_gate", "expected": expected_enum})

    scope_passed = (
        report.get("s1_evidence_parameter_comparison_executed") is False
        and report.get("input_access_audit", {}).get("s1_winner_executed") is False
        and report.get("input_access_audit", {}).get("reserved_service_counterfactual_replay_executed") is False
        and report.get("shadow_executed") is False
        and report.get("active_executed") is False
        and report.get("inference_runs") == 0
    )
    if not scope_passed:
        errors.append({"reason": "forbidden_execution_declaration"})

    integrity_passed = not errors
    return {
        "schema_version": "m6c-c-independent-preregistration-validation-v1",
        "validation_passed": integrity_passed,
        "ready_for_s1_replay": integrity_passed and expected_enum == "PREREGISTRATION_READY_FOR_S1_REPLAY",
        "validation_result": (
            "PASS_READY" if integrity_passed and command_gate_resolved
            else "PASS_NON_READY_COMMAND_GATE_UNRESOLVED" if integrity_passed
            else "FAIL"
        ),
        "preregistration_path": str(report_path),
        "preregistration_sha256": actual_report_sha,
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "schema_validation_passed": schema_passed,
        "schema_error": schema_error,
        "detached_sha_validation_passed": detached_passed,
        "artifact_manifest_validation_passed": all(item["passed"] for item in artifact_records) and len(artifact_records) == 3,
        "artifact_records": artifact_records,
        "authority_validation_passed": all(item["passed"] for item in authority_records),
        "authority_records": authority_records,
        "split_validation_passed": split_passed,
        "calibration_streams": stream_checks,
        "independent_calibration_statistics": independent_stats,
        "independent_derived_age_values_ns": independent_ages,
        "derived_age_validation_passed": ages_passed,
        "candidate_allowlist_validation_passed": candidates_passed,
        "command_gate_resolved": command_gate_resolved,
        "m6c_d_module_available": module_available,
        "final_enum": report.get("final_enum"),
        "final_enum_validation_passed": final_enum_passed,
        "scope_validation_passed": scope_passed,
        "errors": errors,
        "s1_winner_executed": False,
        "reserved_service_counterfactual_replay_executed": False,
        "physical_system_reexecuted": False,
        "performance_claim": False,
    }


def _write_closed(path: Path, value: dict[str, Any]) -> None:
    stream = path.open("x", encoding="utf-8")
    try:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
    finally:
        stream.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-output", required=True)
    args = parser.parse_args(argv)
    result = validate_preregistration(Path(args.repo_root), Path(args.output_dir))
    _write_closed(Path(args.validation_output), result)
    print(json.dumps({
        "validation_passed": result["validation_passed"],
        "ready_for_s1_replay": result["ready_for_s1_replay"],
        "validation_result": result["validation_result"],
    }, sort_keys=True))
    return 0 if result["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
