#!/usr/bin/env python3
"""Independent closed-artifact and M6C-C v2 inheritance validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import inspect_closed_jsonl
from .evidence import sha256_file


ALLOWED_V2_CHANGED_PATHS = {
    "final_enum",
    "expected_m6c_d_command.command",
    "expected_m6c_d_command.module_available",
    "expected_m6c_d_command.module_source_sha256",
    "expected_m6c_d_command.output_dir",
    "expected_m6c_d_command.unresolved_reason",
}
ALLOWED_V2_ADDITIONS = {"command_gate_resolved", "runner_test_and_audit"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_artifact(root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    relative = Path(str(metadata.get("path", "")))
    path = (root / relative).resolve()
    errors: list[str] = []
    try:
        path.relative_to(root.resolve())
    except ValueError:
        errors.append("path_escape")
    if not path.is_file():
        errors.append("missing")
        actual_size = actual_sha = None
    else:
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != metadata.get("size_bytes"):
            errors.append("size_mismatch")
        if actual_sha != metadata.get("sha256"):
            errors.append("sha256_mismatch")
    if metadata.get("line_count") is not None and path.is_file():
        inspection = inspect_closed_jsonl(
            path,
            expected_line_count=int(metadata["line_count"]),
        )
        if not inspection["passed"]:
            errors.append("jsonl_validation_failed")
    return {
        "path": str(relative),
        "passed": not errors,
        "errors": errors,
        "reported_size_bytes": metadata.get("size_bytes"),
        "actual_size_bytes": actual_size,
        "reported_sha256": metadata.get("sha256"),
        "actual_sha256": actual_sha,
    }


def validate_stage_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    errors: list[str] = []
    try:
        manifest_path.relative_to(root)
    except ValueError:
        errors.append("manifest_path_escape")
    try:
        manifest = _load_json(manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"manifest_unreadable:{exc}"], "artifacts": []}
    records = [
        _validate_artifact(root, item)
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict)
    ]
    if len(records) != len(manifest.get("artifacts", [])) or not records:
        errors.append("artifact_records_invalid")
    if manifest.get("hashes_computed_after_close") is not True:
        errors.append("post_close_hash_declaration_missing")
    if not all(record["passed"] for record in records):
        errors.append("artifact_mismatch")
    return {
        "schema_version": "m6c-d-independent-stage-validation-v1",
        "passed": not errors,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifacts": records,
        "errors": errors,
    }


def validate_output(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    root_manifest_path = output_dir / "artifact_manifest.json"
    try:
        root_manifest = _load_json(root_manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"root_manifest_unreadable:{exc}"]}
    errors: list[str] = []
    report_record = _validate_artifact(output_dir, root_manifest.get("report", {}))
    if not report_record["passed"]:
        errors.append("root_report_mismatch")
    stage_validations: dict[str, Any] = {}
    for role, stage in root_manifest.get("stages", {}).items():
        manifest_record = stage.get("manifest", {})
        record = _validate_artifact(output_dir, manifest_record)
        stage_manifest_path = output_dir / str(manifest_record.get("path", ""))
        validation = validate_stage_manifest(output_dir, stage_manifest_path)
        stage_validations[role] = {
            "manifest_artifact": record,
            "stage_validation": validation,
        }
        if not record["passed"] or not validation["passed"]:
            errors.append(f"{role}_stage_mismatch")
    if not stage_validations:
        errors.append("no_finalized_stage")
    return {
        "schema_version": "m6c-d-independent-output-validation-v1",
        "passed": not errors,
        "root_manifest_path": str(root_manifest_path),
        "root_manifest_sha256": sha256_file(root_manifest_path),
        "root_report": report_record,
        "stages": stage_validations,
        "errors": errors,
    }


def _diff_paths(left: Any, right: Any, prefix: str = "") -> tuple[set[str], set[str], set[str]]:
    changed: set[str] = set()
    added: set[str] = set()
    removed: set[str] = set()
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else key
            if key not in left:
                added.add(path)
            elif key not in right:
                removed.add(path)
            else:
                c, a, r = _diff_paths(left[key], right[key], path)
                changed |= c
                added |= a
                removed |= r
    elif left != right:
        changed.add(prefix)
    return changed, added, removed


def validate_preregistration_v2(v1_dir: Path, v2_dir: Path, runner_path: Path) -> dict[str, Any]:
    v1_dir, v2_dir, runner_path = v1_dir.resolve(), v2_dir.resolve(), runner_path.resolve()
    errors: list[str] = []
    v1_path = v1_dir / "m6c_c_preregistration.json"
    v2_path = v2_dir / "m6c_c_preregistration.json"
    try:
        v1 = _load_json(v1_path)
        v2 = _load_json(v2_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"preregistration_unreadable:{exc}"]}

    changed, added, removed = _diff_paths(v1, v2)
    if changed - ALLOWED_V2_CHANGED_PATHS:
        errors.append("frozen_v1_fields_changed")
    if added - ALLOWED_V2_ADDITIONS:
        errors.append("unapproved_v2_fields_added")
    if removed:
        errors.append("v1_fields_removed")
    runner_sha = sha256_file(runner_path)
    command = v2.get("expected_m6c_d_command", {})
    if v2.get("final_enum") != "PREREGISTRATION_READY_FOR_S1_REPLAY":
        errors.append("ready_enum_missing")
    if v2.get("command_gate_resolved") is not True:
        errors.append("command_gate_not_resolved")
    if command.get("module") != "m6c_offline.m6c_d_runner":
        errors.append("runner_module_mismatch")
    if command.get("module_source_sha256") != runner_sha:
        errors.append("runner_sha_mismatch")
    if command.get("module_available") is not True or command.get("unresolved_reason") is not None:
        errors.append("command_availability_mismatch")
    audit = v2.get("runner_test_and_audit", {})
    if audit.get("base_v1_sha256") != sha256_file(v1_path):
        errors.append("base_v1_sha_mismatch")
    validator_path = runner_path.parent / "m6c_d_validator.py"
    test_path = runner_path.parents[1] / "tests/test_m6c_offline_m6c_d.py"
    if audit.get("independent_validator_sha256") != sha256_file(validator_path):
        errors.append("independent_validator_sha_mismatch")
    if audit.get("runner_test_sha256") != sha256_file(test_path):
        errors.append("runner_test_sha_mismatch")
    if audit.get("tests", {}).get("passed") is not True:
        errors.append("runner_tests_not_passed")
    if audit.get("synthetic_three_stage", {}).get("passed") is not True:
        errors.append("synthetic_three_stage_not_passed")
    if audit.get("a3_s1_replay_executed") is not False:
        errors.append("a3_s1_replay_scope_violation")

    schema_path = runner_path.parent / "preregistration_schema.json"
    try:
        import jsonschema
    except ImportError as exc:
        schema_passed = False
        schema_error = str(exc)
        errors.append("schema_validation_failed")
    else:
        try:
            jsonschema.validate(v2, _load_json(schema_path))
            schema_passed = True
            schema_error = None
        except jsonschema.ValidationError as exc:
            schema_passed = False
            schema_error = str(exc)
            errors.append("schema_validation_failed")

    detached = (v2_dir / "m6c_c_preregistration.sha256").read_text(encoding="utf-8").strip().split()
    detached_passed = len(detached) == 2 and detached[0] == sha256_file(v2_path) and detached[1] == v2_path.name
    if not detached_passed:
        errors.append("detached_sha_mismatch")
    manifest = _load_json(v2_dir / "artifact_manifest.json")
    artifact_records = [_validate_artifact(v2_dir, item) for item in manifest.get("artifacts", [])]
    if len(artifact_records) != 3 or not all(item["passed"] for item in artifact_records):
        errors.append("artifact_manifest_mismatch")
    return {
        "schema_version": "m6c-c-v2-independent-validation-v1",
        "passed": not errors,
        "v1_path": str(v1_path),
        "v1_sha256": sha256_file(v1_path),
        "v2_path": str(v2_path),
        "v2_sha256": sha256_file(v2_path),
        "runner_path": str(runner_path),
        "runner_sha256": runner_sha,
        "changed_paths": sorted(changed),
        "added_paths": sorted(added),
        "removed_paths": sorted(removed),
        "schema_validation_passed": schema_passed,
        "schema_error": schema_error,
        "detached_sha_validation_passed": detached_passed,
        "artifact_records": artifact_records,
        "frozen_v1_inheritance_passed": not (changed - ALLOWED_V2_CHANGED_PATHS or added - ALLOWED_V2_ADDITIONS or removed),
        "a3_s1_replay_executed": False,
        "errors": errors,
    }


__all__ = ["validate_output", "validate_preregistration_v2", "validate_stage_manifest"]
