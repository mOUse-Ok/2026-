"""Independent validator for finalized M6C-B.2 reports and decision streams."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .artifacts import inspect_closed_jsonl
from .evidence import COUNTERFACTUAL_DECLARATION, sha256_file, validate_evidence_index


EXPECTED_SUPERSEDES = "m6c_b_offline_20260805_a3_s0_v3"
EXPECTED_SUPERSESSION_REASON = (
    "decision stream SHA calculated before buffered stream finalization"
)
EXPECTED_V3_CLASSIFICATION = "technically informative but non-canonical"
ALLOWED_FINAL_ENUMS = {
    "M6C_B_EVIDENCE_REPAIRED_AND_CONFIRMED",
    "HASH_FINALIZATION_FIX_FAILED",
    "DECISION_STREAM_CONTENT_NOT_DETERMINISTIC",
    "S0_RESULT_CHANGED_AFTER_REPAIR",
    "ADDITIONAL_EVIDENCE_MISMATCH_FOUND",
    "M6C_B_REQUIRES_REIMPLEMENTATION",
}


def _ratio(matched: int, total: int) -> dict[str, Any]:
    return {
        "matched": matched,
        "total": total,
        "display": f"{matched}/{total}",
        "passed": matched == total,
    }


def _resolve_repo_artifact(repo_root: Path, value: Any) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value:
        return None, "missing_reported_path"
    relative = Path(value)
    if relative.is_absolute():
        return None, "artifact_path_must_be_repo_relative"
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return None, "artifact_path_escapes_repo_root"
    return resolved, None


def validate_report_artifacts(
    repo_root: Path,
    report_path: Path,
    *,
    expected_stream_count: int = 30,
) -> dict[str, Any]:
    """Reread a written report, source Evidence, and every final stream."""

    repo_root = repo_root.resolve()
    report_path = report_path.resolve()
    errors: list[dict[str, Any]] = []
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "m6c-b2-independent-artifact-validation-v1",
            "passed": False,
            "report_path": str(report_path),
            "errors": [{"reason": "report_unreadable", "detail": str(exc)}],
            "streams": [],
        }
    if not isinstance(report, dict):
        return {
            "schema_version": "m6c-b2-independent-artifact-validation-v1",
            "passed": False,
            "report_path": str(report_path),
            "errors": [{"reason": "report_is_not_object"}],
            "streams": [],
        }

    final_enum = report.get("final_enum")
    if final_enum not in ALLOWED_FINAL_ENUMS:
        errors.append({"reason": "invalid_final_enum", "actual": final_enum})
    if report.get("schema_version") != "m6c-b2-evidence-finalization-report-v1":
        errors.append({"reason": "unexpected_report_schema", "actual": report.get("schema_version")})
    for field, expected in COUNTERFACTUAL_DECLARATION.items():
        if report.get(field) != expected:
            errors.append({"reason": "declaration_mismatch", "field": field})
    for field in (
        "formal_parameters_selected",
        "s1_evidence_parameter_comparison_executed",
    ):
        if report.get(field) is not False:
            errors.append({"reason": "forbidden_scope_declaration", "field": field})
    if report.get("supersedes") != EXPECTED_SUPERSEDES:
        errors.append({"reason": "supersedes_mismatch"})
    if report.get("supersession_reason") != EXPECTED_SUPERSESSION_REASON:
        errors.append({"reason": "supersession_reason_mismatch"})
    if report.get("superseded_output_classification") != EXPECTED_V3_CLASSIFICATION:
        errors.append({"reason": "v3_classification_mismatch"})

    reconstructability = report.get("reconstructability", {})
    reported_evidence = reconstructability.get("evidence_hash", {})
    evidence_path_value = reported_evidence.get("evidence_index_path")
    evidence_path = Path(evidence_path_value).resolve() if isinstance(evidence_path_value, str) else None
    if evidence_path is None or not evidence_path.is_file():
        actual_evidence = {"passed": False, "reason": "evidence_index_unavailable"}
        errors.append({"reason": "evidence_index_unavailable", "path": evidence_path_value})
    else:
        actual_evidence = validate_evidence_index(repo_root, evidence_path)
        if (
            not actual_evidence.get("passed")
            or actual_evidence.get("evidence_index_sha256") != reported_evidence.get("evidence_index_sha256")
            or actual_evidence.get("checked") != reported_evidence.get("checked")
        ):
            errors.append({"reason": "evidence_source_hash_mismatch"})

    replay = report.get("s0_replay", {})
    runs = replay.get("runs")
    if not isinstance(runs, list):
        runs = []
        errors.append({"reason": "s0_runs_missing"})
    if len(runs) != expected_stream_count:
        errors.append({
            "reason": "stream_count_mismatch",
            "expected": expected_stream_count,
            "actual": len(runs),
        })

    stream_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_run_ids: set[str] = set()
    sha_matches = 0
    size_matches = 0
    line_matches = 0
    parseable = 0
    final_lines = 0
    finalized_after_close = 0
    deterministic_matches = 0
    total_actual_lines = 0

    for run in runs:
        run_errors: list[dict[str, Any]] = []
        run_id = run.get("run_id") if isinstance(run, dict) else None
        if not isinstance(run_id, str) or not run_id or run_id in seen_run_ids:
            run_errors.append({"reason": "invalid_or_duplicate_run_id"})
        else:
            seen_run_ids.add(run_id)
        reported_path = run.get("decision_stream") if isinstance(run, dict) else None
        final_path, path_error = _resolve_repo_artifact(repo_root, reported_path)
        if path_error:
            run_errors.append({"reason": path_error})
        if isinstance(reported_path, str):
            if reported_path in seen_paths:
                run_errors.append({"reason": "duplicate_decision_stream_path"})
            seen_paths.add(reported_path)

        expected_lines = run.get("s0_decisions_executed") if isinstance(run, dict) else None
        if not isinstance(expected_lines, int) or expected_lines <= 0:
            run_errors.append({"reason": "invalid_expected_decision_count"})
            expected_lines = None
        if final_path is None:
            actual = {
                "passed": False,
                "path": None,
                "size_bytes": None,
                "line_count": 0,
                "sha256": None,
                "jsonl_parseable": False,
                "final_line_complete": False,
                "errors": [],
            }
        else:
            actual = inspect_closed_jsonl(final_path, expected_line_count=expected_lines)
        total_actual_lines += int(actual.get("line_count") or 0)

        this_sha_match = actual.get("sha256") == run.get("decision_stream_sha256")
        this_size_match = actual.get("size_bytes") == run.get("decision_stream_size_bytes")
        this_line_match = (
            actual.get("line_count") == run.get("decision_stream_line_count") == expected_lines
        )
        this_parseable = actual.get("jsonl_parseable") is True
        this_final_line = actual.get("final_line_complete") is True
        this_finalized = run.get("decision_stream_finalized_after_close") is True
        deterministic = run.get("deterministic_rerun", {})
        this_deterministic = (
            deterministic.get("executed") is True
            and deterministic.get("content_sha256") == actual.get("sha256")
            and deterministic.get("line_count") == actual.get("line_count")
            and deterministic.get("s0_result_matches") is True
            and deterministic.get("content_matches_finalized_stream") is True
        )

        sha_matches += int(this_sha_match)
        size_matches += int(this_size_match)
        line_matches += int(this_line_match)
        parseable += int(this_parseable)
        final_lines += int(this_final_line)
        finalized_after_close += int(this_finalized)
        deterministic_matches += int(this_deterministic)
        for reason, passed in (
            ("reported_sha_mismatch", this_sha_match),
            ("reported_size_mismatch", this_size_match),
            ("reported_line_count_mismatch", this_line_match),
            ("jsonl_not_parseable", this_parseable),
            ("partial_final_line", this_final_line),
            ("not_finalized_after_close", this_finalized),
            ("deterministic_rerun_mismatch", this_deterministic),
        ):
            if not passed:
                run_errors.append({"reason": reason})
        if actual.get("errors"):
            run_errors.extend(actual["errors"])

        stream_records.append({
            "run_id": run_id,
            "reported_path": reported_path,
            "final_path": str(final_path) if final_path is not None else None,
            "reported_size_bytes": run.get("decision_stream_size_bytes"),
            "actual_size_bytes": actual.get("size_bytes"),
            "reported_line_count": run.get("decision_stream_line_count"),
            "actual_line_count": actual.get("line_count"),
            "reported_sha256": run.get("decision_stream_sha256"),
            "actual_sha256": actual.get("sha256"),
            "sha_report_matches_actual": this_sha_match,
            "size_report_matches_actual": this_size_match,
            "line_count_matches": this_line_match,
            "jsonl_parseable": this_parseable,
            "final_line_complete": this_final_line,
            "finalized_after_close": this_finalized,
            "deterministic_rerun_matches": this_deterministic,
            "passed": not run_errors,
            "errors": run_errors,
        })

    for item in stream_records:
        if not item["passed"]:
            errors.append({"reason": "decision_stream_validation_failed", "run_id": item["run_id"]})

    expected_total = sum(
        int(run.get("s0_decisions_executed", 0)) for run in runs if isinstance(run, dict)
    )
    gate_checks = {
        "reconstructable_runs_30": reconstructability.get("status_counts", {}).get("RECONSTRUCTABLE") == expected_stream_count,
        "reported_total_decisions_matches_runs": replay.get("total_decisions") == expected_total,
        "actual_total_lines_matches_decisions": total_actual_lines == expected_total,
        "s0_oracle_winner_mismatches_zero": replay.get("s0_oracle_winner_mismatches") == 0,
        "b0_runtime_winner_mismatches_zero": replay.get("runtime_deadline_score_winner_mismatches") == 0,
        "full_store_scan_zero": replay.get("full_store_scan_count") == 0,
        "stale_handle_zero": replay.get("stale_handle_count") == 0,
        "all_invariants_passed": replay.get("all_invariants_passed") is True,
        "all_final_queues_empty": len(runs) == expected_stream_count and all(
            isinstance(run, dict) and run.get("final_queue_empty") is True for run in runs
        ),
        "all_runs_passed": len(runs) == expected_stream_count and all(
            isinstance(run, dict) and run.get("passed") is True for run in runs
        ),
    }
    for name, passed in gate_checks.items():
        if not passed:
            errors.append({"reason": "replay_gate_failed", "gate": name})

    summary = {
        "decision_stream_count": _ratio(len(runs), expected_stream_count),
        "decision_stream_sha_report_matches_actual": _ratio(sha_matches, expected_stream_count),
        "decision_stream_size_report_matches_actual": _ratio(size_matches, expected_stream_count),
        "decision_stream_line_count_matches": _ratio(line_matches, expected_stream_count),
        "decision_stream_jsonl_parseable": _ratio(parseable, expected_stream_count),
        "decision_stream_final_line_complete": _ratio(final_lines, expected_stream_count),
        "decision_stream_finalized_after_close": _ratio(finalized_after_close, expected_stream_count),
        "deterministic_rerun_matches": _ratio(deterministic_matches, expected_stream_count),
        "total_actual_lines": total_actual_lines,
        "expected_total_decisions": expected_total,
    }
    reported_summary = report.get("artifact_finalization", {})
    for field in (
        "decision_stream_sha_report_matches_actual",
        "decision_stream_line_count_matches",
        "decision_stream_jsonl_parseable",
    ):
        if reported_summary.get(field) != summary[field]["display"]:
            errors.append({"reason": "reported_artifact_summary_mismatch", "field": field})

    return {
        "schema_version": "m6c-b2-independent-artifact-validation-v1",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "expected_stream_count": expected_stream_count,
        "evidence_source_hash_validation": actual_evidence,
        "summary": summary,
        "replay_gate_checks": gate_checks,
        "streams": stream_records,
        "errors": errors,
        **COUNTERFACTUAL_DECLARATION,
        "formal_parameters_selected": False,
        "s1_evidence_parameter_comparison_executed": False,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = validate_report_artifacts(Path(args.repo_root), Path(args.report))
    _write_json(Path(args.output), result)
    print(json.dumps({
        "passed": result["passed"],
        "report": result["report_path"],
        "sha_matches": result.get("summary", {}).get(
            "decision_stream_sha_report_matches_actual", {}
        ).get("display"),
    }, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
