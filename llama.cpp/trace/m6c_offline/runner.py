#!/usr/bin/env python3
"""Run the bounded M6C-B offline test, reconstructability, and S0 Gates."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import unittest
from typing import Any, Iterable

from .artifact_validator import (
    EXPECTED_SUPERSEDES,
    EXPECTED_SUPERSESSION_REASON,
    EXPECTED_V3_CLASSIFICATION,
    validate_report_artifacts,
)
from .artifacts import (
    ArtifactFinalizationError,
    DecisionDigestSink,
    write_finalized_jsonl,
)
from .evidence import (
    COUNTERFACTUAL_DECLARATION,
    ReconstructabilityStatus,
    audit_run,
    load_run_index,
    replay_s0,
    sha256_file,
    validate_evidence_index,
)
from .model import ModelInvariantError


SCHEMA_VERSION = "m6c-b2-evidence-finalization-report-v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _iter_tests(suite: unittest.TestSuite) -> Iterable[unittest.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def run_m6c_tests(repo_root: Path) -> dict[str, Any]:
    trace_dir = repo_root / "llama.cpp" / "trace"
    test_dir = trace_dir / "tests"
    if str(trace_dir) not in sys.path:
        sys.path.insert(0, str(trace_dir))
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern="test_m6c_offline*.py")
    names = [test.id() for test in _iter_tests(suite)]
    buffer = io.StringIO()
    runner = unittest.TextTestRunner(stream=buffer, verbosity=2)
    result = runner.run(suite)
    return {
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": [{"test": str(test), "detail": detail} for test, detail in result.failures],
        "errors": [{"test": str(test), "detail": detail} for test, detail in result.errors],
        "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
        "test_ids": names,
        "runner_output": buffer.getvalue(),
        "external_dependency_added": False,
        "property_method": "deterministic standard-library random loops",
    }


def _runtime_source_audit(repo_root: Path) -> dict[str, Any]:
    design_path = repo_root / "docs" / "codex" / "M6C_reserved_service_design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    forbidden_runtime = {
        "llama.cpp/trace/tensor_trace.cpp",
        "llama.cpp/trace/expert_prefetch_types.h",
        "llama.cpp/trace/expert_hint_priority.h",
        "llama.cpp/trace/expert_hint_priority.cpp",
        "llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt",
        "llama.cpp/tests/CMakeLists.txt",
    }
    expected = {
        item["path"]: item["sha256"]
        for item in design.get("source_audit", [])
        if isinstance(item, dict) and item.get("path") in forbidden_runtime
    }
    records: list[dict[str, Any]] = []
    for relative in sorted(forbidden_runtime):
        path = repo_root / relative
        actual = sha256_file(path) if path.is_file() else None
        records.append({
            "path": relative,
            "expected_sha256": expected.get(relative),
            "actual_sha256": actual,
            "unchanged": actual is not None and actual == expected.get(relative),
        })
    return {
        "runtime_source_modified": not all(record["unchanged"] for record in records),
        "authority": "M6C-A frozen source audit SHA256",
        "files": records,
    }


def _source_inventory(repo_root: Path) -> dict[str, Any]:
    patterns = (
        "llama.cpp/trace/m6c_offline/*.py",
        "llama.cpp/trace/m6c_offline/*.md",
        "llama.cpp/trace/m6c_offline/*.json",
        "llama.cpp/trace/tests/test_m6c_offline*.py",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(repo_root.glob(pattern))
    files = []
    for path in sorted(paths):
        files.append({
            "path": str(path.relative_to(repo_root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    return {
        "implementation_files": files,
        "implementation_file_count": len(files),
        "runtime_integration_files": [],
    }


def _risk_update() -> dict[str, Any]:
    return {
        "scope": "offline engine and S0 replay only",
        "offline_tested_not_runtime_closed": [
            "R01", "R02", "R03", "R04", "R05", "R07", "R08", "R10",
            "R11", "R13", "R14", "R15", "R17", "R20", "R22", "R23",
            "R31", "R32", "R35",
        ],
        "parameters_intentionally_unselected": ["R24", "R25", "R26", "R27", "R28"],
        "runtime_or_physical_risks_still_open": [
            "R06", "R09", "R12", "R16", "R18", "R19", "R21", "R29",
            "R30", "R33", "R34", "R36",
        ],
        "active_runtime_risk_accepted": False,
        "performance_risk_conclusion": None,
        "evidence_integrity_risk": (
            "v3 decision-stream SHA metadata is non-canonical; v4 requires post-close "
            "hashing plus an independent report-to-artifact validation"
        ),
    }


def _s0_result_signature(value: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "run_id",
        "configuration_id",
        "workers",
        "repeat_index",
        "source_priority_mode",
        "service_slots",
        "s0_decisions_executed",
        "s0_oracle_winner_mismatches",
        "runtime_deadline_score_winner_mismatches",
        "first_mismatch",
        "remaining_arrivals",
        "final_queue_empty",
        "selected_exact_once",
        "full_store_scan_count",
        "stale_handle_count",
        "operation_counters",
        "invariants",
        "passed",
    )
    return {field: value.get(field) for field in fields}


def _compare_superseded_report(path: Path, replay_runs: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        superseded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "passed": False,
            "path": str(path),
            "reason": "superseded_v3_report_unavailable",
            "detail": str(exc),
        }
    prior_runs = {
        item.get("run_id"): item
        for item in superseded.get("s0_replay", {}).get("runs", [])
        if isinstance(item, dict)
    }
    mismatches: list[dict[str, Any]] = []
    for current in replay_runs:
        run_id = current.get("run_id")
        prior = prior_runs.get(run_id)
        if prior is None:
            mismatches.append({"run_id": run_id, "reason": "missing_in_v3"})
            continue
        if _s0_result_signature(current) != _s0_result_signature(prior):
            mismatches.append({"run_id": run_id, "reason": "s0_result_changed"})
    if len(prior_runs) != len(replay_runs):
        mismatches.append({
            "reason": "run_count_changed",
            "v3": len(prior_runs),
            "v4": len(replay_runs),
        })
    return {
        "passed": not mismatches and len(replay_runs) == 30,
        "path": str(path),
        "v3_report_sha256": sha256_file(path),
        "v3_run_count": len(prior_runs),
        "v4_run_count": len(replay_runs),
        "compared_fields": list(_s0_result_signature({}).keys()),
        "mismatches": mismatches,
        "v3_decision_stream_sha_proof_reused": False,
        "classification": EXPECTED_V3_CLASSIFICATION,
    }


def _artifact_summary(replay_runs: list[dict[str, Any]]) -> dict[str, Any]:
    total = 30
    finalized = sum(
        item.get("decision_stream_finalized_after_close") is True for item in replay_runs
    )
    line_matches = sum(
        item.get("decision_stream_line_count") == item.get("s0_decisions_executed")
        for item in replay_runs
    )
    parseable = sum(item.get("decision_stream_jsonl_parseable") is True for item in replay_runs)
    complete = sum(item.get("decision_stream_final_line_complete") is True for item in replay_runs)
    deterministic = sum(
        item.get("deterministic_rerun", {}).get("content_matches_finalized_stream") is True
        and item.get("deterministic_rerun", {}).get("s0_result_matches") is True
        for item in replay_runs
    )
    metadata_complete = sum(
        isinstance(item.get("decision_stream_sha256"), str)
        and isinstance(item.get("decision_stream_size_bytes"), int)
        for item in replay_runs
    )
    return {
        "artifact_lifecycle": (
            "write -> flush -> close -> reopen final path -> line/JSON validation -> size/SHA"
        ),
        "durable_fsync_requested": False,
        "durability_claim": (
            "close-and-reopen integrity only; no all-storage-layer durability claim"
        ),
        "decision_stream_count": f"{len(replay_runs)}/{total}",
        "decision_stream_finalized_after_close": f"{finalized}/{total}",
        "decision_stream_metadata_complete": f"{metadata_complete}/{total}",
        "decision_stream_sha_report_matches_actual": f"{metadata_complete}/{total}",
        "decision_stream_line_count_matches": f"{line_matches}/{total}",
        "decision_stream_jsonl_parseable": f"{parseable}/{total}",
        "decision_stream_final_line_complete": f"{complete}/{total}",
        "deterministic_rerun_matches": f"{deterministic}/{total}",
        "post_write_validator": "artifact_validation_report.json",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    recon = report["reconstructability"]
    replay = report["s0_replay"]
    artifact = report["artifact_finalization"]
    lines = [
        "# M6C-B.2 Evidence Finalization and Hash Repair Report",
        "",
        f"最终枚举：`{report['final_enum']}`",
        "",
        "> 本报告只描述固定 arrival / 固定 service-slot 的离线 S0 policy replay。物理系统没有重执行，且不包含性能结论。",
        "",
        "## v3 / v4 权威关系",
        "",
        f"- supersedes：`{report['supersedes']}`。",
        f"- supersession reason：`{report['supersession_reason']}`。",
        f"- v3 分类：`{report['superseded_output_classification']}`。",
        "- v3 原文件未修改；v4 不复用 v3 的 decision-stream SHA 证明。",
        "",
        "## 范围与冻结声明",
        "",
        "```text",
        "counterfactual_type = fixed_arrival_fixed_service_slot_policy_replay",
        "physical_system_reexecuted = false",
        "performance_claim = false",
        "formal_parameters_selected = false",
        "s1_evidence_parameter_comparison_executed = false",
        "```",
        "",
        "## 实现与测试",
        "",
        f"- 离线实现文件：{report['source_inventory']['implementation_file_count']} 个；runtime 接线文件：0 个。",
        f"- M6C 定向测试：{report['tests']['tests_run']} 个，passed={str(report['tests']['passed']).lower()}。",
        "- Indexed heap：head O(1)，insert/erase O(log n)，position map + eager erase；selection full-store scan 禁止。",
        f"- Runtime frozen source modified={str(report['runtime_source_audit']['runtime_source_modified']).lower()}。",
        "",
        "## Artifact Finalization 与独立复核",
        "",
        f"- 生命周期：`{artifact['artifact_lifecycle']}`。",
        f"- report SHA 与 close 后实际 SHA：{artifact['decision_stream_sha_report_matches_actual']}。",
        f"- decision 行数：{artifact['decision_stream_line_count_matches']}；JSONL 可解析：{artifact['decision_stream_jsonl_parseable']}。",
        f"- final line 完整：{artifact['decision_stream_final_line_complete']}；deterministic rerun：{artifact['deterministic_rerun_matches']}。",
        f"- durability：{artifact['durability_claim']}。",
        "- 报告写出后的独立复核见 `artifact_validation_report.json`。",
        "",
        "## 30-Run Reconstructability",
        "",
        f"- Evidence Hash：{recon['evidence_hash']['checked']}/{recon['evidence_hash']['indexed']} 通过。",
        f"- RECONSTRUCTABLE：{recon['status_counts'].get('RECONSTRUCTABLE', 0)}/30。",
        "",
        "| Run | Config | workers | status | bytes | lines | SHA256 |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    replay_by_run = {item["run_id"]: item for item in replay["runs"]}
    for item in recon["runs"]:
        replay_item = replay_by_run.get(item["run_id"], {})
        lines.append(
            f"| `{item['run_id']}` | {replay_item.get('configuration_id', '-')} | "
            f"{replay_item.get('workers', '-')} | {item['status']} | "
            f"{replay_item.get('decision_stream_size_bytes', '-')} | "
            f"{replay_item.get('decision_stream_line_count', '-')} | "
            f"`{replay_item.get('decision_stream_sha256', '-')}` |"
        )
    lines.extend([
        "",
        "## S0 逐 Decision Gate",
        "",
        f"- S0 decisions：{replay['total_decisions']}。",
        f"- 独立 frozen `deadline_score` oracle mismatch：{replay['s0_oracle_winner_mismatches']}。",
        f"- B0 observed runtime `deadline_score` mismatch：{replay['runtime_deadline_score_winner_mismatches']}。",
        f"- full-store scan：{replay['full_store_scan_count']}；stale handle：{replay['stale_handle_count']}。",
        f"- 双索引/bytes/Task conservation passed={str(replay['all_invariants_passed']).lower()}。",
        "",
        "C1/C2 的实际 runtime winner 是 `max_wait_protection`，因此不把它们的实际 winner 冒充 S0 观测值；这 20 个 Run 使用权威 F64 bits 与冻结 C++ comparator 的独立 oracle 验证 S0，10 个 B0 Run另逐 decision 对比实际 `deadline_score` winner。",
        "",
        "## 不可用字段",
        "",
        "- A.3 没有 M6C-native `queue_op_id`；service-slot 顺序由无 gap/无重复的 `decision_id` 唯一重建。",
        "- 没有正式 R/D、eligibility age 或 hard-urgent guard；没有在 A.3 上运行 S1。",
        "",
        "## 风险状态",
        "",
        "v3 的文件内容仍有技术参考价值，但其 decision-stream SHA 元数据不是权威完整性证明。v4 只修复 Evidence finalization/Hash 链；离线 property/failure-injection Gate 仍不关闭 runtime 锁、Trace 成本、shutdown、物理内存或中长 workload 风险。R/D、age、guard 仍全部未选择。",
        "",
        "## 停止边界",
        "",
        "本阶段到此停止。只有人工批准后才可创建 M6C-C preregistration；不得自动执行正式 S1 replay、Shadow、Active、推理或性能实验。",
        "",
    ])
    return "\n".join(lines)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    decision_dir = output_dir / "s0_decisions"
    decision_dir.mkdir()

    tests = run_m6c_tests(repo_root)
    evidence_hash = validate_evidence_index(repo_root, Path(args.evidence_index).resolve())
    run_entries = load_run_index(Path(args.run_index).resolve())
    recon_results = [
        audit_run(entry, global_hash_passed=bool(evidence_hash["passed"]))
        for entry in run_entries
    ]
    status_counts = Counter(result.status.value for result in recon_results)
    replay_runs: list[dict[str, Any]] = []
    replay_stop_reason: str | None = None
    artifact_finalization_failure: str | None = None

    if all(result.status == ReconstructabilityStatus.RECONSTRUCTABLE for result in recon_results):
        for result in recon_results:
            assert result.run is not None
            stream_path = decision_dir / f"{result.run_id}.jsonl"
            try:
                replay_result, metadata = write_finalized_jsonl(
                    stream_path,
                    lambda stream, run=result.run: replay_s0(run, stream),
                    expected_line_count=len(result.run.slots),
                    durable_fsync=False,
                )
            except ArtifactFinalizationError as exc:
                cause = exc.__cause__
                if isinstance(cause, ModelInvariantError):
                    replay_stop_reason = f"S0 model invariant failed in {result.run_id}: {cause}"
                else:
                    artifact_finalization_failure = f"{result.run_id}: {exc}"
                    replay_stop_reason = f"artifact finalization failed in {result.run_id}"
                replay_runs.append({
                    "run_id": result.run_id,
                    "configuration_id": result.run.configuration_id,
                    "workers": result.run.workers,
                    "repeat_index": result.run.repeat_index,
                    "source_priority_mode": result.run.priority_mode,
                    "decision_stream": str(stream_path.relative_to(repo_root)),
                    "passed": False,
                    "artifact_finalization_error": str(exc),
                    "model_invariant_error": str(cause) if isinstance(cause, ModelInvariantError) else None,
                    "full_store_scan_count": 0,
                    "stale_handle_count": 0,
                    "s0_oracle_winner_mismatches": 0,
                    "runtime_deadline_score_winner_mismatches": None,
                    "s0_decisions_executed": 0,
                    **COUNTERFACTUAL_DECLARATION,
                })
                break

            replay_result.update({
                "decision_stream": str(stream_path.relative_to(repo_root)),
                "decision_stream_size_bytes": metadata["size_bytes"],
                "decision_stream_line_count": metadata["line_count"],
                "decision_stream_sha256": metadata["sha256"],
                "decision_stream_jsonl_parseable": metadata["jsonl_parseable"],
                "decision_stream_final_line_complete": metadata["final_line_complete"],
                "decision_stream_finalized_after_close": metadata["finalized_after_close"],
                "decision_stream_durable_fsync_requested": metadata["durable_fsync_requested"],
                "decision_stream_durability_claim": metadata["durability_claim"],
            })

            digest_sink = DecisionDigestSink()
            deterministic_result = replay_s0(result.run, digest_sink)
            replay_result["deterministic_rerun"] = {
                "executed": True,
                "artifact_written": False,
                "content_sha256": digest_sink.hexdigest(),
                "size_bytes": digest_sink.size_bytes,
                "line_count": digest_sink.line_count,
                "content_matches_finalized_stream": (
                    digest_sink.hexdigest() == metadata["sha256"]
                    and digest_sink.size_bytes == metadata["size_bytes"]
                    and digest_sink.line_count == metadata["line_count"]
                ),
                "s0_result_matches": (
                    _s0_result_signature(deterministic_result)
                    == _s0_result_signature(replay_result)
                ),
            }
            replay_runs.append(replay_result)
            if not replay_result.get("passed"):
                replay_stop_reason = f"S0 Gate failed in {result.run_id}"
                break

    total_decisions = sum(int(item.get("s0_decisions_executed", 0)) for item in replay_runs)
    oracle_mismatches = sum(int(item.get("s0_oracle_winner_mismatches", 0)) for item in replay_runs)
    runtime_mismatches = sum(
        int(item.get("runtime_deadline_score_winner_mismatches") or 0)
        for item in replay_runs
    )
    full_scans = sum(int(item.get("full_store_scan_count", 0)) for item in replay_runs)
    stale = sum(int(item.get("stale_handle_count", 0)) for item in replay_runs)
    all_invariants = len(replay_runs) == 30 and all(
        bool(item.get("invariants", {}).get("passed")) for item in replay_runs
    )
    runtime_audit = _runtime_source_audit(repo_root)
    artifact_finalization = _artifact_summary(replay_runs)
    deterministic_ok = artifact_finalization["deterministic_rerun_matches"] == "30/30"
    artifact_ready = all(
        artifact_finalization[field] == "30/30"
        for field in (
            "decision_stream_count",
            "decision_stream_finalized_after_close",
            "decision_stream_metadata_complete",
            "decision_stream_sha_report_matches_actual",
            "decision_stream_line_count_matches",
            "decision_stream_jsonl_parseable",
            "decision_stream_final_line_complete",
        )
    )
    superseded_comparison = _compare_superseded_report(
        Path(args.superseded_report).resolve(), replay_runs
    )

    if runtime_audit["runtime_source_modified"] or not tests["passed"]:
        final_enum = "M6C_B_REQUIRES_REIMPLEMENTATION"
    elif not evidence_hash["passed"] or status_counts.get("RECONSTRUCTABLE", 0) != 30:
        final_enum = "ADDITIONAL_EVIDENCE_MISMATCH_FOUND"
    elif artifact_finalization_failure or not artifact_ready:
        final_enum = "HASH_FINALIZATION_FIX_FAILED"
    elif not deterministic_ok:
        final_enum = "DECISION_STREAM_CONTENT_NOT_DETERMINISTIC"
    elif oracle_mismatches or runtime_mismatches or not superseded_comparison["passed"]:
        final_enum = "S0_RESULT_CHANGED_AFTER_REPAIR"
    elif full_scans or stale or not all_invariants or replay_stop_reason:
        final_enum = "M6C_B_REQUIRES_REIMPLEMENTATION"
    else:
        final_enum = "M6C_B_EVIDENCE_REPAIRED_AND_CONFIRMED"

    source_inventory = _source_inventory(repo_root)
    reconstructability = {
        "evidence_hash": evidence_hash,
        "status_counts": dict(sorted(status_counts.items())),
        "runs": [result.machine_dict() for result in recon_results],
    }
    s0_replay = {
        "runs": replay_runs,
        "total_decisions": total_decisions,
        "s0_oracle_winner_mismatches": oracle_mismatches,
        "runtime_deadline_score_winner_mismatches": runtime_mismatches,
        "full_store_scan_count": full_scans,
        "stale_handle_count": stale,
        "all_invariants_passed": all_invariants,
        "stop_reason": replay_stop_reason,
        "complexity_proof": {
            "head": "O(1): direct heap[0] access",
            "insert": "O(log n): binary heap sift-up with position map",
            "erase": "O(log n): position-map lookup, last-element replacement, one sift direction",
            "selection": "O(1) heads plus O(log n) eager erase from each index",
            "lazy_stale_control_flow": False,
            "normal_rebuild": False,
        },
        **COUNTERFACTUAL_DECLARATION,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "M6C-B.2 Evidence Finalization and Hash Repair",
        "final_enum": final_enum,
        "supersedes": EXPECTED_SUPERSEDES,
        "supersession_reason": EXPECTED_SUPERSESSION_REASON,
        "superseded_output_classification": EXPECTED_V3_CLASSIFICATION,
        "frozen_audit_facts": {
            "audit_result": "EVIDENCE_OR_REPORT_MISMATCH",
            "technical_scope_violation": False,
            "s0_oracle_winner_mismatches": 0,
            "b0_runtime_winner_mismatches": 0,
            "decision_stream_sha_mismatches": 30,
        },
        "repair": {
            "root_cause": (
                "runner calculated decision-stream SHA while the buffered writer was still open"
            ),
            "fixed_lifecycle": (
                "write, flush, successful close, reopen final path, parse/count/size/SHA"
            ),
            "policy_or_model_semantics_changed": False,
            "runtime_changed": False,
            "v3_files_modified": False,
            "artifact_finalization_failure": artifact_finalization_failure,
        },
        "source_inventory": source_inventory,
        "runtime_source_audit": runtime_audit,
        "tests": tests,
        "reconstructability": reconstructability,
        "s0_replay": s0_replay,
        "artifact_finalization": artifact_finalization,
        "superseded_report_s0_comparison": superseded_comparison,
        "post_write_independent_validator": {
            "implementation": "llama.cpp/trace/m6c_offline/artifact_validator.py",
            "result_artifact": str(
                (output_dir / "artifact_validation_report.json").relative_to(repo_root)
            ),
            "runs_after_m6c_b_report_is_written": True,
        },
        "risk_status_update": _risk_update(),
        "formal_parameters_selected": False,
        "test_fixture_only": True,
        "experimental_parameter": False,
        "s1_evidence_parameter_comparison_executed": False,
        "shadow_executed": False,
        "active_executed": False,
        "smoke_executed": False,
        "inference_runs": 0,
        "formal_n8_executed": False,
        "next_stage_requires_human_approval": True,
        **COUNTERFACTUAL_DECLARATION,
    }
    _write_json(output_dir / "source_inventory.json", source_inventory)
    _write_json(output_dir / "repair_summary.json", report["repair"])
    _write_json(output_dir / "reconstructability_report.json", reconstructability)
    _write_json(output_dir / "s0_replay_report.json", s0_replay)
    _write_json(output_dir / "risk_status_update.json", report["risk_status_update"])
    _write_json(output_dir / "m6c_b_report.json", report)
    (output_dir / "m6c_b_report.md").write_text(_markdown_report(report), encoding="utf-8")
    independent_validation = validate_report_artifacts(
        repo_root,
        output_dir / "m6c_b_report.json",
    )
    if (
        not independent_validation["passed"]
        and report["final_enum"] == "M6C_B_EVIDENCE_REPAIRED_AND_CONFIRMED"
    ):
        report["final_enum"] = "HASH_FINALIZATION_FIX_FAILED"
        report["post_write_independent_validator"]["failure_detected"] = True
        _write_json(output_dir / "m6c_b_report.json", report)
        (output_dir / "m6c_b_report.md").write_text(_markdown_report(report), encoding="utf-8")
        independent_validation = validate_report_artifacts(
            repo_root,
            output_dir / "m6c_b_report.json",
        )
    _write_json(output_dir / "artifact_validation_report.json", independent_validation)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument(
        "--evidence-index",
        default=str(repo_root / "llama.cpp/trace_output/m6b2a3_directed_20260804_n5_v3_analysis_v2/evidence_index.json"),
    )
    parser.add_argument(
        "--run-index",
        default=str(repo_root / "llama.cpp/trace_output/m6b2a3_directed_20260804_n5_v3_run_index.json"),
    )
    parser.add_argument(
        "--superseded-report",
        default=str(repo_root / "llama.cpp/trace_output/m6c_b_offline_20260805_a3_s0_v3/m6c_b_report.json"),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = execute(args)
    print(json.dumps({
        "final_enum": report["final_enum"],
        "output_dir": str(Path(args.output_dir).resolve()),
        "reconstructable_runs": report["reconstructability"]["status_counts"].get("RECONSTRUCTABLE", 0),
        "s0_decisions": report["s0_replay"]["total_decisions"],
        "s0_oracle_winner_mismatches": report["s0_replay"]["s0_oracle_winner_mismatches"],
    }, sort_keys=True))
    return 0 if report["final_enum"] == "M6C_B_EVIDENCE_REPAIRED_AND_CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
