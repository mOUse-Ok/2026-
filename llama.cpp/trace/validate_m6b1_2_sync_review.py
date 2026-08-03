#!/usr/bin/env python3
"""Validate the M6B1.2 single- and fixed-multithread bit-stability gate."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from typing import Any

import audit_router_score_determinism as audit


SCHEMA_VERSION = "m6b1.2-router-sync-review-v1"
SYNC_NAME = "producer_barrier_hook_release_barrier"
SYNC_PROTOCOL = "m6b1.2-v1"


def pair_gate(pair: dict[str, Any]) -> dict[str, Any]:
    replays = (
        pair["replay_a_with_b_scores"], pair["replay_b_with_a_scores"]
    )
    checks = {
        "pair_valid": bool(pair["valid"]),
        "zero_unmatched_tasks": (
            pair["matching"]["unmatched_a_tasks"] == 0
            and pair["matching"]["unmatched_b_tasks"] == 0
        ),
        "zero_unmatched_routes": (
            pair["matching"]["unmatched_a_route_records"] == 0
            and pair["matching"]["unmatched_b_route_records"] == 0
        ),
        "raw_f32_bits_equal": pair["router"]["raw_bit_different_items"] == 0,
        "read_f32_bits_equal": pair["router"]["f32_bit_different_items"] == 0,
        "topk_set_equal": pair["router"]["topk_set_changed_records"] == 0,
        "topk_rank_equal": pair["router"]["topk_rank_changed_items"] == 0,
        "task_f64_bits_equal": pair["tasks"]["score_f64_bit_different_tasks"] == 0,
        "serialized_task_scores_equal": (
            pair["tasks"]["serialized_score_different_tasks"] == 0
        ),
        "source_to_task_exact_widening": (
            pair["tasks"]["source_to_task_widening_mismatches"] == 0
        ),
        "lifecycle_score_bits_stable": (
            pair["tasks"]["lifecycle_score_bit_mismatches"] == 0
        ),
        "same_run_replays_passed": all(replay["passed"] for replay in replays),
        "zero_self_replay_mismatches": all(
            replay["self_replay_winner_mismatches"] == 0 for replay in replays
        ),
        "zero_score_injection_winner_changes": all(
            replay["score_injection_winner_changed_decisions"] == 0
            for replay in replays
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def run_gate(run: audit.AuditRun, expected_threads: str, affinity: str) -> dict[str, Any]:
    environment = run.manifest.get("environment", {})
    route_sync_complete = bool(run.routes) and all(
        route.get("observation_sync") == SYNC_NAME
        and route.get("observation_sync_protocol") == SYNC_PROTOCOL
        for route in run.routes
    )
    checks = {
        "run_valid": bool(run.validation["passed"]),
        "deadline_score_mode": run.mode == "deadline_score",
        "model_threads_frozen": run.model_threads == expected_threads,
        "hint_workers_frozen": run.hint_workers == "2",
        "affinity_manifest_frozen": (
            environment.get("LLM_MEM_TRACE_AUDIT_CPU_AFFINITY") == affinity
        ),
        "router_score_diagnostic_enabled": (
            environment.get("LLM_MEM_TRACE_ROUTER_SCORE_DIAGNOSTIC") == "1"
        ),
        "sync_protocol_manifest": (
            environment.get("LLM_MEM_TRACE_ROUTER_TENSOR_SYNC_PROTOCOL")
            == SYNC_PROTOCOL
        ),
        "sync_protocol_trace": route_sync_complete,
        "trace_zero_drop": bool(run.validation["trace_complete"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# M6B1.2 Router Tensor 同模式稳定性复审",
        "",
        f"> Review ID：`{summary['review_id']}`",
        "",
        f"Gate：**{'PASS' if summary['passed'] else 'FAIL'}**。",
        "",
        "本结果只验证同步修复后的 Router/Task bit 稳定性，不形成性能结论。",
        "",
        "## Run",
        "",
        "| Label | Threads | Hint workers | Valid | Sync Trace | Output Hash |",
        "|---|---:|---:|---|---|---|",
    ]
    for run in summary["runs"]:
        gate = summary["run_gates"][run["label"]]
        lines.append(
            f"| {run['label']} | {run['model_threads']} | {run['hint_workers']} | "
            f"{gate['checks']['run_valid']} | {gate['checks']['sync_protocol_trace']} | "
            f"`{run['output_sha256']}` |"
        )
    lines.extend([
        "",
        "## Pair",
        "",
        "| Pair | Raw F32 mismatch | Task F64 mismatch | Top-K set/rank | Replay mismatch | Pass |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for pair in summary["pairs"]:
        replay_mismatch = sum(
            pair[side]["self_replay_winner_mismatches"]
            for side in ("replay_a_with_b_scores", "replay_b_with_a_scores")
        )
        lines.append(
            f"| {pair['comparison_id']} | {pair['router']['raw_bit_different_items']} | "
            f"{pair['tasks']['score_f64_bit_different_tasks']} | "
            f"{pair['router']['topk_set_changed_records']} / "
            f"{pair['router']['topk_rank_changed_items']} | {replay_mismatch} | "
            f"{summary['pair_gates'][pair['comparison_id']]['passed']} |"
        )
    lines.extend([
        "",
        "## 判定",
        "",
        "只有本 Gate PASS，才允许执行 workers=2 strict A/B Smoke。任意 raw F32 或 Task F64 bit mismatch 都是零容忍失败。",
        "",
        "## Evidence",
        "",
        "- `sync_review.json`：机器 Gate；",
        "- `task_correspondence.jsonl`：全部 Task 对应；",
        "- `router_score_differences.jsonl`：Router 差异（PASS 时为空）；",
        "- 各 Run 目录：manifest、raw Trace、output Hash、metrics。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(
    review_id: str,
    single_dirs: list[Path],
    multi_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    if len(single_dirs) != 2 or len(multi_dirs) != 3:
        raise ValueError("M6B1.2 requires exactly 2 single-thread and 3 fixed-multithread runs")

    labels_and_paths = [
        ("ST_1", single_dirs[0]), ("ST_2", single_dirs[1]),
        ("MT_1", multi_dirs[0]), ("MT_2", multi_dirs[1]),
        ("MT_3", multi_dirs[2]),
    ]
    runs = {
        label: audit.load_run(label, path.resolve())
        for label, path in labels_and_paths
    }
    comparison_specs = (
        ("B_SINGLE_THREAD", "ST_1", "ST_2"),
        ("C_FIXED_MULTI_12", "MT_1", "MT_2"),
        ("C_FIXED_MULTI_13", "MT_1", "MT_3"),
        ("C_FIXED_MULTI_23", "MT_2", "MT_3"),
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    pairs: list[dict[str, Any]] = []
    with ExitStack() as stack:
        route_stream = stack.enter_context(
            (output_dir / "router_score_differences.jsonl").open("w", encoding="utf-8")
        )
        task_stream = stack.enter_context(
            (output_dir / "task_correspondence.jsonl").open("w", encoding="utf-8")
        )
        for comparison_id, left, right in comparison_specs:
            pairs.append(audit.compare_pair(
                comparison_id, runs[left], runs[right], route_stream, task_stream
            ))

    run_gates = {
        label: run_gate(
            run,
            "1" if label.startswith("ST_") else "8",
            "0" if label.startswith("ST_") else "0-7",
        )
        for label, run in runs.items()
    }
    pair_gates = {
        pair["comparison_id"]: pair_gate(pair) for pair in pairs
    }
    binary_hashes = {run.manifest.get("binary", {}).get("sha256") for run in runs.values()}
    model_hashes = {run.manifest.get("model", {}).get("sha256") for run in runs.values()}
    prompt_hashes = {run.manifest.get("prompt", {}).get("sha256") for run in runs.values()}
    output_hashes = {run.validation["output_sha256"] for run in runs.values()}
    global_checks = {
        "one_binary_hash": len(binary_hashes) == 1,
        "one_model_hash": len(model_hashes) == 1,
        "one_prompt_hash": len(prompt_hashes) == 1,
        "one_output_hash": len(output_hashes) == 1,
        "all_runs_passed": all(gate["passed"] for gate in run_gates.values()),
        "all_pairs_passed": all(gate["passed"] for gate in pair_gates.values()),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "review_id": review_id,
        "passed": all(global_checks.values()),
        "global_checks": global_checks,
        "runs": [audit.run_identity(runs[label]) for label, _ in labels_and_paths],
        "run_gates": run_gates,
        "pairs": pairs,
        "pair_gates": pair_gates,
        "sync_contract": {
            "name": SYNC_NAME,
            "protocol": SYNC_PROTOCOL,
            "participants": "active_graph_compute_threads",
            "hook_position": "between_producer_and_release_barriers",
        },
        "next_gate": "workers_2_strict_ab" if all(global_checks.values()) else "stop",
    }
    (output_dir / "sync_review.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(summary, output_dir / "sync_review.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--single", action="append", required=True, type=Path)
    parser.add_argument("--multi", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = validate(
        args.review_id, args.single, args.multi, args.output_dir.resolve()
    )
    print(json.dumps({
        "passed": summary["passed"],
        "next_gate": summary["next_gate"],
        "raw_router_mismatches": sum(
            pair["router"]["raw_bit_different_items"] for pair in summary["pairs"]
        ),
        "task_f64_mismatches": sum(
            pair["tasks"]["score_f64_bit_different_tasks"] for pair in summary["pairs"]
        ),
    }, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit("M6B1.2 sync review failed; evidence was retained")


if __name__ == "__main__":
    main()
