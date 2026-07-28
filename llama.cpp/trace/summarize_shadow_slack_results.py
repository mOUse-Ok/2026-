#!/usr/bin/env python3
"""Build the machine- and human-readable M4A.1 acceptance report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


SUMMARY_METRICS = (
    "process_wall_time_s",
    "decode_avg_latency_us",
    "decode_p50_latency_us",
    "decode_p95_latency_us",
    "decode_p99_latency_us",
    "decode_throughput_tokens_per_s",
    "trace_bytes",
)
DEFAULT_PRIMARY = (
    "phase_layer_stage_p25|queue_depth_worker_ewma|residual_quantile"
)


def ms(value: float | int | None) -> str:
    return "null" if value is None else f"{float(value) / 1e6:.3f}"


def ratio(value: float | None) -> str:
    return "null" if value is None else f"{value:.4f}"


def pct(value: float | None) -> str:
    return "null" if value is None else f"{100.0 * value:.2f}%"


def metric(container: dict[str, Any], key: str) -> dict[str, Any]:
    return container.get(key, {"count": 0})


def load_summary_run(run_dir: Path) -> dict[str, Any]:
    metrics = json.loads((run_dir / "analysis/metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    cache = json.loads((run_dir / "cache_preparation.json").read_text(encoding="utf-8"))
    sink_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    environment = manifest.get("environment", {})
    experiment = manifest.get("experiment", {})
    sinks = sink_summary.get("sinks", {})
    return {
        "run_dir": str(run_dir.resolve()),
        "run_name": manifest.get("run_name", run_dir.name),
        "repeat_index": experiment.get("repeat_index"),
        "order_position": experiment.get("order_position"),
        "active_workers": int(environment.get(
            "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS", 0
        )),
        "shadow_mode": environment.get("LLM_MEM_TRACE_OPT_EXPERT_SLACK_MODE", "off"),
        "process_wall_time_s": metrics.get("process_wall_time_s"),
        "decode_avg_latency_us": metrics.get("decode_avg_latency_us"),
        "decode_p50_latency_us": metrics.get("decode_p50_latency_us"),
        "decode_p95_latency_us": metrics.get("decode_p95_latency_us"),
        "decode_p99_latency_us": metrics.get("decode_p99_latency_us"),
        "decode_throughput_tokens_per_s": metrics.get("decode_throughput_tokens_per_s"),
        "trace_bytes": sum(path.stat().st_size for path in run_dir.glob("*_trace.jsonl")),
        "output_sha256": (run_dir / "output.sha256").read_text(encoding="utf-8").strip(),
        "binary_sha256": manifest.get("binary", {}).get("sha256"),
        "git_dirty": manifest.get("git_dirty"),
        "cache_preparation": cache,
        "cgroup": experiment.get("cgroup", {}),
        "trace_integrity": all(
            not sink.get("enabled")
            or (sink.get("enqueued") == sink.get("written") and sink.get("dropped") == 0)
            for sink in sinks.values()
        ),
        "sinks": sinks,
    }


def medians(runs: list[dict[str, Any]]) -> dict[str, float | None]:
    return {
        field: (
            statistics.median([
                float(run[field]) for run in runs
                if isinstance(run.get(field), (int, float))
            ])
            if any(isinstance(run.get(field), (int, float)) for run in runs)
            else None
        )
        for field in SUMMARY_METRICS
    }


def summary_comparison(off_dirs: list[Path], shadow_dirs: list[Path]) -> dict[str, Any]:
    off = [load_summary_run(path) for path in off_dirs]
    shadow = [load_summary_run(path) for path in shadow_dirs]
    workers = sorted({run["active_workers"] for run in off + shadow})
    by_workers: dict[str, Any] = {}
    for worker in workers:
        off_worker = [run for run in off if run["active_workers"] == worker]
        shadow_worker = [run for run in shadow if run["active_workers"] == worker]
        off_median = medians(off_worker)
        shadow_median = medians(shadow_worker)
        by_workers[str(worker)] = {
            "off_runs": off_worker,
            "shadow_runs": shadow_worker,
            "off_median": off_median,
            "shadow_median": shadow_median,
            "relative_delta": {
                field: (
                    (shadow_median[field] - off_median[field]) / off_median[field]
                    if off_median[field] not in (None, 0)
                    and shadow_median[field] is not None
                    else None
                )
                for field in SUMMARY_METRICS
            },
        }
    return {
        "off_runs": off,
        "shadow_runs": shadow,
        "by_active_workers": by_workers,
        "same_binary": len({run["binary_sha256"] for run in off + shadow}) == 1,
        "all_cold_cache": all(
            run["cache_preparation"].get("mode") == "cold" for run in off + shadow
        ),
        "all_zero_drop": all(run["trace_integrity"] for run in off + shadow),
        "delegated_cgroup": all(
            run["cgroup"].get("memory.max") not in (None, "max") for run in off + shadow
        ),
    }


def conservative_decode_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    qualified: list[dict[str, Any]] = []
    for name, candidate in result.get("candidates", {}).items():
        for target in ("issue", "return"):
            scopes = candidate[target].get("mature_by_phase_workers", {})
            selections = []
            for workers in (2, 4):
                scope = scopes.get(f"DECODE|workers={workers}", {})
                rows = [
                    row for row in scope.get("negative_thresholds", [])
                    if row.get("predicted_late_precision") is not None
                    and row["predicted_late_precision"] >= 0.90
                    and row.get("predicted_late_count", 0) > 0
                ]
                if not rows:
                    selections = []
                    break
                selections.append({"workers": workers, "threshold": rows[0]})
            if len(selections) == 2:
                qualified.append({
                    "candidate": name,
                    "target": target,
                    "workers": selections,
                })
    return qualified


def candidate_table(lines: list[str], candidates: dict[str, Any]) -> None:
    lines.extend((
        "| Candidate | First-use MAE ms | Issue late precision | Issue late recall | Return late precision | Return late recall | Issue fallback | Return fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for name, candidate in sorted(candidates.items()):
        first = candidate["first_use"]["operational"]
        issue = candidate["issue"]["operational"]
        returned = candidate["return"]["operational"]
        lines.append(
            f"| `{name}` | {ms(first.get('mae_ns'))} | "
            f"{ratio(issue.get('predicted_late_precision'))} | "
            f"{ratio(issue.get('predicted_late_recall'))} | "
            f"{ratio(returned.get('predicted_late_precision'))} | "
            f"{ratio(returned.get('predicted_late_recall'))} | "
            f"{issue.get('fallback_count', 0)} | {returned.get('fallback_count', 0)} |"
        )
    lines.append("")


def target_table(lines: list[str], title: str, target: dict[str, Any]) -> None:
    lines.extend((
        f"### {title}",
        "",
        "| Scope | N | Unavailable | Mature | TP | TN | FP | FN | Late precision | Late recall | Late F1 | False-reject rate | Late prevalence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    scopes = {
        "Overall": target["operational"],
        "PREFILL": metric(target["by_phase"], "PREFILL"),
        "DECODE": metric(target["by_phase"], "DECODE"),
        "EARLY": metric(target["by_stage"], "EARLY"),
        "LATE": metric(target["by_stage"], "LATE"),
        "workers=2": metric(target["by_active_workers"], "2"),
        "workers=4": metric(target["by_active_workers"], "4"),
        "DECODE,w=2": metric(target["by_phase_workers"], "DECODE|workers=2"),
        "DECODE,w=4": metric(target["by_phase_workers"], "DECODE|workers=4"),
        "Mature DECODE,w=2": metric(
            target["mature_by_phase_workers"], "DECODE|workers=2"
        ),
        "Mature DECODE,w=4": metric(
            target["mature_by_phase_workers"], "DECODE|workers=4"
        ),
        "Fallback DECODE,w=2": metric(
            target["fallback_by_phase_workers"], "DECODE|workers=2"
        ),
        "Fallback DECODE,w=4": metric(
            target["fallback_by_phase_workers"], "DECODE|workers=4"
        ),
    }
    mature_count = target["mature_exact"].get("count", 0)
    for scope, value in scopes.items():
        lines.append(
            f"| {scope} | {value.get('count', 0)} | {value.get('unavailable_count', 0)} | "
            f"{mature_count if scope == 'Overall' else '-'} | "
            f"{value.get('true_positive', 0)} | {value.get('true_negative', 0)} | "
            f"{value.get('false_positive', 0)} | {value.get('false_negative', 0)} | "
            f"{ratio(value.get('predicted_late_precision'))} | "
            f"{ratio(value.get('predicted_late_recall'))} | "
            f"{ratio(value.get('predicted_late_f1'))} | "
            f"{pct(value.get('false_reject_candidate_rate'))} | "
            f"{pct(value.get('late_prevalence'))} |"
        )
    lines.append("")


def oracle_diagnosis(
    oracle: dict[str, Any], target: str, dimension: str, scope: str
) -> list[tuple[str, float | None]]:
    suffix = "+predicted_syscall_service" if target == "return" else ""
    base_name = "predicted_first_use+predicted_queue+predicted_pre_issue" + suffix
    replacements = {
        "first-use": "actual_first_use+predicted_queue+predicted_pre_issue" + suffix,
        "queue wait": "predicted_first_use+actual_queue+predicted_pre_issue" + suffix,
        "pre-issue": "predicted_first_use+predicted_queue+actual_pre_issue" + suffix,
    }
    if target == "return":
        replacements["syscall service"] = (
            "predicted_first_use+predicted_queue+predicted_pre_issue"
            "+actual_syscall_service"
        )
    base = oracle.get(base_name, {}).get(dimension, {}).get(scope, {}).get("mae_ns")
    return [
        (
            component,
            base - oracle.get(name, {}).get(dimension, {}).get(scope, {}).get("mae_ns")
            if base is not None
            and oracle.get(name, {}).get(dimension, {}).get(scope, {}).get("mae_ns")
            is not None
            else None,
        )
        for component, name in replacements.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--final-json", required=True, type=Path)
    parser.add_argument("--detail-run-dir", action="append", required=True, type=Path)
    parser.add_argument("--summary-off", action="append", default=[], type=Path)
    parser.add_argument("--summary-shadow", action="append", default=[], type=Path)
    parser.add_argument("--equivalence", action="append", default=[], type=Path)
    args = parser.parse_args()
    if len(args.summary_off) != len(args.summary_shadow) or not args.summary_off:
        raise SystemExit("summary off/shadow lists must be non-empty and equally sized")

    detail_payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = detail_payload["combined"]
    summary = summary_comparison(args.summary_off, args.summary_shadow)
    equivalence = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.equivalence
    ]
    qualified = conservative_decode_candidates(result)
    semantic_ok = (
        result.get("semantic_violations", 1) == 0
        and result.get("timestamp_regressions", 1) == 0
        and result.get("causality_errors", 1) == 0
    )
    model_verdict = (
        "仅 Decode 候选可能可用，等待人工批准"
        if semantic_ok and qualified
        else "需要继续改进 Shadow"
    )
    all_equivalent = bool(equivalence) and all(item.get("passed") for item in equivalence)
    primary_name = DEFAULT_PRIMARY if DEFAULT_PRIMARY in result["candidates"] else next(
        iter(result["candidates"]), ""
    )
    primary = result["candidates"].get(primary_name, {})
    oracle_primary_name = primary_name.replace("|residual_quantile", "|raw")
    oracle_primary = result.get("oracle_attribution", {}).get(oracle_primary_name, {})
    detail_bytes = sum(
        path.stat().st_size
        for directory in args.detail_run_dir
        for path in directory.glob("*_trace.jsonl")
    )
    runtime_summaries = {
        run_id: value.get("runtime_summary")
        for run_id, value in detail_payload.get("per_run", {}).items()
    }

    final_payload = {
        "schema_version": 3,
        "milestone": "M4A.1",
        "artifacts": {
            "detail_model_comparison_json": str(args.input.resolve()),
            "human_summary_markdown": str(args.output.resolve()),
            "final_machine_report_json": str(args.final_json.resolve()),
            "detail_run_dirs": [str(path.resolve()) for path in args.detail_run_dir],
        },
        "prediction_targets": result.get("prediction_targets"),
        "semantic_audit": {
            "passed": semantic_ok,
            "semantic_violations": result.get("semantic_violations"),
            "causality_errors": result.get("causality_errors"),
            "timestamp_regressions": result.get("timestamp_regressions"),
            "missing_time_components": result.get("missing_time_components"),
            "multi_syscall_audit": result.get("multi_syscall_audit"),
        },
        "runtime_summaries": runtime_summaries,
        "detail_trace_bytes": detail_bytes,
        "all_model_results": result.get("candidates"),
        "oracle_attribution": result.get("oracle_attribution"),
        "queue_models": result.get("queue_models"),
        "duration_models": result.get("duration_models"),
        "time_decomposition": result.get("time_decomposition"),
        "stage_comparisons": result.get("stage_comparisons"),
        "summary_off_on": summary,
        "equivalence": equivalence,
        "all_equivalence_checks_passed": all_equivalent,
        "decode_control_candidates": qualified,
        "acceptance": {
            "engineering": "通过" if semantic_ok and all_equivalent else "未通过",
            "model": model_verdict,
            "enter_m4b": False,
            "human_approval_required": True,
        },
        "limitations": [
            "Shadow-only evidence; no performance benefit is claimed",
            "Oracle substitutions are offline diagnostics and never runtime inputs",
            "fallback samples are reported separately and excluded from mature_exact",
            (
                "no delegated cgroup; formal isolation is unavailable"
                if not summary["delegated_cgroup"]
                else "delegated cgroup limits were recorded"
            ),
            "ISSUE and RETURN are not page-in completion or physical residency",
        ],
    }

    lines = [
        "# M4A.1 Shadow Slack 语义对齐、误差归因与在线校准报告",
        "",
        "本报告只评价 Shadow 预测、记录和离线校准；没有结果写回 Comparator、Admission、Task 集合、Hint 或其他运行时决策。",
        "",
        "## 1. 验收结论",
        "",
        f"- 工程验收：**{final_payload['acceptance']['engineering']}**；",
        f"- 模型结论：**{model_verdict}**；",
        "- M4B：**未进入**，任何 Active Control 都需要后续人工批准；",
        f"- Detail Task={result.get('valid_unique_tasks', 0)}，候选={result.get('candidate_count', 0)}，workers 分层包含 2 和 4；",
        f"- semantic/causality/timestamp violations={result.get('semantic_violations', 0)}/{result.get('causality_errors', 0)}/{result.get('timestamp_regressions', 0)}。",
        "",
        "## 2. 唯一预测目标与时间组成",
        "",
        "- Issue Slack：`predicted_first_use - prediction_ts - queue_wait - pre_issue_overhead`；实际标签仅为 `issue_ts < logical_first_use_ts`。",
        "- Return Slack：在 Issue 公式上再减 `hint_syscall_service`；实际标签仅为 `final_enabled_hint_return_ts < logical_first_use_ts`。",
        "- Queue wait=`ENQUEUE→DEQUEUE`；pre-issue=`DEQUEUE→ISSUE`；syscall service=`ISSUE→RETURN`；worker occupied=`DEQUEUE→RETURN`。",
        "- ISSUE/RETURN 均不代表 page-in complete，RETURN 也不代表页面已驻留。",
        "",
        "| Actual component | N | Mean ms | Median ms | p95 ms | Max ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in result.get("time_decomposition", {}).items():
        lines.append(
            f"| {name} | {value.get('count', 0)} | {ms(value.get('mean_ns'))} | "
            f"{ms(value.get('median_ns'))} | {ms(value.get('p95_ns'))} | "
            f"{ms(value.get('max_ns'))} |"
        )
    lines.extend((
        "",
        "## 3. 全部模型结果",
        "",
        "下表同时列出 raw 与仅依赖更早成熟样本的 residual-quantile 版本。late precision 是 Active Control 最关键的方向性指标。",
        "",
    ))
    candidate_table(lines, result.get("candidates", {}))

    if primary:
        lines.extend((
            "## 4. Issue / Return 分类与 Calibration",
            "",
            f"详细展示候选 `{primary_name}`；完整 36 候选、PREFILL/DECODE、EARLY/LATE、workers=2/4 及交叉分层保存在机器 JSON。",
            "",
        ))
        target_table(lines, "Issue Slack", primary["issue"])
        target_table(lines, "Return Slack", primary["return"])
        for target_name, target in (("Issue", primary["issue"]), ("Return", primary["return"])):
            lines.extend((
                f"### {target_name} threshold 附近 Calibration",
                "",
                "| Slack bucket | N | Actual on-time rate | Predicted-late precision |",
                "|---|---:|---:|---:|",
            ))
            for label, value in target["operational"].get("calibration", {}).items():
                lines.append(
                    f"| {label} | {value.get('count', 0)} | "
                    f"{pct(value.get('actual_on_time_rate'))} | "
                    f"{ratio(value.get('predicted_late_precision'))} |"
                )
            monotonic = target["operational"].get("calibration_monotonicity", {})
            lines.extend((
                "",
                f"非空桶={monotonic.get('nonempty_bucket_count', 0)}，相邻反向={monotonic.get('adjacent_decrease_count', 0)}，nondecreasing={monotonic.get('nondecreasing')}。",
                "",
            ))

    lines.extend((
        "## 5. Oracle 误差归因",
        "",
        f"对 raw Queue A 候选 `{oracle_primary_name}`，Issue 有 8 个 H/Q/P 替换组合，Return 有 16 个 H/Q/P/S 替换组合；full oracle 必须为零误差。每个组合在 JSON 中保留 unavailable/fallback/mature，并按 phase、stage、workers 分层。Oracle 只用于离线诊断，不进入运行时。",
        "",
        "| Target | Oracle combination | Overall MAE ms | PREFILL | DECODE | EARLY | LATE | w2 | w4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for target in ("issue", "return"):
        for name, value in sorted(oracle_primary.get(target, {}).items()):
            lines.append(
                f"| {target} | {name} | {ms(value.get('overall', {}).get('mae_ns'))} | "
                f"{ms(value.get('by_phase', {}).get('PREFILL', {}).get('mae_ns'))} | "
                f"{ms(value.get('by_phase', {}).get('DECODE', {}).get('mae_ns'))} | "
                f"{ms(value.get('by_stage', {}).get('EARLY', {}).get('mae_ns'))} | "
                f"{ms(value.get('by_stage', {}).get('LATE', {}).get('mae_ns'))} | "
                f"{ms(value.get('by_active_workers', {}).get('2', {}).get('mae_ns'))} | "
                f"{ms(value.get('by_active_workers', {}).get('4', {}).get('mae_ns'))} |"
            )
    lines.append("")
    for target in ("issue", "return"):
        for phase in ("PREFILL", "DECODE"):
            diagnosis = oracle_diagnosis(
                oracle_primary.get(target, {}), target, "by_phase", phase
            )
            ordered = sorted(
                diagnosis,
                key=lambda item: float("-inf") if item[1] is None else item[1],
                reverse=True,
            )
            lines.append(
                f"- {target}/{phase} 单项替换的 MAE 改善：" + ", ".join(
                    f"{name}={ms(value)} ms" for name, value in ordered
                ) + (f"；主要瓶颈为 {ordered[0][0]}。" if ordered else "")
            )
        for workers in ("2", "4"):
            diagnosis = oracle_diagnosis(
                oracle_primary.get(target, {}), target,
                "by_active_workers", workers,
            )
            ordered = sorted(
                diagnosis,
                key=lambda item: float("-inf") if item[1] is None else item[1],
                reverse=True,
            )
            lines.append(
                f"- {target}/workers={workers} 单项替换的 MAE 改善：" + ", ".join(
                    f"{name}={ms(value)} ms" for name, value in ordered
                ) + (f"；主要瓶颈为 {ordered[0][0]}。" if ordered else "")
            )
    lines.extend((
        "",
        "## 6. Queue 与 Worker 时间模型",
        "",
        "Queue A 使用前序任务的 worker occupied（DEQUEUE→RETURN）；当前 Task 的 Issue 目标只减 pre-issue，Return 再减 syscall service。Queue B 始终作为对照保留，无论结果优劣都不删除。",
        "",
        "| Model | N | Mature | MAE ms | p95 ms | Fallback |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for name, value in result.get("queue_models", {}).items():
        all_value = value["all_available"]
        mature = value["mature_exact"]
        lines.append(
            f"| Queue {name} | {all_value.get('count', 0)} | {mature.get('count', 0)} | "
            f"{ms(all_value.get('mae_ns'))} | {ms(all_value.get('p95_absolute_error_ns'))} | "
            f"{all_value.get('fallback_count', 0)} |"
        )
    for name, value in result.get("duration_models", {}).items():
        all_value = value["all_available"]
        mature = value["mature_exact"]
        lines.append(
            f"| {name} | {all_value.get('count', 0)} | {mature.get('count', 0)} | "
            f"{ms(all_value.get('mae_ns'))} | {ms(all_value.get('p95_absolute_error_ns'))} | "
            f"{all_value.get('fallback_count', 0)} |"
        )
    lines.extend((
        "",
        "## 7. Shadow off/on 工程等价性",
        "",
        f"等价性文件={len(equivalence)}，全部通过={all_equivalent}，same binary={summary['same_binary']}，zero-drop={summary['all_zero_drop']}，delegated cgroup={summary['delegated_cgroup']}。",
        "",
        "各 workers=2/4 均比较 Task lifecycle 字段、Hint multiset、输出 Hash 和 sink drop；Summary 只用于工程观测，不宣称性能收益。",
        "",
        "## 8. Active Control 人工门槛",
        "",
        f"仅使用 mature、非 fallback 样本后，满足 workers=2/4 下 Decode predicted-late precision≥90% 且非零覆盖的候选数={len(qualified)}。这只是候选筛查；threshold calibration 和跨 worker 一致性仍须人工判断。",
        "",
        "## 9. 产物",
        "",
        f"- 完整机器 JSON：`{args.final_json.resolve()}`",
        f"- 全模型与 Oracle JSON：`{args.input.resolve()}`",
        f"- 人类报告：`{args.output.resolve()}`",
        f"- Detail Evidence：{', '.join(f'`{path.resolve()}`' for path in args.detail_run_dir)}",
        "",
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.final_json.parent.mkdir(parents=True, exist_ok=True)
    args.final_json.write_text(
        json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
