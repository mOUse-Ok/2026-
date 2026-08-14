#!/usr/bin/env python3
"""Summarize bounded Router-prefetch Scenario B without inferring wins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GROUPS = ("baseline", "performance")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def last_event(path: Path, name: str) -> dict[str, Any]:
    try:
        records = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in reversed(records):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == name:
            return record
    return {}


def stat_value(snapshot: dict[str, Any] | None, key: str) -> int | None:
    if not snapshot:
        return None
    stat = snapshot.get("memory_stat")
    if not isinstance(stat, dict):
        return None
    value = stat.get(key)
    return value if isinstance(value, int) else None


def collect(base: Path, prefix: str, group: str) -> dict[str, Any]:
    directory = base / f"{prefix}_{group}"
    startup = load_json(directory / "startup.json") or {}
    client = load_json(directory / "request_metrics.json") or {}
    before = load_json(directory / "cgroup_before_prepare.json")
    after = load_json(directory / "cgroup_after_requests.json")
    major_before = stat_value(before, "pgmajfault")
    major_after = stat_value(after, "pgmajfault")
    major_delta = None
    if major_before is not None and major_after is not None and major_after >= major_before:
        major_delta = major_after - major_before
    return {
        "group": group,
        "directory": str(directory),
        "startup_total_s": startup.get("startup_total_s"),
        "first_ttft_ms": client.get("first_request_ttft_ms"),
        "subsequent_stream_event_interarrival_p95_ms": client.get(
            "subsequent_stream_event_interarrival_p95_ms"
        ),
        "subsequent_server_decode_mean_p95_ms": client.get(
            "subsequent_server_decode_mean_p95_ms"
        ),
        "major_fault_delta": major_delta,
        "output_sha256": client.get("combined_output_sha256"),
        "prefetch_selection": last_event(
            directory / "memory_trace.jsonl", "EXPERT_PREFETCH_SELECTION_SUMMARY"
        ),
        "prefetch_tasks": last_event(directory / "memory_trace.jsonl", "EXPERT_TASK_SUMMARY"),
    }


def fmt(value: Any) -> str:
    return "—" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [collect(args.base_dir, args.run_prefix, group) for group in GROUPS]
    baseline, performance = rows
    comparable_output = bool(baseline["output_sha256"] and performance["output_sha256"])
    output_match = comparable_output and baseline["output_sha256"] == performance["output_sha256"]
    prefetch_tasks = performance["prefetch_tasks"]
    prefetch_selection = performance["prefetch_selection"]
    prefetch_issued = int(prefetch_tasks.get("issued", 0))
    result = {
        "schema_version": 1,
        "scenario": "B_bounded_memory_router_prefetch",
        "rows": rows,
        "outputs_match": output_match,
        "performance_prefetch_issued": prefetch_issued,
        "performance_prefetch_created": int(prefetch_tasks.get("created", 0)),
        "performance_selected_experts": int(prefetch_selection.get("selected_experts", 0)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scenario_b_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_rows = [
        "| {group} | {startup} | {ttft} | {stream_p95} | {server_p95} | {faults} | {selected} | {issued} |".format(
            group=row["group"],
            startup=fmt(row["startup_total_s"]),
            ttft=fmt(row["first_ttft_ms"]),
            stream_p95=fmt(row["subsequent_stream_event_interarrival_p95_ms"]),
            server_p95=fmt(row["subsequent_server_decode_mean_p95_ms"]),
            faults=fmt(row["major_fault_delta"]),
            selected=int(row["prefetch_selection"].get("selected_experts", 0)),
            issued=int(row["prefetch_tasks"].get("issued", 0)),
        )
        for row in rows
    ]
    if not output_match:
        conclusion = "输出一致性：未通过或没有完整数据；不得进行性能结论。"
    elif prefetch_issued == 0:
        conclusion = "输出一致性：通过；但 performance 未实际发出 hint，不得进行性能结论。"
    else:
        conclusion = "输出一致性：通过；performance 已实际发出 Router-driven hint。性能差异需结合重复运行报告。"
    report = [
        "# 场景 B：中等内存压力下的 Router 预取",
        "",
        "MemoryMax 默认 12 GiB：基线可完成，但模型不能完整驻留。两组均关闭启动期批量预加载。",
        "performance 仅对 Router 每层选中的 1 个 Expert 发出 hint，基础压力预算为 64 MiB。",
        "冷启动总时间从启动 scoped server 到 `/health` 可用，包含模型加载；它不是 Router hint 的收益指标。",
        "后续 p95 的 stream 指标是非空生成事件的相邻到达间隔；server 指标是每个后续请求返回的平均 decode/token 时间的 p95。",
        "",
        "| 组别 | 冷启动 s | 首请求 TTFT ms | 后续 stream 间隔 p95 ms | 后续 server decode 均值 p95 ms | Major Fault delta | Router selected | hint issued |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *markdown_rows,
        "",
        conclusion,
        "只有 performance 的 `hint issued > 0`，且输出一致时，才可比较两组的 decode 和 Major Fault。",
    ]
    (args.output_dir / "SCENARIO_B_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
