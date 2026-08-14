#!/usr/bin/env python3
"""Summarize the persistent-server Scenario-B comparison without inferring wins."""

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


def preload_summary(path: Path) -> dict[str, Any]:
    try:
        records = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in reversed(records):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "EXPERT_PRELOAD_SUMMARY":
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
        "preload": preload_summary(directory / "memory_trace.jsonl"),
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
    preload = performance["preload"]
    preload_issued = int(preload.get("populate_issued", 0)) + int(preload.get("willneed_issued", 0))
    result = {
        "schema_version": 1,
        "scenario": "B_memory_sufficient_persistent_service",
        "rows": rows,
        "outputs_match": output_match,
        "performance_preload_issued": preload_issued,
        "preload_decision": preload.get("decision"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scenario_b_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_rows = [
        "| {group} | {startup} | {ttft} | {stream_p95} | {server_p95} | {faults} | {decision} | {issued} |".format(
            group=row["group"],
            startup=fmt(row["startup_total_s"]),
            ttft=fmt(row["first_ttft_ms"]),
            stream_p95=fmt(row["subsequent_stream_event_interarrival_p95_ms"]),
            server_p95=fmt(row["subsequent_server_decode_mean_p95_ms"]),
            faults=fmt(row["major_fault_delta"]),
            decision=fmt(row["preload"].get("decision")),
            issued=int(row["preload"].get("populate_issued", 0)) + int(row["preload"].get("willneed_issued", 0)),
        )
        for row in rows
    ]
    conclusion = (
        "输出一致性：通过。" if output_match else "输出一致性：未通过或没有完整数据；不得进行性能结论。"
    )
    report = [
        "# 场景 B：内存充足下的持久服务",
        "",
        "冷启动总时间从启动 scoped server 到 `/health` 可用，包含模型加载、预算判定和预加载。",
        "后续 p95 的 stream 指标是非空生成事件的相邻到达间隔；server 指标是每个后续请求返回的平均 decode/token 时间的 p95。",
        "",
        "| 组别 | 冷启动 s | 首请求 TTFT ms | 后续 stream 间隔 p95 ms | 后续 server decode 均值 p95 ms | Major Fault delta | preload decision | advice issued |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        *markdown_rows,
        "",
        conclusion,
        "预加载 advice 仅在 `memory.current + model_size + KV reserve + buffer reserve < 0.8 × cgroup limit` 时执行。",
    ]
    (args.output_dir / "SCENARIO_B_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
