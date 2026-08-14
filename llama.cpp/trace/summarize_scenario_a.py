#!/usr/bin/env python3
"""Create a defence-ready, factual summary for the Scenario-A triplet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GROUPS = ("plain", "survival_static", "survival_reclaim")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def number(value: Any) -> int | float | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def collect_memory_summary(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "EXPERT_MEMORY_OBJECT_SUMMARY":
            return record
    return {}


def collect_group(base: Path, prefix: str, group: str) -> dict[str, Any]:
    directory = base / f"{prefix}_{group}"
    process = load_json(directory / "process_metrics.json") or {}
    after = load_json(directory / "cgroup_after_inference.json") or {}
    events = after.get("memory_events", {}) if isinstance(after.get("memory_events"), dict) else {}
    memory = after.get("memory", {}) if isinstance(after.get("memory"), dict) else {}
    output_hash = None
    try:
        output_hash = (directory / "output.sha256").read_text(encoding="utf-8").strip() or None
    except OSError:
        pass
    launch_status = (
        number((directory / "launch_status.txt").read_text().strip())
        if (directory / "launch_status.txt").is_file()
        else None
    )
    process_exit = number(process.get("exit_code"))
    effective_exit = process_exit if process_exit is not None else launch_status
    has_events = bool(after)
    oom_kill = number(events.get("oom_kill")) if has_events else None
    output_completed = bool(output_hash) and process_exit == 0 and oom_kill in (None, 0)
    if oom_kill and oom_kill > 0:
        outcome = "cgroup OOM kill"
    elif output_completed:
        outcome = "completed"
    elif effective_exit is not None:
        outcome = f"exit {effective_exit}"
    else:
        outcome = "incomplete"
    return {
        "group": group,
        "directory": str(directory),
        "launch_status": launch_status,
        "process_exit_code": process_exit,
        "effective_exit_code": effective_exit,
        "completed": output_completed,
        "outcome": outcome,
        "wall_time_s": process.get("wall_time_s"),
        "max_rss_kb": process.get("max_rss_kb"),
        "major_faults": process.get("major_faults"),
        "memory_peak_bytes": number(memory.get("memory.peak")),
        "memory_max_bytes": number(memory.get("memory.max")),
        "oom_events": number(events.get("oom")) if has_events else None,
        "oom_kill_events": oom_kill,
        "output_sha256": output_hash,
        "reclaim": collect_memory_summary(directory / "memory_trace.jsonl"),
    }


def fmt_bytes(value: Any) -> str:
    value = number(value)
    if value is None:
        return "—"
    return f"{value / (1024 ** 2):.1f} MiB"


def fmt(value: Any) -> str:
    return "—" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    groups = [collect_group(args.base_dir, args.run_prefix, group) for group in GROUPS]
    successful = [item for item in groups if item["completed"]]
    hashes = {item["output_sha256"] for item in successful}
    plain, static, reclaim = groups
    presence_proof = (
        plain["oom_kill_events"] is not None
        and plain["oom_kill_events"] > 0
        and reclaim["completed"]
    )
    conclusion = (
        "支持：Plain 触发 cgroup OOM kill，而 survival_reclaim 完成。"
        if presence_proof
        else "未建立 Plain 失败、survival_reclaim 完成的完成性证明；答辩不得声称该结论。"
    )
    summary = {
        "schema_version": 1,
        "scenario": "A_memory_constrained",
        "groups": groups,
        "successful_output_hashes_match": len(hashes) <= 1,
        "successful_output_hash_count": len(hashes),
        "completion_presence_proof": presence_proof,
        "conclusion": conclusion,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scenario_a_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for item in groups:
        reclaim_info = item["reclaim"]
        rows.append(
            "| {group} | {exit_code} | {wall} | {peak} / {max_mem} | {faults} | "
            "{oom}/{oom_kill} | {dontneed} | {hash_value} |".format(
                group=item["group"],
                exit_code=item["outcome"],
                wall=fmt(item["wall_time_s"]),
                peak=fmt_bytes(item["memory_peak_bytes"]),
                max_mem=fmt_bytes(item["memory_max_bytes"]),
                faults=fmt(item["major_faults"]),
                oom=fmt(item["oom_events"]),
                oom_kill=fmt(item["oom_kill_events"]),
                dontneed=fmt(reclaim_info.get("madv_dontneed_issued")),
                hash_value=(item["output_sha256"] or "—")[:12],
            )
        )
    report = [
        "# 场景 A：受限内存完成性实验",
        "",
        "所有组均在独立的 cgroup v2 scope 中运行；`memory.max`、`memory.swap.max` 和终态事件由 scope 内进程读取。",
        "",
        "| 组别 | 结果 | wall s | cgroup peak / max | major faults | oom / oom_kill | MADV_DONTNEED issued | 输出 hash |",
        "| --- | ---: | ---: | --- | ---: | --- | ---: | --- |",
        *rows,
        "",
        f"结论：{conclusion}",
        "",
        "只有完整输出且未触发 OOM kill 的组参与输出一致性比较；成功组 hash 一致："
        + ("是。" if len(hashes) <= 1 else "否，结果不可用于正确性声明。"),
        "",
        "组别定义：`plain` 为未编译 Trace 的 llama-cli，使用原始 F16 KV、ctx=2048、ubatch=512；"
        "`survival_static` 使用 survival 启动模板而不启用回收；`survival_reclaim` 在同一 survival 模板上启用"
        "文件映射检查、grace step、in-flight 检查、压力门控和每 Decode step 64 MiB 的 MADV_DONTNEED。",
    ]
    (args.output_dir / "SCENARIO_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
