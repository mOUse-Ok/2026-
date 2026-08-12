#!/usr/bin/env python3
"""Aggregate Experiment 4B object-level residency observations.

The C++ observer emits one RESIDENCY_DEMAND record immediately before a
semantic tensor demand. The two metrics stay separate:

* demand-weighted missing bytes: repeated demands remain repeated;
* unique missing pages: read from the observer's exact mincore page summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CLASSES = [
    "Routed Expert",
    "Shared Expert",
    "Attention",
    "SSM",
    "Router/Gate",
    "Embedding",
    "Output",
    "Norm",
    "Other",
]
PHASES = ["PREFILL", "DECODE", "ALL"]


def blank_stats() -> dict[str, int]:
    return {
        "demand_events": 0,
        "demanded_bytes": 0,
        "resident_before_use_bytes": 0,
        "nonresident_before_use_bytes": 0,
        "total_pages": 0,
        "resident_pages": 0,
        "nonresident_pages": 0,
        "unique_demand_pages": 0,
        "unique_missing_pages": 0,
        "exact_events": 0,
        "sampled_events": 0,
        "mapping_missing_events": 0,
    }


def blank_table() -> dict[str, dict[str, dict[str, int]]]:
    return {
        phase: {klass: blank_stats() for klass in CLASSES}
        for phase in PHASES
    }


def safe_int(record: dict[str, Any], name: str) -> int:
    try:
        return int(record.get(name, 0))
    except (TypeError, ValueError):
        return 0


def add_event(stats: dict[str, int], record: dict[str, Any]) -> None:
    stats["demand_events"] += 1
    stats["demanded_bytes"] += safe_int(record, "tensor_bytes")
    for name in (
        "resident_before_use_bytes",
        "nonresident_before_use_bytes",
        "total_pages",
        "resident_pages",
        "nonresident_pages",
    ):
        source = "resident_bytes" if name == "resident_before_use_bytes" else (
            "nonresident_bytes" if name == "nonresident_before_use_bytes" else name
        )
        stats[name] += safe_int(record, source)
    if record.get("resident_exact") is True:
        stats["exact_events"] += 1
    else:
        stats["sampled_events"] += 1
    if not record.get("file_backed", False):
        stats["mapping_missing_events"] += 1


def finalize_stats(stats: dict[str, int], total_missing: int) -> dict[str, Any]:
    demanded = stats["demanded_bytes"]
    missing = stats["nonresident_before_use_bytes"]
    pages = stats["total_pages"]
    result: dict[str, Any] = dict(stats)
    result["resident_ratio"] = (
        stats["resident_before_use_bytes"] / demanded if demanded else 0.0
    )
    result["missing_ratio"] = missing / demanded if demanded else 0.0
    result["share_of_all_missing_bytes"] = (
        missing / total_missing if total_missing else 0.0
    )
    result["unique_missing_ratio"] = (
        stats["unique_missing_pages"] / stats["unique_demand_pages"]
        if stats["unique_demand_pages"]
        else 0.0
    )
    result["page_missing_ratio"] = (
        stats["nonresident_pages"] / pages if pages else 0.0
    )
    return result


def read_records(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield line_number, record


def read_summary(trace_dir: Path) -> dict[str, Any] | None:
    summary: dict[str, Any] | None = None
    for _, record in read_records(trace_dir / "memory_trace.jsonl"):
        if record.get("event") == "RESIDENCY_ATTRIBUTION_SUMMARY":
            summary = record
    return summary


def merge_unique_stats(
    table: dict[str, dict[str, dict[str, int]]],
    summary: dict[str, Any] | None,
) -> bool:
    if not summary:
        return False
    by_phase = summary.get("by_phase", {})
    for phase in PHASES:
        source = by_phase.get(phase, {})
        for klass in CLASSES:
            values = source.get(klass)
            if not isinstance(values, dict):
                continue
            for name in ("unique_demand_pages", "unique_missing_pages"):
                table[phase][klass][name] = safe_int(values, name)
    return bool(summary.get("unique_pages_complete", False))


def analyze(trace_dir: Path) -> dict[str, Any]:
    table = blank_table()
    event_count = 0
    successful_count = 0
    error_count = 0
    exact_count = 0
    sampled_count = 0
    file_backed_count = 0
    tensor_subclasses: dict[str, dict[str, dict[str, int]]] = {
        phase: defaultdict(lambda: defaultdict(int)) for phase in PHASES
    }
    phases_seen: set[str] = set()
    steps_seen: dict[str, set[int]] = defaultdict(set)

    for _, record in read_records(trace_dir / "tensor_trace.jsonl"):
        if record.get("event") != "RESIDENCY_DEMAND":
            continue
        event_count += 1
        phase = str(record.get("phase", "UNKNOWN"))
        phase_bucket = phase if phase in ("PREFILL", "DECODE") else None
        klass = str(record.get("object_class", "Other"))
        if klass not in CLASSES:
            klass = "Other"
        if phase_bucket is not None:
            phases_seen.add(phase_bucket)
            steps_seen[phase_bucket].add(safe_int(record, "step"))
        if not record.get("residency_available", False):
            error_count += 1
            continue
        successful_count += 1
        exact_count += int(record.get("resident_exact") is True)
        sampled_count += int(record.get("resident_exact") is not True)
        file_backed_count += int(record.get("file_backed", False))
        if phase_bucket is None:
            continue
        add_event(table[phase_bucket][klass], record)
        add_event(table["ALL"][klass], record)
        subclass = str(record.get("tensor_subclass", ""))
        if klass == "Routed Expert" and subclass:
            for bucket in (phase_bucket, "ALL"):
                tensor_subclasses[bucket][subclass]["demand_events"] += 1
                tensor_subclasses[bucket][subclass]["demanded_bytes"] += safe_int(
                    record, "tensor_bytes"
                )
                tensor_subclasses[bucket][subclass][
                    "nonresident_before_use_bytes"
                ] += safe_int(record, "nonresident_bytes")

    summary = read_summary(trace_dir)
    unique_complete = merge_unique_stats(table, summary)
    for phase in PHASES:
        total_missing = sum(
            table[phase][klass]["nonresident_before_use_bytes"] for klass in CLASSES
        )
        for klass in CLASSES:
            table[phase][klass] = finalize_stats(
                table[phase][klass], total_missing
            )

    subclass_output: dict[str, dict[str, dict[str, Any]]] = {}
    for phase in PHASES:
        subclass_output[phase] = {}
        total_missing = sum(
            values["nonresident_before_use_bytes"]
            for values in tensor_subclasses[phase].values()
        )
        for name, values in tensor_subclasses[phase].items():
            demanded = values["demanded_bytes"]
            missing = values["nonresident_before_use_bytes"]
            subclass_output[phase][name] = {
                **values,
                "missing_ratio": missing / demanded if demanded else 0.0,
                "share_of_expert_missing_bytes": (
                    missing / total_missing if total_missing else 0.0
                ),
            }

    return {
        "schema_version": 1,
        "trace_dir": str(trace_dir.resolve()),
        "observer": {
            "event_count": event_count,
            "successful_count": successful_count,
            "residency_error_events": error_count,
            "exact_event_count": exact_count,
            "sampled_event_count": sampled_count,
            "file_backed_event_count": file_backed_count,
            "unique_pages_complete": unique_complete,
            "summary_present": summary is not None,
        },
        "phases_seen": sorted(phases_seen),
        "steps_seen": {phase: sorted(values) for phase, values in steps_seen.items()},
        "stats": table,
        "expert_tensor_subclasses": subclass_output,
    }


def write_csv(result: dict[str, Any], path: Path) -> None:
    fields = [
        "phase",
        "object_class",
        "demand_events",
        "demanded_bytes",
        "resident_before_use_bytes",
        "nonresident_before_use_bytes",
        "resident_ratio",
        "missing_ratio",
        "share_of_all_missing_bytes",
        "unique_demand_pages",
        "unique_missing_pages",
        "unique_missing_ratio",
        "exact_events",
        "sampled_events",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for phase in PHASES:
            for klass in CLASSES:
                values = result["stats"][phase][klass]
                writer.writerow(
                    {
                        "phase": phase,
                        "object_class": klass,
                        **{field: values[field] for field in fields[2:]},
                    }
                )


def fmt_bytes(value: int | float) -> str:
    value = float(value)
    units = ("B", "KiB", "MiB", "GiB")
    unit = 0
    while abs(value) >= 1024 and unit < len(units) - 1:
        value /= 1024
        unit += 1
    return f"{value:.2f} {units[unit]}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    decode = result["stats"]["DECODE"]
    lines = [
        "# Experiment 4B — Object-Level Residency Attribution",
        "",
        f"- Trace: '{result['trace_dir']}'",
        f"- Residency demand events: '{result['observer']['successful_count']}' successful / "
        f"'{result['observer']['event_count']}' emitted",
        f"- Exact mincore events: '{result['observer']['exact_event_count']}'; "
        f"sampled: '{result['observer']['sampled_event_count']}'",
        f"- Unique-page summary complete: '{result['observer']['unique_pages_complete']}'",
        "",
        "## DECODE demand-weighted missing bytes",
        "",
        "| Object Class | Demand Bytes | Nonresident Before Use | Missing Ratio | Share of All Missing Bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    order = sorted(
        CLASSES,
        key=lambda klass: (
            -decode[klass]["nonresident_before_use_bytes"],
            CLASSES.index(klass),
        ),
    )
    for klass in order:
        values = decode[klass]
        lines.append(
            f"| {klass} | {fmt_bytes(values['demanded_bytes'])} | "
            f"{fmt_bytes(values['nonresident_before_use_bytes'])} | "
            f"{values['missing_ratio'] * 100:.2f}% | "
            f"{values['share_of_all_missing_bytes'] * 100:.2f}% |"
        )
    lines += [
        "",
        "The missing-byte share is the primary ranking metric. It is not a "
        "Major Fault causal attribution.",
        "",
        "## Expert tensor split",
        "",
        "| Phase | Expert Tensor | Demand Bytes | Missing Bytes | Missing Ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for phase in ("PREFILL", "DECODE", "ALL"):
        for name, values in result["expert_tensor_subclasses"][phase].items():
            lines.append(
                f"| {phase} | {name} | {fmt_bytes(values['demanded_bytes'])} | "
                f"{fmt_bytes(values['nonresident_before_use_bytes'])} | "
                f"{values['missing_ratio'] * 100:.2f}% |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(
    result: dict[str, Any], output_dir: Path, condition_label: str
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    decode = result["stats"]["DECODE"]
    order = sorted(
        CLASSES,
        key=lambda klass: decode[klass]["share_of_all_missing_bytes"],
    )
    order = [
        klass for klass in order
        if decode[klass]["nonresident_before_use_bytes"] > 0
    ]
    if order:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        values = [
            decode[klass]["share_of_all_missing_bytes"] * 100 for klass in order
        ]
        colors = [
            "#d97706" if klass == "Routed Expert" else "#64748b"
            for klass in order
        ]
        bars = ax.barh(order, values, color=colors)
        ax.set_xlabel("Share of all missing bytes (%)")
        ax.set_title(f"{condition_label} Decode Missing Bytes Breakdown")
        ax.grid(axis="x", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                value + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=9,
            )
        fig.tight_layout()
        filename = (
            "figure_1_7gb_decode_missing_breakdown.png"
            if condition_label == "MemoryMax=7G"
            else "figure_decode_missing_breakdown.png"
        )
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition-label", default="")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(args.trace_dir)
    (args.output_dir / "residency_attribution.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(result, args.output_dir / "residency_attribution.csv")
    write_markdown(result, args.output_dir / "residency_attribution.md")
    condition_label = args.condition_label
    if not condition_label:
        condition_label = "Unlimited"
        try:
            manifest = json.loads(
                (args.trace_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("experiment", {}).get("requested_memory_max"):
                condition_label = "MemoryMax=7G"
        except (OSError, json.JSONDecodeError):
            pass
    make_figures(result, args.output_dir, condition_label)
    print(json.dumps(result["observer"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
