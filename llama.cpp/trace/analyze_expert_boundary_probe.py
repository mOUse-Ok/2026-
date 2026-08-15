#!/usr/bin/env python3
"""Summarize EXPERT_BOUNDARY_PROBE JSONL events without runtime dependencies."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_input(spec: str) -> tuple[str, Path]:
    """Accept either a run directory/path or a convenient LABEL=PATH form."""
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        if label and raw_path:
            return label, Path(raw_path)
    path = Path(spec)
    return path.name, path


def trace_path(path: Path) -> Path:
    return path / "memory_trace.jsonl" if path.is_dir() else path


def load_events(path: Path, label: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with trace_path(path).open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "EXPERT_BOUNDARY_PROBE":
                continue
            record["_config"] = label
            events.append(record)
    return events


def is_valid(event: dict[str, Any]) -> bool:
    # The one-shot format used status=complete before the explicit valid field
    # was added; accepting it makes this tool usable on earlier runs too.
    return bool(event.get("valid", event.get("status") == "complete"))


def page_size(event: dict[str, Any]) -> int:
    selected_before = event.get("selected_before", {})
    return int(selected_before.get("page_size", event.get("page_size", 4096)))


def event_value(event: dict[str, Any], short: str, legacy: str) -> int:
    return int(event.get(short, event.get(legacy, 0)))


def summarize(label: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [event for event in events if is_valid(event)]
    a_new = [event_value(event, "A_new_pages", "selected_new_pages") for event in valid]
    b_new = [event_value(event, "B_new_pages", "neighbor_new_pages") for event in valid]
    page_sizes = [page_size(event) for event in valid]
    total_a = sum(a_new)
    total_b = sum(b_new)
    positive = sum(value > 0 for value in b_new)
    return {
        "config": label,
        "events": len(events),
        "invalid_pairs": sum(not is_valid(event) for event in events),
        "valid_pairs": len(valid),
        "positive_pairs": positive,
        "positive_rate": positive / max(len(valid), 1),
        "sum_A_new_pages": total_a,
        "sum_B_new_pages": total_b,
        "sum_A_new_bytes": sum(value * size for value, size in zip(a_new, page_sizes)),
        "sum_B_new_bytes": sum(value * size for value, size in zip(b_new, page_sizes)),
        "weighted_overfetch_ratio": total_b / max(total_a + total_b, 1),
        "overfetch_per_useful_page": total_b / max(total_a, 1),
        "median_B_new_pages": statistics.median(b_new) if b_new else 0,
        "max_B_new_pages": max(b_new, default=0),
        "_valid_events": valid,
    }


def matched_pairs(summaries: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    by_key: dict[tuple[Any, ...], dict[str, int]] = defaultdict(dict)
    for summary in summaries:
        for event in summary["_valid_events"]:
            key = (
                event.get("step"), event.get("layer"), event.get("tensor"),
                event.get("selected_expert"), event.get("neighbor_expert"),
            )
            by_key[key][summary["config"]] = event_value(
                event, "B_new_pages", "neighbor_new_pages")

    baseline = next((summary["config"] for summary in summaries
                     if summary["config"] == "baseline"), summaries[0]["config"] if summaries else "")
    rows: list[dict[str, Any]] = []
    for key, values in sorted(by_key.items()):
        if len(values) < 2 or baseline not in values:
            continue
        for config, b_new in sorted(values.items()):
            if config == baseline:
                continue
            rows.append({
                "step": key[0], "layer": key[1], "tensor": key[2],
                "selected_expert": key[3], "neighbor_expert": key[4],
                "config": config, "baseline_B_new_pages": values[baseline],
                "alternative_B_new_pages": b_new,
                "delta_pages": b_new - values[baseline],
            })
    return len({
        (row["step"], row["layer"], row["tensor"],
         row["selected_expert"], row["neighbor_expert"])
        for row in rows
    }), rows


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    widths = [len(header) for header in headers]
    rendered = [[str(value) for value in row] for row in rows]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def serializable(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "_valid_events"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize EXPERT_BOUNDARY_PROBE events from one or more trace directories.")
    parser.add_argument("inputs", nargs="+", help="RUN_DIR, JSONL, or LABEL=RUN_DIR")
    parser.add_argument("--json-output", type=Path, help="Write the aggregate as JSON as well")
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for spec in args.inputs:
        label, path = parse_input(spec)
        source = trace_path(path)
        if not source.is_file():
            parser.error(f"trace not found: {source}")
        summaries.append(summarize(label, load_events(path, label)))

    print_table(
        ["config", "valid", "positive", "positive_rate", "A_new_pages", "B_new_pages",
         "overfetch_ratio", "overfetch_per_A", "median_B", "max_B"],
        [[summary["config"], summary["valid_pairs"], summary["positive_pairs"],
          f"{summary['positive_rate']:.3f}", summary["sum_A_new_pages"],
          summary["sum_B_new_pages"], f"{summary['weighted_overfetch_ratio']:.3f}",
          f"{summary['overfetch_per_useful_page']:.3f}", summary["median_B_new_pages"],
          summary["max_B_new_pages"]] for summary in summaries],
    )

    matched_pair_keys, pairs = matched_pairs(summaries)
    print(f"\nmatched_pair_keys={matched_pair_keys} matched_comparisons={len(pairs)}")
    if pairs:
        print_table(
            ["step", "layer", "tensor", "A", "B", "config", "baseline_B", "alternative_B", "delta"],
            [[pair["step"], pair["layer"], pair["tensor"], pair["selected_expert"],
              pair["neighbor_expert"], pair["config"], pair["baseline_B_new_pages"],
              pair["alternative_B_new_pages"], pair["delta_pages"]] for pair in pairs],
        )

    if args.json_output:
        args.json_output.write_text(json.dumps({
            "runs": [serializable(summary) for summary in summaries],
            "matched_pair_keys": matched_pair_keys,
            "matched_pairs": pairs,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
