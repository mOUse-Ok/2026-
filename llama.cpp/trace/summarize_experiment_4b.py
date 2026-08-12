#!/usr/bin/env python3
"""Build the two-run Experiment 4B report and core figures."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from analyze_residency_attribution import CLASSES, analyze, fmt_bytes


def load_or_analyze(run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "analysis" / "residency_attribution.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    return analyze(run_dir)


def decode_tps(run_dir: Path) -> float | None:
    try:
        step_records = []
        token_records = []
        with (run_dir / "memory_trace.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("phase") != "DECODE":
                    continue
                if record.get("event") == "STEP_END":
                    step_records.append(record)
                elif record.get("event") == "TOKEN_END":
                    token_records.append(record)
        # STEP_END already represents the whole decode step. Prefer it when
        # available so TOKEN_END is not double-counted.
        records = step_records or token_records
        total_ns = sum(int(record.get("latency_ns", 0)) for record in records)
        tokens = sum(int(record.get("n_tokens", 1)) for record in records)
        return tokens / (total_ns / 1e9) if total_ns else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def write_comparison_csv(
    unlimited: dict[str, Any], limited: dict[str, Any], output: Path
) -> None:
    fields = [
        "object_class",
        "unlimited_demanded_bytes",
        "unlimited_nonresident_bytes",
        "unlimited_missing_ratio",
        "limited_demanded_bytes",
        "limited_nonresident_bytes",
        "limited_missing_ratio",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for klass in CLASSES:
            u = unlimited["stats"]["DECODE"][klass]
            l = limited["stats"]["DECODE"][klass]
            writer.writerow(
                {
                    "object_class": klass,
                    "unlimited_demanded_bytes": u["demanded_bytes"],
                    "unlimited_nonresident_bytes": u[
                        "nonresident_before_use_bytes"
                    ],
                    "unlimited_missing_ratio": u["missing_ratio"],
                    "limited_demanded_bytes": l["demanded_bytes"],
                    "limited_nonresident_bytes": l[
                        "nonresident_before_use_bytes"
                    ],
                    "limited_missing_ratio": l["missing_ratio"],
                }
            )


def make_figure_2(
    unlimited: dict[str, Any], limited: dict[str, Any], output: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    core = ["Routed Expert", "Attention", "SSM"]
    u_values = [unlimited["stats"]["DECODE"][name]["missing_ratio"] * 100 for name in core]
    l_values = [limited["stats"]["DECODE"][name]["missing_ratio"] * 100 for name in core]
    x = np.arange(len(core))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(x - width / 2, u_values, width, label="Unlimited", color="#64748b")
    ax.bar(x + width / 2, l_values, width, label="MemoryMax=7G", color="#d97706")
    ax.set_ylabel("Missing ratio (%)")
    ax.set_title("Unlimited vs 7GB Missing Ratio")
    ax.set_xticks(x, core)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def table_lines(result: dict[str, Any]) -> list[str]:
    values = result["stats"]["DECODE"]
    order = sorted(
        CLASSES,
        key=lambda klass: (
            -values[klass]["nonresident_before_use_bytes"],
            CLASSES.index(klass),
        ),
    )
    lines = [
        "| Object Class | Demand Bytes | Nonresident Before Use | Missing Ratio | Share of All Missing Bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for klass in order:
        item = values[klass]
        lines.append(
            f"| {klass} | {fmt_bytes(item['demanded_bytes'])} | "
            f"{fmt_bytes(item['nonresident_before_use_bytes'])} | "
            f"{item['missing_ratio'] * 100:.2f}% | "
            f"{item['share_of_all_missing_bytes'] * 100:.2f}% |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlimited-dir", required=True, type=Path)
    parser.add_argument("--memory-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    unlimited = load_or_analyze(args.unlimited_dir)
    limited = load_or_analyze(args.memory_dir)
    (args.output_dir / "experiment_4b_comparison.json").write_text(
        json.dumps(
            {"unlimited": unlimited, "memorymax_7g": limited},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_comparison_csv(
        unlimited, limited, args.output_dir / "experiment_4b_comparison.csv"
    )
    source_figure = (
        args.memory_dir / "analysis" / "figure_1_7gb_decode_missing_breakdown.png"
    )
    if source_figure.is_file():
        shutil.copy2(
            source_figure,
            args.output_dir / "figure_1_7gb_decode_missing_breakdown.png",
        )
    make_figure_2(
        unlimited,
        limited,
        args.output_dir / "figure_2_unlimited_vs_7gb_missing_ratio.png",
    )

    routed_share = limited["stats"]["DECODE"]["Routed Expert"][
        "share_of_all_missing_bytes"
    ]
    routed_missing = limited["stats"]["DECODE"]["Routed Expert"][
        "nonresident_before_use_bytes"
    ]
    lines = [
        "# Experiment 4B — Object-Level Residency Attribution",
        "",
        "This report is generated from fresh local runs. Existing repository "
        "traces are not used as results.",
        "",
        "## 7GB DECODE",
        "",
        *table_lines(limited),
        "",
        f"Routed Expert share of all Decode missing bytes: "
        f"**{routed_share * 100:.2f}%** "
        f"({fmt_bytes(routed_missing)}).",
        "",
        "## Unlimited sanity check",
        "",
        *table_lines(unlimited),
        "",
        "## Interpretation boundary",
        "",
        "The primary ranking is demand-weighted nonresident bytes before "
        "logical use. This is a residency attribution result, not a causal "
        "attribution of Major Faults or Decode slowdown.",
        "",
        "## Decode throughput observer check",
        "",
        f"- Unlimited decode TPS: {decode_tps(args.unlimited_dir)}",
        f"- MemoryMax=7G decode TPS: {decode_tps(args.memory_dir)}",
        "",
        "See the two PNG figures in this directory for the defense visuals.",
    ]
    (args.output_dir / "experiment_4b_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "routed_expert_decode_missing_share": routed_share,
                "unlimited_decode_tps": decode_tps(args.unlimited_dir),
                "memorymax_7g_decode_tps": decode_tps(args.memory_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
