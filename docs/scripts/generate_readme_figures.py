#!/usr/bin/env python3
"""Generate evidence-faithful README/PPT figures from final-readme-evidence-v2.

This is presentation tooling only.  It does not run inference or modify project
runtime code.  Constants are transcribed from docs/final-readme-evidence-v2.md;
the Runtime Rescue timeline additionally reads the closed raw JSONL of rescue_on_4.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"
DATA = ROOT / "docs" / "data"
RESCUE_TRACE = ROOT / "llama.cpp" / "trace_output" / "final-readme-v2" / "rescue_on_4" / "memory_trace.jsonl"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

NAVY = "#183A5A"
TEAL = "#2E7D78"
SLATE = "#5E6C84"
AMBER = "#B7791F"
RUST = "#A35A3A"
INK = "#1F2933"
MUTED = "#667085"
GRID = "#D9E1E8"
PALE_BLUE = "#EDF4F8"
PALE_TEAL = "#E9F4F2"
PALE_AMBER = "#FBF3E4"
PALE_RUST = "#F8ECE8"


def configure() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    # The TTC's internal family name is different from its fontconfig alias;
    # registering it explicitly keeps Chinese labels intact in PNG and SVG.
    font_manager.fontManager.addfont(FONT_PATH)
    cjk_font = font_manager.FontProperties(fname=FONT_PATH).get_name()
    mpl.rcParams.update({
        "font.family": cjk_font,
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "axes.edgecolor": "#B8C4CE",
        "axes.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ASSETS / f"{name}.svg", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(ASSETS / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def panel_style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.75, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, *,
        edge: str = NAVY, face: str = PALE_BLUE, fontsize: float = 9.0,
        weight: str = "normal") -> FancyBboxPatch:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                           linewidth=1.25, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=INK,
            fontsize=fontsize, fontweight=weight, wrap=True)
    return patch


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *,
          color: str = MUTED, rad: float = 0.0, style: str = "-|>") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12,
                                 linewidth=1.1, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(15.2, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("系统真实架构：LLM 语义到 Linux VM 的跨层桥梁", loc="left", color=INK, fontsize=16, pad=16)

    # Canonical Runtime -> OS path.  Individual nodes deliberately preserve
    # the semantic hand-off boundaries documented by the V2 evidence report.
    qwen = (0.03, 0.78, 0.13, 0.10, "Qwen MoE\nLLM runtime")
    ggml = (0.19, 0.78, 0.13, 0.10, "llama.cpp /\nGGML")
    router = (0.35, 0.78, 0.13, 0.10, "MoE Router")
    score = (0.51, 0.78, 0.13, 0.10, "Expert ID /\nScore")
    registry = (0.67, 0.78, 0.16, 0.10, "Expert Tensor\nRegistry")
    slice_ = (0.80, 0.55, 0.15, 0.10, "Expert Slice")
    memory = (0.60, 0.55, 0.15, 0.10, "Memory Object\nDemand Lifecycle")
    hint = (0.41, 0.55, 0.14, 0.10, "Async Hint\nTask")
    madvise = (0.23, 0.55, 0.13, 0.10, "Linux\nmadvise")
    vm = (0.04, 0.55, 0.14, 0.10, "Page Cache /\nVM")
    main_nodes = [qwen, ggml, router, score, registry, slice_, memory, hint, madvise, vm]
    for x, y, w, h, label in main_nodes:
        box(ax, x, y, w, h, label, edge=NAVY, face=PALE_BLUE, fontsize=8.5)
    for current, following in zip(main_nodes[:5], main_nodes[1:6]):
        x, y, w, h, _ = current
        nx, ny, nw, nh, _ = following
        if ny == y:
            arrow(ax, (x + w, y + h / 2), (nx, ny + nh / 2))
        else:
            arrow(ax, (x + w / 2, y), (nx + nw / 2, ny + nh), rad=-0.04)
    for current, following in zip(main_nodes[5:], main_nodes[6:]):
        x, y, w, h, _ = current
        nx, ny, nw, nh, _ = following
        arrow(ax, (x, y + h / 2), (nx + nw, ny + nh / 2))

    # Stable mechanisms.
    box(ax, 0.78, 0.37, 0.17, 0.09, "稳定机制\nSemantic Working Set", edge=TEAL, face=PALE_TEAL, fontsize=8.4, weight="bold")
    arrow(ax, (0.81, 0.46), (0.69, 0.55), color=TEAL, rad=0.06)

    # Research controls.
    box(ax, 0.58, 0.37, 0.15, 0.09, "研究控制机制\nMADV_COLD", edge=AMBER, face=PALE_AMBER, fontsize=8.3, weight="bold")
    box(ax, 0.39, 0.37, 0.14, 0.09, "研究控制机制\nRuntime Rescue", edge=AMBER, face=PALE_AMBER, fontsize=8.1, weight="bold")
    arrow(ax, (0.60, 0.415), (0.30, 0.55), color=AMBER, rad=0.10)
    arrow(ax, (0.53, 0.415), (0.58, 0.415), color=AMBER)

    # Observation side path.
    box(ax, 0.03, 0.18, 0.28, 0.10, "旁路观测：Router / Expert / Task /\nFirst-use / OS metrics", edge=SLATE, face="#F4F6F8", fontsize=8.4)
    box(ax, 0.36, 0.18, 0.15, 0.10, "Trace Writer\n有界 sink", edge=SLATE, face="#F4F6F8", fontsize=8.5)
    box(ax, 0.56, 0.18, 0.13, 0.10, "JSONL", edge=SLATE, face="#F4F6F8", fontsize=8.8)
    box(ax, 0.74, 0.18, 0.21, 0.10, "Analysis\n完整性校验 · 图表 · 证据报告", edge=SLATE, face="#F4F6F8", fontsize=8.2)
    arrow(ax, (0.31, 0.23), (0.36, 0.23), color=SLATE)
    arrow(ax, (0.51, 0.23), (0.56, 0.23), color=SLATE)
    arrow(ax, (0.69, 0.23), (0.74, 0.23), color=SLATE)
    ax.text(0.04, 0.04, "注：Calibration shadow 为 observation-only，未画为默认主线。\n稳定机制与研究控制机制分开标注；madvise 是 Linux 建议，不等价于物理页回收。",
            color=MUTED, fontsize=8.7, va="bottom")
    save(fig, "system-architecture")


def evolution() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 12.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("项目认知演化：从语义可见性到运行时保护", loc="left", fontsize=16, color=INK, pad=14)
    stages = [
        (0.88, "Plain llama.cpp", NAVY, PALE_BLUE),
        (0.78, "建立 Trace\n获得 Router / Task / OS 语义观测", NAVY, PALE_BLUE),
        (0.68, "Router 暴露未来 Expert 需求\n→ Router-driven Expert Prefetch", NAVY, PALE_BLUE),
        (0.56, "严格 5×5 实验\nPrefetch ≠ 自动加速", RUST, PALE_RUST),
        (0.44, "Expert Slice → Memory Object\nDemand Lifecycle + Semantic Working Set", TEAL, PALE_TEAL),
        (0.32, "尝试 MADV_COLD 研究策略\nCold Object ≠ 应立即回收", RUST, PALE_RUST),
        (0.20, "Runtime Rescue\n静态策略 → 运行时反馈保护", AMBER, PALE_AMBER),
    ]
    for i, (y, label, edge, face) in enumerate(stages):
        box(ax, 0.19, y - 0.045, 0.54, 0.08, label, edge=edge, face=face, fontsize=10,
            weight="bold" if i in (3, 5) else "normal")
        if i < len(stages) - 1:
            arrow(ax, (0.46, y - 0.045), (0.46, stages[i + 1][0] + 0.035), color=MUTED)
    box(ax, 0.77, 0.52, 0.18, 0.12, "转折 1\n不把可预测性\n误当作加速", edge=RUST, face=PALE_RUST, fontsize=9, weight="bold")
    arrow(ax, (0.77, 0.58), (0.73, 0.56), color=RUST)
    box(ax, 0.77, 0.27, 0.18, 0.12, "转折 2\n不把冷对象\n误当作可立即回收", edge=RUST, face=PALE_RUST, fontsize=9, weight="bold")
    arrow(ax, (0.77, 0.33), (0.73, 0.32), color=RUST)
    ax.text(0.19, 0.06, "演化的是问题认识与证据边界，不是 Git commit timeline。", color=MUTED, fontsize=9)
    save(fig, "project-evolution")


def staircase() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 10.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Evidence Staircase：每一级都有不同类型的证据", loc="left", fontsize=16, color=INK, pad=14)
    stages = [
        ("Plain llama.cpp", "Trace / Semantic Instrumentation", "minimal trace overhead +3.66%", NAVY, PALE_BLUE, "定量成本"),
        ("Router-driven Expert Prefetch", "", "current HEAD N=5×5: no stable speedup", RUST, PALE_RUST, "负性能结果"),
        ("Memory Object Lifecycle", "", "demand / activation / completion / slot all closed", TEAL, PALE_TEAL, "机制闭环"),
        ("Semantic Working Set", "", "budgeted admit / evict / readmit / protection", TEAL, PALE_TEAL, "机制行为"),
        ("MADV_COLD", "", "current N=3: wall +6.47%, faults +9.38%", RUST, PALE_RUST, "负性能结果"),
        ("Runtime Rescue", "", "bad state → suspend COLD\n→ gate bypass → hint issuance restored", AMBER, PALE_AMBER, "状态机证据"),
    ]
    ys = [0.85, 0.70, 0.56, 0.42, 0.28, 0.14]
    for i, ((a, b, evidence, edge, face, kind), y) in enumerate(zip(stages, ys)):
        label = a if not b else f"{a}\n↓\n{b}"
        box(ax, 0.08, y - 0.055, 0.31, 0.105, label, edge=edge, face=face, fontsize=9.6, weight="bold")
        box(ax, 0.47, y - 0.047, 0.43, 0.09, evidence, edge=edge, face="white", fontsize=9.4)
        ax.text(0.93, y, kind, ha="left", va="center", fontsize=8.5, color=edge, fontweight="bold")
        arrow(ax, (0.39, y), (0.47, y), color=edge)
        if i < len(stages) - 1:
            arrow(ax, (0.235, y - 0.055), (0.235, ys[i + 1] + 0.05), color=MUTED)
    ax.legend(handles=[Patch(facecolor=PALE_BLUE, edgecolor=NAVY, label="量化成本 / 基础设施"),
                       Patch(facecolor=PALE_TEAL, edgecolor=TEAL, label="机制闭环 / 行为"),
                       Patch(facecolor=PALE_RUST, edgecolor=RUST, label="负性能结果"),
                       Patch(facecolor=PALE_AMBER, edgecolor=AMBER, label="研究状态机")],
              loc="lower left", bbox_to_anchor=(0.06, -0.04), ncol=2, frameon=False, fontsize=8.5)
    save(fig, "evidence-staircase")


def trace_overhead() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    labels = ["Plain\ntrace compiled out", "Trace\nminimal, controller=off"]
    means = [53.394, 55.348]
    errors = [0.687, 1.052]
    bars = ax.bar(labels, means, yerr=errors, capsize=5, width=0.56, color=[NAVY, TEAL], zorder=3)
    panel_style(ax)
    ax.set_ylabel("wall time (s), mean ± SD, N=5")
    ax.set_ylim(0, 64)
    ax.set_title("获取 LLM 语义观测的运行成本", loc="left", fontsize=14)
    for bar, val, err in zip(bars, means, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.7, f"{val:.3f} ± {err:.3f} s", ha="center", fontweight="bold")
    ax.text(0.5, 61, "+3.66%", ha="center", color=RUST, fontweight="bold")
    ax.text(0.02, -0.25, "同一输出 SHA-256 · trace dropped = 0 · major faults -0.23% · RSS +0.79%\n固定 CPU/Qwen/7040 MiB 冷缓存条件；不是通用常数。",
            transform=ax.transAxes, color=MUTED, fontsize=8.6, va="top")
    save(fig, "trace-overhead")


def prefetch_ablation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    labels = ["Controller off", "expert_prefetch"]
    colors = [SLATE, RUST]
    values = [[286_064.831, 321_072.612], [802_297.2, 798_301.8]]
    titles = ["Decode average", "Major faults"]
    units = ["µs", "count"]
    deltas = ["+12.24%", "-0.50%"]
    for ax, vals, title, unit, delta in zip(axes, values, titles, units, deltas):
        bars = ax.bar(labels, vals, color=colors, width=0.58, zorder=3)
        panel_style(ax)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel(unit)
        for bar, value in zip(bars, vals):
            label = f"{value:,.3f}" if unit == "µs" else f"{value:,.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.015, label, ha="center", fontsize=9)
        ax.text(0.5, 0.91, delta, ha="center", transform=ax.transAxes, color=RUST if "+" in delta else TEAL,
                fontweight="bold", fontsize=11)
    fig.suptitle("Current-head expert prefetch ablation", x=0.07, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.07, 0.02, "N=5×5；tested 7040 MiB cold-cache setup。No stable speedup observed.\n两图为独立坐标轴；不使用双 Y 轴，也不把 major-fault 微小变化解释为加速。", color=MUTED, fontsize=8.6)
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    save(fig, "expert-prefetch-ablation")


def working_set_budget() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3))
    budgets = ["256 MiB", "512 MiB", "1024 MiB"]
    evictions = [40_191, 33_986, 30_103]
    readmits = [17_778, 12_165, 9_467]
    faults = [492_277, 432_455, 431_597]
    x = list(range(3))
    width = 0.34
    ax = axes[0]
    ax.bar([v - width / 2 for v in x], evictions, width, label="evictions", color=NAVY, zorder=3)
    ax.bar([v + width / 2 for v in x], readmits, width, label="readmissions", color=TEAL, zorder=3)
    panel_style(ax)
    ax.set_xticks(x, budgets)
    ax.set_ylabel("count")
    ax.set_title("Working Set mechanism behavior", loc="left", fontsize=12)
    ax.legend(frameon=False)
    ax = axes[1]
    bars = ax.bar(budgets, faults, color=SLATE, zorder=3)
    panel_style(ax)
    ax.set_ylabel("major faults (count)")
    ax.set_title("Recorded historical workload metric", loc="left", fontsize=12)
    for b, v in zip(bars, faults):
        ax.text(b.get_x() + b.get_width() / 2, v + 9_000, f"{v:,}", ha="center", fontsize=9)
    fig.suptitle("Semantic Working Set budget scan (historical Qwen, 16 decode tokens)", x=0.07, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.07, 0.02, "budget 增大时 eviction/readmission 下降，说明 capacity-constrained semantic working set 在工作。\n256 MiB 曾出现 protected objects 导致暂时 unresolved；不是 strict physical memory cap，也不以此图主张性能提升。", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    save(fig, "working-set-budget")


def cold_ablation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.3))
    labels = ["Shadow-only", "Shadow + COLD"]
    colors = [TEAL, RUST]
    series = [("wall time (s)", [82.120, 87.430], "+6.47%"),
              ("major faults", [634_192.7, 693_656.7], "+9.38%")]
    for ax, (ylabel, vals, delta) in zip(axes, series):
        bars = ax.bar(labels, vals, color=colors, width=0.58, zorder=3)
        panel_style(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, loc="left", fontsize=12)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.014, f"{v:,.3f}" if ylabel.startswith("wall") else f"{v:,.1f}", ha="center", fontsize=9)
        ax.text(0.5, 0.91, delta, ha="center", transform=ax.transAxes, color=RUST, fontweight="bold", fontsize=11)
    fig.suptitle("Current-head COLD ablation (N=3 per group)", x=0.07, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.07, 0.02, "COLD-enabled runs were associated with higher wall time and major faults in this controlled A/B.\n每个 COLD run：46,256 issued · 0 failed · 20.97 GB advised · 26,192 post-COLD readmissions。\n成功 madvise 不等价于物理页已回收。", color=MUTED, fontsize=8.4)
    fig.tight_layout(rect=(0, 0.09, 1, 0.90))
    save(fig, "cold-ablation")


def rescue_steps() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with RESCUE_TRACE.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") == "EXPERT_RUNTIME_RESCUE_STEP":
                rows.append({
                    "decode_step": event["decode_step"],
                    "issued": event["issued"],
                    "major_fault_delta": event["major_fault_delta"],
                    "latency_ns": event["latency_ns"],
                    "state": event["state"],
                })
    return rows


def rescue_timeline() -> None:
    rows = rescue_steps()
    csv_path = DATA / "runtime-rescue-on4-step-data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    focus = [r for r in rows if int(r["decode_step"]) <= 12]
    x = [int(r["decode_step"]) for r in focus]
    issued = [int(r["issued"]) for r in focus]
    faults = [int(r["major_fault_delta"]) for r in focus]
    states = [str(r["state"]) for r in focus]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.0), sharex=True, gridspec_kw={"hspace": 0.16})
    for ax in axes:
        ax.axvspan(3.5, 8.5, color=PALE_AMBER, zorder=0, label="gate recovery")
        ax.axvspan(8.5, 12.5, color=PALE_BLUE, zorder=0, label="COLD suspended")
        ax.axvline(3, color=AMBER, linestyle="--", linewidth=1.5, zorder=2)
        panel_style(ax)
    axes[0].plot(x, issued, marker="o", color=NAVY, linewidth=2.2, zorder=3)
    axes[0].set_ylabel("prefetch issued / step")
    axes[0].set_ylim(-70, 1100)
    axes[0].set_title("Runtime Rescue timeline — current HEAD rescue_on_4", loc="left", fontsize=14)
    axes[0].annotate("trigger at decode step 3\nissued = 0", xy=(3, 0), xytext=(1.0, 810), color=RUST,
                     arrowprops={"arrowstyle": "->", "color": RUST}, fontsize=9, fontweight="bold")
    axes[0].annotate("gate bypass (5 steps)\nissued = 960 / step", xy=(4, 960), xytext=(6.7, 980), color=AMBER,
                     ha="center", fontsize=9, fontweight="bold")
    axes[0].annotate("COLD suspended\nfrom step 9", xy=(9, 960), xytext=(10.6, 780), color=NAVY,
                     ha="center", fontsize=8.7, fontweight="bold")
    axes[1].plot(x, faults, marker="o", color=RUST, linewidth=2.2, zorder=3)
    axes[1].set_ylabel("major fault delta / step")
    axes[1].set_xlabel("decode step")
    axes[1].set_xticks(x)
    axes[1].annotate("pre-trigger: 14,680 faults\nsteps 1–3", xy=(3, 6897), xytext=(4.8, 6000), color=RUST,
                     arrowprops={"arrowstyle": "->", "color": RUST}, fontsize=8.6)
    axes[0].legend(handles=[Patch(facecolor=PALE_AMBER, edgecolor=AMBER, label="gate recovery"),
                            Patch(facecolor=PALE_BLUE, edgecolor=NAVY, label="COLD suspended")],
                   loc="lower right", frameon=False, fontsize=8.5)
    fig.text(0.08, 0.01, "Mechanism evidence, not causal end-to-end speedup proof. 原始数据：rescue_on_4 的 EXPERT_RUNTIME_RESCUE_STEP。\n触发后 first 5 steps 共 issued=4,800；状态带来自 trace 的 control state，不虚构逐步 COLD 物理回收量。", color=MUTED, fontsize=8.4)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.91, hspace=0.16)
    save(fig, "runtime-rescue-timeline")


def correctness() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Correctness / Reliability Summary", loc="left", fontsize=16, color=INK, pad=12)
    cards = [
        ("38 / 38", "Qwen runs\nexit code = 0"),
        ("38 / 38", "output SHA-256\nidentical"),
        ("33 / 33", "trace runs\nzero dropped events"),
        ("0 / 0 / 0", "Memory Object\npending / active / violations"),
        ("8 / 8", "targeted CTest\npassed"),
    ]
    for i, (headline, caption) in enumerate(cards):
        x = 0.035 + i * 0.192
        box(ax, x, 0.25, 0.16, 0.43, "", edge=TEAL, face=PALE_TEAL)
        ax.text(x + 0.08, 0.54, headline, ha="center", va="center", fontsize=18, fontweight="bold", color=NAVY)
        ax.text(x + 0.08, 0.37, caption, ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(0.035, 0.10, "证据范围：final-readme-evidence-v2 的 38 个 fresh Qwen runs。\n这些检查说明输出一致、trace 完整和状态收尾；不构成性能或跨环境泛化主张。", color=MUTED, fontsize=9)
    save(fig, "correctness-summary")


def write_mermaid() -> None:
    (DATA / "system-architecture-mermaid.md").write_text("""# System Architecture Mermaid Backup

```mermaid
flowchart LR
    Q[Qwen MoE\nLLM Runtime Semantic] --> L[llama.cpp / GGML]
    L --> R[MoE Router]
    R --> I[Expert ID / Score]
    I --> T[Expert Tensor Registry]
    T --> S[Expert Slice]
    S --> M[Memory Object\nDemand Lifecycle]
    M --> A[Async Hint Task]
    A --> D[Linux madvise]
    D --> V[Page Cache / VM]

    M --- W[Semantic Working Set\nStable mechanism]
    C[MADV_COLD\nResearch control] --> D
    X[Runtime Rescue\nResearch control] -. suspend COLD .-> C
    X -. gate bypass .-> A

    R -. Router metrics .-> TW[Trace Writer]
    T -. Expert metrics .-> TW
    A -. Task / first-use metrics .-> TW
    V -. OS metrics .-> TW
    TW --> J[JSONL]
    J --> AN[Analysis / validation]
```

`MADV_COLD` is an advisory Linux call, not a claim of physical page reclamation. Calibration shadow is observation-only and intentionally omitted from the default main path.
""", encoding="utf-8")


def main() -> None:
    configure()
    architecture()
    evolution()
    staircase()
    trace_overhead()
    prefetch_ablation()
    working_set_budget()
    cold_ablation()
    rescue_timeline()
    correctness()
    write_mermaid()


if __name__ == "__main__":
    main()
