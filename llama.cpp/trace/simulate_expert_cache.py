#!/usr/bin/env python3
"""Trace-driven simulator for MoE expert-slice cache policies.

The simulator consumes existing LLM_MEM_TRACE JSONL files and estimates how
LRU/LFU/window-LFU/least-stale policies behave under fixed cache budgets. It is
intentionally lightweight: it does not rerun inference and uses tensor load
sizes plus observed expert ids to approximate per-expert slice sizes.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICIES = ["route", "lru", "lfu", "window_lfu", "least_stale"]
DEFAULT_BUDGETS_MB = [128, 256, 512, 768, 1024]
EXPERT_TENSOR_MARKERS = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_gate_up_exps.weight",
)


@dataclass
class TensorInfo:
    layer: int
    name: str
    size: int
    slice_size: int = 0


@dataclass
class Access:
    step: int
    layer: int
    expert: int
    tensor_name: str
    size: int
    score: float

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.expert}:{self.tensor_name}"


@dataclass
class CacheItem:
    key: str
    size: int
    first_step: int
    last_step: int
    hits: int = 1
    recent_hits: int = 1
    recent_epoch: int = 0
    avg_gap: int = 0
    score: float = 0.0


@dataclass
class SimResult:
    policy: str
    budget_mb: int
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    skips: int = 0
    prefetch_events: int = 0
    evictions: int = 0
    miss_bytes: int = 0
    evict_bytes: int = 0
    peak_cache_bytes: int = 0

    @property
    def hit_rate_pct(self) -> float:
        return self.hits / max(1, self.hits + self.misses) * 100.0

    @property
    def miss_gib(self) -> float:
        return self.miss_bytes / (1024**3)

    @property
    def evict_gib(self) -> float:
        return self.evict_bytes / (1024**3)

    @property
    def peak_cache_mb(self) -> float:
        return self.peak_cache_bytes / (1024**2)

    @property
    def estimated_hint_events(self) -> int:
        return self.prefetch_events + self.evictions


class PolicyCache:
    def __init__(self, policy: str, budget_mb: int, ttl_steps: int):
        self.policy = policy
        self.capacity = budget_mb * 1024 * 1024
        self.ttl_steps = max(1, ttl_steps)
        self.items: dict[str, CacheItem] = {}
        self.bytes = 0
        self.result = SimResult(policy=policy, budget_mb=budget_mb)
        self.route_seen: set[tuple[int, str]] = set()
        self.last_prune_step = -1

    def access(self, access: Access) -> None:
        self.result.accesses += 1
        if self.policy == "route":
            step_key = (access.step, access.key)
            if step_key in self.route_seen:
                self.result.hits += 1
                return
            self.route_seen.add(step_key)
            self.result.misses += 1
            self.result.prefetch_events += 1
            self.result.miss_bytes += access.size
            return

        if self.capacity <= 0 or access.size > self.capacity:
            self.result.skips += 1
            return

        item = self.items.get(access.key)
        if item is not None:
            self.result.hits += 1
            self._update_item(item, access.step, access.score)
            return

        self.result.misses += 1
        self.result.prefetch_events += 1
        self.result.miss_bytes += access.size
        self.items[access.key] = CacheItem(
            key=access.key,
            size=access.size,
            first_step=access.step,
            last_step=access.step,
            recent_epoch=access.step,
            score=access.score,
        )
        self.bytes += access.size
        self.result.peak_cache_bytes = max(self.result.peak_cache_bytes, self.bytes)
        if self.last_prune_step != access.step:
            self._evict_stale(access.step, protected_key=access.key)
            self.last_prune_step = access.step
        self._evict_to_budget(access.step, protected_key=access.key)

    def _update_item(self, item: CacheItem, step: int, score: float) -> None:
        if step > item.last_step:
            gap = step - item.last_step
            item.avg_gap = gap if item.avg_gap == 0 else (item.avg_gap * 3 + gap + 2) // 4
        if step > item.recent_epoch + self.ttl_steps:
            windows = min((step - item.recent_epoch) // self.ttl_steps, 8)
            item.recent_hits >>= windows
            item.recent_epoch = step
        item.last_step = step
        item.hits += 1
        item.recent_hits += 1
        item.score = score

    def _evict_stale(self, step: int, protected_key: str) -> None:
        stale = [
            key
            for key, item in self.items.items()
            if key != protected_key and step > item.last_step and step - item.last_step > self.ttl_steps
        ]
        for key in stale:
            self._evict_key(key)

    def _evict_to_budget(self, step: int, protected_key: str) -> None:
        while self.bytes > self.capacity and self.items:
            victim = self._victim_key(step, protected_key)
            if victim is None:
                break
            self._evict_key(victim)

    def _evict_key(self, key: str) -> None:
        item = self.items.pop(key)
        self.bytes -= min(self.bytes, item.size)
        self.result.evictions += 1
        self.result.evict_bytes += item.size

    def _victim_key(self, step: int, protected_key: str) -> str | None:
        candidates = [item for key, item in self.items.items() if key != protected_key or len(self.items) == 1]
        if not candidates:
            return None
        if self.policy == "lru":
            return min(candidates, key=lambda item: (item.last_step, item.hits)).key
        if self.policy == "lfu":
            return min(candidates, key=lambda item: (item.hits, item.last_step)).key
        if self.policy == "window_lfu":
            return min(candidates, key=lambda item: (item.recent_hits, item.last_step)).key
        if self.policy == "least_stale":
            return max(candidates, key=lambda item: (self._least_stale_score(item, step), -item.hits)).key
        return min(candidates, key=lambda item: item.last_step).key

    def _least_stale_score(self, item: CacheItem, step: int) -> int:
        gap = item.avg_gap if item.avg_gap > 0 else self.ttl_steps
        predicted_next = item.last_step + gap
        if predicted_next <= step:
            return 1_000_000_000 + step - predicted_next
        return predicted_next - step


def iter_jsonl(path: Path, required_text: str):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if required_text and required_text not in line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def is_expert_tensor(name: str) -> bool:
    return "blk." in name and any(marker in name for marker in EXPERT_TENSOR_MARKERS)


def load_tensors(trace_dir: Path, max_expert_by_layer: dict[int, int]) -> dict[int, list[TensorInfo]]:
    tensors: dict[int, list[TensorInfo]] = {}
    seen: set[tuple[int, str]] = set()
    path = trace_dir / "tensor_trace.jsonl"
    if not path.exists():
        return tensors
    found_load = False
    non_load_after_found = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if '"TENSOR_LOAD"' not in line:
                if found_load:
                    non_load_after_found += 1
                    if non_load_after_found > 200000:
                        break
                continue
            found_load = True
            non_load_after_found = 0
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "TENSOR_LOAD":
                continue
            name = str(row.get("tensor", ""))
            layer = int(row.get("layer", -1))
            size = int(row.get("size", 0))
            if layer < 0 or size <= 0 or not is_expert_tensor(name):
                continue
            key = (layer, name)
            if key in seen:
                continue
            seen.add(key)
            n_experts = max(1, max_expert_by_layer.get(layer, 0) + 1)
            tensors.setdefault(layer, []).append(TensorInfo(layer=layer, name=name, size=size, slice_size=max(1, size // n_experts)))
    return tensors


def valid_experts(row: dict[str, Any]) -> list[int]:
    experts = row.get("experts", [])
    if not isinstance(experts, list):
        return []
    return [int(e) for e in experts if isinstance(e, int) and e >= 0]


def load_routes(trace_dir: Path) -> tuple[list[dict[str, Any]], dict[int, int]]:
    routes: list[dict[str, Any]] = []
    max_expert_by_layer: dict[int, int] = {}
    for row in iter_jsonl(trace_dir / "expert_trace.jsonl", '"EXPERT_ROUTE"'):
        if row.get("event") != "EXPERT_ROUTE":
            continue
        layer = int(row.get("layer", -1))
        experts = valid_experts(row)
        if layer < 0 or not experts:
            continue
        max_expert_by_layer[layer] = max(max_expert_by_layer.get(layer, 0), max(experts))
        routes.append(row)
    return routes, max_expert_by_layer


def build_accesses(trace_dir: Path, prefetch_topk: int) -> list[Access]:
    routes, max_expert_by_layer = load_routes(trace_dir)
    tensors_by_layer = load_tensors(trace_dir, max_expert_by_layer)
    accesses: list[Access] = []
    for row in routes:
        layer = int(row.get("layer", -1))
        tensors = tensors_by_layer.get(layer, [])
        if not tensors:
            continue
        experts = valid_experts(row)
        if prefetch_topk > 0:
            experts = experts[:prefetch_topk]
        scores = row.get("scores", [])
        step = int(row.get("step", 0))
        for idx, expert in enumerate(experts):
            score = float(scores[idx]) if isinstance(scores, list) and idx < len(scores) else 0.0
            for tensor in tensors:
                accesses.append(
                    Access(
                        step=step,
                        layer=layer,
                        expert=expert,
                        tensor_name=tensor.name,
                        size=tensor.slice_size,
                        score=score,
                    )
                )
    return accesses


def simulate(accesses: list[Access], policies: list[str], budgets_mb: list[int], ttl_steps: int) -> list[SimResult]:
    results: list[SimResult] = []
    for policy in policies:
        for budget in budgets_mb:
            cache = PolicyCache(policy, budget, ttl_steps)
            for access in accesses:
                cache.access(access)
            results.append(cache.result)
    return results


def write_csv(results: list[SimResult], out_dir: Path) -> Path:
    path = out_dir / "expert_cache_simulation.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "policy",
                "budget_mb",
                "accesses",
                "hits",
                "misses",
                "hit_rate_pct",
                "prefetch_events",
                "evictions",
                "estimated_hint_events",
                "miss_gib",
                "evict_gib",
                "peak_cache_mb",
                "skips",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.policy,
                    r.budget_mb,
                    r.accesses,
                    r.hits,
                    r.misses,
                    f"{r.hit_rate_pct:.4f}",
                    r.prefetch_events,
                    r.evictions,
                    r.estimated_hint_events,
                    f"{r.miss_gib:.4f}",
                    f"{r.evict_gib:.4f}",
                    f"{r.peak_cache_mb:.4f}",
                    r.skips,
                ]
            )
    return path


def plot_results(results: list[SimResult], out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    policies = list(dict.fromkeys(r.policy for r in results))
    colors = ["#78909C", "#42A5F5", "#66BB6A", "#AB47BC", "#FFA726"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, policy in enumerate(policies):
        rows = sorted([r for r in results if r.policy == policy], key=lambda r: r.budget_mb)
        ax.plot([r.budget_mb for r in rows], [r.hit_rate_pct for r in rows], marker="o", label=policy, color=colors[idx % len(colors)])
    ax.set_title("Expert Cache Hit Rate by Budget")
    ax.set_xlabel("Cache budget (MiB)")
    ax.set_ylabel("Hit rate (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = out_dir / "01_expert_cache_hit_rate.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, policy in enumerate(policies):
        rows = sorted([r for r in results if r.policy == policy], key=lambda r: r.budget_mb)
        ax.plot([r.budget_mb for r in rows], [r.estimated_hint_events for r in rows], marker="o", label=policy, color=colors[idx % len(colors)])
    ax.set_title("Estimated OS Hint Events by Budget")
    ax.set_xlabel("Cache budget (MiB)")
    ax.set_ylabel("prefetch + evict events")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = out_dir / "02_expert_cache_hint_events.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def pareto_front(results: list[SimResult]) -> list[SimResult]:
    def dominates(a: SimResult, b: SimResult) -> bool:
        better_or_equal = (
            a.hit_rate_pct >= b.hit_rate_pct
            and a.estimated_hint_events <= b.estimated_hint_events
            and a.evict_bytes <= b.evict_bytes
            and a.peak_cache_bytes <= b.peak_cache_bytes
        )
        strictly_better = (
            a.hit_rate_pct > b.hit_rate_pct
            or a.estimated_hint_events < b.estimated_hint_events
            or a.evict_bytes < b.evict_bytes
            or a.peak_cache_bytes < b.peak_cache_bytes
        )
        return better_or_equal and strictly_better

    return [r for r in results if not any(dominates(other, r) for other in results if other is not r)]


def write_summary(results: list[SimResult], out_dir: Path) -> Path:
    front = pareto_front(results)
    candidates = sorted(
        [r for r in results if r.policy != "route"],
        key=lambda r: (-r.hit_rate_pct, r.estimated_hint_events, r.evict_bytes, r.budget_mb),
    )[:2]
    lines = [
        "# Expert Cache Policy Simulation",
        "",
        "This is a trace-driven estimate from existing LLM_MEM_TRACE output. It ranks policies before running expensive inference experiments.",
        "",
        "## Suggested Real Runs",
        "",
    ]
    if candidates:
        for r in candidates:
            lines.append(
                f"- `{r.policy}_{r.budget_mb}mb`: hit_rate={r.hit_rate_pct:.2f}%, "
                f"estimated_hint_events={r.estimated_hint_events:,}, miss={r.miss_gib:.2f} GiB, "
                f"evict={r.evict_gib:.2f} GiB."
            )
    else:
        lines.append("- No cache-policy candidate found; check trace inputs.")
    lines.extend(["", "## Pareto Front", ""])
    for r in sorted(front, key=lambda x: (x.policy, x.budget_mb)):
        lines.append(
            f"- `{r.policy}` budget={r.budget_mb} MiB, hit_rate={r.hit_rate_pct:.2f}%, "
            f"hints={r.estimated_hint_events:,}, peak_cache={r.peak_cache_mb:.1f} MiB."
        )
    path = out_dir / "simulation_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument("--budgets-mb", default=",".join(str(x) for x in DEFAULT_BUDGETS_MB))
    parser.add_argument("--ttl-steps", type=int, default=4)
    parser.add_argument("--prefetch-topk", type=int, default=0)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.output_dir) if args.output_dir else trace_dir / "expert_cache_simulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    accesses = build_accesses(trace_dir, args.prefetch_topk)
    results = simulate(accesses, args.policies, parse_int_list(args.budgets_mb), args.ttl_steps)
    csv_path = write_csv(results, out_dir)
    plots = plot_results(results, out_dir)
    summary_path = write_summary(results, out_dir)

    print(f"[OK] accesses={len(accesses):,}")
    print(f"[OK] {csv_path}")
    for path in plots:
        print(f"[OK] {path}")
    print(f"[OK] {summary_path}")


if __name__ == "__main__":
    main()
