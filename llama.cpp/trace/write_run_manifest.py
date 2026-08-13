#!/usr/bin/env python3
"""Write a reproducibility manifest without touching the model page cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cgroup_dir() -> Path | None:
    text = read_text(Path("/proc/self/cgroup"))
    if not text:
        return None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            return Path("/sys/fs/cgroup") / parts[2].lstrip("/")
    return None


def cgroup_values() -> dict[str, str | None]:
    directory = cgroup_dir()
    names = (
        "memory.current",
        "memory.high",
        "memory.max",
        "memory.peak",
        "memory.swap.current",
        "memory.swap.max",
        "memory.swap.peak",
        "memory.events",
        "memory.events.local",
        "memory.stat",
        "memory.pressure",
    )
    values = {name: read_text(directory / name) if directory else None for name in names}
    values["source_path"] = str(directory) if directory else None
    try:
        values["source_inode"] = str(directory.stat().st_ino) if directory else None
    except OSError:
        values["source_inode"] = None
    return values


def git_output(project: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_model() -> str | None:
    text = read_text(Path("/proc/cpuinfo"))
    if text:
        for line in text.splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def cpuinfo_value(name: str) -> str | None:
    text = read_text(Path("/proc/cpuinfo"))
    if text:
        prefix = name.lower()
        for line in text.splitlines():
            if line.lower().startswith(prefix) and ":" in line:
                return line.split(":", 1)[1].strip()
    return None


def cpu_affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--llama-cli", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--trace-profile", required=True)
    parser.add_argument("--cache-mode", required=True)
    parser.add_argument("--expert-profile", required=True)
    parser.add_argument("--cache-type-k", required=True)
    parser.add_argument("--cache-type-v", required=True)
    parser.add_argument("--flash-attn", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--ubatch-size", required=True, type=int)
    parser.add_argument("--ctx-size", required=True, type=int)
    parser.add_argument("--repeat-index", default="")
    parser.add_argument("--order-position", default="")
    parser.add_argument("--order-mode", default="")
    parser.add_argument("--order-seed", default="")
    parser.add_argument("--memory-max", default="")
    parser.add_argument("--memory-swap-max", default="")
    parser.add_argument("--model-sha256", default="")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    model = args.model.resolve()
    prompt = args.prompt.resolve()
    llama_cli = args.llama_cli.resolve()
    model_stat = model.stat()
    cli_stat = llama_cli.stat()
    selected_env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("LLM_MEM_TRACE_")
        or key
        in {
            "NUM_TOKENS_PREDICT",
            "NUM_THREADS",
            "BATCH_SIZE",
            "UBATCH_SIZE",
            "CTX_SIZE",
            "EXPERT_PROFILE",
            "KV_CACHE_TYPE_K",
            "KV_CACHE_TYPE_V",
            "FLASH_ATTN",
            "TEMP",
            "SEED",
            "GPU_LAYERS",
            "TRACE_PROFILE",
            "OMP_NUM_THREADS",
            "OMP_DYNAMIC",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "MKL_DYNAMIC",
            "BLIS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "LC_ALL",
            "LANG",
        }
    }

    dirty = bool(git_output(args.project, "status", "--porcelain"))
    if dirty and args.require_clean:
        raise SystemExit("repository has uncommitted changes; commit or set ALLOW_DIRTY_REPO=1")

    cgroup = cgroup_values()
    if args.memory_max and cgroup.get("memory.max") in (None, "max"):
        raise SystemExit("requested memory limit is not active in the current cgroup")
    if args.memory_swap_max and cgroup.get("memory.swap.max") in (None, "max"):
        raise SystemExit("requested swap limit is not active in the current cgroup")

    manifest = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "git_commit": git_output(args.project, "rev-parse", "HEAD"),
        "git_dirty": dirty,
        "model": {
            "path": str(model),
            "size_bytes": model_stat.st_size,
            "mtime_ns": model_stat.st_mtime_ns,
            "sha256": args.model_sha256 or None,
        },
        "prompt": {
            "path": str(prompt),
            "size_bytes": prompt.stat().st_size,
            "sha256": sha256_file(prompt),
        },
        "binary": {
            "path": str(llama_cli),
            "size_bytes": cli_stat.st_size,
            "mtime_ns": cli_stat.st_mtime_ns,
            "sha256": sha256_file(llama_cli),
        },
        "host": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "cpu": cpu_model(),
            "microcode": cpuinfo_value("microcode"),
            "logical_cpus": os.cpu_count(),
            "cpu_affinity": cpu_affinity(),
        },
        "experiment": {
            "trace_profile": args.trace_profile,
            "cache_mode": args.cache_mode,
            "inference_config": {
                "expert_profile": args.expert_profile,
                "cache_type_k": args.cache_type_k,
                "cache_type_v": args.cache_type_v,
                "flash_attn": args.flash_attn,
                "batch_size": args.batch_size,
                "ubatch_size": args.ubatch_size,
                "ctx_size": args.ctx_size,
            },
            "repeat_index": args.repeat_index or None,
            "order_position": args.order_position or None,
            "order_mode": args.order_mode or None,
            "order_seed": args.order_seed or None,
            "requested_memory_max": args.memory_max or None,
            "requested_memory_swap_max": args.memory_swap_max or None,
            "cgroup": cgroup,
        },
        "environment": selected_env,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
