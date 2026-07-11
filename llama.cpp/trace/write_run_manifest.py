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
        "memory.pressure",
    )
    return {name: read_text(directory / name) if directory else None for name in names}


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
            "CTX_SIZE",
            "TEMP",
            "SEED",
            "GPU_LAYERS",
            "TRACE_PROFILE",
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
        "schema_version": 2,
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
            "logical_cpus": os.cpu_count(),
        },
        "experiment": {
            "trace_profile": args.trace_profile,
            "cache_mode": args.cache_mode,
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
