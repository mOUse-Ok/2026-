#!/usr/bin/env python3
"""Capture cgroup v2 memory state from inside the inference scope."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def parse_kv(text: str | None) -> dict[str, int | str] | None:
    if text is None:
        return None
    values: dict[str, int | str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(" ")
        if not sep or not key:
            continue
        try:
            values[key] = int(value)
        except ValueError:
            values[key] = value
    return values


def current_cgroup_dir() -> Path:
    cgroup_text = read_text(Path("/proc/self/cgroup"))
    if cgroup_text is None:
        raise SystemExit("cannot read /proc/self/cgroup")
    for line in cgroup_text.splitlines():
        hierarchy, _, relative = line.partition("::")
        if hierarchy == "0" and relative:
            directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
            if directory.is_dir():
                return directory
            raise SystemExit(f"cgroup directory is unavailable: {directory}")
    raise SystemExit("cgroup v2 entry is unavailable in /proc/self/cgroup")


def expected_bytes(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        pass
    suffix = value[-1:].upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if suffix not in multipliers:
        raise SystemExit(f"unsupported expected memory limit: {value}")
    try:
        return int(value[:-1]) * multipliers[suffix]
    except ValueError as exc:
        raise SystemExit(f"unsupported expected memory limit: {value}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--expected-memory-max")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite snapshot: {args.output}")

    directory = current_cgroup_dir()
    scalar_names = (
        "memory.current",
        "memory.high",
        "memory.max",
        "memory.peak",
        "memory.swap.current",
        "memory.swap.max",
        "memory.swap.peak",
    )
    values = {name: read_text(directory / name) for name in scalar_names}
    if args.expected_memory_max:
        actual_max = values["memory.max"]
        if actual_max is None or actual_max == "max" or int(actual_max) != expected_bytes(args.expected_memory_max):
            raise SystemExit(
                "requested memory limit is not active in the inference cgroup: "
                f"expected {args.expected_memory_max}, got {actual_max}"
            )

    snapshot = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "cgroup_path": str(directory),
        "memory": values,
        "memory_events": parse_kv(read_text(directory / "memory.events")),
        "memory_events_local": parse_kv(read_text(directory / "memory.events.local")),
        "memory_stat": parse_kv(read_text(directory / "memory.stat")),
        "memory_pressure": parse_kv(read_text(directory / "memory.pressure")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
