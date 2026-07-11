#!/usr/bin/env python3
"""Fail a benchmark run when an enabled trace sink silently lost events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--allow-dropped", action="store_true")
    parser.add_argument("--expect-sink", action="append", default=[])
    args = parser.parse_args()

    if not args.summary.is_file():
        raise SystemExit(f"missing trace summary: {args.summary}")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    sinks = summary.get("sinks")
    if not isinstance(sinks, dict) or not sinks:
        raise SystemExit("trace summary has no sink accounting")

    for name in args.expect_sink:
        counts = sinks.get(name)
        if not isinstance(counts, dict) or not counts.get("enabled"):
            raise SystemExit(f"expected trace sink is not enabled: {name}")

    dropped = {
        name: int(values.get("dropped", 0))
        for name, values in sinks.items()
        if isinstance(values, dict) and values.get("enabled")
    }
    incomplete = {
        name: values
        for name, values in sinks.items()
        if isinstance(values, dict)
        and values.get("enabled")
        and int(values.get("enqueued", 0)) != int(values.get("written", 0))
    }
    if incomplete:
        raise SystemExit(f"trace sink write mismatch: {incomplete}")
    if sum(dropped.values()) and not args.allow_dropped:
        raise SystemExit(f"trace events were dropped: {dropped}")
    print(json.dumps({"complete": True, "dropped": dropped}, sort_keys=True))


if __name__ == "__main__":
    main()
