#!/usr/bin/env python3
"""Wait until a local llama-server health endpoint returns HTTP 200."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    started = time.monotonic_ns()
    error = None
    while (time.monotonic_ns() - started) / 1e9 < args.timeout_s:
        try:
            with urllib.request.urlopen(args.url, timeout=2.0) as response:
                if response.status == 200:
                    elapsed = (time.monotonic_ns() - started) / 1e9
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps({"health_url": args.url, "wait_s": elapsed}, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    return
        except Exception as exc:  # endpoint is expected to be unavailable while loading
            error = str(exc)
        time.sleep(0.2)
    raise SystemExit(f"server did not become healthy within {args.timeout_s}s: {error}")


if __name__ == "__main__":
    main()
