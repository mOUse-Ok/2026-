#!/usr/bin/env python3
"""Prepare a model file for a controlled cold or warm cache experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--mode", choices=("cold", "warm", "as-is"), default="cold")
    parser.add_argument("--chunk-mb", type=int, default=16)
    args = parser.parse_args()

    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit(f"model file not found: {model}")

    result: dict[str, object] = {
        "schema_version": 1,
        "mode": args.mode,
        "model": str(model),
        "size_bytes": model.stat().st_size,
        "method": "none",
    }

    if args.mode == "cold":
        if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
            raise SystemExit("cold mode requires os.posix_fadvise/POSIX_FADV_DONTNEED on Linux")
        with model.open("rb", buffering=0) as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        result["method"] = "posix_fadvise_dontneed"
    elif args.mode == "warm":
        chunk_size = max(1, args.chunk_mb) * 1024 * 1024
        bytes_read = 0
        with model.open("rb", buffering=0) as stream:
            while chunk := stream.read(chunk_size):
                bytes_read += len(chunk)
        result["method"] = "sequential_read"
        result["bytes_read"] = bytes_read

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
