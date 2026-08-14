#!/usr/bin/env python3
"""Issue sequential streaming requests and retain service-side latency evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any


PROMPT = """Summarize the operating-system issues in memory-constrained MoE inference.
Address demand paging, KV cache allocation, expert-weight residency, and the
tradeoff between latency and a hard memory limit. Use a concise numbered list.
"""


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def streaming_request(endpoint: str, request_id: int, n_predict: int, timeout: float) -> dict[str, Any]:
    payload = {
        "prompt": PROMPT + f"\nRequest id: {request_id}.",
        "n_predict": n_predict,
        "stream": True,
        "cache_prompt": False,
        "temperature": 0.0,
        "seed": 1234,
    }
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ns = time.monotonic_ns()
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    token_event_intervals_ms: list[float] = []
    content_parts: list[str] = []
    event_count = 0
    final_timing: dict[str, Any] | None = None

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            event = json.loads(body)
            event_count += 1
            piece = event.get("content")
            if isinstance(piece, str) and piece:
                now_ns = time.monotonic_ns()
                if first_token_ns is None:
                    first_token_ns = now_ns
                elif last_token_ns is not None:
                    token_event_intervals_ms.append((now_ns - last_token_ns) / 1e6)
                last_token_ns = now_ns
                content_parts.append(piece)
            timings = event.get("timings")
            if isinstance(timings, dict):
                final_timing = timings

    completed_ns = time.monotonic_ns()
    output = "".join(content_parts)
    return {
        "request_id": request_id,
        "wall_time_ms": (completed_ns - started_ns) / 1e6,
        "ttft_ms": None if first_token_ns is None else (first_token_ns - started_ns) / 1e6,
        "stream_event_intervals_ms": token_event_intervals_ms,
        "stream_event_count": event_count,
        "output": output,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "server_timings": final_timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()
    if args.request_count < 2:
        raise SystemExit("request-count must be at least 2 to measure subsequent-request latency")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    requests = [
        streaming_request(args.endpoint, request_id, args.n_predict, args.timeout_s)
        for request_id in range(args.request_count)
    ]
    subsequent_intervals = [
        interval
        for item in requests[1:]
        for interval in item["stream_event_intervals_ms"]
    ]
    subsequent_decode_means = [
        float(item["server_timings"]["predicted_per_token_ms"])
        for item in requests[1:]
        if isinstance(item.get("server_timings"), dict)
        and item["server_timings"].get("predicted_per_token_ms") is not None
    ]
    combined_output = "".join(item["output"] for item in requests)
    result = {
        "schema_version": 1,
        "endpoint": args.endpoint,
        "request_count": args.request_count,
        "n_predict": args.n_predict,
        "requests": requests,
        "first_request_ttft_ms": requests[0]["ttft_ms"],
        "subsequent_stream_event_interarrival_p95_ms": percentile(subsequent_intervals, 0.95),
        "subsequent_server_decode_mean_p95_ms": percentile(subsequent_decode_means, 0.95),
        "combined_output_sha256": hashlib.sha256(combined_output.encode("utf-8")).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.metrics_output:
        if args.metrics_output.exists():
            raise SystemExit(f"refusing to overwrite metrics output: {args.metrics_output}")
        metrics_endpoint = args.endpoint.rsplit("/completion", 1)[0] + "/metrics"
        with urllib.request.urlopen(metrics_endpoint, timeout=args.timeout_s) as response:
            metrics = response.read()
        args.metrics_output.write_bytes(metrics)


if __name__ == "__main__":
    main()
