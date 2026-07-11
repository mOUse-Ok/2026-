#!/usr/bin/env python3
"""Dependency-free core metrics for LLM memory trace records."""

from __future__ import annotations

from statistics import mean


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def inference_latency_records(memory_records: list[dict]) -> tuple[list[dict], str]:
    """Return one authoritative latency record per processed ubatch."""
    step_ends = [
        record
        for record in memory_records
        if record.get("event") == "STEP_END" and "latency_ns" in record
    ]
    if step_ends:
        return step_ends, "step_end"
    token_ends = [
        record
        for record in memory_records
        if record.get("event") == "TOKEN_END" and "latency_ns" in record
    ]
    return token_ends, "token_end_legacy"


def collect_core_metrics(memory_records: list[dict]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    mem_events = [
        record
        for record in memory_records
        if record.get("event") == "MEMORY_STAT" and "rss_bytes" in record
    ]
    if mem_events:
        rss_values = [int(record["rss_bytes"]) for record in mem_events]
        metrics["rss_peak_gb"] = max(rss_values) / (1024**3)
        metrics["rss_avg_gb"] = mean(rss_values) / (1024**3)
        first_minor = int(mem_events[0].get("minor_faults", 0))
        last_minor = int(mem_events[-1].get("minor_faults", first_minor))
        first_major = int(mem_events[0].get("major_faults", 0))
        last_major = int(mem_events[-1].get("major_faults", first_major))
        metrics["trace_window_minor_faults"] = max(0, last_minor - first_minor)
        metrics["trace_window_major_faults"] = max(0, last_major - first_major)
        metrics["total_minor_faults"] = metrics["trace_window_minor_faults"]
        metrics["total_major_faults"] = metrics["trace_window_major_faults"]
        metrics["fault_metric_source"] = "trace_window"
        metrics["minor_faults_cumulative_initial"] = first_minor
        metrics["major_faults_cumulative_initial"] = first_major
        metrics["minor_faults_cumulative_final"] = last_minor
        metrics["major_faults_cumulative_final"] = last_major
        metrics["mmap_count"] = int(mem_events[-1].get("mmap_count", 0))

        smaps_events = [record for record in mem_events if "pss_bytes" in record]
        if smaps_events:
            metrics["pss_peak_gb"] = max(int(record.get("pss_bytes", 0)) for record in smaps_events) / (1024**3)
            metrics["pss_avg_gb"] = mean(int(record.get("pss_bytes", 0)) for record in smaps_events) / (1024**3)
            metrics["shared_clean_peak_gb"] = max(int(record.get("shared_clean_bytes", 0)) for record in smaps_events) / (1024**3)
            metrics["private_dirty_peak_gb"] = max(int(record.get("private_dirty_bytes", 0)) for record in smaps_events) / (1024**3)
            metrics["swap_peak_mb"] = max(int(record.get("swap_bytes", 0)) for record in smaps_events) / (1024**2)

    latency_events, latency_source = inference_latency_records(memory_records)
    if latency_events:
        prefill = [record for record in latency_events if record.get("phase") == "PREFILL"]
        decode = [record for record in latency_events if record.get("phase") == "DECODE"]
        metrics["latency_metric_source"] = latency_source
        metrics["prefill_steps"] = len(prefill)
        metrics["decode_steps"] = len(decode)
        metrics["prefill_tokens"] = sum(int(record.get("n_tokens", 1)) for record in prefill)
        metrics["decode_tokens"] = sum(int(record.get("n_tokens", 1)) for record in decode)
        prefill_us = [float(record["latency_ns"]) / 1e3 for record in prefill]
        decode_us = [float(record["latency_ns"]) / 1e3 for record in decode]
        metrics["prefill_avg_latency_us"] = mean(prefill_us) if prefill_us else 0.0
        metrics["prefill_total_latency_ms"] = sum(prefill_us) / 1e3 if prefill_us else 0.0
        metrics["decode_avg_latency_us"] = mean(decode_us) if decode_us else 0.0
        metrics["decode_p50_latency_us"] = percentile(decode_us, 50)
        metrics["decode_p95_latency_us"] = percentile(decode_us, 95)
        metrics["decode_p99_latency_us"] = percentile(decode_us, 99)
        decode_seconds = sum(decode_us) / 1e6
        metrics["decode_throughput_tokens_per_s"] = (
            int(metrics["decode_tokens"]) / decode_seconds if decode_seconds > 0 else 0.0
        )

        trace_starts = [
            record
            for record in memory_records
            if record.get("event") == "TRACE_START" and "ts_ns" in record
        ]
        if trace_starts and decode:
            metrics["trace_to_first_decode_ms"] = max(
                0.0,
                (
                    min(int(record["ts_ns"]) for record in decode)
                    - min(int(record["ts_ns"]) for record in trace_starts)
                )
                / 1e6,
            )

    return metrics
