#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TRACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACE_DIR))

from summarize_repeat_runs import validate_runs  # noqa: E402


class RepeatValidationTest(unittest.TestCase):
    def write_run(
            self,
            base: Path,
            run: str,
            *,
            commit: str = "abc",
            output_hash: str = "a" * 64) -> None:
        run_dir = base / run
        (run_dir / "analysis").mkdir(parents=True)
        manifest = {
            "git_commit": commit,
            "git_dirty": False,
            "model": {"sha256": "model-sha", "size_bytes": 123, "mtime_ns": 1},
            "prompt": {"sha256": "prompt-sha"},
            "binary": {"sha256": "binary-sha"},
            "host": {"kernel": "test", "cpu": "test"},
            "experiment": {
                "trace_profile": "benchmark",
                "cache_mode": "cold",
                "repeat_index": "1",
                "order_position": "1",
                "requested_memory_max": "8G",
                "requested_memory_swap_max": "1G",
                "cgroup": {"memory.max": "8589934592", "memory.swap.max": "1073741824"},
            },
            "environment": {
                "NUM_TOKENS_PREDICT": "80",
                "NUM_THREADS": "8",
                "BATCH_SIZE": "512",
                "CTX_SIZE": "2048",
                "TEMP": "0.0",
                "SEED": "1234",
                "GPU_LAYERS": "0",
            },
        }
        files = {
            "run_manifest.json": manifest,
            "cache_preparation.json": {"mode": "cold"},
            "process_metrics.json": {"exit_code": 0},
            "summary.json": {
                "sinks": {"memory": {"enabled": True, "enqueued": 5, "written": 5, "dropped": 0}},
            },
            "analysis/metrics.json": {
                "fault_metric_source": "gnu_time_process",
                "latency_metric_source": "step_end",
                "decode_steps": 80,
            },
        }
        for relative, content in files.items():
            (run_dir / relative).write_text(json.dumps(content), encoding="utf-8")
        (run_dir / "output.sha256").write_text(output_hash + "\n", encoding="ascii")

    def test_valid_runs_share_one_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.write_run(base, "baseline")
            self.write_run(base, "optimized")
            result = validate_runs(base, ["baseline", "optimized"])
            self.assertTrue(result["valid"])
            self.assertEqual(result["run_count"], 2)

    def test_mismatched_commit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.write_run(base, "baseline", commit="abc")
            self.write_run(base, "optimized", commit="def")
            with self.assertRaisesRegex(ValueError, "manifests disagree"):
                validate_runs(base, ["baseline", "optimized"])

    def test_mismatched_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.write_run(base, "baseline", output_hash="a" * 64)
            self.write_run(base, "optimized", output_hash="b" * 64)
            with self.assertRaisesRegex(ValueError, "output hashes disagree"):
                validate_runs(base, ["baseline", "optimized"])


if __name__ == "__main__":
    unittest.main()
