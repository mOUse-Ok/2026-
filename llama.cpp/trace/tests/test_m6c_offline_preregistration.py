from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from m6c_offline.preregistration import (
    CALIBRATION,
    HOLDOUT,
    ROBUSTNESS,
    _write_closed_text,
    ceil_quantum,
    derive_ages,
    nearest_rank,
    read_s0_waits,
    validate_run_split,
)


class TestM6CPreregistration(unittest.TestCase):
    def test_nearest_rank_is_integer_and_deterministic(self) -> None:
        values = [40, 10, 30, 20]
        self.assertEqual(nearest_rank(values, 0.75), 30)
        self.assertEqual(nearest_rank(values, 0.90), 40)
        with self.assertRaises(ValueError):
            nearest_rank([], 0.75)

    def test_frozen_ceil_quantum(self) -> None:
        self.assertEqual(ceil_quantum(1), 1_000_000)
        self.assertEqual(ceil_quantum(1_000_000), 1_000_000)
        self.assertEqual(ceil_quantum(1_000_001), 2_000_000)

    def test_derive_ages_uses_cross_worker_max_and_rejects_collapse(self) -> None:
        stats, ages = derive_ages({
            2: [100_000, 1_100_000, 2_100_000, 3_100_000],
            4: [200_000, 1_200_000, 4_200_000, 6_200_000],
        })
        self.assertEqual(stats["4"]["p75_ns"], 4_200_000)
        self.assertEqual(ages, {"AGE_MODERATE": 5_000_000, "AGE_SPARSE": 7_000_000})
        with self.assertRaises(ValueError):
            derive_ages({2: [1, 1], 4: [1, 1]})

    def test_exact_split_contract(self) -> None:
        runs = []
        for run_id, workers, repeat, configuration in CALIBRATION + HOLDOUT + ROBUSTNESS:
            runs.append({
                "run_id": run_id,
                "workers": workers,
                "repeat_index": repeat,
                "configuration_id": configuration,
                "service_slots": 29_262,
                "source_priority_mode": "deadline_score" if configuration == "B0" else "max_wait_protection",
                "runtime_deadline_score_winner_mismatches": 0 if configuration == "B0" else None,
            })
        result = validate_run_split({"runs": runs})
        self.assertTrue(result["passed"])
        runs[0]["repeat_index"] = 9
        self.assertFalse(validate_run_split({"runs": runs})["passed"])

    def test_s0_wait_reader_rejects_reserved_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            stream_path = root / "stream.jsonl"
            record = {
                "decision_id": 0,
                "mode": "s1",
                "selected_source": "reserved",
                "source_priority_mode": "deadline_score",
                "reserved_due": True,
                "credit_before": 0,
                "credit_accrued": 1,
                "credit_after": 1,
                "debt_before": False,
                "debt_after": False,
                "decision_ts_ns": 20,
                "selected_task": {"enqueued_ts_ns": 10, "waiting_ns": 10},
            }
            stream_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            run = {
                "run_id": "fixture",
                "decision_stream": "stream.jsonl",
                "decision_stream_size_bytes": stream_path.stat().st_size,
                "decision_stream_line_count": 1,
                "decision_stream_sha256": hashlib.sha256(stream_path.read_bytes()).hexdigest(),
            }
            with self.assertRaisesRegex(ValueError, "non-S0"):
                read_s0_waits(root, run)

    def test_closed_text_metadata_uses_reopened_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            metadata = _write_closed_text(path, '{"ok": true}\n', parse_json=True)
            self.assertTrue(metadata["closed_before_hash"])
            self.assertEqual(metadata["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
