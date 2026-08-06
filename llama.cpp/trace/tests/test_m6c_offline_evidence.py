from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from m6c_offline.evidence import (
    COUNTERFACTUAL_DECLARATION,
    ReconstructabilityStatus,
    audit_run,
    replay_s0,
    validate_evidence_index,
)


def bits(value: float) -> str:
    return "0x" + struct.pack(">d", value).hex()


class TestM6COfflineEvidence(unittest.TestCase):
    def _write_fixture(
        self,
        root: Path,
        *,
        missing_bits: bool = False,
        ambiguous_time: bool = False,
    ) -> dict:
        run_id = "fixture_run"
        run_dir = root / run_id
        run_dir.mkdir()
        manifest = {
            "run_name": run_id,
            "environment": {
                "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE": "deadline_score",
                "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE": "16",
            },
        }
        summary = {
            "sinks": {
                "memory": {"enabled": True, "enqueued": 14, "written": 14, "dropped": 0}
            }
        }
        validation = {
            "passed": True,
            "selection_winner_replay": {"passed": True},
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "m6b2a3_validation.json").write_text(json.dumps(validation), encoding="utf-8")

        records = []
        for task_id, enqueue_time, deadline in ((1, 10, 100), (2, 20, 200)):
            common = {
                "event": "EXPERT_TASK",
                "task_id": task_id,
                "step": 1,
                "layer": 0,
                "expert": task_id,
                "phase": "PREFILL",
                "stage": "LATE",
                "tensor": "x",
                "nbytes": 10,
                "score_f64_bits": None if missing_bits and task_id == 1 else bits(0.1),
                "sequence": task_id - 1,
                "deadline_ts_ns": deadline,
                "enqueued_ts_ns": 0,
            }
            for lifecycle, state, timestamp in (
                ("CREATE", "CREATED", enqueue_time - 2),
                ("ADMIT", "ADMITTED", enqueue_time - 1),
                ("ENQUEUE", "ENQUEUED", enqueue_time),
            ):
                record = dict(common)
                record.update({"lifecycle_event": lifecycle, "state": state, "ts_ns": timestamp})
                if lifecycle == "ENQUEUE":
                    record["enqueued_ts_ns"] = enqueue_time
                records.append(record)
        first_decision = 10 if ambiguous_time else 30
        for decision_id, winner, now, depth in ((0, 1, first_decision, 2), (1, 2, 40, 1)):
            records.append({
                "event": "EXPERT_QUEUE_OVERHEAD_SELECTION",
                "decision_id": decision_id,
                "batch_id": decision_id,
                "batch_slot": 0,
                "worker_id": decision_id % 2,
                "batch_decision_ts_ns": now,
                "winner_task_id": winner,
                "queue_depth_before": depth,
                "priority_mode": "deadline_score",
            })
            records.append({
                "event": "EXPERT_PRIORITY_SELECTION",
                "task_id": winner,
                "decision_ts_ns": now + 1,
                "candidate_count": depth,
            })
            for lifecycle, state, timestamp in (
                ("DEQUEUE", "DEQUEUED", now + 2),
                ("ISSUE", "ISSUED", now + 3),
            ):
                records.append({
                    "event": "EXPERT_TASK",
                    "lifecycle_event": lifecycle,
                    "state": state,
                    "task_id": winner,
                    "ts_ns": timestamp,
                    "nbytes": 10,
                    "score_f64_bits": None if missing_bits and winner == 1 else bits(0.1),
                    "sequence": winner - 1,
                    "deadline_ts_ns": 100 if winner == 1 else 200,
                    "enqueued_ts_ns": 10 if winner == 1 else 20,
                })
        with (run_dir / "memory_trace.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        return {
            "run_id": run_id,
            "evidence_dir": str(run_dir),
            "slot": {
                "configuration_id": "B0",
                "workers": 2,
                "repeat_index": 1,
            },
        }

    def test_reconstructable_fixture_and_s0_runtime_winner_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entry = self._write_fixture(Path(directory))
            result = audit_run(entry, global_hash_passed=True)
            self.assertEqual(result.status, ReconstructabilityStatus.RECONSTRUCTABLE)
            self.assertIsNotNone(result.run)
            replay = replay_s0(result.run)
            self.assertTrue(replay["passed"])
            self.assertEqual(replay["s0_oracle_winner_mismatches"], 0)
            self.assertEqual(replay["runtime_deadline_score_winner_mismatches"], 0)
            self.assertEqual(replay["full_store_scan_count"], 0)

    def test_missing_route_score_bits_is_not_filled_from_decimal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entry = self._write_fixture(Path(directory), missing_bits=True)
            result = audit_run(entry, global_hash_passed=True)
            self.assertEqual(result.status, ReconstructabilityStatus.MISSING_ROUTE_SCORE_BITS)
            self.assertIsNone(result.run)

    def test_equal_enqueue_and_decision_without_queue_op_is_order_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entry = self._write_fixture(Path(directory), ambiguous_time=True)
            result = audit_run(entry, global_hash_passed=True)
            self.assertEqual(result.status, ReconstructabilityStatus.ORDER_UNAVAILABLE)
            self.assertIsNone(result.run)

    def test_global_hash_failure_stops_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entry = self._write_fixture(Path(directory))
            result = audit_run(entry, global_hash_passed=False)
            self.assertEqual(result.status, ReconstructabilityStatus.HASH_MISMATCH)

    def test_evidence_index_recomputes_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("immutable\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            index = root / "index.json"
            index.write_text(json.dumps({
                "schema_version": "fixture",
                "source_run_membership": "explicit fixture",
                "source_artifacts": [{
                    "path": "source.txt",
                    "size_bytes": source.stat().st_size,
                    "sha256": digest,
                }],
            }), encoding="utf-8")
            result = validate_evidence_index(root, index)
            self.assertTrue(result["passed"])
            source.write_text("changed\n", encoding="utf-8")
            result = validate_evidence_index(root, index)
            self.assertFalse(result["passed"])

    def test_counterfactual_declaration_is_exact(self) -> None:
        self.assertEqual(COUNTERFACTUAL_DECLARATION, {
            "counterfactual_type": "fixed_arrival_fixed_service_slot_policy_replay",
            "physical_system_reexecuted": False,
            "performance_claim": False,
        })


if __name__ == "__main__":
    unittest.main()
