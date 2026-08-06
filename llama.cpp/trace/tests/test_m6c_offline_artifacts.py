from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from m6c_offline.artifact_validator import validate_report_artifacts
from m6c_offline.artifacts import (
    ArtifactFinalizationError,
    inspect_closed_jsonl,
    write_finalized_jsonl,
)
from m6c_offline.evidence import COUNTERFACTUAL_DECLARATION, validate_evidence_index


RECORDS = ({"decision_id": 0, "winner": 7}, {"decision_id": 1, "winner": 9})


def write_records(stream) -> dict:
    for record in RECORDS:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return {"passed": True, "decisions": len(RECORDS)}


class StreamProxy:
    def __init__(self, wrapped, *, flush_error: bool = False, close_error: bool = False) -> None:
        self.wrapped = wrapped
        self.flush_error = flush_error
        self.close_error = close_error

    @property
    def closed(self) -> bool:
        return self.wrapped.closed

    def write(self, value: str) -> int:
        return self.wrapped.write(value)

    def flush(self) -> None:
        if self.flush_error:
            raise OSError("injected flush failure")
        self.wrapped.flush()

    def close(self) -> None:
        self.wrapped.close()
        if self.close_error:
            raise OSError("injected close failure")

    def fileno(self) -> int:
        return self.wrapped.fileno()


class TestM6CArtifactFinalization(unittest.TestCase):
    def _finalized(self, root: Path, name: str = "decisions.jsonl"):
        return write_finalized_jsonl(
            root / name,
            write_records,
            expected_line_count=len(RECORDS),
        )

    def test_01_pre_close_buffered_hash_can_differ_from_final_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "buffered.jsonl"
            stream = path.open("x", encoding="utf-8", buffering=8192)
            stream.write(json.dumps(RECORDS[0]) + "\n")
            pre_close = hashlib.sha256(path.read_bytes()).hexdigest()
            stream.close()
            final = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertNotEqual(pre_close, final)

    def test_02_production_inspector_runs_only_after_successful_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {}

            def opener(path: Path):
                proxy = StreamProxy(path.open("x", encoding="utf-8"))
                state["stream"] = proxy
                return proxy

            def inspector(path: Path, *, expected_line_count: int):
                self.assertTrue(state["stream"].closed)
                return inspect_closed_jsonl(path, expected_line_count=expected_line_count)

            _, metadata = write_finalized_jsonl(
                root / "decisions.jsonl",
                write_records,
                expected_line_count=2,
                opener=opener,
                inspector=inspector,
            )
            self.assertTrue(metadata["finalized_after_close"])

    def test_03_reported_size_equals_closed_file_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, metadata = self._finalized(root)
            self.assertEqual(metadata["size_bytes"], (root / "decisions.jsonl").stat().st_size)

    def test_04_reported_sha_equals_independent_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, metadata = self._finalized(root)
            independent = hashlib.sha256((root / "decisions.jsonl").read_bytes()).hexdigest()
            self.assertEqual(metadata["sha256"], independent)

    def test_05_jsonl_line_count_equals_decision_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, metadata = self._finalized(Path(directory))
            self.assertEqual(metadata["line_count"], len(RECORDS))
            self.assertTrue(metadata["line_count_matches"])

    def test_06_every_jsonl_line_is_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, metadata = self._finalized(root)
            self.assertTrue(metadata["jsonl_parseable"])
            for line in (root / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
                self.assertIsInstance(json.loads(line), dict)

    def test_07_final_jsonl_line_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, metadata = self._finalized(root)
            self.assertTrue(metadata["final_line_complete"])
            self.assertTrue((root / "decisions.jsonl").read_bytes().endswith(b"\n"))

    def test_08_write_exception_never_returns_pass_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.jsonl"

            def failing_write(stream):
                stream.write(json.dumps(RECORDS[0]) + "\n")
                raise RuntimeError("injected write failure")

            with self.assertRaises(ArtifactFinalizationError):
                write_finalized_jsonl(path, failing_write, expected_line_count=2)
            self.assertFalse(inspect_closed_jsonl(path, expected_line_count=2)["passed"])

    def test_09_flush_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flush.jsonl"
            with self.assertRaisesRegex(ArtifactFinalizationError, "write or flush"):
                write_finalized_jsonl(
                    path,
                    write_records,
                    expected_line_count=2,
                    opener=lambda value: StreamProxy(
                        value.open("x", encoding="utf-8"), flush_error=True
                    ),
                )

    def test_10_close_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "close.jsonl"
            with self.assertRaisesRegex(ArtifactFinalizationError, "close failed"):
                write_finalized_jsonl(
                    path,
                    write_records,
                    expected_line_count=2,
                    opener=lambda value: StreamProxy(
                        value.open("x", encoding="utf-8"), close_error=True
                    ),
                )

    def test_11_identical_input_has_identical_content_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, first = self._finalized(root, "first.jsonl")
            _, second = self._finalized(root, "second.jsonl")
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["size_bytes"], second["size_bytes"])


class TestM6CIndependentArtifactValidator(unittest.TestCase):
    def _report_fixture(self, root: Path, *, reported_sha: str | None = None) -> tuple[Path, Path]:
        artifact_dir = root / "artifacts"
        artifact_dir.mkdir()
        stream_path = artifact_dir / "run_1.jsonl"
        _, metadata = write_finalized_jsonl(
            stream_path,
            write_records,
            expected_line_count=2,
        )
        source_path = root / "source.txt"
        source_path.write_text("immutable evidence\n", encoding="utf-8")
        evidence_index = root / "evidence_index.json"
        evidence_index.write_text(json.dumps({
            "schema_version": "fixture",
            "source_run_membership": "one fixture run",
            "source_artifacts": [{
                "path": "source.txt",
                "size_bytes": source_path.stat().st_size,
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }],
        }), encoding="utf-8")
        evidence_hash = validate_evidence_index(root, evidence_index)
        run = {
            "run_id": "run_1",
            "decision_stream": "artifacts/run_1.jsonl",
            "decision_stream_size_bytes": metadata["size_bytes"],
            "decision_stream_line_count": metadata["line_count"],
            "decision_stream_sha256": reported_sha or metadata["sha256"],
            "decision_stream_jsonl_parseable": True,
            "decision_stream_final_line_complete": True,
            "decision_stream_finalized_after_close": True,
            "s0_decisions_executed": 2,
            "s0_oracle_winner_mismatches": 0,
            "runtime_deadline_score_winner_mismatches": 0,
            "full_store_scan_count": 0,
            "stale_handle_count": 0,
            "final_queue_empty": True,
            "passed": True,
            "deterministic_rerun": {
                "executed": True,
                "content_sha256": metadata["sha256"],
                "line_count": 2,
                "s0_result_matches": True,
                "content_matches_finalized_stream": True,
            },
        }
        report = {
            "schema_version": "m6c-b2-evidence-finalization-report-v1",
            "final_enum": "M6C_B_EVIDENCE_REPAIRED_AND_CONFIRMED",
            "supersedes": "m6c_b_offline_20260805_a3_s0_v3",
            "supersession_reason": "decision stream SHA calculated before buffered stream finalization",
            "superseded_output_classification": "technically informative but non-canonical",
            "reconstructability": {
                "status_counts": {"RECONSTRUCTABLE": 1},
                "evidence_hash": evidence_hash,
            },
            "s0_replay": {
                "runs": [run],
                "total_decisions": 2,
                "s0_oracle_winner_mismatches": 0,
                "runtime_deadline_score_winner_mismatches": 0,
                "full_store_scan_count": 0,
                "stale_handle_count": 0,
                "all_invariants_passed": True,
            },
            "artifact_finalization": {
                "decision_stream_sha_report_matches_actual": "1/1",
                "decision_stream_line_count_matches": "1/1",
                "decision_stream_jsonl_parseable": "1/1",
            },
            "formal_parameters_selected": False,
            "s1_evidence_parameter_comparison_executed": False,
            **COUNTERFACTUAL_DECLARATION,
        }
        report_path = root / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path, stream_path

    def test_12_validator_detects_one_byte_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report_path, stream_path = self._report_fixture(root)
            self.assertTrue(validate_report_artifacts(root, report_path, expected_stream_count=1)["passed"])
            data = stream_path.read_bytes().replace(b'"winner": 7', b'"winner": 8', 1)
            stream_path.write_bytes(data)
            result = validate_report_artifacts(root, report_path, expected_stream_count=1)
            self.assertFalse(result["passed"])
            self.assertEqual(
                result["summary"]["decision_stream_sha_report_matches_actual"]["display"],
                "0/1",
            )

    def test_13_v3_pre_close_sha_regression_pattern_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            empty_buffer_sha = hashlib.sha256(b"").hexdigest()
            report_path, _ = self._report_fixture(root, reported_sha=empty_buffer_sha)
            result = validate_report_artifacts(root, report_path, expected_stream_count=1)
            self.assertFalse(result["passed"])
            self.assertEqual(
                result["summary"]["decision_stream_sha_report_matches_actual"]["display"],
                "0/1",
            )
            self.assertIn(
                "reported_sha_mismatch",
                {error["reason"] for error in result["streams"][0]["errors"]},
            )


if __name__ == "__main__":
    unittest.main()
