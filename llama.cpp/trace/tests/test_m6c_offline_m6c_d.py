from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from m6c_offline.evidence import EvidenceRun
from m6c_offline.m6c_d_runner import (
    _candidate_map,
    _policy_config,
    _safety_pass,
    build_synthetic_runs,
    create_preregistration_v2,
    execute_pipeline,
    replay_case,
    select_finalists,
)
from m6c_offline.m6c_d_validator import validate_output, validate_preregistration_v2


REPO_ROOT = Path(__file__).resolve().parents[3]
V1_DIR = REPO_ROOT / "llama.cpp/trace_output/m6c_c_preregistration_20260805_v1"


def load_v1() -> tuple[dict, str]:
    path = V1_DIR / "m6c_c_preregistration.json"
    return json.loads(path.read_text(encoding="utf-8")), hashlib.sha256(path.read_bytes()).hexdigest()


class TestM6CDRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preregistration, cls.preregistration_sha = load_v1()
        cls.runs = build_synthetic_runs(cls.preregistration)
        cls.candidates = _candidate_map(cls.preregistration)

    def first_run(self) -> EvidenceRun:
        run_id = self.preregistration["run_split"]["calibration"][0]["run_id"]
        return self.runs[run_id]

    def test_four_candidate_states_are_isolated(self) -> None:
        run = self.first_run()
        before = {}
        after = {}
        for candidate_id, candidate in self.candidates.items():
            config = _policy_config(candidate, self.preregistration_sha, synthetic=True)
            first = replay_case(run, config)
            before[candidate_id] = first["task_metrics"][0]["task_id"]
            after[candidate_id] = first["decision_digest_sha256"]
        repeated = replay_case(
            run,
            _policy_config(self.candidates["S1-A"], self.preregistration_sha, synthetic=True),
        )
        self.assertEqual(after["S1-A"], repeated["decision_digest_sha256"])
        self.assertEqual(set(before), set(self.candidates))
        self.assertTrue(all(value > 0 for value in before.values()))

    def test_single_run_requires_dense_decision_order(self) -> None:
        broken = deepcopy(self.first_run())
        slots = list(broken.slots)
        slots[1], slots[2] = slots[2], slots[1]
        broken.slots = slots
        with self.assertRaisesRegex(RuntimeError, "decision IDs"):
            replay_case(broken, None)

    def test_repeated_input_is_deterministic(self) -> None:
        config = _policy_config(self.candidates["S1-C"], self.preregistration_sha, synthetic=True)
        first = replay_case(self.first_run(), config)
        second = replay_case(self.first_run(), config)
        self.assertTrue(first["deterministic_rerun"]["passed"])
        self.assertEqual(first["decision_digest_sha256"], second["decision_digest_sha256"])
        self.assertEqual(first["operation_counters"], second["operation_counters"])

    def test_safety_blockers_fail_closed(self) -> None:
        base = replay_case(self.first_run(), None)
        for field in (
            "hard_urgent_violation",
            "stale_handle_count",
            "full_store_scan_count",
            "task_conservation_error",
        ):
            value = deepcopy(base)
            value["safety"][field] = 1
            passed, blockers = _safety_pass(value)
            self.assertFalse(passed, field)
            self.assertIn(field, blockers)

    def test_calibration_selects_at_most_two(self) -> None:
        run = self.first_run()
        baseline = replay_case(run, None)
        candidate_results = {}
        from m6c_offline.m6c_d_runner import pair_results

        for candidate_id, candidate in self.candidates.items():
            paired = pair_results(
                baseline,
                replay_case(run, _policy_config(candidate, self.preregistration_sha, synthetic=True)),
            )
            # Calibration requires two Runs in each worker stratum.  Replication
            # here tests only the immutable maximum-finalist truncation.
            candidate_results[candidate_id] = [
                {**deepcopy(paired), "run_id": f"w2-{index}", "workers": 2}
                for index in range(2)
            ] + [
                {**deepcopy(paired), "run_id": f"w4-{index}", "workers": 4}
                for index in range(2)
            ]
        selection = select_finalists(candidate_results, self.candidates, stage="calibration")
        self.assertLessEqual(len(selection["selected_candidates"]), 2)
        self.assertEqual(selection["maximum_finalists"], 2)

    def test_no_unique_recommendation_never_loads_robustness(self) -> None:
        loaded = []

        def loader(role: str, record: dict) -> EvidenceRun:
            loaded.append(role)
            return self.runs[record["run_id"]]

        calls = 0

        def selection(candidate_results, candidates, *, stage):
            nonlocal calls
            calls += 1
            if stage == "calibration":
                return {
                    "stage": stage,
                    "candidate_gates": {},
                    "passing_candidates_in_frozen_order": ["S1-C"],
                    "selected_candidates": ["S1-C"],
                    "maximum_finalists": 2,
                    "tie_break": [],
                }
            return {
                "stage": stage,
                "candidate_gates": {},
                "passing_candidates_in_frozen_order": [],
                "selected_candidates": [],
                "maximum_finalists": 1,
                "tie_break": [],
            }

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "m6c_offline.m6c_d_runner.select_finalists", side_effect=selection
        ):
            report = execute_pipeline(
                self.preregistration,
                self.preregistration_sha,
                Path(directory) / "out",
                loader,
                synthetic=True,
            )
        self.assertEqual(calls, 2)
        self.assertNotIn("robustness", loaded)
        self.assertFalse(report["robustness_executed"])

    def test_full_synthetic_three_stage_and_holdout_scope(self) -> None:
        loaded = []

        def loader(role: str, record: dict) -> EvidenceRun:
            loaded.append((role, record["run_id"]))
            return self.runs[record["run_id"]]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            report = execute_pipeline(
                self.preregistration,
                self.preregistration_sha,
                output,
                loader,
                synthetic=True,
            )
            holdout = json.loads((output / "02_holdout/results.json").read_text(encoding="utf-8"))
            self.assertEqual(set(holdout["candidates"]), set(report["finalists"]))
            self.assertEqual(len(report["finalists"]), 2)
            self.assertIsNotNone(report["unique_recommendation"])
            self.assertEqual(report["robustness_runs_loaded"], 20)
            self.assertTrue(validate_output(output)["passed"])
        self.assertEqual(sum(role == "calibration" for role, _ in loaded), 4)
        self.assertEqual(sum(role == "holdout" for role, _ in loaded), 6)
        self.assertEqual(sum(role == "robustness" for role, _ in loaded), 20)

    def test_post_close_hash_and_tamper_detection(self) -> None:
        def loader(role: str, record: dict) -> EvidenceRun:
            return self.runs[record["run_id"]]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            execute_pipeline(
                self.preregistration,
                self.preregistration_sha,
                output,
                loader,
                synthetic=True,
            )
            manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["hashes_computed_after_close"])
            target = output / "01_calibration/calibration_selection.json"
            target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
            validation = validate_output(output)
            self.assertFalse(validation["passed"])
            self.assertIn("calibration_stage_mismatch", validation["errors"])

    def test_v2_inherits_v1_and_resolves_only_command_gate(self) -> None:
        audit = {"passed": True, "tests_run": 9}
        synthetic = {
            "passed": True,
            "stages": ["calibration", "holdout", "robustness"],
            "a3_evidence_opened": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            v2_dir = Path(directory) / "v2"
            result = create_preregistration_v2(
                V1_DIR,
                v2_dir,
                test_audit=audit,
                synthetic_audit=synthetic,
            )
            validation = validate_preregistration_v2(
                V1_DIR,
                v2_dir,
                REPO_ROOT / "llama.cpp/trace/m6c_offline/m6c_d_runner.py",
            )
            self.assertTrue(result["validation_passed"])
            self.assertTrue(validation["passed"])
            self.assertEqual(validation["added_paths"], ["command_gate_resolved", "runner_test_and_audit"])


if __name__ == "__main__":
    unittest.main()
