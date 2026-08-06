import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_m6c_f_high_pressure_active_ab as m6c_f


class M6CFHighPressureTest(unittest.TestCase):
    def test_wait_distribution(self) -> None:
        result = m6c_f.wait_distribution([1, 25_000_000, 50_000_000, 100_000_000])
        self.assertEqual(result["ge_25ms_count"], 3)
        self.assertEqual(result["ge_50ms_count"], 2)
        self.assertEqual(result["ge_100ms_count"], 1)
        self.assertEqual(result["max_ns"], 100_000_000)

    def test_confirmation_requires_stable_winner_change(self) -> None:
        analyses = []
        for scenario in ("P1", "P2", "P3"):
            analyses.append({
                "scenario": scenario,
                "stable_nonzero_winner_change": scenario == "P2",
                "majority_pair_count": 2,
                "tail_improvement_pair_count": 2 if scenario == "P2" else 0,
                "physical_improvement_pair_count": 0,
                "B1_winner_change_rates": [0.01, 0.02, 0.03] if scenario == "P2" else [0, 0, 0],
            })
        self.assertEqual(m6c_f.confirmation_choice(analyses)[0], "P2")
        analyses[1]["stable_nonzero_winner_change"] = False
        self.assertIsNone(m6c_f.confirmation_choice(analyses)[0])

    def test_artifact_validator_detects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "sample.json"
            artifact.write_text('{"ok":true}\n', encoding="utf-8")
            index = {
                "schema_version": "m6c-f-artifact-index-v1",
                "file_count": 1,
                "files": [m6c_f.file_record(artifact, root)],
            }
            m6c_f.base.write_json(root / "artifact_index.json", index)
            self.assertEqual(m6c_f.validate_artifacts(root), 0)
            artifact.write_text('{"ok":false}\n', encoding="utf-8")
            self.assertEqual(m6c_f.validate_artifacts(root), 2)
            validation = json.loads((root / "artifact_validation.json").read_text())
            self.assertFalse(validation["passed"])
            self.assertTrue(any("mismatch" in error for error in validation["errors"]))


if __name__ == "__main__":
    unittest.main()
