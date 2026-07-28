import json
import sys
import tempfile
import unittest
from pathlib import Path

TRACE_DIR = Path(__file__).resolve().parents[1]
if str(TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(TRACE_DIR))

import pressure_shadow_analysis as psa


def obs(value, timestamp, status="available", error=None):
    return {
        "value": value if status == "available" else None,
        "status": status,
        "read_ts_ns": timestamp,
        "error": error,
        "run_id": "fixture",
    }


def sample(ready, **fields):
    value = {
        "event": "PRESSURE_SHADOW_SAMPLE",
        "run_id": "fixture",
        "sample_ready_ts_ns": ready,
        "phase": {
            "value": "DECODE",
            "status": "available",
            "read_ts_ns": ready - 1,
            "error": None,
        },
    }
    value.update(fields)
    return value


class PressureShadowCausalityTests(unittest.TestCase):
    def test_null_is_not_zero(self):
        value = psa.observation(
            {"field": obs(None, 10, status="field_missing", error="missing")},
            "field",
        )
        self.assertIsNone(value["value"])
        self.assertEqual(value["status"], "field_missing")
        absent = psa.observation({}, "field")
        self.assertIsNone(absent["value"])
        self.assertEqual(absent["status"], "field_missing")

    def test_counter_baseline_is_first_read_strictly_after_state(self):
        samples = [
            sample(100, counter=obs(10, 90)),
            sample(120, counter=obs(11, 100)),
            sample(150, counter=obs(12, 140)),
            sample(190, counter=obs(15, 200)),
        ]
        result = psa.counter_future(samples, 0, "counter", 100)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["baseline_ts_ns"], 140)
        self.assertEqual(result["baseline_value"], 12)
        self.assertEqual(result["magnitude"], 3)
        self.assertEqual(result["timeline"], [(200, 3)])

    def test_window_is_open_left_closed_right(self):
        samples = [
            sample(100, counter=obs(1, 90)),
            sample(120, counter=obs(1, 110)),
            sample(150, counter=obs(2, 150)),
        ]
        result = psa.counter_future(samples, 0, "counter", 50)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["magnitude"], 1)
        self.assertEqual(result["timeline"], [(150, 1)])

    def test_counter_reset_is_unavailable(self):
        samples = [
            sample(100, counter=obs(9, 90)),
            sample(120, counter=obs(8, 110)),
            sample(140, counter=obs(2, 130)),
        ]
        result = psa.counter_future(samples, 0, "counter", 50)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "counter_reset")

    def test_next_decode_must_begin_strictly_after_state(self):
        steps = [
            psa.DecodeStep(1, 100, 120, 20),
            psa.DecodeStep(2, 101, 140, 39),
        ]
        self.assertEqual(psa.next_decode_step(steps, 100).step, 2)
        self.assertIsNone(psa.next_decode_step(steps, 101))

    def test_label_onset_is_first_threshold_crossing(self):
        raw = {
            "eligible": True,
            "magnitude": 5,
            "timeline": [(120, 1), (140, 4), (160, 5)],
        }
        label = psa.label_outcome(raw, 4)
        self.assertTrue(label["positive"])
        self.assertEqual(label["onset_ts_ns"], 140)


class PressureShadowStateTests(unittest.TestCase):
    def test_first_past_delta_is_unavailable(self):
        samples = [
            sample(100, counter=obs(10, 90)),
            sample(125, counter=obs(13, 115)),
        ]
        first = psa.past_counter_delta(samples, 0, "counter")
        second = psa.past_counter_delta(samples, 1, "counter")
        self.assertEqual(first["status"], "not_sampled")
        self.assertEqual(second["status"], "available")
        self.assertEqual(second["value"], 3)

    def test_runtime_delta_unavailability_is_not_backfilled(self):
        pressure = sample(
            100,
            memory_current_bytes=obs(90, 99),
            memory_high_bytes=obs(100, 99),
            memory_max_bytes=obs(None, 99, "available"),
            psi_some_delta_us=obs(None, 99, "no_previous_sample"),
            psi_full_delta_us=obs(None, 99, "no_previous_sample"),
            workingset_refault_anon_delta=obs(
                None, 99, "no_previous_sample"
            ),
            workingset_refault_file_delta=obs(
                None, 99, "no_previous_sample"
            ),
            pgscan_delta=obs(None, 99, "no_previous_sample"),
            pgsteal_delta=obs(None, 99, "no_previous_sample"),
            queue_depth=obs(0, 99),
            queued_bytes=obs(0, 99),
            worker_count=obs(2, 99),
        )
        run = psa.RunData(
            "fixture", Path("."), [pressure], [], [], 2, "100", "0"
        )
        feature = psa.sample_features(run)[0]
        self.assertEqual(feature["stall_status"], "unavailable")
        self.assertIsNone(
            feature["stall_values"]["refault_sum_delta"]["value"]
        )

    def test_unavailable_core_dimension_does_not_become_low(self):
        feature = {
            "run_id": "r1",
            "sample_index": 0,
            "sample_ready_ts_ns": 100,
            "phase": "DECODE",
            "memory_ratio": 0.5,
            "memory_status": "available",
            "stall_status": "unavailable",
            "stall_values": {},
            "queue_status": "available",
            "depth_per_worker": 0.0,
            "queued_bytes": 0.0,
        }
        thresholds = {
            "memory_ratio": {"medium": 0.75, "high": 0.90},
            "stall": {},
            "queue": {
                "depth_per_worker": {"medium": 1, "high": 2},
                "queued_bytes": {"medium": 1, "high": 2},
            },
        }
        states = psa.candidate_states([feature], thresholds)
        self.assertEqual(states["A_memory_only"][0]["state"], "LOW")
        for name in ("B_memory_and_risk", "C_two_of_three", "D_linear"):
            self.assertEqual(states[name][0]["state"], "UNAVAILABLE")

    def test_grouped_folds_do_not_overlap(self):
        folds = psa.grouped_folds(["r1", "r2", "r3"])
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertFalse(
                set(fold["training_run_ids"]) & set(fold["evaluation_run_ids"])
            )
            self.assertEqual(fold["status"], "run_separated")

    def test_evaluation_magnitude_does_not_change_training_threshold(self):
        def row(magnitude):
            windows = {}
            for window in psa.WINDOWS_NS:
                outcomes = {}
                for name in (
                    *psa.COUNTER_OUTCOMES,
                    "refault_sum_burst",
                    *psa.GAUGE_OUTCOMES,
                ):
                    outcomes[name] = {
                        "eligible": True,
                        "magnitude": magnitude,
                        "timeline": [(1, magnitude)],
                    }
                windows[window] = outcomes
            return {"windows": windows}

        raw = {"train": [row(2), row(4)], "eval": [row(1000)]}
        first = psa.training_outcome_thresholds(raw, ["train"])
        raw["eval"] = [row(1_000_000)]
        second = psa.training_outcome_thresholds(raw, ["train"])
        self.assertEqual(first, second)

    def test_decode_p95_uses_training_runs_only(self):
        train = psa.RunData(
            "train", Path("."), [], [psa.DecodeStep(1, 1, 11, 10)], [], 2, "8G", "2G"
        )
        evaluation = psa.RunData(
            "eval", Path("."), [], [psa.DecodeStep(1, 1, 1001, 1000)], [], 2, "8G", "2G"
        )
        rules = psa.decode_thresholds(
            {"train": train, "eval": evaluation}, ["train"]
        )
        self.assertEqual(
            rules["by_stratum"]["workers=2|memory_max=8G"]["value_ns"], 10
        )


class PressureShadowEpisodeTests(unittest.TestCase):
    def test_high_episode_merge_and_gap(self):
        states = [
            {"state": "HIGH", "sample_ready_ts_ns": 100},
            {"state": "HIGH", "sample_ready_ts_ns": 125},
            {"state": "MEDIUM", "sample_ready_ts_ns": 150},
            {"state": "HIGH", "sample_ready_ts_ns": 300},
        ]
        episodes = psa.merge_high_episodes(states, 50)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["sample_count"], 2)
        self.assertEqual(episodes[0]["duration_ns"], 25)

    def test_duration_does_not_bridge_large_gap(self):
        states = [
            {"state": "HIGH", "sample_ready_ts_ns": 100},
            {"state": "HIGH", "sample_ready_ts_ns": 125},
            {"state": "LOW", "sample_ready_ts_ns": 300},
        ]
        result = psa.state_durations(states, 50)
        self.assertEqual(result["duration_ns"]["HIGH"], 25)
        self.assertEqual(result["duration_ns"]["UNAVAILABLE"], 175)

    def test_outcome_before_high_is_not_prediction(self):
        states = [
            {"state": "HIGH", "sample_ready_ts_ns": 200},
            {"state": "LOW", "sample_ready_ts_ns": 225},
        ]
        labels = [{"positive": True, "onset_ts_ns": 150}, {"positive": False}]
        result = psa.episode_metrics(states, labels, [], 100, 50)
        self.assertEqual(result["hit_episode_count"], 0)
        self.assertIsNone(result["lead_time_ns"]["median"])
        self.assertEqual(result["evidence_status"], "insufficient")

    def test_lead_distribution_and_task_opportunity(self):
        states = [
            {"state": "HIGH", "sample_ready_ts_ns": 100},
            {"state": "LOW", "sample_ready_ts_ns": 125},
            {"state": "HIGH", "sample_ready_ts_ns": 200},
            {"state": "LOW", "sample_ready_ts_ns": 225},
        ]
        labels = [
            {"positive": True, "onset_ts_ns": 160},
            {"positive": False},
            {"positive": True, "onset_ts_ns": 260},
            {"positive": False},
        ]
        result = psa.episode_metrics(states, labels, [130, 230], 100, 50)
        self.assertEqual(result["hit_episode_count"], 2)
        self.assertEqual(result["lead_time_ns"]["median"], 60)
        self.assertEqual(result["lead_time_ns"]["p25"], 60)
        self.assertEqual(result["lead_time_ns"]["p95"], 60)
        self.assertEqual(result["episodes"][0]["task_opportunity_count"], 1)


class PressureShadowLegacyTests(unittest.TestCase):
    def test_old_expert_pressure_is_not_a_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "memory_trace.jsonl").write_text(
                json.dumps(
                    {
                        "event": "EXPERT_PRESSURE",
                        "ts_ns": 100,
                        "pressure_level": "critical",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            run = psa.load_run(run_dir)
            self.assertEqual(run.samples, [])

    def test_strict_contract_rejects_run_id_mismatch(self):
        pressure = sample(100)
        pressure.update(
            {
                "sample_seq": 1,
                "sample_start_ts_ns": 90,
                "ts_ns": 100,
                "sources": {
                    "cgroup_memory": {
                        "scope": "current_process_cgroup",
                        "path": "/sys/fs/cgroup/test",
                    }
                },
            }
        )
        pressure["phase"]["run_id"] = "fixture"
        run = psa.RunData(
            "fixture",
            Path("."),
            [pressure],
            [],
            [],
            2,
            "8G",
            "2G",
            {"run_name": "different"},
        )
        with self.assertRaises(ValueError):
            psa.validate_sample_contract(run)

    def test_two_run_end_to_end_analysis_is_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dirs = []
            for run_number, workers in enumerate((2, 4), 1):
                run_dir = Path(directory) / f"run_{run_number}"
                run_dir.mkdir()
                run_dirs.append(run_dir)
                manifest = {
                    "run_name": f"run_{run_number}",
                    "environment": {
                        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS": str(workers)
                    },
                    "experiment": {
                        "requested_memory_max": "8589934592",
                        "requested_memory_swap_max": "2147483648",
                        "cgroup": {
                            "memory.max": "8589934592",
                            "memory.swap.max": "2147483648",
                        },
                    },
                }
                (run_dir / "run_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                records = []
                for index in range(16):
                    ready = 1_000_000_000 + index * 25_000_000
                    read_ts = ready - 1_000
                    pressure = sample(
                        ready,
                        memory_current_bytes=obs(6_000_000_000 + index, read_ts),
                        memory_high_bytes=obs(8_000_000_000, read_ts),
                        memory_max_bytes=obs(8_589_934_592, read_ts),
                        swap_current_bytes=obs(index * 4096, read_ts),
                        memory_events_high=obs(index // 4, read_ts),
                        workingset_refault_anon=obs(index, read_ts),
                        workingset_refault_file=obs(index * 2, read_ts),
                        workingset_refault_anon_delta=obs(
                            1 if index else None,
                            read_ts,
                            "available" if index else "no_previous_sample",
                        ),
                        workingset_refault_file_delta=obs(
                            2 if index else None,
                            read_ts,
                            "available" if index else "no_previous_sample",
                        ),
                        cgroup_pgfault=obs(index * 10, read_ts),
                        cgroup_pgmajfault=obs(index // 3, read_ts),
                        pgscan=obs(index * 2, read_ts),
                        pgsteal=obs(index, read_ts),
                        pgscan_delta=obs(
                            2 if index else None,
                            read_ts,
                            "available" if index else "no_previous_sample",
                        ),
                        pgsteal_delta=obs(
                            1 if index else None,
                            read_ts,
                            "available" if index else "no_previous_sample",
                        ),
                        psi_some_total_us=obs(index * 50, read_ts),
                        psi_full_total_us=obs(index * 10, read_ts),
                        psi_some_delta_us=obs(50 if index else None, read_ts,
                                              "available" if index else "not_sampled"),
                        psi_full_delta_us=obs(10 if index else None, read_ts,
                                              "available" if index else "not_sampled"),
                        process_major_faults=obs(index // 3, read_ts),
                        queue_depth=obs(index % 5, read_ts),
                        queued_bytes=obs((index % 5) * 4096, read_ts),
                        worker_count=obs(workers, read_ts),
                    )
                    pressure["run_id"] = f"run_{run_number}"
                    pressure["phase"]["run_id"] = f"run_{run_number}"
                    records.append(pressure)
                    records.append(
                        {
                            "event": "STEP_BEGIN",
                            "phase": "DECODE",
                            "step": index,
                            "ts_ns": ready + 2_000,
                        }
                    )
                    records.append(
                        {
                            "event": "STEP_END",
                            "phase": "DECODE",
                            "step": index,
                            "ts_ns": ready + 5_000,
                            "latency_ns": 3_000 + index,
                        }
                    )
                (run_dir / "memory_trace.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )

            result = psa.analyze(run_dirs)
            self.assertEqual(result["run_count"], 2)
            self.assertEqual(len(result["folds"]), 2)
            self.assertEqual(len(result["labeled_samples"]), 32)
            json.dumps(result)


if __name__ == "__main__":
    unittest.main()
