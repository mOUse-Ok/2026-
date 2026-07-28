import json
import sys
import tempfile
import unittest
from pathlib import Path

TRACE_DIR = Path(__file__).resolve().parents[1]
if str(TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(TRACE_DIR))

import summarize_pressure_shadow_results as validation


def observation(run_id, value, timestamp, status="available"):
    return {
        "run_id": run_id,
        "status": status,
        "read_ts_ns": timestamp,
        "error": None if status == "available" else status,
        "value": value if status == "available" else None,
    }


class PressureShadowValidationTests(unittest.TestCase):
    def make_run(self, root: Path) -> tuple[str, dict]:
        run_id = "validation_fixture"
        run_dir = root / run_id
        (run_dir / "analysis").mkdir(parents=True)
        cgroup_path = "/sys/fs/cgroup/user.slice/fixture.scope"
        records = []
        for index in range(3):
            ready = 100_000_000 + index * 25_000_000
            sample = {
                "event": "PRESSURE_SHADOW_SAMPLE",
                "schema_version": 1,
                "run_id": run_id,
                "sample_seq": index + 1,
                "sample_start_ts_ns": ready - 1_000_000,
                "sample_ready_ts_ns": ready,
                "ts_ns": ready,
                "target_interval_ns": 25_000_000,
                "actual_interval_ns": None if index == 0 else 25_000_000,
                "deadline_lateness_ns": 1000,
                "missed_samples_since_previous": 0,
                "sample_wall_time_ns": 1_000_000,
                "sample_thread_cpu_time_ns": 100_000,
                "sources": {
                    "cgroup_memory": {
                        "scope": "current_process_cgroup",
                        "path": cgroup_path,
                        "status": "available",
                        "errno": None,
                        "error": None,
                    }
                },
            }
            for field in validation.REQUIRED_SAMPLE_FIELDS:
                status = (
                    "no_previous_sample"
                    if index == 0 and field.endswith("_delta")
                    else "available"
                )
                value = (
                    "DECODE" if field == "phase"
                    else "available" if field == "queue_status"
                    else 1
                )
                sample[field] = observation(
                    run_id, value, ready - 1000, status
                )
            records.append(sample)
        records.append(
            {
                "event": "PRESSURE_SHADOW_SUMMARY",
                "schema_version": 1,
                "run_id": run_id,
                "sample_interval_ms": 25,
                "started_ts_ns": 90_000_000,
                "stopped_ts_ns": 200_000_000,
                "sample_count": 3,
                "detail_events": 3,
                "missed_intervals": 0,
                "cgroup_path": cgroup_path,
                "sampler_cpu_cost": {"total_ns": 1_000_000},
                "pss_sampler": {"cpu_total_ns": 1_000_000},
            }
        )
        (run_dir / "memory_trace.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        manifest = {
            "run_name": run_id,
            "git_dirty": False,
            "environment": {
                "LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER": "off",
                "LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK": "0",
                "LLM_MEM_TRACE_OPT_EXPERT_SLACK": "0",
                "LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE": "0",
                "LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT": "0",
            },
            "experiment": {
                "cgroup": {
                    "source_path": cgroup_path,
                    "memory.max": "8589934592",
                    "memory.swap.max": "2147483648",
                }
            },
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "sinks": {
                        "memory": {
                            "enabled": True,
                            "enqueued": 4,
                            "written": 4,
                            "dropped": 0,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "analysis" / "metrics.json").write_text(
            json.dumps(
                {
                    "expert_task_rejected": 0,
                    "expert_task_cancelled": 0,
                    "expert_task_invalid_transitions": 0,
                }
            ),
            encoding="utf-8",
        )
        return run_id, {
            "run_dir": str(run_dir),
            "workers": 2,
            "memory_max": "8589934592",
            "memory_swap_max": "2147483648",
        }

    def test_valid_detail_contract_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            run_id, run_info = self.make_run(Path(directory))
            result = validation.validate_detail_run(run_id, run_info)
            self.assertTrue(result["passed"], result["checks"])
            self.assertEqual(result["sequence_errors"], 0)
            self.assertEqual(result["source_errors"], 0)

    def test_wilson_empty_is_explicitly_unavailable(self):
        self.assertEqual(
            validation.wilson(0, 0), {"low": None, "high": None}
        )


if __name__ == "__main__":
    unittest.main()
