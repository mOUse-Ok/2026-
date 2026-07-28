#!/usr/bin/env python3
"""Stream and aggregate M4A.1 Detail records across run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shadow_slack_analysis import analyze_shadow_slack


class RunRecordStream:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.run_id = str(manifest.get("run_name") or run_dir.name)

    def __iter__(self):
        memory_path = self.run_dir / "memory_trace.jsonl"
        with memory_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if (
                    '"event":"EXPERT_SHADOW_SLACK' not in line
                    and '"event":"OS_HINT"' not in line
                ):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(
                        f"{memory_path}:{line_number}: invalid JSON: {error}"
                    ) from error
                record.setdefault("run_id", self.run_id)
                yield record


class CombinedRecordStream:
    def __init__(self, streams: list[RunRecordStream]):
        self.streams = streams

    def __iter__(self):
        for stream in self.streams:
            yield from stream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    streams: list[RunRecordStream] = []
    run_ids: list[str] = []
    per_run: dict[str, dict] = {}
    for run_dir in args.run_dir:
        stream = RunRecordStream(run_dir)
        streams.append(stream)
        run_ids.append(stream.run_id)
        cached = run_dir / "analysis" / "shadow_slack_calibration.json"
        per_run[stream.run_id] = (
            json.loads(cached.read_text(encoding="utf-8"))
            if cached.exists()
            else analyze_shadow_slack(stream)
        )

    output = {
        "schema_version": 2,
        "run_ids": run_ids,
        "combined": analyze_shadow_slack(CombinedRecordStream(streams)),
        "per_run": per_run,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
