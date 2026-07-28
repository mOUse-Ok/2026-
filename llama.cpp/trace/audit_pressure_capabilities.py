#!/usr/bin/env python3
"""Audit cgroup/PSI sources required by M5A Pressure Shadow.

This utility does not run model inference. It reads the current process
cgroup and /proc sources, benchmarks their read/parse cost, and optionally
probes a transient systemd user scope whose lifetime ends with the probe.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


STATUS_AVAILABLE = "available"
STATUS_UNAVAILABLE = "unavailable"
STATUS_PERMISSION_DENIED = "permission_denied"
STATUS_FIELD_MISSING = "field_missing"
STATUS_PARSE_ERROR = "parse_error"
STATUS_IO_ERROR = "io_error"


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cost_summary(values: list[int]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def error_status(error: OSError) -> str:
    if error.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
        return STATUS_PERMISSION_DENIED
    if error.errno in (errno.ENOENT, errno.ENOTDIR):
        return STATUS_FIELD_MISSING
    return STATUS_IO_ERROR


def read_text(path: Path) -> tuple[str | None, str, int | None, str | None]:
    try:
        return path.read_text(encoding="utf-8").strip(), STATUS_AVAILABLE, None, None
    except OSError as error:
        return None, error_status(error), error.errno, error.strerror


def parse_scalar(text: str, allow_max: bool = False) -> dict[str, Any]:
    value = text.strip()
    if allow_max and value == "max":
        return {"value": None, "raw_value": "max", "unlimited": True}
    if not value or not value.isdecimal():
        raise ValueError("expected an unsigned decimal integer" + (" or max" if allow_max else ""))
    parsed = int(value)
    if parsed < 0 or parsed > (1 << 64) - 1:
        raise ValueError("unsigned integer is out of uint64 range")
    return {"value": parsed, "raw_value": value, "unlimited": False}


def parse_kv(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdecimal():
            raise ValueError(f"line {line_number}: expected '<field> <uint64>'")
        if parts[0] in result:
            raise ValueError(f"line {line_number}: duplicate field {parts[0]}")
        result[parts[0]] = int(parts[1])
    if not result:
        raise ValueError("no key/value fields")
    return result


def parse_psi(text: str) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) < 2 or parts[0] not in {"some", "full"}:
            raise ValueError(f"line {line_number}: invalid PSI category")
        fields: dict[str, float | int] = {}
        for token in parts[1:]:
            if "=" not in token:
                raise ValueError(f"line {line_number}: invalid PSI token {token}")
            key, raw = token.split("=", 1)
            if key == "total":
                if not raw.isdecimal():
                    raise ValueError(f"line {line_number}: invalid PSI total")
                fields[key] = int(raw)
            else:
                parsed = float(raw)
                if not math.isfinite(parsed):
                    raise ValueError(f"line {line_number}: non-finite PSI average")
                fields[key] = parsed
        if "total" not in fields:
            raise ValueError(f"line {line_number}: PSI total is missing")
        result[parts[0]] = fields
    if "some" not in result:
        raise ValueError("PSI some line is missing")
    return result


def parse_proc_stat(text: str) -> dict[str, int]:
    close = text.rfind(")")
    if close < 0:
        raise ValueError("/proc stat comm terminator is missing")
    fields = text[close + 2 :].split()
    # fields[0] is field 3 (state).
    if len(fields) < 22:
        raise ValueError("/proc stat has too few fields")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return {
        "minor_faults": int(fields[7]),
        "major_faults": int(fields[9]),
        "vms_bytes": int(fields[20]),
        "rss_bytes": int(fields[21]) * page_size,
    }


def parse_smaps_rollup(text: str) -> dict[str, int]:
    wanted = {
        "Rss": "rss_bytes",
        "Pss": "pss_bytes",
        "Swap": "swap_bytes",
        "Anonymous": "anonymous_bytes",
        "Private_Dirty": "private_dirty_bytes",
    }
    result: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        if key not in wanted:
            continue
        parts = rest.split()
        if not parts or not parts[0].isdecimal():
            raise ValueError(f"invalid smaps value for {key}")
        result[wanted[key]] = int(parts[0]) * 1024
    if "rss_bytes" not in result or "pss_bytes" not in result:
        raise ValueError("smaps_rollup is missing RSS or PSS")
    return result


@dataclass(frozen=True)
class MountInfo:
    mount_point: Path
    mount_options: list[str]
    super_options: list[str]


def find_cgroup2_mount() -> MountInfo | None:
    text, status, _, _ = read_text(Path("/proc/self/mountinfo"))
    if status != STATUS_AVAILABLE or text is None:
        return None
    for line in text.splitlines():
        if " - cgroup2 " not in line:
            continue
        before, after = line.split(" - ", 1)
        left = before.split()
        right = after.split()
        if len(left) < 6 or len(right) < 3:
            continue
        return MountInfo(
            mount_point=Path(left[4]),
            mount_options=left[5].split(","),
            super_options=right[2].split(","),
        )
    return None


def current_cgroup_relative() -> tuple[str | None, str]:
    text, status, _, _ = read_text(Path("/proc/self/cgroup"))
    if status != STATUS_AVAILABLE or text is None:
        return None, status
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            return parts[2] or "/", STATUS_AVAILABLE
    return None, STATUS_PARSE_ERROR


def audit_field(
    path: Path,
    parser: Callable[[str], Any],
    samples: int,
    *,
    source_scope: str,
    field_name: str,
) -> dict[str, Any]:
    costs: list[int] = []
    last_text: str | None = None
    last_parsed: Any = None
    status = STATUS_UNAVAILABLE
    error_number: int | None = None
    error_name: str | None = None
    error_message: str | None = None
    parse_status = "not_attempted"

    for _ in range(samples):
        started = time.monotonic_ns()
        text, read_status, current_errno, current_error = read_text(path)
        parsed: Any = None
        current_status = read_status
        current_parse_status = "not_attempted"
        if read_status == STATUS_AVAILABLE and text is not None:
            try:
                parsed = parser(text)
                current_parse_status = "ok"
            except (ValueError, OverflowError) as error:
                current_status = STATUS_PARSE_ERROR
                current_parse_status = "error"
                current_error = str(error)
        costs.append(time.monotonic_ns() - started)
        status = current_status
        error_number = current_errno
        error_name = errno.errorcode.get(current_errno) if current_errno is not None else None
        error_message = current_error
        parse_status = current_parse_status
        last_text = text
        last_parsed = parsed

    return {
        "source_scope": source_scope,
        "source_path": str(path),
        "field_name": field_name,
        "status": status,
        "parse_status": parse_status,
        "errno": error_number,
        "error_name": error_name,
        "error": error_message,
        "raw_format": "newline-delimited text",
        "raw_size_bytes": len(last_text.encode("utf-8")) if last_text is not None else None,
        "value": last_parsed,
        "sampling_cost_ns": cost_summary(costs),
        "audit_ts_ns": time.monotonic_ns(),
    }


def command_result(command: list[str], timeout: int = 10) -> dict[str, Any]:
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "duration_ns": time.monotonic_ns() - started,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
            "duration_ns": time.monotonic_ns() - started,
        }


def observe_scope_cleanup(source_path: str | None, timeout_s: float = 1.0) -> dict[str, Any]:
    if source_path is None:
        return {
            "status": "unknown",
            "path_exists": None,
            "populated": None,
            "process_count": None,
        }

    path = Path(source_path)
    deadline = time.monotonic() + timeout_s
    while path.exists() and time.monotonic() < deadline:
        events_text, events_status, _, _ = read_text(path / "cgroup.events")
        procs_text, procs_status, _, _ = read_text(path / "cgroup.procs")
        events = parse_kv(events_text) if events_status == STATUS_AVAILABLE and events_text else {}
        process_count = (
            len(procs_text.splitlines())
            if procs_status == STATUS_AVAILABLE and procs_text
            else 0 if procs_status == STATUS_AVAILABLE else None
        )
        if events.get("populated") == 0 and process_count == 0:
            return {
                "status": "inactive_empty",
                "path_exists": True,
                "populated": 0,
                "process_count": 0,
            }
        time.sleep(0.025)

    if not path.exists():
        return {
            "status": "removed",
            "path_exists": False,
            "populated": 0,
            "process_count": 0,
        }

    events_text, events_status, _, _ = read_text(path / "cgroup.events")
    procs_text, procs_status, _, _ = read_text(path / "cgroup.procs")
    events = parse_kv(events_text) if events_status == STATUS_AVAILABLE and events_text else {}
    process_count = (
        len(procs_text.splitlines())
        if procs_status == STATUS_AVAILABLE and procs_text
        else 0 if procs_status == STATUS_AVAILABLE else None
    )
    return {
        "status": (
            "inactive_empty"
            if events.get("populated") == 0 and process_count == 0
            else "still_populated"
        ),
        "path_exists": True,
        "populated": events.get("populated"),
        "process_count": process_count,
    }


def probe_systemd_user_scope() -> dict[str, Any]:
    systemctl = command_result(["systemctl", "--user", "is-system-running"])
    probe_script = (
        'cg_rel=$(cut -d: -f3 /proc/self/cgroup); '
        'cg_dir=/sys/fs/cgroup${cg_rel}; '
        'printf "relative=%s\\n" "$cg_rel"; '
        'printf "memory.max="; cat "$cg_dir/memory.max"; '
        'printf "memory.swap.max="; cat "$cg_dir/memory.swap.max"; '
        'printf "memory.current="; cat "$cg_dir/memory.current"; '
        'printf "pressure.lines="; wc -l < "$cg_dir/memory.pressure"'
    )
    scope = command_result(
        [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "-p",
            "MemoryMax=67108864",
            "-p",
            "MemorySwapMax=67108864",
            "bash",
            "-c",
            probe_script,
        ]
    )
    values: dict[str, str] = {}
    for line in scope.get("stdout", "").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    relative = values.get("relative")
    source_path = str(Path("/sys/fs/cgroup") / relative.lstrip("/")) if relative else None
    cleanup = observe_scope_cleanup(source_path)
    ok = (
        systemctl["returncode"] == 0
        and scope["returncode"] == 0
        and values.get("memory.max") == "67108864"
        and values.get("memory.swap.max") == "67108864"
        and relative not in (None, "", "/")
    )
    return {
        "status": STATUS_AVAILABLE if ok else STATUS_UNAVAILABLE,
        "systemctl": systemctl,
        "scope_probe": scope,
        "scope_values": values,
        "source_path": source_path,
        "child_entered_scope": relative not in (None, "", "/"),
        "limits_applied": (
            values.get("memory.max") == "67108864"
            and values.get("memory.swap.max") == "67108864"
        ),
        "cleanup_status": cleanup["status"],
        "cleanup": cleanup,
    }


def audit_runner(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    text, status, number, message = read_text(path)
    return {
        "path": str(path),
        "status": status,
        "exists": path.is_file(),
        "executable": os.access(path, os.X_OK),
        "uses_systemd_user_scope": bool(text and "systemd-run --user --scope" in text),
        "uses_direct_cgroup_creation": bool(text and "mkdir -p \"$cgdir\"" in text),
        "errno": number,
        "error": message,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# M5A Pressure Capability Audit",
        "",
        "> Capability audit only. No model inference was executed.",
        "",
        "## Decision",
        "",
        f"- Audit ID: `{audit['audit_id']}`",
        f"- Current cgroup: `{audit['cgroup']['current_source_path']}`",
        f"- Direct child creation: `{audit['delegation']['direct']['status']}`",
        f"- systemd user scope: `{audit['delegation']['systemd_user_scope']['status']}`",
        f"- Selected runner: `{audit['decision']['selected_runner']}`",
        f"- Delegated cgroup available: `{str(audit['decision']['delegated_cgroup_available']).lower()}`",
        f"- Formal restricted-memory evidence allowed: `{str(audit['decision']['formal_evidence_allowed']).lower()}`",
        "",
        "## Sources",
        "",
        "| Field | Status | Source | Value summary | Mean cost ns | p95 cost ns |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for field in audit["fields"]:
        value = field.get("value")
        if isinstance(value, dict):
            if "raw_value" in value:
                summary = value["raw_value"]
            else:
                summary = ",".join(sorted(value)[:8])
        else:
            summary = str(value)
        cost = field["sampling_cost_ns"]
        lines.append(
            f"| `{field['field_name']}` | `{field['status']}` | "
            f"`{field['source_path']}` | `{summary[:120]}` | "
            f"{cost['mean'] if cost['mean'] is not None else '-'} | "
            f"{cost['p95'] if cost['p95'] is not None else '-'} |"
        )
    lines.extend(
        [
            "",
            "## cgroup and delegation",
            "",
            f"- cgroup v2 mount: `{audit['cgroup']['mount_point']}`",
            f"- mount options: `{','.join(audit['cgroup']['mount_options'])}`",
            f"- controllers: `{audit['cgroup']['controllers']}`",
            f"- subtree control: `{audit['cgroup']['subtree_control']}`",
            f"- direct path writable: `{str(audit['delegation']['direct']['writable']).lower()}`",
            f"- user bus probe return code: `{audit['delegation']['systemd_user_scope']['systemctl']['returncode']}`",
            f"- transient scope limits applied: `{str(audit['delegation']['systemd_user_scope']['limits_applied']).lower()}`",
            f"- transient scope cleanup: `{audit['delegation']['systemd_user_scope']['cleanup_status']}`",
            "",
            "## Limitations",
            "",
            "- The current `/init.scope` is readable but is not a delegated writable parent for this user.",
            "- Formal runs must launch the tested process inside the verified transient user scope.",
            "- Host-global PSI is not used as a fallback for current-cgroup metrics.",
            "- Sampling costs describe this audit process and must be re-measured by the runtime sampler.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--skip-systemd-probe", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0 or args.samples > 1000:
        raise SystemExit("--samples must be in 1..1000")

    mount = find_cgroup2_mount()
    relative, cgroup_status = current_cgroup_relative()
    current_path = (
        mount.mount_point / relative.lstrip("/")
        if mount is not None and relative is not None
        else None
    )
    controllers_text, controllers_status, _, _ = (
        read_text(current_path / "cgroup.controllers")
        if current_path is not None
        else (None, STATUS_UNAVAILABLE, None, None)
    )
    subtree_text, subtree_status, _, _ = (
        read_text(current_path / "cgroup.subtree_control")
        if current_path is not None
        else (None, STATUS_UNAVAILABLE, None, None)
    )

    field_specs: list[tuple[str, str, Callable[[str], Any]]] = [
        ("memory.current", "memory.current", lambda text: parse_scalar(text)),
        ("memory.high", "memory.high", lambda text: parse_scalar(text, allow_max=True)),
        ("memory.max", "memory.max", lambda text: parse_scalar(text, allow_max=True)),
        ("memory.swap.current", "memory.swap.current", lambda text: parse_scalar(text)),
        ("memory.swap.max", "memory.swap.max", lambda text: parse_scalar(text, allow_max=True)),
        ("memory.events", "memory.events", parse_kv),
        ("memory.events.local", "memory.events.local", parse_kv),
        ("memory.stat", "memory.stat", parse_kv),
        ("memory.pressure", "memory.pressure", parse_psi),
    ]
    fields: list[dict[str, Any]] = []
    if current_path is not None:
        for filename, field_name, field_parser in field_specs:
            fields.append(
                audit_field(
                    current_path / filename,
                    field_parser,
                    args.samples,
                    source_scope="current_process_cgroup",
                    field_name=field_name,
                )
            )
    fields.append(
        audit_field(
            Path("/proc/self/stat"),
            parse_proc_stat,
            args.samples,
            source_scope="current_process",
            field_name="proc.self.stat",
        )
    )
    fields.append(
        audit_field(
            Path("/proc/self/smaps_rollup"),
            parse_smaps_rollup,
            max(1, min(args.samples, 4)),
            source_scope="current_process",
            field_name="proc.self.smaps_rollup",
        )
    )

    direct_writable = bool(
        current_path is not None
        and current_path.is_dir()
        and os.access(current_path, os.W_OK)
        and mount is not None
        and "ro" not in mount.mount_options
    )
    direct_status = STATUS_AVAILABLE if direct_writable else STATUS_PERMISSION_DENIED
    systemd_probe = (
        {
            "status": STATUS_UNAVAILABLE,
            "reason": "skipped_by_request",
            "systemctl": {"returncode": None},
            "limits_applied": False,
            "cleanup_status": "not_attempted",
        }
        if args.skip_systemd_probe
        else probe_systemd_user_scope()
    )

    repo = args.repo.resolve()
    runners = [
        audit_runner(repo, "llama.cpp/trace/run_cgroup_memory_matrix.sh"),
        audit_runner(repo, "llama.cpp/trace/run_finalist_repeat_matrix.sh"),
        audit_runner(repo, "llama.cpp/trace/run_stage_deadline_score_repeat_matrix.sh"),
    ]
    systemd_available = systemd_probe["status"] == STATUS_AVAILABLE
    selected_runner = (
        "systemd-run --user --scope via M5A dedicated runner"
        if systemd_available
        else
        "direct delegated cgroup parent"
        if direct_writable
        else
        "none"
    )
    audit_id = (
        "m5a_pressure_capability_audit_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_{os.getpid()}"
    )
    audit = {
        "schema_version": 1,
        "audit_id": audit_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_ts_ns": time.monotonic_ns(),
        "no_model_inference": True,
        "host": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "uid": os.getuid(),
            "gid": os.getgid(),
        },
        "cgroup": {
            "status": cgroup_status,
            "version": 2 if mount is not None else None,
            "mount_point": str(mount.mount_point) if mount is not None else None,
            "mount_options": mount.mount_options if mount is not None else [],
            "super_options": mount.super_options if mount is not None else [],
            "relative_path": relative,
            "current_source_path": str(current_path) if current_path is not None else None,
            "controllers_status": controllers_status,
            "controllers": controllers_text,
            "subtree_control_status": subtree_status,
            "subtree_control": subtree_text,
            "swap_controller_available": any(
                field["field_name"] == "memory.swap.current"
                and field["status"] == STATUS_AVAILABLE
                for field in fields
            ),
        },
        "delegation": {
            "direct": {
                "status": direct_status,
                "source_path": str(current_path) if current_path is not None else None,
                "writable": direct_writable,
                "probe_created": False,
                "reason": (
                    "current cgroup/mount is not writable by this user"
                    if not direct_writable
                    else
                    "writable delegated parent detected"
                ),
            },
            "systemd_user_scope": systemd_probe,
        },
        "fields": fields,
        "runners": runners,
        "decision": {
            "selected_runner": selected_runner,
            "delegated_cgroup_available": systemd_available or direct_writable,
            "formal_evidence_allowed": systemd_available or direct_writable,
            "host_global_fallback_allowed": False,
        },
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "M5A_pressure_capability_audit.json"
    markdown_path = output_dir / "M5A_pressure_capability_audit.md"
    commands_path = output_dir / "M5A_pressure_capability_audit_commands.txt"
    for path in (json_path, markdown_path, commands_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing audit artifact: {path}")
    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")
    commands = [
        f"python3 {Path(__file__).resolve()} --output-dir {output_dir} --samples {args.samples}",
        "systemctl --user is-system-running",
        (
            "systemd-run --user --scope --quiet "
            "-p MemoryMax=67108864 -p MemorySwapMax=67108864 <probe>"
        ),
    ]
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit_id": audit_id,
        "json": str(json_path),
        "markdown": str(markdown_path),
        "delegated_cgroup_available": audit["decision"]["delegated_cgroup_available"],
        "selected_runner": selected_runner,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
