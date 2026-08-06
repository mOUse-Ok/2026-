"""Fail-closed JSONL artifact writing and post-close finalization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Generic, TextIO, TypeVar


T = TypeVar("T")


class ArtifactFinalizationError(RuntimeError):
    """The artifact did not reach a verifiable closed-file state."""


class DecisionDigestSink:
    """Hash deterministic replay output without representing it as an artifact."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.size_bytes = 0
        self.line_count = 0

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8")
        self._digest.update(encoded)
        self.size_bytes += len(encoded)
        self.line_count += encoded.count(b"\n")
        return len(value)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def inspect_closed_jsonl(
    path: Path,
    *,
    expected_line_count: int | None = None,
) -> dict[str, Any]:
    """Reopen a final path and independently measure and parse every JSONL line."""

    path = path.resolve()
    errors: list[dict[str, Any]] = []
    if not path.is_file():
        return {
            "path": str(path),
            "passed": False,
            "errors": [{"reason": "missing_final_artifact"}],
            "size_bytes": None,
            "bytes_read": 0,
            "line_count": 0,
            "expected_line_count": expected_line_count,
            "line_count_matches": False,
            "sha256": None,
            "jsonl_parseable": False,
            "final_line_complete": False,
        }

    digest = hashlib.sha256()
    bytes_read = 0
    line_count = 0
    jsonl_parseable = True
    final_line_complete = True
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            digest.update(raw_line)
            bytes_read += len(raw_line)
            line_count += 1
            if not raw_line.endswith(b"\n"):
                final_line_complete = False
                errors.append({"reason": "partial_final_line", "line": line_number})
            try:
                json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                jsonl_parseable = False
                errors.append({
                    "reason": "jsonl_parse_failure",
                    "line": line_number,
                    "detail": str(exc),
                })

    size_bytes = path.stat().st_size
    if bytes_read != size_bytes:
        errors.append({
            "reason": "size_changed_while_reading",
            "stat_size": size_bytes,
            "bytes_read": bytes_read,
        })
    line_count_matches = expected_line_count is None or line_count == expected_line_count
    if not line_count_matches:
        errors.append({
            "reason": "line_count_mismatch",
            "expected": expected_line_count,
            "actual": line_count,
        })
    if line_count == 0:
        final_line_complete = False
        errors.append({"reason": "empty_jsonl_artifact"})

    return {
        "path": str(path),
        "passed": not errors,
        "errors": errors,
        "size_bytes": size_bytes,
        "bytes_read": bytes_read,
        "line_count": line_count,
        "expected_line_count": expected_line_count,
        "line_count_matches": line_count_matches,
        "sha256": digest.hexdigest(),
        "jsonl_parseable": jsonl_parseable,
        "final_line_complete": final_line_complete,
    }


def write_finalized_jsonl(
    path: Path,
    write_callback: Callable[[TextIO], T],
    *,
    expected_line_count: int,
    durable_fsync: bool = False,
    opener: Callable[[Path], TextIO] | None = None,
    inspector: Callable[..., dict[str, Any]] = inspect_closed_jsonl,
) -> tuple[T, dict[str, Any]]:
    """Write, flush, close, then reopen and validate a final JSONL artifact."""

    if expected_line_count <= 0:
        raise ValueError("expected_line_count must be positive")
    path = path.resolve()
    open_stream = opener or (lambda final_path: final_path.open("x", encoding="utf-8"))
    stream = open_stream(path)
    result: T
    try:
        result = write_callback(stream)
        stream.flush()
        if durable_fsync:
            os.fsync(stream.fileno())
    except BaseException as exc:
        try:
            stream.close()
        except BaseException:
            pass
        raise ArtifactFinalizationError("artifact write or flush failed") from exc

    try:
        stream.close()
    except BaseException as exc:
        raise ArtifactFinalizationError("artifact close failed") from exc
    if not stream.closed:
        raise ArtifactFinalizationError("artifact writer did not reach closed state")

    metadata = inspector(path, expected_line_count=expected_line_count)
    if not metadata.get("passed"):
        raise ArtifactFinalizationError(
            f"closed artifact validation failed: {metadata.get('errors', [])}"
        )
    metadata.update({
        "finalized_after_close": True,
        "durable_fsync_requested": durable_fsync,
        "durability_claim": (
            "fsync requested before close; no claim beyond the local filesystem contract"
            if durable_fsync
            else "close-and-reopen integrity only; no all-storage-layer durability claim"
        ),
    })
    return result, metadata
