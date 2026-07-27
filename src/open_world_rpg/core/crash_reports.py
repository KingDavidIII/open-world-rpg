"""Structured crash-report persistence for release diagnostics."""

from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, TypeAlias

CRASH_REPORT_SCHEMA_VERSION: Final = 1
CrashContextValue: TypeAlias = str | int | float | bool | None


class CrashReportError(RuntimeError):
    """Base exception for crash-report failures."""


class CrashReportWriteError(CrashReportError):
    """Raised when a crash report cannot be written atomically."""


def write_crash_report(
    *,
    directory: Path,
    error: BaseException,
    application_version: str,
    command: Sequence[str] = (),
    context: Mapping[str, CrashContextValue] | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Write one atomic JSON crash report and return its final path."""
    if not isinstance(directory, Path):
        raise TypeError("directory must be a pathlib.Path instance.")
    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception.")
    if not isinstance(application_version, str):
        raise TypeError("application_version must be a string.")
    version = application_version.strip()
    if not version:
        raise ValueError("application_version must not be empty.")
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TypeError("command must be a sequence of strings.")
    normalised_command: list[str] = []
    for argument in command:
        if not isinstance(argument, str):
            raise TypeError("command entries must be strings.")
        normalised_command.append(argument)
    if context is not None and not isinstance(context, Mapping):
        raise TypeError("context must be a mapping or None.")
    normalised_context: dict[str, CrashContextValue] = {}
    if context is not None:
        for key, value in context.items():
            if not isinstance(key, str):
                raise TypeError("context keys must be strings.")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError("context values must be JSON scalar values.")
            normalised_context[key] = value
    timestamp = datetime.now(tz=UTC) if created_at is None else created_at
    if not isinstance(timestamp, datetime):
        raise TypeError("created_at must be a datetime or None.")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware.")
    timestamp = timestamp.astimezone(UTC)

    report_directory = directory.expanduser().resolve(strict=False)
    file_stamp = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    destination = report_directory / f"open-world-rpg-crash-{file_stamp}-{os.getpid()}.json"
    payload = {
        "schema_version": CRASH_REPORT_SCHEMA_VERSION,
        "timestamp": timestamp.isoformat(),
        "application": "Open World RPG",
        "application_version": version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "command": normalised_command,
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.TracebackException.from_exception(error).format()),
        },
        "context": normalised_context,
    }

    temporary_path: Path | None = None
    try:
        report_directory.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=report_directory,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
        raise CrashReportWriteError("Could not write the crash report.") from exc

    return destination
