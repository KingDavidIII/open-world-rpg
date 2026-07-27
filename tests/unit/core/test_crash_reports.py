"""Tests for atomic structured crash reports."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest

import open_world_rpg.core.crash_reports as crash_module
from open_world_rpg.core import (
    CRASH_REPORT_SCHEMA_VERSION,
    CrashReportWriteError,
    write_crash_report,
)

CREATED_AT = datetime(2026, 7, 27, 4, 30, 15, 123456, tzinfo=UTC)


def test_write_crash_report_persists_diagnostic_payload(tmp_path: Path) -> None:
    try:
        raise RuntimeError("renderer failed")
    except RuntimeError as error:
        path = write_crash_report(
            directory=tmp_path / "reports",
            error=error,
            application_version="0.9.0",
            command=("--smoke-test", "--world-seed", "42"),
            context={"world_seed": 42, "smoke_test": True, "save_path": None},
            created_at=CREATED_AT,
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.parent == (tmp_path / "reports").resolve()
    assert path.name.startswith("open-world-rpg-crash-20260727T043015.123456Z-")
    assert payload["schema_version"] == CRASH_REPORT_SCHEMA_VERSION
    assert payload["timestamp"] == CREATED_AT.isoformat()
    assert payload["application"] == "Open World RPG"
    assert payload["application_version"] == "0.9.0"
    assert payload["command"] == ["--smoke-test", "--world-seed", "42"]
    assert payload["context"] == {
        "save_path": None,
        "smoke_test": True,
        "world_seed": 42,
    }
    assert payload["exception"]["type"] == "RuntimeError"
    assert payload["exception"]["message"] == "renderer failed"
    assert "RuntimeError: renderer failed" in payload["exception"]["traceback"]
    assert payload["python_version"]
    assert payload["platform"]
    assert payload["executable"]


def test_write_crash_report_uses_current_utc_time_and_empty_defaults(tmp_path: Path) -> None:
    path = write_crash_report(
        directory=tmp_path,
        error=ValueError("bad"),
        application_version=" 1.0.0 ",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["application_version"] == "1.0.0"
    assert payload["command"] == []
    assert payload["context"] == {}
    assert datetime.fromisoformat(payload["timestamp"]).tzinfo is not None


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"directory": "reports"}, TypeError, "directory"),
        ({"error": object()}, TypeError, "error"),
        ({"application_version": 1}, TypeError, "application_version"),
        ({"application_version": "  "}, ValueError, "must not be empty"),
        ({"command": "--run"}, TypeError, "command"),
        ({"command": [1]}, TypeError, "command entries"),
        ({"context": []}, TypeError, "context"),
        ({"context": {1: "value"}}, TypeError, "context keys"),
        ({"context": {"bad": object()}}, TypeError, "context values"),
        ({"created_at": "now"}, TypeError, "created_at"),
        ({"created_at": datetime(2026, 7, 27)}, ValueError, "timezone-aware"),
    ],
)
def test_write_crash_report_rejects_invalid_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "directory": tmp_path,
        "error": RuntimeError("failure"),
        "application_version": "0.9.0",
        "created_at": CREATED_AT,
    }
    arguments.update(kwargs)

    with pytest.raises(error_type, match=message):
        write_crash_report(**cast(Any, arguments))


def test_write_failure_is_wrapped_and_temporary_file_removed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
    ) -> NoReturn:
        del source, destination
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(CrashReportWriteError, match="Could not write") as error:
        write_crash_report(
            directory=tmp_path,
            error=RuntimeError("failure"),
            application_version="0.9.0",
            created_at=CREATED_AT,
        )

    assert isinstance(error.value.__cause__, OSError)
    assert list(tmp_path.glob("*.tmp")) == []


def test_report_directory_or_temporary_file_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "blocked"
    report_path.write_text("file", encoding="utf-8")

    with pytest.raises(CrashReportWriteError) as directory_error:
        write_crash_report(
            directory=report_path,
            error=RuntimeError("failure"),
            application_version="0.9.0",
            created_at=CREATED_AT,
        )
    assert isinstance(directory_error.value.__cause__, OSError)

    def fail_temporary_file(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise OSError("temporary unavailable")

    monkeypatch.setattr(crash_module, "NamedTemporaryFile", fail_temporary_file)

    with pytest.raises(CrashReportWriteError) as temporary_error:
        write_crash_report(
            directory=tmp_path / "reports",
            error=RuntimeError("failure"),
            application_version="0.9.0",
            created_at=CREATED_AT,
        )
    assert isinstance(temporary_error.value.__cause__, OSError)


def test_cleanup_failure_does_not_hide_report_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
    ) -> NoReturn:
        del source, destination
        raise OSError("replace failed")

    def fail_unlink(self: Path, missing_ok: bool = False) -> NoReturn:
        del self, missing_ok
        raise OSError("cleanup failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(CrashReportWriteError) as error:
        write_crash_report(
            directory=tmp_path,
            error=RuntimeError("failure"),
            application_version="0.9.0",
            created_at=CREATED_AT,
        )

    assert str(error.value.__cause__) == "replace failed"
