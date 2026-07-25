"""Tests for application construction and execution."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TextIO, cast

import pytest

from open_world_rpg.application.bootstrap import (
    create_application,
    run_application,
)
from open_world_rpg.application.runtime import ApplicationState
from open_world_rpg.core import RuntimeEnvironment


class BrokenOutput:
    """Output stream that simulates a write failure."""

    def write(self, value: str) -> int:
        del value
        raise OSError("output unavailable")

    def flush(self) -> None:
        return None


def test_create_application_uses_current_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    application = create_application()

    assert application.config.paths.project_root == tmp_path.resolve()
    assert application.config.environment is RuntimeEnvironment.DEVELOPMENT
    assert application.state is ApplicationState.CREATED


def test_create_application_accepts_explicit_configuration(
    tmp_path: Path,
) -> None:
    application = create_application(
        project_root=tmp_path,
        environment=RuntimeEnvironment.TEST,
    )

    assert application.config.paths.project_root == tmp_path.resolve()
    assert application.config.environment is RuntimeEnvironment.TEST


def test_run_application_returns_success_and_stops(
    tmp_path: Path,
) -> None:
    application = create_application(
        project_root=tmp_path,
        environment=RuntimeEnvironment.TEST,
    )
    output = StringIO()

    exit_code = run_application(application, output=output)

    assert exit_code == 0
    assert application.state is ApplicationState.STOPPED
    assert output.getvalue() == ("Open World RPG v0.1.0 - runtime initialised.\n")


def test_run_application_uses_standard_output_by_default(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    application = create_application(project_root=tmp_path)

    exit_code = run_application(application)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "Open World RPG v0.1.0 - runtime initialised.\n"


def test_run_application_marks_output_failure(
    tmp_path: Path,
) -> None:
    application = create_application(project_root=tmp_path)
    output = cast(TextIO, BrokenOutput())

    with pytest.raises(OSError, match="output unavailable"):
        run_application(application, output=output)

    assert application.state is ApplicationState.FAILED
