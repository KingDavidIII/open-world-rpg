"""Tests for application construction and execution."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import TextIO, cast

import pytest

from open_world_rpg.application.bootstrap import (
    create_application,
    run_application,
)
from open_world_rpg.application.runtime import ApplicationState
from open_world_rpg.application.session import GameMode, SessionState
from open_world_rpg.core import LOGGER_NAME, RuntimeEnvironment
from open_world_rpg.core.diagnostics import reset_runtime_logging


class BrokenOutput:
    """Output stream that simulates a write failure."""

    def write(self, value: str) -> int:
        del value
        raise OSError("output unavailable")

    def flush(self) -> None:
        return None


@pytest.fixture(autouse=True)
def clean_runtime_logger() -> None:
    reset_runtime_logging()
    yield
    reset_runtime_logging()


def test_create_application_uses_current_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    application = create_application()

    assert application.config.paths.project_root == tmp_path.resolve()
    assert application.config.environment is RuntimeEnvironment.DEVELOPMENT
    assert application.context.game_mode is GameMode.NEW_GAME
    assert application.context.world_seed == application.config.simulation.world_seed
    assert application.logger.name == LOGGER_NAME
    assert application.state is ApplicationState.CREATED
    assert application.context.state is SessionState.CREATED
    assert (tmp_path / "logs" / "open-world-rpg.log").is_file()


def test_create_application_accepts_explicit_configuration_and_logger(
    tmp_path: Path,
) -> None:
    logger = logging.Logger("injected")

    application = create_application(
        project_root=tmp_path,
        environment=RuntimeEnvironment.TEST,
        game_mode=GameMode.LOADED_GAME,
        logger=logger,
    )

    assert application.config.paths.project_root == tmp_path.resolve()
    assert application.config.environment is RuntimeEnvironment.TEST
    assert application.context.game_mode is GameMode.LOADED_GAME
    assert application.logger is logger


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
    assert application.context.state is SessionState.TERMINATED
    assert output.getvalue() == ("Open World RPG v0.8.0 - runtime initialised.\n")


def test_run_application_uses_standard_output_by_default(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    application = create_application(project_root=tmp_path)

    exit_code = run_application(application)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "Open World RPG v0.8.0 - runtime initialised.\n"


def test_run_application_marks_output_failure(
    tmp_path: Path,
) -> None:
    application = create_application(project_root=tmp_path)
    output = cast(TextIO, BrokenOutput())

    with pytest.raises(OSError, match="output unavailable"):
        run_application(application, output=output)

    assert application.state is ApplicationState.FAILED
    assert application.context.state is SessionState.FAILED
