"""Tests for application runtime lifecycle management."""

from __future__ import annotations

from pathlib import Path

import pytest

from open_world_rpg.application.runtime import (
    ApplicationLifecycleError,
    ApplicationState,
    GameApplication,
)
from open_world_rpg.core import GameConfig, RuntimeEnvironment


def create_test_application(tmp_path: Path) -> GameApplication:
    return GameApplication(
        config=GameConfig.create_default(
            project_root=tmp_path,
            environment=RuntimeEnvironment.TEST,
        )
    )


def test_application_starts_in_created_state(tmp_path: Path) -> None:
    application = create_test_application(tmp_path)

    assert application.state is ApplicationState.CREATED
    assert application.is_running is False


def test_start_creates_runtime_directories(tmp_path: Path) -> None:
    application = create_test_application(tmp_path)

    application.start()

    assert application.state is ApplicationState.RUNNING
    assert application.is_running is True
    assert application.config.paths.save_directory.is_dir()
    assert application.config.paths.log_directory.is_dir()


def test_application_can_be_stopped(tmp_path: Path) -> None:
    application = create_test_application(tmp_path)
    application.start()

    application.stop()

    assert application.state is ApplicationState.STOPPED
    assert application.is_running is False


def test_stopping_an_already_stopped_application_is_harmless(
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)
    application.start()
    application.stop()

    application.stop()

    assert application.state is ApplicationState.STOPPED


def test_start_rejects_duplicate_start(tmp_path: Path) -> None:
    application = create_test_application(tmp_path)
    application.start()

    with pytest.raises(
        ApplicationLifecycleError,
        match="Cannot start application",
    ):
        application.start()


def test_stop_rejects_application_that_has_not_started(
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)

    with pytest.raises(
        ApplicationLifecycleError,
        match="Cannot stop application",
    ):
        application.stop()


def test_start_failure_marks_application_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)

    def raise_directory_error(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        del self, mode, parents, exist_ok
        raise OSError("runtime storage unavailable")

    monkeypatch.setattr(Path, "mkdir", raise_directory_error)

    with pytest.raises(OSError, match="runtime storage unavailable"):
        application.start()

    assert application.state is ApplicationState.FAILED
    assert application.is_running is False


def test_application_can_be_marked_failed(tmp_path: Path) -> None:
    application = create_test_application(tmp_path)

    application.fail()

    assert application.state is ApplicationState.FAILED


def test_stopped_application_cannot_be_marked_failed(
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)
    application.start()
    application.stop()

    with pytest.raises(
        ApplicationLifecycleError,
        match="cannot be marked as failed",
    ):
        application.fail()
