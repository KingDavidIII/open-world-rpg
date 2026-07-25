"""Tests for application runtime lifecycle management."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from open_world_rpg.application.runtime import (
    ApplicationLifecycleError,
    ApplicationState,
    GameApplication,
)
from open_world_rpg.application.session import (
    GameMode,
    RuntimeContext,
    SessionState,
    SessionTransitionError,
)
from open_world_rpg.core import GameConfig, RuntimeEnvironment, SimulationConfig

FIXED_TIME = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
SESSION_ID = UUID("12345678-1234-5678-1234-567812345678")


def create_test_application(
    tmp_path: Path,
    *,
    world_seed: int = 0,
) -> GameApplication:
    config = GameConfig(
        environment=RuntimeEnvironment.TEST,
        simulation=SimulationConfig(world_seed=world_seed),
        paths=GameConfig.create_default(
            project_root=tmp_path,
            environment=RuntimeEnvironment.TEST,
        ).paths,
    )
    context = RuntimeContext.create(
        game_mode=GameMode.NEW_GAME,
        world_seed=world_seed,
        clock=lambda: FIXED_TIME,
        session_id=SESSION_ID,
    )

    return GameApplication(
        config=config,
        context=context,
    )


def test_application_starts_in_created_state(tmp_path: Path) -> None:
    application = create_test_application(tmp_path)

    assert application.state is ApplicationState.CREATED
    assert application.context.state is SessionState.CREATED
    assert application.is_running is False


def test_start_creates_directories_and_activates_session(
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)

    application.start()

    assert application.state is ApplicationState.RUNNING
    assert application.context.state is SessionState.ACTIVE
    assert application.is_running is True
    assert application.config.paths.save_directory.is_dir()
    assert application.config.paths.log_directory.is_dir()


def test_application_can_pause_and_resume_gameplay(tmp_path: Path) -> None:
    application = create_test_application(tmp_path)
    application.start()

    application.pause()

    assert application.state is ApplicationState.RUNNING
    assert application.context.state is SessionState.PAUSED

    application.resume()

    assert application.state is ApplicationState.RUNNING
    assert application.context.state is SessionState.ACTIVE


def test_application_stop_terminates_session(tmp_path: Path) -> None:
    application = create_test_application(tmp_path)
    application.start()

    application.stop()

    assert application.state is ApplicationState.STOPPED
    assert application.context.state is SessionState.TERMINATED
    assert application.is_running is False


def test_paused_application_can_be_stopped(tmp_path: Path) -> None:
    application = create_test_application(tmp_path)
    application.start()
    application.pause()

    application.stop()

    assert application.state is ApplicationState.STOPPED
    assert application.context.state is SessionState.TERMINATED


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


@pytest.mark.parametrize("operation", ["pause", "resume", "stop"])
def test_unstarted_application_rejects_runtime_operations(
    operation: str,
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)

    with pytest.raises(
        ApplicationLifecycleError,
        match=f"Cannot {operation} application",
    ):
        getattr(application, operation)()


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
    assert application.context.state is SessionState.CREATED


def test_stop_failure_marks_application_failed(tmp_path: Path) -> None:
    application = create_test_application(tmp_path)
    application.start()
    application.context.fail()

    with pytest.raises(
        SessionTransitionError,
        match="Cannot terminate session",
    ):
        application.stop()

    assert application.state is ApplicationState.FAILED


def test_application_can_be_marked_failed(tmp_path: Path) -> None:
    application = create_test_application(tmp_path)
    application.start()

    application.fail()

    assert application.state is ApplicationState.FAILED
    assert application.context.state is SessionState.FAILED


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


def test_application_rejects_invalid_constructor_values(
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)

    with pytest.raises(TypeError, match="config"):
        GameApplication(
            config=cast(Any, object()),
            context=application.context,
        )

    with pytest.raises(TypeError, match="context"):
        GameApplication(
            config=application.config,
            context=cast(Any, object()),
        )


def test_application_rejects_context_seed_mismatch(
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path, world_seed=10)
    mismatched_context = RuntimeContext.create(
        game_mode=GameMode.NEW_GAME,
        world_seed=20,
        clock=lambda: FIXED_TIME,
        session_id=SESSION_ID,
    )

    with pytest.raises(ValueError, match="seed must match"):
        GameApplication(
            config=application.config,
            context=mismatched_context,
        )
