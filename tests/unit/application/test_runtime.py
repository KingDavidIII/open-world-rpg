"""Tests for application runtime lifecycle management."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from io import StringIO
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
from open_world_rpg.core import (
    GameConfig,
    JsonLogFormatter,
    RuntimeEnvironment,
    SimulationConfig,
)

FIXED_TIME = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
SESSION_ID = UUID("12345678-1234-5678-1234-567812345678")


def create_test_logger(stream: StringIO | None = None) -> logging.Logger:
    logger = logging.Logger(
        "test.open_world_rpg",
        level=logging.DEBUG,
    )
    logger.propagate = False

    if stream is not None:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)

    return logger


def create_test_application(
    tmp_path: Path,
    *,
    world_seed: int = 0,
    logger: logging.Logger | None = None,
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
        logger=create_test_logger() if logger is None else logger,
    )


def read_log_payloads(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


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


def test_lifecycle_emits_structured_diagnostic_events(
    tmp_path: Path,
) -> None:
    stream = StringIO()
    application = create_test_application(
        tmp_path,
        logger=create_test_logger(stream),
    )

    application.start()
    application.pause()
    application.resume()
    application.stop()

    payloads = read_log_payloads(stream)

    assert [payload["event"] for payload in payloads] == [
        "application.starting",
        "session.activated",
        "application.running",
        "session.paused",
        "session.resumed",
        "application.stopping",
        "session.terminated",
        "application.stopped",
    ]

    for payload in payloads:
        assert payload["session_id"] == str(SESSION_ID)
        assert payload["world_seed"] == 0
        assert "application_state" in payload
        assert "session_state" in payload


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


def test_start_failure_marks_application_failed_and_logs_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stream = StringIO()
    application = create_test_application(
        tmp_path,
        logger=create_test_logger(stream),
    )

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

    payloads = read_log_payloads(stream)

    assert application.state is ApplicationState.FAILED
    assert application.context.state is SessionState.CREATED
    assert payloads[-1]["event"] == "application.start_failed"
    assert "OSError: runtime storage unavailable" in payloads[-1]["exception"]


def test_pause_failure_is_logged(tmp_path: Path) -> None:
    stream = StringIO()
    application = create_test_application(
        tmp_path,
        logger=create_test_logger(stream),
    )
    application.start()
    application.pause()

    with pytest.raises(SessionTransitionError, match="Cannot pause session"):
        application.pause()

    assert read_log_payloads(stream)[-1]["event"] == "session.pause_failed"


def test_resume_failure_is_logged(tmp_path: Path) -> None:
    stream = StringIO()
    application = create_test_application(
        tmp_path,
        logger=create_test_logger(stream),
    )
    application.start()

    with pytest.raises(SessionTransitionError, match="Cannot resume session"):
        application.resume()

    assert read_log_payloads(stream)[-1]["event"] == "session.resume_failed"


def test_stop_failure_marks_application_failed_and_logs_error(
    tmp_path: Path,
) -> None:
    stream = StringIO()
    application = create_test_application(
        tmp_path,
        logger=create_test_logger(stream),
    )
    application.start()
    application.context.fail()

    with pytest.raises(
        SessionTransitionError,
        match="Cannot terminate session",
    ):
        application.stop()

    payload = read_log_payloads(stream)[-1]

    assert application.state is ApplicationState.FAILED
    assert payload["event"] == "application.stop_failed"


def test_application_can_be_marked_failed(tmp_path: Path) -> None:
    stream = StringIO()
    application = create_test_application(
        tmp_path,
        logger=create_test_logger(stream),
    )
    application.start()

    application.fail()

    payloads = read_log_payloads(stream)

    assert application.state is ApplicationState.FAILED
    assert application.context.state is SessionState.FAILED
    assert [payload["event"] for payload in payloads[-2:]] == [
        "session.failed",
        "application.failed",
    ]


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
            logger=application.logger,
        )

    with pytest.raises(TypeError, match="context"):
        GameApplication(
            config=application.config,
            context=cast(Any, object()),
            logger=application.logger,
        )

    with pytest.raises(TypeError, match="logger"):
        GameApplication(
            config=application.config,
            context=application.context,
            logger=cast(Any, object()),
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
            logger=application.logger,
        )
