"""Tests for runtime session identity and state transitions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast
from uuid import UUID

import pytest

from open_world_rpg.application.session import (
    Clock,
    GameMode,
    RuntimeContext,
    SessionClockError,
    SessionState,
    SessionTransitionError,
)
from open_world_rpg.core.config import MAX_WORLD_SEED

SESSION_ID = UUID("12345678-1234-5678-1234-567812345678")
BASE_TIME = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)


def sequence_clock(*values: datetime) -> Clock:
    iterator = iter(values)
    return lambda: next(iterator)


def create_context(
    *,
    clock: Clock | None = None,
    game_mode: GameMode = GameMode.NEW_GAME,
    world_seed: int = 42,
) -> RuntimeContext:
    return RuntimeContext.create(
        game_mode=game_mode,
        world_seed=world_seed,
        clock=clock or sequence_clock(BASE_TIME),
        session_id=SESSION_ID,
    )


def test_game_mode_and_session_state_values() -> None:
    assert GameMode.NEW_GAME.value == "new_game"
    assert GameMode.LOADED_GAME.value == "loaded_game"

    assert SessionState.CREATED.value == "created"
    assert SessionState.ACTIVE.value == "active"
    assert SessionState.PAUSED.value == "paused"
    assert SessionState.TERMINATED.value == "terminated"
    assert SessionState.FAILED.value == "failed"


def test_create_context_uses_supplied_identity_and_clock() -> None:
    local_time = datetime(
        2026,
        7,
        25,
        11,
        0,
        tzinfo=timezone(timedelta(hours=1)),
    )

    context = RuntimeContext.create(
        game_mode=GameMode.LOADED_GAME,
        world_seed=9001,
        clock=sequence_clock(local_time),
        session_id=SESSION_ID,
    )

    assert context.session_id == SESSION_ID
    assert context.game_mode is GameMode.LOADED_GAME
    assert context.world_seed == 9001
    assert context.created_at == BASE_TIME
    assert context.last_transition_at == BASE_TIME
    assert context.state is SessionState.CREATED
    assert context.started_at is None
    assert context.paused_at is None
    assert context.resumed_at is None
    assert context.terminated_at is None
    assert context.failed_at is None
    assert context.is_active is False
    assert context.is_paused is False
    assert context.is_terminal is False


def test_create_context_generates_default_identity_and_time() -> None:
    context = RuntimeContext.create(
        game_mode=GameMode.NEW_GAME,
        world_seed=0,
    )

    assert isinstance(context.session_id, UUID)
    assert context.created_at.tzinfo is UTC


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("session_id", "invalid", TypeError),
        ("game_mode", "new_game", TypeError),
        ("world_seed", "42", TypeError),
        ("world_seed", True, TypeError),
        ("world_seed", -1, ValueError),
        ("world_seed", MAX_WORLD_SEED + 1, ValueError),
    ],
)
def test_context_rejects_invalid_identity_and_configuration(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, Any] = {
        "session_id": SESSION_ID,
        "game_mode": GameMode.NEW_GAME,
        "world_seed": 42,
        "created_at": BASE_TIME,
        "_clock": sequence_clock(BASE_TIME),
    }
    values[field_name] = value

    with pytest.raises(error_type, match=field_name):
        RuntimeContext(**values)


def test_context_rejects_naive_creation_timestamp() -> None:
    with pytest.raises(ValueError, match="created_at"):
        RuntimeContext(
            session_id=SESSION_ID,
            game_mode=GameMode.NEW_GAME,
            world_seed=42,
            created_at=datetime(2026, 7, 25, 10, 0),
            _clock=sequence_clock(BASE_TIME),
        )


def test_context_rejects_non_callable_clock() -> None:
    with pytest.raises(TypeError, match="clock"):
        RuntimeContext(
            session_id=SESSION_ID,
            game_mode=GameMode.NEW_GAME,
            world_seed=42,
            created_at=BASE_TIME,
            _clock=cast(Any, 100),
        )


def test_context_rejects_invalid_clock_result() -> None:
    invalid_clock = cast(Clock, lambda: cast(Any, "not-a-datetime"))

    with pytest.raises(TypeError, match="clock result"):
        RuntimeContext.create(
            game_mode=GameMode.NEW_GAME,
            world_seed=42,
            clock=invalid_clock,
            session_id=SESSION_ID,
        )


def test_context_rejects_naive_clock_result() -> None:
    naive_clock = sequence_clock(datetime(2026, 7, 25, 10, 0))

    with pytest.raises(ValueError, match="clock result"):
        RuntimeContext.create(
            game_mode=GameMode.NEW_GAME,
            world_seed=42,
            clock=naive_clock,
            session_id=SESSION_ID,
        )


def test_session_start_records_timestamp() -> None:
    started_at = BASE_TIME + timedelta(seconds=1)
    context = create_context(
        clock=sequence_clock(BASE_TIME, started_at),
    )

    context.start()

    assert context.state is SessionState.ACTIVE
    assert context.started_at == started_at
    assert context.last_transition_at == started_at
    assert context.is_active is True


def test_session_pause_and_resume_record_timestamps() -> None:
    started_at = BASE_TIME + timedelta(seconds=1)
    paused_at = BASE_TIME + timedelta(seconds=2)
    resumed_at = BASE_TIME + timedelta(seconds=3)
    context = create_context(
        clock=sequence_clock(
            BASE_TIME,
            started_at,
            paused_at,
            resumed_at,
        )
    )

    context.start()
    context.pause()

    assert context.state is SessionState.PAUSED
    assert context.paused_at == paused_at
    assert context.is_paused is True

    context.resume()

    assert context.state is SessionState.ACTIVE
    assert context.resumed_at == resumed_at
    assert context.last_transition_at == resumed_at


@pytest.mark.parametrize("pause_before_termination", [False, True])
def test_session_can_terminate_from_active_or_paused_state(
    pause_before_termination: bool,
) -> None:
    timestamps = [
        BASE_TIME,
        BASE_TIME + timedelta(seconds=1),
        BASE_TIME + timedelta(seconds=2),
        BASE_TIME + timedelta(seconds=3),
    ]
    context = create_context(clock=sequence_clock(*timestamps))

    context.start()
    if pause_before_termination:
        context.pause()

    context.terminate()

    expected_time = (
        BASE_TIME + timedelta(seconds=3)
        if pause_before_termination
        else BASE_TIME + timedelta(seconds=2)
    )
    assert context.state is SessionState.TERMINATED
    assert context.terminated_at == expected_time
    assert context.is_terminal is True


def test_session_failure_is_recorded_and_idempotent() -> None:
    failed_at = BASE_TIME + timedelta(seconds=1)
    context = create_context(
        clock=sequence_clock(BASE_TIME, failed_at),
    )

    context.fail()
    context.fail()

    assert context.state is SessionState.FAILED
    assert context.failed_at == failed_at
    assert context.is_terminal is True


@pytest.mark.parametrize(
    ("operation", "expected_message"),
    [
        ("pause", "Cannot pause session"),
        ("resume", "Cannot resume session"),
        ("terminate", "Cannot terminate session"),
    ],
)
def test_created_session_rejects_invalid_transitions(
    operation: str,
    expected_message: str,
) -> None:
    context = create_context()

    with pytest.raises(SessionTransitionError, match=expected_message):
        getattr(context, operation)()


def test_active_session_rejects_duplicate_start() -> None:
    context = create_context(
        clock=sequence_clock(
            BASE_TIME,
            BASE_TIME + timedelta(seconds=1),
        )
    )
    context.start()

    with pytest.raises(SessionTransitionError, match="Cannot start session"):
        context.start()


def test_terminated_session_cannot_fail() -> None:
    context = create_context(
        clock=sequence_clock(
            BASE_TIME,
            BASE_TIME + timedelta(seconds=1),
            BASE_TIME + timedelta(seconds=2),
        )
    )
    context.start()
    context.terminate()

    with pytest.raises(
        SessionTransitionError,
        match="cannot be marked as failed",
    ):
        context.fail()


def test_runtime_clock_cannot_move_backwards() -> None:
    context = create_context(
        clock=sequence_clock(
            BASE_TIME,
            BASE_TIME - timedelta(seconds=1),
        )
    )

    with pytest.raises(SessionClockError, match="moved backwards"):
        context.start()

    assert context.state is SessionState.CREATED


def test_runtime_context_configuration_is_immutable() -> None:
    context = create_context()

    with pytest.raises(FrozenInstanceError):
        context.world_seed = 100  # type: ignore[misc]


def test_create_context_rejects_non_callable_clock() -> None:
    with pytest.raises(TypeError, match="clock must be callable"):
        RuntimeContext.create(
            game_mode=GameMode.NEW_GAME,
            world_seed=42,
            clock=cast(Any, 100),
            session_id=SESSION_ID,
        )
