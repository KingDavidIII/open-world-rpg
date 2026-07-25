"""Tests for save-document models and runtime snapshots."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

import pytest

from open_world_rpg.application.session import (
    GameMode,
    RuntimeContext,
    SessionState,
)
from open_world_rpg.core.config import MAX_WORLD_SEED
from open_world_rpg.persistence.document import (
    CURRENT_SAVE_SCHEMA_VERSION,
    SaveCompatibilityError,
    SaveDocument,
    SessionSnapshot,
)

SESSION_ID = UUID("12345678-1234-5678-1234-567812345678")
CREATED_AT = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
SAVED_AT = datetime(2026, 7, 25, 10, 5, tzinfo=UTC)


def create_active_context() -> RuntimeContext:
    timestamps = iter(
        [
            CREATED_AT,
            CREATED_AT + timedelta(seconds=1),
        ]
    )
    context = RuntimeContext.create(
        game_mode=GameMode.NEW_GAME,
        world_seed=42,
        clock=lambda: next(timestamps),
        session_id=SESSION_ID,
    )
    context.start()
    return context


def create_paused_context() -> RuntimeContext:
    timestamps = iter(
        [
            CREATED_AT,
            CREATED_AT + timedelta(seconds=1),
            CREATED_AT + timedelta(seconds=2),
        ]
    )
    context = RuntimeContext.create(
        game_mode=GameMode.LOADED_GAME,
        world_seed=84,
        clock=lambda: next(timestamps),
        session_id=SESSION_ID,
    )
    context.start()
    context.pause()
    return context


def test_current_schema_version() -> None:
    assert CURRENT_SAVE_SCHEMA_VERSION == 1


@pytest.mark.parametrize(
    "state",
    [
        SessionState.ACTIVE,
        SessionState.PAUSED,
    ],
)
def test_session_snapshot_accepts_resumable_states(
    state: SessionState,
) -> None:
    snapshot = SessionSnapshot(
        session_id=SESSION_ID,
        game_mode=GameMode.NEW_GAME,
        world_seed=42,
        state=state,
    )

    assert snapshot.state is state


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("session_id", "invalid", TypeError),
        ("game_mode", "new_game", TypeError),
        ("world_seed", "42", TypeError),
        ("world_seed", True, TypeError),
        ("world_seed", -1, ValueError),
        ("world_seed", MAX_WORLD_SEED + 1, ValueError),
        ("state", "active", TypeError),
        ("state", SessionState.CREATED, ValueError),
        ("state", SessionState.TERMINATED, ValueError),
        ("state", SessionState.FAILED, ValueError),
    ],
)
def test_session_snapshot_rejects_invalid_values(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, Any] = {
        "session_id": SESSION_ID,
        "game_mode": GameMode.NEW_GAME,
        "world_seed": 42,
        "state": SessionState.ACTIVE,
    }
    values[field_name] = value

    with pytest.raises(error_type):
        SessionSnapshot(**values)


def test_snapshot_is_created_from_active_context() -> None:
    context = create_active_context()

    snapshot = SessionSnapshot.from_runtime_context(context)

    assert snapshot.session_id == SESSION_ID
    assert snapshot.game_mode is GameMode.NEW_GAME
    assert snapshot.world_seed == 42
    assert snapshot.state is SessionState.ACTIVE


def test_snapshot_rejects_invalid_context() -> None:
    with pytest.raises(TypeError, match="context"):
        SessionSnapshot.from_runtime_context(cast(Any, object()))


def test_document_is_created_from_active_context() -> None:
    context = create_active_context()

    document = SaveDocument.from_runtime_context(
        context=context,
        saved_at=SAVED_AT,
        payload={
            "player": {
                "name": "Adebayo",
                "level": 7,
            }
        },
    )

    assert document.schema_version == CURRENT_SAVE_SCHEMA_VERSION
    assert document.saved_at == SAVED_AT
    assert document.session.state is SessionState.ACTIVE
    assert document.payload["player"] == {
        "name": "Adebayo",
        "level": 7,
    }


def test_document_is_created_from_paused_context() -> None:
    context = create_paused_context()

    document = SaveDocument.from_runtime_context(
        context=context,
        saved_at=SAVED_AT,
    )

    assert document.session.game_mode is GameMode.LOADED_GAME
    assert document.session.world_seed == 84
    assert document.session.state is SessionState.PAUSED
    assert document.payload == {}


def test_document_uses_current_utc_time_by_default() -> None:
    before = datetime.now(UTC)
    document = SaveDocument.from_runtime_context(
        context=create_active_context(),
    )
    after = datetime.now(UTC)

    assert before <= document.saved_at <= after


def test_document_payload_is_deeply_copied() -> None:
    payload: dict[str, Any] = {
        "player": {
            "inventory": ["sword"],
        }
    }

    document = SaveDocument.from_runtime_context(
        context=create_active_context(),
        saved_at=SAVED_AT,
        payload=payload,
    )

    payload["player"]["inventory"].append("shield")

    assert document.payload == {
        "player": {
            "inventory": ["sword"],
        }
    }


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        ("1", TypeError),
        (True, TypeError),
        (2, SaveCompatibilityError),
    ],
)
def test_document_rejects_invalid_schema_version(
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        SaveDocument(
            schema_version=cast(Any, value),
            saved_at=SAVED_AT,
            session=SessionSnapshot(
                session_id=SESSION_ID,
                game_mode=GameMode.NEW_GAME,
                world_seed=42,
                state=SessionState.ACTIVE,
            ),
        )


def test_document_rejects_invalid_saved_at() -> None:
    snapshot = SessionSnapshot(
        session_id=SESSION_ID,
        game_mode=GameMode.NEW_GAME,
        world_seed=42,
        state=SessionState.ACTIVE,
    )

    with pytest.raises(TypeError, match="saved_at"):
        SaveDocument(
            schema_version=CURRENT_SAVE_SCHEMA_VERSION,
            saved_at=cast(Any, "2026-07-25"),
            session=snapshot,
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        SaveDocument(
            schema_version=CURRENT_SAVE_SCHEMA_VERSION,
            saved_at=datetime(2026, 7, 25, 10, 0),
            session=snapshot,
        )


def test_document_rejects_invalid_session() -> None:
    with pytest.raises(TypeError, match="session"):
        SaveDocument(
            schema_version=CURRENT_SAVE_SCHEMA_VERSION,
            saved_at=SAVED_AT,
            session=cast(Any, object()),
        )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"invalid": b"bytes"},
        {"invalid": ("tuple",)},
        {"invalid": float("nan")},
        {"invalid": float("inf")},
        {1: "invalid key"},
    ],
)
def test_document_rejects_invalid_payload(
    payload: object,
) -> None:
    snapshot = SessionSnapshot(
        session_id=SESSION_ID,
        game_mode=GameMode.NEW_GAME,
        world_seed=42,
        state=SessionState.ACTIVE,
    )

    with pytest.raises((TypeError, ValueError)):
        SaveDocument(
            schema_version=CURRENT_SAVE_SCHEMA_VERSION,
            saved_at=SAVED_AT,
            session=snapshot,
            payload=cast(Any, payload),
        )


def test_document_rejects_invalid_runtime_context() -> None:
    with pytest.raises(TypeError, match="context"):
        SaveDocument.from_runtime_context(
            context=cast(Any, object()),
        )


@pytest.mark.parametrize(
    "state_transition",
    [
        "none",
        "terminate",
        "fail",
    ],
)
def test_document_rejects_non_resumable_context(
    state_transition: str,
) -> None:
    timestamps = iter(
        [
            CREATED_AT,
            CREATED_AT + timedelta(seconds=1),
            CREATED_AT + timedelta(seconds=2),
        ]
    )
    context = RuntimeContext.create(
        game_mode=GameMode.NEW_GAME,
        world_seed=42,
        clock=lambda: next(timestamps),
        session_id=SESSION_ID,
    )

    if state_transition == "terminate":
        context.start()
        context.terminate()
    elif state_transition == "fail":
        context.fail()

    with pytest.raises(ValueError, match="active or paused"):
        SaveDocument.from_runtime_context(
            context=context,
            saved_at=SAVED_AT,
        )
