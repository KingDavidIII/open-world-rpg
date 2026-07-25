"""Tests for restoring save documents into runtime sessions."""

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
from open_world_rpg.application.session_restore import (
    RestoredGameSession,
    SessionConfigurationMismatchError,
    restore_game_session,
)
from open_world_rpg.core import (
    GameConfig,
    ProjectPaths,
    RuntimeEnvironment,
    SimulationConfig,
)
from open_world_rpg.persistence.document import (
    CURRENT_SAVE_SCHEMA_VERSION,
    SaveDocument,
    SessionSnapshot,
)

SESSION_ID = UUID("12345678-1234-5678-1234-567812345678")
SAVED_AT = datetime(2026, 7, 25, 10, 5, tzinfo=UTC)
RESTORED_AT = datetime(2026, 7, 25, 11, 0, tzinfo=UTC)


def sequence_clock(*values: datetime):
    iterator = iter(values)
    return lambda: next(iterator)


def create_config(
    tmp_path,
    *,
    world_seed: int = 42,
) -> GameConfig:
    return GameConfig(
        environment=RuntimeEnvironment.TEST,
        simulation=SimulationConfig(world_seed=world_seed),
        paths=ProjectPaths.from_project_root(tmp_path),
    )


def create_document(
    *,
    state: SessionState = SessionState.ACTIVE,
    world_seed: int = 42,
) -> SaveDocument:
    return SaveDocument(
        schema_version=CURRENT_SAVE_SCHEMA_VERSION,
        saved_at=SAVED_AT,
        session=SessionSnapshot(
            session_id=SESSION_ID,
            game_mode=GameMode.LOADED_GAME,
            world_seed=world_seed,
            state=state,
        ),
        payload={
            "player": {
                "level": 12,
                "inventory": ["sword"],
            }
        },
    )


def test_restore_game_session_preserves_active_metadata(
    tmp_path,
) -> None:
    document = create_document()
    started_at = RESTORED_AT + timedelta(seconds=1)

    restored = restore_game_session(
        document=document,
        config=create_config(tmp_path),
        clock=sequence_clock(RESTORED_AT, started_at),
    )

    assert restored.document is document
    assert restored.context.session_id == SESSION_ID
    assert restored.context.game_mode is GameMode.LOADED_GAME
    assert restored.context.world_seed == 42
    assert restored.context.state is SessionState.ACTIVE
    assert restored.context.created_at == RESTORED_AT
    assert restored.context.started_at == started_at


def test_restore_game_session_recreates_paused_state(
    tmp_path,
) -> None:
    document = create_document(state=SessionState.PAUSED)

    restored = restore_game_session(
        document=document,
        config=create_config(tmp_path),
        clock=sequence_clock(
            RESTORED_AT,
            RESTORED_AT + timedelta(seconds=1),
            RESTORED_AT + timedelta(seconds=2),
        ),
    )

    assert restored.context.state is SessionState.PAUSED
    assert restored.context.is_paused is True


def test_restored_payload_is_isolated_from_document(
    tmp_path,
) -> None:
    restored = restore_game_session(
        document=create_document(),
        config=create_config(tmp_path),
        clock=sequence_clock(
            RESTORED_AT,
            RESTORED_AT + timedelta(seconds=1),
        ),
    )

    payload = restored.payload
    player = cast(dict[str, Any], payload["player"])
    inventory = cast(list[str], player["inventory"])
    inventory.append("shield")

    assert restored.document.payload == {
        "player": {
            "level": 12,
            "inventory": ["sword"],
        }
    }


def test_restore_rejects_world_seed_mismatch(tmp_path) -> None:
    with pytest.raises(
        SessionConfigurationMismatchError,
        match="saved=42, configured=99",
    ):
        restore_game_session(
            document=create_document(),
            config=create_config(tmp_path, world_seed=99),
        )


def test_restore_rejects_invalid_arguments(tmp_path) -> None:
    document = create_document()
    config = create_config(tmp_path)

    with pytest.raises(TypeError, match="document"):
        restore_game_session(
            document=cast(Any, object()),
            config=config,
        )

    with pytest.raises(TypeError, match="config"):
        restore_game_session(
            document=document,
            config=cast(Any, object()),
        )


def test_restored_session_rejects_invalid_dependencies(
    tmp_path,
) -> None:
    document = create_document()
    context = RuntimeContext.restore(
        session_id=SESSION_ID,
        game_mode=GameMode.LOADED_GAME,
        world_seed=42,
        state=SessionState.ACTIVE,
        clock=sequence_clock(
            RESTORED_AT,
            RESTORED_AT + timedelta(seconds=1),
        ),
    )

    with pytest.raises(TypeError, match="context"):
        RestoredGameSession(
            context=cast(Any, object()),
            document=document,
        )

    with pytest.raises(TypeError, match="document"):
        RestoredGameSession(
            context=context,
            document=cast(Any, object()),
        )


@pytest.mark.parametrize(
    "mismatch",
    [
        "session_id",
        "game_mode",
        "world_seed",
        "state",
    ],
)
def test_restored_session_enforces_document_invariants(
    mismatch: str,
) -> None:
    document = create_document()
    context = RuntimeContext.restore(
        session_id=(
            UUID("87654321-4321-8765-4321-876543218765") if mismatch == "session_id" else SESSION_ID
        ),
        game_mode=(GameMode.NEW_GAME if mismatch == "game_mode" else GameMode.LOADED_GAME),
        world_seed=99 if mismatch == "world_seed" else 42,
        state=(SessionState.PAUSED if mismatch == "state" else SessionState.ACTIVE),
        clock=sequence_clock(
            RESTORED_AT,
            RESTORED_AT + timedelta(seconds=1),
            RESTORED_AT + timedelta(seconds=2),
        ),
    )

    with pytest.raises(ValueError, match="does not match"):
        RestoredGameSession(
            context=context,
            document=document,
        )
