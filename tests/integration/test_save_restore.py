"""Integration tests for the complete save and restore workflow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from open_world_rpg.application.session import (
    GameMode,
    RuntimeContext,
    SessionState,
)
from open_world_rpg.application.session_restore import restore_game_session
from open_world_rpg.core import (
    GameConfig,
    ProjectPaths,
    RuntimeEnvironment,
    SimulationConfig,
)
from open_world_rpg.persistence import (
    RuntimeStorage,
    SaveDocument,
    SaveRepository,
    SaveSlot,
)


def test_save_load_and_restore_paused_session(
    tmp_path: Path,
) -> None:
    session_id = UUID("12345678-1234-5678-1234-567812345678")
    created_at = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
    timestamps = iter(
        [
            created_at,
            created_at + timedelta(seconds=1),
            created_at + timedelta(seconds=2),
        ]
    )

    original = RuntimeContext.create(
        session_id=session_id,
        game_mode=GameMode.NEW_GAME,
        world_seed=77,
        clock=lambda: next(timestamps),
    )
    original.start()
    original.pause()

    document = SaveDocument.from_runtime_context(
        context=original,
        saved_at=created_at + timedelta(minutes=5),
        payload={"player": {"level": 15}},
    )

    repository = SaveRepository(
        storage=RuntimeStorage(paths=ProjectPaths.from_project_root(tmp_path))
    )
    slot = SaveSlot("campaign-01")
    repository.save(slot=slot, document=document)

    loaded = repository.load(slot)
    config = GameConfig(
        environment=RuntimeEnvironment.TEST,
        simulation=SimulationConfig(world_seed=77),
        paths=ProjectPaths.from_project_root(tmp_path),
    )

    restored_at = created_at + timedelta(hours=1)
    restore_times = iter(
        [
            restored_at,
            restored_at + timedelta(seconds=1),
            restored_at + timedelta(seconds=2),
        ]
    )
    restored = restore_game_session(
        document=loaded,
        config=config,
        clock=lambda: next(restore_times),
    )

    assert restored.context.session_id == session_id
    assert restored.context.game_mode is GameMode.NEW_GAME
    assert restored.context.world_seed == 77
    assert restored.context.state is SessionState.PAUSED
    assert restored.payload == {"player": {"level": 15}}
