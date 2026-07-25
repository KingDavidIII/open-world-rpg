"""Tests for application-level world and engine construction helpers."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from open_world_rpg.application import (
    GameApplication,
    GameMode,
    RuntimeContext,
    create_world_engine_runtime,
    create_world_model,
    create_world_runtime,
)
from open_world_rpg.core import (
    GameConfig,
    ProjectPaths,
    RuntimeEnvironment,
    SimulationConfig,
)
from open_world_rpg.engine import EventBus
from open_world_rpg.world import (
    WorldRuntime,
    WorldState,
)

SESSION_ID = UUID("12345678-1234-5678-1234-567812345678")
CREATED_AT = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)


def create_application(tmp_path: Path) -> GameApplication:
    config = GameConfig(
        title="Open World RPG",
        environment=RuntimeEnvironment.TEST,
        simulation=SimulationConfig(
            world_seed=77,
            tick_rate=20,
        ),
        paths=ProjectPaths.from_project_root(tmp_path),
    )
    context = RuntimeContext.create(
        game_mode=GameMode.NEW_GAME,
        world_seed=77,
        session_id=SESSION_ID,
        clock=lambda: CREATED_AT,
    )
    return GameApplication(
        config=config,
        context=context,
        logger=logging.Logger("test.world.bootstrap"),
    )


def test_create_world_model_reuses_application_rules_and_session_identity(
    tmp_path: Path,
) -> None:
    application = create_application(tmp_path)

    model = create_world_model(application=application)

    assert model.metadata.name == application.config.title
    assert model.metadata.seed == 77
    assert model.metadata.created_at == application.context.created_at
    assert model.metadata.world_id.value == application.context.session_id
    assert model.metadata.state is WorldState.CREATED
    assert model.specification.seed.value == 77
    assert model.specification.time_config.ticks_per_second == 20
    assert model.clock.current.tick == 0


def test_create_world_model_accepts_explicit_name(tmp_path: Path) -> None:
    model = create_world_model(
        application=create_application(tmp_path),
        name="The Northern Reach",
    )

    assert model.metadata.name == "The Northern Reach"
    assert model.specification.name == "The Northern Reach"


def test_world_helpers_reject_invalid_application_and_event_bus(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="application must be a GameApplication"):
        create_world_model(application=cast(Any, object()))

    with pytest.raises(TypeError, match="event_bus must be an EventBus or None"):
        create_world_runtime(
            application=create_application(tmp_path),
            event_bus=cast(Any, object()),
        )


def test_create_world_runtime_uses_optional_event_bus(tmp_path: Path) -> None:
    event_bus = EventBus()

    runtime = create_world_runtime(
        application=create_application(tmp_path),
        event_bus=event_bus,
    )
    runtime.initialise()

    assert runtime.revision == 1
    assert event_bus.pending_event_count == 1


def test_create_world_engine_runtime_wires_service_and_subsystem(
    tmp_path: Path,
) -> None:
    application = create_application(tmp_path)

    engine = create_world_engine_runtime(application=application)
    runtime = engine.context.resolve(WorldRuntime)

    assert engine.registry.subsystem_names == ("world",)
    assert runtime.model.metadata.state is WorldState.CREATED
    assert runtime.model.specification.seed.value == 77
    assert engine.context.event_bus is engine.event_bus
