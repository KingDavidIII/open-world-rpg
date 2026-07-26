"""End-to-end acceptance coverage for the procedural terrain foundation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn
from uuid import UUID

import pytest

from open_world_rpg.application import (
    GameApplication,
    GameMode,
    RuntimeContext,
    create_application_terrain_runtime,
    create_terrain_runtime,
    create_world_model,
)
from open_world_rpg.core import (
    GameConfig,
    ProjectPaths,
    RuntimeEnvironment,
    SimulationConfig,
)
from open_world_rpg.engine import EventBus
from open_world_rpg.world import (
    ChunkCoordinate,
    ChunkState,
    InMemoryTerrainRepository,
    TerrainChunkActivated,
    TerrainChunkDeclared,
    TerrainChunkGenerated,
    TerrainChunkGenerationStarted,
    TerrainChunkSuspended,
    TerrainChunkUnloaded,
    TerrainGenerationConfig,
    TerrainGenerationService,
    TerrainRuntimeGenerationError,
    TerrainRuntimeRepositoryError,
)
from open_world_rpg.world.model import WorldSpecification
from open_world_rpg.world.terrain import TerrainGenerationError

SESSION_ID = UUID("87654321-4321-8765-4321-876543218765")
CREATED_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


class RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def create_application(tmp_path: Path, logger: logging.Logger) -> GameApplication:
    return GameApplication(
        config=GameConfig(
            title="Open World RPG",
            environment=RuntimeEnvironment.TEST,
            simulation=SimulationConfig(world_seed=77, tick_rate=20),
            paths=ProjectPaths.from_project_root(tmp_path),
        ),
        context=RuntimeContext.create(
            game_mode=GameMode.NEW_GAME,
            world_seed=77,
            session_id=SESSION_ID,
            clock=lambda: CREATED_AT,
        ),
        logger=logger,
    )


class FailingGenerator:
    def generate(
        self,
        *,
        specification: WorldSpecification,
        coordinate: ChunkCoordinate,
    ) -> NoReturn:
        del specification, coordinate
        raise TerrainGenerationError("forced generation failure")


class FailingEvictionRepository(InMemoryTerrainRepository):
    def remove(self, coordinate: ChunkCoordinate) -> NoReturn:
        del coordinate
        raise RuntimeError("forced eviction failure")


def test_application_to_terrain_runtime_acceptance(tmp_path: Path) -> None:
    logger = logging.Logger("test.integration.terrain", level=logging.DEBUG)
    handler = RecordingHandler()
    logger.addHandler(handler)
    application = create_application(tmp_path, logger)
    world = create_world_model(application=application)
    event_bus = EventBus()
    config = TerrainGenerationConfig(octave_count=1)
    runtime = create_application_terrain_runtime(
        application=application,
        world=world,
        config=config,
        event_bus=event_bus,
    )
    coordinate = ChunkCoordinate(x=-17, y=16)
    events: list[object] = []
    event_types = (
        TerrainChunkDeclared,
        TerrainChunkGenerationStarted,
        TerrainChunkGenerated,
        TerrainChunkActivated,
        TerrainChunkSuspended,
        TerrainChunkUnloaded,
    )
    for event_type in event_types:
        event_bus.subscribe(event_type, events.append)

    first = runtime.get_or_generate(coordinate)
    runtime.activate(coordinate)
    runtime.suspend(coordinate)
    runtime.unload(coordinate)
    second = runtime.generate(coordinate)
    event_bus.dispatch_pending()

    assert first == second
    assert first is not second
    assert world.metadata.created_at == application.context.created_at == CREATED_AT
    assert runtime.metadata_at(coordinate).state is ChunkState.READY
    assert runtime.revision == 8
    assert runtime.service.snapshot().repository_revision == 3
    assert runtime.service.snapshot().cache_hits == 0
    assert runtime.service.snapshot().cache_misses == 0
    assert runtime.service.snapshot().successful_generations == 2
    assert runtime.service.snapshot().failed_generations == 0
    assert runtime.service.snapshot().evictions == 1
    assert [type(event) for event in events] == [
        TerrainChunkDeclared,
        TerrainChunkGenerationStarted,
        TerrainChunkGenerated,
        TerrainChunkActivated,
        TerrainChunkSuspended,
        TerrainChunkUnloaded,
        TerrainChunkGenerationStarted,
        TerrainChunkGenerated,
    ]
    assert [event.runtime_revision for event in events] == list(range(1, 9))
    assert [
        record.event
        for record in handler.records
        if getattr(record, "event", "").startswith("terrain.")
    ] == [
        "terrain.chunk_declared",
        "terrain.generation_started",
        "terrain.chunk_generated",
        "terrain.chunk_activated",
        "terrain.chunk_suspended",
        "terrain.chunk_unloaded",
        "terrain.generation_started",
        "terrain.chunk_generated",
    ]

    failing_service = TerrainGenerationService(
        specification=world.specification,
        config=config,
        generator=FailingGenerator(),
    )
    failing_runtime = create_terrain_runtime(
        world=world,
        service=failing_service,
        logger=logger,
    )
    failed_coordinate = ChunkCoordinate(x=-18, y=16)
    with pytest.raises(TerrainRuntimeGenerationError):
        failing_runtime.get_or_generate(failed_coordinate)
    assert failing_runtime.metadata_at(failed_coordinate).state is ChunkState.FAILED
    assert len(failing_service.repository) == 0
    assert failing_service.snapshot().repository_revision == 0

    base_service = TerrainGenerationService(
        specification=world.specification,
        config=config,
    )
    repository = FailingEvictionRepository(scope=base_service.repository.scope)
    eviction_service = TerrainGenerationService(
        specification=world.specification,
        config=config,
        repository=repository,
    )
    eviction_runtime = create_terrain_runtime(
        world=world.specification,
        service=eviction_service,
        logger=logger,
    )
    cached = eviction_runtime.get_or_generate(coordinate)
    revision = eviction_runtime.revision
    with pytest.raises(TerrainRuntimeRepositoryError):
        eviction_runtime.unload(coordinate)
    assert eviction_runtime.metadata_at(coordinate).state is ChunkState.READY
    assert eviction_runtime.terrain_at(coordinate) is cached
    assert eviction_runtime.revision == revision
