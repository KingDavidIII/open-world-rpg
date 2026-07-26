"""Tests for controlled terrain lifecycle orchestration and events."""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError
from typing import Any, NoReturn, cast

import pytest

import open_world_rpg.world as world
from open_world_rpg.engine import EventBus
from open_world_rpg.world import (
    ChunkCoordinate,
    ChunkMetadata,
    ChunkState,
    IncompatibleTerrainRuntimeError,
    InMemoryTerrainRepository,
    TerrainChunkActivated,
    TerrainChunkDeclared,
    TerrainChunkFailed,
    TerrainChunkGenerated,
    TerrainChunkGenerationStarted,
    TerrainChunkSuspended,
    TerrainChunkUnloaded,
    TerrainGenerationConfig,
    TerrainGenerationInProgressError,
    TerrainGenerationService,
    TerrainRuntime,
    TerrainRuntimeError,
    TerrainRuntimeGenerationError,
    TerrainRuntimeInvalidOperationError,
    TerrainRuntimeMissingChunkError,
    TerrainRuntimeRepositoryError,
    TerrainRuntimeSnapshot,
    TerrainUnavailableError,
    WorldSeed,
    WorldSpecification,
)
from open_world_rpg.world.terrain import TerrainGenerationError

CONFIG = TerrainGenerationConfig(octave_count=1)
SPECIFICATION = WorldSpecification(name="Terrain Runtime", seed=WorldSeed(value=42))

EVENT_TYPES = (
    TerrainChunkDeclared,
    TerrainChunkGenerationStarted,
    TerrainChunkGenerated,
    TerrainChunkActivated,
    TerrainChunkSuspended,
    TerrainChunkUnloaded,
    TerrainChunkFailed,
)


def create_service(
    *,
    generator: object | None = None,
    repository: InMemoryTerrainRepository | None = None,
    specification: WorldSpecification = SPECIFICATION,
) -> TerrainGenerationService:
    return TerrainGenerationService(
        specification=specification,
        config=CONFIG,
        generator=cast(Any, generator),
        repository=repository,
    )


def create_runtime(
    *,
    event_bus: EventBus | None = None,
    service: TerrainGenerationService | None = None,
    specification: WorldSpecification = SPECIFICATION,
    logger: logging.Logger | None = None,
) -> TerrainRuntime:
    return TerrainRuntime(
        specification=specification,
        service=create_service(specification=specification) if service is None else service,
        event_bus=event_bus,
        logger=logger,
    )


def capture_events(event_bus: EventBus) -> list[object]:
    events: list[object] = []
    for event_type in EVENT_TYPES:
        event_bus.subscribe(event_type, events.append)
    return events


def dispatch(event_bus: EventBus, events: list[object]) -> list[object]:
    event_bus.dispatch_pending()
    return events


def test_construction_is_empty_compatible_and_does_not_generate() -> None:
    service = create_service()
    runtime = create_runtime(service=service)

    assert runtime.specification is SPECIFICATION
    assert runtime.service is service
    assert runtime.revision == 0
    assert runtime.coordinates() == ()
    assert service.snapshot().successful_generations == 0
    assert isinstance(runtime.logger, logging.Logger)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("specification", object(), "specification must be"),
        ("service", object(), "service must be"),
        ("event_bus", object(), "event_bus must be"),
        ("logger", object(), "logger must be"),
    ],
)
def test_construction_validates_dependency_types(
    field_name: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "specification": SPECIFICATION,
        "service": create_service(),
        field_name: value,
    }
    with pytest.raises(TypeError, match=message):
        TerrainRuntime(**cast(Any, values))


def test_construction_rejects_incompatible_service_specification_and_format() -> None:
    other = WorldSpecification(name="Other", seed=WorldSeed(value=43))
    with pytest.raises(IncompatibleTerrainRuntimeError, match="must match"):
        TerrainRuntime(specification=SPECIFICATION, service=create_service(specification=other))

    service = create_service()
    object.__setattr__(service.config, "generation_format_version", "v2")
    with pytest.raises(IncompatibleTerrainRuntimeError, match="must match"):
        TerrainRuntime(specification=SPECIFICATION, service=service)
    object.__setattr__(service.config, "generation_format_version", "v1")


@pytest.mark.parametrize(
    "method_name",
    [
        "declare",
        "generate",
        "get_or_generate",
        "activate",
        "suspend",
        "unload",
        "fail",
        "terrain_at",
        "metadata_at",
        "contains",
    ],
)
def test_coordinate_operations_validate_types(method_name: str) -> None:
    runtime = create_runtime()

    with pytest.raises(TypeError, match="coordinate must be"):
        getattr(runtime, method_name)(cast(Any, object()))


def test_declare_is_revisioned_and_duplicate_is_identity_no_op() -> None:
    event_bus = EventBus()
    events = capture_events(event_bus)
    runtime = create_runtime(event_bus=event_bus)
    coordinate = ChunkCoordinate(x=-17, y=16)

    metadata = runtime.declare(coordinate)
    duplicate = runtime.declare(coordinate)

    assert duplicate is metadata
    assert metadata.state is ChunkState.DECLARED
    assert metadata.region_coordinate == coordinate.to_region()
    assert runtime.revision == 1
    assert runtime.metadata_at(coordinate) is metadata
    assert runtime.contains(coordinate)
    assert dispatch(event_bus, events) == [
        TerrainChunkDeclared(
            coordinate=coordinate,
            region_coordinate=coordinate.to_region(),
            previous_state=None,
            current_state=ChunkState.DECLARED,
            runtime_revision=1,
        )
    ]


def test_missing_metadata_and_terrain_are_explicit() -> None:
    runtime = create_runtime()
    coordinate = ChunkCoordinate(x=1, y=2)

    with pytest.raises(TerrainRuntimeMissingChunkError) as caught:
        runtime.metadata_at(coordinate)
    assert isinstance(caught.value.__cause__, KeyError)

    runtime.declare(coordinate)
    with pytest.raises(TerrainUnavailableError, match="unavailable"):
        runtime.terrain_at(coordinate)


def test_successful_generation_has_exact_lifecycle_revisions_and_events() -> None:
    event_bus = EventBus()
    events = capture_events(event_bus)
    runtime = create_runtime(event_bus=event_bus)
    coordinate = ChunkCoordinate(x=-2, y=3)
    runtime.declare(coordinate)

    terrain = runtime.generate(coordinate)
    metadata = runtime.metadata_at(coordinate)

    assert metadata.state is ChunkState.READY
    assert metadata.revision == 2
    assert runtime.revision == 3
    assert runtime.terrain_at(coordinate) is terrain
    assert runtime.service.snapshot().repository_revision == 1
    assert runtime.service.snapshot().successful_generations == 1

    published = dispatch(event_bus, events)
    assert [type(event) for event in published] == [
        TerrainChunkDeclared,
        TerrainChunkGenerationStarted,
        TerrainChunkGenerated,
    ]
    started = cast(TerrainChunkGenerationStarted, published[1])
    generated = cast(TerrainChunkGenerated, published[2])
    assert started.terrain_seed == metadata.terrain_seed
    assert started.runtime_revision == 2
    assert generated.runtime_revision == 3
    assert generated.repository_revision == 1
    assert generated.minimum_elevation == terrain.minimum_elevation
    assert generated.maximum_elevation == terrain.maximum_elevation
    assert dict(generated.terrain_type_counts) == dict(terrain.terrain_type_counts)


def test_get_or_generate_auto_declares_then_hits_without_lifecycle_change() -> None:
    runtime = create_runtime()
    coordinate = ChunkCoordinate(x=4, y=-5)

    terrain = runtime.get_or_generate(coordinate)
    revision = runtime.revision
    metadata = runtime.metadata_at(coordinate)
    service_snapshot = runtime.service.snapshot()

    assert revision == 3
    assert metadata.state is ChunkState.READY
    assert runtime.get_or_generate(coordinate) is terrain
    assert runtime.revision == revision
    assert runtime.metadata_at(coordinate) is metadata
    assert runtime.service.snapshot() == service_snapshot


def test_activation_suspension_and_idempotent_operations() -> None:
    event_bus = EventBus()
    events = capture_events(event_bus)
    runtime = create_runtime(event_bus=event_bus)
    coordinate = ChunkCoordinate(x=0, y=0)
    runtime.get_or_generate(coordinate)

    active = runtime.activate(coordinate)
    duplicate_active = runtime.activate(coordinate)
    suspended = runtime.suspend(coordinate)
    duplicate_suspended = runtime.suspend(coordinate)
    resumed = runtime.activate(coordinate)

    assert duplicate_active is active
    assert duplicate_suspended is suspended
    assert resumed.state is ChunkState.ACTIVE
    assert runtime.revision == 6
    published = dispatch(event_bus, events)
    assert [type(event) for event in published[-3:]] == [
        TerrainChunkActivated,
        TerrainChunkSuspended,
        TerrainChunkActivated,
    ]
    assert [cast(Any, event).runtime_revision for event in published[-3:]] == [4, 5, 6]


@pytest.mark.parametrize(
    ("operation", "initial_state"),
    [
        ("activate", ChunkState.DECLARED),
        ("suspend", ChunkState.READY),
        ("unload", ChunkState.ACTIVE),
        ("generate", ChunkState.READY),
        ("generate", ChunkState.SUSPENDED),
        ("activate", ChunkState.UNLOADED),
        ("suspend", ChunkState.FAILED),
    ],
)
def test_invalid_lifecycle_operations_preserve_state_and_revision(
    operation: str,
    initial_state: ChunkState,
) -> None:
    runtime = create_runtime()
    coordinate = ChunkCoordinate(x=8, y=9)
    metadata = ChunkMetadata.create(world_seed=SPECIFICATION.seed, coordinate=coordinate)
    for state in {
        ChunkState.GENERATING: (ChunkState.GENERATING,),
        ChunkState.READY: (ChunkState.GENERATING, ChunkState.READY),
        ChunkState.ACTIVE: (
            ChunkState.GENERATING,
            ChunkState.READY,
            ChunkState.ACTIVE,
        ),
        ChunkState.SUSPENDED: (
            ChunkState.GENERATING,
            ChunkState.READY,
            ChunkState.ACTIVE,
            ChunkState.SUSPENDED,
        ),
        ChunkState.UNLOADED: (
            ChunkState.GENERATING,
            ChunkState.READY,
            ChunkState.UNLOADED,
        ),
        ChunkState.FAILED: (ChunkState.FAILED,),
    }.get(initial_state, ()):
        metadata = metadata.transition_to(state)
    runtime._metadata[coordinate] = metadata  # type: ignore[attr-defined]
    original_revision = runtime.revision

    with pytest.raises(TerrainRuntimeInvalidOperationError):
        getattr(runtime, operation)(coordinate)

    assert runtime.metadata_at(coordinate) is metadata
    assert runtime.revision == original_revision


def test_generation_in_progress_has_dedicated_error() -> None:
    runtime = create_runtime()
    coordinate = ChunkCoordinate(x=3, y=3)
    metadata = ChunkMetadata.create(
        world_seed=SPECIFICATION.seed,
        coordinate=coordinate,
    ).transition_to(ChunkState.GENERATING)
    runtime._metadata[coordinate] = metadata  # type: ignore[attr-defined]

    with pytest.raises(TerrainGenerationInProgressError):
        runtime.generate(coordinate)
    with pytest.raises(TerrainGenerationInProgressError):
        runtime.get_or_generate(coordinate)


def test_transition_error_is_translated_without_runtime_mutation() -> None:
    runtime = create_runtime()
    coordinate = ChunkCoordinate(x=11, y=12)
    metadata = runtime.declare(coordinate)
    revision = runtime.revision

    with pytest.raises(TerrainRuntimeInvalidOperationError) as caught:
        runtime._transition(metadata, ChunkState.ACTIVE)  # type: ignore[attr-defined]

    assert caught.value.__cause__ is not None
    assert runtime.metadata_at(coordinate) is metadata
    assert runtime.revision == revision


def test_unload_ready_and_suspended_is_evict_first_and_idempotent() -> None:
    runtime = create_runtime()
    ready_coordinate = ChunkCoordinate(x=1, y=0)
    suspended_coordinate = ChunkCoordinate(x=2, y=0)
    runtime.get_or_generate(ready_coordinate)
    runtime.get_or_generate(suspended_coordinate)
    runtime.activate(suspended_coordinate)
    runtime.suspend(suspended_coordinate)
    service_before = runtime.service.snapshot()

    ready = runtime.unload(ready_coordinate)
    suspended = runtime.unload(suspended_coordinate)
    revision = runtime.revision
    duplicate = runtime.unload(ready_coordinate)

    assert ready.state is suspended.state is ChunkState.UNLOADED
    assert duplicate is ready
    assert not runtime.service.contains(ready_coordinate)
    assert not runtime.service.contains(suspended_coordinate)
    assert runtime.revision == revision
    service_after = runtime.service.snapshot()
    assert service_after.evictions == service_before.evictions + 2
    assert service_after.repository_revision == service_before.repository_revision + 2


def test_unloaded_chunk_can_generate_again() -> None:
    runtime = create_runtime()
    coordinate = ChunkCoordinate(x=-1, y=-1)
    first = runtime.get_or_generate(coordinate)
    runtime.unload(coordinate)
    revision = runtime.revision

    second = runtime.generate(coordinate)

    assert second == first
    assert second is not first
    assert runtime.metadata_at(coordinate).state is ChunkState.READY
    assert runtime.revision == revision + 2
    assert runtime.service.snapshot().successful_generations == 2


class FailingGenerator:
    def generate(
        self,
        *,
        specification: WorldSpecification,
        coordinate: ChunkCoordinate,
    ) -> NoReturn:
        del specification, coordinate
        raise TerrainGenerationError("generation exploded")


def test_generation_failure_transitions_to_failed_and_chains_error() -> None:
    event_bus = EventBus()
    events = capture_events(event_bus)
    service = create_service(generator=FailingGenerator())
    runtime = create_runtime(event_bus=event_bus, service=service)
    coordinate = ChunkCoordinate(x=-7, y=12)
    runtime.declare(coordinate)

    with pytest.raises(TerrainRuntimeGenerationError) as caught:
        runtime.generate(coordinate)

    assert caught.value.__cause__ is not None
    assert runtime.metadata_at(coordinate).state is ChunkState.FAILED
    assert runtime.revision == 3
    assert not runtime.service.contains(coordinate)
    published = dispatch(event_bus, events)
    assert [type(event) for event in published] == [
        TerrainChunkDeclared,
        TerrainChunkGenerationStarted,
        TerrainChunkFailed,
    ]
    failure = cast(TerrainChunkFailed, published[-1])
    assert failure.previous_state is ChunkState.GENERATING
    assert failure.current_state is ChunkState.FAILED
    assert failure.runtime_revision == 3
    assert failure.error_type_name == "TerrainGenerationServiceError"
    with pytest.raises(TerrainRuntimeInvalidOperationError):
        runtime.generate(coordinate)
    with pytest.raises(TerrainRuntimeInvalidOperationError):
        runtime.get_or_generate(coordinate)


class FailingEvictionRepository(InMemoryTerrainRepository):
    def remove(self, coordinate: ChunkCoordinate) -> NoReturn:
        del coordinate
        raise RuntimeError("eviction exploded")


def test_unload_eviction_failure_preserves_ready_metadata_and_cached_terrain() -> None:
    base_service = create_service()
    repository = FailingEvictionRepository(scope=base_service.repository.scope)
    service = create_service(repository=repository)
    runtime = create_runtime(service=service)
    coordinate = ChunkCoordinate(x=5, y=6)
    terrain = runtime.get_or_generate(coordinate)
    revision = runtime.revision

    with pytest.raises(TerrainRuntimeRepositoryError) as caught:
        runtime.unload(coordinate)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert runtime.metadata_at(coordinate).state is ChunkState.READY
    assert runtime.terrain_at(coordinate) is terrain
    assert runtime.revision == revision
    assert service.snapshot().evictions == 0


@pytest.mark.parametrize(
    "state",
    [
        ChunkState.DECLARED,
        ChunkState.GENERATING,
        ChunkState.READY,
        ChunkState.ACTIVE,
        ChunkState.SUSPENDED,
        ChunkState.UNLOADED,
    ],
)
def test_explicit_failure_from_every_non_terminal_state(state: ChunkState) -> None:
    runtime = create_runtime()
    coordinate = ChunkCoordinate(x=state.value.__len__(), y=0)
    metadata = ChunkMetadata.create(world_seed=SPECIFICATION.seed, coordinate=coordinate)
    paths = {
        ChunkState.DECLARED: (),
        ChunkState.GENERATING: (ChunkState.GENERATING,),
        ChunkState.READY: (ChunkState.GENERATING, ChunkState.READY),
        ChunkState.ACTIVE: (ChunkState.GENERATING, ChunkState.READY, ChunkState.ACTIVE),
        ChunkState.SUSPENDED: (
            ChunkState.GENERATING,
            ChunkState.READY,
            ChunkState.ACTIVE,
            ChunkState.SUSPENDED,
        ),
        ChunkState.UNLOADED: (
            ChunkState.GENERATING,
            ChunkState.READY,
            ChunkState.UNLOADED,
        ),
    }
    for transition in paths[state]:
        metadata = metadata.transition_to(transition)
    runtime._metadata[coordinate] = metadata  # type: ignore[attr-defined]

    failed = runtime.fail(coordinate)
    revision = runtime.revision
    duplicate = runtime.fail(coordinate)

    assert failed.state is ChunkState.FAILED
    assert duplicate is failed
    assert runtime.revision == revision == 1


def test_fail_does_not_evict_cached_terrain() -> None:
    runtime = create_runtime()
    coordinate = ChunkCoordinate(x=10, y=10)
    terrain = runtime.get_or_generate(coordinate)

    runtime.fail(coordinate)

    assert runtime.metadata_at(coordinate).state is ChunkState.FAILED
    assert runtime.terrain_at(coordinate) is terrain


def test_coordinates_and_snapshot_are_deterministic_and_non_mutating() -> None:
    runtime = create_runtime()
    coordinates = (
        ChunkCoordinate(x=10**100, y=0),
        ChunkCoordinate(x=-2, y=1),
        ChunkCoordinate(x=5, y=-1),
        ChunkCoordinate(x=-(10**100), y=0),
    )
    for coordinate in coordinates:
        runtime.declare(coordinate)
    runtime.fail(coordinates[1])
    revision = runtime.revision
    service_snapshot = runtime.service.snapshot()

    snapshot = runtime.snapshot()

    assert snapshot.coordinates == (
        ChunkCoordinate(x=5, y=-1),
        ChunkCoordinate(x=-(10**100), y=0),
        ChunkCoordinate(x=10**100, y=0),
        ChunkCoordinate(x=-2, y=1),
    )
    assert snapshot.tracked_chunk_count == 4
    assert snapshot.state_counts[ChunkState.DECLARED] == 3
    assert snapshot.state_counts[ChunkState.FAILED] == 1
    assert sum(snapshot.state_counts.values()) == 4
    assert tuple(snapshot.state_counts) == tuple(ChunkState)
    assert snapshot.terrain_service == service_snapshot
    assert runtime.revision == revision
    assert runtime.service.snapshot() == service_snapshot
    with pytest.raises(TypeError):
        snapshot.state_counts[ChunkState.READY] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type", "message"),
    [
        ("revision", True, TypeError, "revision must be"),
        ("revision", -1, ValueError, "revision must be greater"),
        ("tracked_chunk_count", "0", TypeError, "tracked_chunk_count must be"),
        ("tracked_chunk_count", -1, ValueError, "tracked_chunk_count must be greater"),
        ("coordinates", [], TypeError, "coordinates must be a tuple"),
        ("coordinates", (object(),), TypeError, "coordinates must be a tuple"),
        ("tracked_chunk_count", 1, ValueError, "coordinate count must match"),
        ("state_counts", {}, ValueError, "every ChunkState"),
        (
            "state_counts",
            {state: (True if state is ChunkState.DECLARED else 0) for state in ChunkState},
            TypeError,
            "values must be integers",
        ),
        (
            "state_counts",
            {state: (-1 if state is ChunkState.DECLARED else 0) for state in ChunkState},
            ValueError,
            "values must be greater",
        ),
        (
            "state_counts",
            {state: (1 if state is ChunkState.DECLARED else 0) for state in ChunkState},
            ValueError,
            "state count sum",
        ),
        ("terrain_service", object(), TypeError, "terrain_service must be"),
    ],
)
def test_runtime_snapshot_validation(
    field_name: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {
        "revision": 0,
        "tracked_chunk_count": 0,
        "coordinates": (),
        "state_counts": dict.fromkeys(ChunkState, 0),
        "terrain_service": create_service().snapshot(),
        field_name: value,
    }
    with pytest.raises(error_type, match=message):
        TerrainRuntimeSnapshot(**cast(Any, values))


def test_snapshot_and_events_are_immutable() -> None:
    runtime = create_runtime()
    snapshot = runtime.snapshot()
    event = TerrainChunkFailed(
        coordinate=ChunkCoordinate(x=0, y=0),
        previous_state=ChunkState.DECLARED,
        current_state=ChunkState.FAILED,
        runtime_revision=1,
        error_type_name="ExplicitTerrainFailure",
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.revision = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        event.runtime_revision = 2  # type: ignore[misc]


def test_no_event_bus_is_fully_supported() -> None:
    runtime = create_runtime()
    coordinate = ChunkCoordinate(x=2, y=2)

    runtime.declare(coordinate)
    runtime.generate(coordinate)
    runtime.activate(coordinate)
    runtime.suspend(coordinate)
    runtime.unload(coordinate)
    runtime.fail(coordinate)

    assert runtime.metadata_at(coordinate).state is ChunkState.FAILED


def test_no_events_for_no_ops_or_failed_validation() -> None:
    event_bus = EventBus()
    runtime = create_runtime(event_bus=event_bus)
    coordinate = ChunkCoordinate(x=0, y=0)
    runtime.declare(coordinate)
    event_bus.clear_pending()

    runtime.declare(coordinate)
    with pytest.raises(TerrainRuntimeInvalidOperationError):
        runtime.activate(coordinate)

    assert event_bus.pending_event_count == 0


class RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_structured_diagnostics_cover_successful_lifecycle_in_order() -> None:
    logger = logging.Logger("test.terrain.runtime", level=logging.DEBUG)
    handler = RecordingHandler()
    logger.addHandler(handler)
    runtime = TerrainRuntime(
        specification=SPECIFICATION,
        service=create_service(),
        logger=logger,
    )
    coordinate = ChunkCoordinate(x=-17, y=16)

    runtime.get_or_generate(coordinate)
    runtime.activate(coordinate)
    runtime.suspend(coordinate)
    runtime.unload(coordinate)
    runtime.fail(coordinate)

    assert [record.event for record in handler.records] == [
        "terrain.chunk_declared",
        "terrain.generation_started",
        "terrain.chunk_generated",
        "terrain.chunk_activated",
        "terrain.chunk_suspended",
        "terrain.chunk_unloaded",
        "terrain.chunk_failed",
    ]
    generated = handler.records[2]
    assert generated.chunk_x == -17
    assert generated.chunk_y == 16
    assert generated.region_x == -2
    assert generated.region_y == 1
    assert generated.chunk_state == "ready"
    assert generated.previous_chunk_state == "generating"
    assert generated.terrain_runtime_revision == 3
    assert generated.terrain_repository_revision == 1
    assert generated.terrain_seed == runtime.metadata_at(coordinate).terrain_seed
    assert generated.terrain_min_elevation <= generated.terrain_max_elevation
    assert generated.terrain_tile_count == 256
    assert generated.terrain_cache_hits == 0
    assert generated.terrain_cache_misses == 0
    assert generated.terrain_successful_generations == 1
    assert generated.terrain_failed_generations == 0
    assert generated.terrain_evictions == 0
    declared = handler.records[0]
    assert not hasattr(declared, "previous_chunk_state")
    assert not hasattr(declared, "terrain_min_elevation")


def test_failure_diagnostic_occurs_after_failed_transition_and_no_ops_do_not_log() -> None:
    logger = logging.Logger("test.terrain.failure", level=logging.DEBUG)
    handler = RecordingHandler()
    logger.addHandler(handler)
    runtime = create_runtime(
        service=create_service(generator=FailingGenerator()),
        logger=logger,
    )
    coordinate = ChunkCoordinate(x=1, y=1)

    runtime.declare(coordinate)
    runtime.declare(coordinate)
    with pytest.raises(TerrainRuntimeGenerationError):
        runtime.generate(coordinate)
    runtime.fail(coordinate)

    assert [record.event for record in handler.records] == [
        "terrain.chunk_declared",
        "terrain.generation_started",
        "terrain.chunk_failed",
    ]
    assert handler.records[-1].chunk_state == "failed"
    assert handler.records[-1].terrain_runtime_revision == 3
    assert handler.records[-1].terrain_failed_generations == 1


def test_runtime_error_hierarchy_and_public_exports() -> None:
    assert issubclass(TerrainRuntimeMissingChunkError, TerrainRuntimeError)
    assert issubclass(TerrainRuntimeInvalidOperationError, TerrainRuntimeError)
    assert issubclass(TerrainGenerationInProgressError, TerrainRuntimeInvalidOperationError)
    assert issubclass(IncompatibleTerrainRuntimeError, TerrainRuntimeError)
    assert issubclass(TerrainUnavailableError, TerrainRuntimeError)
    assert issubclass(TerrainRuntimeGenerationError, TerrainRuntimeError)
    assert issubclass(TerrainRuntimeRepositoryError, TerrainRuntimeError)
    names = {
        "IncompatibleTerrainRuntimeError",
        "TerrainChunkActivated",
        "TerrainChunkDeclared",
        "TerrainChunkFailed",
        "TerrainChunkGenerated",
        "TerrainChunkGenerationStarted",
        "TerrainChunkSuspended",
        "TerrainChunkUnloaded",
        "TerrainGenerationInProgressError",
        "TerrainRuntime",
        "TerrainRuntimeError",
        "TerrainRuntimeGenerationError",
        "TerrainRuntimeInvalidOperationError",
        "TerrainRuntimeMissingChunkError",
        "TerrainRuntimeRepositoryError",
        "TerrainRuntimeSnapshot",
        "TerrainUnavailableError",
    }
    assert names <= set(world.__all__)
    assert all(hasattr(world, name) for name in names)
