"""Tests for the immutable deterministic world aggregate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import pytest

from open_world_rpg.world import (
    CHUNK_SIZE,
    REGION_SIZE_IN_CHUNKS,
    REGION_SIZE_IN_TILES,
    SUPPORTED_GENERATION_FORMAT_VERSION,
    InconsistentWorldModelError,
    InvalidWorldTimeOperationError,
    UnsupportedWorldSpecificationError,
    WorldClock,
    WorldDateTime,
    WorldId,
    WorldInstant,
    WorldMetadata,
    WorldModel,
    WorldModelError,
    WorldSeed,
    WorldSnapshot,
    WorldSpecification,
    WorldState,
    WorldTimeConfig,
    WorldTransitionError,
)

WORLD_ID = WorldId(value=UUID("12345678-1234-5678-1234-567812345678"))
CREATED_AT = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)


def create_specification(
    *,
    seed: int = 42,
    time_config: WorldTimeConfig | None = None,
) -> WorldSpecification:
    return WorldSpecification(
        name="Test World",
        seed=WorldSeed(value=seed),
        time_config=WorldTimeConfig() if time_config is None else time_config,
    )


def create_model(
    *,
    state: WorldState = WorldState.CREATED,
    tick: int = 0,
    specification: WorldSpecification | None = None,
) -> WorldModel:
    resolved_specification = create_specification() if specification is None else specification
    return WorldModel(
        metadata=WorldMetadata(
            world_id=WORLD_ID,
            name=resolved_specification.name,
            seed=resolved_specification.seed.value,
            created_at=CREATED_AT,
            state=state,
        ),
        specification=resolved_specification,
        clock=WorldClock(
            config=resolved_specification.time_config,
            current=WorldInstant(tick=tick),
        ),
    )


def test_default_specification_defines_compatibility_rules() -> None:
    specification = WorldSpecification(
        name="  Test World  ",
        seed=WorldSeed(value=42),
    )

    assert specification.name == "Test World"
    assert specification.seed == WorldSeed(value=42)
    assert specification.time_config == WorldTimeConfig()
    assert specification.chunk_size_tiles == CHUNK_SIZE
    assert specification.region_size_chunks == REGION_SIZE_IN_CHUNKS
    assert specification.tiles_per_region_axis == REGION_SIZE_IN_TILES
    assert specification.generation_format_version == SUPPORTED_GENERATION_FORMAT_VERSION == "v1"


def test_specification_accepts_custom_time_configuration() -> None:
    time_config = WorldTimeConfig(
        ticks_per_second=5,
        seconds_per_minute=6,
        minutes_per_hour=7,
        hours_per_day=8,
        days_per_year=9,
    )

    specification = create_specification(time_config=time_config)

    assert specification.time_config is time_config


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("chunk_size_tiles", CHUNK_SIZE - 1, "chunk_size_tiles"),
        ("chunk_size_tiles", CHUNK_SIZE + 1, "chunk_size_tiles"),
        ("chunk_size_tiles", True, "chunk_size_tiles"),
        ("chunk_size_tiles", "16", "chunk_size_tiles"),
        ("region_size_chunks", REGION_SIZE_IN_CHUNKS - 1, "region_size_chunks"),
        ("region_size_chunks", REGION_SIZE_IN_CHUNKS + 1, "region_size_chunks"),
        ("region_size_chunks", False, "region_size_chunks"),
        ("region_size_chunks", 16.0, "region_size_chunks"),
        ("generation_format_version", "v0", "generation_format_version"),
        ("generation_format_version", "v2", "generation_format_version"),
        ("generation_format_version", b"v1", "generation_format_version"),
    ],
)
def test_specification_rejects_unsupported_compatibility_values(
    field_name: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, Any] = {
        "name": "Test World",
        "seed": WorldSeed(value=42),
    }
    values[field_name] = value

    with pytest.raises(UnsupportedWorldSpecificationError, match=message):
        WorldSpecification(**values)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("seed", 42, "seed must be a WorldSeed"),
        ("time_config", object(), "time_config must be a WorldTimeConfig"),
    ],
)
def test_specification_rejects_invalid_component_types(
    field_name: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, Any] = {
        "name": "Test World",
        "seed": WorldSeed(value=42),
    }
    values[field_name] = value

    with pytest.raises(TypeError, match=message):
        WorldSpecification(**values)


def test_specification_reuses_strict_world_name_validation() -> None:
    with pytest.raises(ValueError, match="name cannot be empty"):
        WorldSpecification(name="   ", seed=WorldSeed(value=42))


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("metadata", object(), "metadata must be WorldMetadata"),
        ("specification", object(), "specification must be WorldSpecification"),
        ("clock", object(), "clock must be WorldClock"),
    ],
)
def test_world_model_rejects_invalid_component_types(
    field_name: str,
    value: object,
    message: str,
) -> None:
    model = create_model()
    values: dict[str, Any] = {
        "metadata": model.metadata,
        "specification": model.specification,
        "clock": model.clock,
    }
    values[field_name] = value

    with pytest.raises(InconsistentWorldModelError, match=message):
        WorldModel(**values)


def test_world_model_rejects_seed_inconsistency() -> None:
    model = create_model()
    inconsistent_metadata = WorldMetadata(
        world_id=WORLD_ID,
        name="Test World",
        seed=99,
        created_at=CREATED_AT,
    )

    with pytest.raises(InconsistentWorldModelError, match="seed must match"):
        WorldModel(
            metadata=inconsistent_metadata,
            specification=model.specification,
            clock=model.clock,
        )


def test_world_model_rejects_name_inconsistency() -> None:
    model = create_model()
    inconsistent_metadata = WorldMetadata(
        world_id=WORLD_ID,
        name="Different World",
        seed=model.specification.seed.value,
        created_at=CREATED_AT,
    )

    with pytest.raises(InconsistentWorldModelError, match="name must match"):
        WorldModel(
            metadata=inconsistent_metadata,
            specification=model.specification,
            clock=model.clock,
        )


def test_world_model_rejects_clock_configuration_inconsistency() -> None:
    model = create_model()

    with pytest.raises(
        InconsistentWorldModelError,
        match="clock configuration must match",
    ):
        WorldModel(
            metadata=model.metadata,
            specification=model.specification,
            clock=WorldClock(
                config=WorldTimeConfig(ticks_per_second=1),
            ),
        )


def test_world_model_create_builds_consistent_zero_tick_world() -> None:
    specification = create_specification()

    model = WorldModel.create(
        specification=specification,
        created_at=CREATED_AT,
        world_id=WORLD_ID,
    )

    assert model.metadata.world_id is WORLD_ID
    assert model.metadata.name == specification.name
    assert model.metadata.seed == specification.seed.value
    assert model.metadata.created_at == CREATED_AT
    assert model.metadata.state is WorldState.CREATED
    assert model.specification is specification
    assert model.clock.config == specification.time_config
    assert model.clock.current == WorldInstant()


def test_world_model_create_generates_default_identity() -> None:
    model = WorldModel.create(
        specification=create_specification(),
        created_at=CREATED_AT,
    )

    assert isinstance(model.metadata.world_id, WorldId)
    assert model.metadata.world_id != WORLD_ID


def test_world_model_create_rejects_invalid_specification() -> None:
    with pytest.raises(
        InconsistentWorldModelError,
        match="specification must be WorldSpecification",
    ):
        WorldModel.create(
            specification=cast(Any, object()),
            created_at=CREATED_AT,
        )


LifecycleOperation = Callable[[WorldModel], WorldModel]


@pytest.mark.parametrize(
    ("start", "operation", "expected"),
    [
        (WorldState.CREATED, lambda world: world.initialise(), WorldState.INITIALISED),
        (WorldState.INITIALISED, lambda world: world.activate(), WorldState.ACTIVE),
        (WorldState.ACTIVE, lambda world: world.pause(), WorldState.PAUSED),
        (WorldState.PAUSED, lambda world: world.resume(), WorldState.ACTIVE),
        (WorldState.ACTIVE, lambda world: world.close(), WorldState.CLOSED),
        (WorldState.PAUSED, lambda world: world.close(), WorldState.CLOSED),
        (WorldState.CREATED, lambda world: world.fail(), WorldState.FAILED),
        (WorldState.INITIALISED, lambda world: world.fail(), WorldState.FAILED),
        (WorldState.ACTIVE, lambda world: world.fail(), WorldState.FAILED),
        (WorldState.PAUSED, lambda world: world.fail(), WorldState.FAILED),
    ],
)
def test_every_lifecycle_operation_delegates_to_metadata(
    start: WorldState,
    operation: LifecycleOperation,
    expected: WorldState,
) -> None:
    original = create_model(state=start, tick=123)

    transitioned = operation(original)

    assert transitioned.metadata.state is expected
    assert transitioned.metadata.world_id is original.metadata.world_id
    assert transitioned.metadata.created_at is original.metadata.created_at
    assert transitioned.specification is original.specification
    assert transitioned.clock is original.clock
    assert original.metadata.state is start


def test_invalid_lifecycle_operation_leaves_original_unchanged() -> None:
    original = create_model(state=WorldState.CREATED, tick=123)

    with pytest.raises(WorldTransitionError, match="Cannot pause world"):
        original.pause()

    assert original == create_model(state=WorldState.CREATED, tick=123)


def test_active_world_advances_ticks_seconds_and_large_values() -> None:
    original = create_model(state=WorldState.ACTIVE)
    huge_advance = 10**100

    by_ticks = original.advance_ticks(huge_advance)
    by_seconds = by_ticks.advance_seconds(3)

    assert original.clock.current.tick == 0
    assert by_ticks.clock.current.tick == huge_advance
    assert by_seconds.clock.current.tick == (
        huge_advance + (3 * original.specification.time_config.ticks_per_second)
    )


@pytest.mark.parametrize(
    "state",
    [
        WorldState.CREATED,
        WorldState.INITIALISED,
        WorldState.PAUSED,
        WorldState.CLOSED,
        WorldState.FAILED,
    ],
)
@pytest.mark.parametrize(
    "operation",
    [
        lambda world: world.advance_ticks(1),
        lambda world: world.advance_seconds(1),
    ],
)
def test_time_advancement_is_rejected_in_every_non_active_state(
    state: WorldState,
    operation: LifecycleOperation,
) -> None:
    original = create_model(state=state, tick=50)

    with pytest.raises(
        InvalidWorldTimeOperationError,
        match="Cannot advance time",
    ):
        operation(original)

    assert original.clock.current.tick == 50
    assert original.metadata.state is state


@pytest.mark.parametrize("state", [WorldState.CREATED, WorldState.INITIALISED])
def test_clock_reset_is_valid_before_activation(state: WorldState) -> None:
    original = create_model(state=state, tick=500)

    zeroed = original.reset_clock()
    repositioned = original.reset_clock(instant=WorldInstant(tick=75))

    assert zeroed.clock.current == WorldInstant()
    assert repositioned.clock.current == WorldInstant(tick=75)
    assert original.clock.current == WorldInstant(tick=500)


@pytest.mark.parametrize(
    "state",
    [
        WorldState.ACTIVE,
        WorldState.PAUSED,
        WorldState.CLOSED,
        WorldState.FAILED,
    ],
)
def test_clock_reset_is_rejected_after_activation(state: WorldState) -> None:
    original = create_model(state=state, tick=500)

    with pytest.raises(
        InvalidWorldTimeOperationError,
        match="Cannot reset clock",
    ):
        original.reset_clock()

    assert original.clock.current == WorldInstant(tick=500)


def test_valid_time_state_still_delegates_advance_and_reset_validation() -> None:
    active = create_model(state=WorldState.ACTIVE)
    created = create_model()

    with pytest.raises(TypeError, match="ticks must be an integer"):
        active.advance_ticks(cast(Any, True))

    with pytest.raises(TypeError, match="seconds must be an integer"):
        active.advance_seconds(cast(Any, False))

    with pytest.raises(TypeError, match="instant must be a WorldInstant"):
        created.reset_clock(instant=cast(Any, 10))

    assert active.clock.current == WorldInstant()
    assert created.clock.current == WorldInstant()


def test_snapshot_contains_complete_projection() -> None:
    model = create_model(state=WorldState.ACTIVE, tick=5_184_060)

    snapshot = model.snapshot()

    assert snapshot == WorldSnapshot(
        world_id=WORLD_ID,
        name="Test World",
        seed=WorldSeed(value=42),
        state=WorldState.ACTIVE,
        created_at=CREATED_AT,
        absolute_world_tick=5_184_060,
        date_time=WorldDateTime(
            instant=WorldInstant(tick=5_184_060),
            config=WorldTimeConfig(),
        ),
        chunk_size_tiles=CHUNK_SIZE,
        region_size_chunks=REGION_SIZE_IN_CHUNKS,
        generation_format_version=SUPPORTED_GENERATION_FORMAT_VERSION,
    )
    assert snapshot.date_time.day_of_year == 2
    assert snapshot.date_time.second == 1


def test_aggregate_values_are_immutable_and_errors_share_base_type() -> None:
    specification = create_specification()
    model = create_model(specification=specification)
    snapshot = model.snapshot()

    assert issubclass(InconsistentWorldModelError, WorldModelError)
    assert issubclass(InvalidWorldTimeOperationError, WorldModelError)
    assert issubclass(UnsupportedWorldSpecificationError, WorldModelError)

    with pytest.raises(FrozenInstanceError):
        specification.name = "Changed"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        model.clock = WorldClock()  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        snapshot.name = "Changed"  # type: ignore[misc]
