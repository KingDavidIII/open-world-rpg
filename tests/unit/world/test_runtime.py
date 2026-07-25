"""Tests for the controlled mutable world runtime boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import pytest

from open_world_rpg.engine import EventBus
from open_world_rpg.world import (
    InvalidWorldTimeOperationError,
    WorldClock,
    WorldId,
    WorldInstant,
    WorldMetadata,
    WorldModel,
    WorldRuntime,
    WorldRuntimeSnapshot,
    WorldSeed,
    WorldSnapshot,
    WorldSpecification,
    WorldState,
    WorldStateChanged,
    WorldTimeAdvanced,
    WorldTransitionError,
)

WORLD_ID = WorldId(value=UUID("12345678-1234-5678-1234-567812345678"))
CREATED_AT = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)


def create_model(
    *,
    state: WorldState = WorldState.CREATED,
    tick: int = 0,
) -> WorldModel:
    specification = WorldSpecification(
        name="Runtime World",
        seed=WorldSeed(value=42),
    )
    return WorldModel(
        metadata=WorldMetadata(
            world_id=WORLD_ID,
            name=specification.name,
            seed=specification.seed.value,
            created_at=CREATED_AT,
            state=state,
        ),
        specification=specification,
        clock=WorldClock(
            config=specification.time_config,
            current=WorldInstant(tick=tick),
        ),
    )


def test_runtime_validates_construction_and_exposes_initial_snapshot() -> None:
    model = create_model()
    runtime = WorldRuntime(model=model)

    assert runtime.model is model
    assert runtime.revision == 0
    assert runtime.snapshot == WorldRuntimeSnapshot(
        revision=0,
        world=model.snapshot(),
    )

    with pytest.raises(TypeError, match="model must be a WorldModel"):
        WorldRuntime(model=cast(Any, object()))

    with pytest.raises(TypeError, match="event_bus must be an EventBus or None"):
        WorldRuntime(model=model, event_bus=cast(Any, object()))


def test_runtime_without_event_bus_supports_every_successful_operation() -> None:
    runtime = WorldRuntime(model=create_model())

    runtime.initialise()
    runtime.activate()
    runtime.advance_tick()
    runtime.advance_ticks(4)
    runtime.pause()
    runtime.resume()
    runtime.close()

    assert runtime.model.metadata.state is WorldState.CLOSED
    assert runtime.model.clock.current.tick == 5
    assert runtime.revision == 7


def test_runtime_fail_operation_replaces_model_once() -> None:
    runtime = WorldRuntime(model=create_model())
    original = runtime.model

    runtime.fail()

    assert runtime.model is not original
    assert runtime.model.metadata.state is WorldState.FAILED
    assert runtime.revision == 1


def test_state_events_are_published_after_success_in_fifo_order() -> None:
    event_bus = EventBus()
    runtime = WorldRuntime(model=create_model(), event_bus=event_bus)
    events: list[WorldStateChanged] = []
    event_bus.subscribe(WorldStateChanged, events.append)

    runtime.initialise()
    runtime.activate()
    runtime.pause()
    runtime.resume()
    runtime.close()

    assert event_bus.pending_event_count == 5
    event_bus.dispatch_pending()

    assert [(event.previous_state, event.current_state, event.revision) for event in events] == [
        (WorldState.CREATED, WorldState.INITIALISED, 1),
        (WorldState.INITIALISED, WorldState.ACTIVE, 2),
        (WorldState.ACTIVE, WorldState.PAUSED, 3),
        (WorldState.PAUSED, WorldState.ACTIVE, 4),
        (WorldState.ACTIVE, WorldState.CLOSED, 5),
    ]
    assert all(event.world_id is WORLD_ID for event in events)
    assert runtime.revision == 5


def test_time_events_publish_exact_tick_changes() -> None:
    event_bus = EventBus()
    runtime = WorldRuntime(
        model=create_model(state=WorldState.ACTIVE, tick=10),
        event_bus=event_bus,
    )
    events: list[WorldTimeAdvanced] = []
    event_bus.subscribe(WorldTimeAdvanced, events.append)

    runtime.advance_tick()
    runtime.advance_ticks(9)
    event_bus.dispatch_pending()

    assert events == [
        WorldTimeAdvanced(
            world_id=WORLD_ID,
            previous_tick=10,
            current_tick=11,
            advanced_ticks=1,
            revision=1,
        ),
        WorldTimeAdvanced(
            world_id=WORLD_ID,
            previous_tick=11,
            current_tick=20,
            advanced_ticks=9,
            revision=2,
        ),
    ]


def test_no_op_advancement_and_reset_do_not_change_revision_or_publish() -> None:
    event_bus = EventBus()
    active = WorldRuntime(
        model=create_model(state=WorldState.ACTIVE, tick=10),
        event_bus=event_bus,
    )
    created = WorldRuntime(
        model=create_model(tick=10),
        event_bus=event_bus,
    )

    active.advance_ticks(0)
    created.reset_clock(instant=WorldInstant(tick=10))

    assert active.revision == 0
    assert created.revision == 0
    assert event_bus.pending_event_count == 0


def test_reset_clock_changes_model_and_revision_without_lifecycle_event() -> None:
    event_bus = EventBus()
    runtime = WorldRuntime(model=create_model(tick=50), event_bus=event_bus)

    runtime.reset_clock()

    assert runtime.model.clock.current == WorldInstant()
    assert runtime.revision == 1
    assert event_bus.pending_event_count == 0


@pytest.mark.parametrize("value", [True, False, 1.5, "1", None])
def test_advance_ticks_rejects_invalid_types_without_changes(value: object) -> None:
    runtime = WorldRuntime(model=create_model(state=WorldState.ACTIVE, tick=5))
    original = runtime.model

    with pytest.raises(TypeError, match="ticks must be an integer"):
        runtime.advance_ticks(cast(Any, value))

    assert runtime.model is original
    assert runtime.revision == 0


def test_advance_ticks_rejects_negative_value_without_changes() -> None:
    runtime = WorldRuntime(model=create_model(state=WorldState.ACTIVE, tick=5))
    original = runtime.model

    with pytest.raises(ValueError, match="ticks must be greater than or equal to zero"):
        runtime.advance_ticks(-1)

    assert runtime.model is original
    assert runtime.revision == 0


def test_failed_operations_preserve_model_revision_and_event_queue() -> None:
    event_bus = EventBus()
    runtime = WorldRuntime(model=create_model(), event_bus=event_bus)
    original = runtime.model

    with pytest.raises(WorldTransitionError, match="Cannot activate world"):
        runtime.activate()

    with pytest.raises(InvalidWorldTimeOperationError, match="Cannot advance time"):
        runtime.advance_tick()

    with pytest.raises(TypeError, match="instant must be a WorldInstant"):
        runtime.reset_clock(instant=cast(Any, 10))

    assert runtime.model is original
    assert runtime.revision == 0
    assert event_bus.pending_event_count == 0


def test_runtime_event_and_snapshot_payloads_are_immutable() -> None:
    runtime_snapshot = WorldRuntime(model=create_model()).snapshot
    state_event = WorldStateChanged(
        world_id=WORLD_ID,
        previous_state=WorldState.CREATED,
        current_state=WorldState.INITIALISED,
        revision=1,
    )
    time_event = WorldTimeAdvanced(
        world_id=WORLD_ID,
        previous_tick=0,
        current_tick=1,
        advanced_ticks=1,
        revision=1,
    )

    assert isinstance(runtime_snapshot.world, WorldSnapshot)

    with pytest.raises(FrozenInstanceError):
        runtime_snapshot.revision = 2  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        state_event.revision = 2  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        time_event.revision = 2  # type: ignore[misc]
