"""Tests for immutable world identity, metadata, and lifecycle state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast
from uuid import UUID

import pytest

from open_world_rpg.core.config import MAX_WORLD_SEED
from open_world_rpg.world import (
    MAX_WORLD_NAME_LENGTH,
    WorldId,
    WorldMetadata,
    WorldState,
    WorldTransitionError,
)

WORLD_UUID = UUID("12345678-1234-5678-1234-567812345678")
CREATED_AT = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)


def create_metadata(
    *,
    state: WorldState = WorldState.CREATED,
) -> WorldMetadata:
    return WorldMetadata(
        world_id=WorldId(value=WORLD_UUID),
        name="Test World",
        seed=42,
        created_at=CREATED_AT,
        state=state,
    )


def test_world_id_creation_generates_uuid_identity() -> None:
    first = WorldId.create()
    second = WorldId.create()

    assert isinstance(first.value, UUID)
    assert first != second


def test_world_id_parsing_and_string_conversion() -> None:
    world_id = WorldId.parse(str(WORLD_UUID).upper())

    assert world_id == WorldId(value=WORLD_UUID)
    assert str(world_id) == str(WORLD_UUID)
    assert hash(world_id) == hash(WorldId(value=WORLD_UUID))


@pytest.mark.parametrize(
    ("operation", "error_type", "message"),
    [
        (
            lambda: WorldId(value=cast(Any, "not-a-uuid")),
            TypeError,
            "value must be a UUID",
        ),
        (
            lambda: WorldId.parse(cast(Any, 100)),
            TypeError,
            "value must be a string",
        ),
        (
            lambda: WorldId.parse("not-a-uuid"),
            ValueError,
            "value must be a valid UUID",
        ),
    ],
)
def test_world_id_rejects_invalid_uuid_values(
    operation: Callable[[], object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        operation()


def test_world_id_is_immutable() -> None:
    world_id = WorldId(value=WORLD_UUID)

    with pytest.raises(FrozenInstanceError):
        world_id.value = UUID(int=0)  # type: ignore[misc]


def test_world_state_values() -> None:
    assert WorldState.CREATED.value == "created"
    assert WorldState.INITIALISED.value == "initialised"
    assert WorldState.ACTIVE.value == "active"
    assert WorldState.PAUSED.value == "paused"
    assert WorldState.CLOSED.value == "closed"
    assert WorldState.FAILED.value == "failed"


def test_metadata_normalises_name_and_timezone() -> None:
    local_time = datetime(
        2026,
        7,
        25,
        11,
        0,
        tzinfo=timezone(timedelta(hours=1)),
    )
    metadata = WorldMetadata(
        world_id=WorldId(value=WORLD_UUID),
        name="  The Northern Reach  ",
        seed=MAX_WORLD_SEED,
        created_at=local_time,
    )

    assert metadata.name == "The Northern Reach"
    assert metadata.seed == MAX_WORLD_SEED
    assert metadata.created_at == CREATED_AT
    assert metadata.created_at.tzinfo is UTC
    assert metadata.state is WorldState.CREATED


def test_metadata_is_value_comparable_and_immutable() -> None:
    metadata = create_metadata()

    assert metadata == create_metadata()
    assert hash(metadata) == hash(create_metadata())

    with pytest.raises(FrozenInstanceError):
        metadata.name = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type", "message"),
    [
        ("world_id", WORLD_UUID, TypeError, "world_id must be a WorldId"),
        ("name", 100, TypeError, "name must be a string"),
        ("name", "", ValueError, "name cannot be empty"),
        ("name", "   ", ValueError, "name cannot be empty"),
        (
            "name",
            "x" * (MAX_WORLD_NAME_LENGTH + 1),
            ValueError,
            "name cannot exceed",
        ),
        ("seed", True, TypeError, "seed must be an integer"),
        ("seed", False, TypeError, "seed must be an integer"),
        ("seed", "42", TypeError, "seed must be an integer"),
        ("seed", -1, ValueError, "seed must be between"),
        ("seed", MAX_WORLD_SEED + 1, ValueError, "seed must be between"),
        ("created_at", "today", TypeError, "created_at must be a datetime"),
        (
            "created_at",
            datetime(2026, 7, 25, 10, 0),
            ValueError,
            "created_at must be timezone-aware",
        ),
        ("state", "created", TypeError, "state must be a WorldState"),
    ],
)
def test_metadata_rejects_invalid_values(
    field_name: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    values: dict[str, Any] = {
        "world_id": WorldId(value=WORLD_UUID),
        "name": "Test World",
        "seed": 42,
        "created_at": CREATED_AT,
        "state": WorldState.CREATED,
    }
    values[field_name] = value

    with pytest.raises(error_type, match=message):
        WorldMetadata(**values)


Transition = Callable[[WorldMetadata], WorldMetadata]


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
def test_every_valid_state_transition_returns_new_metadata(
    start: WorldState,
    operation: Transition,
    expected: WorldState,
) -> None:
    original = create_metadata(state=start)

    transitioned = operation(original)

    assert transitioned is not original
    assert transitioned.state is expected
    assert transitioned.world_id == original.world_id
    assert transitioned.name == original.name
    assert transitioned.seed == original.seed
    assert transitioned.created_at == original.created_at
    assert original.state is start


OPERATIONS: tuple[tuple[str, Transition, frozenset[WorldState]], ...] = (
    (
        "initialise",
        lambda world: world.initialise(),
        frozenset({WorldState.CREATED}),
    ),
    (
        "activate",
        lambda world: world.activate(),
        frozenset({WorldState.INITIALISED}),
    ),
    (
        "pause",
        lambda world: world.pause(),
        frozenset({WorldState.ACTIVE}),
    ),
    (
        "resume",
        lambda world: world.resume(),
        frozenset({WorldState.PAUSED}),
    ),
    (
        "close",
        lambda world: world.close(),
        frozenset({WorldState.ACTIVE, WorldState.PAUSED}),
    ),
    (
        "fail",
        lambda world: world.fail(),
        frozenset(
            {
                WorldState.CREATED,
                WorldState.INITIALISED,
                WorldState.ACTIVE,
                WorldState.PAUSED,
            }
        ),
    ),
)

INVALID_TRANSITIONS = [
    (state, name, operation)
    for name, operation, allowed in OPERATIONS
    for state in WorldState
    if state not in allowed
]


@pytest.mark.parametrize(
    ("state", "operation_name", "operation"),
    INVALID_TRANSITIONS,
)
def test_every_invalid_state_transition_is_rejected(
    state: WorldState,
    operation_name: str,
    operation: Transition,
) -> None:
    metadata = create_metadata(state=state)

    with pytest.raises(
        WorldTransitionError,
        match=rf"Cannot {operation_name} world",
    ):
        operation(metadata)
