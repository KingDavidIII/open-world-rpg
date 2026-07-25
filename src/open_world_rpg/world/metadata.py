"""World identity, descriptive metadata, and lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final
from uuid import UUID, uuid4

from open_world_rpg.core.config import MAX_WORLD_SEED, MIN_WORLD_SEED

MAX_WORLD_NAME_LENGTH: Final = 100


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldId:
    """Strongly typed UUID identity for one world."""

    value: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.value, UUID):
            raise TypeError("value must be a UUID.")

    @classmethod
    def create(cls) -> WorldId:
        """Create a new unique world identity."""
        return cls(value=uuid4())

    @classmethod
    def parse(cls, value: str) -> WorldId:
        """Parse a world identity from its UUID text representation."""
        if not isinstance(value, str):
            raise TypeError("value must be a string.")

        try:
            parsed = UUID(value)
        except ValueError as exc:
            raise ValueError("value must be a valid UUID.") from exc

        return cls(value=parsed)

    def __str__(self) -> str:
        """Return the canonical UUID text representation."""
        return str(self.value)


class WorldState(StrEnum):
    """Lifecycle states for a persisted world."""

    CREATED = "created"
    INITIALISED = "initialised"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    FAILED = "failed"


class WorldTransitionError(RuntimeError):
    """Raised when a world lifecycle transition is not permitted."""


def _normalise_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("name must be a string.")

    normalised = value.strip()
    if not normalised:
        raise ValueError("name cannot be empty.")

    if len(normalised) > MAX_WORLD_NAME_LENGTH:
        raise ValueError(f"name cannot exceed {MAX_WORLD_NAME_LENGTH} characters.")

    return normalised


def _validate_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an integer.")

    if value < MIN_WORLD_SEED or value > MAX_WORLD_SEED:
        raise ValueError(f"seed must be between {MIN_WORLD_SEED} and {MAX_WORLD_SEED}.")

    return value


def _normalise_created_at(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("created_at must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware.")

    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldMetadata:
    """Immutable identity, description, and lifecycle state for one world."""

    world_id: WorldId
    name: str
    seed: int
    created_at: datetime
    state: WorldState = WorldState.CREATED

    def __post_init__(self) -> None:
        if not isinstance(self.world_id, WorldId):
            raise TypeError("world_id must be a WorldId.")

        if not isinstance(self.state, WorldState):
            raise TypeError("state must be a WorldState.")

        object.__setattr__(self, "name", _normalise_name(self.name))
        object.__setattr__(self, "seed", _validate_seed(self.seed))
        object.__setattr__(
            self,
            "created_at",
            _normalise_created_at(self.created_at),
        )

    def initialise(self) -> WorldMetadata:
        """Move a newly created world into its initialised state."""
        return self._transition(
            target=WorldState.INITIALISED,
            allowed=(WorldState.CREATED,),
            operation="initialise",
        )

    def activate(self) -> WorldMetadata:
        """Activate an initialised world."""
        return self._transition(
            target=WorldState.ACTIVE,
            allowed=(WorldState.INITIALISED,),
            operation="activate",
        )

    def pause(self) -> WorldMetadata:
        """Pause an active world."""
        return self._transition(
            target=WorldState.PAUSED,
            allowed=(WorldState.ACTIVE,),
            operation="pause",
        )

    def resume(self) -> WorldMetadata:
        """Resume a paused world."""
        return self._transition(
            target=WorldState.ACTIVE,
            allowed=(WorldState.PAUSED,),
            operation="resume",
        )

    def close(self) -> WorldMetadata:
        """Close an active or paused world."""
        return self._transition(
            target=WorldState.CLOSED,
            allowed=(
                WorldState.ACTIVE,
                WorldState.PAUSED,
            ),
            operation="close",
        )

    def fail(self) -> WorldMetadata:
        """Mark any non-terminal world as failed."""
        return self._transition(
            target=WorldState.FAILED,
            allowed=(
                WorldState.CREATED,
                WorldState.INITIALISED,
                WorldState.ACTIVE,
                WorldState.PAUSED,
            ),
            operation="fail",
        )

    def _transition(
        self,
        *,
        target: WorldState,
        allowed: tuple[WorldState, ...],
        operation: str,
    ) -> WorldMetadata:
        if self.state not in allowed:
            expected = ", ".join(repr(state.value) for state in allowed)
            raise WorldTransitionError(
                f"Cannot {operation} world while state is "
                f"{self.state.value!r}; expected {expected}."
            )

        return replace(self, state=target)
