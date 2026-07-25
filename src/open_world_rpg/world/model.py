"""Immutable aggregate model for deterministic world state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Final

from open_world_rpg.world.coordinates import CHUNK_SIZE, REGION_SIZE_IN_CHUNKS
from open_world_rpg.world.generation import DERIVATION_VERSION, WorldSeed
from open_world_rpg.world.metadata import (
    WorldId,
    WorldMetadata,
    WorldState,
    _normalise_name,
)
from open_world_rpg.world.time import (
    WorldClock,
    WorldDateTime,
    WorldInstant,
    WorldTimeConfig,
)

SUPPORTED_GENERATION_FORMAT_VERSION: Final = DERIVATION_VERSION.decode("ascii")


class WorldModelError(RuntimeError):
    """Base error for invalid world aggregate operations."""


class InconsistentWorldModelError(WorldModelError):
    """Raised when aggregate components describe incompatible worlds."""


class InvalidWorldTimeOperationError(WorldModelError):
    """Raised when lifecycle state forbids a world-time operation."""


class UnsupportedWorldSpecificationError(WorldModelError):
    """Raised when a specification uses unsupported compatibility values."""


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldSpecification:
    """Immutable compatibility rules required to interpret a world."""

    name: str
    seed: WorldSeed
    time_config: WorldTimeConfig = field(default_factory=WorldTimeConfig)
    chunk_size_tiles: int = CHUNK_SIZE
    region_size_chunks: int = REGION_SIZE_IN_CHUNKS
    generation_format_version: str = SUPPORTED_GENERATION_FORMAT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalise_name(self.name))

        if not isinstance(self.seed, WorldSeed):
            raise TypeError("seed must be a WorldSeed.")

        if not isinstance(self.time_config, WorldTimeConfig):
            raise TypeError("time_config must be a WorldTimeConfig.")

        if (
            isinstance(self.chunk_size_tiles, bool)
            or not isinstance(self.chunk_size_tiles, int)
            or self.chunk_size_tiles != CHUNK_SIZE
        ):
            raise UnsupportedWorldSpecificationError(
                f"chunk_size_tiles must be the supported value {CHUNK_SIZE}."
            )

        if (
            isinstance(self.region_size_chunks, bool)
            or not isinstance(self.region_size_chunks, int)
            or self.region_size_chunks != REGION_SIZE_IN_CHUNKS
        ):
            raise UnsupportedWorldSpecificationError(
                f"region_size_chunks must be the supported value {REGION_SIZE_IN_CHUNKS}."
            )

        if (
            not isinstance(self.generation_format_version, str)
            or self.generation_format_version != SUPPORTED_GENERATION_FORMAT_VERSION
        ):
            raise UnsupportedWorldSpecificationError(
                "generation_format_version must be the supported value "
                f"{SUPPORTED_GENERATION_FORMAT_VERSION!r}."
            )

    @property
    def tiles_per_region_axis(self) -> int:
        """Return the supported number of tiles along one region axis."""
        return self.chunk_size_tiles * self.region_size_chunks


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldSnapshot:
    """Immutable diagnostic and future-persistence projection of a world."""

    world_id: WorldId
    name: str
    seed: WorldSeed
    state: WorldState
    created_at: datetime
    absolute_world_tick: int
    date_time: WorldDateTime
    chunk_size_tiles: int
    region_size_chunks: int
    generation_format_version: str


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldModel:
    """Aggregate root combining world metadata, rules, and simulation time."""

    metadata: WorldMetadata
    specification: WorldSpecification
    clock: WorldClock

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, WorldMetadata):
            raise InconsistentWorldModelError("metadata must be WorldMetadata.")

        if not isinstance(self.specification, WorldSpecification):
            raise InconsistentWorldModelError("specification must be WorldSpecification.")

        if not isinstance(self.clock, WorldClock):
            raise InconsistentWorldModelError("clock must be WorldClock.")

        if self.metadata.seed != self.specification.seed.value:
            raise InconsistentWorldModelError("metadata seed must match the specification seed.")

        if self.metadata.name != self.specification.name:
            raise InconsistentWorldModelError("metadata name must match the specification name.")

        if self.clock.config != self.specification.time_config:
            raise InconsistentWorldModelError(
                "clock configuration must match the specification time configuration."
            )

    @classmethod
    def create(
        cls,
        *,
        specification: WorldSpecification,
        created_at: datetime,
        world_id: WorldId | None = None,
    ) -> WorldModel:
        """Create a new aggregate at world tick zero."""
        if not isinstance(specification, WorldSpecification):
            raise InconsistentWorldModelError("specification must be WorldSpecification.")

        resolved_world_id = WorldId.create() if world_id is None else world_id
        metadata = WorldMetadata(
            world_id=resolved_world_id,
            name=specification.name,
            seed=specification.seed.value,
            created_at=created_at,
        )
        return cls(
            metadata=metadata,
            specification=specification,
            clock=WorldClock(config=specification.time_config),
        )

    def initialise(self) -> WorldModel:
        """Return this world after its delegated initialise transition."""
        return replace(self, metadata=self.metadata.initialise())

    def activate(self) -> WorldModel:
        """Return this world after its delegated activate transition."""
        return replace(self, metadata=self.metadata.activate())

    def pause(self) -> WorldModel:
        """Return this world after its delegated pause transition."""
        return replace(self, metadata=self.metadata.pause())

    def resume(self) -> WorldModel:
        """Return this world after its delegated resume transition."""
        return replace(self, metadata=self.metadata.resume())

    def close(self) -> WorldModel:
        """Return this world after its delegated close transition."""
        return replace(self, metadata=self.metadata.close())

    def fail(self) -> WorldModel:
        """Return this world after its delegated failure transition."""
        return replace(self, metadata=self.metadata.fail())

    def advance_ticks(self, ticks: int) -> WorldModel:
        """Advance time by ticks while the world is active."""
        self._require_time_state(
            operation="advance time",
            allowed=(WorldState.ACTIVE,),
        )
        return replace(self, clock=self.clock.advance_ticks(ticks))

    def advance_seconds(self, seconds: int) -> WorldModel:
        """Advance time by seconds while the world is active."""
        self._require_time_state(
            operation="advance time",
            allowed=(WorldState.ACTIVE,),
        )
        return replace(self, clock=self.clock.advance_seconds(seconds))

    def reset_clock(
        self,
        *,
        instant: WorldInstant | None = None,
    ) -> WorldModel:
        """Reset time explicitly before the world becomes active."""
        self._require_time_state(
            operation="reset clock",
            allowed=(
                WorldState.CREATED,
                WorldState.INITIALISED,
            ),
        )
        return replace(self, clock=self.clock.reset(instant=instant))

    def snapshot(self) -> WorldSnapshot:
        """Return an immutable diagnostic projection of this aggregate."""
        return WorldSnapshot(
            world_id=self.metadata.world_id,
            name=self.metadata.name,
            seed=self.specification.seed,
            state=self.metadata.state,
            created_at=self.metadata.created_at,
            absolute_world_tick=self.clock.current.tick,
            date_time=self.clock.date_time,
            chunk_size_tiles=self.specification.chunk_size_tiles,
            region_size_chunks=self.specification.region_size_chunks,
            generation_format_version=self.specification.generation_format_version,
        )

    def _require_time_state(
        self,
        *,
        operation: str,
        allowed: tuple[WorldState, ...],
    ) -> None:
        if self.metadata.state in allowed:
            return

        expected = ", ".join(repr(state.value) for state in allowed)
        raise InvalidWorldTimeOperationError(
            f"Cannot {operation} while world state is "
            f"{self.metadata.state.value!r}; expected {expected}."
        )
