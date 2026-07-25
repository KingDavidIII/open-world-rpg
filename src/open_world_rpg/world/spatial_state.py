"""Immutable generation state contracts for regions and chunks.

Chunk and region metadata contain generation identities, not generated terrain
or gameplay payloads. ``FAILED`` is terminal. ``UNLOADED`` is reloadable:
chunks may return to ``GENERATING`` and regions may return to ``INDEXING``.
Same-state transitions are no-ops that return the original value.

Region chunk indexes iterate in row-major order: rows use increasing world
chunk ``y`` and coordinates within each row use increasing world chunk ``x``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum

from open_world_rpg.world.coordinates import (
    REGION_SIZE_IN_CHUNKS,
    ChunkCoordinate,
    RegionCoordinate,
)
from open_world_rpg.world.generation import (
    MAX_DERIVED_SEED,
    ChunkGenerationKey,
    RegionGenerationKey,
    WorldGenerationStage,
    WorldSeed,
)


class ChunkState(StrEnum):
    """Lifecycle states for one deterministic chunk contract."""

    DECLARED = "declared"
    GENERATING = "generating"
    READY = "ready"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    UNLOADED = "unloaded"
    FAILED = "failed"


class RegionState(StrEnum):
    """Lifecycle states for one deterministic region contract."""

    DECLARED = "declared"
    INDEXING = "indexing"
    READY = "ready"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    UNLOADED = "unloaded"
    FAILED = "failed"


class ChunkTransitionError(RuntimeError):
    """Raised when a chunk lifecycle transition is illegal."""


class RegionTransitionError(RuntimeError):
    """Raised when a region lifecycle transition is illegal."""


_CHUNK_TRANSITIONS = {
    ChunkState.DECLARED: frozenset({ChunkState.GENERATING, ChunkState.FAILED}),
    ChunkState.GENERATING: frozenset({ChunkState.READY, ChunkState.FAILED}),
    ChunkState.READY: frozenset(
        {
            ChunkState.ACTIVE,
            ChunkState.UNLOADED,
            ChunkState.FAILED,
        }
    ),
    ChunkState.ACTIVE: frozenset({ChunkState.SUSPENDED, ChunkState.FAILED}),
    ChunkState.SUSPENDED: frozenset(
        {
            ChunkState.ACTIVE,
            ChunkState.UNLOADED,
            ChunkState.FAILED,
        }
    ),
    ChunkState.UNLOADED: frozenset({ChunkState.GENERATING, ChunkState.FAILED}),
    ChunkState.FAILED: frozenset(),
}

_REGION_TRANSITIONS = {
    RegionState.DECLARED: frozenset({RegionState.INDEXING, RegionState.FAILED}),
    RegionState.INDEXING: frozenset({RegionState.READY, RegionState.FAILED}),
    RegionState.READY: frozenset(
        {
            RegionState.ACTIVE,
            RegionState.UNLOADED,
            RegionState.FAILED,
        }
    ),
    RegionState.ACTIVE: frozenset({RegionState.SUSPENDED, RegionState.FAILED}),
    RegionState.SUSPENDED: frozenset(
        {
            RegionState.ACTIVE,
            RegionState.UNLOADED,
            RegionState.FAILED,
        }
    ),
    RegionState.UNLOADED: frozenset({RegionState.INDEXING, RegionState.FAILED}),
    RegionState.FAILED: frozenset(),
}


def _validate_seed(*, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if not 0 <= value <= MAX_DERIVED_SEED:
        raise ValueError(f"{name} must be between 0 and {MAX_DERIVED_SEED}.")


def _validate_revision(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("revision must be an integer.")

    if value < 0:
        raise ValueError("revision must be greater than or equal to zero.")


def _chunk_seed(
    *,
    world_seed: WorldSeed,
    coordinate: ChunkCoordinate,
    stage: WorldGenerationStage,
) -> int:
    return ChunkGenerationKey(
        world_seed=world_seed,
        coordinate=coordinate,
        stage=stage,
    ).derived_seed


def _region_seed(
    *,
    world_seed: WorldSeed,
    coordinate: RegionCoordinate,
    stage: WorldGenerationStage,
) -> int:
    return RegionGenerationKey(
        world_seed=world_seed,
        coordinate=coordinate,
        stage=stage,
    ).derived_seed


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkMetadata:
    """Immutable lifecycle and generation identity for one chunk."""

    world_seed: WorldSeed
    coordinate: ChunkCoordinate
    region_coordinate: RegionCoordinate
    terrain_seed: int
    climate_seed: int
    biome_seed: int
    feature_seed: int
    resource_seed: int
    structure_seed: int
    entity_seed: int
    state: ChunkState = ChunkState.DECLARED
    revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.world_seed, WorldSeed):
            raise TypeError("world_seed must be a WorldSeed.")

        if not isinstance(self.coordinate, ChunkCoordinate):
            raise TypeError("coordinate must be a ChunkCoordinate.")

        if not isinstance(self.region_coordinate, RegionCoordinate):
            raise TypeError("region_coordinate must be a RegionCoordinate.")

        if self.region_coordinate != self.coordinate.to_region():
            raise ValueError("region_coordinate must contain the chunk coordinate.")

        if not isinstance(self.state, ChunkState):
            raise TypeError("state must be a ChunkState.")

        _validate_revision(self.revision)
        self._validate_generation_seeds()

    @classmethod
    def create(
        cls,
        *,
        world_seed: WorldSeed,
        coordinate: ChunkCoordinate,
    ) -> ChunkMetadata:
        """Create declared chunk metadata with all stage seeds derived."""
        if not isinstance(world_seed, WorldSeed):
            raise TypeError("world_seed must be a WorldSeed.")

        if not isinstance(coordinate, ChunkCoordinate):
            raise TypeError("coordinate must be a ChunkCoordinate.")

        return cls(
            world_seed=world_seed,
            coordinate=coordinate,
            region_coordinate=coordinate.to_region(),
            terrain_seed=_chunk_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.TERRAIN,
            ),
            climate_seed=_chunk_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.CLIMATE,
            ),
            biome_seed=_chunk_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.BIOMES,
            ),
            feature_seed=_chunk_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.FEATURES,
            ),
            resource_seed=_chunk_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.RESOURCES,
            ),
            structure_seed=_chunk_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.STRUCTURES,
            ),
            entity_seed=_chunk_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.ENTITIES,
            ),
        )

    def transition_to(self, state: ChunkState) -> ChunkMetadata:
        """Return metadata after one validated lifecycle transition."""
        if not isinstance(state, ChunkState):
            raise TypeError("state must be a ChunkState.")

        if state is self.state:
            return self

        if state not in _CHUNK_TRANSITIONS[self.state]:
            raise ChunkTransitionError(
                f"Cannot transition chunk from {self.state.value!r} to {state.value!r}."
            )

        return replace(self, state=state, revision=self.revision + 1)

    def snapshot(self) -> ChunkSnapshot:
        """Return an immutable diagnostic projection."""
        return ChunkSnapshot(
            world_seed=self.world_seed,
            coordinate=self.coordinate,
            region_coordinate=self.region_coordinate,
            terrain_seed=self.terrain_seed,
            climate_seed=self.climate_seed,
            biome_seed=self.biome_seed,
            feature_seed=self.feature_seed,
            resource_seed=self.resource_seed,
            structure_seed=self.structure_seed,
            entity_seed=self.entity_seed,
            state=self.state,
            revision=self.revision,
        )

    def _validate_generation_seeds(self) -> None:
        seeds = (
            ("terrain_seed", self.terrain_seed, WorldGenerationStage.TERRAIN),
            ("climate_seed", self.climate_seed, WorldGenerationStage.CLIMATE),
            ("biome_seed", self.biome_seed, WorldGenerationStage.BIOMES),
            ("feature_seed", self.feature_seed, WorldGenerationStage.FEATURES),
            ("resource_seed", self.resource_seed, WorldGenerationStage.RESOURCES),
            ("structure_seed", self.structure_seed, WorldGenerationStage.STRUCTURES),
            ("entity_seed", self.entity_seed, WorldGenerationStage.ENTITIES),
        )
        for name, value, stage in seeds:
            _validate_seed(name=name, value=value)
            expected = _chunk_seed(
                world_seed=self.world_seed,
                coordinate=self.coordinate,
                stage=stage,
            )
            if value != expected:
                raise ValueError(f"{name} must match its deterministic generation key.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RegionMetadata:
    """Immutable lifecycle and generation identity for one region."""

    world_seed: WorldSeed
    coordinate: RegionCoordinate
    terrain_seed: int
    climate_seed: int
    biome_seed: int
    feature_seed: int
    resource_seed: int
    structure_seed: int
    entity_seed: int
    state: RegionState = RegionState.DECLARED
    revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.world_seed, WorldSeed):
            raise TypeError("world_seed must be a WorldSeed.")

        if not isinstance(self.coordinate, RegionCoordinate):
            raise TypeError("coordinate must be a RegionCoordinate.")

        if not isinstance(self.state, RegionState):
            raise TypeError("state must be a RegionState.")

        _validate_revision(self.revision)
        self._validate_generation_seeds()

    @classmethod
    def create(
        cls,
        *,
        world_seed: WorldSeed,
        coordinate: RegionCoordinate,
    ) -> RegionMetadata:
        """Create declared region metadata with all stage seeds derived."""
        if not isinstance(world_seed, WorldSeed):
            raise TypeError("world_seed must be a WorldSeed.")

        if not isinstance(coordinate, RegionCoordinate):
            raise TypeError("coordinate must be a RegionCoordinate.")

        return cls(
            world_seed=world_seed,
            coordinate=coordinate,
            terrain_seed=_region_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.TERRAIN,
            ),
            climate_seed=_region_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.CLIMATE,
            ),
            biome_seed=_region_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.BIOMES,
            ),
            feature_seed=_region_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.FEATURES,
            ),
            resource_seed=_region_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.RESOURCES,
            ),
            structure_seed=_region_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.STRUCTURES,
            ),
            entity_seed=_region_seed(
                world_seed=world_seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.ENTITIES,
            ),
        )

    def transition_to(self, state: RegionState) -> RegionMetadata:
        """Return metadata after one validated lifecycle transition."""
        if not isinstance(state, RegionState):
            raise TypeError("state must be a RegionState.")

        if state is self.state:
            return self

        if state not in _REGION_TRANSITIONS[self.state]:
            raise RegionTransitionError(
                f"Cannot transition region from {self.state.value!r} to {state.value!r}."
            )

        return replace(self, state=state, revision=self.revision + 1)

    def snapshot(self) -> RegionSnapshot:
        """Return an immutable diagnostic projection."""
        return RegionSnapshot(
            world_seed=self.world_seed,
            coordinate=self.coordinate,
            terrain_seed=self.terrain_seed,
            climate_seed=self.climate_seed,
            biome_seed=self.biome_seed,
            feature_seed=self.feature_seed,
            resource_seed=self.resource_seed,
            structure_seed=self.structure_seed,
            entity_seed=self.entity_seed,
            state=self.state,
            revision=self.revision,
        )

    def _validate_generation_seeds(self) -> None:
        seeds = (
            ("terrain_seed", self.terrain_seed, WorldGenerationStage.TERRAIN),
            ("climate_seed", self.climate_seed, WorldGenerationStage.CLIMATE),
            ("biome_seed", self.biome_seed, WorldGenerationStage.BIOMES),
            ("feature_seed", self.feature_seed, WorldGenerationStage.FEATURES),
            ("resource_seed", self.resource_seed, WorldGenerationStage.RESOURCES),
            ("structure_seed", self.structure_seed, WorldGenerationStage.STRUCTURES),
            ("entity_seed", self.entity_seed, WorldGenerationStage.ENTITIES),
        )
        for name, value, stage in seeds:
            _validate_seed(name=name, value=value)
            expected = _region_seed(
                world_seed=self.world_seed,
                coordinate=self.coordinate,
                stage=stage,
            )
            if value != expected:
                raise ValueError(f"{name} must match its deterministic generation key.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkSnapshot:
    """Immutable diagnostic projection of chunk metadata."""

    world_seed: WorldSeed
    coordinate: ChunkCoordinate
    region_coordinate: RegionCoordinate
    terrain_seed: int
    climate_seed: int
    biome_seed: int
    feature_seed: int
    resource_seed: int
    structure_seed: int
    entity_seed: int
    state: ChunkState
    revision: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RegionSnapshot:
    """Immutable diagnostic projection of region metadata."""

    world_seed: WorldSeed
    coordinate: RegionCoordinate
    terrain_seed: int
    climate_seed: int
    biome_seed: int
    feature_seed: int
    resource_seed: int
    structure_seed: int
    entity_seed: int
    state: RegionState
    revision: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RegionLocalChunkCoordinate:
    """Zero-based chunk offset within a region."""

    x: int
    y: int

    def __post_init__(self) -> None:
        if isinstance(self.x, bool) or not isinstance(self.x, int):
            raise TypeError("x must be an integer.")

        if isinstance(self.y, bool) or not isinstance(self.y, int):
            raise TypeError("y must be an integer.")

        if not 0 <= self.x < REGION_SIZE_IN_CHUNKS:
            raise ValueError(f"x must be between 0 and {REGION_SIZE_IN_CHUNKS - 1}.")

        if not 0 <= self.y < REGION_SIZE_IN_CHUNKS:
            raise ValueError(f"y must be between 0 and {REGION_SIZE_IN_CHUNKS - 1}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RegionChunkIndex:
    """Lazy immutable row-major description of one region's chunks."""

    region_coordinate: RegionCoordinate

    def __post_init__(self) -> None:
        if not isinstance(self.region_coordinate, RegionCoordinate):
            raise TypeError("region_coordinate must be a RegionCoordinate.")

    @property
    def minimum(self) -> ChunkCoordinate:
        """Return the inclusive minimum world chunk coordinate."""
        return self.region_coordinate.to_chunk_origin()

    @property
    def maximum(self) -> ChunkCoordinate:
        """Return the inclusive maximum world chunk coordinate."""
        origin = self.minimum
        return ChunkCoordinate(
            x=origin.x + REGION_SIZE_IN_CHUNKS - 1,
            y=origin.y + REGION_SIZE_IN_CHUNKS - 1,
        )

    @property
    def chunk_count(self) -> int:
        """Return the exact number of chunks in this square region."""
        return REGION_SIZE_IN_CHUNKS**2

    def __iter__(self) -> Iterator[ChunkCoordinate]:
        """Yield world chunk coordinates in deterministic row-major order."""
        origin = self.minimum
        for local_y in range(REGION_SIZE_IN_CHUNKS):
            for local_x in range(REGION_SIZE_IN_CHUNKS):
                yield ChunkCoordinate(
                    x=origin.x + local_x,
                    y=origin.y + local_y,
                )

    def contains(self, coordinate: ChunkCoordinate) -> bool:
        """Return whether a world chunk belongs to this region."""
        if not isinstance(coordinate, ChunkCoordinate):
            raise TypeError("coordinate must be a ChunkCoordinate.")
        return coordinate.to_region() == self.region_coordinate

    def local_coordinate(
        self,
        coordinate: ChunkCoordinate,
    ) -> RegionLocalChunkCoordinate:
        """Return a contained world chunk's zero-based region offset."""
        if not self.contains(coordinate):
            raise ValueError("coordinate does not belong to this region.")

        origin = self.minimum
        return RegionLocalChunkCoordinate(
            x=coordinate.x - origin.x,
            y=coordinate.y - origin.y,
        )

    def chunk_at(
        self,
        local: RegionLocalChunkCoordinate,
    ) -> ChunkCoordinate:
        """Return the world chunk at a validated local region offset."""
        if not isinstance(local, RegionLocalChunkCoordinate):
            raise TypeError("local must be a RegionLocalChunkCoordinate.")

        origin = self.minimum
        return ChunkCoordinate(
            x=origin.x + local.x,
            y=origin.y + local.y,
        )
