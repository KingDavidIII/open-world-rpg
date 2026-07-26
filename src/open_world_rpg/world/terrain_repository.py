"""Controlled in-memory repository boundary for immutable chunk terrain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from open_world_rpg.world.coordinates import CHUNK_SIZE, ChunkCoordinate
from open_world_rpg.world.generation import (
    ChunkGenerationKey,
    WorldGenerationStage,
    WorldSeed,
)
from open_world_rpg.world.model import SUPPORTED_GENERATION_FORMAT_VERSION
from open_world_rpg.world.terrain import ChunkTerrain
from open_world_rpg.world.terrain_sampling import TerrainGenerationConfig


class TerrainRepositoryError(RuntimeError):
    """Base error for terrain repository access and compatibility."""


class TerrainRepositoryAccessError(TerrainRepositoryError):
    """Raised when a repository operation cannot access requested terrain."""


class TerrainMissingError(TerrainRepositoryAccessError):
    """Raised when requested terrain is absent."""


class IncompatibleTerrainRepositoryScopeError(TerrainRepositoryError):
    """Raised when repository data violates its compatibility scope."""


class TerrainRepositoryConflictError(TerrainRepositoryError):
    """Raised when storing different terrain at an occupied coordinate."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainRepositoryScope:
    """Immutable compatibility guard for one in-memory terrain repository."""

    world_seed: WorldSeed
    chunk_size_tiles: int
    generation_format_version: str
    terrain_config: TerrainGenerationConfig

    def __post_init__(self) -> None:
        if not isinstance(self.world_seed, WorldSeed):
            raise TypeError("world_seed must be a WorldSeed.")
        if isinstance(self.chunk_size_tiles, bool) or not isinstance(self.chunk_size_tiles, int):
            raise TypeError("chunk_size_tiles must be an integer.")
        if self.chunk_size_tiles != CHUNK_SIZE:
            raise IncompatibleTerrainRepositoryScopeError(
                f"chunk_size_tiles must be the supported value {CHUNK_SIZE}."
            )
        if not isinstance(self.generation_format_version, str):
            raise TypeError("generation_format_version must be a string.")
        if not isinstance(self.terrain_config, TerrainGenerationConfig):
            raise TypeError("terrain_config must be a TerrainGenerationConfig.")
        if (
            self.generation_format_version != SUPPORTED_GENERATION_FORMAT_VERSION
            or self.terrain_config.generation_format_version != self.generation_format_version
        ):
            raise IncompatibleTerrainRepositoryScopeError(
                "Repository and terrain configuration generation formats must agree."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainRepositorySnapshot:
    """Immutable repository diagnostics without terrain tile payloads."""

    scope: TerrainRepositoryScope
    revision: int
    chunk_count: int
    coordinates: tuple[ChunkCoordinate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, TerrainRepositoryScope):
            raise TypeError("scope must be a TerrainRepositoryScope.")
        for name, value in (
            ("revision", self.revision),
            ("chunk_count", self.chunk_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be greater than or equal to zero.")
        if not isinstance(self.coordinates, tuple) or not all(
            isinstance(coordinate, ChunkCoordinate) for coordinate in self.coordinates
        ):
            raise TypeError("coordinates must be a tuple of ChunkCoordinate values.")
        if len(self.coordinates) != self.chunk_count:
            raise ValueError("coordinate count must match chunk_count.")


@runtime_checkable
class TerrainRepository(Protocol):
    """Controlled mutable repository interface for immutable terrain."""

    @property
    def scope(self) -> TerrainRepositoryScope:
        """Return this repository's immutable compatibility scope."""

    def get(self, coordinate: ChunkCoordinate) -> ChunkTerrain:
        """Return stored terrain or raise TerrainMissingError."""

    def contains(self, coordinate: ChunkCoordinate) -> bool:
        """Return whether terrain exists without mutating repository state."""

    def store(self, terrain: ChunkTerrain) -> None:
        """Store new terrain idempotently without replacement."""

    def remove(self, coordinate: ChunkCoordinate) -> None:
        """Remove terrain when present."""

    def clear(self) -> None:
        """Remove every stored terrain payload."""

    def coordinates(self) -> tuple[ChunkCoordinate, ...]:
        """Return coordinates ordered by increasing y, then x."""

    def snapshot(self) -> TerrainRepositorySnapshot:
        """Return immutable repository diagnostics."""

    def __len__(self) -> int:
        """Return the number of stored chunks."""


@dataclass(slots=True, kw_only=True)
class InMemoryTerrainRepository:
    """Canonical in-memory mapping with explicit content revision semantics."""

    scope: TerrainRepositoryScope
    _terrain: dict[ChunkCoordinate, ChunkTerrain] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _revision: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scope, TerrainRepositoryScope):
            raise TypeError("scope must be a TerrainRepositoryScope.")

    @property
    def revision(self) -> int:
        """Return the number of successful content-changing operations."""
        return self._revision

    def __len__(self) -> int:
        """Return the number of stored chunks."""
        return len(self._terrain)

    def get(self, coordinate: ChunkCoordinate) -> ChunkTerrain:
        """Return stored terrain without mutating repository state."""
        self._validate_coordinate(coordinate)
        try:
            return self._terrain[coordinate]
        except KeyError as error:
            raise TerrainMissingError(
                f"Terrain for chunk ({coordinate.x}, {coordinate.y}) is not stored."
            ) from error

    def contains(self, coordinate: ChunkCoordinate) -> bool:
        """Return whether terrain exists without mutating repository state."""
        self._validate_coordinate(coordinate)
        return coordinate in self._terrain

    def store(self, terrain: ChunkTerrain) -> None:
        """Store new terrain, reject replacement, and preserve idempotency."""
        self._validate_terrain(terrain)
        existing = self._terrain.get(terrain.chunk_coordinate)
        if existing is not None:
            if existing == terrain:
                return
            raise TerrainRepositoryConflictError(
                "Different terrain is already stored for chunk "
                f"({terrain.chunk_coordinate.x}, {terrain.chunk_coordinate.y})."
            )
        self._terrain[terrain.chunk_coordinate] = terrain
        self._revision += 1

    def remove(self, coordinate: ChunkCoordinate) -> None:
        """Remove terrain if present and increment revision exactly once."""
        self._validate_coordinate(coordinate)
        if coordinate not in self._terrain:
            return
        del self._terrain[coordinate]
        self._revision += 1

    def clear(self) -> None:
        """Clear non-empty content and increment revision exactly once."""
        if not self._terrain:
            return
        self._terrain.clear()
        self._revision += 1

    def coordinates(self) -> tuple[ChunkCoordinate, ...]:
        """Return deterministic increasing-y then increasing-x ordering."""
        return tuple(sorted(self._terrain, key=lambda coordinate: (coordinate.y, coordinate.x)))

    def snapshot(self) -> TerrainRepositorySnapshot:
        """Return immutable diagnostics without copying tile payloads."""
        coordinates = self.coordinates()
        return TerrainRepositorySnapshot(
            scope=self.scope,
            revision=self._revision,
            chunk_count=len(coordinates),
            coordinates=coordinates,
        )

    @staticmethod
    def _validate_coordinate(coordinate: object) -> None:
        if not isinstance(coordinate, ChunkCoordinate):
            raise TypeError("coordinate must be a ChunkCoordinate.")

    def _validate_terrain(self, terrain: object) -> None:
        if not isinstance(terrain, ChunkTerrain):
            raise TypeError("terrain must be a ChunkTerrain.")
        if terrain.world_seed != self.scope.world_seed:
            raise IncompatibleTerrainRepositoryScopeError(
                "Terrain world seed must match the repository scope."
            )
        if (
            terrain.width != self.scope.chunk_size_tiles
            or terrain.height != self.scope.chunk_size_tiles
        ):
            raise IncompatibleTerrainRepositoryScopeError(
                "Terrain dimensions must match the repository scope."
            )
        if terrain.generation_format_version != self.scope.generation_format_version:
            raise IncompatibleTerrainRepositoryScopeError(
                "Terrain generation format must match the repository scope."
            )
        expected_seed = ChunkGenerationKey(
            world_seed=self.scope.world_seed,
            coordinate=terrain.chunk_coordinate,
            stage=WorldGenerationStage.TERRAIN,
        ).derived_seed
        if terrain.terrain_seed != expected_seed:
            raise IncompatibleTerrainRepositoryScopeError(
                "Terrain seed must match its TERRAIN-stage generation key."
            )
