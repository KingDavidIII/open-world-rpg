"""Immutable terrain data contracts and generation interface.

Terrain elevation is stored as whole metres relative to world sea level. The
supported inclusive range from ``MIN_TERRAIN_ELEVATION`` through
``MAX_TERRAIN_ELEVATION`` is a persistence and generation compatibility
policy.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

from open_world_rpg.world.coordinates import (
    CHUNK_SIZE,
    ChunkCoordinate,
    LocalTileCoordinate,
)
from open_world_rpg.world.generation import (
    MAX_DERIVED_SEED,
    ChunkGenerationKey,
    WorldGenerationStage,
    WorldSeed,
)
from open_world_rpg.world.model import (
    SUPPORTED_GENERATION_FORMAT_VERSION,
    WorldSpecification,
)

MIN_TERRAIN_ELEVATION: Final = -32_768
MAX_TERRAIN_ELEVATION: Final = 32_767


class TerrainGenerationError(RuntimeError):
    """Base error for terrain contracts and generation failures."""


class InvalidTerrainPayloadError(TerrainGenerationError):
    """Raised when a terrain payload contains invalid values."""


class IncompleteTerrainCoverageError(InvalidTerrainPayloadError):
    """Raised when a chunk payload does not cover every local tile."""


class DuplicateTerrainCoordinateError(InvalidTerrainPayloadError):
    """Raised when multiple tiles use the same local coordinate."""


class IncompatibleTerrainDimensionsError(InvalidTerrainPayloadError):
    """Raised when terrain dimensions differ from supported chunk dimensions."""


class TerrainGeneratorExecutionError(TerrainGenerationError):
    """Raised when a terrain generator cannot produce a valid payload."""


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class TerrainElevation:
    """Whole-metre terrain elevation relative to world sea level."""

    metres: int

    def __post_init__(self) -> None:
        if isinstance(self.metres, bool) or not isinstance(self.metres, int):
            raise TypeError("metres must be an integer.")

        if not MIN_TERRAIN_ELEVATION <= self.metres <= MAX_TERRAIN_ELEVATION:
            raise ValueError(
                f"metres must be between {MIN_TERRAIN_ELEVATION} and {MAX_TERRAIN_ELEVATION}."
            )


class TerrainType(StrEnum):
    """Foundational non-biome terrain categories."""

    DEEP_WATER = "deep_water"
    SHALLOW_WATER = "shallow_water"
    COAST = "coast"
    PLAINS = "plains"
    HILLS = "hills"
    MOUNTAINS = "mountains"


def _validate_revision(*, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero.")


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainTile:
    """Immutable terrain data for one local chunk tile."""

    coordinate: LocalTileCoordinate
    elevation: TerrainElevation
    terrain_type: TerrainType
    revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate, LocalTileCoordinate):
            raise TypeError("coordinate must be a LocalTileCoordinate.")

        if not isinstance(self.elevation, TerrainElevation):
            raise TypeError("elevation must be a TerrainElevation.")

        if not isinstance(self.terrain_type, TerrainType):
            raise TypeError("terrain_type must be a TerrainType.")

        _validate_revision(name="revision", value=self.revision)


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkTerrainSnapshot:
    """Immutable diagnostic and future-persistence terrain projection."""

    world_seed: WorldSeed
    chunk_coordinate: ChunkCoordinate
    terrain_seed: int
    width: int
    height: int
    tiles: tuple[TerrainTile, ...]
    revision: int
    generation_format_version: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkTerrain:
    """Complete immutable row-major terrain payload for one chunk."""

    world_seed: WorldSeed
    chunk_coordinate: ChunkCoordinate
    terrain_seed: int
    width: int
    height: int
    tiles: tuple[TerrainTile, ...]
    revision: int = 0
    generation_format_version: str = SUPPORTED_GENERATION_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.world_seed, WorldSeed):
            raise TypeError("world_seed must be a WorldSeed.")

        if not isinstance(self.chunk_coordinate, ChunkCoordinate):
            raise TypeError("chunk_coordinate must be a ChunkCoordinate.")

        self._validate_dimensions()
        self._validate_terrain_seed()
        _validate_revision(name="revision", value=self.revision)

        if (
            not isinstance(self.generation_format_version, str)
            or self.generation_format_version != SUPPORTED_GENERATION_FORMAT_VERSION
        ):
            raise InvalidTerrainPayloadError(
                "generation_format_version must match the supported world generation format."
            )

        if not isinstance(self.tiles, tuple):
            raise TypeError("tiles must be a tuple of TerrainTile values.")

        coordinates: set[LocalTileCoordinate] = set()
        for tile in self.tiles:
            if not isinstance(tile, TerrainTile):
                raise TypeError("tiles must contain only TerrainTile values.")

            if tile.coordinate in coordinates:
                raise DuplicateTerrainCoordinateError(
                    f"Duplicate terrain tile coordinate: {tile.coordinate!r}."
                )
            coordinates.add(tile.coordinate)

        expected_count = self.width * self.height
        if len(self.tiles) != expected_count:
            raise IncompleteTerrainCoverageError(
                f"Chunk terrain must contain exactly {expected_count} local tile coordinates."
            )

        ordered = tuple(
            sorted(
                self.tiles,
                key=lambda tile: (
                    tile.coordinate.y,
                    tile.coordinate.x,
                ),
            )
        )
        object.__setattr__(self, "tiles", ordered)

    def __iter__(self) -> Iterator[TerrainTile]:
        """Iterate tiles by increasing y, then increasing x within each row."""
        return iter(self.tiles)

    def __len__(self) -> int:
        """Return the exact number of local terrain tiles."""
        return len(self.tiles)

    def tile_at(self, coordinate: LocalTileCoordinate) -> TerrainTile:
        """Return the terrain tile at one validated local coordinate."""
        if not isinstance(coordinate, LocalTileCoordinate):
            raise TypeError("coordinate must be a LocalTileCoordinate.")

        return self.tiles[(coordinate.y * self.width) + coordinate.x]

    @property
    def minimum_elevation(self) -> TerrainElevation:
        """Return the minimum elevation present in this chunk."""
        return min(tile.elevation for tile in self.tiles)

    @property
    def maximum_elevation(self) -> TerrainElevation:
        """Return the maximum elevation present in this chunk."""
        return max(tile.elevation for tile in self.tiles)

    @property
    def terrain_type_counts(self) -> Mapping[TerrainType, int]:
        """Return immutable counts in TerrainType declaration order."""
        counts = dict.fromkeys(TerrainType, 0)
        for tile in self.tiles:
            counts[tile.terrain_type] += 1
        return MappingProxyType(counts)

    def snapshot(self) -> ChunkTerrainSnapshot:
        """Return an immutable terrain projection."""
        return ChunkTerrainSnapshot(
            world_seed=self.world_seed,
            chunk_coordinate=self.chunk_coordinate,
            terrain_seed=self.terrain_seed,
            width=self.width,
            height=self.height,
            tiles=self.tiles,
            revision=self.revision,
            generation_format_version=self.generation_format_version,
        )

    def _validate_dimensions(self) -> None:
        if (
            isinstance(self.width, bool)
            or not isinstance(self.width, int)
            or self.width != CHUNK_SIZE
        ):
            raise IncompatibleTerrainDimensionsError(
                f"width must match the supported chunk size {CHUNK_SIZE}."
            )

        if (
            isinstance(self.height, bool)
            or not isinstance(self.height, int)
            or self.height != CHUNK_SIZE
        ):
            raise IncompatibleTerrainDimensionsError(
                f"height must match the supported chunk size {CHUNK_SIZE}."
            )

    def _validate_terrain_seed(self) -> None:
        if isinstance(self.terrain_seed, bool) or not isinstance(self.terrain_seed, int):
            raise TypeError("terrain_seed must be an integer.")

        if not 0 <= self.terrain_seed <= MAX_DERIVED_SEED:
            raise ValueError(f"terrain_seed must be between 0 and {MAX_DERIVED_SEED}.")

        expected = ChunkGenerationKey(
            world_seed=self.world_seed,
            coordinate=self.chunk_coordinate,
            stage=WorldGenerationStage.TERRAIN,
        ).derived_seed
        if self.terrain_seed != expected:
            raise InvalidTerrainPayloadError(
                "terrain_seed must match the deterministic TERRAIN generation key."
            )


@runtime_checkable
class TerrainGenerator(Protocol):
    """Runtime-checkable contract for future deterministic terrain generators."""

    def generate(
        self,
        *,
        specification: WorldSpecification,
        coordinate: ChunkCoordinate,
    ) -> ChunkTerrain:
        """Generate complete deterministic terrain for one chunk."""
