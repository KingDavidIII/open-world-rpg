"""Deterministic two-dimensional world coordinate value objects.

World positions are expressed in tiles. Each chunk contains ``CHUNK_SIZE``
tiles per axis, and each region contains ``REGION_SIZE_IN_CHUNKS`` chunks per
axis. Container coordinates use floor division, so negative world positions
belong to negative chunks and regions. Local tile coordinates use modulo and
are therefore always in the inclusive range ``0`` to ``CHUNK_SIZE - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

CHUNK_SIZE: Final = 16
REGION_SIZE_IN_CHUNKS: Final = 16
REGION_SIZE_IN_TILES: Final = CHUNK_SIZE * REGION_SIZE_IN_CHUNKS


def _require_coordinate(*, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")


def _require_type(*, name: str, value: object, expected: type[object]) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be a {expected.__name__}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldPosition:
    """Absolute tile position in the unbounded two-dimensional world."""

    x: int
    y: int

    def __post_init__(self) -> None:
        _require_coordinate(name="x", value=self.x)
        _require_coordinate(name="y", value=self.y)

    def to_chunk(self) -> ChunkCoordinate:
        """Return the chunk containing this world position."""
        return ChunkCoordinate(
            x=self.x // CHUNK_SIZE,
            y=self.y // CHUNK_SIZE,
        )

    def to_region(self) -> RegionCoordinate:
        """Return the region containing this world position."""
        return RegionCoordinate(
            x=self.x // REGION_SIZE_IN_TILES,
            y=self.y // REGION_SIZE_IN_TILES,
        )

    def to_local_tile(self) -> LocalTileCoordinate:
        """Return this position's tile offset within its containing chunk."""
        return LocalTileCoordinate(
            x=self.x % CHUNK_SIZE,
            y=self.y % CHUNK_SIZE,
        )

    @classmethod
    def from_chunk_and_local(
        cls,
        *,
        chunk: ChunkCoordinate,
        local: LocalTileCoordinate,
    ) -> WorldPosition:
        """Combine a chunk coordinate and local tile offset."""
        _require_type(name="chunk", value=chunk, expected=ChunkCoordinate)
        _require_type(name="local", value=local, expected=LocalTileCoordinate)
        return cls(
            x=(chunk.x * CHUNK_SIZE) + local.x,
            y=(chunk.y * CHUNK_SIZE) + local.y,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkCoordinate:
    """Coordinate of a chunk in the unbounded chunk grid."""

    x: int
    y: int

    def __post_init__(self) -> None:
        _require_coordinate(name="x", value=self.x)
        _require_coordinate(name="y", value=self.y)

    def to_region(self) -> RegionCoordinate:
        """Return the region containing this chunk."""
        return RegionCoordinate(
            x=self.x // REGION_SIZE_IN_CHUNKS,
            y=self.y // REGION_SIZE_IN_CHUNKS,
        )

    def to_world_origin(self) -> WorldPosition:
        """Return the world position at this chunk's minimum tile corner."""
        return WorldPosition(
            x=self.x * CHUNK_SIZE,
            y=self.y * CHUNK_SIZE,
        )

    @classmethod
    def from_world(cls, position: WorldPosition) -> ChunkCoordinate:
        """Return the chunk containing a world position."""
        _require_type(name="position", value=position, expected=WorldPosition)
        return position.to_chunk()


@dataclass(frozen=True, slots=True, kw_only=True)
class RegionCoordinate:
    """Coordinate of a region in the unbounded region grid."""

    x: int
    y: int

    def __post_init__(self) -> None:
        _require_coordinate(name="x", value=self.x)
        _require_coordinate(name="y", value=self.y)

    def to_chunk_origin(self) -> ChunkCoordinate:
        """Return the chunk at this region's minimum chunk corner."""
        return ChunkCoordinate(
            x=self.x * REGION_SIZE_IN_CHUNKS,
            y=self.y * REGION_SIZE_IN_CHUNKS,
        )

    def to_world_origin(self) -> WorldPosition:
        """Return the world position at this region's minimum tile corner."""
        return WorldPosition(
            x=self.x * REGION_SIZE_IN_TILES,
            y=self.y * REGION_SIZE_IN_TILES,
        )

    @classmethod
    def from_world(cls, position: WorldPosition) -> RegionCoordinate:
        """Return the region containing a world position."""
        _require_type(name="position", value=position, expected=WorldPosition)
        return position.to_region()

    @classmethod
    def from_chunk(cls, chunk: ChunkCoordinate) -> RegionCoordinate:
        """Return the region containing a chunk."""
        _require_type(name="chunk", value=chunk, expected=ChunkCoordinate)
        return chunk.to_region()


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalTileCoordinate:
    """Tile offset within a chunk, from zero through ``CHUNK_SIZE - 1``."""

    x: int
    y: int

    def __post_init__(self) -> None:
        _require_coordinate(name="x", value=self.x)
        _require_coordinate(name="y", value=self.y)

        if not 0 <= self.x < CHUNK_SIZE:
            raise ValueError(f"x must be between 0 and {CHUNK_SIZE - 1}.")

        if not 0 <= self.y < CHUNK_SIZE:
            raise ValueError(f"y must be between 0 and {CHUNK_SIZE - 1}.")

    @classmethod
    def from_world(cls, position: WorldPosition) -> LocalTileCoordinate:
        """Return a world position's tile offset within its chunk."""
        _require_type(name="position", value=position, expected=WorldPosition)
        return position.to_local_tile()
