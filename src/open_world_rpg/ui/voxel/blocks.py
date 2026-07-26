"""Deterministic mapping from domain terrain to display-scale voxel columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from open_world_rpg.world import BlockMaterial, TerrainType

ELEVATION_METRES_PER_BLOCK: Final = 30
HIGH_ELEVATION_METRES_PER_BLOCK: Final = 45
HIGH_ELEVATION_START_METRES: Final = 300
DISPLAY_SEA_LEVEL: Final = 12
MIN_DISPLAY_HEIGHT: Final = 1
MAX_DISPLAY_HEIGHT: Final = 64
SNOW_HEIGHT: Final = 48


BlockType = BlockMaterial


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockColumn:
    """Compact visible column description rather than a per-block allocation."""

    ground_height: int
    surface_height: int
    surface: BlockType
    subsurface: BlockType
    water: BlockType | None = None


def elevation_to_height(elevation_metres: int) -> int:
    """Project metres with stronger highland relief around an explicit sea level.

    Elevations through 300 m use 30 m/block. Higher terrain uses 45 m/block
    after that ten-block lowland band, limiting noise while amplifying relief.
    """
    if isinstance(elevation_metres, bool) or not isinstance(elevation_metres, int):
        raise TypeError("elevation_metres must be an integer.")
    if elevation_metres <= HIGH_ELEVATION_START_METRES:
        height = DISPLAY_SEA_LEVEL + elevation_metres // ELEVATION_METRES_PER_BLOCK
    else:
        height = (
            DISPLAY_SEA_LEVEL
            + HIGH_ELEVATION_START_METRES // ELEVATION_METRES_PER_BLOCK
            + (elevation_metres - HIGH_ELEVATION_START_METRES) // HIGH_ELEVATION_METRES_PER_BLOCK
        )
    return max(MIN_DISPLAY_HEIGHT, min(MAX_DISPLAY_HEIGHT, height))


def column_from_terrain(
    *, terrain_type: TerrainType, elevation_metres: int, steep: bool = False
) -> BlockColumn:
    """Return one deterministic display column from existing terrain values."""
    if not isinstance(terrain_type, TerrainType):
        raise TypeError("terrain_type must be a TerrainType.")
    if not isinstance(steep, bool):
        raise TypeError("steep must be a boolean.")
    ground = elevation_to_height(elevation_metres)
    if terrain_type is TerrainType.DEEP_WATER:
        return BlockColumn(
            ground_height=ground,
            surface_height=DISPLAY_SEA_LEVEL,
            surface=BlockType.STONE,
            subsurface=BlockType.STONE,
            water=BlockType.WATER,
        )
    if terrain_type is TerrainType.SHALLOW_WATER:
        return BlockColumn(
            ground_height=ground,
            surface_height=DISPLAY_SEA_LEVEL,
            surface=BlockType.SAND,
            subsurface=BlockType.SAND,
            water=BlockType.WATER,
        )
    if terrain_type is TerrainType.COAST:
        return BlockColumn(
            ground_height=ground,
            surface_height=ground,
            surface=BlockType.SAND,
            subsurface=BlockType.SAND,
        )
    if terrain_type is TerrainType.MOUNTAINS:
        material = BlockType.SNOW if ground >= SNOW_HEIGHT else BlockType.STONE
        return BlockColumn(
            ground_height=ground,
            surface_height=ground,
            surface=material,
            subsurface=BlockType.STONE,
        )
    if terrain_type is TerrainType.HILLS and steep:
        return BlockColumn(
            ground_height=ground,
            surface_height=ground,
            surface=BlockType.GRASS,
            subsurface=BlockType.STONE,
        )
    return BlockColumn(
        ground_height=ground,
        surface_height=ground,
        surface=BlockType.GRASS,
        subsurface=BlockType.DIRT,
    )
