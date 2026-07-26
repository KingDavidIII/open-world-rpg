"""Deterministic, renderer-independent terrain styling calculations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from open_world_rpg.world import TerrainType

RgbColour = tuple[int, int, int]
VISUAL_STYLE_REVISION: Final = 2

TERRAIN_GRADIENTS: Final[dict[TerrainType, tuple[RgbColour, RgbColour]]] = {
    TerrainType.DEEP_WATER: ((8, 28, 54), (20, 67, 105)),
    TerrainType.SHALLOW_WATER: ((24, 91, 126), (52, 151, 156)),
    TerrainType.COAST: ((151, 125, 76), (226, 205, 139)),
    TerrainType.PLAINS: ((48, 103, 54), (112, 159, 78)),
    TerrainType.HILLS: ((65, 78, 43), (132, 119, 69)),
    TerrainType.MOUNTAINS: ((73, 70, 68), (202, 207, 205)),
}
TERRAIN_PALETTE: Final[dict[TerrainType, RgbColour]] = {
    terrain_type: colours[1] for terrain_type, colours in TERRAIN_GRADIENTS.items()
}

_ELEVATION_RANGES: Final[dict[TerrainType, tuple[int, int]]] = {
    TerrainType.DEEP_WATER: (-3_000, -200),
    TerrainType.SHALLOW_WATER: (-199, -1),
    TerrainType.COAST: (0, 20),
    TerrainType.PLAINS: (21, 300),
    TerrainType.HILLS: (301, 1_200),
    TerrainType.MOUNTAINS: (1_201, 3_000),
}


def _clamp(value: int, minimum: int = 0, maximum: int = 255) -> int:
    return max(minimum, min(maximum, value))


def stable_visual_value(*, seed: int, world_x: int, world_y: int, channel: str) -> int:
    """Return a stable unsigned 64-bit visual identity."""
    payload = (
        b"open-world-rpg/terrain-visual/v2\0"
        + seed.to_bytes(8, "big", signed=False)
        + world_x.to_bytes(16, "big", signed=True)
        + world_y.to_bytes(16, "big", signed=True)
        + channel.encode("ascii")
    )
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def terrain_colour(*, terrain_type: TerrainType, elevation: int, light: int = 0) -> RgbColour:
    """Interpolate a natural gradient and apply bounded directional light."""
    if not isinstance(terrain_type, TerrainType):
        raise TypeError("terrain_type must be a TerrainType.")
    if isinstance(elevation, bool) or not isinstance(elevation, int):
        raise TypeError("elevation must be an integer.")
    if isinstance(light, bool) or not isinstance(light, int):
        raise TypeError("light must be an integer.")
    low, high = _ELEVATION_RANGES[terrain_type]
    span = max(1, high - low)
    position = max(0, min(span, elevation - low))
    start, end = TERRAIN_GRADIENTS[terrain_type]
    return (
        _clamp(start[0] + ((end[0] - start[0]) * position // span) + light),
        _clamp(start[1] + ((end[1] - start[1]) * position // span) + light),
        _clamp(start[2] + ((end[2] - start[2]) * position // span) + light),
    )


def slope_light(
    *,
    centre: int,
    west: int,
    east: int,
    north: int,
    south: int,
    strength: int = 1,
) -> int:
    """Estimate upper-left illumination from neighbouring elevations."""
    values = (centre, west, east, north, south, strength)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("slope inputs must be integers.")
    if strength <= 0:
        raise ValueError("strength must be greater than zero.")
    gradient = (west - east) + (north - south)
    return max(-36, min(36, gradient * strength // 45))


def transition_mask(
    *,
    seed: int,
    world_x: int,
    world_y: int,
    terrain_type: TerrainType,
    neighbour_type: TerrainType,
) -> bool:
    """Select deterministic irregular edge overlays between adjacent categories."""
    if not isinstance(terrain_type, TerrainType) or not isinstance(neighbour_type, TerrainType):
        raise TypeError("terrain types must be TerrainType values.")
    if terrain_type is neighbour_type:
        return False
    value = stable_visual_value(
        seed=seed,
        world_x=world_x,
        world_y=world_y,
        channel=f"edge:{terrain_type.value}:{neighbour_type.value}",
    )
    return value % 5 < 2


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualDetail:
    """One deterministic UI-only mark inside a world tile."""

    kind: str
    offset_x_eighths: int
    offset_y_eighths: int


def visual_details(
    *, seed: int, world_x: int, world_y: int, terrain_type: TerrainType
) -> tuple[VisualDetail, ...]:
    """Return sparse deterministic detail placement without random state."""
    if not isinstance(terrain_type, TerrainType):
        raise TypeError("terrain_type must be a TerrainType.")
    value = stable_visual_value(
        seed=seed,
        world_x=world_x,
        world_y=world_y,
        channel=f"detail:{terrain_type.value}",
    )
    frequency = {
        TerrainType.DEEP_WATER: 11,
        TerrainType.SHALLOW_WATER: 7,
        TerrainType.COAST: 6,
        TerrainType.PLAINS: 4,
        TerrainType.HILLS: 5,
        TerrainType.MOUNTAINS: 3,
    }[terrain_type]
    if value % frequency:
        return ()
    kind = {
        TerrainType.DEEP_WATER: "wave",
        TerrainType.SHALLOW_WATER: "ripple",
        TerrainType.COAST: "shell",
        TerrainType.PLAINS: "grass",
        TerrainType.HILLS: "shrub",
        TerrainType.MOUNTAINS: "rock",
    }[terrain_type]
    return (
        VisualDetail(
            kind=kind,
            offset_x_eighths=1 + ((value >> 8) % 6),
            offset_y_eighths=1 + ((value >> 16) % 6),
        ),
    )


def water_wave_phase(*, world_x: int, world_y: int, animation_tick: int) -> int:
    """Return a cheap repeating wave phase without changing static terrain."""
    values = (world_x, world_y, animation_tick)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("water wave inputs must be integers.")
    return (world_x * 3 + world_y * 5 + animation_tick) % 24
