"""Pure height-field collision, jumping, spawning, and voxel ray casting."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .camera import PlayerState, Vector3

HeightLookup = Callable[[int, int], int]
SolidLookup = Callable[[int, int, int], bool]


@dataclass(frozen=True, slots=True, kw_only=True)
class Aabb:
    """Player collision dimensions."""

    half_width: float = 0.3
    height: float = 1.8


@dataclass(frozen=True, slots=True, kw_only=True)
class RayHit:
    """First solid voxel intersected by a camera ray."""

    x: int
    y: int
    z: int
    distance: float


def safe_spawn_height(*, world_x: int, world_z: int, height_at: HeightLookup) -> float:
    """Spawn eye/feet safely one block above solid terrain."""
    return float(height_at(world_x, world_z) + 1)


def move_player(
    *,
    player: PlayerState,
    delta_x: float,
    delta_z: float,
    delta_seconds: float,
    height_at: HeightLookup,
    jump: bool = False,
) -> PlayerState:
    """Advance immutable player physics over a deterministic height field."""
    if delta_seconds < 0 or not math.isfinite(delta_seconds):
        raise ValueError("delta_seconds must be finite and non-negative.")
    x = player.x + delta_x
    z = player.z + delta_z
    if player.flying:
        return PlayerState(
            x=x,
            y=player.y,
            z=z,
            vertical_velocity=0.0,
            flying=True,
        )
    floor_height = float(height_at(math.floor(x), math.floor(z)) + 1)
    velocity = 7.5 if jump and player.grounded else player.vertical_velocity
    velocity -= 20.0 * delta_seconds
    y = player.y + velocity * delta_seconds
    grounded = y <= floor_height
    if grounded:
        y = floor_height
        velocity = 0.0
    return PlayerState(
        x=x,
        y=y,
        z=z,
        vertical_velocity=velocity,
        grounded=grounded,
    )


def ray_cast(
    *,
    origin: tuple[float, float, float],
    direction: Vector3,
    solid_at: SolidLookup,
    maximum_distance: float = 6.0,
    step: float = 0.1,
) -> RayHit | None:
    """Step a short deterministic ray and return the first occupied block."""
    if maximum_distance <= 0 or step <= 0:
        raise ValueError("ray distances must be greater than zero.")
    distance = 0.0
    last: tuple[int, int, int] | None = None
    while distance <= maximum_distance:
        block = (
            math.floor(origin[0] + direction[0] * distance),
            math.floor(origin[1] + direction[1] * distance),
            math.floor(origin[2] + direction[2] * distance),
        )
        if block != last and solid_at(*block):
            return RayHit(x=block[0], y=block[1], z=block[2], distance=distance)
        last = block
        distance += step
    return None
