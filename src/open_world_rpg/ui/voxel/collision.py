"""Pure height-field collision, jumping, spawning, and voxel ray casting."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from open_world_rpg.world import BlockMaterial, WorldBlockCoordinate

from .camera import PlayerState, Vector3

HeightLookup = Callable[[int, int], int]
SolidLookup = Callable[[int, int, int], bool]
MaterialLookup = Callable[[int, int, int], BlockMaterial]


@dataclass(frozen=True, slots=True, kw_only=True)
class Aabb:
    """Player collision dimensions."""

    half_width: float = 0.3
    height: float = 1.8


DEFAULT_PLAYER_AABB = Aabb()


@dataclass(frozen=True, slots=True, kw_only=True)
class RayHit:
    """First solid voxel intersected by a camera ray."""

    x: int
    y: int
    z: int
    distance: float
    material: BlockMaterial = BlockMaterial.STONE
    face_normal: tuple[int, int, int] = (0, 0, 0)

    @property
    def coordinate(self) -> WorldBlockCoordinate:
        return WorldBlockCoordinate(x=self.x, y=self.y, z=self.z)

    @property
    def adjacent_coordinate(self) -> WorldBlockCoordinate:
        return self.coordinate.offset(
            x=self.face_normal[0],
            y=self.face_normal[1],
            z=self.face_normal[2],
        )


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
    solid_at: SolidLookup | None = None,
    jump: bool = False,
) -> PlayerState:
    """Advance immutable player physics over a deterministic height field."""
    if delta_seconds < 0 or not math.isfinite(delta_seconds):
        raise ValueError("delta_seconds must be finite and non-negative.")
    if solid_at is not None:
        return _move_player_in_voxels(
            player=player,
            delta_x=delta_x,
            delta_z=delta_z,
            delta_seconds=delta_seconds,
            solid_at=solid_at,
            jump=jump,
        )
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


def player_intersects_block(
    *,
    player: PlayerState,
    coordinate: WorldBlockCoordinate,
    body: Aabb = DEFAULT_PLAYER_AABB,
) -> bool:
    """Return whether an integer block overlaps the player's collision body."""
    if not isinstance(player, PlayerState):
        raise TypeError("player must be a PlayerState.")
    if not isinstance(coordinate, WorldBlockCoordinate):
        raise TypeError("coordinate must be a WorldBlockCoordinate.")
    return (
        coordinate.x < player.x + body.half_width
        and coordinate.x + 1 > player.x - body.half_width
        and coordinate.y < player.y + body.height
        and coordinate.y + 1 > player.y
        and coordinate.z < player.z + body.half_width
        and coordinate.z + 1 > player.z - body.half_width
    )


def _body_collides(
    *,
    x: float,
    y: float,
    z: float,
    solid_at: SolidLookup,
    body: Aabb = DEFAULT_PLAYER_AABB,
) -> bool:
    xs = (math.floor(x - body.half_width), math.floor(x + body.half_width))
    zs = (math.floor(z - body.half_width), math.floor(z + body.half_width))
    first_y = math.floor(y + 0.001)
    last_y = math.floor(y + body.height - 0.001)
    return any(
        solid_at(block_x, block_y, block_z)
        for block_x in xs
        for block_z in zs
        for block_y in range(first_y, last_y + 1)
    )


def _move_player_in_voxels(
    *,
    player: PlayerState,
    delta_x: float,
    delta_z: float,
    delta_seconds: float,
    solid_at: SolidLookup,
    jump: bool,
) -> PlayerState:
    x = player.x + delta_x
    if _body_collides(x=x, y=player.y, z=player.z, solid_at=solid_at):
        x = player.x
    z = player.z + delta_z
    if _body_collides(x=x, y=player.y, z=z, solid_at=solid_at):
        z = player.z
    if player.flying:
        return PlayerState(x=x, y=player.y, z=z, flying=True)
    velocity = 7.5 if jump and player.grounded else player.vertical_velocity
    velocity -= 20.0 * delta_seconds
    y = player.y + velocity * delta_seconds
    if velocity > 0 and _body_collides(x=x, y=y, z=z, solid_at=solid_at):
        y = player.y
        velocity = 0.0
    support_y = math.floor(y - 0.001)
    support = any(
        solid_at(block_x, support_y, block_z)
        for block_x in (
            math.floor(x - DEFAULT_PLAYER_AABB.half_width),
            math.floor(x + DEFAULT_PLAYER_AABB.half_width),
        )
        for block_z in (
            math.floor(z - DEFAULT_PLAYER_AABB.half_width),
            math.floor(z + DEFAULT_PLAYER_AABB.half_width),
        )
    )
    grounded = velocity <= 0 and support
    if grounded:
        y = float(support_y + 1)
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
    block_at: MaterialLookup | None = None,
    solid_at: SolidLookup | None = None,
    maximum_distance: float = 6.0,
    step: float = 0.1,
) -> RayHit | None:
    """Step a short deterministic ray and return the first occupied block."""
    if maximum_distance <= 0 or step <= 0:
        raise ValueError("ray distances must be greater than zero.")
    if block_at is None and solid_at is None:
        raise TypeError("block_at or solid_at must be provided.")
    distance = 0.0
    last: tuple[int, int, int] | None = None
    while distance <= maximum_distance:
        block = (
            math.floor(origin[0] + direction[0] * distance),
            math.floor(origin[1] + direction[1] * distance),
            math.floor(origin[2] + direction[2] * distance),
        )
        if block != last:
            material = (
                block_at(*block)
                if block_at is not None
                else BlockMaterial.STONE
                if solid_at is not None and solid_at(*block)
                else BlockMaterial.AIR
            )
            if material.is_solid:
                normal = (
                    (0, 0, 0)
                    if last is None
                    else (
                        last[0] - block[0],
                        last[1] - block[1],
                        last[2] - block[2],
                    )
                )
                return RayHit(
                    x=block[0],
                    y=block[1],
                    z=block[2],
                    distance=distance,
                    material=material,
                    face_normal=normal,
                )
        last = block
        distance += step
    return None
