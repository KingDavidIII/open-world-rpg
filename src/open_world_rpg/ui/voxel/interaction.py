"""Pure breaking, placement, cooldown, and mesh-invalidation policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from open_world_rpg.world import (
    CHUNK_SIZE,
    BlockEditStore,
    BlockMaterial,
    ChunkCoordinate,
    WorldBlockCoordinate,
)

from .camera import PlayerState
from .collision import RayHit, player_intersects_block
from .editable_world import EditableVoxelWorld


class InteractionResult(StrEnum):
    NONE = "none"
    BROKEN = "block broken"
    PLACED = "block placed"
    COOLDOWN = "interaction cooling down"
    NO_TARGET = "no target"
    WATER = "water cannot be broken"
    EMPTY_SLOT = "selected slot is empty"
    OCCUPIED = "placement destination occupied"
    PLAYER_INTERSECTION = "placement intersects player"
    OUT_OF_BOUNDS = "placement outside supported height"


@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionOutcome:
    result: InteractionResult
    coordinate: WorldBlockCoordinate | None = None
    invalidated_chunks: tuple[ChunkCoordinate, ...] = ()

    @property
    def changed(self) -> bool:
        return self.result in (InteractionResult.BROKEN, InteractionResult.PLACED)


def invalidated_chunks_for_edit(
    coordinate: WorldBlockCoordinate,
) -> tuple[ChunkCoordinate, ...]:
    """Return the owner and horizontal boundary neighbours in stable order."""
    if not isinstance(coordinate, WorldBlockCoordinate):
        raise TypeError("coordinate must be a WorldBlockCoordinate.")
    owner = coordinate.chunk_coordinate
    local = coordinate.local_coordinate
    affected = {owner}
    if local.x == 0:
        affected.add(ChunkCoordinate(x=owner.x - 1, y=owner.y))
    if local.x == CHUNK_SIZE - 1:
        affected.add(ChunkCoordinate(x=owner.x + 1, y=owner.y))
    if local.y == 0:
        affected.add(ChunkCoordinate(x=owner.x, y=owner.y - 1))
    if local.y == CHUNK_SIZE - 1:
        affected.add(ChunkCoordinate(x=owner.x, y=owner.y + 1))
    return tuple(sorted(affected, key=lambda item: (item.y, item.x)))


class VoxelInteractionController:
    """Apply edge-triggered creative edits with deterministic cooldowns."""

    def __init__(
        self,
        *,
        world: EditableVoxelWorld,
        edits: BlockEditStore,
        break_cooldown: float = 0.18,
        placement_cooldown: float = 0.18,
    ) -> None:
        if not isinstance(world, EditableVoxelWorld):
            raise TypeError("world must be an EditableVoxelWorld.")
        if not isinstance(edits, BlockEditStore):
            raise TypeError("edits must be a BlockEditStore.")
        if break_cooldown < 0 or placement_cooldown < 0:
            raise ValueError("interaction cooldowns must be non-negative.")
        self._world = world
        self._edits = edits
        self._break_cooldown = break_cooldown
        self._placement_cooldown = placement_cooldown
        self._last_break = float("-inf")
        self._last_place = float("-inf")

    def break_block(self, *, target: RayHit | None, now: float) -> InteractionOutcome:
        if now - self._last_break < self._break_cooldown:
            return InteractionOutcome(result=InteractionResult.COOLDOWN)
        if target is None:
            return InteractionOutcome(result=InteractionResult.NO_TARGET)
        if target.material is BlockMaterial.WATER:
            return InteractionOutcome(result=InteractionResult.WATER)
        if target.material is BlockMaterial.AIR:
            return InteractionOutcome(result=InteractionResult.NO_TARGET)
        self._edits.set_block(target.coordinate, BlockMaterial.AIR)
        self._last_break = now
        return InteractionOutcome(
            result=InteractionResult.BROKEN,
            coordinate=target.coordinate,
            invalidated_chunks=invalidated_chunks_for_edit(target.coordinate),
        )

    def place_block(
        self,
        *,
        target: RayHit | None,
        material: BlockMaterial | None,
        player: PlayerState,
        now: float,
    ) -> InteractionOutcome:
        if now - self._last_place < self._placement_cooldown:
            return InteractionOutcome(result=InteractionResult.COOLDOWN)
        if target is None:
            return InteractionOutcome(result=InteractionResult.NO_TARGET)
        if material is None or material in (BlockMaterial.AIR, BlockMaterial.WATER):
            return InteractionOutcome(result=InteractionResult.EMPTY_SLOT)
        destination = target.adjacent_coordinate
        if not self._world.supports(destination):
            return InteractionOutcome(result=InteractionResult.OUT_OF_BOUNDS)
        if self._world.block_at(destination) is not BlockMaterial.AIR:
            return InteractionOutcome(result=InteractionResult.OCCUPIED)
        if player_intersects_block(player=player, coordinate=destination):
            return InteractionOutcome(result=InteractionResult.PLAYER_INTERSECTION)
        self._edits.set_block(destination, material)
        self._last_place = now
        return InteractionOutcome(
            result=InteractionResult.PLACED,
            coordinate=destination,
            invalidated_chunks=invalidated_chunks_for_edit(destination),
        )
