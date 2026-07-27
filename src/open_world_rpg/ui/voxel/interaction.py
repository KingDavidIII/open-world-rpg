"""Pure breaking, placement, reach, cooldown, and mesh-invalidation policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from open_world_rpg.gameplay import (
    ItemType,
    PlayerInventory,
    item_for_material,
    item_policy,
)
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
    OUT_OF_REACH = "target out of reach"
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
    dropped_item: ItemType | None = None

    @property
    def changed(self) -> bool:
        return self.result in (InteractionResult.BROKEN, InteractionResult.PLACED)

    @property
    def allowed(self) -> bool:
        """Return whether a preview represents an actionable interaction."""
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
    """Validate and apply first-person voxel interactions deterministically."""

    def __init__(
        self,
        *,
        world: EditableVoxelWorld,
        edits: BlockEditStore,
        break_cooldown: float = 0.18,
        placement_cooldown: float = 0.18,
        maximum_reach: float = 5.5,
    ) -> None:
        if not isinstance(world, EditableVoxelWorld):
            raise TypeError("world must be an EditableVoxelWorld.")
        if not isinstance(edits, BlockEditStore):
            raise TypeError("edits must be a BlockEditStore.")
        for name, value in (
            ("break_cooldown", break_cooldown),
            ("placement_cooldown", placement_cooldown),
            ("maximum_reach", maximum_reach),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number.")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if break_cooldown < 0 or placement_cooldown < 0:
            raise ValueError("interaction cooldowns must be non-negative.")
        if maximum_reach <= 0:
            raise ValueError("maximum_reach must be greater than zero.")
        self._world = world
        self._edits = edits
        self._break_cooldown = float(break_cooldown)
        self._placement_cooldown = float(placement_cooldown)
        self._maximum_reach = float(maximum_reach)
        self._last_break = float("-inf")
        self._last_place = float("-inf")

    @property
    def maximum_reach(self) -> float:
        return self._maximum_reach

    def preview_break(self, *, target: RayHit | None) -> InteractionOutcome:
        """Validate a mining target without mutating world state or cooldowns."""
        target_result = self._target_result(target)
        if target_result is not None:
            return target_result
        assert target is not None
        if target.material is BlockMaterial.WATER:
            return InteractionOutcome(
                result=InteractionResult.WATER,
                coordinate=target.coordinate,
            )
        if target.material is BlockMaterial.AIR:
            return InteractionOutcome(result=InteractionResult.NO_TARGET)
        return InteractionOutcome(
            result=InteractionResult.BROKEN,
            coordinate=target.coordinate,
            dropped_item=item_for_material(target.material),
        )

    def preview_place(
        self,
        *,
        target: RayHit | None,
        material: BlockMaterial | None,
        player: PlayerState,
    ) -> InteractionOutcome:
        """Validate placement and expose its destination without mutating state."""
        if not isinstance(player, PlayerState):
            raise TypeError("player must be a PlayerState.")
        target_result = self._target_result(target)
        if target_result is not None:
            return target_result
        assert target is not None
        destination = target.adjacent_coordinate
        if material is None or material in (BlockMaterial.AIR, BlockMaterial.WATER):
            return InteractionOutcome(
                result=InteractionResult.EMPTY_SLOT,
                coordinate=destination,
            )
        if not self._world.supports(destination):
            return InteractionOutcome(
                result=InteractionResult.OUT_OF_BOUNDS,
                coordinate=destination,
            )
        if self._world.block_at(destination) is not BlockMaterial.AIR:
            return InteractionOutcome(
                result=InteractionResult.OCCUPIED,
                coordinate=destination,
            )
        if player_intersects_block(player=player, coordinate=destination):
            return InteractionOutcome(
                result=InteractionResult.PLAYER_INTERSECTION,
                coordinate=destination,
            )
        return InteractionOutcome(
            result=InteractionResult.PLACED,
            coordinate=destination,
        )

    def break_block(self, *, target: RayHit | None, now: float) -> InteractionOutcome:
        if now - self._last_break < self._break_cooldown:
            return InteractionOutcome(result=InteractionResult.COOLDOWN)
        preview = self.preview_break(target=target)
        if preview.result is not InteractionResult.BROKEN:
            return preview
        assert preview.coordinate is not None
        self._edits.set_block(preview.coordinate, BlockMaterial.AIR)
        self._last_break = now
        return InteractionOutcome(
            result=InteractionResult.BROKEN,
            coordinate=preview.coordinate,
            invalidated_chunks=invalidated_chunks_for_edit(preview.coordinate),
            dropped_item=preview.dropped_item,
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
        preview = self.preview_place(target=target, material=material, player=player)
        if preview.result is not InteractionResult.PLACED:
            return preview
        assert preview.coordinate is not None
        assert material is not None
        self._edits.set_block(preview.coordinate, material)
        self._last_place = now
        return InteractionOutcome(
            result=InteractionResult.PLACED,
            coordinate=preview.coordinate,
            invalidated_chunks=invalidated_chunks_for_edit(preview.coordinate),
        )

    def place_inventory_block(
        self,
        *,
        target: RayHit | None,
        inventory: PlayerInventory,
        player: PlayerState,
        now: float,
    ) -> InteractionOutcome:
        """Place from the selected slot and consume exactly one on success."""
        if not isinstance(inventory, PlayerInventory):
            raise TypeError("inventory must be a PlayerInventory.")
        stack = inventory.selected_stack
        outcome = self.place_block(
            target=target,
            material=(None if stack is None else item_policy(stack.item).placeable_material),
            player=player,
            now=now,
        )
        if outcome.result is InteractionResult.PLACED and not inventory.remove_from_slot(
            inventory.selected_hotbar_index, 1
        ):
            raise RuntimeError("Validated placement inventory consumption failed.")
        return outcome

    def _target_result(self, target: RayHit | None) -> InteractionOutcome | None:
        if target is None:
            return InteractionOutcome(result=InteractionResult.NO_TARGET)
        if not math.isfinite(target.distance) or target.distance < 0:
            return InteractionOutcome(
                result=InteractionResult.OUT_OF_REACH,
                coordinate=target.coordinate,
            )
        if target.distance > self._maximum_reach:
            return InteractionOutcome(
                result=InteractionResult.OUT_OF_REACH,
                coordinate=target.coordinate,
            )
        return None
