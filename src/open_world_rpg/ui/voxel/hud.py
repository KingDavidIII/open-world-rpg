"""Pure voxel diagnostic HUD projection."""

from __future__ import annotations

import math
from dataclasses import dataclass

from open_world_rpg.world import (
    CHUNK_SIZE,
    REGION_SIZE_IN_CHUNKS,
    BlockMaterial,
    ChunkCoordinate,
)

from .camera import PlayerState
from .collision import RayHit


@dataclass(frozen=True, slots=True, kw_only=True)
class VoxelHudSnapshot:
    """Renderer-independent compact and expanded diagnostic values."""

    fps: float
    position: tuple[float, float, float]
    block: tuple[int, int, int]
    chunk: ChunkCoordinate
    region: ChunkCoordinate
    seed: int
    active_chunks: int
    cached_chunks: int
    mesh_count: int
    triangles: int
    render_distance: int
    mode: str
    target: tuple[int, int, int] | None
    target_material: BlockMaterial | None
    target_face: tuple[int, int, int] | None
    selected_material: BlockMaterial | None
    edit_revision: int
    edited_block_count: int
    last_interaction: str
    save_path: str | None
    dirty: bool
    loading: bool
    selected_item: str | None
    selected_quantity: int
    inventory_revision: int
    occupied_slots: int
    total_inventory_items: int
    active_dropped_items: int
    nearest_drop_distance: float | None
    last_pickup: str
    last_placement_consumption: str
    selected_slot_kind: str = "empty"
    tool_durability: tuple[int, int] | None = None
    mining_progress: float = 0.0
    health: float = 100.0
    stamina: float = 100.0
    fall_distance: float = 0.0
    last_fall_damage: int = 0
    death_count: int = 0
    vitals_revision: int = 0
    mouse_captured: bool = False
    target_distance: float | None = None
    break_preview: str = "no target"
    placement_preview: str = "no target"
    placement_target: tuple[int, int, int] | None = None
    interaction_prompt: str = "Click to capture the mouse"

    @classmethod
    def create(
        cls,
        *,
        fps: float,
        player: PlayerState,
        seed: int,
        active_chunks: int,
        cached_chunks: int,
        mesh_count: int,
        triangles: int,
        render_distance: int,
        target: RayHit | None,
        loading: bool,
        selected_material: BlockMaterial | None = None,
        edit_revision: int = 0,
        edited_block_count: int = 0,
        last_interaction: str = "none",
        save_path: str | None = None,
        dirty: bool = False,
        selected_item: str | None = None,
        selected_quantity: int = 0,
        inventory_revision: int = 0,
        occupied_slots: int = 0,
        total_inventory_items: int = 0,
        active_dropped_items: int = 0,
        nearest_drop_distance: float | None = None,
        last_pickup: str = "none",
        last_placement_consumption: str = "none",
        selected_slot_kind: str = "empty",
        tool_durability: tuple[int, int] | None = None,
        mining_progress: float = 0.0,
        health: float = 100.0,
        stamina: float = 100.0,
        fall_distance: float = 0.0,
        last_fall_damage: int = 0,
        death_count: int = 0,
        vitals_revision: int = 0,
        mouse_captured: bool = False,
        break_preview: str = "no target",
        placement_preview: str = "no target",
        placement_target: tuple[int, int, int] | None = None,
        interaction_prompt: str = "Click to capture the mouse",
    ) -> VoxelHudSnapshot:
        block = (math.floor(player.x), math.floor(player.y), math.floor(player.z))
        chunk = ChunkCoordinate(
            x=block[0] // CHUNK_SIZE,
            y=block[2] // CHUNK_SIZE,
        )
        return cls(
            fps=fps,
            position=(player.x, player.y, player.z),
            block=block,
            chunk=chunk,
            region=ChunkCoordinate(
                x=chunk.x // REGION_SIZE_IN_CHUNKS,
                y=chunk.y // REGION_SIZE_IN_CHUNKS,
            ),
            seed=seed,
            active_chunks=active_chunks,
            cached_chunks=cached_chunks,
            mesh_count=mesh_count,
            triangles=triangles,
            render_distance=render_distance,
            mode="FLY" if player.flying else "WALK",
            target=None if target is None else (target.x, target.y, target.z),
            target_material=None if target is None else target.material,
            target_face=None if target is None else target.face_normal,
            selected_material=selected_material,
            edit_revision=edit_revision,
            edited_block_count=edited_block_count,
            last_interaction=last_interaction,
            save_path=save_path,
            dirty=dirty,
            loading=loading,
            selected_item=selected_item,
            selected_quantity=selected_quantity,
            inventory_revision=inventory_revision,
            occupied_slots=occupied_slots,
            total_inventory_items=total_inventory_items,
            active_dropped_items=active_dropped_items,
            nearest_drop_distance=nearest_drop_distance,
            last_pickup=last_pickup,
            last_placement_consumption=last_placement_consumption,
            selected_slot_kind=selected_slot_kind,
            tool_durability=tool_durability,
            mining_progress=mining_progress,
            health=health,
            stamina=stamina,
            fall_distance=fall_distance,
            last_fall_damage=last_fall_damage,
            death_count=death_count,
            vitals_revision=vitals_revision,
            mouse_captured=mouse_captured,
            target_distance=None if target is None else target.distance,
            break_preview=break_preview,
            placement_preview=placement_preview,
            placement_target=placement_target,
            interaction_prompt=interaction_prompt,
        )
