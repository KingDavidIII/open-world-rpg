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
        )
