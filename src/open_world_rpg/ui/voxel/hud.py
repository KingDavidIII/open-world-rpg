"""Pure voxel diagnostic HUD projection."""

from __future__ import annotations

import math
from dataclasses import dataclass

from open_world_rpg.world import CHUNK_SIZE, REGION_SIZE_IN_CHUNKS, ChunkCoordinate

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
            loading=loading,
        )
