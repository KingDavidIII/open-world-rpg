"""Public, renderer-independent voxel presentation contracts."""

from .blocks import (
    DISPLAY_SEA_LEVEL,
    ELEVATION_METRES_PER_BLOCK,
    BlockColumn,
    BlockType,
    column_from_terrain,
)
from .camera import FirstPersonCamera, PlayerState, camera_vectors
from .collision import (
    Aabb,
    RayHit,
    move_player,
    ray_cast,
    safe_spawn_height,
)
from .hud import VoxelHudSnapshot
from .meshing import VoxelChunkMesh, build_chunk_mesh, mesh_cache_key
from .scenery import SceneryKind, SceneryPlacement, scenery_at
from .spawn import select_spawn
from .streaming import streaming_chunks
from .texture_atlas import FaceTexture, atlas_uv, generate_texture_atlas

__all__ = [
    "DISPLAY_SEA_LEVEL",
    "ELEVATION_METRES_PER_BLOCK",
    "Aabb",
    "BlockColumn",
    "BlockType",
    "FaceTexture",
    "FirstPersonCamera",
    "PlayerState",
    "RayHit",
    "SceneryKind",
    "SceneryPlacement",
    "VoxelChunkMesh",
    "VoxelHudSnapshot",
    "atlas_uv",
    "build_chunk_mesh",
    "camera_vectors",
    "column_from_terrain",
    "generate_texture_atlas",
    "mesh_cache_key",
    "move_player",
    "ray_cast",
    "safe_spawn_height",
    "scenery_at",
    "select_spawn",
    "streaming_chunks",
]
