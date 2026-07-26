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
    player_intersects_block,
    ray_cast,
    safe_spawn_height,
)
from .editable_world import MAX_EDITABLE_BLOCK_Y, MIN_EDITABLE_BLOCK_Y, EditableVoxelWorld
from .hotbar import DEFAULT_HOTBAR_SLOTS, HOTBAR_SIZE, VoxelHotbar
from .hud import VoxelHudSnapshot
from .interaction import (
    InteractionOutcome,
    InteractionResult,
    VoxelInteractionController,
    invalidated_chunks_for_edit,
)
from .meshing import VoxelChunkMesh, build_chunk_mesh, mesh_cache_key
from .scenery import SceneryKind, SceneryPlacement, scenery_at
from .spawn import select_spawn
from .streaming import streaming_chunks
from .texture_atlas import FaceTexture, atlas_uv, generate_texture_atlas

__all__ = [
    "DEFAULT_HOTBAR_SLOTS",
    "DISPLAY_SEA_LEVEL",
    "ELEVATION_METRES_PER_BLOCK",
    "HOTBAR_SIZE",
    "MAX_EDITABLE_BLOCK_Y",
    "MIN_EDITABLE_BLOCK_Y",
    "Aabb",
    "BlockColumn",
    "BlockType",
    "EditableVoxelWorld",
    "FaceTexture",
    "FirstPersonCamera",
    "InteractionOutcome",
    "InteractionResult",
    "PlayerState",
    "RayHit",
    "SceneryKind",
    "SceneryPlacement",
    "VoxelChunkMesh",
    "VoxelHotbar",
    "VoxelHudSnapshot",
    "VoxelInteractionController",
    "atlas_uv",
    "build_chunk_mesh",
    "camera_vectors",
    "column_from_terrain",
    "generate_texture_atlas",
    "invalidated_chunks_for_edit",
    "mesh_cache_key",
    "move_player",
    "player_intersects_block",
    "ray_cast",
    "safe_spawn_height",
    "scenery_at",
    "select_spawn",
    "streaming_chunks",
]
