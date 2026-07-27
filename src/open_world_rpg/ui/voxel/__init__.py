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
from .controls import (
    DEFAULT_CONTROL_HINTS,
    ControlHint,
    MovementAxes,
    normalise_movement_axes,
)
from .editable_world import MAX_EDITABLE_BLOCK_Y, MIN_EDITABLE_BLOCK_Y, EditableVoxelWorld
from .game_flow import (
    GameFlowAction,
    GameFlowController,
    MenuOption,
    VoxelScreen,
)
from .hotbar import DEFAULT_HOTBAR_SLOTS, HOTBAR_SIZE, VoxelHotbar
from .hud import VoxelHudSnapshot
from .interaction import (
    InteractionOutcome,
    InteractionResult,
    VoxelInteractionController,
    invalidated_chunks_for_edit,
)
from .inventory_ui import InventoryActionResult, InventoryScreenController
from .meshing import ChunkMeshSnapshot, VoxelChunkMesh, build_chunk_mesh, mesh_cache_key
from .natural_blocks import TreeShape, natural_blocks_in_area, tree_shape
from .performance import FrameTimeTracker, FrameTimingSnapshot
from .scenery import SceneryKind, SceneryPlacement, scenery_at
from .spawn import select_spawn
from .streaming import streaming_chunks
from .texture_atlas import FaceTexture, atlas_uv, generate_texture_atlas

__all__ = [
    "DEFAULT_CONTROL_HINTS",
    "DEFAULT_HOTBAR_SLOTS",
    "DISPLAY_SEA_LEVEL",
    "ELEVATION_METRES_PER_BLOCK",
    "HOTBAR_SIZE",
    "MAX_EDITABLE_BLOCK_Y",
    "MIN_EDITABLE_BLOCK_Y",
    "Aabb",
    "BlockColumn",
    "BlockType",
    "ChunkMeshSnapshot",
    "ControlHint",
    "EditableVoxelWorld",
    "FaceTexture",
    "FirstPersonCamera",
    "FrameTimeTracker",
    "FrameTimingSnapshot",
    "GameFlowAction",
    "GameFlowController",
    "InteractionOutcome",
    "InteractionResult",
    "InventoryActionResult",
    "InventoryScreenController",
    "MenuOption",
    "MovementAxes",
    "PlayerState",
    "RayHit",
    "SceneryKind",
    "SceneryPlacement",
    "TreeShape",
    "VoxelChunkMesh",
    "VoxelHotbar",
    "VoxelHudSnapshot",
    "VoxelInteractionController",
    "VoxelScreen",
    "atlas_uv",
    "build_chunk_mesh",
    "camera_vectors",
    "column_from_terrain",
    "generate_texture_atlas",
    "invalidated_chunks_for_edit",
    "mesh_cache_key",
    "move_player",
    "natural_blocks_in_area",
    "normalise_movement_axes",
    "player_intersects_block",
    "ray_cast",
    "safe_spawn_height",
    "scenery_at",
    "select_spawn",
    "streaming_chunks",
    "tree_shape",
]
