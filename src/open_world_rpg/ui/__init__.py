"""Desktop user-interface prototypes kept outside domain packages."""

from open_world_rpg.ui.terrain_style import (
    TERRAIN_GRADIENTS,
    TERRAIN_PALETTE,
    VISUAL_STYLE_REVISION,
    VisualDetail,
    slope_light,
    stable_visual_value,
    terrain_colour,
    transition_mask,
    visual_details,
    water_wave_phase,
)
from open_world_rpg.ui.terrain_view import (
    CameraState,
    TerrainHudSnapshot,
    TerrainViewport,
    ZoomState,
    terrain_surface_cache_key,
)

__all__ = [
    "TERRAIN_GRADIENTS",
    "TERRAIN_PALETTE",
    "VISUAL_STYLE_REVISION",
    "CameraState",
    "TerrainHudSnapshot",
    "TerrainViewport",
    "VisualDetail",
    "ZoomState",
    "slope_light",
    "stable_visual_value",
    "terrain_colour",
    "terrain_surface_cache_key",
    "transition_mask",
    "visual_details",
    "water_wave_phase",
]
