"""Pure camera, viewport, palette, HUD, and cache calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from open_world_rpg.world import (
    CHUNK_SIZE,
    ChunkCoordinate,
    ChunkTerrain,
    RegionCoordinate,
    TerrainGenerationServiceSnapshot,
    TerrainRuntime,
    TerrainType,
    WorldPosition,
)

RgbColour = tuple[int, int, int]

TERRAIN_PALETTE: Final[dict[TerrainType, RgbColour]] = {
    TerrainType.DEEP_WATER: (18, 52, 96),
    TerrainType.SHALLOW_WATER: (38, 104, 166),
    TerrainType.COAST: (218, 199, 134),
    TerrainType.PLAINS: (91, 153, 78),
    TerrainType.HILLS: (112, 126, 72),
    TerrainType.MOUNTAINS: (116, 116, 124),
}


def _require_positive_integer(*, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


@dataclass(frozen=True, slots=True, kw_only=True)
class CameraState:
    """Floating UI camera position measured in world tiles."""

    x_tiles: float = 0.0
    y_tiles: float = 0.0
    movement_speed_tiles_per_second: float = 18.0
    fast_multiplier: float = 3.0

    def __post_init__(self) -> None:
        for name, value in (
            ("x_tiles", self.x_tiles),
            ("y_tiles", self.y_tiles),
            ("movement_speed_tiles_per_second", self.movement_speed_tiles_per_second),
            ("fast_multiplier", self.fast_multiplier),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number.")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.movement_speed_tiles_per_second <= 0:
            raise ValueError("movement_speed_tiles_per_second must be greater than zero.")
        if self.fast_multiplier <= 0:
            raise ValueError("fast_multiplier must be greater than zero.")

    def moved(
        self,
        *,
        horizontal: int,
        vertical: int,
        delta_seconds: float,
        fast: bool = False,
    ) -> CameraState:
        """Return frame-rate-independent movement with diagonal normalisation."""
        for name, value in (("horizontal", horizontal), ("vertical", vertical)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value not in (-1, 0, 1):
                raise ValueError(f"{name} must be -1, 0, or 1.")
        if isinstance(delta_seconds, bool) or not isinstance(delta_seconds, (int, float)):
            raise TypeError("delta_seconds must be a number.")
        if not math.isfinite(delta_seconds) or delta_seconds < 0:
            raise ValueError("delta_seconds must be finite and non-negative.")
        if not isinstance(fast, bool):
            raise TypeError("fast must be a boolean.")
        if horizontal == 0 and vertical == 0:
            return self
        length = math.sqrt((horizontal * horizontal) + (vertical * vertical))
        distance = (
            self.movement_speed_tiles_per_second
            * delta_seconds
            * (self.fast_multiplier if fast else 1.0)
        )
        return CameraState(
            x_tiles=self.x_tiles + (horizontal / length * distance),
            y_tiles=self.y_tiles + (vertical / length * distance),
            movement_speed_tiles_per_second=self.movement_speed_tiles_per_second,
            fast_multiplier=self.fast_multiplier,
        )

    @property
    def world_tile(self) -> WorldPosition:
        """Return the tile containing the camera centre using floor semantics."""
        return WorldPosition(x=math.floor(self.x_tiles), y=math.floor(self.y_tiles))


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainViewport:
    """Pure conversion policy for a pixel viewport over world tiles."""

    width_pixels: int
    height_pixels: int
    tile_size_pixels: int

    def __post_init__(self) -> None:
        _require_positive_integer(name="width_pixels", value=self.width_pixels)
        _require_positive_integer(name="height_pixels", value=self.height_pixels)
        _require_positive_integer(name="tile_size_pixels", value=self.tile_size_pixels)

    def visible_chunks(
        self,
        *,
        camera: CameraState,
        preload_margin_chunks: int = 0,
    ) -> tuple[ChunkCoordinate, ...]:
        """Return intersecting chunks ordered by increasing y, then x."""
        if not isinstance(camera, CameraState):
            raise TypeError("camera must be a CameraState.")
        if isinstance(preload_margin_chunks, bool) or not isinstance(preload_margin_chunks, int):
            raise TypeError("preload_margin_chunks must be an integer.")
        if preload_margin_chunks < 0:
            raise ValueError("preload_margin_chunks must be non-negative.")
        half_width = self.width_pixels / (2 * self.tile_size_pixels)
        half_height = self.height_pixels / (2 * self.tile_size_pixels)
        minimum_x = math.floor(camera.x_tiles - half_width) // CHUNK_SIZE
        maximum_x = math.ceil(camera.x_tiles + half_width) - 1
        maximum_x //= CHUNK_SIZE
        minimum_y = math.floor(camera.y_tiles - half_height) // CHUNK_SIZE
        maximum_y = math.ceil(camera.y_tiles + half_height) - 1
        maximum_y //= CHUNK_SIZE
        return tuple(
            ChunkCoordinate(x=x, y=y)
            for y in range(
                minimum_y - preload_margin_chunks,
                maximum_y + preload_margin_chunks + 1,
            )
            for x in range(
                minimum_x - preload_margin_chunks,
                maximum_x + preload_margin_chunks + 1,
            )
        )

    def world_to_screen(
        self,
        *,
        camera: CameraState,
        world_x: float,
        world_y: float,
    ) -> tuple[int, int]:
        """Convert a world tile corner into screen pixels."""
        if not isinstance(camera, CameraState):
            raise TypeError("camera must be a CameraState.")
        return (
            round((world_x - camera.x_tiles) * self.tile_size_pixels + self.width_pixels / 2),
            round((world_y - camera.y_tiles) * self.tile_size_pixels + self.height_pixels / 2),
        )

    def screen_to_world(
        self,
        *,
        camera: CameraState,
        screen_x: int,
        screen_y: int,
    ) -> WorldPosition:
        """Return the world tile containing one screen pixel."""
        if not isinstance(camera, CameraState):
            raise TypeError("camera must be a CameraState.")
        for name, value in (("screen_x", screen_x), ("screen_y", screen_y)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        return WorldPosition(
            x=math.floor(
                camera.x_tiles + (screen_x - self.width_pixels / 2) / self.tile_size_pixels
            ),
            y=math.floor(
                camera.y_tiles + (screen_y - self.height_pixels / 2) / self.tile_size_pixels
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainHudSnapshot:
    """Renderer-independent values displayed by the terrain demo HUD."""

    camera_tile: WorldPosition
    chunk_coordinate: ChunkCoordinate
    region_coordinate: RegionCoordinate
    world_seed: int
    visible_chunk_count: int
    cached_chunk_count: int
    terrain_runtime_revision: int
    repository_revision: int
    cache_hits: int
    cache_misses: int
    successful_generations: int
    failed_generations: int

    @classmethod
    def from_runtime(
        cls,
        *,
        camera: CameraState,
        runtime: TerrainRuntime,
        visible_chunk_count: int,
    ) -> TerrainHudSnapshot:
        """Project current pure HUD data from camera and runtime state."""
        if not isinstance(camera, CameraState):
            raise TypeError("camera must be a CameraState.")
        if not isinstance(runtime, TerrainRuntime):
            raise TypeError("runtime must be a TerrainRuntime.")
        if isinstance(visible_chunk_count, bool) or not isinstance(visible_chunk_count, int):
            raise TypeError("visible_chunk_count must be an integer.")
        if visible_chunk_count < 0:
            raise ValueError("visible_chunk_count must be non-negative.")
        tile = camera.world_tile
        chunk = tile.to_chunk()
        service: TerrainGenerationServiceSnapshot = runtime.service.snapshot()
        return cls(
            camera_tile=tile,
            chunk_coordinate=chunk,
            region_coordinate=chunk.to_region(),
            world_seed=runtime.specification.seed.value,
            visible_chunk_count=visible_chunk_count,
            cached_chunk_count=service.cached_chunk_count,
            terrain_runtime_revision=runtime.revision,
            repository_revision=service.repository_revision,
            cache_hits=service.cache_hits,
            cache_misses=service.cache_misses,
            successful_generations=service.successful_generations,
            failed_generations=service.failed_generations,
        )


def terrain_surface_cache_key(
    terrain: ChunkTerrain,
    *,
    tile_size_pixels: int,
) -> tuple[ChunkCoordinate, int, int, str, int]:
    """Return fields whose change requires rebuilding a chunk surface."""
    if not isinstance(terrain, ChunkTerrain):
        raise TypeError("terrain must be a ChunkTerrain.")
    _require_positive_integer(name="tile_size_pixels", value=tile_size_pixels)
    return (
        terrain.chunk_coordinate,
        terrain.terrain_seed,
        terrain.revision,
        terrain.generation_format_version,
        tile_size_pixels,
    )
