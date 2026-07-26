"""Runnable pygame terrain prototype: ``python -m open_world_rpg.ui.terrain_demo``."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

import pygame

from open_world_rpg.application import create_terrain_runtime
from open_world_rpg.world import (
    CHUNK_SIZE,
    ChunkCoordinate,
    ChunkState,
    TerrainGenerationConfig,
    WorldId,
    WorldModel,
    WorldSeed,
    WorldSpecification,
)

from .terrain_view import (
    TERRAIN_PALETTE,
    CameraState,
    TerrainHudSnapshot,
    TerrainViewport,
    terrain_surface_cache_key,
)


class TerrainPrototypeError(RuntimeError):
    """Raised when the visual terrain prototype cannot run."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainPrototypeConfig:
    """Desktop-only prototype configuration."""

    width_pixels: int = 1280
    height_pixels: int = 720
    tile_size_pixels: int = 20
    preload_margin_chunks: int = 1
    target_fps: int = 60
    world_seed: int = 0
    terrain_config: TerrainGenerationConfig = field(
        default_factory=lambda: TerrainGenerationConfig(octave_count=1)
    )


class TerrainPrototypeApplication:
    """Own pygame, terrain infrastructure, camera, loading, rendering, and HUD."""

    def __init__(self, *, config: TerrainPrototypeConfig | None = None) -> None:
        self.config = TerrainPrototypeConfig() if config is None else config
        if not isinstance(self.config, TerrainPrototypeConfig):
            raise TypeError("config must be a TerrainPrototypeConfig or None.")
        specification = WorldSpecification(
            name="Terrain Prototype",
            seed=WorldSeed(value=self.config.world_seed),
        )
        self.world = WorldModel.create(
            specification=specification,
            created_at=datetime(1970, 1, 1, tzinfo=UTC),
            world_id=WorldId(value=UUID(int=0)),
        )
        self.runtime = create_terrain_runtime(
            world=self.world,
            config=self.config.terrain_config,
        )
        self.camera = CameraState()
        self.show_help = True
        self.show_grid = False
        self.show_chunk_boundaries = True
        self.running = False
        self.screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._last_frame: pygame.Surface | None = None
        self._surface_cache: dict[
            ChunkCoordinate,
            tuple[tuple[ChunkCoordinate, int, int, str, int], pygame.Surface],
        ] = {}
        self._visible_coordinates: tuple[ChunkCoordinate, ...] = ()

    def initialise(self) -> None:
        """Initialise pygame and create a resizable display."""
        try:
            pygame.init()
            pygame.font.init()
            try:
                self.screen = pygame.display.set_mode(
                    (self.config.width_pixels, self.config.height_pixels),
                    pygame.RESIZABLE,
                    vsync=1,
                )
            except (pygame.error, TypeError):
                self.screen = pygame.display.set_mode(
                    (self.config.width_pixels, self.config.height_pixels),
                    pygame.RESIZABLE,
                )
            pygame.display.set_caption("Open World RPG - Terrain Prototype")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.Font(None, 22)
            self.running = True
        except Exception as error:
            pygame.quit()
            raise TerrainPrototypeError("Could not initialise the terrain prototype.") from error

    def run(self, *, max_frames: int | None = None) -> int:
        """Run until exit, or for a bounded frame count used by smoke tests."""
        if max_frames is not None:
            if isinstance(max_frames, bool) or not isinstance(max_frames, int):
                raise TypeError("max_frames must be an integer or None.")
            if max_frames <= 0:
                raise ValueError("max_frames must be greater than zero.")
        if not self.running:
            self.initialise()
        frames = 0
        try:
            while self.running and (max_frames is None or frames < max_frames):
                assert self._clock is not None
                delta_seconds = self._clock.tick(self.config.target_fps) / 1000.0
                self.process_events()
                self.update(delta_seconds)
                self.render()
                frames += 1
            return 0
        finally:
            self.shutdown()

    def process_events(self) -> None:
        """Handle window and discrete keyboard controls."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key in (pygame.K_F1, pygame.K_h):
                    self.show_help = not self.show_help
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                    self._surface_cache.clear()
                elif event.key == pygame.K_c:
                    self.show_chunk_boundaries = not self.show_chunk_boundaries
                elif event.key == pygame.K_r:
                    self.camera = CameraState()

    def update(self, delta_seconds: float) -> None:
        """Apply continuous input and synchronously maintain nearby terrain."""
        keys = pygame.key.get_pressed()
        horizontal = int(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - int(
            keys[pygame.K_a] or keys[pygame.K_LEFT]
        )
        vertical = int(keys[pygame.K_s] or keys[pygame.K_DOWN]) - int(
            keys[pygame.K_w] or keys[pygame.K_UP]
        )
        self.camera = self.camera.moved(
            horizontal=horizontal,
            vertical=vertical,
            delta_seconds=delta_seconds,
            fast=bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]),
        )
        viewport = self._viewport()
        visible = viewport.visible_chunks(camera=self.camera)
        nearby = viewport.visible_chunks(
            camera=self.camera,
            preload_margin_chunks=self.config.preload_margin_chunks,
        )
        visible_set = set(visible)
        nearby_set = set(nearby)
        for coordinate in nearby:
            self.runtime.get_or_generate(coordinate)
            metadata = self.runtime.metadata_at(coordinate)
            if coordinate in visible_set and metadata.state in (
                ChunkState.READY,
                ChunkState.SUSPENDED,
            ):
                self.runtime.activate(coordinate)
        for coordinate in self.runtime.coordinates():
            metadata = self.runtime.metadata_at(coordinate)
            if coordinate not in nearby_set and metadata.state is ChunkState.ACTIVE:
                self.runtime.suspend(coordinate)
        self._visible_coordinates = visible

    def render(self) -> None:
        """Render visible cached chunks and readable diagnostics."""
        if self.screen is None:
            raise TerrainPrototypeError("Terrain prototype is not initialised.")
        self.screen.fill((10, 12, 18))
        viewport = self._viewport()
        for coordinate in self._visible_coordinates:
            terrain = self.runtime.terrain_at(coordinate)
            surface = self._chunk_surface(terrain)
            origin = coordinate.to_world_origin()
            position = viewport.world_to_screen(
                camera=self.camera,
                world_x=origin.x,
                world_y=origin.y,
            )
            self.screen.blit(surface, position)
            if self.show_chunk_boundaries:
                pygame.draw.rect(
                    self.screen,
                    (255, 220, 80),
                    (*position, surface.get_width(), surface.get_height()),
                    2,
                )
        self._render_hud()
        self._last_frame = self.screen.copy()
        pygame.display.flip()

    def shutdown(self) -> None:
        """Suspend active chunks and always release pygame resources."""
        for coordinate in self.runtime.coordinates():
            if self.runtime.metadata_at(coordinate).state is ChunkState.ACTIVE:
                self.runtime.suspend(coordinate)
        self.running = False
        pygame.quit()

    def _viewport(self) -> TerrainViewport:
        if self.screen is None:
            width, height = self.config.width_pixels, self.config.height_pixels
        else:
            width, height = self.screen.get_size()
        return TerrainViewport(
            width_pixels=width,
            height_pixels=height,
            tile_size_pixels=self.config.tile_size_pixels,
        )

    def _chunk_surface(self, terrain: object) -> pygame.Surface:
        from open_world_rpg.world import ChunkTerrain

        if not isinstance(terrain, ChunkTerrain):
            raise TypeError("terrain must be a ChunkTerrain.")
        key = terrain_surface_cache_key(
            terrain,
            tile_size_pixels=self.config.tile_size_pixels,
        )
        cached = self._surface_cache.get(terrain.chunk_coordinate)
        if cached is not None and cached[0] == key:
            return cached[1]
        size = CHUNK_SIZE * self.config.tile_size_pixels
        surface = pygame.Surface((size, size))
        for tile in terrain:
            rectangle = pygame.Rect(
                tile.coordinate.x * self.config.tile_size_pixels,
                tile.coordinate.y * self.config.tile_size_pixels,
                self.config.tile_size_pixels,
                self.config.tile_size_pixels,
            )
            surface.fill(TERRAIN_PALETTE[tile.terrain_type], rectangle)
            if self.show_grid:
                pygame.draw.rect(surface, (25, 25, 25), rectangle, 1)
        self._surface_cache[terrain.chunk_coordinate] = (key, surface)
        return surface

    def _render_hud(self) -> None:
        if self.screen is None or self._font is None or self._clock is None:
            return
        hud = TerrainHudSnapshot.from_runtime(
            camera=self.camera,
            runtime=self.runtime,
            visible_chunk_count=len(self._visible_coordinates),
        )
        lines = [
            f"FPS: {self._clock.get_fps():5.1f}",
            f"Camera tile: ({hud.camera_tile.x}, {hud.camera_tile.y})",
            f"Chunk: ({hud.chunk_coordinate.x}, {hud.chunk_coordinate.y})",
            f"Region: ({hud.region_coordinate.x}, {hud.region_coordinate.y})",
            f"Seed: {hud.world_seed}",
            f"Visible / cached: {hud.visible_chunk_count} / {hud.cached_chunk_count}",
            (
                "Runtime / repository revision: "
                f"{hud.terrain_runtime_revision} / {hud.repository_revision}"
            ),
            f"Cache hits / misses: {hud.cache_hits} / {hud.cache_misses}",
            f"Generated / failed: {hud.successful_generations} / {hud.failed_generations}",
        ]
        if self.show_help:
            lines.extend(
                (
                    "",
                    "WASD / arrows: move   Shift: faster   R: origin",
                    "G: tile grid   C: chunk borders   F1/H: help   Esc: exit",
                )
            )
        line_height = self._font.get_linesize()
        overlay = pygame.Surface(
            (610, (len(lines) * line_height) + 16),
            pygame.SRCALPHA,
        )
        overlay.fill((0, 0, 0, 185))
        for index, line in enumerate(lines):
            text = self._font.render(line, True, (245, 245, 245))
            overlay.blit(text, (8, 8 + index * line_height))
        self.screen.blit(overlay, (10, 10))


def main() -> int:
    """Run the interactive terrain prototype."""
    try:
        return TerrainPrototypeApplication().run()
    except Exception:
        logging.getLogger("open_world_rpg").exception("Terrain prototype failed.")
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised as a module entry point
    raise SystemExit(main())
