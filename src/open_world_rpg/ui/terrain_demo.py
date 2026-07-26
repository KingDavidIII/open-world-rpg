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
    ChunkTerrain,
    LocalTileCoordinate,
    TerrainGenerationConfig,
    TerrainTile,
    TerrainType,
    WorldId,
    WorldModel,
    WorldSeed,
    WorldSpecification,
)

from .terrain_style import (
    TERRAIN_PALETTE,
    slope_light,
    terrain_colour,
    transition_mask,
    visual_details,
    water_wave_phase,
)
from .terrain_view import (
    CameraState,
    TerrainHudSnapshot,
    TerrainViewport,
    ZoomState,
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
        default_factory=lambda: TerrainGenerationConfig(octave_count=2)
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
        self.zoom = ZoomState(tile_size_pixels=self.config.tile_size_pixels)
        self.show_help = True
        self.show_grid = False
        self.show_chunk_boundaries = False
        self.show_debug = False
        self.loading = False
        self.running = False
        self.screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._last_frame: pygame.Surface | None = None
        self._surface_cache: dict[
            ChunkCoordinate,
            tuple[tuple[ChunkCoordinate, int, int, str, int, int], pygame.Surface],
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
            elif event.type == pygame.MOUSEWHEEL:
                self.zoom = self.zoom.changed(steps=event.y)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key in (pygame.K_F1, pygame.K_h):
                    self.show_help = not self.show_help
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_c:
                    self.show_chunk_boundaries = not self.show_chunk_boundaries
                elif event.key == pygame.K_r:
                    self.camera = CameraState()
                elif event.key == pygame.K_F3:
                    self.show_debug = not self.show_debug
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.zoom = self.zoom.changed(steps=1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.zoom = self.zoom.changed(steps=-1)

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
            delta_seconds=delta_seconds * (20 / self.zoom.tile_size_pixels),
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
        self.loading = any(not self.runtime.contains(coordinate) for coordinate in nearby)
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
        self.loading = False

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
            if self.show_grid:
                self._draw_grid(position=position, surface=surface)
            if self.show_chunk_boundaries:
                pygame.draw.rect(
                    self.screen,
                    (255, 220, 80),
                    (*position, surface.get_width(), surface.get_height()),
                    2,
                )
        self._render_water_overlay(viewport)
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
            tile_size_pixels=self.zoom.tile_size_pixels,
        )

    def _chunk_surface(self, terrain: object) -> pygame.Surface:
        if not isinstance(terrain, ChunkTerrain):
            raise TypeError("terrain must be a ChunkTerrain.")
        key = terrain_surface_cache_key(
            terrain,
            tile_size_pixels=self.zoom.tile_size_pixels,
        )
        cached = self._surface_cache.get(terrain.chunk_coordinate)
        if cached is not None and cached[0] == key:
            return cached[1]
        size = CHUNK_SIZE * self.zoom.tile_size_pixels
        surface = pygame.Surface((size, size))
        for tile in terrain:
            world_x = terrain.chunk_coordinate.x * CHUNK_SIZE + tile.coordinate.x
            world_y = terrain.chunk_coordinate.y * CHUNK_SIZE + tile.coordinate.y
            west = self._tile_at_world(world_x - 1, world_y)
            east = self._tile_at_world(world_x + 1, world_y)
            north = self._tile_at_world(world_x, world_y - 1)
            south = self._tile_at_world(world_x, world_y + 1)
            strength = 2 if tile.terrain_type in (TerrainType.HILLS, TerrainType.MOUNTAINS) else 1
            light = slope_light(
                centre=tile.elevation.metres,
                west=west.elevation.metres,
                east=east.elevation.metres,
                north=north.elevation.metres,
                south=south.elevation.metres,
                strength=strength,
            )
            rectangle = pygame.Rect(
                tile.coordinate.x * self.zoom.tile_size_pixels,
                tile.coordinate.y * self.zoom.tile_size_pixels,
                self.zoom.tile_size_pixels,
                self.zoom.tile_size_pixels,
            )
            surface.fill(
                terrain_colour(
                    terrain_type=tile.terrain_type,
                    elevation=tile.elevation.metres,
                    light=light,
                ),
                rectangle,
            )
            self._draw_transition(
                surface=surface,
                rectangle=rectangle,
                seed=terrain.terrain_seed,
                world_x=world_x,
                world_y=world_y,
                terrain_type=tile.terrain_type,
                neighbours=(
                    west.terrain_type,
                    east.terrain_type,
                    north.terrain_type,
                    south.terrain_type,
                ),
            )
            self._draw_details(
                surface=surface,
                rectangle=rectangle,
                seed=terrain.terrain_seed,
                world_x=world_x,
                world_y=world_y,
                terrain_type=tile.terrain_type,
            )
        self._surface_cache[terrain.chunk_coordinate] = (key, surface)
        return surface

    def _draw_grid(self, *, position: tuple[int, int], surface: pygame.Surface) -> None:
        assert self.screen is not None
        size = self.zoom.tile_size_pixels
        for offset in range(0, surface.get_width() + 1, size):
            pygame.draw.line(
                self.screen,
                (34, 39, 37),
                (position[0] + offset, position[1]),
                (position[0] + offset, position[1] + surface.get_height()),
                1,
            )
        for offset in range(0, surface.get_height() + 1, size):
            pygame.draw.line(
                self.screen,
                (34, 39, 37),
                (position[0], position[1] + offset),
                (position[0] + surface.get_width(), position[1] + offset),
                1,
            )

    def _tile_at_world(self, world_x: int, world_y: int) -> TerrainTile:
        coordinate = ChunkCoordinate(x=world_x // CHUNK_SIZE, y=world_y // CHUNK_SIZE)
        terrain = self.runtime.get_or_generate(coordinate)
        return terrain.tile_at(LocalTileCoordinate(x=world_x % CHUNK_SIZE, y=world_y % CHUNK_SIZE))

    def _draw_transition(
        self,
        *,
        surface: pygame.Surface,
        rectangle: pygame.Rect,
        seed: int,
        world_x: int,
        world_y: int,
        terrain_type: TerrainType,
        neighbours: tuple[TerrainType, ...],
    ) -> None:
        for index, neighbour in enumerate(neighbours):
            if not transition_mask(
                seed=seed,
                world_x=world_x,
                world_y=world_y,
                terrain_type=terrain_type,
                neighbour_type=neighbour,
            ):
                continue
            colour = TERRAIN_PALETTE[neighbour]
            inset = max(1, self.zoom.tile_size_pixels // 5)
            strips = (
                pygame.Rect(rectangle.left, rectangle.top, inset, rectangle.height),
                pygame.Rect(rectangle.right - inset, rectangle.top, inset, rectangle.height),
                pygame.Rect(rectangle.left, rectangle.top, rectangle.width, inset),
                pygame.Rect(rectangle.left, rectangle.bottom - inset, rectangle.width, inset),
            )
            overlay = pygame.Surface(strips[index].size, pygame.SRCALPHA)
            overlay.fill((*colour, 72))
            surface.blit(overlay, strips[index].topleft)

    def _draw_details(
        self,
        *,
        surface: pygame.Surface,
        rectangle: pygame.Rect,
        seed: int,
        world_x: int,
        world_y: int,
        terrain_type: TerrainType,
    ) -> None:
        colours = {
            "wave": (115, 174, 196),
            "ripple": (127, 205, 204),
            "shell": (242, 226, 174),
            "grass": (38, 83, 42),
            "shrub": (47, 68, 39),
            "rock": (49, 47, 46),
        }
        for detail in visual_details(
            seed=seed,
            world_x=world_x,
            world_y=world_y,
            terrain_type=terrain_type,
        ):
            x = rectangle.left + detail.offset_x_eighths * rectangle.width // 8
            y = rectangle.top + detail.offset_y_eighths * rectangle.height // 8
            radius = max(1, rectangle.width // 12)
            pygame.draw.circle(surface, colours[detail.kind], (x, y), radius)

    def _render_water_overlay(self, viewport: TerrainViewport) -> None:
        assert self.screen is not None
        animation_tick = pygame.time.get_ticks() // 180
        for coordinate in self._visible_coordinates:
            terrain = self.runtime.terrain_at(coordinate)
            origin = coordinate.to_world_origin()
            position = viewport.world_to_screen(
                camera=self.camera,
                world_x=origin.x,
                world_y=origin.y,
            )
            for tile in terrain:
                if tile.terrain_type not in (TerrainType.DEEP_WATER, TerrainType.SHALLOW_WATER):
                    continue
                world_x = origin.x + tile.coordinate.x
                world_y = origin.y + tile.coordinate.y
                if (
                    water_wave_phase(
                        world_x=world_x,
                        world_y=world_y,
                        animation_tick=animation_tick,
                    )
                    != 0
                ):
                    continue
                start = (
                    position[0] + tile.coordinate.x * self.zoom.tile_size_pixels,
                    position[1] + tile.coordinate.y * self.zoom.tile_size_pixels,
                )
                pygame.draw.line(
                    self.screen,
                    (126, 204, 215),
                    start,
                    (start[0] + max(2, self.zoom.tile_size_pixels // 2), start[1]),
                    1,
                )

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
            f"World  {hud.camera_tile.x}, {hud.camera_tile.y}",
            f"Chunk  {hud.chunk_coordinate.x}, {hud.chunk_coordinate.y}   "
            f"Region  {hud.region_coordinate.x}, {hud.region_coordinate.y}",
            f"Seed {hud.world_seed}   Chunks {hud.cached_chunk_count}",
        ]
        if self.loading:
            lines.append("Generating terrain...")
        if self.show_debug:
            lines.extend(
                (
                    "Runtime / repository revision: "
                    f"{hud.terrain_runtime_revision} / {hud.repository_revision}",
                    f"Cache hits / misses: {hud.cache_hits} / {hud.cache_misses}",
                    f"Generated / failed: {hud.successful_generations} / {hud.failed_generations}",
                )
            )
        if self.show_help:
            lines.extend(
                (
                    "",
                    "WASD / arrows: move   Shift: faster   R: origin",
                    "+/- or wheel: zoom   G: grid   C: chunks   F3: debug",
                    "F1/H: help   Esc: exit",
                )
            )
        line_height = self._font.get_linesize()
        overlay = pygame.Surface(
            (520, (len(lines) * line_height) + 16),
            pygame.SRCALPHA,
        )
        overlay.fill((0, 0, 0, 185))
        pygame.draw.rect(overlay, (104, 124, 116, 210), overlay.get_rect(), 1, 6)
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
