"""Headless SDL smoke coverage for the playable terrain prototype."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

import open_world_rpg.ui.terrain_demo as terrain_demo
from open_world_rpg.ui.terrain_demo import (
    TerrainPrototypeApplication,
    TerrainPrototypeConfig,
    TerrainPrototypeError,
)
from open_world_rpg.world import ChunkState, TerrainGenerationConfig, TerrainType


def small_config() -> TerrainPrototypeConfig:
    return TerrainPrototypeConfig(
        width_pixels=64,
        height_pixels=64,
        tile_size_pixels=16,
        preload_margin_chunks=0,
        target_fps=120,
        world_seed=9,
        terrain_config=TerrainGenerationConfig(octave_count=1),
    )


def test_headless_demo_generates_renders_pixels_and_shuts_down() -> None:
    application = TerrainPrototypeApplication(config=small_config())
    application.initialise()
    assert application.screen is not None
    application.camera = application.camera.moved(
        horizontal=1,
        vertical=1,
        delta_seconds=0.025,
    )

    assert application.run(max_frames=2) == 0
    assert application._last_frame is not None  # type: ignore[attr-defined]
    assert application._last_frame.get_at((32, 32))[:3] != (10, 12, 18)  # type: ignore[attr-defined]
    cached_surface = next(iter(application._surface_cache.values()))[1]  # type: ignore[attr-defined]
    assert cached_surface.get_at((1, 1))[:3] != (10, 12, 18)
    assert len(application.runtime.service.repository) > 0
    assert application.runtime.revision > 0
    assert application.runtime.service.snapshot().successful_generations > 0
    assert all(
        application.runtime.metadata_at(coordinate).state is ChunkState.SUSPENDED
        for coordinate in application.runtime.coordinates()
        if coordinate in application._visible_coordinates  # type: ignore[attr-defined]
    )
    assert not pygame.get_init()


def test_demo_controls_resize_and_surface_cache_invalidation() -> None:
    application = TerrainPrototypeApplication(config=small_config())
    application.initialise()
    application.update(0.0)
    application.render()
    assert application._surface_cache  # type: ignore[attr-defined]
    original_help = application.show_help
    original_boundaries = application.show_chunk_boundaries
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F3))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, y=1))
    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, size=(80, 48)))

    application.process_events()

    assert application.show_help is not original_help
    assert application.show_grid
    assert application.show_chunk_boundaries is not original_boundaries
    assert application.camera.x_tiles == application.camera.y_tiles == 0.0
    assert application.show_debug
    assert application.zoom.tile_size_pixels == 18
    assert application.screen is not None
    assert application.screen.get_size() == (80, 48)
    assert application._surface_cache  # type: ignore[attr-defined]
    application.loading = True
    application.render()
    application.shutdown()


def test_demo_escape_and_window_close_stop_loop() -> None:
    application = TerrainPrototypeApplication(config=small_config())
    application.initialise()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    application.process_events()
    assert not application.running
    application.shutdown()

    application = TerrainPrototypeApplication(config=small_config())
    application.initialise()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    application.process_events()
    assert not application.running
    application.shutdown()

    application = TerrainPrototypeApplication(config=small_config())
    application.initialise()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    application.process_events()
    assert application.camera.x_tiles == application.camera.y_tiles == 0.0
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert application.run() == 0


def test_demo_validates_config_run_and_render_state() -> None:
    with pytest.raises(TypeError, match="config must be"):
        TerrainPrototypeApplication(config=object())  # type: ignore[arg-type]
    application = TerrainPrototypeApplication(config=small_config())
    with pytest.raises(TypeError, match="max_frames"):
        application.run(max_frames=True)
    with pytest.raises(ValueError, match="greater than zero"):
        application.run(max_frames=0)
    with pytest.raises(TerrainPrototypeError, match="not initialised"):
        application.render()
    with pytest.raises(TypeError, match="terrain must be"):
        application._chunk_surface(object())  # type: ignore[attr-defined]


def test_demo_initialisation_fallback_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_set_mode = pygame.display.set_mode
    calls = 0

    def set_mode_with_vsync_fallback(
        size: tuple[int, int],
        flags: int = 0,
        depth: int = 0,
        display: int = 0,
        vsync: int = 0,
    ) -> pygame.Surface:
        nonlocal calls
        calls += 1
        if vsync:
            raise pygame.error("vsync unavailable")
        return original_set_mode(size, flags, depth, display, vsync)

    monkeypatch.setattr(pygame.display, "set_mode", set_mode_with_vsync_fallback)
    application = TerrainPrototypeApplication(config=small_config())
    application.initialise()
    assert calls == 2
    application.shutdown()

    monkeypatch.setattr(pygame, "init", lambda: (_ for _ in ()).throw(RuntimeError()))
    application = TerrainPrototypeApplication(config=small_config())
    with pytest.raises(TerrainPrototypeError, match="Could not initialise"):
        application.initialise()
    assert not pygame.get_init()


def test_demo_auto_initialises_caches_draws_grid_and_suspends_far_chunks() -> None:
    application = TerrainPrototypeApplication(config=small_config())
    assert application._viewport().width_pixels == 64  # type: ignore[attr-defined]
    assert application.run(max_frames=1) == 0

    application.initialise()
    application.show_help = False
    application.show_grid = True
    application.show_chunk_boundaries = False
    application.update(0.0)
    terrain = application.runtime.terrain_at(application._visible_coordinates[0])  # type: ignore[attr-defined]
    first = application._chunk_surface(terrain)  # type: ignore[attr-defined]
    assert application._chunk_surface(terrain) is first  # type: ignore[attr-defined]
    application.render()

    original_visible = application._visible_coordinates  # type: ignore[attr-defined]
    application.camera = application.camera.moved(
        horizontal=1,
        vertical=0,
        delta_seconds=100.0,
        fast=True,
    )
    application.update(0.0)
    assert all(
        application.runtime.metadata_at(coordinate).state is ChunkState.SUSPENDED
        for coordinate in original_visible
    )
    application.shutdown()


def test_hud_guard_and_main_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = TerrainPrototypeApplication(config=small_config())
    application._render_hud()  # type: ignore[attr-defined]

    class SuccessfulApplication:
        def run(self) -> int:
            return 0

    class FailedApplication:
        def run(self) -> int:
            raise RuntimeError("failure")

    monkeypatch.setattr(terrain_demo, "TerrainPrototypeApplication", SuccessfulApplication)
    assert terrain_demo.main() == 0
    monkeypatch.setattr(terrain_demo, "TerrainPrototypeApplication", FailedApplication)
    assert terrain_demo.main() == 1


def test_transition_and_non_water_overlay_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = TerrainPrototypeApplication(
        config=TerrainPrototypeConfig(
            width_pixels=320,
            height_pixels=160,
            tile_size_pixels=16,
            preload_margin_chunks=0,
            world_seed=0,
            terrain_config=TerrainGenerationConfig(octave_count=1),
        )
    )
    application.initialise()
    surface = pygame.Surface((16, 16))
    monkeypatch.setattr(terrain_demo, "transition_mask", lambda **_: True)
    application._draw_transition(  # type: ignore[attr-defined]
        surface=surface,
        rectangle=pygame.Rect(0, 0, 16, 16),
        seed=1,
        world_x=0,
        world_y=0,
        terrain_type=TerrainType.PLAINS,
        neighbours=(
            TerrainType.HILLS,
            TerrainType.COAST,
            TerrainType.MOUNTAINS,
            TerrainType.SHALLOW_WATER,
        ),
    )
    assert surface.get_at((0, 0)).a > 0

    application.update(0.0)
    monkeypatch.setattr(terrain_demo, "water_wave_phase", lambda **_: 0)
    application._render_water_overlay(application._viewport())  # type: ignore[attr-defined]
    application.shutdown()
