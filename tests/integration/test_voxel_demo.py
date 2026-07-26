"""Bounded OpenGL smoke acceptance for the voxel terrain prototype."""

from __future__ import annotations

import pygame
import pytest

from open_world_rpg.ui.voxel.application import (
    VoxelContextUnavailableError,
    VoxelPrototypeApplication,
    VoxelPrototypeConfig,
)
from open_world_rpg.ui.voxel.collision import RayHit
from open_world_rpg.world import (
    CHUNK_SIZE,
    BlockMaterial,
    ChunkState,
    TerrainGenerationConfig,
    WorldBlockCoordinate,
)


@pytest.fixture(autouse=True)
def isolated_opengl_sdl(monkeypatch: pytest.MonkeyPatch):
    """Start SDL clean and restore the caller's driver configuration."""
    pygame.quit()
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)
    try:
        yield
    finally:
        pygame.quit()


def test_voxel_demo_builds_uploads_and_draws_bounded_frame() -> None:
    application = VoxelPrototypeApplication(
        config=VoxelPrototypeConfig(
            width_pixels=320,
            height_pixels=180,
            render_distance=0,
            target_fps=120,
            hidden_window=True,
            terrain_config=TerrainGenerationConfig(octave_count=1),
        )
    )
    try:
        try:
            assert application.run(max_frames=2) == 0
        except VoxelContextUnavailableError as error:
            pytest.skip(str(error))
        assert application.runtime.coordinates()
        assert application.runtime.service.snapshot().successful_generations > 0
        assert application.context is None
    finally:
        application.shutdown()


def test_voxel_controls_targeting_and_shutdown_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication(
        config=VoxelPrototypeConfig(
            width_pixels=320,
            height_pixels=180,
            render_distance=0,
            target_fps=120,
            terrain_config=TerrainGenerationConfig(octave_count=1),
        )
    )
    try:
        try:
            application.initialise()
        except VoxelContextUnavailableError as error:
            pytest.skip(str(error))
        original_camera = application.camera
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, rel=(8, -4)))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F3))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        application.process_events()
        assert application.camera != original_camera
        assert not application.show_help
        assert application.show_debug
        assert application.player.flying

        surface_y = application._height_at(application.spawn_x, application.spawn_z)  # type: ignore[attr-defined]
        surface = WorldBlockCoordinate(
            x=application.spawn_x,
            y=surface_y,
            z=application.spawn_z,
        )
        application.target = RayHit(
            x=surface.x,
            y=surface.y,
            z=surface.z,
            distance=1.0,
            material=application.editable_world.block_at(surface),
            face_normal=(0, 1, 0),
        )
        revision = application.edits.revision
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1))
        application.process_events()
        assert application.edits.revision == revision + 1
        assert application.editable_world.block_at(surface) is BlockMaterial.AIR
        support = surface.offset(y=-1)
        application.target = RayHit(
            x=support.x,
            y=support.y,
            z=support.z,
            distance=1.0,
            material=application.editable_world.block_at(support),
            face_normal=(0, 1, 0),
        )
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3))
        application.process_events()
        assert application.editable_world.block_at(surface) is BlockMaterial.GRASS
        assert application.edits.revision == revision + 2
        application._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.0  # type: ignore[attr-defined]
        application.render()

        class FlyingKeys:
            def __getitem__(self, key: int) -> bool:
                return key in (pygame.K_w, pygame.K_d, pygame.K_LSHIFT, pygame.K_SPACE)

        monkeypatch.setattr(pygame.key, "get_pressed", FlyingKeys)
        old_y = application.player.y
        application.update(0.01)
        assert application.player.y > old_y

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
        application.process_events()
        assert application.player.grounded
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        application.process_events()

        application.target = RayHit(x=8, y=8, z=8, distance=1.0)
        application.render()
        assert "target 8,8,8" in application._caption(1)  # type: ignore[attr-defined]
        sky = application._sky_array  # type: ignore[attr-defined]
        outline = application._outline_program  # type: ignore[attr-defined]
        application._sky_array = None  # type: ignore[attr-defined]
        application._outline_program = None  # type: ignore[attr-defined]
        existing = next(iter(application._gpu_chunks.values()))  # type: ignore[attr-defined]
        water_array = existing.water_array
        existing.water_array = None
        application.render()
        existing.water_array = water_array
        application._sky_array = sky  # type: ignore[attr-defined]
        application._outline_program = outline  # type: ignore[attr-defined]
        crosshair = application._crosshair_array  # type: ignore[attr-defined]
        application._crosshair_array = None  # type: ignore[attr-defined]
        application.render()
        application._crosshair_array = crosshair  # type: ignore[attr-defined]
        application._stream_signature = None  # type: ignore[attr-defined]
        application._stream()  # type: ignore[attr-defined]
        existing.key = (
            existing.key[0],
            -1,
            existing.key[2],
            existing.key[3],
            existing.key[4],
        )
        application._stream_signature = None  # type: ignore[attr-defined]
        application._stream()  # type: ignore[attr-defined]
        previous_coordinates = set(application._gpu_chunks)  # type: ignore[attr-defined]
        application.player = application.player.__class__(
            x=CHUNK_SIZE * 3,
            y=application.player.y,
            z=CHUNK_SIZE * 3,
            flying=True,
        )
        application._stream()  # type: ignore[attr-defined]
        assert previous_coordinates.isdisjoint(application._gpu_chunks)  # type: ignore[attr-defined]

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        application.process_events()
        assert not application.mouse_captured
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1))
        application.process_events()
        assert application.mouse_captured
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        application.process_events()
        assert not application.mouse_captured
        assert application.running
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        assert application.run() == 0
    finally:
        application.shutdown()
    assert not pygame.get_init()
    assert not application.mouse_captured
    assert all(
        application.runtime.metadata_at(coordinate).state is not ChunkState.ACTIVE
        for coordinate in application.runtime.coordinates()
    )


def test_spawn_generation_failure_is_wrapped_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication(
        config=VoxelPrototypeConfig(
            width_pixels=160,
            height_pixels=90,
            render_distance=0,
            terrain_config=TerrainGenerationConfig(octave_count=1),
        )
    )

    def fail_stream() -> None:
        raise RuntimeError("generation failed")

    monkeypatch.setattr(application, "_stream", fail_stream)
    try:
        application.initialise()
    except VoxelContextUnavailableError as error:
        pytest.skip(str(error))
    except Exception as error:
        assert "terrain startup" in str(error)
        assert isinstance(error.__cause__, RuntimeError)
    else:
        pytest.fail("Expected terrain startup to fail.")
    finally:
        application.shutdown()
    assert not pygame.get_init()
