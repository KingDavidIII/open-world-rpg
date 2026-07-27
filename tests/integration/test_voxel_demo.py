"""Bounded OpenGL smoke acceptance for the voxel terrain prototype."""

from __future__ import annotations

from pathlib import Path

import pygame
import pytest

from open_world_rpg.gameplay import ItemType
from open_world_rpg.ui.voxel.application import (
    VoxelContextUnavailableError,
    VoxelPrototypeApplication,
    VoxelPrototypeConfig,
    VoxelPrototypeError,
)
from open_world_rpg.ui.voxel.collision import RayHit, ray_cast
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
        assert application.show_help
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
        application._update_mining(10_000_000)  # type: ignore[attr-defined]
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
        application.inventory.select_hotbar(2)
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


def test_voxel_edits_survive_save_restart_and_atomic_reload(tmp_path: Path) -> None:
    save_path = tmp_path / "voxel-acceptance.json"
    config = VoxelPrototypeConfig(
        width_pixels=320,
        height_pixels=180,
        render_distance=0,
        target_fps=120,
        hidden_window=True,
        save_path=save_path,
        terrain_config=TerrainGenerationConfig(octave_count=1),
    )
    first = VoxelPrototypeApplication(config=config)
    try:
        try:
            first.initialise()
        except VoxelContextUnavailableError as error:
            pytest.skip(str(error))
        removed = WorldBlockCoordinate(
            x=first.spawn_x,
            y=first._height_at(first.spawn_x, first.spawn_z),  # type: ignore[attr-defined]
            z=first.spawn_z,
        )
        first.edits.set_block(removed, BlockMaterial.AIR)
        grass = WorldBlockCoordinate(x=-1, y=20, z=-1)
        boundary = WorldBlockCoordinate(x=16, y=21, z=0)
        first.edits.set_block(grass, BlockMaterial.GRASS)
        first.edits.set_block(boundary, BlockMaterial.STONE)
        first.inventory.select_hotbar(4)
        placement = first.interactions.place_inventory_block(
            target=RayHit(
                x=30,
                y=29,
                z=30,
                distance=1,
                material=BlockMaterial.STONE,
                face_normal=(0, 1, 0),
            ),
            inventory=first.inventory,
            player=first.player,
            now=0,
        )
        first._apply_interaction(placement)  # type: ignore[attr-defined]
        assert first.inventory.total_quantity(ItemType.STONE_BLOCK) == 7
        drop_source = WorldBlockCoordinate(x=-2, y=20, z=-2)
        broken = first.interactions.break_block(
            target=RayHit(
                x=drop_source.x,
                y=drop_source.y,
                z=drop_source.z,
                distance=1,
                material=BlockMaterial.GRASS,
            ),
            now=0,
        )
        first._apply_interaction(broken)  # type: ignore[attr-defined]
        assert len(first.dropped_items) == 1
        first.dirty = True
        saved_revision = first.edits.revision
        assert first._save_edits()  # type: ignore[attr-defined]
    finally:
        first.shutdown()

    second = VoxelPrototypeApplication(
        config=VoxelPrototypeConfig(
            width_pixels=320,
            height_pixels=180,
            render_distance=0,
            target_fps=120,
            hidden_window=True,
            save_path=save_path,
            load_on_start=True,
            terrain_config=TerrainGenerationConfig(octave_count=1),
        )
    )
    try:
        try:
            second.initialise()
        except VoxelContextUnavailableError as error:
            pytest.skip(str(error))
        assert second.editable_world.block_at(removed) is BlockMaterial.AIR
        assert second.editable_world.block_at(grass) is BlockMaterial.GRASS
        assert second.editable_world.block_at(boundary) is BlockMaterial.STONE
        assert second.edits.revision == saved_revision
        assert second.inventory.selected_hotbar_index == 4
        assert second.inventory.total_quantity(ItemType.STONE_BLOCK) == 7
        assert len(second.dropped_items) == 1
        assert not second.dirty
        assert second._solid_at(boundary.x, boundary.y, boundary.z)  # type: ignore[attr-defined]
        hit = ray_cast(
            origin=(boundary.x + 0.5, boundary.y + 2.5, boundary.z + 0.5),
            direction=(0.0, -1.0, 0.0),
            block_at=second.editable_world.material_at,
        )
        assert hit is not None
        assert hit.coordinate == boundary
        second.render()

        newer = WorldBlockCoordinate(x=-16, y=22, z=15)
        second.edits.set_block(newer, BlockMaterial.SAND)
        second.dirty = True
        assert second._save_edits()  # type: ignore[attr-defined]
        second.edits.set_block(newer, BlockMaterial.SNOW)
        second.dirty = True
        assert second._load_edits()  # type: ignore[attr-defined]
        assert second.editable_world.block_at(newer) is BlockMaterial.SAND
        assert not second.dirty
    finally:
        second.shutdown()


def test_requested_missing_voxel_save_fails_startup_cleanly(tmp_path: Path) -> None:
    application = VoxelPrototypeApplication(
        config=VoxelPrototypeConfig(
            width_pixels=160,
            height_pixels=90,
            render_distance=0,
            hidden_window=True,
            save_path=tmp_path / "missing.json",
            load_on_start=True,
            terrain_config=TerrainGenerationConfig(octave_count=1),
        )
    )
    try:
        try:
            application.initialise()
        except VoxelContextUnavailableError as error:
            pytest.skip(str(error))
        except VoxelPrototypeError as error:
            assert "save restoration" in str(error)
        else:
            pytest.fail("Expected the requested missing save to fail startup.")
    finally:
        application.shutdown()
