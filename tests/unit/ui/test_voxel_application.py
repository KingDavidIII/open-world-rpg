"""Deterministic controller, matrix, entry-point, and cleanup coverage."""

from __future__ import annotations

import struct
from typing import Any, cast

import pygame
import pytest

import open_world_rpg.ui.voxel_demo as voxel_demo
from open_world_rpg.ui.voxel.application import (
    GpuChunk,
    VoxelContextUnavailableError,
    VoxelPrototypeApplication,
    VoxelPrototypeConfig,
    VoxelPrototypeError,
    _perspective,
    _view_matrix,
    _water_render_order,
)
from open_world_rpg.ui.voxel.collision import RayHit
from open_world_rpg.ui.voxel.interaction import InteractionOutcome, InteractionResult
from open_world_rpg.world import (
    BlockMaterial,
    ChunkCoordinate,
    TerrainGenerationConfig,
    WorldBlockCoordinate,
)


def test_projection_and_view_matrices_are_finite_deterministic_matrices() -> None:
    projection = _perspective(field_of_view=72.0, aspect=16 / 9, near=0.1, far=100.0)
    view = _view_matrix(position=(1.0, 2.0, 3.0), forward=(0.0, 0.0, -1.0))
    vertical_view = _view_matrix(position=(0.0, 0.0, 0.0), forward=(0.0, 1.0, 0.0))
    assert len(struct.unpack("16f", projection)) == 16
    assert len(struct.unpack("16f", view)) == 16
    assert len(struct.unpack("16f", vertical_view)) == 16
    assert projection == _perspective(field_of_view=72.0, aspect=16 / 9, near=0.1, far=100.0)


def test_water_chunks_render_far_to_near_with_deterministic_ties() -> None:
    coordinates = (
        ChunkCoordinate(x=0, y=0),
        ChunkCoordinate(x=2, y=0),
        ChunkCoordinate(x=-1, y=0),
    )
    assert _water_render_order(coordinates, player_x=8.0, player_z=8.0) == (
        ChunkCoordinate(x=2, y=0),
        ChunkCoordinate(x=-1, y=0),
        ChunkCoordinate(x=0, y=0),
    )


def test_run_validates_bounded_frame_count() -> None:
    application = VoxelPrototypeApplication()
    with pytest.raises(TypeError, match="max_frames"):
        application.run(max_frames=True)
    with pytest.raises(ValueError, match="greater than zero"):
        application.run(max_frames=0)
    with pytest.raises(VoxelPrototypeError, match="not initialised"):
        application.render()
    application._render_target_outline(  # type: ignore[attr-defined]
        RayHit(x=0, y=0, z=0, distance=0.0)
    )
    with pytest.raises(TypeError):
        VoxelPrototypeConfig(interaction_reach=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VoxelPrototypeConfig(interaction_reach=float("inf"))
    with pytest.raises(ValueError):
        VoxelPrototypeConfig(interaction_reach=0)
    with pytest.raises(ValueError):
        VoxelPrototypeConfig(break_cooldown=-1)
    with pytest.raises(ValueError):
        VoxelPrototypeConfig(placement_cooldown=-1)


def test_uncaptured_escape_exits_and_unhandled_key_is_harmless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.running = True
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE),
        ],
    )
    application.process_events()
    assert not application.running


def test_render_distance_controls_clamp_between_one_and_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    stream_calls = 0

    def record_stream() -> None:
        nonlocal stream_calls
        stream_calls += 1

    monkeypatch.setattr(application, "_stream", record_stream)
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [
            *(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F5) for _ in range(3)),
            *(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F6) for _ in range(6)),
        ],
    )
    application.process_events()
    assert application.render_distance == 4
    assert stream_calls == 9


def test_hotbar_keyboard_wheel_and_mouse_edit_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.mouse_captured = True
    application.target = RayHit(
        x=0,
        y=1,
        z=0,
        distance=1.0,
        material=BlockMaterial.STONE,
        face_normal=(0, 1, 0),
    )
    calls: list[str] = []

    def break_block(**_kwargs: object) -> InteractionOutcome:
        calls.append("break")
        return InteractionOutcome(result=InteractionResult.NO_TARGET)

    def place_block(**_kwargs: object) -> InteractionOutcome:
        calls.append("place")
        return InteractionOutcome(result=InteractionResult.NO_TARGET)

    monkeypatch.setattr(application.interactions, "break_block", break_block)
    monkeypatch.setattr(application.interactions, "place_block", place_block)
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_5),
            pygame.event.Event(pygame.MOUSEWHEEL, y=-1),
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=4),
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=5),
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1),
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3),
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=2),
        ],
    )
    application.process_events()
    assert application.hotbar.selected_index == 5
    assert application.hotbar.selected_material is None
    assert calls == ["break", "place"]


def test_successful_interaction_releases_only_invalidated_cached_meshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Cached:
        released = False

        def release(self) -> None:
            self.released = True

    application = VoxelPrototypeApplication()
    affected = ChunkCoordinate(x=0, y=0)
    retained = ChunkCoordinate(x=1, y=1)
    removed = Cached()
    kept = Cached()
    application._gpu_chunks[affected] = cast(Any, removed)  # type: ignore[attr-defined]
    application._gpu_chunks[retained] = cast(Any, kept)  # type: ignore[attr-defined]
    stream_calls = 0

    def stream() -> None:
        nonlocal stream_calls
        stream_calls += 1

    monkeypatch.setattr(application, "_stream", stream)
    monkeypatch.setattr(
        "open_world_rpg.ui.voxel.application.ray_cast",
        lambda **_kwargs: None,
    )
    application._apply_interaction(  # type: ignore[attr-defined]
        InteractionOutcome(
            result=InteractionResult.BROKEN,
            coordinate=WorldBlockCoordinate(x=0, y=1, z=0),
            invalidated_chunks=(affected, ChunkCoordinate(x=9, y=9)),
        )
    )
    assert removed.released
    assert not kept.released
    assert affected not in application._gpu_chunks  # type: ignore[attr-defined]
    assert retained in application._gpu_chunks  # type: ignore[attr-defined]
    assert stream_calls == 1
    application._apply_interaction(  # type: ignore[attr-defined]
        InteractionOutcome(result=InteractionResult.PLAYER_INTERSECTION)
    )
    assert application.last_interaction is InteractionResult.PLAYER_INTERSECTION


def test_caption_and_hud_cover_help_debug_loading_and_uninitialised_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.show_help = True
    application.show_debug = False
    assert "WASD move" in application._caption(0)  # type: ignore[attr-defined]
    application.show_debug = True
    application.target = RayHit(x=1, y=2, z=3, distance=4.0)
    assert "target 1,2,3" in application._caption(12)  # type: ignore[attr-defined]

    application._render_hud(0)  # type: ignore[attr-defined]
    application._draw_hotbar(pygame.Surface((1024, 512), pygame.SRCALPHA))  # type: ignore[attr-defined]

    class Resource:
        def write(self, _data: bytes) -> None:
            return None

        def use(self, *, location: int) -> None:
            assert location == 1

        def render(self, _mode: int) -> None:
            return None

    class Context:
        def disable(self, _flag: int) -> None:
            return None

        def enable(self, _flag: int) -> None:
            return None

    class Font:
        def render(
            self, _text: str, _antialias: bool, _colour: tuple[int, int, int]
        ) -> pygame.Surface:
            return pygame.Surface((1, 1), pygame.SRCALPHA)

    application.context = cast(Any, Context())
    application._hud_texture = cast(Any, Resource())  # type: ignore[attr-defined]
    application._hud_array = cast(Any, Resource())  # type: ignore[attr-defined]
    application._font = cast(Any, Font())  # type: ignore[attr-defined]
    application.loading = True
    application._selection_changed_at = 0.0  # type: ignore[attr-defined]
    application._feedback_until = 2.0  # type: ignore[attr-defined]
    application.last_interaction = InteractionResult.BROKEN
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 1000)
    application._render_hud(12)  # type: ignore[attr-defined]
    assert application.hud_snapshot is not None
    assert application.hud_snapshot.loading


def test_stream_without_context_maintains_domain_cache_only() -> None:
    application = VoxelPrototypeApplication(
        config=VoxelPrototypeConfig(
            render_distance=0,
            terrain_config=application_config(),
        )
    )
    application._stream()  # type: ignore[attr-defined]
    assert application.runtime.coordinates()
    assert not application._gpu_chunks  # type: ignore[attr-defined]
    application.shutdown()


def application_config() -> TerrainGenerationConfig:
    return TerrainGenerationConfig(octave_count=1)


def test_context_failure_is_precise_and_always_quits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    quit_calls = 0

    def fail_initialisation() -> tuple[int, int]:
        raise pygame.error("driver unavailable")

    def record_quit() -> None:
        nonlocal quit_calls
        quit_calls += 1

    monkeypatch.setattr(pygame, "init", fail_initialisation)
    monkeypatch.setattr(pygame, "quit", record_quit)
    monkeypatch.setattr(application, "_capture_mouse", lambda _captured: None)
    with pytest.raises(VoxelContextUnavailableError) as raised:
        application.initialise()
    assert isinstance(raised.value.__cause__, pygame.error)
    assert quit_calls == 1


def test_resource_failure_is_not_misreported_as_context_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenContext:
        blend_func: tuple[int, int]

        def enable(self, _flags: int) -> None:
            return None

        def program(self, **_kwargs: object) -> None:
            raise RuntimeError("shader compilation failed")

        def release(self) -> None:
            return None

    application = VoxelPrototypeApplication()
    monkeypatch.setattr(pygame.display, "set_mode", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("moderngl.create_context", lambda **_kwargs: BrokenContext())
    monkeypatch.setattr(application, "_capture_mouse", lambda _captured: None)
    with pytest.raises(VoxelPrototypeError, match="during resources") as raised:
        application.initialise()
    assert not isinstance(raised.value, VoxelContextUnavailableError)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert not pygame.get_init()


def test_gpu_chunk_releases_buffer_even_if_vertex_array_release_fails() -> None:
    class FailedArray:
        def release(self) -> None:
            raise RuntimeError("array")

    class Buffer:
        released = False

        def release(self) -> None:
            self.released = True

    buffer = Buffer()
    chunk = GpuChunk(
        key=Any,  # type: ignore[arg-type]
        opaque_buffer=buffer,  # type: ignore[arg-type]
        opaque_array=FailedArray(),  # type: ignore[arg-type]
        water_buffer=None,
        water_array=None,
        mesh=Any,  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="array"):
        chunk.release()
    assert buffer.released


def test_shutdown_tolerates_resource_release_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingResource:
        def release(self) -> None:
            raise RuntimeError("release")

    application = VoxelPrototypeApplication()
    application._gpu_chunks[ChunkCoordinate(x=0, y=0)] = cast(  # type: ignore[attr-defined]
        Any, FailingResource()
    )
    application._target_array = cast(Any, FailingResource())  # type: ignore[attr-defined]
    application.context = cast(Any, FailingResource())

    def fail_mouse(_captured: bool) -> None:
        raise pygame.error("mouse")

    monkeypatch.setattr(application, "_capture_mouse", fail_mouse)
    application.shutdown()
    assert not application.mouse_captured
    assert application.context is None


def test_voxel_entry_point_selects_smoke_and_interactive_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[VoxelPrototypeConfig, int | None]] = []

    class Application:
        def __init__(self, *, config: VoxelPrototypeConfig) -> None:
            self.config = config

        def run(self, *, max_frames: int | None = None) -> int:
            calls.append((self.config, max_frames))
            return 0

    monkeypatch.setattr(voxel_demo, "VoxelPrototypeApplication", Application)
    assert voxel_demo.main(["--smoke-test"]) == 0
    assert calls[-1][0].hidden_window
    assert calls[-1][1] == 3
    assert voxel_demo.main([]) == 0
    assert not calls[-1][0].hidden_window
    assert calls[-1][1] is None


def test_voxel_entry_point_reports_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Application:
        def __init__(self, *, config: VoxelPrototypeConfig) -> None:
            pass

        def run(self, *, max_frames: int | None = None) -> int:
            raise RuntimeError("failure")

    monkeypatch.setattr(voxel_demo, "VoxelPrototypeApplication", Application)
    assert voxel_demo.main([]) == 1
