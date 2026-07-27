"""Deterministic controller, matrix, entry-point, and cleanup coverage."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pygame
import pytest

import open_world_rpg.ui.voxel.application as voxel_application
import open_world_rpg.ui.voxel_demo as voxel_demo
from open_world_rpg.gameplay import (
    CraftingResult,
    ItemStack,
    ItemType,
    MiningStatus,
    PickupResult,
    PlayerVitalsSnapshot,
    ToolInstance,
)
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
from open_world_rpg.ui.voxel.blocks import BlockColumn
from open_world_rpg.ui.voxel.camera import PlayerState
from open_world_rpg.ui.voxel.collision import RayHit
from open_world_rpg.ui.voxel.game_flow import GameFlowAction, VoxelScreen
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
    with pytest.raises(TypeError):
        VoxelPrototypeConfig(save_path="save.json")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        VoxelPrototypeConfig(load_on_start=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        VoxelPrototypeConfig(autosave=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        VoxelPrototypeConfig(bootstrap_inventory=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        VoxelPrototypeConfig(game_flow_enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VoxelPrototypeConfig(load_on_start=True)
    with pytest.raises(ValueError):
        VoxelPrototypeConfig(autosave=True)


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


def test_mouse_capture_reacquires_only_after_escape_and_not_on_immediate_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.running = True
    application.mouse_captured = True
    monkeypatch.setattr(pygame.event, "set_grab", lambda _: None)
    monkeypatch.setattr(pygame.mouse, "set_visible", lambda _: None)
    monkeypatch.setattr(pygame.mouse, "get_rel", lambda: (0, 0))
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE),
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1),
        ],
    )
    application.process_events()
    assert not application.mouse_captured
    assert not application._mining_held  # type: ignore[attr-defined]


def test_mouse_button_down_captures_mouse_when_not_already_captured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.running = True
    application.mouse_captured = False
    monkeypatch.setattr(pygame.event, "set_grab", lambda _: None)
    monkeypatch.setattr(pygame.mouse, "set_visible", lambda _: None)
    monkeypatch.setattr(pygame.mouse, "get_rel", lambda: (0, 0))
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1)],
    )
    application.process_events()
    assert application.mouse_captured


def test_inventory_noop_controls_drop_update_pickup_and_uninitialised_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application._refresh_drop_gpu()  # type: ignore[attr-defined]
    application.mouse_captured = True
    monkeypatch.setattr(application.inventory, "cycle_hotbar", lambda _direction: False)
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [
            pygame.event.Event(pygame.MOUSEWHEEL, y=0),
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=4),
        ],
    )
    application.process_events()
    assert not application.dirty

    class Keys:
        def __getitem__(self, _key: int) -> bool:
            return False

    monkeypatch.setattr(pygame.key, "get_pressed", Keys)
    monkeypatch.setattr(
        voxel_application,
        "move_player",
        lambda **_kwargs: application.player,
    )
    monkeypatch.setattr(voxel_application, "ray_cast", lambda **_kwargs: None)
    monkeypatch.setattr(application, "_stream", lambda: None)
    monkeypatch.setattr(application.dropped_items, "update", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        application.dropped_items,
        "pickup_near",
        lambda **_kwargs: (PickupResult(item=ItemType.STONE_BLOCK, accepted=1),),
    )
    application.update(0.01)
    assert application.dirty
    assert application.last_pickup == "Picked up Stone Block x1"


def test_update_reuses_frame_column_cache_and_preserves_block_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.mouse_captured = True
    edited = WorldBlockCoordinate(x=2, y=10, z=0)
    application.edits.set_block(edited, BlockMaterial.STONE)

    columns = {
        (0, 0): BlockColumn(
            ground_height=5,
            surface_height=5,
            surface=BlockMaterial.GRASS,
            subsurface=BlockMaterial.DIRT,
        ),
        (1, 0): BlockColumn(
            ground_height=2,
            surface_height=5,
            surface=BlockMaterial.SAND,
            subsurface=BlockMaterial.SAND,
            water=BlockMaterial.WATER,
        ),
        (2, 0): BlockColumn(
            ground_height=2,
            surface_height=2,
            surface=BlockMaterial.GRASS,
            subsurface=BlockMaterial.DIRT,
        ),
    }
    lookups: list[tuple[int, int]] = []

    def column_at(x: int, z: int) -> BlockColumn:
        lookups.append((x, z))
        return columns[(x, z)]

    class Keys:
        def __getitem__(self, _key: int) -> bool:
            return False

    def inspect_cached_collision(**kwargs: object) -> PlayerState:
        solid_at = cast(Any, kwargs["solid_at"])
        assert solid_at(0, 5, 0)
        assert solid_at(0, 4, 0)
        assert solid_at(0, 1, 0)
        assert solid_at(0, 5, 0)
        assert not solid_at(0, 6, 0)
        assert not solid_at(1, 3, 0)
        assert not solid_at(1, 5, 0)
        assert solid_at(2, 10, 0)
        return application.player

    monkeypatch.setattr(pygame.key, "get_pressed", Keys)
    monkeypatch.setattr(application, "_column_at", column_at)
    monkeypatch.setattr(voxel_application, "move_player", inspect_cached_collision)
    monkeypatch.setattr(voxel_application, "ray_cast", lambda **_kwargs: None)
    monkeypatch.setattr(application, "_stream", lambda: None)
    monkeypatch.setattr(application.dropped_items, "update", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(application.dropped_items, "pickup_near", lambda **_kwargs: ())

    application.update(0.01)

    assert lookups == [(0, 0), (1, 0), (2, 0)]


def test_survival_update_covers_sprint_jump_fall_damage_and_respawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.mouse_captured = True
    monkeypatch.setattr(application, "_stream", lambda: None)
    monkeypatch.setattr(voxel_application, "ray_cast", lambda **_kwargs: None)
    monkeypatch.setattr(application.dropped_items, "update", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(application.dropped_items, "pickup_near", lambda **_kwargs: ())

    class Keys:
        def __init__(self) -> None:
            self.active = {pygame.K_w, pygame.K_LSHIFT}

        def __getitem__(self, key: int) -> bool:
            return key in self.active

    keys = Keys()
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: keys)
    monkeypatch.setattr(voxel_application, "move_player", lambda **_kwargs: application.player)
    application.update(1.0)
    assert application.vitals.snapshot.stamina == 82
    assert application.dirty

    keys.active = {pygame.K_SPACE}
    application.player = PlayerState(x=1, y=10, z=1, grounded=True)
    application._jump_was_pressed = False  # type: ignore[attr-defined]
    application.update(0.01)
    assert application.vitals.snapshot.stamina == 70
    application.vitals.restore(PlayerVitalsSnapshot(stamina_milli=0))
    application._jump_was_pressed = False  # type: ignore[attr-defined]
    application.update(0.01)
    assert "Not enough stamina" in application.save_message

    keys.active = set()
    application.player = PlayerState(x=1, y=10, z=1, grounded=False)
    monkeypatch.setattr(
        voxel_application,
        "move_player",
        lambda **_kwargs: PlayerState(x=1, y=9, z=1, grounded=False),
    )
    application.update(0.01)
    assert application.vitals.snapshot.accumulated_fall_milli == 1_000

    application.vitals.restore(
        PlayerVitalsSnapshot(
            health_milli=16_000,
            stamina_milli=50_000,
            grounded=False,
            accumulated_fall_milli=5_000,
        )
    )
    application.player = PlayerState(x=1, y=9, z=1, grounded=False)
    monkeypatch.setattr(
        voxel_application,
        "move_player",
        lambda **_kwargs: PlayerState(x=1, y=8, z=1, grounded=True),
    )
    monkeypatch.setattr(voxel_application, "safe_spawn_height", lambda **_kwargs: 20.0)
    application.update(0.01)
    assert application.vitals.snapshot.death_count == 1
    assert application.save_message == "You died \N{EM DASH} respawned"

    application.vitals.restore(
        PlayerVitalsSnapshot(
            grounded=False,
            accumulated_fall_milli=2_999,
        )
    )
    application.player = PlayerState(x=1, y=9, z=1, grounded=False)
    application.update(0.01)
    assert application.vitals.snapshot.last_fall_damage == 0


def test_application_mining_completion_cancellation_and_tool_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.mouse_captured = True
    application._mining_held = True  # type: ignore[attr-defined]
    application.target = RayHit(
        x=1,
        y=2,
        z=3,
        distance=1,
        material=BlockMaterial.STONE,
    )
    outcomes: list[InteractionOutcome] = []
    monkeypatch.setattr(application, "_apply_interaction", outcomes.append)
    broken = InteractionOutcome(
        result=InteractionResult.BROKEN,
        coordinate=application.target.coordinate,
    )
    monkeypatch.setattr(application.interactions, "break_block", lambda **_kwargs: broken)
    application._update_mining(1)  # type: ignore[attr-defined]
    assert application.mining.snapshot.status is MiningStatus.ACTIVE
    application._update_mining(10_000_000)  # type: ignore[attr-defined]
    assert len(outcomes) == 1
    assert application.inventory.selected_tool is not None
    assert application.inventory.selected_tool.current_durability == 63

    application.inventory.set_slot(
        0,
        ToolInstance(
            item=ItemType.WOODEN_PICKAXE,
            current_durability=1,
            maximum_durability=64,
        ),
    )
    application._update_mining(10_000_000)  # type: ignore[attr-defined]
    assert application.inventory.selected_tool is None
    assert application.save_message == "Wooden Pickaxe broke"

    application._update_mining(1)  # type: ignore[attr-defined]
    application._mining_held = False  # type: ignore[attr-defined]
    application._update_mining(1)  # type: ignore[attr-defined]
    assert application.mining.snapshot.status is MiningStatus.CANCELLED
    application._mining_held = True  # type: ignore[attr-defined]
    monkeypatch.setattr(
        application.interactions,
        "break_block",
        lambda **_kwargs: InteractionOutcome(result=InteractionResult.NO_TARGET),
    )
    application._update_mining(10_000_000)  # type: ignore[attr-defined]
    assert application.mining.snapshot.status is MiningStatus.IDLE

    application.inventory.select_hotbar(7)
    monkeypatch.setattr(application.interactions, "break_block", lambda **_kwargs: broken)
    application._update_mining(10_000_000)  # type: ignore[attr-defined]
    assert len(outcomes) == 3

    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.MOUSEBUTTONUP, button=1)],
    )
    application.process_events()
    assert not application._mining_held  # type: ignore[attr-defined]


def test_mouse_button_up_does_not_cancel_non_primary_buttons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.running = True
    application._mining_held = True  # type: ignore[attr-defined]
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.MOUSEBUTTONUP, button=2)],
    )
    application.process_events()
    assert application._mining_held  # type: ignore[attr-defined]


def test_drop_gpu_batch_reuses_equal_sized_buffer() -> None:
    class Resource:
        def __init__(self) -> None:
            self.writes = 0
            self.released = False

        def write(self, _data: bytes) -> None:
            self.writes += 1

        def release(self) -> None:
            self.released = True

    class Context:
        def buffer(self, _data: bytes) -> Resource:
            return Resource()

        def vertex_array(self, _program: object, _content: object) -> Resource:
            return Resource()

    application = VoxelPrototypeApplication()
    application.context = cast(Any, Context())
    application.program = cast(Any, object())
    application.dropped_items.spawn(
        item=ItemType.STONE_BLOCK,
        quantity=1,
        position=(0, 2, 0),
    )
    application._refresh_drop_gpu()  # type: ignore[attr-defined]
    buffer = cast(Any, application._drop_buffer)  # type: ignore[attr-defined]
    vertex_array = cast(Any, application._drop_array)  # type: ignore[attr-defined]
    application._refresh_drop_gpu()  # type: ignore[attr-defined]
    assert buffer.writes == 0
    application.dropped_items.update(0.01, solid_at=lambda _x, _y, _z: False)
    application._refresh_drop_gpu()  # type: ignore[attr-defined]
    assert application._drop_buffer is buffer  # type: ignore[attr-defined]
    assert buffer.writes == 1
    application.dropped_items.update(0.5, solid_at=lambda _x, _y, _z: True)
    application._refresh_drop_gpu()  # type: ignore[attr-defined]
    settled_writes = buffer.writes
    application.dropped_items.update(0.01, solid_at=lambda _x, _y, _z: True)
    application._refresh_drop_gpu()  # type: ignore[attr-defined]
    assert buffer.writes == settled_writes
    application.dropped_items.spawn(
        item=ItemType.DIRT_BLOCK,
        quantity=1,
        position=(1, 2, 0),
    )
    application._refresh_drop_gpu()  # type: ignore[attr-defined]
    assert buffer.released
    assert vertex_array.released


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
    save_load_calls: list[str] = []
    monkeypatch.setattr(
        application,
        "_save_edits",
        lambda: save_load_calls.append("save") is None,
    )
    monkeypatch.setattr(
        application,
        "_load_edits",
        lambda: save_load_calls.append("load") is None,
    )
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_5),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F7),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F8),
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
    assert application.hotbar.selected_material is BlockMaterial.SAND
    assert calls == ["place"]
    assert save_load_calls == ["save", "load"]


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
            dropped_item=ItemType.STONE_BLOCK,
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
    application._apply_interaction(  # type: ignore[attr-defined]
        InteractionOutcome(result=InteractionResult.COOLDOWN)
    )
    assert application.save_message == ""
    application._apply_interaction(  # type: ignore[attr-defined]
        InteractionOutcome(
            result=InteractionResult.PLACED,
            coordinate=WorldBlockCoordinate(x=1, y=2, z=3),
        )
    )
    assert application.last_placement_consumption == "consumed 1 selected item"


def test_voxel_save_load_is_atomic_dirty_and_world_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_path = tmp_path / "voxel.json"
    application = VoxelPrototypeApplication(
        config=VoxelPrototypeConfig(save_path=save_path, render_distance=0)
    )
    coordinate = WorldBlockCoordinate(x=-16, y=8, z=15)
    application.edits.set_block(coordinate, BlockMaterial.STONE)
    application.dirty = True
    assert application._save_edits()  # type: ignore[attr-defined]
    assert not application.dirty
    assert application.save_message == "World saved"

    application.edits.set_block(coordinate, BlockMaterial.DIRT)
    application.dirty = True
    monkeypatch.setattr(application, "_stream", lambda: None)
    monkeypatch.setattr(
        "open_world_rpg.ui.voxel.application.ray_cast",
        lambda **_kwargs: None,
    )
    assert application._load_edits()  # type: ignore[attr-defined]
    assert application.edits.get(coordinate).material is BlockMaterial.STONE  # type: ignore[union-attr]
    assert not application.dirty
    assert application.save_message == "World loaded"

    current = application.edits.snapshot()
    save_path.write_text("{", encoding="utf-8")
    application.dirty = True
    assert not application._load_edits()  # type: ignore[attr-defined]
    assert application.edits.snapshot() == current
    assert application.dirty
    assert application.save_message == "Load failed"

    assert application._save_edits()  # type: ignore[attr-defined]
    raw = json.loads(save_path.read_text(encoding="utf-8"))
    raw["session"]["session_id"] = str(UUID(int=2))
    save_path.write_text(json.dumps(raw), encoding="utf-8")
    assert not application._load_edits()  # type: ignore[attr-defined]
    assert application.edits.snapshot() == current


def test_voxel_save_controls_fail_cleanly_and_autosave_only_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled = VoxelPrototypeApplication()
    assert not disabled._save_edits()  # type: ignore[attr-defined]
    assert not disabled._load_edits()  # type: ignore[attr-defined]

    application = VoxelPrototypeApplication(
        config=VoxelPrototypeConfig(
            save_path=tmp_path / "auto.json",
            autosave=True,
        )
    )
    assert application._save_service is not None  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "open_world_rpg.ui.voxel.application.GameSaveService.save",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk")),
    )
    application.dirty = True
    assert not application._save_edits()  # type: ignore[attr-defined]
    assert application.dirty
    with pytest.raises(ValueError, match=r"\.json"):
        VoxelPrototypeApplication(config=VoxelPrototypeConfig(save_path=tmp_path / "invalid.txt"))


def test_clean_run_autosaves_dirty_edits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    application = VoxelPrototypeApplication(
        config=VoxelPrototypeConfig(save_path=tmp_path / "auto.json", autosave=True)
    )

    class Clock:
        def tick(self, _fps: int) -> int:
            return 16

        def get_fps(self) -> float:
            return 60.0

    application.running = True
    application._clock = cast(Any, Clock())  # type: ignore[attr-defined]
    application.dirty = True
    saved = 0

    def save() -> bool:
        nonlocal saved
        saved += 1
        return True

    monkeypatch.setattr(application, "_save_edits", save)
    monkeypatch.setattr(application, "process_events", lambda: None)
    monkeypatch.setattr(application, "update", lambda _delta: None)
    monkeypatch.setattr(application, "render", lambda: None)
    monkeypatch.setattr(application, "shutdown", lambda: None)
    assert application.run(max_frames=1) == 0
    assert saved == 1


def test_caption_and_hud_cover_help_debug_loading_and_uninitialised_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.show_help = True
    application.show_debug = False
    assert "CURSOR" in application._caption(0)  # type: ignore[attr-defined]
    assert "F1 controls" in application._caption(0)  # type: ignore[attr-defined]
    application.mouse_captured = True
    assert "PLAY" in application._caption(0)  # type: ignore[attr-defined]
    application.mouse_captured = False
    application.show_debug = True
    application.target = RayHit(x=1, y=2, z=3, distance=4.0)
    assert "target 1,2,3" in application._caption(12)  # type: ignore[attr-defined]

    blank_surface = pygame.Surface((1024, 512), pygame.SRCALPHA)
    application._render_hud(0)  # type: ignore[attr-defined]
    application._draw_hotbar(blank_surface)  # type: ignore[attr-defined]
    application._draw_help_panel(blank_surface)  # type: ignore[attr-defined]
    application._draw_interaction_prompt(blank_surface, "")  # type: ignore[attr-defined]
    application._draw_capture_prompt(blank_surface)  # type: ignore[attr-defined]

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
    application.mining.begin(
        target=WorldBlockCoordinate(x=1, y=2, z=3),
        material=BlockMaterial.STONE,
        tool=application.inventory.selected_tool,
    )
    application.mining.advance(100_000)
    application.inventory.select_hotbar(2)
    application._refresh_interaction_previews()  # type: ignore[attr-defined]
    application._render_hud(12)  # type: ignore[attr-defined]
    assert application.hud_snapshot is not None
    assert application.hud_snapshot.loading
    assert application.hud_snapshot.placement_target is not None
    application.mouse_captured = True
    application.target = None
    application._refresh_interaction_previews()  # type: ignore[attr-defined]
    application._render_hud(12)  # type: ignore[attr-defined]


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
    tmp_path: Path,
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
    assert not calls[-1][0].game_flow_enabled
    assert calls[-1][0].save_path is None
    assert calls[-1][1] == 3
    assert voxel_demo.main([]) == 0
    assert not calls[-1][0].hidden_window
    assert calls[-1][0].game_flow_enabled
    assert calls[-1][0].save_path == Path("saves/voxel.json")
    assert calls[-1][1] is None
    assert voxel_demo.main(["--direct-play"]) == 0
    assert not calls[-1][0].game_flow_enabled
    save_path = tmp_path / "voxel.json"
    assert (
        voxel_demo.main(
            [
                "--smoke-test",
                "--save-path",
                str(save_path),
                "--load",
                "--autosave",
            ]
        )
        == 0
    )
    assert calls[-1][0].save_path == save_path
    assert calls[-1][0].load_on_start
    assert calls[-1][0].autosave
    assert voxel_demo.main(["--load"]) == 0
    assert calls[-1][0].load_on_start
    assert calls[-1][0].save_path == Path("saves/voxel.json")
    with pytest.raises(SystemExit):
        voxel_demo.main(["--smoke-test", "--load"])
    with pytest.raises(SystemExit):
        voxel_demo.main(["--smoke-test", "--autosave"])


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


def test_window_focus_loss_releases_mouse_and_cancels_mining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.running = True
    application.mouse_captured = True
    application._mining_held = True  # type: ignore[attr-defined]
    monkeypatch.setattr(pygame.event, "set_grab", lambda _captured: None)
    monkeypatch.setattr(pygame.mouse, "set_visible", lambda _visible: None)
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.WINDOWFOCUSLOST)],
    )

    application.process_events()

    assert not application.mouse_captured
    assert not application._mining_held  # type: ignore[attr-defined]
    assert application.running

    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.WINDOWFOCUSLOST)],
    )
    application.process_events()
    assert not application.mouse_captured


def test_update_normalises_diagonal_motion_and_pauses_input_with_free_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.mouse_captured = True
    application.player = PlayerState(x=8.0, y=20.0, z=8.0, grounded=True)
    captured_moves: list[dict[str, object]] = []

    class Keys:
        def __getitem__(self, key: int) -> bool:
            return key in {
                pygame.K_w,
                pygame.K_d,
                pygame.K_LSHIFT,
                pygame.K_SPACE,
            }

    def capture_move(**kwargs: object) -> PlayerState:
        captured_moves.append(kwargs)
        return application.player

    monkeypatch.setattr(pygame.key, "get_pressed", Keys)
    monkeypatch.setattr(voxel_application, "move_player", capture_move)
    monkeypatch.setattr(voxel_application, "ray_cast", lambda **_kwargs: None)
    monkeypatch.setattr(application, "_stream", lambda: None)
    monkeypatch.setattr(application.dropped_items, "update", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(application.dropped_items, "pickup_near", lambda **_kwargs: ())

    application.update(0.01)

    first = captured_moves[-1]
    diagonal_distance = (float(first["delta_x"]) ** 2 + float(first["delta_z"]) ** 2) ** 0.5
    assert diagonal_distance == pytest.approx(0.1)
    assert first["jump"] is True

    application.mouse_captured = False
    application._jump_was_pressed = False  # type: ignore[attr-defined]
    application.update(0.01)

    paused = captured_moves[-1]
    assert paused["delta_x"] == 0.0
    assert paused["delta_z"] == 0.0
    assert paused["jump"] is False


def test_contextual_interaction_prompt_covers_capture_target_tool_block_and_empty_slot() -> None:
    application = VoxelPrototypeApplication()
    assert "Click to capture" in application._interaction_prompt()  # type: ignore[attr-defined]

    application.mouse_captured = True
    assert "Aim at a block" in application._interaction_prompt()  # type: ignore[attr-defined]

    application.target = RayHit(
        x=0,
        y=40,
        z=0,
        distance=1.0,
        material=BlockMaterial.GRASS,
        face_normal=(0, 1, 0),
    )
    application._refresh_interaction_previews()  # type: ignore[attr-defined]
    assert "Wooden Pickaxe" in application._interaction_prompt()  # type: ignore[attr-defined]

    application.inventory.select_hotbar(2)
    application._refresh_interaction_previews()  # type: ignore[attr-defined]
    prompt = application._interaction_prompt()  # type: ignore[attr-defined]
    assert "RMB to place Grass Block" in prompt

    application.target = RayHit(
        x=0,
        y=4,
        z=0,
        distance=1.0,
        material=BlockMaterial.GRASS,
        face_normal=(0, -1, 0),
    )
    application._refresh_interaction_previews()  # type: ignore[attr-defined]
    assert "placement destination occupied" in application._interaction_prompt()  # type: ignore[attr-defined]

    application.inventory.select_hotbar(7)
    application._refresh_interaction_previews()  # type: ignore[attr-defined]
    assert "Select a block to place" in application._interaction_prompt()  # type: ignore[attr-defined]


def test_invalid_break_preview_cancels_active_mining() -> None:
    application = VoxelPrototypeApplication()
    application.mouse_captured = True
    application._mining_held = True  # type: ignore[attr-defined]
    application.target = RayHit(
        x=0,
        y=5,
        z=0,
        distance=1.0,
        material=BlockMaterial.WATER,
    )
    application._refresh_interaction_previews()  # type: ignore[attr-defined]
    application._update_mining(1)  # type: ignore[attr-defined]
    assert application.mining.snapshot.status is MiningStatus.IDLE

    application.mining.begin(
        target=application.target.coordinate,
        material=BlockMaterial.STONE,
        tool=application.inventory.selected_tool,
    )
    application._update_mining(1)  # type: ignore[attr-defined]

    assert application.mining.snapshot.status is MiningStatus.CANCELLED
    assert application.mining.snapshot.last_cancellation_reason == InteractionResult.WATER.value
    assert "Mining: water cannot be broken" in application._interaction_prompt()  # type: ignore[attr-defined]


def test_invalid_left_click_reports_preview_without_starting_mining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.mouse_captured = True
    application.target = RayHit(
        x=0,
        y=5,
        z=0,
        distance=1.0,
        material=BlockMaterial.WATER,
    )
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1)],
    )

    application.process_events()

    assert not application._mining_held  # type: ignore[attr-defined]
    assert application.last_interaction is InteractionResult.WATER
    assert application.save_message == "Water cannot be broken"


def test_flying_vertical_input_respects_mouse_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication()
    application.mouse_captured = True
    application.player = PlayerState(x=8.0, y=20.0, z=8.0, flying=True)

    class Keys:
        def __getitem__(self, key: int) -> bool:
            return key == pygame.K_LCTRL

    monkeypatch.setattr(pygame.key, "get_pressed", Keys)
    monkeypatch.setattr(voxel_application, "move_player", lambda **_kwargs: application.player)
    monkeypatch.setattr(voxel_application, "ray_cast", lambda **_kwargs: None)
    monkeypatch.setattr(application, "_stream", lambda: None)
    monkeypatch.setattr(application.dropped_items, "update", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(application.dropped_items, "pickup_near", lambda **_kwargs: ())

    application.update(0.1)
    captured_y = application.player.y
    assert captured_y == pytest.approx(19.5)

    application.mouse_captured = False
    application.update(0.1)
    assert application.player.y == captured_y


def test_render_colours_valid_invalid_and_feedback_previews(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Uniform:
        value: object = None

        def write(self, _value: bytes) -> None:
            return None

    class Program:
        def __init__(self) -> None:
            self.uniforms: dict[str, Uniform] = {}

        def __getitem__(self, name: str) -> Uniform:
            return self.uniforms.setdefault(name, Uniform())

    class Context:
        viewport: tuple[int, int, int, int]
        depth_mask = True

        def clear(self, *_args: object, **_kwargs: object) -> None:
            return None

        def disable(self, _flag: int) -> None:
            return None

        def enable(self, _flag: int) -> None:
            return None

    application = VoxelPrototypeApplication()
    application.context = cast(Any, Context())
    application.program = cast(Any, Program())
    application._visible = ()  # type: ignore[attr-defined]
    application.inventory.select_hotbar(2)
    application.target = RayHit(
        x=0,
        y=40,
        z=0,
        distance=1.0,
        material=BlockMaterial.GRASS,
        face_normal=(0, 1, 0),
    )
    application.break_preview = InteractionOutcome(
        result=InteractionResult.BROKEN,
        coordinate=application.target.coordinate,
    )
    application.placement_preview = InteractionOutcome(
        result=InteractionResult.PLACED,
        coordinate=WorldBlockCoordinate(x=0, y=41, z=0),
    )
    application._feedback_coordinate = WorldBlockCoordinate(x=1, y=2, z=3)  # type: ignore[attr-defined]
    application._feedback_until = 2.0  # type: ignore[attr-defined]
    colours: list[tuple[float, float, float, float]] = []

    monkeypatch.setattr(pygame.display, "get_window_size", lambda: (1280, 720))
    monkeypatch.setattr(pygame.display, "set_caption", lambda _caption: None)
    monkeypatch.setattr(pygame.display, "flip", lambda: None)
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 1000)
    monkeypatch.setattr(application, "_refresh_drop_gpu", lambda: None)
    monkeypatch.setattr(application, "_render_hud", lambda _triangles: None)
    monkeypatch.setattr(
        application,
        "_render_target_outline",
        lambda _target, *, colour=(1.0, 0.93, 0.32, 1.0): colours.append(colour),
    )

    application.render()
    assert colours == [
        (1.0, 0.93, 0.32, 1.0),
        (0.30, 0.95, 0.48, 1.0),
        (0.35, 0.90, 1.0, 1.0),
    ]

    colours.clear()
    application.break_preview = InteractionOutcome(result=InteractionResult.WATER)
    application.placement_preview = InteractionOutcome(
        result=InteractionResult.OCCUPIED,
        coordinate=WorldBlockCoordinate(x=0, y=39, z=0),
    )
    application._feedback_until = 0.5  # type: ignore[attr-defined]
    application.render()
    assert colours == [
        (1.0, 0.35, 0.28, 1.0),
        (1.0, 0.30, 0.24, 1.0),
    ]

    colours.clear()
    application.inventory.select_hotbar(0)
    application.target = None
    application.placement_preview = InteractionOutcome(result=InteractionResult.NO_TARGET)
    application.render()
    assert colours == []


def test_game_flow_initial_state_layout_helpers_and_overlay_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    save_path = tmp_path / "world.json"
    save_path.write_text("{}", encoding="utf-8")
    application = VoxelPrototypeApplication(
        config=VoxelPrototypeConfig(game_flow_enabled=True, save_path=save_path)
    )
    assert application.flow.screen is VoxelScreen.MAIN_MENU
    assert application.flow.continue_available
    assert not application.mouse_captured

    direct = VoxelPrototypeApplication()
    assert direct.flow.screen is VoxelScreen.PLAYING
    assert not direct.flow.continue_available

    assert application._inventory_slot_at(76, 138) == 0  # type: ignore[attr-defined]
    assert application._inventory_slot_at(516, 248) == 26  # type: ignore[attr-defined]
    assert application._inventory_slot_at(124, 138) is None  # type: ignore[attr-defined]
    assert application._inventory_slot_at(75, 138) is None  # type: ignore[attr-defined]
    assert application._recipe_at(626, 116) == 0  # type: ignore[attr-defined]
    assert application._recipe_at(969, 153) == 0  # type: ignore[attr-defined]
    assert application._recipe_at(626, 154) is None  # type: ignore[attr-defined]
    assert application._recipe_at(625, 116) is None  # type: ignore[attr-defined]
    assert application._recipe_at(626, 1000) is None  # type: ignore[attr-defined]
    assert application._menu_option_at(340, 220) == 0  # type: ignore[attr-defined]
    assert application._menu_option_at(683, 261) == 0  # type: ignore[attr-defined]
    assert application._menu_option_at(340, 262) is None  # type: ignore[attr-defined]
    assert application._menu_option_at(100, 220) is None  # type: ignore[attr-defined]
    application.flow.screen = VoxelScreen.INVENTORY
    assert application._menu_option_at(340, 220) is None  # type: ignore[attr-defined]

    monkeypatch.setattr(pygame.display, "get_window_size", lambda: (2048, 1024))
    assert application._hud_pointer((1024, 512)) == (512, 256)  # type: ignore[attr-defined]
    monkeypatch.setattr(pygame.display, "get_window_size", lambda: (0, 0))
    assert application._hud_pointer((1, 1)) == (1024, 512)  # type: ignore[attr-defined]

    application.target = RayHit(x=0, y=0, z=0, distance=1.0)
    application._mining_held = True  # type: ignore[attr-defined]
    refreshed: list[bool] = []
    monkeypatch.setattr(
        application,
        "_refresh_interaction_previews",
        lambda: refreshed.append(True),
    )
    application.update(0.01)
    assert application.target is None
    assert not application._mining_held  # type: ignore[attr-defined]
    assert refreshed == [True]


def test_game_flow_actions_new_continue_pause_save_death_and_quit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication(config=VoxelPrototypeConfig(game_flow_enabled=True))
    captures: list[bool] = []
    resets: list[str] = []
    respawns: list[str] = []
    save_results: list[bool] = []
    load_results: list[bool] = []

    def capture_mouse(captured: bool) -> None:
        captures.append(captured)
        application.mouse_captured = captured

    monkeypatch.setattr(application, "_capture_mouse", capture_mouse)
    monkeypatch.setattr(application, "_reset_new_world", lambda: resets.append("new"))
    monkeypatch.setattr(application, "_respawn_after_death", lambda: respawns.append("respawn"))
    monkeypatch.setattr(application, "_save_edits", lambda: save_results.pop(0))
    monkeypatch.setattr(application, "_load_edits", lambda: load_results.pop(0))

    application._activate_flow_action(GameFlowAction.NEW_WORLD)  # type: ignore[attr-defined]
    assert resets == ["new"]
    assert application.flow.screen is VoxelScreen.PLAYING
    assert captures[-1]

    application.flow.return_to_main_menu()
    application.flow.set_continue_available(True)
    load_results.extend((False, True))
    application._activate_flow_action(GameFlowAction.CONTINUE)  # type: ignore[attr-defined]
    assert application.flow.screen is VoxelScreen.MAIN_MENU
    application._activate_flow_action(GameFlowAction.CONTINUE)  # type: ignore[attr-defined]
    assert application.flow.screen is VoxelScreen.PLAYING

    application.flow.pause()
    application._activate_flow_action(GameFlowAction.RESUME)  # type: ignore[attr-defined]
    assert application.flow.screen is VoxelScreen.PLAYING

    save_results.extend((True, False, True))
    application._activate_flow_action(GameFlowAction.SAVE)  # type: ignore[attr-defined]
    application.running = True
    application._activate_flow_action(GameFlowAction.SAVE_AND_QUIT)  # type: ignore[attr-defined]
    assert application.running
    application._activate_flow_action(GameFlowAction.SAVE_AND_QUIT)  # type: ignore[attr-defined]
    assert not application.running

    application.flow.mark_dead()
    application._activate_flow_action(GameFlowAction.RESPAWN)  # type: ignore[attr-defined]
    assert respawns == ["respawn"]
    assert application.flow.screen is VoxelScreen.PLAYING

    application.flow.mark_dead()
    application._activate_flow_action(GameFlowAction.QUIT)  # type: ignore[attr-defined]
    assert application.flow.screen is VoxelScreen.MAIN_MENU
    application.flow.start_new_world()
    application.running = True
    application._activate_flow_action(GameFlowAction.QUIT)  # type: ignore[attr-defined]
    assert not application.running
    application._activate_flow_action(GameFlowAction.NONE)  # type: ignore[attr-defined]


def test_game_flow_overlay_keyboard_mouse_and_inventory_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication(config=VoxelPrototypeConfig(game_flow_enabled=True))
    captured: list[bool] = []

    def capture_mouse(value: bool) -> None:
        captured.append(value)
        application.mouse_captured = value

    monkeypatch.setattr(application, "_capture_mouse", capture_mouse)
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 2000)

    application.flow.start_new_world()
    application._open_inventory_screen()  # type: ignore[attr-defined]
    assert application.flow.screen is VoxelScreen.INVENTORY
    assert captured[-1] is False
    for key in (pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT, pygame.K_UP):
        application._process_overlay_key(key)  # type: ignore[attr-defined]
    for key in (pygame.K_PAGEDOWN, pygame.K_RIGHTBRACKET, pygame.K_PAGEUP, pygame.K_LEFTBRACKET):
        application._process_overlay_key(key)  # type: ignore[attr-defined]

    application.inventory_screen.select_slot(2)
    application._process_overlay_key(pygame.K_RETURN)  # type: ignore[attr-defined]
    application.inventory_screen.select_slot(9)
    application._process_overlay_key(pygame.K_KP_ENTER)  # type: ignore[attr-defined]
    assert application.inventory.slot(9) is not None
    application._process_overlay_key(pygame.K_q)  # type: ignore[attr-defined]

    application.inventory.add(ItemType.WOOD_LOG, 1)
    application.inventory_screen.selected_recipe_index = 0
    application._process_overlay_key(pygame.K_c)  # type: ignore[attr-defined]
    assert application.inventory.total_quantity(ItemType.WOOD_PLANK) == 4
    assert application.save_message == "Crafted Wood Plank x4"
    assert application.dirty

    application.inventory_screen.selected_recipe_index = 5
    application._process_overlay_key(pygame.K_c)  # type: ignore[attr-defined]
    assert application.save_message == CraftingResult.MISSING_INGREDIENTS.value.capitalize()

    application._process_overlay_key(pygame.K_e)  # type: ignore[attr-defined]
    assert application.flow.screen is VoxelScreen.PLAYING
    assert captured[-1] is True

    application._open_pause_menu()  # type: ignore[attr-defined]
    application._process_overlay_key(pygame.K_ESCAPE)  # type: ignore[attr-defined]
    assert application.flow.screen is VoxelScreen.PLAYING
    application.flow.return_to_main_menu()
    application.running = True
    application._process_overlay_key(pygame.K_ESCAPE)  # type: ignore[attr-defined]
    assert not application.running
    application.flow.mark_dead()
    application._process_overlay_key(pygame.K_ESCAPE)  # type: ignore[attr-defined]
    assert application.flow.screen is VoxelScreen.MAIN_MENU

    actions: list[GameFlowAction] = []
    monkeypatch.setattr(application, "_activate_flow_action", actions.append)
    application._process_overlay_key(pygame.K_DOWN)  # type: ignore[attr-defined]
    application._process_overlay_key(pygame.K_UP)  # type: ignore[attr-defined]
    application._process_overlay_key(pygame.K_SPACE)  # type: ignore[attr-defined]
    assert actions[-1] is GameFlowAction.NEW_WORLD
    application.flow.mark_dead()
    application._process_overlay_key(pygame.K_r)  # type: ignore[attr-defined]
    assert actions[-1] is GameFlowAction.RESPAWN

    monkeypatch.setattr(application, "_hud_pointer", lambda position: position)
    application.flow.screen = VoxelScreen.INVENTORY
    application._process_overlay_click(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(76, 138))
    )
    application._process_overlay_click(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(76, 138))
    )
    application._process_overlay_click(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(626, 116))
    )
    application._process_overlay_click(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(0, 0))
    )
    application.flow.return_to_main_menu()
    application._process_overlay_click(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(340, 220))
    )
    assert actions[-1] is GameFlowAction.NEW_WORLD


def test_game_flow_event_router_and_gameplay_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication(config=VoxelPrototypeConfig(game_flow_enabled=True))
    application.running = True
    application._process_flow_event(pygame.event.Event(pygame.QUIT))  # type: ignore[attr-defined]
    assert not application.running

    routed: list[int] = []
    application.flow.start_new_world()
    monkeypatch.setattr(
        application,
        "_process_flow_gameplay_event",
        lambda event: routed.append(event.type),
    )
    application._process_flow_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))  # type: ignore[attr-defined]
    assert routed == [pygame.KEYDOWN]

    application = VoxelPrototypeApplication(config=VoxelPrototypeConfig(game_flow_enabled=True))
    captures: list[bool] = []
    pauses: list[bool] = []
    inventories: list[bool] = []
    streams: list[bool] = []
    saves: list[bool] = []
    loads: list[bool] = []
    inventory_changes: list[str] = []
    interactions: list[InteractionOutcome] = []

    def capture_mouse(value: bool) -> None:
        captures.append(value)
        application.mouse_captured = value

    monkeypatch.setattr(application, "_capture_mouse", capture_mouse)
    monkeypatch.setattr(application, "_open_pause_menu", lambda: pauses.append(True))
    monkeypatch.setattr(application, "_open_inventory_screen", lambda: inventories.append(True))
    monkeypatch.setattr(application, "_place_player_at_spawn", lambda: None)
    monkeypatch.setattr(application, "_stream", lambda: streams.append(True))
    monkeypatch.setattr(application, "_save_edits", lambda: saves.append(True) or True)
    monkeypatch.setattr(application, "_load_edits", lambda: loads.append(True) or True)
    monkeypatch.setattr(application, "_on_inventory_changed", inventory_changes.append)
    monkeypatch.setattr(application, "_refresh_interaction_previews", lambda: None)
    monkeypatch.setattr(application, "_apply_interaction", interactions.append)
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 1000)
    application.flow.start_new_world()

    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.WINDOWFOCUSLOST)
    )
    for key in (
        pygame.K_ESCAPE,
        pygame.K_e,
        pygame.K_F1,
        pygame.K_h,
        pygame.K_F3,
        pygame.K_f,
        pygame.K_r,
        pygame.K_F5,
        pygame.K_F6,
        pygame.K_F7,
        pygame.K_F8,
        pygame.K_1,
        pygame.K_2,
    ):
        application._process_flow_gameplay_event(  # type: ignore[attr-defined]
            pygame.event.Event(pygame.KEYDOWN, key=key)
        )
    assert len(pauses) == 2
    assert inventories == [True]
    assert streams == [True, True]
    assert saves == [True]
    assert loads == [True]
    assert inventory_changes == ["selected tool changed"]

    application.mouse_captured = True
    old_camera = application.camera
    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEMOTION, rel=(2, -1))
    )
    assert application.camera != old_camera
    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEWHEEL, y=1)
    )
    assert inventory_changes[-1] == "selected tool changed"

    application.mouse_captured = False
    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1)
    )
    assert captures[-1] is True
    application.mouse_captured = True
    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=4)
    )
    application.break_preview = InteractionOutcome(result=InteractionResult.WATER)
    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1)
    )
    assert interactions[-1].result is InteractionResult.WATER
    application.break_preview = InteractionOutcome(
        result=InteractionResult.BROKEN,
        coordinate=WorldBlockCoordinate(x=0, y=0, z=0),
    )
    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1)
    )
    assert application._mining_held  # type: ignore[attr-defined]
    placement = InteractionOutcome(result=InteractionResult.PLACED)
    monkeypatch.setattr(
        application.interactions,
        "place_inventory_block",
        lambda **_kwargs: placement,
    )
    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3)
    )
    assert interactions[-1] is placement
    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONUP, button=1)
    )
    assert not application._mining_held  # type: ignore[attr-defined]


def test_new_world_spawn_and_flow_overlay_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pygame.init()
    pygame.font.init()
    try:
        application = VoxelPrototypeApplication(config=VoxelPrototypeConfig(game_flow_enabled=True))
        application._font = pygame.font.Font(None, 22)  # type: ignore[attr-defined]
        surface = pygame.Surface((1024, 512), pygame.SRCALPHA)
        application.save_message = "Status"
        application._draw_flow_overlay(surface)  # type: ignore[attr-defined]
        application.flow.screen = VoxelScreen.PAUSED
        application._draw_flow_overlay(surface)  # type: ignore[attr-defined]
        application.flow.screen = VoxelScreen.DEAD
        application._draw_flow_overlay(surface)  # type: ignore[attr-defined]

        application.inventory.set_slot(0, ItemStack(item=ItemType.WOOD_LOG, quantity=2))
        application.inventory.set_slot(1, ItemStack(item=ItemType.WOOD_PLANK, quantity=2))
        application.inventory.set_slot(2, ItemStack(item=ItemType.STICK, quantity=2))
        application.inventory.set_slot(3, ItemStack(item=ItemType.GRASS_BLOCK, quantity=2))
        application.inventory.set_slot(
            4,
            ToolInstance(
                item=ItemType.WOODEN_PICKAXE,
                current_durability=1,
                maximum_durability=64,
            ),
        )
        application.inventory.set_slot(5, ToolInstance.create(ItemType.STONE_PICKAXE))
        application.flow.screen = VoxelScreen.INVENTORY
        application.inventory_screen.source_slot_index = 0
        application._draw_flow_overlay(surface)  # type: ignore[attr-defined]
        application.save_message = ""
        application._draw_flow_overlay(surface)  # type: ignore[attr-defined]
        application._draw_hotbar(surface)  # type: ignore[attr-defined]
        application._draw_inventory_value(  # type: ignore[attr-defined]
            surface,
            surface,
            pygame.Rect(0, 0, 48, 48),
            None,
        )
        saved_font = application._font  # type: ignore[attr-defined]
        application._font = None  # type: ignore[attr-defined]
        application._draw_flow_overlay(surface)  # type: ignore[attr-defined]
        application._draw_menu_screen(surface)  # type: ignore[attr-defined]
        application._draw_inventory_screen(surface)  # type: ignore[attr-defined]
        application._draw_inventory_value(  # type: ignore[attr-defined]
            surface,
            surface,
            pygame.Rect(0, 0, 48, 48),
            application.inventory.slot(0),
        )
        application._font = saved_font  # type: ignore[attr-defined]

        monkeypatch.setattr(voxel_application, "safe_spawn_height", lambda **_kwargs: 12.5)
        application.spawn_x = 4
        application.spawn_z = 6
        application._place_player_at_spawn()  # type: ignore[attr-defined]
        assert application.player == PlayerState(x=4.5, y=12.5, z=6.5, grounded=True)

        calls: list[str] = []
        monkeypatch.setattr(application, "_place_player_at_spawn", lambda: calls.append("spawn"))
        monkeypatch.setattr(
            application,
            "_refresh_interaction_previews",
            lambda: calls.append("preview"),
        )
        monkeypatch.setattr(application, "_stream", lambda: calls.append("stream"))
        application.edits.set_block(
            WorldBlockCoordinate(x=0, y=1, z=0),
            BlockMaterial.STONE,
        )
        application._reset_new_world()  # type: ignore[attr-defined]
        assert calls == ["spawn", "preview", "stream"]
        assert application.edits.revision == 0
        assert application.save_message == "New world started"
        assert not application.dirty
    finally:
        pygame.quit()


def test_game_flow_overlay_event_router_and_process_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication(config=VoxelPrototypeConfig(game_flow_enabled=True))
    captures: list[bool] = []
    keys: list[int] = []
    clicks: list[int] = []

    def capture_mouse(value: bool) -> None:
        captures.append(value)
        application.mouse_captured = value

    monkeypatch.setattr(application, "_capture_mouse", capture_mouse)
    monkeypatch.setattr(application, "_process_overlay_key", keys.append)
    monkeypatch.setattr(
        application,
        "_process_overlay_click",
        lambda event: clicks.append(event.button),
    )
    application._process_flow_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.WINDOWFOCUSLOST)
    )
    application._process_flow_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN)
    )
    application._process_flow_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(0, 0))
    )
    application.flow.screen = VoxelScreen.INVENTORY
    selected = application.inventory_screen.selected_recipe_index
    application._process_flow_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEWHEEL, y=-1)
    )
    assert captures == [False]
    assert keys == [pygame.K_DOWN]
    assert clicks == [1]
    assert application.inventory_screen.selected_recipe_index == selected + 1

    application.running = True
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.QUIT)],
    )
    application.process_events()
    assert not application.running


def test_game_flow_noop_event_and_action_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication(config=VoxelPrototypeConfig(game_flow_enabled=True))
    application._process_flow_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEMOTION, rel=(0, 0))
    )

    application.flow.start_new_world()
    application._process_overlay_key(pygame.K_ESCAPE)  # type: ignore[attr-defined]
    application.flow.return_to_main_menu()
    application._process_overlay_key(pygame.K_F3)  # type: ignore[attr-defined]
    application.flow.mark_dead()
    application._process_overlay_key(pygame.K_F3)  # type: ignore[attr-defined]

    monkeypatch.setattr(application, "_hud_pointer", lambda position: position)
    application.flow.screen = VoxelScreen.INVENTORY
    application._process_overlay_key(pygame.K_F3)  # type: ignore[attr-defined]
    application._process_overlay_click(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=2, pos=(76, 138))
    )
    application._process_overlay_click(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=2, pos=(626, 116))
    )
    application.flow.return_to_main_menu()
    application._process_overlay_click(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(0, 0))
    )

    application.flow.return_to_main_menu()
    application.flow.set_continue_available(False)
    monkeypatch.setattr(application, "_load_edits", lambda: True)
    captures: list[bool] = []
    monkeypatch.setattr(application, "_capture_mouse", captures.append)
    application._activate_flow_action(GameFlowAction.CONTINUE)  # type: ignore[attr-defined]
    assert application.flow.screen is VoxelScreen.MAIN_MENU
    assert captures == []

    application._activate_flow_action(GameFlowAction.RESUME)  # type: ignore[attr-defined]
    application._activate_flow_action(GameFlowAction.RESPAWN)  # type: ignore[attr-defined]
    application._open_pause_menu()  # type: ignore[attr-defined]
    application._open_inventory_screen()  # type: ignore[attr-defined]

    application.flow.start_new_world()
    monkeypatch.setattr(application.inventory, "cycle_hotbar", lambda _direction: False)
    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEWHEEL, y=0)
    )
    application.mouse_captured = True
    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=4)
    )
    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=2)
    )
    application._process_flow_gameplay_event(  # type: ignore[attr-defined]
        pygame.event.Event(pygame.MOUSEBUTTONUP, button=2)
    )


def test_game_flow_death_transition_waits_for_explicit_respawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = VoxelPrototypeApplication(config=VoxelPrototypeConfig(game_flow_enabled=True))
    application.flow.start_new_world()
    application.mouse_captured = True
    application.vitals.restore(
        PlayerVitalsSnapshot(
            health_milli=16_000,
            stamina_milli=50_000,
            grounded=False,
            accumulated_fall_milli=5_000,
        )
    )
    application.player = PlayerState(x=1, y=9, z=1, grounded=False)

    class Keys:
        def __getitem__(self, _key: int) -> bool:
            return False

    captures: list[bool] = []
    monkeypatch.setattr(pygame.key, "get_pressed", Keys)
    monkeypatch.setattr(
        voxel_application,
        "move_player",
        lambda **_kwargs: PlayerState(x=1, y=8, z=1, grounded=True),
    )
    monkeypatch.setattr(voxel_application, "ray_cast", lambda **_kwargs: None)
    monkeypatch.setattr(application, "_stream", lambda: None)
    monkeypatch.setattr(application.dropped_items, "update", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(application.dropped_items, "pickup_near", lambda **_kwargs: ())
    monkeypatch.setattr(application, "_capture_mouse", captures.append)

    application.update(0.01)

    assert application.flow.screen is VoxelScreen.DEAD
    assert application.vitals.snapshot.death_count == 0
    assert not application._mining_held  # type: ignore[attr-defined]
    assert application.target is None
    assert captures == [False]


def test_game_flow_overlay_is_composed_by_the_authoritative_hud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    pygame.init()
    pygame.font.init()
    try:
        application = VoxelPrototypeApplication(config=VoxelPrototypeConfig(game_flow_enabled=True))
        application.context = cast(Any, Context())
        application._hud_texture = cast(Any, Resource())  # type: ignore[attr-defined]
        application._hud_array = cast(Any, Resource())  # type: ignore[attr-defined]
        application._font = pygame.font.Font(None, 22)  # type: ignore[attr-defined]
        application.inventory.set_slot(0, ItemStack(item=ItemType.WOOD_LOG, quantity=2))
        application.save_message = ""
        monkeypatch.setattr(pygame.time, "get_ticks", lambda: 10_000)

        application._render_hud(0)  # type: ignore[attr-defined]

        assert application.hud_snapshot is not None
        assert application.hud_snapshot.selected_item == ItemType.WOOD_LOG.value
        assert application.hud_snapshot.selected_material is None
        assert application.flow.screen is VoxelScreen.MAIN_MENU
    finally:
        pygame.quit()
