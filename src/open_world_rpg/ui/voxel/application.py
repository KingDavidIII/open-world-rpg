"""Pygame/ModernGL first-person voxel prototype controller."""

from __future__ import annotations

import logging
import math
import struct
import time
from array import array
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast
from uuid import UUID

import moderngl
import pygame

from open_world_rpg.application import GameMode, RuntimeContext, create_terrain_runtime
from open_world_rpg.application.save_service import GameSaveService
from open_world_rpg.core import MAX_WORLD_SEED, ProjectPaths
from open_world_rpg.gameplay import (
    GUIDE_PAGES,
    CraftingResult,
    DroppedItemManager,
    ItemStack,
    ItemType,
    MiningStatus,
    PickupResult,
    PlayerVitals,
    SurvivalProgression,
    TimedMiningController,
    ToolClassification,
    ToolInstance,
    create_bootstrap_inventory,
    item_policy,
)
from open_world_rpg.persistence import RuntimeStorage, SaveRepository, SaveSlot
from open_world_rpg.world import (
    CHUNK_SIZE,
    BlockEditStore,
    BlockMaterial,
    ChunkCoordinate,
    ChunkState,
    LocalTileCoordinate,
    TerrainGenerationConfig,
    WorldBlockCoordinate,
    WorldId,
    WorldModel,
    WorldSeed,
    WorldSpecification,
)

from .blocks import BlockColumn, column_from_terrain
from .camera import FirstPersonCamera, PlayerState
from .collision import (
    RayHit,
    move_player,
    player_intersects_block,
    ray_cast,
    safe_spawn_height,
)
from .controls import DEFAULT_CONTROL_HINTS, normalise_movement_axes
from .editable_world import MAX_EDITABLE_BLOCK_Y, EditableVoxelWorld
from .game_flow import GameFlowAction, GameFlowController, VoxelScreen
from .hotbar import VoxelHotbar
from .hud import VoxelHudSnapshot
from .interaction import (
    InteractionOutcome,
    InteractionResult,
    VoxelInteractionController,
    invalidated_chunks_for_edit,
)
from .inventory_ui import InventoryScreenController
from .item_rendering import build_dropped_item_vertices
from .meshing import ChunkMeshSnapshot, VoxelChunkMesh, mesh_cache_key
from .natural_blocks import natural_blocks_in_area, tree_shape
from .performance import FrameTimeTracker
from .scenery import scenery_at
from .shaders import (
    FRAGMENT_SHADER,
    HUD_FRAGMENT_SHADER,
    HUD_VERTEX_SHADER,
    OUTLINE_FRAGMENT_SHADER,
    OUTLINE_VERTEX_SHADER,
    OVERLAY_FRAGMENT_SHADER,
    OVERLAY_VERTEX_SHADER,
    SKY_FRAGMENT_SHADER,
    SKY_VERTEX_SHADER,
    VERTEX_SHADER,
)
from .spawn import select_spawn
from .streaming import streaming_chunks
from .texture_atlas import (
    ATLAS_COLUMNS,
    ATLAS_SIZE,
    ATLAS_TILE_SIZE,
    FaceTexture,
    generate_texture_atlas,
)


class VoxelPrototypeError(RuntimeError):
    """Raised when the voxel renderer cannot initialise or render."""


class VoxelContextUnavailableError(VoxelPrototypeError):
    """Raised only when SDL cannot provide the required OpenGL context."""


HUD_REFRESH_INTERVAL_SECONDS: Final = 0.1


@dataclass(frozen=True, slots=True, kw_only=True)
class VoxelPrototypeConfig:
    """Conservative desktop defaults for synchronous voxel streaming."""

    width_pixels: int = 1280
    height_pixels: int = 720
    target_fps: int = 60
    render_distance: int = 1
    world_seed: int = 0
    hidden_window: bool = False
    vsync_enabled: bool = False
    interaction_reach: float = 5.5
    break_cooldown: float = 0.18
    placement_cooldown: float = 0.18
    save_path: Path | None = None
    load_on_start: bool = False
    autosave: bool = False
    bootstrap_inventory: bool = True
    game_flow_enabled: bool = False
    progression_enabled: bool = False
    terrain_config: TerrainGenerationConfig = field(
        default_factory=lambda: TerrainGenerationConfig(octave_count=2)
    )

    def __post_init__(self) -> None:
        for name, integer_value, minimum, maximum in (
            ("width_pixels", self.width_pixels, 160, 7680),
            ("height_pixels", self.height_pixels, 90, 4320),
            ("target_fps", self.target_fps, 1, 360),
            ("render_distance", self.render_distance, 0, 8),
            ("world_seed", self.world_seed, 0, MAX_WORLD_SEED),
        ):
            if isinstance(integer_value, bool) or not isinstance(integer_value, int):
                raise TypeError(f"{name} must be an integer.")
            if not minimum <= integer_value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        if not isinstance(self.hidden_window, bool):
            raise TypeError("hidden_window must be a boolean.")
        if not isinstance(self.vsync_enabled, bool):
            raise TypeError("vsync_enabled must be a boolean.")
        if not isinstance(self.terrain_config, TerrainGenerationConfig):
            raise TypeError("terrain_config must be a TerrainGenerationConfig.")
        for name, numeric_value in (
            ("interaction_reach", self.interaction_reach),
            ("break_cooldown", self.break_cooldown),
            ("placement_cooldown", self.placement_cooldown),
        ):
            if isinstance(numeric_value, bool) or not isinstance(numeric_value, (int, float)):
                raise TypeError(f"{name} must be a number.")
            if not math.isfinite(numeric_value):
                raise ValueError(f"{name} must be finite.")
        if self.interaction_reach <= 0:
            raise ValueError("interaction_reach must be greater than zero.")
        if self.break_cooldown < 0 or self.placement_cooldown < 0:
            raise ValueError("interaction cooldowns must be non-negative.")
        if self.save_path is not None:
            if not isinstance(self.save_path, Path):
                raise TypeError("save_path must be a pathlib.Path or None.")
            if self.save_path.suffix.lower() != ".json":
                raise ValueError("save_path must use the .json suffix.")
            SaveSlot(self.save_path.stem)
        if not isinstance(self.load_on_start, bool):
            raise TypeError("load_on_start must be a boolean.")
        if not isinstance(self.autosave, bool):
            raise TypeError("autosave must be a boolean.")
        if not isinstance(self.bootstrap_inventory, bool):
            raise TypeError("bootstrap_inventory must be a boolean.")
        if not isinstance(self.game_flow_enabled, bool):
            raise TypeError("game_flow_enabled must be a boolean.")
        if not isinstance(self.progression_enabled, bool):
            raise TypeError("progression_enabled must be a boolean.")
        if self.progression_enabled and not self.game_flow_enabled:
            raise ValueError("progression requires game_flow_enabled.")
        if (self.load_on_start or self.autosave) and self.save_path is None:
            raise ValueError("load and autosave require a save_path.")


@dataclass(slots=True)
class GpuChunk:
    """Owned ModernGL resources for one cached render chunk."""

    key: tuple[ChunkCoordinate, int, tuple[int, int, int, int], str, int, int]
    opaque_buffer: moderngl.Buffer
    opaque_array: moderngl.VertexArray
    water_buffer: moderngl.Buffer | None
    water_array: moderngl.VertexArray | None
    mesh: VoxelChunkMesh

    def release(self) -> None:
        try:
            self.opaque_array.release()
        finally:
            self.opaque_buffer.release()
        if self.water_array is not None:
            self.water_array.release()
        if self.water_buffer is not None:
            self.water_buffer.release()


def _water_render_order(
    coordinates: tuple[ChunkCoordinate, ...], *, player_x: float, player_z: float
) -> tuple[ChunkCoordinate, ...]:
    """Order transparent chunks far-to-near with deterministic coordinate ties."""
    return tuple(
        sorted(
            coordinates,
            key=lambda coordinate: (
                -(
                    (coordinate.x * CHUNK_SIZE + CHUNK_SIZE / 2 - player_x) ** 2
                    + (coordinate.y * CHUNK_SIZE + CHUNK_SIZE / 2 - player_z) ** 2
                ),
                coordinate.y,
                coordinate.x,
            ),
        )
    )


def _perspective(*, field_of_view: float, aspect: float, near: float, far: float) -> bytes:
    scale = 1.0 / math.tan(math.radians(field_of_view) / 2.0)
    return struct.pack(
        "16f",
        scale / aspect,
        0,
        0,
        0,
        0,
        scale,
        0,
        0,
        0,
        0,
        (far + near) / (near - far),
        -1,
        0,
        0,
        (2 * far * near) / (near - far),
        0,
    )


def _view_matrix(
    *, position: tuple[float, float, float], forward: tuple[float, float, float]
) -> bytes:
    fx, fy, fz = forward
    right_length = math.hypot(fx, fz) or 1.0
    rx, ry, rz = -fz / right_length, 0.0, fx / right_length
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    px, py, pz = position
    return struct.pack(
        "16f",
        rx,
        ux,
        -fx,
        0,
        ry,
        uy,
        -fy,
        0,
        rz,
        uz,
        -fz,
        0,
        -(rx * px + ry * py + rz * pz),
        -(ux * px + uy * py + uz * pz),
        fx * px + fy * py + fz * pz,
        1,
    )


class VoxelPrototypeApplication:
    """Own the OpenGL context, domain runtime, streaming, physics, and rendering."""

    def __init__(self, *, config: VoxelPrototypeConfig | None = None) -> None:
        self.config = VoxelPrototypeConfig() if config is None else config
        self.world_id = UUID(int=1)
        specification = WorldSpecification(
            name="Voxel Prototype", seed=WorldSeed(value=self.config.world_seed)
        )
        world = WorldModel.create(
            specification=specification,
            created_at=datetime(1970, 1, 1, tzinfo=UTC),
            world_id=WorldId(value=self.world_id),
        )
        self.world = world
        self.runtime = create_terrain_runtime(world=world, config=self.config.terrain_config)
        self.session_context = RuntimeContext.create(
            game_mode=GameMode.NEW_GAME,
            world_seed=self.config.world_seed,
            session_id=self.world_id,
        )
        self.session_context.start()
        self.save_path: Path | None = (
            None
            if self.config.save_path is None
            else self.config.save_path.expanduser().resolve(strict=False)
        )
        self._save_slot: SaveSlot | None = None
        self._save_service: GameSaveService | None = None
        if self.save_path is not None:
            self._save_slot = SaveSlot(self.save_path.stem)
            paths = ProjectPaths(
                project_root=self.save_path.parent,
                save_directory=self.save_path.parent,
                log_directory=self.save_path.parent / "logs",
            )
            storage = RuntimeStorage(paths=paths)
            self.save_path = storage.save_path(self._save_slot)
            self._save_service = GameSaveService(
                repository=SaveRepository(storage=storage),
                context=self.session_context,
                logger=logging.getLogger("open_world_rpg"),
            )
        self.edits = BlockEditStore()
        self._natural_chunk_cache: dict[
            ChunkCoordinate, dict[WorldBlockCoordinate, BlockMaterial]
        ] = {}
        self.editable_world = EditableVoxelWorld(
            column_at=self._column_at,
            edits=self.edits,
            natural_material_at=self._natural_material_at,
        )
        self.interactions = VoxelInteractionController(
            world=self.editable_world,
            edits=self.edits,
            break_cooldown=self.config.break_cooldown,
            placement_cooldown=self.config.placement_cooldown,
            maximum_reach=self.config.interaction_reach,
        )
        self.inventory = create_bootstrap_inventory(
            enabled=self.config.bootstrap_inventory and not self.config.progression_enabled
        )
        self.flow = GameFlowController(
            initial_screen=(
                VoxelScreen.MAIN_MENU if self.config.game_flow_enabled else VoxelScreen.PLAYING
            ),
            continue_available=self.save_path is not None and self.save_path.exists(),
        )
        self.inventory_screen = InventoryScreenController()
        self.progression = SurvivalProgression()
        self.mining = TimedMiningController()
        self.vitals = PlayerVitals()
        self._mining_held = False
        self._jump_was_pressed = False
        self.dropped_items = DroppedItemManager()
        self.last_interaction = InteractionResult.NONE
        self.last_pickup = "none"
        self.last_placement_consumption = "none"
        self.save_message = ""
        self.dirty = False
        self._selection_changed_at = float("-inf")
        self._feedback_until = 0.0
        self._feedback_coordinate: WorldBlockCoordinate | None = None
        self.camera = FirstPersonCamera(yaw_degrees=25.0, pitch_degrees=-12.0)
        self.player = PlayerState(x=8.0, y=20.0, z=8.0)
        self.spawn_x = 8
        self.spawn_z = 8
        self.running = False
        self.fps = 0.0
        self.frame_timing = FrameTimeTracker()
        self.render_distance = self.config.render_distance
        self.loading = False
        self.show_help = False
        self.show_debug = False
        self.mouse_captured = False
        self.target: RayHit | None = None
        self.break_preview = InteractionOutcome(result=InteractionResult.NO_TARGET)
        self.placement_preview = InteractionOutcome(result=InteractionResult.NO_TARGET)
        self.context: moderngl.Context | None = None
        self.program: moderngl.Program | None = None
        self._overlay_program: moderngl.Program | None = None
        self._outline_program: moderngl.Program | None = None
        self._atlas: moderngl.Texture | None = None
        self._sky_program: moderngl.Program | None = None
        self._sky_buffer: moderngl.Buffer | None = None
        self._sky_array: moderngl.VertexArray | None = None
        self._hud_program: moderngl.Program | None = None
        self._hud_buffer: moderngl.Buffer | None = None
        self._hud_array: moderngl.VertexArray | None = None
        self._hud_texture: moderngl.Texture | None = None
        self._drop_buffer: moderngl.Buffer | None = None
        self._drop_array: moderngl.VertexArray | None = None
        self._drop_render_revision = -1
        self._drop_buffer_size = 0
        self._drop_render_key: tuple[object, ...] | None = None
        self._font: pygame.font.Font | None = None
        self._inventory_atlas_surface: pygame.Surface | None = None
        self.hud_snapshot: VoxelHudSnapshot | None = None
        self._crosshair_buffer: moderngl.Buffer | None = None
        self._crosshair_array: moderngl.VertexArray | None = None
        self._target_buffer: moderngl.Buffer | None = None
        self._target_array: moderngl.VertexArray | None = None
        self._clock: pygame.time.Clock | None = None
        self._gpu_chunks: dict[ChunkCoordinate, GpuChunk] = {}
        self._visible: tuple[ChunkCoordinate, ...] = ()
        self._stream_signature: tuple[int, int, int] | None = None
        self._wanted_chunks: tuple[ChunkCoordinate, ...] = ()
        self._terrain_queue: deque[ChunkCoordinate] = deque()
        self._mesh_queue: deque[ChunkCoordinate] = deque()
        self._mesh_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="open-world-rpg-mesh",
        )
        self._mesh_futures: dict[
            ChunkCoordinate,
            tuple[
                tuple[ChunkCoordinate, int, tuple[int, int, int, int], str, int, int],
                Future[VoxelChunkMesh],
            ],
        ] = {}
        self._generation_seconds = 0.0
        self._mesh_seconds = 0.0
        self._next_hud_refresh = float("-inf")
        self._movement_velocity_x = 0.0
        self._movement_velocity_z = 0.0
        self._walk_cycle = 0.0
        self._view_bob = 0.0
        self._current_fov = 72.0

    @property
    def hotbar(self) -> VoxelHotbar:
        """Compatibility projection over the authoritative inventory hotbar."""
        return VoxelHotbar(
            slots=tuple(
                None
                if not isinstance(slot, ItemStack)
                else item_policy(slot.item).placeable_material
                for slot in self.inventory.slots()[:9]
            ),
            selected_index=self.inventory.selected_hotbar_index,
        )

    def initialise(self) -> None:
        """Create a core OpenGL context and safely spawn above generated terrain."""
        stage = "context"
        try:
            pygame.init()
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
            )
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
            if self.config.hidden_window:
                flags |= pygame.HIDDEN
            pygame.display.set_mode(
                (self.config.width_pixels, self.config.height_pixels),
                flags,
                vsync=int(self.config.vsync_enabled),
            )
            pygame.event.clear()
            self.context = moderngl.create_context(require=330)
            stage = "resources"
            self.context.enable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE)
            self.context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.program = self.context.program(
                vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
            )
            self._overlay_program = self.context.program(
                vertex_shader=OVERLAY_VERTEX_SHADER,
                fragment_shader=OVERLAY_FRAGMENT_SHADER,
            )
            self._outline_program = self.context.program(
                vertex_shader=OUTLINE_VERTEX_SHADER,
                fragment_shader=OUTLINE_FRAGMENT_SHADER,
            )
            self._sky_program = self.context.program(
                vertex_shader=SKY_VERTEX_SHADER,
                fragment_shader=SKY_FRAGMENT_SHADER,
            )
            sky = array("f", (-1.0, -1.0, 3.0, -1.0, -1.0, 3.0))
            self._sky_buffer = self.context.buffer(sky.tobytes())
            self._sky_array = self.context.vertex_array(
                self._sky_program,
                [(self._sky_buffer, "2f", "in_position")],
            )
            self._hud_program = self.context.program(
                vertex_shader=HUD_VERTEX_SHADER,
                fragment_shader=HUD_FRAGMENT_SHADER,
            )
            hud_quad = array(
                "f",
                (
                    -1.0,
                    1.0,
                    0.0,
                    1.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    1.0,
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    -1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
            )
            self._hud_buffer = self.context.buffer(hud_quad.tobytes())
            self._hud_array = self.context.vertex_array(
                self._hud_program,
                [(self._hud_buffer, "2f 2f", "in_position", "in_uv")],
            )
            self._hud_texture = self.context.texture((1024, 512), 4)
            self._hud_texture.filter = moderngl.NEAREST, moderngl.NEAREST
            self._hud_texture.use(location=1)
            cast(moderngl.Uniform, self._hud_program["hud_texture"]).value = 1
            self._font = pygame.font.Font(None, 22)
            self._atlas = self.context.texture(
                (ATLAS_SIZE, ATLAS_SIZE),
                4,
                generate_texture_atlas(),
            )
            self._atlas.filter = moderngl.NEAREST, moderngl.NEAREST
            self._atlas.repeat_x = False
            self._atlas.repeat_y = False
            self._atlas.use(location=0)
            cast(moderngl.Uniform, self.program["atlas"]).value = 0
            crosshair = array("f", (-0.012, 0.0, 0.012, 0.0, 0.0, -0.02, 0.0, 0.02))
            self._crosshair_buffer = self.context.buffer(crosshair.tobytes())
            self._crosshair_array = self.context.vertex_array(
                self._overlay_program,
                [(self._crosshair_buffer, "2f", "in_position")],
            )
            self._target_buffer = self.context.buffer(reserve=24 * 7 * 4)
            self._target_array = self.context.vertex_array(
                self._outline_program,
                [(self._target_buffer, "3f 4f", "in_position", "in_colour")],
            )
            self._clock = pygame.time.Clock()
            stage = "terrain startup"
            self._stream()
            spawn_x, spawn_z = select_spawn(
                column_at=self._column_at,
                blocked_at=self._has_scenery,
            )
            self.spawn_x = spawn_x
            self.spawn_z = spawn_z
            self.player = PlayerState(
                x=float(spawn_x) + 0.5,
                y=safe_spawn_height(
                    world_x=spawn_x,
                    world_z=spawn_z,
                    height_at=self._height_at,
                ),
                z=float(spawn_z) + 0.5,
                grounded=True,
            )
            if self.config.load_on_start:
                stage = "save restoration"
                if not self._load_edits():
                    raise VoxelPrototypeError("Could not load the requested voxel save.")
            self._stream()
            self._capture_mouse(self.flow.gameplay_active)
            self.running = True
        except Exception as error:
            self.shutdown()
            if stage == "context":
                raise VoxelContextUnavailableError(
                    "OpenGL 3.3 context unavailable in this environment."
                ) from error
            raise VoxelPrototypeError(
                f"Could not initialise the voxel prototype during {stage}."
            ) from error

    def run(self, *, max_frames: int | None = None) -> int:
        """Run interactively or for a bounded graphical smoke test."""
        if max_frames is not None:
            if isinstance(max_frames, bool) or not isinstance(max_frames, int):
                raise TypeError("max_frames must be an integer or None.")
            if max_frames <= 0:
                raise ValueError("max_frames must be greater than zero.")
        if not self.running:
            self.initialise()
        frames = 0
        completed = False
        try:
            while self.running and (max_frames is None or frames < max_frames):
                frame_started = time.perf_counter()
                assert self._clock is not None
                delta = min(0.05, self._clock.tick(self.config.target_fps) / 1000.0)
                self.process_events()
                self.update(delta)
                self.render()
                elapsed = max(time.perf_counter() - frame_started, 1e-9)
                self.frame_timing.record(elapsed)
                self.fps = self.frame_timing.snapshot.average_fps
                frames += 1
            completed = True
            return 0
        finally:
            if completed and self.config.autosave and self.dirty:
                self._save_edits()
            self.shutdown()

    def process_events(self) -> None:
        """Handle mouse capture and discrete prototype controls."""
        events = pygame.event.get()
        if self.config.game_flow_enabled:
            for event in events:
                self._process_flow_event(event)
            return
        capture_just_lost = False
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == getattr(pygame, "WINDOWFOCUSLOST", -1):
                if self.mouse_captured:
                    self._capture_mouse(False)
                    capture_just_lost = True
            elif event.type == pygame.MOUSEMOTION and self.mouse_captured:
                self.camera = self.camera.looked(
                    delta_x=float(event.rel[0]), delta_y=float(event.rel[1])
                )
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.mouse_captured:
                        self._capture_mouse(False)
                        capture_just_lost = True
                    else:
                        self.running = False
                elif event.key in (pygame.K_F1, pygame.K_h):
                    self.show_help = not self.show_help
                elif event.key == pygame.K_F3:
                    self.show_debug = not self.show_debug
                elif event.key == pygame.K_f:
                    self.player = PlayerState(
                        x=self.player.x,
                        y=self.player.y,
                        z=self.player.z,
                        flying=not self.player.flying,
                    )
                elif event.key == pygame.K_r:
                    self.player = PlayerState(
                        x=float(self.spawn_x) + 0.5,
                        y=safe_spawn_height(
                            world_x=self.spawn_x,
                            world_z=self.spawn_z,
                            height_at=self._height_at,
                        ),
                        z=float(self.spawn_z) + 0.5,
                        grounded=True,
                    )
                elif event.key == pygame.K_F5:
                    self.render_distance = max(1, self.render_distance - 1)
                    self._stream_signature = None
                    self._stream(blocking=False)
                elif event.key == pygame.K_F6:
                    self.render_distance = min(4, self.render_distance + 1)
                    self._stream_signature = None
                    self._stream(blocking=False)
                elif event.key == pygame.K_F7:
                    self._save_edits()
                elif event.key == pygame.K_F8:
                    self._load_edits()
                elif pygame.K_1 <= event.key <= pygame.K_9 and self.inventory.select_hotbar(
                    event.key - pygame.K_1
                ):
                    self.mining.cancel("selected tool changed")
                    self.dirty = True
                    self._selection_changed_at = pygame.time.get_ticks() / 1000.0
            elif event.type == pygame.MOUSEWHEEL:
                if self.inventory.cycle_hotbar(event.y):
                    self.mining.cancel("selected tool changed")
                    self.dirty = True
                    self._selection_changed_at = pygame.time.get_ticks() / 1000.0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not self.mouse_captured:
                    if capture_just_lost:
                        continue
                    self._capture_mouse(True)
                elif event.button in (4, 5):
                    if self.inventory.cycle_hotbar(1 if event.button == 4 else -1):
                        self.mining.cancel("selected tool changed")
                        self.dirty = True
                        self._selection_changed_at = pygame.time.get_ticks() / 1000.0
                elif event.button == 1:
                    self._refresh_interaction_previews()
                    self._mining_held = self.break_preview.allowed
                    if not self._mining_held:
                        self._apply_interaction(self.break_preview)
                elif event.button == 3:
                    self._mining_held = False
                    self.mining.cancel("block placement began")
                    self._apply_interaction(
                        self.interactions.place_inventory_block(
                            target=self.target,
                            inventory=self.inventory,
                            player=self.player,
                            now=pygame.time.get_ticks() / 1000.0,
                        )
                    )
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._mining_held = False
                self.mining.cancel("mining input released")

    def _process_flow_event(self, event: Any) -> None:
        """Route one event through the active menu, inventory, or gameplay screen."""
        if event.type == pygame.QUIT:
            self.running = False
            return
        if self.flow.screen is VoxelScreen.PLAYING:
            self._process_flow_gameplay_event(event)
            return
        if event.type == getattr(pygame, "WINDOWFOCUSLOST", -1):
            self._capture_mouse(False)
            return
        if event.type == pygame.KEYDOWN:
            self._process_overlay_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self._process_overlay_click(event)
        elif event.type == pygame.MOUSEWHEEL and self.flow.screen is VoxelScreen.INVENTORY:
            self.inventory_screen.move_recipe_selection(-event.y)

    def _process_flow_gameplay_event(self, event: Any) -> None:
        """Handle gameplay controls while the v0.9.0 screen flow is enabled."""
        if event.type == getattr(pygame, "WINDOWFOCUSLOST", -1):
            self._open_pause_menu()
        elif event.type == pygame.MOUSEMOTION and self.mouse_captured:
            self.camera = self.camera.looked(
                delta_x=float(event.rel[0]), delta_y=float(event.rel[1])
            )
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._open_pause_menu()
            elif event.key in (pygame.K_e, pygame.K_TAB):
                self._open_inventory_screen()
            elif event.key in (pygame.K_F1, pygame.K_h):
                self.show_help = not self.show_help
            elif event.key == pygame.K_F3:
                self.show_debug = not self.show_debug
            elif event.key == pygame.K_f:
                self.player = PlayerState(
                    x=self.player.x,
                    y=self.player.y,
                    z=self.player.z,
                    flying=not self.player.flying,
                )
            elif event.key == pygame.K_r:
                self._place_player_at_spawn()
            elif event.key == pygame.K_F5:
                self.render_distance = max(1, self.render_distance - 1)
                self._stream_signature = None
                self._stream(blocking=False)
            elif event.key == pygame.K_F6:
                self.render_distance = min(4, self.render_distance + 1)
                self._stream_signature = None
                self._stream(blocking=False)
            elif event.key == pygame.K_F7:
                self._save_edits()
            elif event.key == pygame.K_F8:
                self._load_edits()
            elif pygame.K_1 <= event.key <= pygame.K_9 and self.inventory.select_hotbar(
                event.key - pygame.K_1
            ):
                self._on_inventory_changed("selected tool changed")
        elif event.type == pygame.MOUSEWHEEL:
            if self.inventory.cycle_hotbar(event.y):
                self._on_inventory_changed("selected tool changed")
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if not self.mouse_captured:
                self._capture_mouse(True)
            elif event.button in (4, 5):
                if self.inventory.cycle_hotbar(1 if event.button == 4 else -1):
                    self._on_inventory_changed("selected tool changed")
            elif event.button == 1:
                self._refresh_interaction_previews()
                self._mining_held = self.break_preview.allowed
                if not self._mining_held:
                    self._apply_interaction(self.break_preview)
            elif event.button == 3:
                self._mining_held = False
                self.mining.cancel("block placement began")
                self._apply_interaction(
                    self.interactions.place_inventory_block(
                        target=self.target,
                        inventory=self.inventory,
                        player=self.player,
                        now=pygame.time.get_ticks() / 1000.0,
                    )
                )
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._mining_held = False
            self.mining.cancel("mining input released")

    def _process_overlay_key(self, key: int) -> None:
        if self.flow.screen is VoxelScreen.GUIDE:
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE, pygame.K_RIGHT):
                self._advance_guide()
            elif key == pygame.K_ESCAPE:
                self.progression.dismiss_guide()
                self.flow.close_guide()
                self._capture_mouse(True)
                self.dirty = True
            return
        if self.flow.screen is VoxelScreen.INVENTORY:
            if key in (pygame.K_ESCAPE, pygame.K_e, pygame.K_TAB):
                self.flow.close_inventory()
                self._capture_mouse(True)
            elif key == pygame.K_LEFT:
                self.inventory_screen.move_slot_selection(columns=9, delta_x=-1, delta_y=0)
            elif key == pygame.K_RIGHT:
                self.inventory_screen.move_slot_selection(columns=9, delta_x=1, delta_y=0)
            elif key == pygame.K_UP:
                self.inventory_screen.move_slot_selection(columns=9, delta_x=0, delta_y=-1)
            elif key == pygame.K_DOWN:
                self.inventory_screen.move_slot_selection(columns=9, delta_x=0, delta_y=1)
            elif key in (pygame.K_PAGEUP, pygame.K_LEFTBRACKET):
                self.inventory_screen.move_recipe_selection(-1)
            elif key in (pygame.K_PAGEDOWN, pygame.K_RIGHTBRACKET):
                self.inventory_screen.move_recipe_selection(1)
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                result = self.inventory_screen.activate_slot(self.inventory)
                self._apply_inventory_ui_result(result.changed, result.message)
            elif key == pygame.K_q:
                result = self.inventory_screen.quick_move_selected(self.inventory)
                self._apply_inventory_ui_result(result.changed, result.message)
            elif key == pygame.K_c:
                self._craft_selected_recipe()
            return

        if key == pygame.K_ESCAPE:
            if self.flow.screen is VoxelScreen.PAUSED:
                self.flow.resume()
                self._capture_mouse(True)
            elif self.flow.screen is VoxelScreen.MAIN_MENU:
                self.running = False
            elif self.flow.screen is VoxelScreen.DEAD:
                self.flow.return_to_main_menu()
            elif self.flow.screen is VoxelScreen.COMPLETED:
                self.flow.continue_playing()
                self._capture_mouse(True)
            return
        if key in (pygame.K_UP, pygame.K_w):
            self.flow.move_selection(-1)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.flow.move_selection(1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._activate_flow_action(self.flow.activate_selected())
        elif self.flow.screen is VoxelScreen.DEAD and key == pygame.K_r:
            self._activate_flow_action(GameFlowAction.RESPAWN)

    def _process_overlay_click(self, event: Any) -> None:
        if self.flow.screen is VoxelScreen.GUIDE:
            if event.button == 1:
                self._advance_guide()
            return
        hud_x, hud_y = self._hud_pointer(event.pos)
        if self.flow.screen is VoxelScreen.INVENTORY:
            slot = self._inventory_slot_at(hud_x, hud_y)
            if slot is not None:
                self.inventory_screen.select_slot(slot)
                if event.button == 1:
                    result = self.inventory_screen.activate_slot(self.inventory)
                    self._apply_inventory_ui_result(result.changed, result.message)
                elif event.button == 3:
                    result = self.inventory_screen.quick_move_selected(self.inventory)
                    self._apply_inventory_ui_result(result.changed, result.message)
                return
            recipe = self._recipe_at(hud_x, hud_y)
            if recipe is not None:
                self.inventory_screen.selected_recipe_index = recipe
                if event.button == 1:
                    self._craft_selected_recipe()
            return
        option = self._menu_option_at(hud_x, hud_y)
        if option is not None:
            self.flow.selected_index = option
            self._activate_flow_action(self.flow.activate_selected())

    def _activate_flow_action(self, action: GameFlowAction) -> None:
        if action is GameFlowAction.NEW_WORLD:
            self._reset_new_world()
            self.flow.start_new_world()
            if self.config.progression_enabled and not self.progression.guide_completed:
                self.flow.open_guide()
                self._capture_mouse(False)
            else:
                self._capture_mouse(True)
        elif action is GameFlowAction.CONTINUE:
            if self._load_edits() and self.flow.continue_world():
                if self.config.progression_enabled and not self.progression.guide_completed:
                    self.flow.open_guide()
                    self._capture_mouse(False)
                else:
                    self._capture_mouse(True)
        elif action is GameFlowAction.RESUME:
            if self.flow.resume():
                self._capture_mouse(True)
        elif action is GameFlowAction.SAVE:
            self._save_edits()
        elif action is GameFlowAction.SAVE_AND_QUIT:
            if self._save_edits():
                self.running = False
        elif action is GameFlowAction.RESPAWN:
            if self.flow.respawn():
                self._respawn_after_death()
                self._capture_mouse(True)
        elif action is GameFlowAction.CONTINUE_PLAYING:
            if self.flow.continue_playing():
                self._capture_mouse(True)
        elif action is GameFlowAction.QUIT:
            if self.flow.screen in (VoxelScreen.DEAD, VoxelScreen.COMPLETED):
                self.flow.return_to_main_menu()
            else:
                self.running = False

    def _advance_guide(self) -> None:
        if self.progression.next_guide_page():
            self.flow.close_guide()
            self._capture_mouse(True)
            self.dirty = True

    def _open_pause_menu(self) -> None:
        if self.flow.pause():
            self._mining_held = False
            self.mining.cancel("game paused")
            self._capture_mouse(False)

    def _open_inventory_screen(self) -> None:
        if self.flow.open_inventory():
            self.inventory_screen.reset()
            self._mining_held = False
            self.mining.cancel("inventory opened")
            self._capture_mouse(False)

    def _on_inventory_changed(self, cancellation_reason: str) -> None:
        self.mining.cancel(cancellation_reason)
        self.dirty = True
        self._selection_changed_at = pygame.time.get_ticks() / 1000.0

    def _apply_inventory_ui_result(self, changed: bool, message: str) -> None:
        self.save_message = message
        self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
        if changed:
            self._on_inventory_changed("inventory changed")

    def _craft_selected_recipe(self) -> None:
        recipe = self.inventory_screen.selected_recipe
        if self.config.progression_enabled and not self.progression.recipe_unlocked(
            recipe.identifier
        ):
            self._apply_inventory_ui_result(False, "Craft a wooden pickaxe first")
            return
        attempt = self.inventory_screen.craft_selected(self.inventory)
        if attempt.result is CraftingResult.CRAFTED:
            assert attempt.recipe is not None
            advanced = (
                self.progression.record_craft(attempt.recipe.output_item, self.inventory)
                if self.config.progression_enabled
                else False
            )
            self._apply_inventory_ui_result(True, f"Crafted {attempt.recipe.output_label}")
            if advanced:
                self._on_progression_advanced()
        else:
            self._apply_inventory_ui_result(False, attempt.result.value.capitalize())

    def _on_progression_advanced(self) -> None:
        self.dirty = True
        if self.progression.completed:
            self.save_message = "Stone Age reached — survival loop complete"
            self._feedback_until = pygame.time.get_ticks() / 1000.0 + 2.5
            self.flow.mark_completed()
            self._capture_mouse(False)
        else:
            objective = self.progression.objective(self.inventory)
            self.save_message = f"New objective: {objective.title}"
            self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.75

    def _reset_new_world(self) -> None:
        self.edits = BlockEditStore()
        self._natural_chunk_cache.clear()
        self.editable_world = EditableVoxelWorld(
            column_at=self._column_at,
            edits=self.edits,
            natural_material_at=self._natural_material_at,
        )
        self.interactions = VoxelInteractionController(
            world=self.editable_world,
            edits=self.edits,
            break_cooldown=self.config.break_cooldown,
            placement_cooldown=self.config.placement_cooldown,
            maximum_reach=self.config.interaction_reach,
        )
        self.inventory = create_bootstrap_inventory(
            enabled=self.config.bootstrap_inventory and not self.config.progression_enabled
        )
        self.dropped_items = DroppedItemManager()
        self.progression = SurvivalProgression()
        if self.config.progression_enabled:
            self._plant_starter_tree()
        self.vitals = PlayerVitals()
        self.last_interaction = InteractionResult.NONE
        self.last_pickup = "none"
        self.last_placement_consumption = "none"
        self.save_message = "New world started"
        self.dirty = False
        self.target = None
        self._stream_signature = None
        self._place_player_at_spawn()
        self._refresh_interaction_previews()
        self._stream(blocking=False)

    def _plant_starter_tree(self) -> None:
        forward_x, _, forward_z = self.camera.forward
        tree_x = math.floor(self.spawn_x + 0.5 + forward_x * 5.0)
        tree_z = math.floor(self.spawn_z + 0.5 + forward_z * 5.0)
        ground_y = self._height_at(tree_x, tree_z)
        shape = tree_shape(world_x=tree_x, ground_y=ground_y, world_z=tree_z)
        for coordinate in shape.trunk:
            self.edits.set_block(coordinate, BlockMaterial.WOOD)
        for coordinate in shape.leaves:
            if coordinate.y <= MAX_EDITABLE_BLOCK_Y:
                self.edits.set_block(coordinate, BlockMaterial.LEAVES)

    def _place_player_at_spawn(self) -> None:
        self.player = PlayerState(
            x=float(self.spawn_x) + 0.5,
            y=safe_spawn_height(
                world_x=self.spawn_x,
                world_z=self.spawn_z,
                height_at=self._height_at,
            ),
            z=float(self.spawn_z) + 0.5,
            grounded=True,
        )

    @staticmethod
    def _inventory_slot_at(hud_x: int, hud_y: int) -> int | None:
        start_x, start_y, size, gap = 76, 138, 48, 7
        column = (hud_x - start_x) // (size + gap)
        row = (hud_y - start_y) // (size + gap)
        if not 0 <= column < 9 or not 0 <= row < 3:
            return None
        within_x = (hud_x - start_x) % (size + gap)
        within_y = (hud_y - start_y) % (size + gap)
        if within_x >= size or within_y >= size:
            return None
        return row * 9 + column

    def _recipe_at(self, hud_x: int, hud_y: int) -> int | None:
        if not 626 <= hud_x < 970 or hud_y < 116:
            return None
        offset = hud_y - 116
        index, within = divmod(offset, 46)
        if within >= 38 or not 0 <= index < len(self.inventory_screen.recipes):
            return None
        return index

    def _menu_option_at(self, hud_x: int, hud_y: int) -> int | None:
        options = self.flow.options
        if not options or not 340 <= hud_x < 684 or hud_y < 220:
            return None
        offset = hud_y - 220
        index, within = divmod(offset, 54)
        if within >= 42 or not 0 <= index < len(options):
            return None
        return index

    @staticmethod
    def _hud_pointer(position: tuple[int, int]) -> tuple[int, int]:
        width, height = pygame.display.get_window_size()
        return (
            round(position[0] * 1024 / max(1, width)),
            round(position[1] * 512 / max(1, height)),
        )

    def update(self, delta_seconds: float) -> None:
        """Apply first-person motion, physics, targeting, and streaming."""
        if self.config.game_flow_enabled and not self.flow.gameplay_active:
            self._mining_held = False
            self.mining.cancel("gameplay overlay active")
            self.target = None
            self._refresh_interaction_previews()
            return
        keys = pygame.key.get_pressed()
        gameplay_active = self.mouse_captured
        raw_forward = int(keys[pygame.K_w]) - int(keys[pygame.K_s]) if gameplay_active else 0
        raw_sideways = int(keys[pygame.K_d]) - int(keys[pygame.K_a]) if gameplay_active else 0
        axes = normalise_movement_axes(forward=raw_forward, sideways=raw_sideways)
        microseconds = max(0, round(delta_seconds * 1_000_000))
        sprinting = bool(
            gameplay_active
            and keys[pygame.K_LSHIFT]
            and axes.active
            and not self.player.flying
            and self.vitals.can_sprint
        )
        if self.vitals.update_stamina(
            microseconds, sprinting=sprinting, active=self.mouse_captured
        ):
            self.dirty = True
        movement_speed = 9.0 if sprinting else 5.2
        flat_forward = (math.sin(math.radians(self.camera.yaw_degrees)), 0.0)
        flat_right = (math.cos(math.radians(self.camera.yaw_degrees)), 0.0)
        desired_velocity_x = (
            flat_forward[0] * axes.forward + flat_right[0] * axes.sideways
        ) * movement_speed
        desired_velocity_z = (
            -math.cos(math.radians(self.camera.yaw_degrees)) * axes.forward
            + math.sin(math.radians(self.camera.yaw_degrees)) * axes.sideways
        ) * movement_speed
        if not gameplay_active:
            self._movement_velocity_x = 0.0
            self._movement_velocity_z = 0.0
        else:
            response = 1.0 - math.exp(-(22.0 if axes.active else 30.0) * delta_seconds)
            self._movement_velocity_x += (desired_velocity_x - self._movement_velocity_x) * response
            self._movement_velocity_z += (desired_velocity_z - self._movement_velocity_z) * response
        delta_x = self._movement_velocity_x * delta_seconds
        delta_z = self._movement_velocity_z * delta_seconds
        target_fov = 78.0 if sprinting and axes.active else 72.0
        self._current_fov += (target_fov - self._current_fov) * (
            1.0 - math.exp(-10.0 * delta_seconds)
        )
        jump_pressed = bool(gameplay_active and keys[pygame.K_SPACE])
        jump_requested = (
            jump_pressed
            and not self._jump_was_pressed
            and self.player.grounded
            and not self.player.flying
        )
        jump_allowed = jump_requested and self.vitals.jump()
        if jump_requested and jump_allowed:
            self.dirty = True
        elif jump_requested:
            self.save_message = "Not enough stamina to jump"
            self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
        self._jump_was_pressed = jump_pressed
        previous_player = self.player

        # Frame-local column cache to avoid regenerating terrain during collision checks
        frame_column_cache: dict[tuple[int, int], BlockColumn] = {}

        def cached_solid_at(x: int, y: int, z: int) -> bool:
            key = (x, z)
            if key not in frame_column_cache:
                frame_column_cache[key] = self._column_at(x, z)
            column = frame_column_cache[key]
            coord = WorldBlockCoordinate(x=x, y=y, z=z)
            edit = self.editable_world.edits.get(coord)
            if edit is not None:
                return edit.material.is_solid
            if y <= column.ground_height:
                if y == column.ground_height or (y >= column.ground_height - 3):
                    return True
                return BlockMaterial.STONE.is_solid
            if column.water is not None and y < column.surface_height:
                return BlockMaterial.WATER.is_solid
            return self._natural_material_at(x, y, z).is_solid

        self.player = move_player(
            player=self.player,
            delta_x=delta_x,
            delta_z=delta_z,
            delta_seconds=delta_seconds,
            height_at=self._height_at,
            solid_at=cached_solid_at,
            jump=jump_allowed,
        )
        if self.player.flying:
            vertical = int(jump_pressed) - int(
                gameplay_active and (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])
            )
            self.player = PlayerState(
                x=self.player.x,
                y=self.player.y + vertical * 5.0 * delta_seconds,
                z=self.player.z,
                flying=True,
            )
            self.vitals.reset_fall()
        elif not self.player.grounded and self.player.y < previous_player.y:
            self.vitals.record_airborne_descent(round((previous_player.y - self.player.y) * 1_000))
        elif self.player.grounded and not previous_player.grounded:
            damage = self.vitals.land()
            if damage:
                self.dirty = True
                self.save_message = f"Fall damage: {damage}"
                self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
            if self.vitals.snapshot.health_milli == 0:
                if self.config.game_flow_enabled:
                    self.flow.mark_dead()
                    self._mining_held = False
                    self.mining.cancel("player died")
                    self.target = None
                    self._capture_mouse(False)
                else:
                    self._respawn_after_death()
        if self.dropped_items.update(delta_seconds, solid_at=self._solid_at):
            self.dirty = True
        pickups = self.dropped_items.pickup_near(
            position=(self.player.x, self.player.y + 0.9, self.player.z),
            inventory=self.inventory,
        )
        self._apply_pickups(pickups)
        horizontal_speed = math.hypot(self._movement_velocity_x, self._movement_velocity_z)
        if self.player.grounded and axes.active and gameplay_active:
            self._walk_cycle += horizontal_speed * delta_seconds * 1.65
            target_bob = math.sin(self._walk_cycle * math.tau) * (0.055 if sprinting else 0.035)
        else:
            target_bob = 0.0
        self._view_bob += (target_bob - self._view_bob) * (1.0 - math.exp(-14.0 * delta_seconds))
        self._stream(blocking=False)
        self.target = ray_cast(
            origin=(self.player.x, self.player.y + 1.62, self.player.z),
            direction=self.camera.forward,
            block_at=self.editable_world.material_at,
            maximum_distance=self.config.interaction_reach,
        )
        self._refresh_interaction_previews()
        self._update_mining(microseconds)

    def _apply_pickups(self, pickups: tuple[PickupResult, ...]) -> None:
        if not pickups:
            return
        self.dirty = True
        progression_advanced = False
        for pickup in pickups:
            item = cast(ItemType, pickup.item)
            if self.config.progression_enabled:
                progression_advanced = (
                    self.progression.record_pickup(item, self.inventory) or progression_advanced
                )
        pickup = pickups[-1]
        item = cast(ItemType, pickup.item)
        self.save_message = f"Picked up {item.display_name} x{pickup.accepted}"
        self.last_pickup = self.save_message
        self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
        if progression_advanced:
            self._on_progression_advanced()

    def render(self) -> None:
        """Draw cached chunk meshes in deterministic order with fog and water motion."""
        if self.context is None or self.program is None:
            raise VoxelPrototypeError("Voxel prototype is not initialised.")
        width, height = pygame.display.get_window_size()
        self.context.viewport = (0, 0, width, height)
        self.context.clear(0.43, 0.68, 0.88, 1.0, depth=1.0)
        if self._sky_array is not None:
            self.context.disable(moderngl.DEPTH_TEST)
            self._sky_array.render(moderngl.TRIANGLES)
            self.context.enable(moderngl.DEPTH_TEST)
        projection = _perspective(
            field_of_view=self._current_fov,
            aspect=width / max(1, height),
            near=0.1,
            far=float((self.render_distance + 2) * CHUNK_SIZE),
        )
        view = _view_matrix(
            position=(self.player.x, self.player.y + 1.62 + self._view_bob, self.player.z),
            forward=self.camera.forward,
        )
        cast(moderngl.Uniform, self.program["projection"]).write(projection)
        cast(moderngl.Uniform, self.program["view"]).write(view)
        if self._outline_program is not None:
            cast(moderngl.Uniform, self._outline_program["projection"]).write(projection)
            cast(moderngl.Uniform, self._outline_program["view"]).write(view)
        cast(moderngl.Uniform, self.program["fog_colour"]).value = (0.43, 0.68, 0.88)
        cast(moderngl.Uniform, self.program["fog_near"]).value = float(
            self.render_distance * CHUNK_SIZE
        )
        fog_far = float((self.render_distance + 1.5) * CHUNK_SIZE)
        cast(moderngl.Uniform, self.program["fog_far"]).value = fog_far
        cast(moderngl.Uniform, self.program["water_time"]).value = pygame.time.get_ticks() / 1000.0
        triangles = 0
        for coordinate in self._visible:
            gpu = self._gpu_chunks.get(coordinate)
            if gpu is None:
                continue
            gpu.opaque_array.render(moderngl.TRIANGLES)
            triangles += gpu.mesh.triangle_count
        self._refresh_drop_gpu()
        if self._drop_array is not None:
            self.context.disable(moderngl.CULL_FACE)
            self._drop_array.render(moderngl.TRIANGLES)
            self.context.enable(moderngl.CULL_FACE)
            triangles += len(self.dropped_items) * 4
        cast(Any, self.context).depth_mask = False
        for coordinate in _water_render_order(
            self._visible,
            player_x=self.player.x,
            player_z=self.player.z,
        ):
            gpu = self._gpu_chunks.get(coordinate)
            water_array = None if gpu is None else gpu.water_array
            if water_array is not None:
                water_array.render(moderngl.TRIANGLES)
        cast(Any, self.context).depth_mask = True
        if self.target is not None:
            target_colour = (
                (1.0, 0.93, 0.32, 1.0) if self.break_preview.allowed else (1.0, 0.35, 0.28, 1.0)
            )
            self._render_target_outline(self.target, colour=target_colour)
        selected_stack = self.inventory.selected_stack
        if selected_stack is not None and self.placement_preview.coordinate is not None:
            destination = self.placement_preview.coordinate
            placement_colour = (
                (0.30, 0.95, 0.48, 1.0)
                if self.placement_preview.allowed
                else (1.0, 0.30, 0.24, 1.0)
            )
            self._render_target_outline(
                RayHit(
                    x=destination.x,
                    y=destination.y,
                    z=destination.z,
                    distance=0.0,
                ),
                colour=placement_colour,
            )
        if (
            self._feedback_coordinate is not None
            and pygame.time.get_ticks() / 1000.0 < self._feedback_until
        ):
            self._render_target_outline(
                RayHit(
                    x=self._feedback_coordinate.x,
                    y=self._feedback_coordinate.y,
                    z=self._feedback_coordinate.z,
                    distance=0.0,
                ),
                colour=(0.35, 0.90, 1.0, 1.0),
            )
        if self._crosshair_array is not None:
            self.context.disable(moderngl.DEPTH_TEST)
            self._crosshair_array.render(moderngl.LINES)
            self.context.enable(moderngl.DEPTH_TEST)
        self._render_hud(triangles)
        pygame.display.set_caption(self._caption(triangles))
        pygame.display.flip()

    def shutdown(self) -> None:
        """Explicitly release GPU objects and suspend active terrain."""
        for _, future in self._mesh_futures.values():
            future.cancel()
        self._mesh_futures.clear()
        self._mesh_queue.clear()
        self._terrain_queue.clear()
        self._mesh_executor.shutdown(wait=False, cancel_futures=True)
        for gpu in self._gpu_chunks.values():
            try:
                gpu.release()
            except Exception:
                logging.getLogger("open_world_rpg").exception(
                    "Could not release a voxel chunk GPU resource."
                )
        self._gpu_chunks.clear()
        for resource in (
            self._drop_array,
            self._drop_buffer,
            self._target_array,
            self._target_buffer,
            self._crosshair_array,
            self._crosshair_buffer,
            self._overlay_program,
            self._outline_program,
            self._atlas,
            self._sky_array,
            self._sky_buffer,
            self._sky_program,
            self._hud_array,
            self._hud_buffer,
            self._hud_texture,
            self._hud_program,
            self.program,
        ):
            if resource is not None:
                try:
                    resource.release()
                except Exception:
                    logging.getLogger("open_world_rpg").exception(
                        "Could not release an OpenGL resource."
                    )
        self._target_array = None
        self._target_buffer = None
        self._crosshair_array = None
        self._crosshair_buffer = None
        self._overlay_program = None
        self._outline_program = None
        self._atlas = None
        self._sky_array = None
        self._sky_buffer = None
        self._sky_program = None
        self._hud_array = None
        self._hud_buffer = None
        self._hud_texture = None
        self._hud_program = None
        self._drop_array = None
        self._drop_buffer = None
        self._drop_buffer_size = 0
        self._drop_render_key = None
        self._font = None
        self._inventory_atlas_surface = None
        self.program = None
        try:
            for coordinate in self.runtime.coordinates():
                if self.runtime.metadata_at(coordinate).state is ChunkState.ACTIVE:
                    self.runtime.suspend(coordinate)
        finally:
            try:
                self._capture_mouse(False)
            except pygame.error:
                self.mouse_captured = False
            self.running = False
            if self.context is not None:
                try:
                    self.context.release()
                except Exception:
                    logging.getLogger("open_world_rpg").exception(
                        "Could not release the OpenGL context."
                    )
            self.context = None
            pygame.quit()

    def _refresh_drop_gpu(self) -> None:
        """Refresh the single drop batch only when authoritative state changes."""
        if self.context is None or self.program is None:
            return
        if self._drop_render_revision == self.dropped_items.revision:
            return
        items = self.dropped_items.items()
        render_key = tuple(
            (
                item.identifier,
                item.item,
                item.quantity,
                item.position,
                item.velocity,
                item.settled,
                math.floor(item.age * 10) if item.settled else item.age,
            )
            for item in items
        )
        if render_key == self._drop_render_key:
            self._drop_render_revision = self.dropped_items.revision
            return
        vertices = build_dropped_item_vertices(items)
        if vertices and self._drop_buffer is not None and len(vertices) == self._drop_buffer_size:
            self._drop_buffer.write(vertices)
            self._drop_render_revision = self.dropped_items.revision
            self._drop_render_key = render_key
            return
        if self._drop_array is not None:
            self._drop_array.release()
        if self._drop_buffer is not None:
            self._drop_buffer.release()
        self._drop_array = None
        self._drop_buffer = None
        self._drop_buffer_size = 0
        if vertices:
            self._drop_buffer = self.context.buffer(vertices)
            self._drop_buffer_size = len(vertices)
            self._drop_array = self.context.vertex_array(
                self.program,
                [
                    (
                        self._drop_buffer,
                        "3f 2f 1f",
                        "in_position",
                        "in_uv",
                        "in_shade",
                    )
                ],
            )
        self._drop_render_revision = self.dropped_items.revision
        self._drop_render_key = render_key

    def _stream(self, *, blocking: bool = True) -> None:
        """Synchronise or incrementally stream the chunks around the player."""
        signature = (
            math.floor(self.player.x) // CHUNK_SIZE,
            math.floor(self.player.z) // CHUNK_SIZE,
            self.render_distance,
        )
        if signature != self._stream_signature:
            wanted = streaming_chunks(
                world_x=self.player.x,
                world_z=self.player.z,
                render_distance=self.render_distance,
            )
            required = self._required_terrain_chunks(wanted)
            self._wanted_chunks = wanted
            self._terrain_queue = deque(
                coordinate for coordinate in required if not self.runtime.contains(coordinate)
            )
            self._mesh_queue = deque(wanted)
            self._stream_signature = signature
            wanted_set = set(wanted)
            for coordinate in self.runtime.coordinates():
                if coordinate not in wanted_set and (
                    self.runtime.metadata_at(coordinate).state is ChunkState.ACTIVE
                ):
                    self.runtime.suspend(coordinate)
            for coordinate in tuple(self._gpu_chunks):
                if coordinate not in wanted_set:
                    self._gpu_chunks.pop(coordinate).release()

        if blocking:
            self._complete_stream_blocking()
        else:
            self._pump_streaming()

    def _required_terrain_chunks(
        self, wanted: tuple[ChunkCoordinate, ...]
    ) -> tuple[ChunkCoordinate, ...]:
        required = {
            ChunkCoordinate(x=coordinate.x + offset_x, y=coordinate.y + offset_z)
            for coordinate in wanted
            for offset_z in (-1, 0, 1)
            for offset_x in (-1, 0, 1)
        }
        centre_x = math.floor(self.player.x) // CHUNK_SIZE
        centre_z = math.floor(self.player.z) // CHUNK_SIZE
        return tuple(
            sorted(
                required,
                key=lambda item: (
                    abs(item.x - centre_x) + abs(item.y - centre_z),
                    item.y,
                    item.x,
                ),
            )
        )

    def _complete_stream_blocking(self) -> None:
        for _, future in self._mesh_futures.values():
            future.cancel()
        self._mesh_futures.clear()
        generation_started = time.perf_counter()
        for coordinate in self._required_terrain_chunks(self._wanted_chunks):
            self.runtime.get_or_generate(coordinate)
        self._generation_seconds += time.perf_counter() - generation_started
        self._terrain_queue.clear()
        if self.context is None or self.program is None:
            self._mesh_queue.clear()
            self._visible = ()
            self.loading = False
            return
        for coordinate in self._wanted_chunks:
            self.runtime.activate(coordinate)
            key = self._mesh_key(coordinate)
            cached = self._gpu_chunks.get(coordinate)
            if cached is not None and cached.key == key:
                continue
            mesh_started = time.perf_counter()
            mesh = self._capture_mesh_snapshot(coordinate).build()
            self._mesh_seconds += time.perf_counter() - mesh_started
            self._install_gpu_mesh(key=key, mesh=mesh)
        self._mesh_queue.clear()
        self._visible = self._wanted_chunks
        self.loading = False

    def _pump_streaming(self) -> None:
        self._collect_mesh_results()
        if self._terrain_queue:
            generation_started = time.perf_counter()
            coordinate = self._terrain_queue.popleft()
            self.runtime.get_or_generate(coordinate)
            self._generation_seconds += time.perf_counter() - generation_started
        for coordinate in self._wanted_chunks:
            if not self.runtime.contains(coordinate):
                continue
            state = self.runtime.metadata_at(coordinate).state
            if state in (ChunkState.READY, ChunkState.SUSPENDED):
                self.runtime.activate(coordinate)
        if self.context is None or self.program is None:
            self._mesh_queue.clear()
        elif not self._mesh_futures:
            self._submit_next_mesh()
        self._visible = tuple(
            coordinate for coordinate in self._wanted_chunks if coordinate in self._gpu_chunks
        )
        pending_mesh_count = len(self._mesh_queue) + len(self._mesh_futures)
        visible_mismatch = int(len(self._visible) != len(self._wanted_chunks))
        gpu_available = int(self.context is not None)
        gpu_backlog = gpu_available * (pending_mesh_count + visible_mismatch)
        self.loading = bool(len(self._terrain_queue) + gpu_backlog)

    def _submit_next_mesh(self) -> None:
        attempts = len(self._mesh_queue)
        for _ in range(attempts):
            coordinate = self._mesh_queue.popleft()
            if coordinate not in self._wanted_chunks:
                continue
            required = self._required_terrain_chunks((coordinate,))
            if any(not self.runtime.contains(item) for item in required):
                self._mesh_queue.append(coordinate)
                continue
            key = self._mesh_key(coordinate)
            cached = self._gpu_chunks.get(coordinate)
            if cached is not None and cached.key == key:
                continue
            snapshot = self._capture_mesh_snapshot(coordinate)
            future = self._mesh_executor.submit(snapshot.build)
            self._mesh_futures[coordinate] = (key, future)
            return

    def _collect_mesh_results(self) -> None:
        completed = tuple(
            coordinate for coordinate, (_, future) in self._mesh_futures.items() if future.done()
        )
        for coordinate in completed:
            key, future = self._mesh_futures.pop(coordinate)
            mesh_started = time.perf_counter()
            mesh = future.result()
            self._mesh_seconds += time.perf_counter() - mesh_started
            if coordinate not in self._wanted_chunks:
                continue
            if key != self._mesh_key(coordinate):
                self._mesh_queue.append(coordinate)
                continue
            self._install_gpu_mesh(key=key, mesh=mesh)

    def _mesh_key(
        self, coordinate: ChunkCoordinate
    ) -> tuple[ChunkCoordinate, int, tuple[int, int, int, int], str, int, int]:
        terrain = self.runtime.terrain_at(coordinate)
        return mesh_cache_key(
            terrain=terrain,
            neighbour_revisions=self._neighbour_revisions(coordinate),
            edit_revision=self._local_edit_revision(coordinate),
        )

    def _local_edit_revision(self, coordinate: ChunkCoordinate) -> int:
        chunks = tuple(
            ChunkCoordinate(x=coordinate.x + offset_x, y=coordinate.y + offset_z)
            for offset_z in (-1, 0, 1)
            for offset_x in (-1, 0, 1)
        )
        return max(
            (edit.revision for chunk in chunks for edit in self.edits.edits_for_chunk(chunk)),
            default=0,
        )

    def _capture_mesh_snapshot(self, coordinate: ChunkCoordinate) -> ChunkMeshSnapshot:
        terrain = self.runtime.terrain_at(coordinate)
        origin = coordinate.to_world_origin()
        minimum_x = origin.x - 1
        maximum_x = origin.x + CHUNK_SIZE
        minimum_z = origin.y - 1
        maximum_z = origin.y + CHUNK_SIZE
        columns = {
            (world_x, world_z): self._column_at(world_x, world_z)
            for world_z in range(minimum_z, maximum_z + 1)
            for world_x in range(minimum_x, maximum_x + 1)
        }
        edits = {
            edit.coordinate: edit.material
            for edit in self.edits.snapshot().edits
            if minimum_x <= edit.coordinate.x <= maximum_x
            and minimum_z <= edit.coordinate.z <= maximum_z
        }
        natural_blocks = (
            natural_blocks_in_area(
                minimum_x=minimum_x,
                maximum_x=maximum_x,
                minimum_z=minimum_z,
                maximum_z=maximum_z,
                column_at=self._column_at,
                terrain_seed_at=self._terrain_seed_at,
            )
            if edits
            else {}
        )
        return ChunkMeshSnapshot(
            terrain=terrain,
            columns=columns,
            edits=edits,
            natural_blocks=natural_blocks,
            editable=bool(edits),
        )

    def _install_gpu_mesh(
        self,
        *,
        key: tuple[ChunkCoordinate, int, tuple[int, int, int, int], str, int, int],
        mesh: VoxelChunkMesh,
    ) -> None:
        if self.context is None or self.program is None:
            return
        cached = self._gpu_chunks.pop(mesh.coordinate, None)
        if cached is not None:
            cached.release()
        opaque_buffer = self.context.buffer(mesh.opaque_vertices)
        opaque_array = self.context.vertex_array(
            self.program,
            [(opaque_buffer, "3f 2f 1f", "in_position", "in_uv", "in_shade")],
        )
        water_buffer = self.context.buffer(mesh.water_vertices) if mesh.water_vertex_count else None
        water_array = (
            self.context.vertex_array(
                self.program,
                [(water_buffer, "3f 2f 1f", "in_position", "in_uv", "in_shade")],
            )
            if water_buffer is not None
            else None
        )
        self._gpu_chunks[mesh.coordinate] = GpuChunk(
            key=key,
            opaque_buffer=opaque_buffer,
            opaque_array=opaque_array,
            water_buffer=water_buffer,
            water_array=water_array,
            mesh=mesh,
        )

    def _column_at(self, world_x: int, world_z: int) -> BlockColumn:
        coordinate = ChunkCoordinate(x=world_x // CHUNK_SIZE, y=world_z // CHUNK_SIZE)
        terrain = self.runtime.get_or_generate(coordinate)
        tile = terrain.tile_at(LocalTileCoordinate(x=world_x % CHUNK_SIZE, y=world_z % CHUNK_SIZE))
        return column_from_terrain(
            terrain_type=tile.terrain_type, elevation_metres=tile.elevation.metres
        )

    def _terrain_seed_at(self, world_x: int, world_z: int) -> int:
        coordinate = ChunkCoordinate(x=world_x // CHUNK_SIZE, y=world_z // CHUNK_SIZE)
        return self.runtime.get_or_generate(coordinate).terrain_seed

    def _natural_blocks_for_chunk(
        self, coordinate: ChunkCoordinate
    ) -> dict[WorldBlockCoordinate, BlockMaterial]:
        cached = self._natural_chunk_cache.get(coordinate)
        if cached is not None:
            return cached
        required = self._required_terrain_chunks((coordinate,))
        if any(not self.runtime.contains(item) for item in required):
            return {}
        origin = coordinate.to_world_origin()
        blocks = natural_blocks_in_area(
            minimum_x=origin.x,
            maximum_x=origin.x + CHUNK_SIZE - 1,
            minimum_z=origin.y,
            maximum_z=origin.y + CHUNK_SIZE - 1,
            column_at=self._column_at,
            terrain_seed_at=self._terrain_seed_at,
        )
        self._natural_chunk_cache[coordinate] = blocks
        return blocks

    def _natural_material_at(self, world_x: int, world_y: int, world_z: int) -> BlockMaterial:
        coordinate = WorldBlockCoordinate(x=world_x, y=world_y, z=world_z)
        return self._natural_blocks_for_chunk(coordinate.chunk_coordinate).get(
            coordinate, BlockMaterial.AIR
        )

    def _height_at(self, world_x: int, world_z: int) -> int:
        return self._column_at(world_x, world_z).ground_height

    def _solid_at(self, world_x: int, world_y: int, world_z: int) -> bool:
        return self.editable_world.solid_at(world_x, world_y, world_z)

    def _selected_placement_material(self) -> BlockMaterial | None:
        stack = self.inventory.selected_stack
        return None if stack is None else item_policy(stack.item).placeable_material

    def _refresh_interaction_previews(self) -> None:
        self.break_preview = self.interactions.preview_break(target=self.target)
        self.placement_preview = self.interactions.preview_place(
            target=self.target,
            material=self._selected_placement_material(),
            player=self.player,
        )

    def _interaction_prompt(self) -> str:
        if not self.mouse_captured:
            return "Click to capture the mouse and resume controls"
        if self.target is None:
            return "Aim at a block within reach"
        target_name = self.target.material.value.replace("_", " ").title()
        mining_prompt = (
            f"Hold LMB to mine {target_name}"
            if self.break_preview.allowed
            else f"Mining: {self.break_preview.result.value}"
        )
        selected = self.inventory.selected_slot
        if isinstance(selected, ItemStack):
            if self.placement_preview.allowed:
                return f"{mining_prompt}  |  RMB to place {selected.item.display_name}"
            return f"{mining_prompt}  |  Placement: {self.placement_preview.result.value}"
        if selected is not None and self.break_preview.allowed:
            return f"{mining_prompt} with {selected.item.display_name}"
        if selected is not None:
            return f"{mining_prompt}  |  Selected {selected.item.display_name}"
        return f"{mining_prompt}  |  Select a block to place"

    def _apply_interaction(self, outcome: InteractionOutcome) -> None:
        self.last_interaction = outcome.result
        if outcome.result is InteractionResult.PLACED:
            self.last_placement_consumption = "consumed 1 selected item"
        self.save_message = ""
        if not outcome.changed:
            if outcome.result not in (InteractionResult.NONE, InteractionResult.COOLDOWN):
                self.save_message = outcome.result.value.capitalize()
                self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
            self._refresh_interaction_previews()
            return
        if (
            outcome.result is InteractionResult.BROKEN
            and outcome.coordinate is not None
            and outcome.dropped_item is not None
        ):
            self.dropped_items.spawn(
                item=outcome.dropped_item,
                quantity=1,
                position=(
                    outcome.coordinate.x + 0.5,
                    outcome.coordinate.y + 0.5,
                    outcome.coordinate.z + 0.5,
                ),
            )
        self._feedback_coordinate = outcome.coordinate
        self._feedback_until = pygame.time.get_ticks() / 1000.0 + 0.22
        self.dirty = True
        self._stream_signature = None
        self._stream(blocking=False)
        self.target = ray_cast(
            origin=(self.player.x, self.player.y + 1.62, self.player.z),
            direction=self.camera.forward,
            block_at=self.editable_world.material_at,
            maximum_distance=self.config.interaction_reach,
        )
        self._refresh_interaction_previews()

    def _update_mining(self, microseconds: int) -> None:
        if not self._mining_held or not self.mouse_captured or self.target is None:
            if self.mining.snapshot.status is MiningStatus.ACTIVE:
                self.mining.cancel("target unavailable")
            return
        self.break_preview = self.interactions.preview_break(target=self.target)
        if not self.break_preview.allowed:
            if self.mining.snapshot.status is MiningStatus.ACTIVE:
                self.mining.cancel(self.break_preview.result.value)
            return
        target = self.target
        snapshot = self.mining.snapshot
        selected_tool = self.inventory.selected_tool
        if self.config.progression_enabled and target.material is BlockMaterial.STONE:
            classification = (
                None
                if selected_tool is None
                else item_policy(selected_tool.item).tool_classification
            )
            if classification is not ToolClassification.PICKAXE:
                self.mining.cancel("stone requires a pickaxe")
                self.save_message = "Craft and select a pickaxe to mine stone"
                self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
                return
        if (
            snapshot.status is not MiningStatus.ACTIVE
            or snapshot.target != target.coordinate
            or snapshot.target_material is not target.material
            or snapshot.selected_tool != selected_tool
        ):
            self.mining.begin(
                target=target.coordinate,
                material=target.material,
                tool=selected_tool,
            )
        snapshot = self.mining.advance(microseconds)
        if snapshot.status is not MiningStatus.COMPLETED:
            return
        outcome = self.interactions.break_block(
            target=target,
            now=pygame.time.get_ticks() / 1000.0,
        )
        if outcome.result is InteractionResult.BROKEN:
            self._apply_interaction(outcome)
            if selected_tool is not None:
                self.inventory.use_tool(self.inventory.selected_hotbar_index)
                if self.inventory.selected_tool is None:
                    self.save_message = f"{selected_tool.item.display_name} broke"
                    self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.5
        self.mining.reset()

    def _respawn_after_death(self) -> None:
        self._mining_held = False
        self.mining.cancel("player died")
        self.vitals.respawn()
        self.player = PlayerState(
            x=float(self.spawn_x) + 0.5,
            y=safe_spawn_height(
                world_x=self.spawn_x,
                world_z=self.spawn_z,
                height_at=self._height_at,
            ),
            z=float(self.spawn_z) + 0.5,
            grounded=True,
        )
        self.target = None
        self._refresh_interaction_previews()
        self.dirty = True
        self.save_message = "You died \N{EM DASH} respawned"
        self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.5

    def _save_edits(self) -> bool:
        if self._save_service is None or self._save_slot is None:
            self.save_message = "Save failed"
            self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
            return False
        try:
            self._save_service.save(
                slot=self._save_slot,
                block_edits=self.edits.snapshot(),
                inventory=self.inventory.snapshot(),
                dropped_items=self.dropped_items.snapshot(),
                vitals=self.vitals.snapshot,
                progression=(
                    self.progression.snapshot if self.config.progression_enabled else None
                ),
            )
        except Exception:
            self.save_message = "Save failed"
            self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
            return False
        self.dirty = False
        self.flow.set_continue_available(True)
        self.save_message = "World saved"
        self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
        return True

    def _load_edits(self) -> bool:
        if self._save_service is None or self._save_slot is None:
            self.save_message = "Load failed"
            self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
            return False
        try:
            load_result = self._save_service.load_with_status(self._save_slot)
            document = load_result.document
            restored = self._save_service.restore_block_edits(
                document,
                expected_world_id=self.world_id,
                expected_world_seed=self.config.world_seed,
            )
            restored_inventory, restored_drops = self._save_service.restore_resources(
                document,
                expected_world_id=self.world_id,
                expected_world_seed=self.config.world_seed,
                legacy_inventory=create_bootstrap_inventory(
                    enabled=(
                        self.config.bootstrap_inventory and not self.config.progression_enabled
                    )
                ).snapshot(),
            )
            restored_vitals = self._save_service.restore_vitals(
                document,
                expected_world_id=self.world_id,
                expected_world_seed=self.config.world_seed,
            )
            restored_progression = (
                self._save_service.restore_progression(
                    document,
                    expected_world_id=self.world_id,
                    expected_world_seed=self.config.world_seed,
                    legacy_inventory=restored_inventory,
                )
                if self.config.progression_enabled
                else SurvivalProgression()
            )
        except Exception:
            self.save_message = "Load failed"
            self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
            return False
        previous = {edit.coordinate: edit.material for edit in self.edits.snapshot().edits}
        replacement = {edit.coordinate: edit.material for edit in restored.snapshot().edits}
        changed = {
            coordinate
            for coordinate in previous.keys() | replacement.keys()
            if previous.get(coordinate) is not replacement.get(coordinate)
        }
        self.edits = restored
        self.inventory = restored_inventory
        self.dropped_items = restored_drops
        self.vitals = restored_vitals
        self.progression = restored_progression
        self._mining_held = False
        self.mining.reset()
        self._drop_render_revision = -1
        self._natural_chunk_cache.clear()
        self.editable_world = EditableVoxelWorld(
            column_at=self._column_at,
            edits=self.edits,
            natural_material_at=self._natural_material_at,
        )
        self.interactions = VoxelInteractionController(
            world=self.editable_world,
            edits=self.edits,
            break_cooldown=self.config.break_cooldown,
            placement_cooldown=self.config.placement_cooldown,
            maximum_reach=self.config.interaction_reach,
        )
        affected = {
            chunk for coordinate in changed for chunk in invalidated_chunks_for_edit(coordinate)
        }
        if affected:
            self._stream_signature = None
        if self._player_intersects_world():
            self.player = PlayerState(
                x=float(self.spawn_x) + 0.5,
                y=safe_spawn_height(
                    world_x=self.spawn_x,
                    world_z=self.spawn_z,
                    height_at=self._height_at,
                ),
                z=float(self.spawn_z) + 0.5,
                grounded=True,
            )
        self._stream_signature = None
        self._stream(blocking=False)
        self.target = ray_cast(
            origin=(self.player.x, self.player.y + 1.62, self.player.z),
            direction=self.camera.forward,
            block_at=self.editable_world.material_at,
            maximum_distance=self.config.interaction_reach,
        )
        self._refresh_interaction_previews()
        self.dirty = False
        self.flow.set_continue_available(True)
        self.save_message = (
            "World recovered from backup" if load_result.recovered_from_backup else "World loaded"
        )
        self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
        return True

    def _player_intersects_world(self) -> bool:
        for x in range(math.floor(self.player.x - 0.3), math.floor(self.player.x + 0.3) + 1):
            for y in range(math.floor(self.player.y), math.floor(self.player.y + 1.8) + 1):
                for z in range(
                    math.floor(self.player.z - 0.3),
                    math.floor(self.player.z + 0.3) + 1,
                ):
                    coordinate = WorldBlockCoordinate(x=x, y=y, z=z)
                    if self.editable_world.block_at(
                        coordinate
                    ).is_solid and player_intersects_block(
                        player=self.player,
                        coordinate=coordinate,
                    ):
                        return True
        return False

    def _has_scenery(self, world_x: int, world_z: int) -> bool:
        coordinate = ChunkCoordinate(
            x=world_x // CHUNK_SIZE,
            y=world_z // CHUNK_SIZE,
        )
        terrain = self.runtime.get_or_generate(coordinate)
        column = self._column_at(world_x, world_z)
        neighbours = (
            self._column_at(world_x - 1, world_z),
            self._column_at(world_x + 1, world_z),
            self._column_at(world_x, world_z - 1),
            self._column_at(world_x, world_z + 1),
        )
        slope = max(abs(item.ground_height - column.ground_height) for item in neighbours)
        return (
            scenery_at(
                seed=terrain.terrain_seed,
                world_x=world_x,
                world_z=world_z,
                column=column,
                slope=slope,
            )
            is not None
        )

    def _neighbour_revisions(self, coordinate: ChunkCoordinate) -> tuple[int, int, int, int]:
        neighbours = (
            ChunkCoordinate(x=coordinate.x - 1, y=coordinate.y),
            ChunkCoordinate(x=coordinate.x + 1, y=coordinate.y),
            ChunkCoordinate(x=coordinate.x, y=coordinate.y - 1),
            ChunkCoordinate(x=coordinate.x, y=coordinate.y + 1),
        )
        return tuple(self.runtime.get_or_generate(item).revision for item in neighbours)  # type: ignore[return-value]

    def _capture_mouse(self, captured: bool) -> None:
        self.mouse_captured = captured
        if not captured:
            self._mining_held = False
            self.mining.cancel("mouse capture lost")
        pygame.event.set_grab(captured)
        pygame.mouse.set_visible(not captured)
        if captured:
            pygame.mouse.get_rel()

    def _render_target_outline(
        self,
        target: RayHit,
        *,
        colour: tuple[float, float, float, float] = (1.0, 0.93, 0.32, 1.0),
    ) -> None:
        if self._target_buffer is None or self._target_array is None:
            return
        epsilon = 0.003
        x0, y0, z0 = target.x - epsilon, target.y - epsilon, target.z - epsilon
        x1, y1, z1 = target.x + 1 + epsilon, target.y + 1 + epsilon, target.z + 1 + epsilon
        corners = (
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        )
        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        data = array("f")
        for start, end in edges:
            data.extend((*corners[start], *colour))
            data.extend((*corners[end], *colour))
        self._target_buffer.write(data.tobytes())
        self._target_array.render(moderngl.LINES)

    def _caption(self, triangles: int) -> str:
        status = " | loading" if self.loading else ""
        mode = "FLY" if self.player.flying else "WALK"
        input_state = "PLAY" if self.mouse_captured else "CURSOR"
        timing = self.frame_timing.snapshot
        basic = f"Open World RPG Voxel | {self.fps:4.0f} FPS | {mode} | {input_state}{status}"
        if self.show_help:
            basic += " | F1 controls | Esc releases mouse"
        if not self.show_debug:
            return basic
        chunk = ChunkCoordinate(
            x=math.floor(self.player.x) // CHUNK_SIZE,
            y=math.floor(self.player.z) // CHUNK_SIZE,
        )
        target = (
            "none" if self.target is None else (f"{self.target.x},{self.target.y},{self.target.z}")
        )
        return (
            f"{basic} | xyz {self.player.x:.1f},{self.player.y:.1f},{self.player.z:.1f}"
            f" | chunk {chunk.x},{chunk.y} | cached {len(self.runtime.coordinates())}"
            f" | meshes {len(self._gpu_chunks)} | triangles {triangles}"
            f" | 1% low {timing.one_percent_low_fps:.0f}"
            f" | worst {timing.worst_frame_ms:.0f}ms"
            f" | target {target}"
        )

    def _render_hud(self, triangles: int) -> None:
        if (
            self.context is None
            or self._hud_texture is None
            or self._hud_array is None
            or self._font is None
        ):
            return
        now = pygame.time.get_ticks() / 1000.0
        if (
            self.hud_snapshot is not None
            and not self.flow.overlay_active
            and now < self._next_hud_refresh
        ):
            self.context.disable(moderngl.DEPTH_TEST)
            self._hud_texture.use(location=1)
            self._hud_array.render(moderngl.TRIANGLES)
            self.context.enable(moderngl.DEPTH_TEST)
            return
        self._next_hud_refresh = now + HUD_REFRESH_INTERVAL_SECONDS
        active = sum(
            self.runtime.metadata_at(coordinate).state is ChunkState.ACTIVE
            for coordinate in self.runtime.coordinates()
        )
        self.hud_snapshot = VoxelHudSnapshot.create(
            fps=self.fps,
            player=self.player,
            seed=self.config.world_seed,
            active_chunks=active,
            cached_chunks=len(self.runtime.coordinates()),
            mesh_count=len(self._gpu_chunks),
            triangles=triangles,
            render_distance=self.render_distance,
            target=self.target,
            loading=self.loading,
            selected_material=(
                None
                if self.inventory.selected_stack is None
                else item_policy(self.inventory.selected_stack.item).placeable_material
            ),
            edit_revision=self.edits.revision,
            edited_block_count=len(self.edits),
            last_interaction=self.last_interaction.value,
            save_path=None if self.save_path is None else str(self.save_path),
            dirty=self.dirty,
            selected_item=(
                None
                if self.inventory.selected_slot is None
                else self.inventory.selected_slot.item.value
            ),
            selected_quantity=(
                0
                if self.inventory.selected_stack is None
                else self.inventory.selected_stack.quantity
            ),
            inventory_revision=self.inventory.revision,
            occupied_slots=self.inventory.occupied_slots,
            total_inventory_items=self.inventory.total_items,
            active_dropped_items=len(self.dropped_items),
            nearest_drop_distance=self.dropped_items.nearest_distance(
                (self.player.x, self.player.y + 0.9, self.player.z)
            ),
            last_pickup=self.last_pickup,
            last_placement_consumption=self.last_placement_consumption,
            selected_slot_kind=(
                "empty"
                if self.inventory.selected_slot is None
                else ("stack" if isinstance(self.inventory.selected_slot, ItemStack) else "tool")
            ),
            tool_durability=(
                None
                if self.inventory.selected_tool is None
                else (
                    self.inventory.selected_tool.current_durability,
                    self.inventory.selected_tool.maximum_durability,
                )
            ),
            mining_progress=self.mining.snapshot.normalised_progress,
            health=self.vitals.snapshot.health,
            stamina=self.vitals.snapshot.stamina,
            fall_distance=self.vitals.snapshot.accumulated_fall_milli / 1_000,
            last_fall_damage=self.vitals.snapshot.last_fall_damage,
            death_count=self.vitals.snapshot.death_count,
            vitals_revision=self.vitals.snapshot.revision,
            mouse_captured=self.mouse_captured,
            break_preview=self.break_preview.result.value,
            placement_preview=self.placement_preview.result.value,
            placement_target=(
                None
                if self.placement_preview.coordinate is None
                else (
                    self.placement_preview.coordinate.x,
                    self.placement_preview.coordinate.y,
                    self.placement_preview.coordinate.z,
                )
            ),
            interaction_prompt=self._interaction_prompt(),
        )
        hud = self.hud_snapshot
        lines = [f"{hud.fps:4.0f} FPS"]
        if hud.loading:
            lines.append("Generating terrain...")
        if self.show_debug:
            lines.extend(
                (
                    f"XYZ {hud.position[0]:.1f} {hud.position[1]:.1f} {hud.position[2]:.1f}",
                    f"Block {hud.block}  Chunk {hud.chunk.x},{hud.chunk.y}",
                    f"Region {hud.region.x},{hud.region.y}  Seed {hud.seed}",
                    f"Active/cached {hud.active_chunks}/{hud.cached_chunks}",
                    f"Meshes {hud.mesh_count}  Triangles {hud.triangles}",
                    f"Mode {hud.mode}  Radius {hud.render_distance}",
                    (
                        f"Frame avg {self.frame_timing.snapshot.average_fps:.1f} FPS  "
                        f"1% low {self.frame_timing.snapshot.one_percent_low_fps:.1f} FPS"
                    ),
                    (
                        f"Frame p95 {self.frame_timing.snapshot.p95_frame_ms:.1f} ms  "
                        f"worst {self.frame_timing.snapshot.worst_frame_ms:.1f} ms  "
                        f"stalls {self.frame_timing.snapshot.stall_count}/"
                        f"{self.frame_timing.snapshot.severe_stall_count}"
                    ),
                    (
                        f"Stream terrain {len(self._terrain_queue)}  "
                        f"meshes {len(self._mesh_queue)}  jobs {len(self._mesh_futures)}"
                    ),
                    f"Selected {hud.selected_material.value if hud.selected_material else 'empty'}",
                    f"Target {hud.target or 'none'} "
                    f"{hud.target_material.value if hud.target_material else ''} "
                    f"face {hud.target_face or 'none'} "
                    f"distance {hud.target_distance:.2f}"
                    if hud.target_distance is not None
                    else "Target none",
                    f"Preview mine={hud.break_preview} place={hud.placement_preview}",
                    f"Edits {hud.edited_block_count}  Revision {hud.edit_revision}",
                    f"Interaction {hud.last_interaction}",
                    f"Inventory {hud.total_inventory_items} items / {hud.occupied_slots} slots "
                    f"(rev {hud.inventory_revision})",
                    f"Drops {hud.active_dropped_items}  nearest {hud.nearest_drop_distance:.2f}"
                    if hud.nearest_drop_distance is not None
                    else f"Drops {hud.active_dropped_items}  nearest none",
                    f"Pickup {hud.last_pickup}",
                    f"Placement {hud.last_placement_consumption}",
                    f"Vitals H {hud.health:.1f} S {hud.stamina:.1f} (rev {hud.vitals_revision})",
                    f"Mining {hud.mining_progress * 100:.0f}%  "
                    f"Fall {hud.fall_distance:.2f}/{hud.last_fall_damage}",
                    f"Deaths {hud.death_count}  Slot {hud.selected_slot_kind}",
                    f"Save {hud.save_path or 'disabled'} ({'modified' if hud.dirty else 'clean'})",
                )
            )
        surface = pygame.Surface((1024, 512), pygame.SRCALPHA)
        surface.fill((8, 13, 20, 175), pygame.Rect(0, 0, 500, len(lines) * 22 + 12))
        pygame.draw.rect(surface, (48, 24, 24, 220), pygame.Rect(18, 445, 200, 14))
        pygame.draw.rect(
            surface,
            (190, 52, 52, 235),
            pygame.Rect(18, 445, round(200 * hud.health / 100), 14),
        )
        pygame.draw.rect(surface, (26, 40, 52, 220), pygame.Rect(18, 464, 200, 12))
        pygame.draw.rect(
            surface,
            (60, 155, 210, 235),
            pygame.Rect(18, 464, round(200 * hud.stamina / 100), 12),
        )
        if hud.mining_progress:
            pygame.draw.rect(surface, (20, 20, 20, 220), pygame.Rect(412, 300, 200, 12))
            pygame.draw.rect(
                surface,
                (238, 190, 62, 240),
                pygame.Rect(412, 300, round(200 * hud.mining_progress), 12),
            )
        for index, line in enumerate(lines):
            surface.blit(
                self._font.render(line, True, (235, 240, 235)),
                (8, 6 + index * 22),
            )
        self._draw_hotbar(surface)
        if self.config.progression_enabled and self.flow.screen is VoxelScreen.PLAYING:
            self._draw_objective_panel(surface)
        if self.config.game_flow_enabled and self.flow.overlay_active:
            self._draw_flow_overlay(surface)
        else:
            self._draw_interaction_prompt(surface, hud.interaction_prompt)
            if self.show_help:
                self._draw_help_panel(surface)
            if not hud.mouse_captured:
                self._draw_capture_prompt(surface)
        self._hud_texture.write(pygame.image.tobytes(surface, "RGBA", True))
        self.context.disable(moderngl.DEPTH_TEST)
        self._hud_texture.use(location=1)
        self._hud_array.render(moderngl.TRIANGLES)
        self.context.enable(moderngl.DEPTH_TEST)

    def _draw_flow_overlay(self, surface: pygame.Surface) -> None:
        if self._font is None:
            return
        pygame.draw.rect(surface, (5, 9, 15, 225), surface.get_rect())
        if self.flow.screen is VoxelScreen.INVENTORY:
            self._draw_inventory_screen(surface)
        elif self.flow.screen is VoxelScreen.GUIDE:
            self._draw_guide_screen(surface)
        else:
            self._draw_menu_screen(surface)

    def _draw_guide_screen(self, surface: pygame.Surface) -> None:
        if self._font is None:
            return
        font = self._font
        title_text, body_text = self.progression.guide_page
        panel = pygame.Rect(166, 116, 692, 280)
        pygame.draw.rect(surface, (12, 22, 31, 245), panel, border_radius=10)
        pygame.draw.rect(surface, (255, 220, 112), panel, 2, border_radius=10)
        title = font.render(title_text, True, (255, 236, 160))
        surface.blit(title, (panel.centerx - title.get_width() // 2, 152))
        words = body_text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= 600:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        for index, line in enumerate(lines):
            rendered = font.render(line, True, (215, 228, 235))
            surface.blit(rendered, (panel.centerx - rendered.get_width() // 2, 218 + index * 30))
        page = font.render(
            f"{self.progression.guide_page_index + 1}/{len(GUIDE_PAGES)}",
            True,
            (150, 175, 190),
        )
        hint = font.render(
            "Enter / Space / click to continue   Escape to skip",
            True,
            (170, 195, 210),
        )
        surface.blit(page, (panel.centerx - page.get_width() // 2, 326))
        surface.blit(hint, (panel.centerx - hint.get_width() // 2, 356))

    def _draw_objective_panel(self, surface: pygame.Surface) -> None:
        if self._font is None:
            return
        objective = self.progression.objective(self.inventory)
        panel = pygame.Rect(636, 18, 368, 92)
        pygame.draw.rect(surface, (8, 15, 22, 220), panel, border_radius=7)
        pygame.draw.rect(
            surface,
            (112, 205, 128) if self.progression.completed else (238, 190, 62),
            panel,
            2,
            border_radius=7,
        )
        title = self._font.render(objective.title.upper(), True, (255, 236, 160))
        instruction = self._font.render(objective.instruction, True, (205, 220, 228))
        progress = self._font.render(objective.progress, True, (145, 205, 235))
        surface.blit(title, (650, 28))
        surface.blit(instruction, (650, 56))
        surface.blit(progress, (650, 82))

    def _draw_menu_screen(self, surface: pygame.Surface) -> None:
        if self._font is None:
            return
        font = self._font
        title_text = {
            VoxelScreen.MAIN_MENU: "OPEN WORLD RPG",
            VoxelScreen.PAUSED: "PAUSED",
            VoxelScreen.DEAD: "YOU DIED",
            VoxelScreen.COMPLETED: "STONE AGE REACHED",
        }.get(self.flow.screen, "OPEN WORLD RPG")
        subtitle_text = {
            VoxelScreen.MAIN_MENU: "v0.9.0 playable survival loop",
            VoxelScreen.PAUSED: "The world is paused",
            VoxelScreen.DEAD: f"Deaths: {self.vitals.snapshot.death_count}",
            VoxelScreen.COMPLETED: "The v0.9.0 survival objective is complete",
        }.get(self.flow.screen, "")
        title = font.render(title_text, True, (255, 236, 160))
        subtitle = font.render(subtitle_text, True, (185, 205, 218))
        surface.blit(title, (512 - title.get_width() // 2, 116))
        surface.blit(subtitle, (512 - subtitle.get_width() // 2, 154))
        for index, option in enumerate(self.flow.options):
            rect = pygame.Rect(340, 220 + index * 54, 344, 42)
            selected = index == self.flow.selected_index
            pygame.draw.rect(
                surface,
                (64, 80, 92, 235) if selected else (18, 27, 36, 225),
                rect,
                border_radius=6,
            )
            pygame.draw.rect(
                surface,
                (255, 226, 120) if selected else (91, 112, 126),
                rect,
                2,
                border_radius=6,
            )
            colour = (245, 245, 235) if option.enabled else (112, 120, 126)
            label = font.render(option.label, True, colour)
            surface.blit(
                label,
                (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2),
            )
        hint = font.render("Arrow keys / mouse to select   Enter to confirm", True, (160, 180, 192))
        surface.blit(hint, (512 - hint.get_width() // 2, 458))
        if self.save_message:
            feedback = font.render(self.save_message, True, (255, 205, 125))
            surface.blit(feedback, (512 - feedback.get_width() // 2, 414))

    def _draw_inventory_screen(self, surface: pygame.Surface) -> None:
        if self._font is None:
            return
        font = self._font
        title = font.render("INVENTORY & CRAFTING", True, (255, 236, 160))
        surface.blit(title, (76, 70))
        instructions = font.render(
            "Arrows: move  Enter: select/move  Q: quick move  C: craft  E/Esc: close",
            True,
            (170, 195, 210),
        )
        surface.blit(instructions, (76, 98))
        atlas = self._inventory_atlas_surface
        if atlas is None:
            atlas = pygame.image.frombytes(
                generate_texture_atlas(),
                (ATLAS_SIZE, ATLAS_SIZE),
                "RGBA",
            )
            self._inventory_atlas_surface = atlas
        start_x, start_y, size, gap = 76, 138, 48, 7
        for index, slot in enumerate(self.inventory.slots()):
            row, column = divmod(index, 9)
            rect = pygame.Rect(
                start_x + column * (size + gap),
                start_y + row * (size + gap),
                size,
                size,
            )
            selected = index == self.inventory_screen.selected_slot_index
            source = index == self.inventory_screen.source_slot_index
            fill_colour = (
                (80, 67, 32, 240)
                if source
                else ((55, 70, 82, 235) if selected else (22, 30, 38, 225))
            )
            pygame.draw.rect(surface, fill_colour, rect)
            pygame.draw.rect(
                surface,
                (255, 230, 132) if selected or source else (90, 107, 118),
                rect,
                2,
            )
            self._draw_inventory_value(surface, atlas, rect, slot)
            number = font.render(str(index + 1), True, (150, 165, 175))
            surface.blit(number, (rect.x + 3, rect.y + 2))
        hotbar_label = font.render("HOTBAR", True, (135, 205, 245))
        backpack_label = font.render("BACKPACK", True, (135, 205, 245))
        surface.blit(hotbar_label, (76, 302))
        surface.blit(backpack_label, (76, 324))

        craft_panel = pygame.Rect(616, 70, 370, 360)
        pygame.draw.rect(surface, (12, 20, 28, 235), craft_panel, border_radius=8)
        pygame.draw.rect(surface, (88, 108, 122), craft_panel, 2, border_radius=8)
        craft_title = font.render("RECIPES", True, (255, 236, 160))
        surface.blit(craft_title, (634, 84))
        for index, recipe in enumerate(self.inventory_screen.recipes):
            y = 116 + index * 46
            selected = index == self.inventory_screen.selected_recipe_index
            unlocked = not self.config.progression_enabled or self.progression.recipe_unlocked(
                recipe.identifier
            )
            can_craft = unlocked and self.inventory_screen.crafting.can_craft(
                self.inventory, recipe
            )
            rect = pygame.Rect(626, y, 344, 38)
            pygame.draw.rect(
                surface,
                (58, 73, 83, 235) if selected else (20, 29, 36, 220),
                rect,
                border_radius=5,
            )
            pygame.draw.rect(
                surface,
                (255, 220, 112) if selected else (72, 90, 101),
                rect,
                2 if selected else 1,
                border_radius=5,
            )
            colour = (235, 245, 235) if can_craft else (140, 145, 148)
            label = font.render(recipe.output_label, True, colour)
            surface.blit(label, (rect.x + 8, rect.y + 4))
            ingredients = ", ".join(
                f"{ingredient.item.display_name} x{ingredient.quantity}"
                for ingredient in recipe.ingredients
            )
            detail = pygame.font.Font(None, 16).render(ingredients, True, (160, 180, 190))
            surface.blit(detail, (rect.x + 8, rect.y + 22))
        selected_recipe = self.inventory_screen.selected_recipe
        selected_unlocked = not self.config.progression_enabled or self.progression.recipe_unlocked(
            selected_recipe.identifier
        )
        craft_hint_text = (
            f"C / click: craft {selected_recipe.output_label}"
            if selected_unlocked
            else "Locked: craft a wooden pickaxe first"
        )
        craft_hint = font.render(
            craft_hint_text,
            True,
            (255, 226, 140),
        )
        surface.blit(craft_hint, (626, 402))
        if self.config.progression_enabled:
            objective = self.progression.objective(self.inventory)
            objective_text = font.render(
                f"Objective: {objective.title} — {objective.progress}",
                True,
                (150, 215, 245),
            )
            surface.blit(objective_text, (76, 360))
        if self.save_message:
            feedback = font.render(self.save_message, True, (255, 205, 125))
            surface.blit(feedback, (76, 390))

    def _draw_inventory_value(
        self,
        surface: pygame.Surface,
        atlas: pygame.Surface,
        rect: pygame.Rect,
        slot: ItemStack | ToolInstance | None,
    ) -> None:
        if self._font is None or slot is None:
            return
        font = self._font
        if isinstance(slot, ItemStack):
            material = item_policy(slot.item).placeable_material
            if material is not None:
                icon_for = {
                    BlockMaterial.GRASS: FaceTexture.GRASS_TOP,
                    BlockMaterial.DIRT: FaceTexture.DIRT,
                    BlockMaterial.STONE: FaceTexture.STONE,
                    BlockMaterial.SAND: FaceTexture.SAND,
                    BlockMaterial.SNOW: FaceTexture.SNOW_TOP,
                }
                texture = icon_for[material]
                atlas_index = tuple(FaceTexture).index(texture)
                source = pygame.Rect(
                    atlas_index % ATLAS_COLUMNS * ATLAS_TILE_SIZE,
                    atlas_index // ATLAS_COLUMNS * ATLAS_TILE_SIZE,
                    ATLAS_TILE_SIZE,
                    ATLAS_TILE_SIZE,
                )
                icon = pygame.transform.scale(atlas.subsurface(source), (32, 32))
                surface.blit(icon, (rect.x + 8, rect.y + 8))
            else:
                resource_colour = {
                    ItemType.WOOD_LOG: (126, 82, 46),
                    ItemType.WOOD_PLANK: (186, 136, 76),
                    ItemType.STICK: (205, 166, 105),
                }.get(slot.item, (150, 150, 150))
                pygame.draw.rect(surface, resource_colour, rect.inflate(-20, -20), border_radius=4)
            quantity = font.render(str(slot.quantity), True, (255, 255, 255))
            surface.blit(
                quantity,
                (rect.right - quantity.get_width() - 3, rect.bottom - quantity.get_height()),
            )
            return
        item = slot
        colour = (166, 119, 69) if "wooden" in item.item.value else (155, 160, 166)
        pygame.draw.line(
            surface,
            colour,
            (rect.x + 13, rect.bottom - 10),
            (rect.right - 12, rect.y + 10),
            5,
        )
        durability = font.render(
            f"{item.current_durability}/{item.maximum_durability}",
            True,
            (255, 110, 90)
            if item.current_durability * 4 <= item.maximum_durability
            else (255, 255, 255),
        )
        surface.blit(
            durability,
            (rect.right - durability.get_width() - 3, rect.bottom - durability.get_height()),
        )

    def _draw_help_panel(self, surface: pygame.Surface) -> None:
        if self._font is None:
            return
        font = self._font
        panel = pygame.Rect(706, 18, 300, 286)
        pygame.draw.rect(surface, (8, 13, 20, 205), panel, border_radius=8)
        pygame.draw.rect(surface, (98, 116, 128, 220), panel, 2, border_radius=8)
        title = font.render("CONTROLS", True, (255, 236, 160))
        surface.blit(title, (panel.x + 12, panel.y + 10))
        for index, hint in enumerate(DEFAULT_CONTROL_HINTS):
            y = panel.y + 38 + index * 22
            binding = font.render(hint.binding, True, (130, 207, 255))
            action = font.render(hint.action, True, (235, 240, 235))
            surface.blit(binding, (panel.x + 12, y))
            surface.blit(action, (panel.x + 132, y))

    def _draw_interaction_prompt(self, surface: pygame.Surface, prompt: str) -> None:
        if self._font is None or not prompt:
            return
        text = self._font.render(prompt, True, (245, 245, 235))
        padding = 8
        rect = pygame.Rect(
            surface.get_width() // 2 - text.get_width() // 2 - padding,
            326,
            text.get_width() + padding * 2,
            text.get_height() + padding,
        )
        pygame.draw.rect(surface, (8, 13, 20, 195), rect, border_radius=6)
        surface.blit(text, (rect.x + padding, rect.y + 4))

    def _draw_capture_prompt(self, surface: pygame.Surface) -> None:
        if self._font is None:
            return
        title = self._font.render("CLICK TO RESUME", True, (255, 236, 160))
        detail = self._font.render(
            "Gameplay input is paused while the cursor is free",
            True,
            (235, 240, 235),
        )
        panel = pygame.Rect(288, 190, 448, 78)
        pygame.draw.rect(surface, (8, 13, 20, 225), panel, border_radius=10)
        pygame.draw.rect(surface, (255, 236, 160, 230), panel, 2, border_radius=10)
        surface.blit(title, (512 - title.get_width() // 2, panel.y + 12))
        surface.blit(detail, (512 - detail.get_width() // 2, panel.y + 42))

    def _draw_hotbar(self, surface: pygame.Surface) -> None:
        if self._font is None:
            return
        font = self._font
        atlas = pygame.image.frombytes(
            generate_texture_atlas(),
            (ATLAS_SIZE, ATLAS_SIZE),
            "RGBA",
        )
        icon_for = {
            BlockMaterial.GRASS: FaceTexture.GRASS_TOP,
            BlockMaterial.DIRT: FaceTexture.DIRT,
            BlockMaterial.STONE: FaceTexture.STONE,
            BlockMaterial.SAND: FaceTexture.SAND,
            BlockMaterial.SNOW: FaceTexture.SNOW_TOP,
        }
        slot_size = 46
        gap = 4
        total_width = 9 * slot_size + 8 * gap
        start_x = (surface.get_width() - total_width) // 2
        y = surface.get_height() - slot_size - 18
        for index, slot in enumerate(self.inventory.slots()[:9]):
            rect = pygame.Rect(start_x + index * (slot_size + gap), y, slot_size, slot_size)
            is_selected = index == self.inventory.selected_hotbar_index
            pygame.draw.rect(
                surface,
                (232, 204, 92, 235) if is_selected else (24, 29, 34, 215),
                rect,
            )
            pygame.draw.rect(
                surface,
                (255, 244, 174) if is_selected else (122, 132, 137),
                rect,
                3,
            )
            if slot is not None:
                if isinstance(slot, ItemStack):
                    material = item_policy(slot.item).placeable_material
                    if material is not None:
                        texture = icon_for[material]
                        atlas_index = tuple(FaceTexture).index(texture)
                        source = pygame.Rect(
                            atlas_index % ATLAS_COLUMNS * ATLAS_TILE_SIZE,
                            atlas_index // ATLAS_COLUMNS * ATLAS_TILE_SIZE,
                            ATLAS_TILE_SIZE,
                            ATLAS_TILE_SIZE,
                        )
                        icon = pygame.transform.scale(atlas.subsurface(source), (34, 34))
                        surface.blit(icon, (rect.x + 6, rect.y + 6))
                    else:
                        resource_colour = {
                            ItemType.WOOD_LOG: (126, 82, 46),
                            ItemType.WOOD_PLANK: (186, 136, 76),
                            ItemType.STICK: (205, 166, 105),
                        }.get(slot.item, (150, 150, 150))
                        pygame.draw.rect(surface, resource_colour, rect.inflate(-18, -18))
                    quantity = font.render(str(slot.quantity), True, (255, 255, 255))
                    surface.blit(
                        quantity,
                        (
                            rect.right - quantity.get_width() - 3,
                            rect.bottom - quantity.get_height(),
                        ),
                    )
                else:
                    colour = (166, 119, 69) if "wooden" in slot.item.value else (155, 160, 166)
                    pygame.draw.line(
                        surface,
                        colour,
                        (rect.x + 13, rect.bottom - 10),
                        (rect.right - 12, rect.y + 10),
                        5,
                    )
                    pygame.draw.line(
                        surface,
                        (210, 210, 205),
                        (rect.right - 21, rect.y + 10),
                        (rect.right - 7, rect.y + 18),
                        5,
                    )
                    durability = font.render(
                        str(slot.current_durability),
                        True,
                        (255, 110, 90)
                        if slot.current_durability * 4 <= slot.maximum_durability
                        else (255, 255, 255),
                    )
                    surface.blit(
                        durability,
                        (
                            rect.right - durability.get_width() - 3,
                            rect.bottom - durability.get_height(),
                        ),
                    )
            number = font.render(str(index + 1), True, (245, 245, 245))
            surface.blit(number, (rect.x + 3, rect.y + 1))
        now = pygame.time.get_ticks() / 1000.0
        selected_slot = self.inventory.selected_slot
        if selected_slot is not None and now - self._selection_changed_at < 1.4:
            label = font.render(selected_slot.item.display_name, True, (255, 245, 190))
            surface.blit(label, (surface.get_width() // 2 - label.get_width() // 2, y - 24))
        if now < self._feedback_until:
            message = self.save_message or self.last_interaction.value
            feedback = font.render(message, True, (255, 218, 145))
            surface.blit(
                feedback,
                (surface.get_width() // 2 - feedback.get_width() // 2, y - 46),
            )
