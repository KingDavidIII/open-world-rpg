"""Pygame/ModernGL first-person voxel prototype controller."""

from __future__ import annotations

import logging
import math
import struct
import time
from array import array
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import moderngl
import pygame

from open_world_rpg.application import GameMode, RuntimeContext, create_terrain_runtime
from open_world_rpg.application.save_service import GameSaveService
from open_world_rpg.core import ProjectPaths
from open_world_rpg.gameplay import (
    DroppedItemManager,
    ItemStack,
    ItemType,
    MiningStatus,
    PlayerVitals,
    TimedMiningController,
    create_bootstrap_inventory,
    material_for_item,
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
from .editable_world import EditableVoxelWorld
from .hotbar import VoxelHotbar
from .hud import VoxelHudSnapshot
from .interaction import (
    InteractionOutcome,
    InteractionResult,
    VoxelInteractionController,
    invalidated_chunks_for_edit,
)
from .item_rendering import build_dropped_item_vertices
from .meshing import VoxelChunkMesh, build_chunk_mesh, mesh_cache_key
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


@dataclass(frozen=True, slots=True, kw_only=True)
class VoxelPrototypeConfig:
    """Conservative desktop defaults for synchronous voxel streaming."""

    width_pixels: int = 1280
    height_pixels: int = 720
    target_fps: int = 60
    render_distance: int = 1
    world_seed: int = 0
    hidden_window: bool = False
    interaction_reach: float = 5.5
    break_cooldown: float = 0.18
    placement_cooldown: float = 0.18
    save_path: Path | None = None
    load_on_start: bool = False
    autosave: bool = False
    bootstrap_inventory: bool = True
    terrain_config: TerrainGenerationConfig = field(
        default_factory=lambda: TerrainGenerationConfig(octave_count=2)
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("interaction_reach", self.interaction_reach),
            ("break_cooldown", self.break_cooldown),
            ("placement_cooldown", self.placement_cooldown),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number.")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.interaction_reach <= 0:
            raise ValueError("interaction_reach must be greater than zero.")
        if self.break_cooldown < 0 or self.placement_cooldown < 0:
            raise ValueError("interaction cooldowns must be non-negative.")
        if self.save_path is not None and not isinstance(self.save_path, Path):
            raise TypeError("save_path must be a pathlib.Path or None.")
        if not isinstance(self.load_on_start, bool):
            raise TypeError("load_on_start must be a boolean.")
        if not isinstance(self.autosave, bool):
            raise TypeError("autosave must be a boolean.")
        if not isinstance(self.bootstrap_inventory, bool):
            raise TypeError("bootstrap_inventory must be a boolean.")
        if (self.load_on_start or self.autosave) and self.save_path is None:
            raise ValueError("load and autosave require a save_path.")


@dataclass(slots=True)
class GpuChunk:
    """Owned ModernGL resources for one cached render chunk."""

    key: tuple[ChunkCoordinate, int, tuple[int, int, int, int], str, int]
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
            if self.save_path.suffix.lower() != ".json":
                raise ValueError("save_path must use the .json suffix.")
            self._save_slot = SaveSlot(self.save_path.stem)
            paths = ProjectPaths(
                project_root=self.save_path.parent,
                save_directory=self.save_path.parent,
                log_directory=self.save_path.parent / "logs",
            )
            self._save_service = GameSaveService(
                repository=SaveRepository(storage=RuntimeStorage(paths=paths)),
                context=self.session_context,
                logger=logging.getLogger("open_world_rpg"),
            )
        self.edits = BlockEditStore()
        self.editable_world = EditableVoxelWorld(column_at=self._column_at, edits=self.edits)
        self.interactions = VoxelInteractionController(
            world=self.editable_world,
            edits=self.edits,
            break_cooldown=self.config.break_cooldown,
            placement_cooldown=self.config.placement_cooldown,
        )
        self.inventory = create_bootstrap_inventory(enabled=self.config.bootstrap_inventory)
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
        self.render_distance = self.config.render_distance
        self.loading = False
        self.show_help = True
        self.show_debug = False
        self.mouse_captured = False
        self.target: RayHit | None = None
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
        self.hud_snapshot: VoxelHudSnapshot | None = None
        self._crosshair_buffer: moderngl.Buffer | None = None
        self._crosshair_array: moderngl.VertexArray | None = None
        self._target_buffer: moderngl.Buffer | None = None
        self._target_array: moderngl.VertexArray | None = None
        self._clock: pygame.time.Clock | None = None
        self._gpu_chunks: dict[ChunkCoordinate, GpuChunk] = {}
        self._visible: tuple[ChunkCoordinate, ...] = ()
        self._stream_signature: tuple[int, int, int] | None = None
        self._generation_seconds = 0.0
        self._mesh_seconds = 0.0

    @property
    def hotbar(self) -> VoxelHotbar:
        """Compatibility projection over the authoritative inventory hotbar."""
        return VoxelHotbar(
            slots=tuple(
                None if not isinstance(slot, ItemStack) else material_for_item(slot.item)
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
                vsync=1,
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
            self._capture_mouse(True)
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
                assert self._clock is not None
                delta = min(0.05, self._clock.tick(self.config.target_fps) / 1000.0)
                self.fps = self._clock.get_fps()
                self.process_events()
                self.update(delta)
                self.render()
                frames += 1
            completed = True
            return 0
        finally:
            if completed and self.config.autosave and self.dirty:
                self._save_edits()
            self.shutdown()

    def process_events(self) -> None:
        """Handle mouse capture and discrete prototype controls."""
        capture_just_lost = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
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
                    self._stream()
                elif event.key == pygame.K_F6:
                    self.render_distance = min(4, self.render_distance + 1)
                    self._stream_signature = None
                    self._stream()
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
                    self._mining_held = True
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

    def update(self, delta_seconds: float) -> None:
        """Apply first-person motion, physics, targeting, and streaming."""
        keys = pygame.key.get_pressed()
        forward = int(keys[pygame.K_w]) - int(keys[pygame.K_s])
        sideways = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
        microseconds = max(0, round(delta_seconds * 1_000_000))
        sprinting = bool(
            keys[pygame.K_LSHIFT]
            and (forward or sideways)
            and not self.player.flying
            and self.vitals.can_sprint
        )
        if self.vitals.update_stamina(
            microseconds, sprinting=sprinting, active=self.mouse_captured
        ):
            self.dirty = True
        speed = (10.0 if sprinting else 5.0) * delta_seconds
        flat_forward = (math.sin(math.radians(self.camera.yaw_degrees)), 0.0)
        flat_right = (math.cos(math.radians(self.camera.yaw_degrees)), 0.0)
        delta_x = (flat_forward[0] * forward + flat_right[0] * sideways) * speed
        delta_z = (
            -math.cos(math.radians(self.camera.yaw_degrees)) * forward
            + math.sin(math.radians(self.camera.yaw_degrees)) * sideways
        ) * speed
        jump_pressed = bool(keys[pygame.K_SPACE])
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
            return False

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
            vertical = int(keys[pygame.K_SPACE]) - int(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])
            self.player = PlayerState(
                x=self.player.x,
                y=self.player.y + vertical * speed,
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
                self._respawn_after_death()
        if self.dropped_items.update(delta_seconds, solid_at=self._solid_at):
            self.dirty = True
        pickups = self.dropped_items.pickup_near(
            position=(self.player.x, self.player.y + 0.9, self.player.z),
            inventory=self.inventory,
        )
        if pickups:
            self.dirty = True
            pickup = pickups[-1]
            item = cast(ItemType, pickup.item)
            self.save_message = f"Picked up {item.display_name} x{pickup.accepted}"
            self.last_pickup = self.save_message
            self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
        self._stream()
        self.target = ray_cast(
            origin=(self.player.x, self.player.y + 1.62, self.player.z),
            direction=self.camera.forward,
            block_at=self.editable_world.material_at,
            maximum_distance=self.config.interaction_reach,
        )
        self._update_mining(microseconds)

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
            field_of_view=72.0,
            aspect=width / max(1, height),
            near=0.1,
            far=float((self.render_distance + 2) * CHUNK_SIZE),
        )
        view = _view_matrix(
            position=(self.player.x, self.player.y + 1.62, self.player.z),
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
        cast(moderngl.Uniform, self.program["fog_far"]).value = float(
            (self.render_distance + 1.5) * CHUNK_SIZE
        )
        cast(moderngl.Uniform, self.program["water_time"]).value = pygame.time.get_ticks() / 1000.0
        triangles = 0
        for coordinate in self._visible:
            gpu = self._gpu_chunks[coordinate]
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
            water_array = self._gpu_chunks[coordinate].water_array
            if water_array is not None:
                water_array.render(moderngl.TRIANGLES)
        cast(Any, self.context).depth_mask = True
        if self.target is not None:
            self._render_target_outline(self.target)
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
                )
            )
        if self._crosshair_array is not None:
            self.context.disable(moderngl.DEPTH_TEST)
            self._crosshair_array.render(moderngl.LINES)
            self.context.enable(moderngl.DEPTH_TEST)
        self._render_hud(triangles)
        pygame.display.set_caption("Open World RPG — Voxel Prototype")
        pygame.display.flip()

    def shutdown(self) -> None:
        """Explicitly release GPU objects and suspend active terrain."""
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

    def _stream(self) -> None:
        signature = (
            math.floor(self.player.x) // CHUNK_SIZE,
            math.floor(self.player.z) // CHUNK_SIZE,
            self.render_distance,
        )
        if signature == self._stream_signature:
            return
        wanted = streaming_chunks(
            world_x=self.player.x,
            world_z=self.player.z,
            render_distance=self.render_distance,
        )
        wanted_set = set(wanted)
        self.loading = any(not self.runtime.contains(item) for item in wanted)
        if self.loading and pygame.display.get_init():
            pygame.display.set_caption("Open World RPG Voxel | Generating terrain...")
        generation_start = time.perf_counter()
        for coordinate in wanted:
            terrain = self.runtime.get_or_generate(coordinate)
            state = self.runtime.metadata_at(coordinate).state
            if state in (ChunkState.READY, ChunkState.SUSPENDED):
                self.runtime.activate(coordinate)
            if self.context is not None:
                revisions = self._neighbour_revisions(coordinate)
                key = mesh_cache_key(terrain=terrain, neighbour_revisions=revisions)
                cached = self._gpu_chunks.get(coordinate)
                if cached is None or cached.key != key:
                    if cached is not None:
                        cached.release()
                    mesh_start = time.perf_counter()
                    mesh = build_chunk_mesh(
                        terrain=terrain,
                        column_at_world=self._column_at,
                        block_at_world=self.editable_world.material_at,
                    )
                    self._mesh_seconds += time.perf_counter() - mesh_start
                    opaque_buffer = self.context.buffer(mesh.opaque_vertices)
                    opaque_array = self.context.vertex_array(
                        self.program,
                        [(opaque_buffer, "3f 2f 1f", "in_position", "in_uv", "in_shade")],
                    )
                    water_buffer = (
                        self.context.buffer(mesh.water_vertices)
                        if mesh.water_vertex_count
                        else None
                    )
                    water_array = (
                        self.context.vertex_array(
                            self.program,
                            [
                                (
                                    water_buffer,
                                    "3f 2f 1f",
                                    "in_position",
                                    "in_uv",
                                    "in_shade",
                                )
                            ],
                        )
                        if water_buffer is not None
                        else None
                    )
                    self._gpu_chunks[coordinate] = GpuChunk(
                        key=key,
                        opaque_buffer=opaque_buffer,
                        opaque_array=opaque_array,
                        water_buffer=water_buffer,
                        water_array=water_array,
                        mesh=mesh,
                    )
        self._generation_seconds += time.perf_counter() - generation_start
        for coordinate in self.runtime.coordinates():
            if coordinate not in wanted_set and (
                self.runtime.metadata_at(coordinate).state is ChunkState.ACTIVE
            ):
                self.runtime.suspend(coordinate)
        for coordinate in tuple(self._gpu_chunks):
            if coordinate not in wanted_set:
                self._gpu_chunks.pop(coordinate).release()
        self._visible = wanted
        self.loading = False
        self._stream_signature = signature

    def _column_at(self, world_x: int, world_z: int) -> BlockColumn:
        coordinate = ChunkCoordinate(x=world_x // CHUNK_SIZE, y=world_z // CHUNK_SIZE)
        terrain = self.runtime.get_or_generate(coordinate)
        tile = terrain.tile_at(LocalTileCoordinate(x=world_x % CHUNK_SIZE, y=world_z % CHUNK_SIZE))
        return column_from_terrain(
            terrain_type=tile.terrain_type, elevation_metres=tile.elevation.metres
        )

    def _height_at(self, world_x: int, world_z: int) -> int:
        return self._column_at(world_x, world_z).ground_height

    def _solid_at(self, world_x: int, world_y: int, world_z: int) -> bool:
        return self.editable_world.solid_at(world_x, world_y, world_z)

    def _apply_interaction(self, outcome: InteractionOutcome) -> None:
        self.last_interaction = outcome.result
        if outcome.result is InteractionResult.PLACED:
            self.last_placement_consumption = "consumed 1 selected item"
        self.save_message = ""
        if not outcome.changed:
            if outcome.result is InteractionResult.PLAYER_INTERSECTION:
                self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
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
        for coordinate in outcome.invalidated_chunks:
            cached = self._gpu_chunks.pop(coordinate, None)
            if cached is not None:
                cached.release()
        self._stream_signature = None
        self._stream()
        self.target = ray_cast(
            origin=(self.player.x, self.player.y + 1.62, self.player.z),
            direction=self.camera.forward,
            block_at=self.editable_world.material_at,
            maximum_distance=self.config.interaction_reach,
        )

    def _update_mining(self, microseconds: int) -> None:
        if not self._mining_held or not self.mouse_captured or self.target is None:
            if self.mining.snapshot.status is MiningStatus.ACTIVE:
                self.mining.cancel("target unavailable")
            return
        target = self.target
        snapshot = self.mining.snapshot
        selected_tool = self.inventory.selected_tool
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
            )
        except Exception:
            self.save_message = "Save failed"
            self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
            return False
        self.dirty = False
        self.save_message = "World saved"
        self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
        return True

    def _load_edits(self) -> bool:
        if self._save_service is None or self._save_slot is None:
            self.save_message = "Load failed"
            self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
            return False
        try:
            document = self._save_service.load(self._save_slot)
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
                    enabled=self.config.bootstrap_inventory
                ).snapshot(),
            )
            restored_vitals = self._save_service.restore_vitals(
                document,
                expected_world_id=self.world_id,
                expected_world_seed=self.config.world_seed,
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
        self._mining_held = False
        self.mining.reset()
        self._drop_render_revision = -1
        self.editable_world = EditableVoxelWorld(
            column_at=self._column_at,
            edits=self.edits,
        )
        self.interactions = VoxelInteractionController(
            world=self.editable_world,
            edits=self.edits,
            break_cooldown=self.config.break_cooldown,
            placement_cooldown=self.config.placement_cooldown,
        )
        affected = {
            chunk for coordinate in changed for chunk in invalidated_chunks_for_edit(coordinate)
        }
        for coordinate in affected:
            cached = self._gpu_chunks.pop(coordinate, None)
            if cached is not None:
                cached.release()
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
        self._stream()
        self.target = ray_cast(
            origin=(self.player.x, self.player.y + 1.62, self.player.z),
            direction=self.camera.forward,
            block_at=self.editable_world.material_at,
            maximum_distance=self.config.interaction_reach,
        )
        self.dirty = False
        self.save_message = "World loaded"
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

    def _render_target_outline(self, target: RayHit) -> None:
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
            data.extend((*corners[start], 1.0, 0.93, 0.32, 1.0))
            data.extend((*corners[end], 1.0, 0.93, 0.32, 1.0))
        self._target_buffer.write(data.tobytes())
        self._target_array.render(moderngl.LINES)

    def _caption(self, triangles: int) -> str:
        status = " | loading" if self.loading else ""
        mode = "FLY" if self.player.flying else "WALK"
        basic = f"Open World RPG Voxel | {self.fps:4.0f} FPS | {mode}{status}"
        if self.show_help:
            basic += " | WASD move, mouse look, Space jump/up, Ctrl down, F fly, F3 debug"
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
                else material_for_item(self.inventory.selected_stack.item)
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
                    f"Selected {hud.selected_material.value if hud.selected_material else 'empty'}",
                    f"Target {hud.target or 'none'} "
                    f"{hud.target_material.value if hud.target_material else ''} "
                    f"face {hud.target_face or 'none'}",
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
        self._hud_texture.write(pygame.image.tobytes(surface, "RGBA", True))
        self.context.disable(moderngl.DEPTH_TEST)
        self._hud_texture.use(location=1)
        self._hud_array.render(moderngl.TRIANGLES)
        self.context.enable(moderngl.DEPTH_TEST)

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
                    material = material_for_item(slot.item)
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
