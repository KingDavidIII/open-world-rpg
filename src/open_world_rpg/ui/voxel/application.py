"""Pygame/ModernGL first-person voxel prototype controller."""

from __future__ import annotations

import logging
import math
import struct
import time
from array import array
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import moderngl
import pygame

from open_world_rpg.application import create_terrain_runtime
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
from .collision import RayHit, move_player, ray_cast, safe_spawn_height
from .editable_world import EditableVoxelWorld
from .hotbar import VoxelHotbar
from .hud import VoxelHudSnapshot
from .interaction import InteractionOutcome, InteractionResult, VoxelInteractionController
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
        specification = WorldSpecification(
            name="Voxel Prototype", seed=WorldSeed(value=self.config.world_seed)
        )
        world = WorldModel.create(
            specification=specification,
            created_at=datetime(1970, 1, 1, tzinfo=UTC),
            world_id=WorldId(value=UUID(int=1)),
        )
        self.runtime = create_terrain_runtime(world=world, config=self.config.terrain_config)
        self.edits = BlockEditStore()
        self.editable_world = EditableVoxelWorld(column_at=self._column_at, edits=self.edits)
        self.interactions = VoxelInteractionController(
            world=self.editable_world,
            edits=self.edits,
            break_cooldown=self.config.break_cooldown,
            placement_cooldown=self.config.placement_cooldown,
        )
        self.hotbar = VoxelHotbar()
        self.last_interaction = InteractionResult.NONE
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
        try:
            while self.running and (max_frames is None or frames < max_frames):
                assert self._clock is not None
                delta = min(0.05, self._clock.tick(self.config.target_fps) / 1000.0)
                self.fps = self._clock.get_fps()
                self.process_events()
                self.update(delta)
                self.render()
                frames += 1
            return 0
        finally:
            self.shutdown()

    def process_events(self) -> None:
        """Handle mouse capture and discrete prototype controls."""
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
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    self.hotbar = self.hotbar.select(event.key - pygame.K_1 + 1)
                    self._selection_changed_at = pygame.time.get_ticks() / 1000.0
            elif event.type == pygame.MOUSEWHEEL:
                self.hotbar = self.hotbar.cycle(event.y)
                self._selection_changed_at = pygame.time.get_ticks() / 1000.0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not self.mouse_captured:
                    self._capture_mouse(True)
                elif event.button in (4, 5):
                    self.hotbar = self.hotbar.cycle(1 if event.button == 4 else -1)
                    self._selection_changed_at = pygame.time.get_ticks() / 1000.0
                elif event.button == 1:
                    self._apply_interaction(
                        self.interactions.break_block(
                            target=self.target,
                            now=pygame.time.get_ticks() / 1000.0,
                        )
                    )
                elif event.button == 3:
                    self._apply_interaction(
                        self.interactions.place_block(
                            target=self.target,
                            material=self.hotbar.selected_material,
                            player=self.player,
                            now=pygame.time.get_ticks() / 1000.0,
                        )
                    )

    def update(self, delta_seconds: float) -> None:
        """Apply first-person motion, physics, targeting, and streaming."""
        keys = pygame.key.get_pressed()
        forward = int(keys[pygame.K_w]) - int(keys[pygame.K_s])
        sideways = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
        speed = (10.0 if keys[pygame.K_LSHIFT] else 5.0) * delta_seconds
        flat_forward = (math.sin(math.radians(self.camera.yaw_degrees)), 0.0)
        flat_right = (math.cos(math.radians(self.camera.yaw_degrees)), 0.0)
        delta_x = (flat_forward[0] * forward + flat_right[0] * sideways) * speed
        delta_z = (
            -math.cos(math.radians(self.camera.yaw_degrees)) * forward
            + math.sin(math.radians(self.camera.yaw_degrees)) * sideways
        ) * speed
        self.player = move_player(
            player=self.player,
            delta_x=delta_x,
            delta_z=delta_z,
            delta_seconds=delta_seconds,
            height_at=self._height_at,
            solid_at=self._solid_at,
            jump=bool(keys[pygame.K_SPACE]),
        )
        if self.player.flying:
            vertical = int(keys[pygame.K_SPACE]) - int(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])
            self.player = PlayerState(
                x=self.player.x,
                y=self.player.y + vertical * speed,
                z=self.player.z,
                flying=True,
            )
        self._stream()
        self.target = ray_cast(
            origin=(self.player.x, self.player.y + 1.62, self.player.z),
            direction=self.camera.forward,
            block_at=self.editable_world.material_at,
            maximum_distance=self.config.interaction_reach,
        )

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
        if not outcome.changed:
            if outcome.result is InteractionResult.PLAYER_INTERSECTION:
                self._feedback_until = pygame.time.get_ticks() / 1000.0 + 1.25
            return
        self._feedback_coordinate = outcome.coordinate
        self._feedback_until = pygame.time.get_ticks() / 1000.0 + 0.22
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
            selected_material=self.hotbar.selected_material,
            edit_revision=self.edits.revision,
            edited_block_count=len(self.edits),
            last_interaction=self.last_interaction.value,
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
                )
            )
        surface = pygame.Surface((1024, 512), pygame.SRCALPHA)
        surface.fill((8, 13, 20, 175), pygame.Rect(0, 0, 500, len(lines) * 22 + 12))
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
        total_width = len(self.hotbar.slots) * slot_size + (len(self.hotbar.slots) - 1) * gap
        start_x = (surface.get_width() - total_width) // 2
        y = surface.get_height() - slot_size - 18
        for index, material in enumerate(self.hotbar.slots):
            rect = pygame.Rect(start_x + index * (slot_size + gap), y, slot_size, slot_size)
            is_selected = index == self.hotbar.selected_index
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
            number = font.render(str(index + 1), True, (245, 245, 245))
            surface.blit(number, (rect.x + 3, rect.y + 1))
        now = pygame.time.get_ticks() / 1000.0
        selected_material = self.hotbar.selected_material
        if selected_material is not None and now - self._selection_changed_at < 1.4:
            label = font.render(selected_material.value.upper(), True, (255, 245, 190))
            surface.blit(label, (surface.get_width() // 2 - label.get_width() // 2, y - 24))
        if now < self._feedback_until:
            feedback = font.render(self.last_interaction.value, True, (255, 218, 145))
            surface.blit(
                feedback,
                (surface.get_width() // 2 - feedback.get_width() // 2, y - 46),
            )
