"""Tests for renderer-independent voxel mapping, physics, and meshing."""

from __future__ import annotations

import struct
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from open_world_rpg.application import create_terrain_runtime
from open_world_rpg.ui.voxel import (
    DISPLAY_SEA_LEVEL,
    BlockColumn,
    BlockType,
    FirstPersonCamera,
    PlayerState,
    build_chunk_mesh,
    camera_vectors,
    column_from_terrain,
    mesh_cache_key,
    move_player,
    ray_cast,
    safe_spawn_height,
    streaming_chunks,
)
from open_world_rpg.ui.voxel.blocks import elevation_to_height
from open_world_rpg.world import (
    CHUNK_SIZE,
    ChunkCoordinate,
    TerrainGenerationConfig,
    TerrainType,
    WorldSeed,
    WorldSpecification,
)


def test_elevation_display_scale_is_bounded_and_negative_safe() -> None:
    assert elevation_to_height(0) == DISPLAY_SEA_LEVEL
    assert elevation_to_height(150) == DISPLAY_SEA_LEVEL + 5
    assert elevation_to_height(-1) == DISPLAY_SEA_LEVEL - 1
    assert elevation_to_height(-999_999) == 1
    assert elevation_to_height(999_999) == 64
    with pytest.raises(TypeError):
        elevation_to_height(cast(Any, True))


@pytest.mark.parametrize(
    ("terrain_type", "expected_surface", "water"),
    [
        (TerrainType.DEEP_WATER, BlockType.STONE, BlockType.DEEP_WATER),
        (TerrainType.SHALLOW_WATER, BlockType.SAND, BlockType.WATER),
        (TerrainType.COAST, BlockType.SAND, None),
        (TerrainType.PLAINS, BlockType.GRASS, None),
        (TerrainType.HILLS, BlockType.GRASS, None),
        (TerrainType.MOUNTAINS, BlockType.STONE, None),
    ],
)
def test_terrain_types_map_to_voxel_columns(
    terrain_type: TerrainType,
    expected_surface: BlockType,
    water: BlockType | None,
) -> None:
    column = column_from_terrain(terrain_type=terrain_type, elevation_metres=600)
    assert column.surface is expected_surface
    assert column.water is water


def test_steep_hills_keep_grass_caps_over_exposed_stone_cliffs() -> None:
    steep_hill = column_from_terrain(
        terrain_type=TerrainType.HILLS, elevation_metres=600, steep=True
    )
    assert steep_hill.surface is BlockType.GRASS
    assert steep_hill.subsurface is BlockType.STONE
    assert (
        column_from_terrain(terrain_type=TerrainType.MOUNTAINS, elevation_metres=3_000).surface
        is BlockType.SNOW
    )
    with pytest.raises(TypeError):
        column_from_terrain(terrain_type=cast(Any, "plains"), elevation_metres=0)
    with pytest.raises(TypeError):
        column_from_terrain(
            terrain_type=TerrainType.PLAINS,
            elevation_metres=0,
            steep=cast(Any, 1),
        )


def test_camera_vectors_look_and_pitch_clamping() -> None:
    forward, right = camera_vectors(yaw_degrees=0.0, pitch_degrees=0.0)
    assert forward == pytest.approx((0.0, 0.0, -1.0))
    assert right == pytest.approx((1.0, 0.0, 0.0))
    camera = FirstPersonCamera().looked(delta_x=100.0, delta_y=-1000.0)
    assert camera.yaw_degrees == 12.0
    assert camera.pitch_degrees == 89.0
    assert camera.forward == camera_vectors(yaw_degrees=12.0, pitch_degrees=89.0)[0]
    assert camera.right == camera_vectors(yaw_degrees=12.0, pitch_degrees=89.0)[1]


def test_player_gravity_ground_jump_and_flying() -> None:
    def height(_x: int, _z: int) -> int:
        return 4

    player = PlayerState(x=0.0, y=5.0, z=0.0, grounded=True)
    jumped = move_player(
        player=player,
        delta_x=1.0,
        delta_z=-1.0,
        delta_seconds=0.1,
        height_at=height,
        jump=True,
    )
    assert jumped.y > 5.0
    landed = move_player(
        player=PlayerState(x=0.0, y=4.0, z=0.0),
        delta_x=0.0,
        delta_z=0.0,
        delta_seconds=1.0,
        height_at=height,
    )
    assert landed.y == 5.0
    assert landed.grounded
    flying = move_player(
        player=PlayerState(x=0.0, y=9.0, z=0.0, flying=True),
        delta_x=2.0,
        delta_z=3.0,
        delta_seconds=1.0,
        height_at=height,
    )
    assert flying == PlayerState(x=2.0, y=9.0, z=3.0, flying=True)
    assert safe_spawn_height(world_x=-1, world_z=-2, height_at=height) == 5.0
    with pytest.raises(ValueError):
        move_player(
            player=player,
            delta_x=0,
            delta_z=0,
            delta_seconds=-1,
            height_at=height,
        )


def test_ray_cast_hits_first_block_and_validates_range() -> None:
    hit = ray_cast(
        origin=(0.5, 2.5, 0.5),
        direction=(0.0, -1.0, 0.0),
        solid_at=lambda _x, y, _z: y <= 0,
    )
    assert hit is not None
    assert (hit.x, hit.y, hit.z) == (0, 0, 0)
    assert (
        ray_cast(
            origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            solid_at=lambda _x, _y, _z: False,
            maximum_distance=1.0,
        )
        is None
    )
    with pytest.raises(ValueError):
        ray_cast(
            origin=(0, 0, 0),
            direction=(1, 0, 0),
            solid_at=lambda _x, _y, _z: False,
            step=0,
        )


def test_streaming_is_row_major_and_supports_negative_coordinates() -> None:
    assert streaming_chunks(world_x=-0.1, world_z=-0.1, render_distance=0) == (
        ChunkCoordinate(x=-1, y=-1),
    )
    assert streaming_chunks(world_x=0, world_z=0, render_distance=1) == tuple(
        ChunkCoordinate(x=x, y=y) for y in (-1, 0, 1) for x in (-1, 0, 1)
    )
    with pytest.raises(TypeError):
        streaming_chunks(world_x=0, world_z=0, render_distance=True)
    with pytest.raises(ValueError):
        streaming_chunks(world_x=0, world_z=0, render_distance=-1)


def test_mesh_is_deterministic_compact_and_cache_key_tracks_neighbours() -> None:
    runtime = create_terrain_runtime(
        world=WorldSpecification(name="Voxel", seed=WorldSeed(value=3)),
        config=TerrainGenerationConfig(octave_count=1),
    )
    terrain = runtime.get_or_generate(ChunkCoordinate(x=-1, y=0))

    def column_at(world_x: int, world_z: int):
        chunk = runtime.get_or_generate(
            ChunkCoordinate(x=world_x // CHUNK_SIZE, y=world_z // CHUNK_SIZE)
        )
        tile = chunk.tiles[(world_z % CHUNK_SIZE) * CHUNK_SIZE + world_x % CHUNK_SIZE]
        return column_from_terrain(
            terrain_type=tile.terrain_type,
            elevation_metres=tile.elevation.metres,
        )

    mesh = build_chunk_mesh(terrain=terrain, column_at_world=column_at)
    assert mesh == build_chunk_mesh(terrain=terrain, column_at_world=column_at)
    assert mesh.vertex_count >= CHUNK_SIZE * CHUNK_SIZE * 6
    assert mesh.triangle_count == mesh.vertex_count // 3
    key = mesh_cache_key(terrain=terrain, neighbour_revisions=(0, 0, 0, 0))
    assert key != mesh_cache_key(terrain=terrain, neighbour_revisions=(1, 0, 0, 0))
    low_column = BlockColumn(
        ground_height=1,
        surface_height=1,
        surface=BlockType.STONE,
        subsurface=BlockType.STONE,
    )
    exposed = build_chunk_mesh(
        terrain=terrain,
        column_at_world=lambda _x, _z: low_column,
    )
    assert exposed.vertex_count > mesh.vertex_count
    values = struct.unpack(
        f"{exposed.opaque_vertex_count * 6}f",
        exposed.opaque_vertices,
    )
    normals: set[tuple[int, int, int]] = set()
    for offset in range(0, len(values), 18):
        points = [values[offset + index : offset + index + 3] for index in (0, 6, 12)]
        first = tuple(points[1][axis] - points[0][axis] for axis in range(3))
        second = tuple(points[2][axis] - points[0][axis] for axis in range(3))
        normal = (
            round(first[1] * second[2] - first[2] * second[1]),
            round(first[2] * second[0] - first[0] * second[2]),
            round(first[0] * second[1] - first[1] * second[0]),
        )
        normals.add(normal)
    assert {(0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)} <= normals
    with pytest.raises(TypeError):
        build_chunk_mesh(terrain=cast(Any, object()), column_at_world=column_at)
    with pytest.raises(FrozenInstanceError):
        mesh.opaque_vertex_count = 0  # type: ignore[misc]
