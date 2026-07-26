"""Tests for voxel atlas, strata, scenery, spawn, water, and HUD policy."""

from __future__ import annotations

import struct

import pytest

from open_world_rpg.gameplay import DroppedItem, ItemType
from open_world_rpg.ui.voxel import (
    BlockColumn,
    BlockType,
    FaceTexture,
    PlayerState,
    RayHit,
    SceneryKind,
    VoxelHudSnapshot,
    atlas_uv,
    generate_texture_atlas,
    scenery_at,
    select_spawn,
)
from open_world_rpg.ui.voxel.item_rendering import build_dropped_item_vertices
from open_world_rpg.ui.voxel.meshing import _material_texture, side_texture, top_texture
from open_world_rpg.ui.voxel.texture_atlas import ATLAS_SIZE


def grass_column(height: int = 12) -> BlockColumn:
    return BlockColumn(
        ground_height=height,
        surface_height=height,
        surface=BlockType.GRASS,
        subsurface=BlockType.DIRT,
    )


def test_texture_atlas_is_complete_crisp_and_deterministic() -> None:
    atlas = generate_texture_atlas()
    assert len(atlas) == ATLAS_SIZE * ATLAS_SIZE * 4
    assert atlas == generate_texture_atlas()
    assert len({atlas_uv(texture) for texture in FaceTexture}) == len(FaceTexture)
    assert top_texture(BlockType.GRASS) is FaceTexture.GRASS_TOP
    assert top_texture(BlockType.WATER) is FaceTexture.SHALLOW_WATER
    assert _material_texture(BlockType.SNOW, top=False) is FaceTexture.SNOW_SIDE
    with pytest.raises(TypeError):
        atlas_uv("grass")  # type: ignore[arg-type]


def test_dropped_item_geometry_is_one_deterministic_atlas_batch() -> None:
    items = (
        DroppedItem(
            identifier=1,
            item=ItemType.STONE_BLOCK,
            quantity=1,
            position=(-1.5, 2.0, 3.5),
            age=1,
            settled=True,
        ),
        DroppedItem(
            identifier=2,
            item=ItemType.SNOW_BLOCK,
            quantity=2,
            position=(0.5, 4.0, 0.5),
        ),
    )
    vertices = build_dropped_item_vertices(items)
    assert len(vertices) == 2 * 12 * 6 * 4
    assert vertices == build_dropped_item_vertices(items)
    assert build_dropped_item_vertices(()) == b""
    with pytest.raises(TypeError):
        build_dropped_item_vertices([])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_dropped_item_vertices(("bad",))  # type: ignore[arg-type]


def test_side_strata_select_grass_dirt_stone_sand_and_snow() -> None:
    grass = grass_column(20)
    assert side_texture(column=grass, block_y=19) is FaceTexture.GRASS_SIDE
    assert side_texture(column=grass, block_y=17) is FaceTexture.DIRT
    assert side_texture(column=grass, block_y=10) is FaceTexture.STONE
    assert side_texture(column=grass, block_y=-100) is FaceTexture.STONE
    sand = BlockColumn(
        ground_height=10,
        surface_height=10,
        surface=BlockType.SAND,
        subsurface=BlockType.SAND,
    )
    assert side_texture(column=sand, block_y=1) is FaceTexture.SAND
    snow = BlockColumn(
        ground_height=50,
        surface_height=50,
        surface=BlockType.SNOW,
        subsurface=BlockType.STONE,
    )
    assert side_texture(column=snow, block_y=49) is FaceTexture.SNOW_SIDE
    assert side_texture(column=snow, block_y=40) is FaceTexture.STONE
    dirt = BlockColumn(
        ground_height=3,
        surface_height=3,
        surface=BlockType.DIRT,
        subsurface=BlockType.DIRT,
    )
    assert side_texture(column=dirt, block_y=2) is FaceTexture.STONE


def test_steep_grassy_ridge_has_no_isolated_stone_top_teeth() -> None:
    """A steep hill keeps a connected grass cap and exposes rock vertically."""
    from open_world_rpg.ui.voxel import column_from_terrain
    from open_world_rpg.world import TerrainType

    ridge = column_from_terrain(
        terrain_type=TerrainType.HILLS,
        elevation_metres=600,
        steep=True,
    )
    assert top_texture(ridge.surface) is FaceTexture.GRASS_TOP
    assert (
        side_texture(
            column=ridge,
            block_y=ridge.ground_height - 1,
        )
        is FaceTexture.GRASS_SIDE
    )
    assert (
        side_texture(
            column=ridge,
            block_y=ridge.ground_height - 2,
        )
        is FaceTexture.STONE
    )


def test_scenery_is_deterministic_restrained_and_owned_by_base_chunk() -> None:
    found = {
        placement.kind: placement
        for x in range(-500, 500)
        if (
            placement := scenery_at(
                seed=7,
                world_x=x,
                world_z=-1,
                column=grass_column(),
                slope=0,
            )
        )
    }
    assert set(found) == set(SceneryKind)
    assert all(
        placement
        == scenery_at(
            seed=7,
            world_x=placement.world_x,
            world_z=-1,
            column=grass_column(),
            slope=0,
        )
        for placement in found.values()
    )
    assert all(placement.owner.x == placement.world_x // 16 for placement in found.values())
    water = BlockColumn(
        ground_height=4,
        surface_height=12,
        surface=BlockType.SAND,
        subsurface=BlockType.SAND,
        water=BlockType.WATER,
    )
    assert scenery_at(seed=1, world_x=0, world_z=0, column=water, slope=0) is None
    assert scenery_at(seed=1, world_x=0, world_z=0, column=grass_column(), slope=2) is None


def test_spawn_prefers_dry_relief_near_water_and_validates_policy() -> None:
    water = BlockColumn(
        ground_height=5,
        surface_height=12,
        surface=BlockType.SAND,
        subsurface=BlockType.SAND,
        water=BlockType.WATER,
    )

    def column_at(x: int, z: int) -> BlockColumn:
        if abs(x - 4) <= 1 and abs(z) <= 1:
            return grass_column(16)
        if abs(x - 4) == 6 or abs(z) == 6:
            return water
        return grass_column(12)

    assert select_spawn(column_at=column_at, origin_x=0, origin_z=0, radius=8, step=4) == (4, 0)
    assert select_spawn(
        column_at=column_at,
        blocked_at=lambda x, z: (x, z) == (4, 0),
        origin_x=0,
        origin_z=0,
        radius=8,
        step=4,
    ) != (4, 0)
    with pytest.raises(ValueError):
        select_spawn(column_at=column_at, radius=-1)
    with pytest.raises(ValueError, match="No safe voxel spawn"):
        select_spawn(column_at=lambda _x, _z: water, radius=0)


def test_hud_snapshot_projects_negative_coordinates_and_modes() -> None:
    snapshot = VoxelHudSnapshot.create(
        fps=60.0,
        player=PlayerState(x=-0.1, y=14.2, z=-16.1, flying=True),
        seed=9,
        active_chunks=4,
        cached_chunks=8,
        mesh_count=4,
        triangles=123,
        render_distance=2,
        target=RayHit(
            x=-1,
            y=14,
            z=-17,
            distance=1.0,
            material=BlockType.STONE,
            face_normal=(0, 1, 0),
        ),
        loading=True,
        selected_material=BlockType.DIRT,
        edit_revision=4,
        edited_block_count=3,
        last_interaction="block placed",
        save_path="C:/saves/voxel.json",
        dirty=True,
    )
    assert snapshot.block == (-1, 14, -17)
    assert (snapshot.chunk.x, snapshot.chunk.y) == (-1, -2)
    assert snapshot.mode == "FLY"
    assert snapshot.loading
    assert snapshot.target_material is BlockType.STONE
    assert snapshot.target_face == (0, 1, 0)
    assert snapshot.selected_material is BlockType.DIRT
    assert snapshot.edit_revision == 4
    assert snapshot.edited_block_count == 3
    assert snapshot.last_interaction == "block placed"
    assert snapshot.save_path == "C:/saves/voxel.json"
    assert snapshot.dirty


def test_mesh_vertex_layout_has_upward_top_face_winding() -> None:
    # First triangle of a textured top face: position/uv/shade, six floats each.
    from array import array

    from open_world_rpg.ui.voxel.meshing import _top

    output = array("f")
    _top(
        output,
        x=0,
        y=1,
        z=0,
        texture=FaceTexture.GRASS_TOP,
    )
    values = struct.unpack(f"{len(output)}f", output.tobytes())
    points = [values[index : index + 3] for index in (0, 6, 12)]
    first = tuple(points[1][i] - points[0][i] for i in range(3))
    second = tuple(points[2][i] - points[0][i] for i in range(3))
    normal_y = first[2] * second[0] - first[0] * second[2]
    assert normal_y > 0
