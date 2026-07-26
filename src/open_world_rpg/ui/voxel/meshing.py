"""Texture-atlas voxel meshing with strata and separate transparent water."""

from __future__ import annotations

from array import array
from collections.abc import Callable
from dataclasses import dataclass

from open_world_rpg.world import ChunkCoordinate, ChunkTerrain

from .blocks import BlockColumn, BlockType, column_from_terrain
from .scenery import SceneryKind, scenery_at
from .texture_atlas import FaceTexture, atlas_uv

ColumnLookup = Callable[[int, int], BlockColumn]
Point = tuple[float, float, float]


@dataclass(frozen=True, slots=True, kw_only=True)
class VoxelChunkMesh:
    """GPU-ready opaque and transparent position/UV/shade triangle streams."""

    coordinate: ChunkCoordinate
    opaque_vertices: bytes
    water_vertices: bytes
    opaque_vertex_count: int
    water_vertex_count: int
    triangle_count: int
    terrain_revision: int

    @property
    def vertex_count(self) -> int:
        return self.opaque_vertex_count + self.water_vertex_count


def mesh_cache_key(
    *, terrain: ChunkTerrain, neighbour_revisions: tuple[int, int, int, int]
) -> tuple[ChunkCoordinate, int, tuple[int, int, int, int], str, int]:
    """Invalidate terrain, edge, format, or visual mesh-contract changes."""
    return (
        terrain.chunk_coordinate,
        terrain.revision,
        neighbour_revisions,
        terrain.generation_format_version,
        2,
    )


def top_texture(block: BlockType) -> FaceTexture:
    """Select the atlas cell used for one upward face."""
    return {
        BlockType.DEEP_WATER: FaceTexture.DEEP_WATER,
        BlockType.WATER: FaceTexture.SHALLOW_WATER,
        BlockType.SAND: FaceTexture.SAND,
        BlockType.GRASS: FaceTexture.GRASS_TOP,
        BlockType.DIRT: FaceTexture.DIRT,
        BlockType.STONE: FaceTexture.STONE,
        BlockType.SNOW: FaceTexture.SNOW_TOP,
    }[block]


def side_texture(*, column: BlockColumn, block_y: int) -> FaceTexture:
    """Select visible strata: surface edge, dirt/sand, then stone."""
    if column.surface is BlockType.SAND:
        return FaceTexture.SAND
    if column.surface is BlockType.STONE:
        return FaceTexture.STONE
    if column.surface is BlockType.SNOW:
        return FaceTexture.SNOW_SIDE if block_y >= column.ground_height - 1 else FaceTexture.STONE
    if column.surface is BlockType.GRASS:
        if block_y >= column.ground_height - 1:
            return FaceTexture.GRASS_SIDE
        if column.subsurface is BlockType.STONE:
            return FaceTexture.STONE
        if block_y >= column.ground_height - 4:
            return FaceTexture.DIRT
    return FaceTexture.STONE


def _quad(
    output: array[float],
    *,
    points: tuple[Point, Point, Point, Point],
    texture: FaceTexture,
    shade: float,
) -> None:
    u0, v0, u1, v1 = atlas_uv(texture)
    uvs = ((u0, v0), (u0, v1), (u1, v1), (u1, v0))
    for index in (0, 1, 2, 0, 2, 3):
        output.extend((*points[index], *uvs[index], shade))


def _top(
    output: array[float],
    *,
    x: int,
    y: int,
    z: int,
    texture: FaceTexture,
    shade: float = 1.08,
) -> None:
    _quad(
        output,
        points=((x, y, z), (x, y, z + 1), (x + 1, y, z + 1), (x + 1, y, z)),
        texture=texture,
        shade=shade,
    )


def _cube(
    output: array[float],
    *,
    minimum: Point,
    maximum: Point,
    top: FaceTexture,
    side: FaceTexture,
) -> None:
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    _quad(
        output,
        points=((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)),
        texture=top,
        shade=1.08,
    )
    faces = (
        (((x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (x0, y0, z0)), 0.72),
        (((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)), 0.88),
        (((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)), 0.96),
        (((x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, z1)), 0.66),
    )
    for points, shade in faces:
        _quad(output, points=points, texture=side, shade=shade)


def _add_scenery(
    output: array[float],
    *,
    kind: SceneryKind,
    x: int,
    y: int,
    z: int,
) -> None:
    if kind is SceneryKind.TREE:
        _cube(
            output,
            minimum=(x + 0.35, y, z + 0.35),
            maximum=(x + 0.65, y + 4, z + 0.65),
            top=FaceTexture.DIRT,
            side=FaceTexture.DIRT,
        )
        _cube(
            output,
            minimum=(x - 0.5, y + 3, z - 0.5),
            maximum=(x + 1.5, y + 5.5, z + 1.5),
            top=FaceTexture.GRASS_TOP,
            side=FaceTexture.GRASS_TOP,
        )
    elif kind is SceneryKind.ROCK:
        _cube(
            output,
            minimum=(x + 0.2, y, z + 0.2),
            maximum=(x + 0.8, y + 0.55, z + 0.8),
            top=FaceTexture.STONE,
            side=FaceTexture.STONE,
        )
    else:
        height = 0.8 if kind is SceneryKind.SHRUB else 0.45
        for points in (
            (
                (x + 0.15, y, z + 0.15),
                (x + 0.15, y + height, z + 0.15),
                (x + 0.85, y + height, z + 0.85),
                (x + 0.85, y, z + 0.85),
            ),
            (
                (x + 0.85, y, z + 0.15),
                (x + 0.85, y + height, z + 0.15),
                (x + 0.15, y + height, z + 0.85),
                (x + 0.15, y, z + 0.85),
            ),
        ):
            _quad(
                output,
                points=points,
                texture=FaceTexture.GRASS_TOP,
                shade=0.95,
            )
            _quad(
                output,
                points=(points[3], points[2], points[1], points[0]),
                texture=FaceTexture.GRASS_TOP,
                shade=0.85,
            )


def build_chunk_mesh(*, terrain: ChunkTerrain, column_at_world: ColumnLookup) -> VoxelChunkMesh:
    """Build opaque terrain first and water into an independent blend stream."""
    if not isinstance(terrain, ChunkTerrain):
        raise TypeError("terrain must be a ChunkTerrain.")
    opaque = array("f")
    water = array("f")
    origin = terrain.chunk_coordinate.to_world_origin()
    for tile in terrain:
        world_x = origin.x + tile.coordinate.x
        world_z = origin.y + tile.coordinate.y
        west = column_at_world(world_x - 1, world_z)
        east = column_at_world(world_x + 1, world_z)
        north = column_at_world(world_x, world_z - 1)
        south = column_at_world(world_x, world_z + 1)
        differences = (
            abs(west.ground_height - east.ground_height),
            abs(north.ground_height - south.ground_height),
        )
        column = column_from_terrain(
            terrain_type=tile.terrain_type,
            elevation_metres=tile.elevation.metres,
            steep=max(differences) >= 3,
        )
        slope = max(differences)
        _top(
            opaque,
            x=world_x,
            y=column.ground_height,
            z=world_z,
            texture=top_texture(column.surface),
        )
        neighbours = (
            (west, 0.72, ((world_x, world_z + 1), (world_x, world_z))),
            (east, 0.88, ((world_x + 1, world_z), (world_x + 1, world_z + 1))),
            (north, 0.96, ((world_x, world_z), (world_x + 1, world_z))),
            (south, 0.66, ((world_x + 1, world_z + 1), (world_x, world_z + 1))),
        )
        for neighbour, shade, ((x1, z1), (x2, z2)) in neighbours:
            for block_y in range(neighbour.ground_height, column.ground_height):
                _quad(
                    opaque,
                    points=(
                        (x1, block_y, z1),
                        (x1, block_y + 1, z1),
                        (x2, block_y + 1, z2),
                        (x2, block_y, z2),
                    ),
                    texture=side_texture(column=column, block_y=block_y),
                    shade=shade,
                )
        if column.water is not None:
            water_texture = top_texture(column.water)
            _top(
                water,
                x=world_x,
                y=column.surface_height,
                z=world_z,
                texture=water_texture,
                shade=1.0,
            )
            for neighbour, shade, ((x1, z1), (x2, z2)) in neighbours:
                if neighbour.water is not None or neighbour.ground_height >= column.surface_height:
                    continue
                water_bottom = max(
                    column.ground_height,
                    neighbour.ground_height,
                )
                _quad(
                    water,
                    points=(
                        (x1, water_bottom, z1),
                        (x1, column.surface_height, z1),
                        (x2, column.surface_height, z2),
                        (x2, water_bottom, z2),
                    ),
                    texture=water_texture,
                    shade=shade,
                )
        scenery = scenery_at(
            seed=terrain.terrain_seed,
            world_x=world_x,
            world_z=world_z,
            column=column,
            slope=slope,
        )
        if scenery is not None:
            _add_scenery(
                opaque,
                kind=scenery.kind,
                x=world_x,
                y=column.ground_height,
                z=world_z,
            )
    opaque_count = len(opaque) // 6
    water_count = len(water) // 6
    return VoxelChunkMesh(
        coordinate=terrain.chunk_coordinate,
        opaque_vertices=opaque.tobytes(),
        water_vertices=water.tobytes(),
        opaque_vertex_count=opaque_count,
        water_vertex_count=water_count,
        triangle_count=(opaque_count + water_count) // 3,
        terrain_revision=terrain.revision,
    )
