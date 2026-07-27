"""Texture-atlas voxel meshing with strata and separate transparent water."""

from __future__ import annotations

from array import array
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from open_world_rpg.world import (
    CHUNK_SIZE,
    BlockMaterial,
    ChunkCoordinate,
    ChunkTerrain,
    WorldBlockCoordinate,
)

from .blocks import BlockColumn, BlockType, column_from_terrain
from .editable_world import MAX_EDITABLE_BLOCK_Y
from .scenery import SceneryKind, scenery_at
from .texture_atlas import FaceTexture, atlas_uv

ColumnLookup = Callable[[int, int], BlockColumn]
BlockLookup = Callable[[int, int, int], BlockMaterial]
Point = tuple[float, float, float]
FaceBuilder = Callable[[int, int, int], tuple[Point, Point, Point, Point]]


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkMeshSnapshot:
    """Thread-safe immutable inputs for one CPU-only chunk mesh build."""

    terrain: ChunkTerrain
    columns: Mapping[tuple[int, int], BlockColumn]
    edits: Mapping[WorldBlockCoordinate, BlockMaterial]
    natural_blocks: Mapping[WorldBlockCoordinate, BlockMaterial]
    editable: bool

    def build(self) -> VoxelChunkMesh:
        def column_at(world_x: int, world_z: int) -> BlockColumn:
            return self.columns[(world_x, world_z)]

        if not self.editable:
            return build_chunk_mesh(terrain=self.terrain, column_at_world=column_at)

        def block_at(world_x: int, world_y: int, world_z: int) -> BlockMaterial:
            coordinate = WorldBlockCoordinate(x=world_x, y=world_y, z=world_z)
            edited = self.edits.get(coordinate)
            if edited is not None:
                return edited
            natural = self.natural_blocks.get(coordinate)
            if natural is not None:
                return natural
            column = column_at(world_x, world_z)
            if world_y <= column.ground_height:
                if world_y == column.ground_height:
                    return column.surface
                if world_y >= column.ground_height - 3:
                    return column.subsurface
                return BlockMaterial.STONE
            if column.water is not None and world_y < column.surface_height:
                return BlockMaterial.WATER
            return BlockMaterial.AIR

        return build_chunk_mesh(
            terrain=self.terrain,
            column_at_world=column_at,
            block_at_world=block_at,
        )


def mesh_cache_key(
    *,
    terrain: ChunkTerrain,
    neighbour_revisions: tuple[int, int, int, int],
    edit_revision: int = 0,
) -> tuple[ChunkCoordinate, int, tuple[int, int, int, int], str, int, int]:
    """Invalidate terrain, edge, format, or visual mesh-contract changes."""
    if isinstance(edit_revision, bool) or not isinstance(edit_revision, int):
        raise TypeError("edit_revision must be an integer.")
    if edit_revision < 0:
        raise ValueError("edit_revision must be non-negative.")
    return (
        terrain.chunk_coordinate,
        terrain.revision,
        neighbour_revisions,
        terrain.generation_format_version,
        3,
        edit_revision,
    )


def top_texture(block: BlockType) -> FaceTexture:
    """Select the atlas cell used for one upward face."""
    return {
        BlockType.WATER: FaceTexture.SHALLOW_WATER,
        BlockType.SAND: FaceTexture.SAND,
        BlockType.GRASS: FaceTexture.GRASS_TOP,
        BlockType.DIRT: FaceTexture.DIRT,
        BlockType.STONE: FaceTexture.STONE,
        BlockType.SNOW: FaceTexture.SNOW_TOP,
    }[block]


def _material_texture(material: BlockMaterial, *, top: bool) -> FaceTexture:
    if material is BlockMaterial.GRASS:
        return FaceTexture.GRASS_TOP if top else FaceTexture.GRASS_SIDE
    if material is BlockMaterial.SNOW:
        return FaceTexture.SNOW_TOP if top else FaceTexture.SNOW_SIDE
    if material is BlockMaterial.WOOD:
        return FaceTexture.LOG_TOP if top else FaceTexture.LOG_SIDE
    return {
        BlockMaterial.DIRT: FaceTexture.DIRT,
        BlockMaterial.STONE: FaceTexture.STONE,
        BlockMaterial.SAND: FaceTexture.SAND,
        BlockMaterial.LEAVES: FaceTexture.LEAVES,
        BlockMaterial.WATER: FaceTexture.SHALLOW_WATER,
    }[material]


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
            minimum=(x, y + 1, z),
            maximum=(x + 1, y + 5, z + 1),
            top=FaceTexture.LOG_TOP,
            side=FaceTexture.LOG_SIDE,
        )
        _cube(
            output,
            minimum=(x - 1, y + 4, z - 1),
            maximum=(x + 2, y + 6, z + 2),
            top=FaceTexture.LEAVES,
            side=FaceTexture.LEAVES,
        )
        _cube(
            output,
            minimum=(x, y + 6, z),
            maximum=(x + 1, y + 7, z + 1),
            top=FaceTexture.LEAVES,
            side=FaceTexture.LEAVES,
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


def build_chunk_mesh(
    *,
    terrain: ChunkTerrain,
    column_at_world: ColumnLookup,
    block_at_world: BlockLookup | None = None,
) -> VoxelChunkMesh:
    """Build opaque terrain first and water into an independent blend stream."""
    if not isinstance(terrain, ChunkTerrain):
        raise TypeError("terrain must be a ChunkTerrain.")
    if block_at_world is not None:
        return _build_editable_chunk_mesh(
            terrain=terrain,
            column_at_world=column_at_world,
            block_at_world=block_at_world,
        )
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


def _build_editable_chunk_mesh(
    *,
    terrain: ChunkTerrain,
    column_at_world: ColumnLookup,
    block_at_world: BlockLookup,
) -> VoxelChunkMesh:
    """Mesh the authoritative edited block resolver with hidden-face culling."""
    opaque = array("f")
    water = array("f")
    origin = terrain.chunk_coordinate.to_world_origin()
    faces: tuple[tuple[tuple[int, int, int], float, FaceBuilder], ...] = (
        (
            (-1, 0, 0),
            0.72,
            lambda x, y, z: ((x, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z), (x, y, z)),
        ),
        (
            (1, 0, 0),
            0.88,
            lambda x, y, z: (
                (x + 1, y, z),
                (x + 1, y + 1, z),
                (x + 1, y + 1, z + 1),
                (x + 1, y, z + 1),
            ),
        ),
        (
            (0, 0, -1),
            0.96,
            lambda x, y, z: ((x, y, z), (x, y + 1, z), (x + 1, y + 1, z), (x + 1, y, z)),
        ),
        (
            (0, 0, 1),
            0.66,
            lambda x, y, z: (
                (x + 1, y, z + 1),
                (x + 1, y + 1, z + 1),
                (x, y + 1, z + 1),
                (x, y, z + 1),
            ),
        ),
        (
            (0, 1, 0),
            1.08,
            lambda x, y, z: (
                (x, y + 1, z),
                (x, y + 1, z + 1),
                (x + 1, y + 1, z + 1),
                (x + 1, y + 1, z),
            ),
        ),
        (
            (0, -1, 0),
            0.58,
            lambda x, y, z: ((x, y, z + 1), (x, y, z), (x + 1, y, z), (x + 1, y, z + 1)),
        ),
    )
    for local_z in range(CHUNK_SIZE):
        for local_x in range(CHUNK_SIZE):
            world_x = origin.x + local_x
            world_z = origin.y + local_z
            column = column_at_world(world_x, world_z)
            highest = max(column.surface_height, column.ground_height, MAX_EDITABLE_BLOCK_Y)
            for y in range(0, highest + 1):
                material = block_at_world(world_x, y, world_z)
                if material is BlockMaterial.AIR:
                    continue
                output = water if material is BlockMaterial.WATER else opaque
                for (dx, dy, dz), shade, points_at in faces:
                    neighbour = block_at_world(world_x + dx, y + dy, world_z + dz)
                    visible = (
                        neighbour is BlockMaterial.AIR
                        if material is BlockMaterial.WATER
                        else neighbour in (BlockMaterial.AIR, BlockMaterial.WATER)
                    )
                    if visible:
                        _quad(
                            output,
                            points=points_at(world_x, y, world_z),
                            texture=_material_texture(material, top=dy > 0),
                            shade=shade,
                        )
            surface_material = block_at_world(world_x, column.ground_height, world_z)
            above_material = block_at_world(world_x, column.ground_height + 1, world_z)
            differences = (
                abs(
                    column_at_world(world_x - 1, world_z).ground_height
                    - column_at_world(world_x + 1, world_z).ground_height
                ),
                abs(
                    column_at_world(world_x, world_z - 1).ground_height
                    - column_at_world(world_x, world_z + 1).ground_height
                ),
            )
            scenery = (
                scenery_at(
                    seed=terrain.terrain_seed,
                    world_x=world_x,
                    world_z=world_z,
                    column=column,
                    slope=max(differences),
                )
                if surface_material is column.surface and above_material is BlockMaterial.AIR
                else None
            )
            if scenery is not None and scenery.kind is not SceneryKind.TREE:
                _add_scenery(
                    opaque,
                    kind=scenery.kind,
                    x=world_x,
                    y=column.ground_height + 1,
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
