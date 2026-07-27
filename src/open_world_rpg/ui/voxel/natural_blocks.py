"""Deterministic block-aligned natural features used by rendering and gameplay."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from open_world_rpg.world import BlockMaterial, WorldBlockCoordinate

from .blocks import BlockColumn
from .scenery import SceneryKind, scenery_at

ColumnLookup = Callable[[int, int], BlockColumn]
TerrainSeedLookup = Callable[[int, int], int]


@dataclass(frozen=True, slots=True, kw_only=True)
class TreeShape:
    """One Minecraft-style tree expressed as stable block coordinates."""

    trunk: tuple[WorldBlockCoordinate, ...]
    leaves: tuple[WorldBlockCoordinate, ...]

    @property
    def blocks(self) -> dict[WorldBlockCoordinate, BlockMaterial]:
        result = dict.fromkeys(self.leaves, BlockMaterial.LEAVES)
        result.update(dict.fromkeys(self.trunk, BlockMaterial.WOOD))
        return result


def tree_shape(*, world_x: int, ground_y: int, world_z: int) -> TreeShape:
    """Return a compact block-aligned tree with a four-block trunk."""
    trunk = tuple(
        WorldBlockCoordinate(x=world_x, y=ground_y + offset, z=world_z) for offset in range(1, 5)
    )
    leaves: set[WorldBlockCoordinate] = set()
    for y in (ground_y + 4, ground_y + 5):
        for x in range(world_x - 1, world_x + 2):
            for z in range(world_z - 1, world_z + 2):
                leaves.add(WorldBlockCoordinate(x=x, y=y, z=z))
    leaves.add(WorldBlockCoordinate(x=world_x, y=ground_y + 6, z=world_z))
    leaves.difference_update(trunk)
    return TreeShape(
        trunk=trunk,
        leaves=tuple(sorted(leaves)),
    )


def natural_blocks_in_area(
    *,
    minimum_x: int,
    maximum_x: int,
    minimum_z: int,
    maximum_z: int,
    column_at: ColumnLookup,
    terrain_seed_at: TerrainSeedLookup,
) -> dict[WorldBlockCoordinate, BlockMaterial]:
    """Resolve tree blocks intersecting an inclusive horizontal area."""
    if minimum_x > maximum_x or minimum_z > maximum_z:
        raise ValueError("area bounds must be ordered.")
    result: dict[WorldBlockCoordinate, BlockMaterial] = {}
    columns: dict[tuple[int, int], BlockColumn] = {}

    def resolve_column(world_x: int, world_z: int) -> BlockColumn:
        key = (world_x, world_z)
        cached = columns.get(key)
        if cached is None:
            cached = column_at(world_x, world_z)
            columns[key] = cached
        return cached

    for centre_x in range(minimum_x - 1, maximum_x + 2):
        for centre_z in range(minimum_z - 1, maximum_z + 2):
            column = resolve_column(centre_x, centre_z)
            neighbours = (
                resolve_column(centre_x - 1, centre_z),
                resolve_column(centre_x + 1, centre_z),
                resolve_column(centre_x, centre_z - 1),
                resolve_column(centre_x, centre_z + 1),
            )
            slope = max(abs(item.ground_height - column.ground_height) for item in neighbours)
            placement = scenery_at(
                seed=terrain_seed_at(centre_x, centre_z),
                world_x=centre_x,
                world_z=centre_z,
                column=column,
                slope=slope,
            )
            if placement is None or placement.kind is not SceneryKind.TREE:
                continue
            for coordinate, material in tree_shape(
                world_x=centre_x,
                ground_y=column.ground_height,
                world_z=centre_z,
            ).blocks.items():
                if (
                    minimum_x <= coordinate.x <= maximum_x
                    and minimum_z <= coordinate.z <= maximum_z
                ):
                    result[coordinate] = material
    return result
