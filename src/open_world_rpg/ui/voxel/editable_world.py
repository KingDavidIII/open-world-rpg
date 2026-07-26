"""Authoritative resolution of generated terrain plus player block edits."""

from __future__ import annotations

from collections.abc import Callable

from open_world_rpg.world import (
    BlockEditStore,
    BlockMaterial,
    WorldBlockCoordinate,
)

from .blocks import MAX_DISPLAY_HEIGHT, BlockColumn

MIN_EDITABLE_BLOCK_Y = 0
MAX_EDITABLE_BLOCK_Y = MAX_DISPLAY_HEIGHT

ColumnLookup = Callable[[int, int], BlockColumn]


class EditableVoxelWorld:
    """Resolve immutable generated columns through a mutable edit overlay."""

    def __init__(self, *, column_at: ColumnLookup, edits: BlockEditStore) -> None:
        if not callable(column_at):
            raise TypeError("column_at must be callable.")
        if not isinstance(edits, BlockEditStore):
            raise TypeError("edits must be a BlockEditStore.")
        self._column_at = column_at
        self._edits = edits

    @property
    def edits(self) -> BlockEditStore:
        return self._edits

    def block_at(self, coordinate: WorldBlockCoordinate) -> BlockMaterial:
        if not isinstance(coordinate, WorldBlockCoordinate):
            raise TypeError("coordinate must be a WorldBlockCoordinate.")
        edit = self._edits.get(coordinate)
        if edit is not None:
            return edit.material
        column = self._column_at(coordinate.x, coordinate.z)
        if coordinate.y <= column.ground_height:
            if coordinate.y == column.ground_height:
                return column.surface
            return (
                column.subsurface
                if coordinate.y >= column.ground_height - 3
                else BlockMaterial.STONE
            )
        if column.water is not None and coordinate.y < column.surface_height:
            return BlockMaterial.WATER
        return BlockMaterial.AIR

    def material_at(self, x: int, y: int, z: int) -> BlockMaterial:
        return self.block_at(WorldBlockCoordinate(x=x, y=y, z=z))

    def solid_at(self, x: int, y: int, z: int) -> bool:
        return self.material_at(x, y, z).is_solid

    @staticmethod
    def supports(coordinate: WorldBlockCoordinate) -> bool:
        if not isinstance(coordinate, WorldBlockCoordinate):
            raise TypeError("coordinate must be a WorldBlockCoordinate.")
        return MIN_EDITABLE_BLOCK_Y <= coordinate.y <= MAX_EDITABLE_BLOCK_Y
