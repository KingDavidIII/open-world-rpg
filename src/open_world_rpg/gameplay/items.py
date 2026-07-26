"""Stable inventory item identities and block-resource mappings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from open_world_rpg.world import BlockMaterial

MAX_STACK_SIZE: Final = 64


class ItemType(StrEnum):
    """Persistence-safe collectible item identities."""

    GRASS_BLOCK = "grass_block"
    DIRT_BLOCK = "dirt_block"
    STONE_BLOCK = "stone_block"
    SAND_BLOCK = "sand_block"
    SNOW_BLOCK = "snow_block"

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()


_MATERIAL_TO_ITEM: Final = {
    BlockMaterial.GRASS: ItemType.GRASS_BLOCK,
    BlockMaterial.DIRT: ItemType.DIRT_BLOCK,
    BlockMaterial.STONE: ItemType.STONE_BLOCK,
    BlockMaterial.SAND: ItemType.SAND_BLOCK,
    BlockMaterial.SNOW: ItemType.SNOW_BLOCK,
}
_ITEM_TO_MATERIAL: Final = {item: material for material, item in _MATERIAL_TO_ITEM.items()}


def item_for_material(material: BlockMaterial) -> ItemType | None:
    """Return the collectible produced by a block, if any."""
    if not isinstance(material, BlockMaterial):
        raise TypeError("material must be a BlockMaterial.")
    return _MATERIAL_TO_ITEM.get(material)


def material_for_item(item: ItemType) -> BlockMaterial:
    """Return the placeable block represented by an item."""
    if not isinstance(item, ItemType):
        raise TypeError("item must be an ItemType.")
    return _ITEM_TO_MATERIAL[item]


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class ItemStack:
    """Immutable bounded quantity of one stable item type."""

    item: ItemType
    quantity: int
    maximum: int = MAX_STACK_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.item, ItemType):
            raise TypeError("item must be an ItemType.")
        for name, value in (("quantity", self.quantity), ("maximum", self.maximum)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.quantity > self.maximum:
            raise ValueError("quantity cannot exceed the stack maximum.")

    def with_quantity(self, quantity: int) -> ItemStack:
        return ItemStack(item=self.item, quantity=quantity, maximum=self.maximum)
