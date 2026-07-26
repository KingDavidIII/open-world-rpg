"""Stable item identities, catalogue policy, and inventory values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from open_world_rpg.world import BlockMaterial

MAX_STACK_SIZE: Final = 64


class ItemType(StrEnum):
    """Persistence-safe item identities."""

    GRASS_BLOCK = "grass_block"
    DIRT_BLOCK = "dirt_block"
    STONE_BLOCK = "stone_block"
    SAND_BLOCK = "sand_block"
    SNOW_BLOCK = "snow_block"
    WOODEN_PICKAXE = "wooden_pickaxe"
    STONE_PICKAXE = "stone_pickaxe"
    WOODEN_SHOVEL = "wooden_shovel"
    STONE_SHOVEL = "stone_shovel"

    @property
    def display_name(self) -> str:
        return item_policy(self).display_name


class ToolClassification(StrEnum):
    PICKAXE = "pickaxe"
    SHOVEL = "shovel"


class ToolTier(StrEnum):
    WOOD = "wood"
    STONE = "stone"


@dataclass(frozen=True, slots=True, kw_only=True)
class ItemPolicy:
    """Renderer-independent policy for one stable item identity."""

    stackable: bool
    maximum_stack_size: int
    placeable_material: BlockMaterial | None
    tool_classification: ToolClassification | None
    tool_tier: ToolTier | None
    maximum_durability: int | None
    display_name: str

    @property
    def placeable(self) -> bool:
        return self.placeable_material is not None


def _block_policy(name: str, material: BlockMaterial) -> ItemPolicy:
    return ItemPolicy(
        stackable=True,
        maximum_stack_size=MAX_STACK_SIZE,
        placeable_material=material,
        tool_classification=None,
        tool_tier=None,
        maximum_durability=None,
        display_name=name,
    )


def _tool_policy(
    name: str, classification: ToolClassification, tier: ToolTier, durability: int
) -> ItemPolicy:
    return ItemPolicy(
        stackable=False,
        maximum_stack_size=1,
        placeable_material=None,
        tool_classification=classification,
        tool_tier=tier,
        maximum_durability=durability,
        display_name=name,
    )


_ITEM_POLICIES: Final = {
    ItemType.GRASS_BLOCK: _block_policy("Grass Block", BlockMaterial.GRASS),
    ItemType.DIRT_BLOCK: _block_policy("Dirt Block", BlockMaterial.DIRT),
    ItemType.STONE_BLOCK: _block_policy("Stone Block", BlockMaterial.STONE),
    ItemType.SAND_BLOCK: _block_policy("Sand Block", BlockMaterial.SAND),
    ItemType.SNOW_BLOCK: _block_policy("Snow Block", BlockMaterial.SNOW),
    ItemType.WOODEN_PICKAXE: _tool_policy(
        "Wooden Pickaxe", ToolClassification.PICKAXE, ToolTier.WOOD, 64
    ),
    ItemType.STONE_PICKAXE: _tool_policy(
        "Stone Pickaxe", ToolClassification.PICKAXE, ToolTier.STONE, 128
    ),
    ItemType.WOODEN_SHOVEL: _tool_policy(
        "Wooden Shovel", ToolClassification.SHOVEL, ToolTier.WOOD, 64
    ),
    ItemType.STONE_SHOVEL: _tool_policy(
        "Stone Shovel", ToolClassification.SHOVEL, ToolTier.STONE, 128
    ),
}


def item_policy(item: ItemType) -> ItemPolicy:
    if not isinstance(item, ItemType):
        raise TypeError("item must be an ItemType.")
    return _ITEM_POLICIES[item]


def item_for_material(material: BlockMaterial) -> ItemType | None:
    """Return the collectible produced by a block, if any."""
    if not isinstance(material, BlockMaterial):
        raise TypeError("material must be a BlockMaterial.")
    return next(
        (item for item, policy in _ITEM_POLICIES.items() if policy.placeable_material is material),
        None,
    )


def material_for_item(item: ItemType) -> BlockMaterial:
    """Return the placeable block represented by an item."""
    material = item_policy(item).placeable_material
    if material is None:
        raise ValueError("item is not placeable.")
    return material


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class ItemStack:
    """Immutable bounded quantity of one stackable item."""

    item: ItemType
    quantity: int
    maximum: int = MAX_STACK_SIZE

    def __post_init__(self) -> None:
        policy = item_policy(self.item)
        if not policy.stackable:
            raise ValueError("tools cannot be represented by ItemStack.")
        for name, value in (("quantity", self.quantity), ("maximum", self.maximum)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.maximum != policy.maximum_stack_size:
            raise ValueError("maximum must match the item catalogue.")
        if self.quantity > self.maximum:
            raise ValueError("quantity cannot exceed the stack maximum.")

    def with_quantity(self, quantity: int) -> ItemStack:
        return ItemStack(item=self.item, quantity=quantity, maximum=self.maximum)


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class ToolInstance:
    """One non-stackable tool with immutable durability."""

    item: ItemType
    current_durability: int
    maximum_durability: int

    def __post_init__(self) -> None:
        policy = item_policy(self.item)
        if policy.maximum_durability is None:
            raise ValueError("item must identify a tool.")
        for name, value in (
            ("current_durability", self.current_durability),
            ("maximum_durability", self.maximum_durability),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.maximum_durability != policy.maximum_durability:
            raise ValueError("maximum_durability must match the item catalogue.")
        if self.current_durability > self.maximum_durability:
            raise ValueError("current durability cannot exceed maximum durability.")

    @classmethod
    def create(cls, item: ItemType) -> ToolInstance:
        maximum = item_policy(item).maximum_durability
        if maximum is None:
            raise ValueError("item must identify a tool.")
        return cls(item=item, current_durability=maximum, maximum_durability=maximum)

    def use(self) -> ToolInstance | None:
        if self.current_durability == 1:
            return None
        return ToolInstance(
            item=self.item,
            current_durability=self.current_durability - 1,
            maximum_durability=self.maximum_durability,
        )
