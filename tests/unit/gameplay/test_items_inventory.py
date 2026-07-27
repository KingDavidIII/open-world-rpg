"""Stable item and controlled inventory contracts."""

from dataclasses import FrozenInstanceError

import pytest

from open_world_rpg.gameplay import (
    INVENTORY_CAPACITY,
    MAX_STACK_SIZE,
    InventoryAddResult,
    ItemStack,
    ItemType,
    PlayerInventory,
    PlayerInventorySnapshot,
    ToolInstance,
    create_bootstrap_inventory,
    item_for_material,
    material_for_item,
)
from open_world_rpg.world import BlockMaterial


def test_item_values_display_and_block_mapping() -> None:
    assert tuple(item.value for item in ItemType) == (
        "grass_block",
        "dirt_block",
        "stone_block",
        "sand_block",
        "snow_block",
        "wood_log",
        "wood_plank",
        "stick",
        "wooden_pickaxe",
        "stone_pickaxe",
        "wooden_shovel",
        "stone_shovel",
    )
    assert ItemType.GRASS_BLOCK.display_name == "Grass Block"
    for material, item in zip(
        (
            BlockMaterial.GRASS,
            BlockMaterial.DIRT,
            BlockMaterial.STONE,
            BlockMaterial.SAND,
            BlockMaterial.SNOW,
        ),
        tuple(ItemType)[:5],
        strict=True,
    ):
        assert item_for_material(material) is item
        assert material_for_item(item) is material
    assert item_for_material(BlockMaterial.AIR) is None
    assert item_for_material(BlockMaterial.WATER) is None
    with pytest.raises(TypeError):
        item_for_material("stone")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        material_for_item("stone_block")  # type: ignore[arg-type]


def test_item_stack_validation_ordering_and_immutability() -> None:
    stack = ItemStack(item=ItemType.STONE_BLOCK, quantity=1)
    assert stack.maximum == MAX_STACK_SIZE
    assert stack.with_quantity(2) > stack
    with pytest.raises(FrozenInstanceError):
        stack.quantity = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        ItemStack(item="stone", quantity=1)  # type: ignore[arg-type]
    for value in (True, 1.5, "1"):
        with pytest.raises(TypeError):
            ItemStack(item=ItemType.STONE_BLOCK, quantity=value)  # type: ignore[arg-type]
    for value in (0, -1):
        with pytest.raises(ValueError):
            ItemStack(item=ItemType.STONE_BLOCK, quantity=value)
    with pytest.raises(ValueError):
        ItemStack(item=ItemType.STONE_BLOCK, quantity=65)
    with pytest.raises(TypeError):
        ItemStack(item=ItemType.STONE_BLOCK, quantity=1, maximum=True)
    with pytest.raises(ValueError):
        ItemStack(item=ItemType.STONE_BLOCK, quantity=1, maximum=0)
    with pytest.raises(ValueError):
        ItemStack(item=ItemType.STONE_BLOCK, quantity=2, maximum=1)


def test_inventory_first_fit_overflow_and_removal_are_atomic() -> None:
    inventory = PlayerInventory()
    assert inventory.slots() == (None,) * INVENTORY_CAPACITY
    assert inventory.add(ItemType.STONE_BLOCK, 65).accepted == 65
    assert inventory.slot(0) == ItemStack(item=ItemType.STONE_BLOCK, quantity=64)
    assert inventory.slot(1) == ItemStack(item=ItemType.STONE_BLOCK, quantity=1)
    inventory.set_slot(0, ItemStack(item=ItemType.STONE_BLOCK, quantity=60))
    assert inventory.add(ItemType.STONE_BLOCK, 5).accepted == 5
    assert inventory.slot(0).quantity == 64  # type: ignore[union-attr]
    assert inventory.slot(1).quantity == 2  # type: ignore[union-attr]
    before = inventory.snapshot()
    assert not inventory.remove(ItemType.STONE_BLOCK, 100)
    assert inventory.snapshot() == before
    assert inventory.remove(ItemType.STONE_BLOCK, 64)
    assert inventory.total_quantity(ItemType.STONE_BLOCK) == 2
    assert inventory.remove_from_slot(1, 1)
    assert inventory.slot(1).quantity == 1  # type: ignore[union-attr]
    assert inventory.remove_from_slot(1, 1)
    assert inventory.slot(1) is None
    assert not inventory.remove_from_slot(1, 1)


def test_inventory_full_remainder_contains_clear_and_revision() -> None:
    inventory = PlayerInventory()
    for index in range(INVENTORY_CAPACITY):
        assert inventory.set_slot(index, ItemStack(item=ItemType.DIRT_BLOCK, quantity=64))
    revision = inventory.revision
    result = inventory.add(ItemType.STONE_BLOCK, 3)
    assert (result.accepted, result.remainder) == (0, 3)
    assert inventory.revision == revision
    assert inventory.contains(ItemType.DIRT_BLOCK, INVENTORY_CAPACITY * 64)
    assert not inventory.contains(ItemType.STONE_BLOCK, 1)
    assert inventory.occupied_slots == INVENTORY_CAPACITY
    assert inventory.total_items == INVENTORY_CAPACITY * 64
    assert inventory.clear()
    assert not inventory.clear()


def test_inventory_selection_snapshot_restore_and_noops() -> None:
    inventory = create_bootstrap_inventory()
    assert inventory.slot(0) == ToolInstance.create(ItemType.WOODEN_PICKAXE)
    assert inventory.slot(1) == ToolInstance.create(ItemType.WOODEN_SHOVEL)
    assert [inventory.slot(index).quantity for index in range(2, 7)] == [8, 8, 8, 4, 4]  # type: ignore[union-attr]
    revision = inventory.revision
    assert not inventory.select_hotbar(0)
    assert inventory.revision == revision
    assert inventory.select_hotbar(2)
    assert inventory.selected_stack == ItemStack(item=ItemType.GRASS_BLOCK, quantity=8)
    assert inventory.cycle_hotbar(1)
    assert inventory.selected_hotbar_index == 1
    assert inventory.cycle_hotbar(-1)
    assert inventory.selected_hotbar_index == 2
    assert not inventory.cycle_hotbar(0)
    snapshot = inventory.snapshot()
    restored = PlayerInventory.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot
    empty = create_bootstrap_inventory(enabled=False)
    assert empty.restore(snapshot)
    assert empty.snapshot() == snapshot
    assert not empty.restore(snapshot)


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_inventory_rejects_invalid_quantities_and_indexes(value: object) -> None:
    inventory = PlayerInventory()
    with pytest.raises(TypeError):
        inventory.add(ItemType.DIRT_BLOCK, value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        inventory.slot(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        inventory.cycle_hotbar(value)  # type: ignore[arg-type]


def test_inventory_validation_branches() -> None:
    inventory = PlayerInventory()
    with pytest.raises(TypeError):
        inventory.add("dirt", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        inventory.remove("dirt", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        inventory.total_quantity("dirt")  # type: ignore[arg-type]
    for value in (0, -1):
        with pytest.raises(ValueError):
            inventory.add(ItemType.DIRT_BLOCK, value)
    for value in (-1, 27):
        with pytest.raises(IndexError):
            inventory.slot(value)
    with pytest.raises(IndexError):
        inventory.select_hotbar(9)
    with pytest.raises(TypeError):
        inventory.set_slot(0, "stack")  # type: ignore[arg-type]
    assert inventory.set_slot(0, ItemStack(item=ItemType.DIRT_BLOCK, quantity=1))
    assert not inventory.set_slot(0, inventory.slot(0))
    with pytest.raises(TypeError):
        InventoryAddResult(accepted=True, remainder=0)
    with pytest.raises(ValueError):
        InventoryAddResult(accepted=0, remainder=-1)
    inventory.set_slot(0, None)
    inventory.set_slot(1, ItemStack(item=ItemType.DIRT_BLOCK, quantity=1))
    inventory.set_slot(2, ItemStack(item=ItemType.STONE_BLOCK, quantity=2))
    assert inventory.remove(ItemType.STONE_BLOCK, 1)

    valid = PlayerInventorySnapshot(revision=0, selected_hotbar_index=0, slots=(None,) * 27)
    with pytest.raises(TypeError):
        PlayerInventory.from_snapshot("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PlayerInventorySnapshot(revision=True, selected_hotbar_index=0, slots=(None,) * 27)
    with pytest.raises(ValueError):
        PlayerInventorySnapshot(revision=-1, selected_hotbar_index=0, slots=(None,) * 27)
    with pytest.raises(TypeError):
        PlayerInventorySnapshot(revision=0, selected_hotbar_index=True, slots=(None,) * 27)
    with pytest.raises(ValueError):
        PlayerInventorySnapshot(revision=0, selected_hotbar_index=9, slots=(None,) * 27)
    with pytest.raises(TypeError):
        PlayerInventorySnapshot(revision=0, selected_hotbar_index=0, slots=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PlayerInventorySnapshot(revision=0, selected_hotbar_index=0, slots=(None,))
    with pytest.raises(TypeError):
        PlayerInventorySnapshot(
            revision=0,
            selected_hotbar_index=0,
            slots=("bad", *valid.slots[1:]),  # type: ignore[arg-type]
        )
