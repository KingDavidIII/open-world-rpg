"""Atomic inventory transfer and crafting catalogue coverage."""

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from open_world_rpg.gameplay import (
    DEFAULT_RECIPES,
    CraftingCatalogue,
    CraftingRecipe,
    CraftingResult,
    CraftingService,
    InventoryTransferResult,
    ItemStack,
    ItemType,
    PlayerInventory,
    RecipeIngredient,
    ToolInstance,
)


def stocked_inventory() -> PlayerInventory:
    inventory = PlayerInventory()
    inventory.add(ItemType.WOOD_LOG, 4)
    inventory.add(ItemType.WOOD_PLANK, 16)
    inventory.add(ItemType.STICK, 16)
    inventory.add(ItemType.STONE_BLOCK, 16)
    return inventory


def test_resource_items_and_default_recipe_catalogue_are_stable() -> None:
    assert ItemType.WOOD_LOG.display_name == "Wood Log"
    assert ItemType.WOOD_PLANK.display_name == "Wood Plank"
    assert ItemType.STICK.display_name == "Stick"
    catalogue = CraftingCatalogue()
    assert catalogue.recipes == DEFAULT_RECIPES
    assert tuple(recipe.identifier for recipe in catalogue.recipes) == (
        "wood_planks",
        "sticks",
        "wooden_pickaxe",
        "wooden_shovel",
        "stone_pickaxe",
        "stone_shovel",
    )
    assert catalogue.recipe("stone_pickaxe") is DEFAULT_RECIPES[4]
    assert catalogue.recipe("missing") is None
    with pytest.raises(TypeError):
        catalogue.recipe(1)  # type: ignore[arg-type]


def test_recipe_values_validate_and_are_immutable() -> None:
    ingredient = RecipeIngredient(item=ItemType.STICK, quantity=2)
    recipe = CraftingRecipe(
        identifier="test",
        display_name="Test Stack",
        ingredients=(ingredient,),
        output_item=ItemType.WOOD_PLANK,
        output_quantity=2,
    )
    assert recipe.output_label == "Wood Plank x2"
    assert DEFAULT_RECIPES[2].output_label == "Wooden Pickaxe"
    with pytest.raises(FrozenInstanceError):
        ingredient.quantity = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="stackable"):
        RecipeIngredient(item=ItemType.WOODEN_PICKAXE, quantity=1)
    with pytest.raises(TypeError):
        RecipeIngredient(item=ItemType.STICK, quantity=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RecipeIngredient(item=ItemType.STICK, quantity=0)
    with pytest.raises(TypeError):
        CraftingRecipe(
            identifier=cast(Any, 1),
            display_name="Bad",
            ingredients=(ingredient,),
            output_item=ItemType.STICK,
        )
    for identifier in ("", " bad", "bad "):
        with pytest.raises(ValueError):
            CraftingRecipe(
                identifier=identifier,
                display_name="Bad",
                ingredients=(ingredient,),
                output_item=ItemType.STICK,
            )
    with pytest.raises(TypeError):
        CraftingRecipe(
            identifier="bad",
            display_name=cast(Any, 1),
            ingredients=(ingredient,),
            output_item=ItemType.STICK,
        )
    with pytest.raises(ValueError):
        CraftingRecipe(
            identifier="bad",
            display_name=" ",
            ingredients=(ingredient,),
            output_item=ItemType.STICK,
        )
    with pytest.raises(TypeError):
        CraftingRecipe(
            identifier="bad",
            display_name="Bad",
            ingredients=cast(Any, [ingredient]),
            output_item=ItemType.STICK,
        )
    with pytest.raises(ValueError):
        CraftingRecipe(
            identifier="bad",
            display_name="Bad",
            ingredients=(),
            output_item=ItemType.STICK,
        )
    with pytest.raises(ValueError, match="repeat"):
        CraftingRecipe(
            identifier="bad",
            display_name="Bad",
            ingredients=(ingredient, ingredient),
            output_item=ItemType.STICK,
        )
    with pytest.raises(TypeError):
        CraftingRecipe(
            identifier="bad",
            display_name="Bad",
            ingredients=(ingredient,),
            output_item=ItemType.STICK,
            output_quantity=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        CraftingRecipe(
            identifier="bad",
            display_name="Bad",
            ingredients=(ingredient,),
            output_item=ItemType.STICK,
            output_quantity=0,
        )
    with pytest.raises(ValueError, match="exactly one"):
        CraftingRecipe(
            identifier="bad",
            display_name="Bad",
            ingredients=(ingredient,),
            output_item=ItemType.WOODEN_PICKAXE,
            output_quantity=2,
        )
    with pytest.raises(ValueError, match="stack maximum"):
        CraftingRecipe(
            identifier="bad",
            display_name="Bad",
            ingredients=(ingredient,),
            output_item=ItemType.STICK,
            output_quantity=65,
        )


def test_catalogue_validation_and_crafting_service_type_guards() -> None:
    with pytest.raises(TypeError):
        CraftingCatalogue(cast(Any, []))
    with pytest.raises(ValueError):
        CraftingCatalogue(())
    with pytest.raises(TypeError):
        CraftingCatalogue((cast(Any, "bad"),))
    with pytest.raises(ValueError, match="unique"):
        CraftingCatalogue((DEFAULT_RECIPES[0], DEFAULT_RECIPES[0]))
    with pytest.raises(TypeError):
        CraftingService(cast(Any, object()))
    service = CraftingService()
    with pytest.raises(TypeError):
        service.can_craft(cast(Any, object()), DEFAULT_RECIPES[0])
    with pytest.raises(TypeError):
        service.can_craft(PlayerInventory(), cast(Any, object()))
    with pytest.raises(TypeError):
        service.craft(cast(Any, object()), "sticks")


def test_crafting_is_atomic_reports_failures_and_increments_once() -> None:
    service = CraftingService()
    inventory = stocked_inventory()
    revision = inventory.revision
    attempt = service.craft(inventory, "wooden_pickaxe")
    assert attempt.crafted
    assert attempt.result is CraftingResult.CRAFTED
    assert inventory.revision == revision + 1
    assert inventory.total_quantity(ItemType.WOOD_PLANK) == 13
    assert inventory.total_quantity(ItemType.STICK) == 14
    assert any(
        isinstance(slot, ToolInstance) and slot.item is ItemType.WOODEN_PICKAXE
        for slot in inventory.slots()
    )

    unknown_before = inventory.snapshot()
    unknown = service.craft(inventory, "unknown")
    assert unknown.result is CraftingResult.UNKNOWN_RECIPE
    assert unknown.recipe is None
    assert inventory.snapshot() == unknown_before

    empty = PlayerInventory()
    missing = service.craft(empty, "stone_pickaxe")
    assert missing.result is CraftingResult.MISSING_INGREDIENTS
    assert not missing.crafted
    assert not service.can_craft(empty, DEFAULT_RECIPES[4])

    full = stocked_inventory()
    for index in range(4, 27):
        full.set_slot(index, ToolInstance.create(ItemType.WOODEN_SHOVEL))
    before = full.snapshot()
    result = service.craft(full, "stone_pickaxe")
    assert result.result is CraftingResult.INVENTORY_FULL
    assert full.snapshot() == before


def test_stack_outputs_merge_and_tool_outputs_cover_every_default_recipe() -> None:
    service = CraftingService()
    inventory = stocked_inventory()
    before_logs = inventory.total_quantity(ItemType.WOOD_LOG)
    assert service.craft(inventory, "wood_planks").crafted
    assert inventory.total_quantity(ItemType.WOOD_LOG) == before_logs - 1
    assert inventory.total_quantity(ItemType.WOOD_PLANK) == 20
    assert service.craft(inventory, "sticks").crafted
    assert inventory.total_quantity(ItemType.STICK) == 20
    for identifier in (
        "wooden_pickaxe",
        "wooden_shovel",
        "stone_pickaxe",
        "stone_shovel",
    ):
        assert service.craft(inventory, identifier).crafted


def test_slot_transfer_supports_move_split_merge_swap_and_tools() -> None:
    inventory = PlayerInventory()
    inventory.set_slot(0, ItemStack(item=ItemType.STONE_BLOCK, quantity=10))
    moved = inventory.move_slot(0, 9, quantity=4)
    assert moved == InventoryTransferResult(
        changed=True,
        moved_quantity=4,
        message="Stack moved",
    )
    assert inventory.slot(0) == ItemStack(item=ItemType.STONE_BLOCK, quantity=6)
    assert inventory.slot(9) == ItemStack(item=ItemType.STONE_BLOCK, quantity=4)
    merged = inventory.move_slot(0, 9)
    assert merged.message == "Stacks merged"
    assert inventory.slot(0) is None
    assert inventory.slot(9) == ItemStack(item=ItemType.STONE_BLOCK, quantity=10)

    inventory.set_slot(0, ItemStack(item=ItemType.DIRT_BLOCK, quantity=2))
    swapped = inventory.move_slot(0, 9)
    assert swapped.swapped
    assert inventory.slot(0) == ItemStack(item=ItemType.STONE_BLOCK, quantity=10)
    assert inventory.slot(9) == ItemStack(item=ItemType.DIRT_BLOCK, quantity=2)

    tool = ToolInstance.create(ItemType.WOODEN_PICKAXE)
    inventory.set_slot(1, tool)
    assert inventory.move_slot(1, 10).message == "Tool moved"
    inventory.set_slot(11, ToolInstance.create(ItemType.WOODEN_SHOVEL))
    assert inventory.move_slot(10, 11).swapped
    assert not inventory.move_slot(11, 12, quantity=2).changed


def test_slot_transfer_and_quick_move_noop_paths() -> None:
    inventory = PlayerInventory()
    assert inventory.move_slot(0, 0).message == "Selection cleared"
    assert inventory.move_slot(0, 1).message == "Source slot is empty"
    inventory.set_slot(0, ItemStack(item=ItemType.STONE_BLOCK, quantity=64))
    inventory.set_slot(1, ItemStack(item=ItemType.STONE_BLOCK, quantity=64))
    assert inventory.move_slot(0, 1).message == "Destination stack is full"
    inventory.set_slot(1, ItemStack(item=ItemType.DIRT_BLOCK, quantity=1))
    assert inventory.move_slot(0, 1, quantity=2).message.startswith("Partial")
    assert inventory.quick_move(2).message == "Source slot is empty"

    inventory = PlayerInventory()
    inventory.set_slot(0, ItemStack(item=ItemType.STONE_BLOCK, quantity=10))
    inventory.set_slot(9, ItemStack(item=ItemType.STONE_BLOCK, quantity=60))
    result = inventory.quick_move(0)
    assert result.changed
    assert inventory.slot(9) == ItemStack(item=ItemType.STONE_BLOCK, quantity=64)
    assert inventory.slot(10) == ItemStack(item=ItemType.STONE_BLOCK, quantity=6)
    assert inventory.quick_move(10).changed

    full = PlayerInventory()
    full.set_slot(0, ItemStack(item=ItemType.STONE_BLOCK, quantity=1))
    for index in range(9, 27):
        full.set_slot(index, ToolInstance.create(ItemType.WOODEN_SHOVEL))
    assert full.quick_move(0).message == "Destination section is full"


def test_transfer_result_validation() -> None:
    with pytest.raises(TypeError):
        InventoryTransferResult(changed=cast(Any, 1))
    with pytest.raises(TypeError):
        InventoryTransferResult(changed=False, moved_quantity=cast(Any, True))
    with pytest.raises(ValueError):
        InventoryTransferResult(changed=False, moved_quantity=-1)
    with pytest.raises(TypeError):
        InventoryTransferResult(changed=False, swapped=cast(Any, 1))
    with pytest.raises(TypeError):
        InventoryTransferResult(changed=False, message=cast(Any, 1))


def test_transfer_input_guards_tool_quick_move_and_partial_section_capacity() -> None:
    inventory = PlayerInventory()
    with pytest.raises(TypeError):
        inventory.move_slot(cast(Any, True), 1)
    with pytest.raises(IndexError):
        inventory.move_slot(-1, 1)
    with pytest.raises(TypeError):
        inventory.move_slot(0, 1, quantity=cast(Any, True))
    with pytest.raises(ValueError):
        inventory.move_slot(0, 1, quantity=0)
    with pytest.raises(IndexError):
        inventory.quick_move(27)

    tool = ToolInstance.create(ItemType.STONE_PICKAXE)
    inventory.set_slot(0, tool)
    revision = inventory.revision
    result = inventory.quick_move(0)
    assert result == InventoryTransferResult(
        changed=True,
        moved_quantity=1,
        message="Quick moved",
    )
    assert inventory.slot(0) is None
    assert inventory.slot(9) == tool
    assert inventory.revision == revision + 1

    constrained = PlayerInventory()
    constrained.set_slot(0, ItemStack(item=ItemType.STONE_BLOCK, quantity=10))
    constrained.set_slot(9, ItemStack(item=ItemType.STONE_BLOCK, quantity=60))
    for index in range(10, 27):
        constrained.set_slot(index, ToolInstance.create(ItemType.WOODEN_SHOVEL))
    before_revision = constrained.revision
    partial = constrained.quick_move(0)
    assert partial.changed
    assert partial.moved_quantity == 4
    assert constrained.slot(0) == ItemStack(item=ItemType.STONE_BLOCK, quantity=6)
    assert constrained.slot(9) == ItemStack(item=ItemType.STONE_BLOCK, quantity=64)
    assert constrained.revision == before_revision + 1


def test_transfer_result_rejects_blank_message() -> None:
    with pytest.raises(ValueError):
        InventoryTransferResult(changed=False, message=" ")


def test_quick_move_exact_merge_and_tool_scans_occupied_destinations() -> None:
    exact = PlayerInventory()
    exact.set_slot(0, ItemStack(item=ItemType.STONE_BLOCK, quantity=4))
    exact.set_slot(9, ItemStack(item=ItemType.STONE_BLOCK, quantity=60))
    result = exact.quick_move(0)
    assert result.moved_quantity == 4
    assert exact.slot(0) is None
    assert exact.slot(9) == ItemStack(item=ItemType.STONE_BLOCK, quantity=64)

    tool_inventory = PlayerInventory()
    tool_inventory.set_slot(0, ToolInstance.create(ItemType.STONE_PICKAXE))
    tool_inventory.set_slot(9, ItemStack(item=ItemType.DIRT_BLOCK, quantity=1))
    assert tool_inventory.quick_move(0).changed
    assert isinstance(tool_inventory.slot(10), ToolInstance)

    blocked = PlayerInventory()
    blocked.set_slot(0, ToolInstance.create(ItemType.STONE_PICKAXE))
    for index in range(9, 27):
        blocked.set_slot(index, ItemStack(item=ItemType.DIRT_BLOCK, quantity=1))
    assert blocked.quick_move(0).message == "Destination section is full"
