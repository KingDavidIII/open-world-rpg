"""Playable wood-to-stone progression and first-run guidance coverage."""

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from open_world_rpg.gameplay import (
    GUIDE_PAGES,
    STONE_BLOCK_TARGET,
    WOOD_LOG_TARGET,
    ItemStack,
    ItemType,
    PlayerInventory,
    ProgressionObjective,
    ProgressionStage,
    SurvivalProgression,
    SurvivalProgressionSnapshot,
    ToolInstance,
)


def test_progression_snapshot_objective_and_guide_validation() -> None:
    snapshot = SurvivalProgressionSnapshot()
    assert snapshot.stage is ProgressionStage.COLLECT_WOOD
    assert not snapshot.guide_completed
    assert snapshot.revision == 0
    with pytest.raises(FrozenInstanceError):
        snapshot.revision = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        SurvivalProgressionSnapshot(stage=cast(Any, "wood"))
    with pytest.raises(TypeError):
        SurvivalProgressionSnapshot(guide_completed=cast(Any, 1))
    with pytest.raises(TypeError):
        SurvivalProgressionSnapshot(revision=cast(Any, True))
    with pytest.raises(ValueError):
        SurvivalProgressionSnapshot(revision=-1)

    objective = ProgressionObjective(title="Title", instruction="Do it", progress="0/1")
    assert objective.title == "Title"
    for name in ("title", "instruction", "progress"):
        values = {"title": "Title", "instruction": "Do it", "progress": "0/1"}
        values[name] = cast(Any, 1)
        with pytest.raises(TypeError):
            ProgressionObjective(**values)
        values[name] = " "
        with pytest.raises(ValueError):
            ProgressionObjective(**values)

    progression = SurvivalProgression()
    assert progression.guide_page == GUIDE_PAGES[0]
    assert progression.guide_page_index == 0
    assert not progression.next_guide_page()
    assert progression.guide_page == GUIDE_PAGES[1]
    assert not progression.next_guide_page()
    assert progression.next_guide_page()
    assert progression.guide_completed
    assert progression.next_guide_page()
    assert not progression.dismiss_guide()

    skipped = SurvivalProgression()
    assert skipped.dismiss_guide()
    assert skipped.guide_completed
    with pytest.raises(TypeError):
        SurvivalProgression(cast(Any, object()))


def test_progression_advances_through_the_playable_loop_once() -> None:
    inventory = PlayerInventory()
    progression = SurvivalProgression()
    objective = progression.objective(inventory)
    assert objective.title == "Gather wood"
    assert objective.progress == f"0/{WOOD_LOG_TARGET} logs"

    inventory.add(ItemType.WOOD_LOG, 2)
    assert not progression.record_pickup(ItemType.WOOD_LOG, inventory)
    inventory.add(ItemType.WOOD_LOG, 1)
    assert progression.record_pickup(ItemType.WOOD_LOG, inventory)
    assert progression.stage is ProgressionStage.CRAFT_PLANKS
    assert not progression.record_pickup(ItemType.WOOD_LOG, inventory)

    inventory.add(ItemType.WOOD_PLANK, 4)
    assert not progression.record_craft(ItemType.STICK, inventory)
    assert progression.record_craft(ItemType.WOOD_PLANK, inventory)
    assert progression.objective(inventory).title == "Craft sticks"

    inventory.add(ItemType.STICK, 4)
    assert progression.record_craft(ItemType.STICK, inventory)
    assert progression.stage is ProgressionStage.CRAFT_WOODEN_PICKAXE
    assert not progression.record_craft(ItemType.WOODEN_PICKAXE, inventory)

    inventory.add_tool(ToolInstance.create(ItemType.WOODEN_PICKAXE))
    assert progression.record_craft(ItemType.WOODEN_PICKAXE, inventory)
    assert progression.stage is ProgressionStage.COLLECT_STONE
    assert progression.recipe_unlocked("stone_pickaxe")

    inventory.add(ItemType.STONE_BLOCK, STONE_BLOCK_TARGET - 1)
    assert not progression.record_pickup(ItemType.STONE_BLOCK, inventory)
    inventory.add(ItemType.STONE_BLOCK, 1)
    assert progression.record_pickup(ItemType.STONE_BLOCK, inventory)
    assert progression.stage is ProgressionStage.CRAFT_STONE_PICKAXE

    inventory.add_tool(ToolInstance.create(ItemType.STONE_PICKAXE))
    assert progression.record_craft(ItemType.STONE_PICKAXE, inventory)
    assert progression.completed
    assert progression.objective(inventory).progress == "Objective complete"
    assert not progression.record_craft(ItemType.STONE_PICKAXE, inventory)
    assert not progression.record_pickup(ItemType.STONE_BLOCK, inventory)
    assert not progression._advance_to(ProgressionStage.COMPLETE)  # type: ignore[attr-defined]
    assert progression.snapshot.revision == 6


def test_progression_recipe_gates_validation_and_inventory_inference() -> None:
    empty = PlayerInventory()
    progression = SurvivalProgression()
    assert progression.recipe_unlocked("wooden_pickaxe")
    assert not progression.recipe_unlocked("stone_pickaxe")
    with pytest.raises(TypeError):
        progression.recipe_unlocked(cast(Any, 1))
    with pytest.raises(TypeError):
        progression.objective(cast(Any, object()))
    with pytest.raises(TypeError):
        progression.record_pickup(cast(Any, "wood"), empty)
    with pytest.raises(TypeError):
        progression.record_pickup(ItemType.WOOD_LOG, cast(Any, object()))
    with pytest.raises(TypeError):
        progression.record_craft(cast(Any, "wood"), empty)
    with pytest.raises(TypeError):
        progression.record_craft(ItemType.WOOD_PLANK, cast(Any, object()))

    assert SurvivalProgression.infer_from_inventory(empty).stage is ProgressionStage.COLLECT_WOOD
    logs = PlayerInventory()
    logs.add(ItemType.WOOD_LOG, WOOD_LOG_TARGET)
    assert (
        SurvivalProgression.infer_from_inventory(logs, guide_completed=False).stage
        is ProgressionStage.CRAFT_PLANKS
    )
    planks = PlayerInventory()
    planks.add(ItemType.WOOD_PLANK, 2)
    assert SurvivalProgression.infer_from_inventory(planks).stage is ProgressionStage.CRAFT_STICKS
    sticks = PlayerInventory()
    sticks.add(ItemType.STICK, 2)
    assert (
        SurvivalProgression.infer_from_inventory(sticks).stage
        is ProgressionStage.CRAFT_WOODEN_PICKAXE
    )
    wood_tool = PlayerInventory()
    wood_tool.add_tool(ToolInstance.create(ItemType.WOODEN_PICKAXE))
    assert (
        SurvivalProgression.infer_from_inventory(wood_tool).stage is ProgressionStage.COLLECT_STONE
    )
    wood_tool.add(ItemType.STONE_BLOCK, STONE_BLOCK_TARGET)
    assert (
        SurvivalProgression.infer_from_inventory(wood_tool).stage
        is ProgressionStage.CRAFT_STONE_PICKAXE
    )
    stone_tool = PlayerInventory()
    stone_tool.add_tool(ToolInstance.create(ItemType.STONE_PICKAXE))
    inferred = SurvivalProgression.infer_from_inventory(stone_tool)
    assert inferred.stage is ProgressionStage.COMPLETE
    assert inferred.guide_completed
    with pytest.raises(TypeError):
        SurvivalProgression.infer_from_inventory(cast(Any, object()))
    with pytest.raises(TypeError):
        SurvivalProgression.infer_from_inventory(empty, guide_completed=cast(Any, 1))


def test_objectives_cover_every_stage() -> None:
    inventory = PlayerInventory()
    inventory.set_slot(0, ItemStack(item=ItemType.WOOD_LOG, quantity=2))
    inventory.set_slot(1, ItemStack(item=ItemType.WOOD_PLANK, quantity=4))
    inventory.set_slot(2, ItemStack(item=ItemType.STICK, quantity=4))
    inventory.set_slot(3, ItemStack(item=ItemType.STONE_BLOCK, quantity=2))
    titles = []
    for stage in ProgressionStage:
        progression = SurvivalProgression(
            SurvivalProgressionSnapshot(stage=stage, guide_completed=True)
        )
        titles.append(progression.objective(inventory).title)
    assert titles == [
        "Gather wood",
        "Craft wood planks",
        "Craft sticks",
        "Craft a wooden pickaxe",
        "Mine stone",
        "Craft a stone pickaxe",
        "Stone Age reached",
    ]


def test_full_progression_playthrough_uses_real_crafting() -> None:
    from open_world_rpg.gameplay import CraftingService

    inventory = PlayerInventory()
    progression = SurvivalProgression()
    crafting = CraftingService()

    inventory.add(ItemType.WOOD_LOG, 3)
    assert progression.record_pickup(ItemType.WOOD_LOG, inventory)
    for _ in range(3):
        attempt = crafting.craft(inventory, "wood_planks")
        assert attempt.crafted
        progression.record_craft(ItemType.WOOD_PLANK, inventory)
    attempt = crafting.craft(inventory, "sticks")
    assert attempt.crafted
    assert progression.record_craft(ItemType.STICK, inventory)
    attempt = crafting.craft(inventory, "wooden_pickaxe")
    assert attempt.crafted
    assert progression.record_craft(ItemType.WOODEN_PICKAXE, inventory)

    inventory.add(ItemType.STONE_BLOCK, 3)
    assert progression.record_pickup(ItemType.STONE_BLOCK, inventory)
    attempt = crafting.craft(inventory, "stone_pickaxe")
    assert attempt.crafted
    assert progression.record_craft(ItemType.STONE_PICKAXE, inventory)
    assert progression.completed
    assert inventory.total_quantity(ItemType.WOOD_LOG) == 0
    assert inventory.total_quantity(ItemType.STONE_BLOCK) == 0
