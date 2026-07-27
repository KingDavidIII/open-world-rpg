"""Menu, inventory-screen, and crafting interaction state coverage."""

from typing import Any, cast

import pytest

from open_world_rpg.gameplay import ItemStack, ItemType, PlayerInventory
from open_world_rpg.ui.voxel import (
    GameFlowAction,
    GameFlowController,
    InventoryActionResult,
    InventoryScreenController,
    MenuOption,
    VoxelScreen,
)


def test_main_pause_dead_and_inventory_flow_transitions() -> None:
    flow = GameFlowController(initial_screen=VoxelScreen.MAIN_MENU)
    assert flow.overlay_active
    assert not flow.gameplay_active
    assert [option.label for option in flow.options] == ["New World", "Continue", "Quit"]
    assert flow.activate_selected() is GameFlowAction.NEW_WORLD
    assert flow.move_selection(1)
    assert flow.selected_index == 2
    assert flow.activate_selected() is GameFlowAction.QUIT
    flow.set_continue_available(True)
    flow.selected_index = 1
    assert flow.activate_selected() is GameFlowAction.CONTINUE
    assert flow.continue_world()
    assert flow.gameplay_active
    assert flow.options == ()
    assert flow.activate_selected() is GameFlowAction.NONE
    assert not flow.move_selection(1)

    assert flow.open_inventory()
    assert flow.screen is VoxelScreen.INVENTORY
    assert not flow.open_inventory()
    assert flow.close_inventory()
    assert not flow.close_inventory()
    assert flow.pause()
    assert flow.screen is VoxelScreen.PAUSED
    assert [option.action for option in flow.options] == [
        GameFlowAction.RESUME,
        GameFlowAction.SAVE,
        GameFlowAction.SAVE_AND_QUIT,
        GameFlowAction.QUIT,
    ]
    assert flow.resume()
    assert not flow.resume()
    assert flow.mark_dead()
    assert not flow.mark_dead()
    assert flow.options[0].action is GameFlowAction.RESPAWN
    assert flow.respawn()
    assert not flow.respawn()
    flow.return_to_main_menu()
    flow.start_new_world()
    assert flow.screen is VoxelScreen.PLAYING


def test_flow_validation_disabled_continue_and_selection_normalisation() -> None:
    with pytest.raises(TypeError):
        GameFlowController(initial_screen=cast(Any, "playing"))
    with pytest.raises(TypeError):
        GameFlowController(continue_available=cast(Any, 1))
    flow = GameFlowController(initial_screen=VoxelScreen.MAIN_MENU)
    assert not flow.continue_world()
    with pytest.raises(TypeError):
        flow.set_continue_available(cast(Any, 1))
    with pytest.raises(TypeError):
        flow.move_selection(cast(Any, True))
    assert not flow.move_selection(0)
    flow.selected_index = 99
    assert flow.activate_selected() is GameFlowAction.NEW_WORLD
    flow.selected_index = 1
    flow.set_continue_available(False)
    assert flow.selected_index == 0
    flow.screen = VoxelScreen.INVENTORY
    flow.selected_index = 9
    flow.set_continue_available(False)
    assert flow.selected_index == 0
    assert not flow.pause()


def test_menu_option_validation() -> None:
    option = MenuOption(label="Resume", action=GameFlowAction.RESUME)
    assert option.enabled
    with pytest.raises(TypeError):
        MenuOption(label=cast(Any, 1), action=GameFlowAction.RESUME)
    with pytest.raises(ValueError):
        MenuOption(label=" ", action=GameFlowAction.RESUME)
    with pytest.raises(TypeError):
        MenuOption(label="Bad", action=cast(Any, "resume"))
    with pytest.raises(TypeError):
        MenuOption(label="Bad", action=GameFlowAction.RESUME, enabled=cast(Any, 1))


def test_inventory_screen_selection_move_quick_move_and_craft() -> None:
    controller = InventoryScreenController()
    inventory = PlayerInventory()
    inventory.set_slot(0, ItemStack(item=ItemType.STONE_BLOCK, quantity=4))
    assert controller.inventory_section(0) == "hotbar"
    assert controller.inventory_section(9) == "backpack"
    assert controller.activate_slot(inventory) == InventoryActionResult(
        changed=False,
        message="Slot selected",
    )
    controller.select_slot(9)
    moved = controller.activate_slot(inventory)
    assert moved.changed
    assert inventory.slot(9) == ItemStack(item=ItemType.STONE_BLOCK, quantity=4)

    assert controller.activate_slot(inventory).message == "Slot selected"
    assert controller.activate_slot(inventory).message == "Selection cleared"
    controller.select_slot(1)
    assert controller.activate_slot(inventory).message == "Empty slot"
    controller.select_slot(9)
    assert controller.quick_move_selected(inventory).changed
    assert inventory.slot(0) == ItemStack(item=ItemType.STONE_BLOCK, quantity=4)

    inventory.add(ItemType.WOOD_PLANK, 3)
    inventory.add(ItemType.STICK, 2)
    controller.selected_recipe_index = 2
    attempt = controller.craft_selected(inventory)
    assert attempt.crafted
    assert controller.selected_recipe.identifier == "wooden_pickaxe"
    controller.reset()
    assert controller.selected_slot_index == 0
    assert controller.source_slot_index is None
    assert controller.selected_recipe_index == 0


def test_inventory_screen_navigation_and_validation() -> None:
    controller = InventoryScreenController()
    assert controller.move_slot_selection(columns=9, delta_x=1, delta_y=0)
    assert controller.selected_slot_index == 1
    assert controller.move_slot_selection(columns=9, delta_x=-2, delta_y=-1)
    assert controller.selected_slot_index == 26
    assert controller.move_recipe_selection(1)
    assert controller.move_recipe_selection(-1)
    assert not controller.move_recipe_selection(0)
    with pytest.raises(TypeError):
        controller.move_recipe_selection(cast(Any, True))
    with pytest.raises(TypeError):
        controller.move_slot_selection(columns=cast(Any, True), delta_x=0, delta_y=0)
    for columns in (0, 4):
        with pytest.raises(ValueError):
            controller.move_slot_selection(columns=columns, delta_x=0, delta_y=0)
    with pytest.raises(TypeError):
        controller.move_slot_selection(columns=9, delta_x=cast(Any, True), delta_y=0)
    with pytest.raises(TypeError):
        controller.select_slot(cast(Any, True))
    with pytest.raises(IndexError):
        controller.select_slot(27)
    with pytest.raises(TypeError):
        controller.activate_slot(cast(Any, object()))
    with pytest.raises(TypeError):
        controller.quick_move_selected(cast(Any, object()))
    with pytest.raises(TypeError):
        controller.inventory_section(cast(Any, True))
    with pytest.raises(IndexError):
        controller.inventory_section(-1)


def test_inventory_action_result_validation() -> None:
    assert InventoryActionResult(changed=True, message="Moved").changed
    with pytest.raises(TypeError):
        InventoryActionResult(changed=cast(Any, 1), message="Moved")
    with pytest.raises(TypeError):
        InventoryActionResult(changed=False, message=cast(Any, 1))
    with pytest.raises(ValueError):
        InventoryActionResult(changed=False, message=" ")


def test_flow_handles_an_all_disabled_custom_menu() -> None:
    class DisabledFlow(GameFlowController):
        @property
        def options(self) -> tuple[MenuOption, ...]:
            return (
                MenuOption(
                    label="Unavailable",
                    action=GameFlowAction.CONTINUE,
                    enabled=False,
                ),
            )

    flow = DisabledFlow(initial_screen=VoxelScreen.MAIN_MENU)
    assert not flow.move_selection(1)
    assert flow.activate_selected() is GameFlowAction.NONE


def test_guide_and_completion_flow_transitions() -> None:
    flow = GameFlowController()
    assert flow.open_guide()
    assert flow.screen is VoxelScreen.GUIDE
    assert flow.overlay_active
    assert not flow.open_guide()
    assert flow.close_guide()
    assert not flow.close_guide()

    assert flow.mark_completed()
    assert not flow.mark_completed()
    assert [option.action for option in flow.options] == [
        GameFlowAction.CONTINUE_PLAYING,
        GameFlowAction.SAVE_AND_QUIT,
        GameFlowAction.QUIT,
    ]
    assert flow.activate_selected() is GameFlowAction.CONTINUE_PLAYING
    assert flow.continue_playing()
    assert not flow.continue_playing()
