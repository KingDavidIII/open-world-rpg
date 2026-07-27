"""Renderer-independent inventory-grid and crafting-screen interaction state."""

from __future__ import annotations

from dataclasses import dataclass

from open_world_rpg.gameplay import (
    HOTBAR_SIZE,
    INVENTORY_CAPACITY,
    CraftingAttempt,
    CraftingCatalogue,
    CraftingRecipe,
    CraftingService,
    PlayerInventory,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class InventoryActionResult:
    changed: bool
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be a boolean.")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string.")
        if not self.message.strip():
            raise ValueError("message must not be empty.")


class InventoryScreenController:
    """Keyboard/mouse-neutral selection, transfer, and crafting policy."""

    def __init__(
        self,
        *,
        catalogue: CraftingCatalogue | None = None,
    ) -> None:
        self.selected_slot_index = 0
        self.source_slot_index: int | None = None
        self.selected_recipe_index = 0
        self.crafting = CraftingService(catalogue)

    @property
    def recipes(self) -> tuple[CraftingRecipe, ...]:
        return self.crafting.catalogue.recipes

    @property
    def selected_recipe(self) -> CraftingRecipe:
        return self.recipes[self.selected_recipe_index]

    def reset(self) -> None:
        self.selected_slot_index = 0
        self.source_slot_index = None
        self.selected_recipe_index = 0

    def move_slot_selection(self, *, columns: int, delta_x: int, delta_y: int) -> bool:
        if isinstance(columns, bool) or not isinstance(columns, int):
            raise TypeError("columns must be an integer.")
        if columns <= 0 or INVENTORY_CAPACITY % columns:
            raise ValueError("columns must evenly divide the inventory capacity.")
        for name, value in (("delta_x", delta_x), ("delta_y", delta_y)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        rows = INVENTORY_CAPACITY // columns
        row, column = divmod(self.selected_slot_index, columns)
        replacement = ((row + delta_y) % rows) * columns + (column + delta_x) % columns
        changed = replacement != self.selected_slot_index
        self.selected_slot_index = replacement
        return changed

    def select_slot(self, index: int) -> bool:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer.")
        if not 0 <= index < INVENTORY_CAPACITY:
            raise IndexError("index is outside the inventory.")
        changed = index != self.selected_slot_index
        self.selected_slot_index = index
        return changed

    def activate_slot(self, inventory: PlayerInventory) -> InventoryActionResult:
        if not isinstance(inventory, PlayerInventory):
            raise TypeError("inventory must be a PlayerInventory.")
        target = self.selected_slot_index
        if self.source_slot_index is None:
            if inventory.slot(target) is None:
                return InventoryActionResult(changed=False, message="Empty slot")
            self.source_slot_index = target
            return InventoryActionResult(changed=False, message="Slot selected")
        source = self.source_slot_index
        self.source_slot_index = None
        if source == target:
            return InventoryActionResult(changed=False, message="Selection cleared")
        transfer = inventory.move_slot(source, target)
        return InventoryActionResult(
            changed=transfer.changed,
            message=transfer.message,
        )

    def quick_move_selected(self, inventory: PlayerInventory) -> InventoryActionResult:
        if not isinstance(inventory, PlayerInventory):
            raise TypeError("inventory must be a PlayerInventory.")
        self.source_slot_index = None
        transfer = inventory.quick_move(self.selected_slot_index)
        return InventoryActionResult(changed=transfer.changed, message=transfer.message)

    def move_recipe_selection(self, direction: int) -> bool:
        if isinstance(direction, bool) or not isinstance(direction, int):
            raise TypeError("direction must be an integer.")
        if direction == 0:
            return False
        before = self.selected_recipe_index
        self.selected_recipe_index = (before + (1 if direction > 0 else -1)) % len(self.recipes)
        return self.selected_recipe_index != before

    def craft_selected(self, inventory: PlayerInventory) -> CraftingAttempt:
        return self.crafting.craft(inventory, self.selected_recipe.identifier)

    @staticmethod
    def inventory_section(index: int) -> str:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer.")
        if not 0 <= index < INVENTORY_CAPACITY:
            raise IndexError("index is outside the inventory.")
        return "hotbar" if index < HOTBAR_SIZE else "backpack"
