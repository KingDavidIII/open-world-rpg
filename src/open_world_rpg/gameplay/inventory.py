"""Controlled deterministic player inventory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .items import MAX_STACK_SIZE, ItemStack, ItemType

INVENTORY_CAPACITY: Final = 27
HOTBAR_SIZE: Final = 9


def _quantity(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("quantity must be an integer.")
    if value <= 0:
        raise ValueError("quantity must be positive.")
    return value


def _slot_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("slot index must be an integer.")
    if not 0 <= value < INVENTORY_CAPACITY:
        raise IndexError("slot index is outside the inventory.")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class InventoryAddResult:
    accepted: int
    remainder: int

    def __post_init__(self) -> None:
        for name, value in (("accepted", self.accepted), ("remainder", self.remainder)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerInventorySnapshot:
    revision: int
    selected_hotbar_index: int
    slots: tuple[ItemStack | None, ...]

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer.")
        if self.revision < 0:
            raise ValueError("revision must be non-negative.")
        if not isinstance(self.selected_hotbar_index, int) or isinstance(
            self.selected_hotbar_index, bool
        ):
            raise TypeError("selected_hotbar_index must be an integer.")
        if not 0 <= self.selected_hotbar_index < HOTBAR_SIZE:
            raise ValueError("selected_hotbar_index must be between zero and eight.")
        if not isinstance(self.slots, tuple):
            raise TypeError("slots must be a tuple.")
        if len(self.slots) != INVENTORY_CAPACITY:
            raise ValueError("inventory must contain exactly 27 slots.")
        if any(stack is not None and not isinstance(stack, ItemStack) for stack in self.slots):
            raise TypeError("slots must contain ItemStack values or None.")


class PlayerInventory:
    """Mutable inventory with one revision per successful public mutation."""

    def __init__(self) -> None:
        self._slots: list[ItemStack | None] = [None] * INVENTORY_CAPACITY
        self._selected = 0
        self._revision = 0

    @classmethod
    def from_snapshot(cls, snapshot: PlayerInventorySnapshot) -> PlayerInventory:
        if not isinstance(snapshot, PlayerInventorySnapshot):
            raise TypeError("snapshot must be a PlayerInventorySnapshot.")
        inventory = cls()
        inventory._slots = list(snapshot.slots)
        inventory._selected = snapshot.selected_hotbar_index
        inventory._revision = snapshot.revision
        return inventory

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def selected_hotbar_index(self) -> int:
        return self._selected

    @property
    def selected_stack(self) -> ItemStack | None:
        return self._slots[self._selected]

    def slot(self, index: int) -> ItemStack | None:
        return self._slots[_slot_index(index)]

    def slots(self) -> tuple[ItemStack | None, ...]:
        return tuple(self._slots)

    def select_hotbar(self, index: int) -> bool:
        index = _slot_index(index)
        if index >= HOTBAR_SIZE:
            raise IndexError("hotbar index must be between zero and eight.")
        if index == self._selected:
            return False
        self._selected = index
        self._revision += 1
        return True

    def cycle_hotbar(self, direction: int) -> bool:
        if isinstance(direction, bool) or not isinstance(direction, int):
            raise TypeError("direction must be an integer.")
        if direction == 0:
            return False
        return self.select_hotbar((self._selected - direction) % HOTBAR_SIZE)

    def add(self, item: ItemType, quantity: int) -> InventoryAddResult:
        if not isinstance(item, ItemType):
            raise TypeError("item must be an ItemType.")
        remaining = _quantity(quantity)
        replacement = list(self._slots)
        for index, stack in enumerate(replacement):
            if stack is not None and stack.item is item and stack.quantity < stack.maximum:
                accepted = min(remaining, stack.maximum - stack.quantity)
                replacement[index] = stack.with_quantity(stack.quantity + accepted)
                remaining -= accepted
                if not remaining:
                    break
        if remaining:
            for index, stack in enumerate(replacement):
                if stack is None:
                    accepted = min(remaining, MAX_STACK_SIZE)
                    replacement[index] = ItemStack(item=item, quantity=accepted)
                    remaining -= accepted
                    if not remaining:
                        break
        accepted_total = quantity - remaining
        if accepted_total:
            self._slots = replacement
            self._revision += 1
        return InventoryAddResult(accepted=accepted_total, remainder=remaining)

    def remove(self, item: ItemType, quantity: int) -> bool:
        if not isinstance(item, ItemType):
            raise TypeError("item must be an ItemType.")
        quantity = _quantity(quantity)
        if self.total_quantity(item) < quantity:
            return False
        remaining = quantity
        replacement = list(self._slots)
        for index, stack in enumerate(replacement):
            if remaining and stack is not None and stack.item is item:
                removed = min(remaining, stack.quantity)
                left = stack.quantity - removed
                replacement[index] = None if left == 0 else stack.with_quantity(left)
                remaining -= removed
        self._slots = replacement
        self._revision += 1
        return True

    def remove_from_slot(self, index: int, quantity: int) -> bool:
        index = _slot_index(index)
        quantity = _quantity(quantity)
        stack = self._slots[index]
        if stack is None or stack.quantity < quantity:
            return False
        left = stack.quantity - quantity
        self._slots[index] = None if left == 0 else stack.with_quantity(left)
        self._revision += 1
        return True

    def set_slot(self, index: int, stack: ItemStack | None) -> bool:
        index = _slot_index(index)
        if stack is not None and not isinstance(stack, ItemStack):
            raise TypeError("stack must be an ItemStack or None.")
        if self._slots[index] == stack:
            return False
        self._slots[index] = stack
        self._revision += 1
        return True

    def contains(self, item: ItemType, quantity: int) -> bool:
        return self.total_quantity(item) >= _quantity(quantity)

    def total_quantity(self, item: ItemType) -> int:
        if not isinstance(item, ItemType):
            raise TypeError("item must be an ItemType.")
        return sum(
            stack.quantity for stack in self._slots if stack is not None and stack.item is item
        )

    def snapshot(self) -> PlayerInventorySnapshot:
        return PlayerInventorySnapshot(
            revision=self._revision,
            selected_hotbar_index=self._selected,
            slots=tuple(self._slots),
        )

    def restore(self, snapshot: PlayerInventorySnapshot) -> bool:
        replacement = PlayerInventory.from_snapshot(snapshot)
        changed = self.snapshot() != snapshot
        self._slots = replacement._slots
        self._selected = replacement._selected
        self._revision = replacement._revision
        return changed

    def clear(self) -> bool:
        if all(stack is None for stack in self._slots):
            return False
        self._slots = [None] * INVENTORY_CAPACITY
        self._revision += 1
        return True

    @property
    def occupied_slots(self) -> int:
        return sum(stack is not None for stack in self._slots)

    @property
    def total_items(self) -> int:
        return sum(stack.quantity for stack in self._slots if stack is not None)


def create_bootstrap_inventory(*, enabled: bool = True) -> PlayerInventory:
    """Create the explicit deterministic manual-play starting inventory."""
    inventory = PlayerInventory()
    if enabled:
        for item, quantity in (
            (ItemType.GRASS_BLOCK, 8),
            (ItemType.DIRT_BLOCK, 8),
            (ItemType.STONE_BLOCK, 8),
            (ItemType.SAND_BLOCK, 4),
            (ItemType.SNOW_BLOCK, 4),
        ):
            inventory.add(item, quantity)
    return inventory
