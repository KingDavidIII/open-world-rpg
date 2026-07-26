"""Controlled deterministic mixed player inventory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

from .items import MAX_STACK_SIZE, ItemStack, ItemType, ToolInstance, item_policy

INVENTORY_CAPACITY: Final = 27
HOTBAR_SIZE: Final = 9
InventorySlot: TypeAlias = ItemStack | ToolInstance | None


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
    slots: tuple[InventorySlot, ...]

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer.")
        if self.revision < 0:
            raise ValueError("revision must be non-negative.")
        if isinstance(self.selected_hotbar_index, bool) or not isinstance(
            self.selected_hotbar_index, int
        ):
            raise TypeError("selected_hotbar_index must be an integer.")
        if not 0 <= self.selected_hotbar_index < HOTBAR_SIZE:
            raise ValueError("selected_hotbar_index must be between zero and eight.")
        if not isinstance(self.slots, tuple):
            raise TypeError("slots must be a tuple.")
        if len(self.slots) != INVENTORY_CAPACITY:
            raise ValueError("inventory must contain exactly 27 slots.")
        if any(
            slot is not None and not isinstance(slot, (ItemStack, ToolInstance))
            for slot in self.slots
        ):
            raise TypeError("slots must contain inventory values or None.")


class PlayerInventory:
    """Mutable inventory with one revision per successful public mutation."""

    def __init__(self) -> None:
        self._slots: list[InventorySlot] = [None] * INVENTORY_CAPACITY
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
    def selected_slot(self) -> InventorySlot:
        return self._slots[self._selected]

    @property
    def selected_stack(self) -> ItemStack | None:
        slot = self.selected_slot
        return slot if isinstance(slot, ItemStack) else None

    @property
    def selected_tool(self) -> ToolInstance | None:
        slot = self.selected_slot
        return slot if isinstance(slot, ToolInstance) else None

    def slot(self, index: int) -> InventorySlot:
        return self._slots[_slot_index(index)]

    def slots(self) -> tuple[InventorySlot, ...]:
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
        policy = item_policy(item)
        quantity = _quantity(quantity)
        if not policy.stackable:
            raise ValueError("tools cannot be added with a quantity.")
        remaining = quantity
        replacement = list(self._slots)
        for index, slot in enumerate(replacement):
            if isinstance(slot, ItemStack) and slot.item is item and slot.quantity < slot.maximum:
                accepted = min(remaining, slot.maximum - slot.quantity)
                replacement[index] = slot.with_quantity(slot.quantity + accepted)
                remaining -= accepted
                if not remaining:
                    break
        if remaining:
            for index, slot in enumerate(replacement):
                if slot is None:
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

    def add_tool(self, tool: ToolInstance, *, slot: int | None = None) -> bool:
        if not isinstance(tool, ToolInstance):
            raise TypeError("tool must be a ToolInstance.")
        replacement = list(self._slots)
        if slot is None:
            try:
                index = replacement.index(None)
            except ValueError:
                return False
        else:
            index = _slot_index(slot)
            if replacement[index] is not None:
                return False
        replacement[index] = tool
        self._slots = replacement
        self._revision += 1
        return True

    def use_tool(self, index: int) -> bool:
        index = _slot_index(index)
        tool = self._slots[index]
        if not isinstance(tool, ToolInstance):
            return False
        self._slots[index] = tool.use()
        self._revision += 1
        return True

    def remove_tool(self, index: int) -> ToolInstance | None:
        index = _slot_index(index)
        tool = self._slots[index]
        if not isinstance(tool, ToolInstance):
            return None
        self._slots[index] = None
        self._revision += 1
        return tool

    def remove(self, item: ItemType, quantity: int) -> bool:
        if not item_policy(item).stackable:
            raise ValueError("tools cannot be removed with a quantity.")
        quantity = _quantity(quantity)
        if self.total_quantity(item) < quantity:
            return False
        remaining = quantity
        replacement = list(self._slots)
        for index, slot in enumerate(replacement):
            if remaining and isinstance(slot, ItemStack) and slot.item is item:
                removed = min(remaining, slot.quantity)
                left = slot.quantity - removed
                replacement[index] = None if left == 0 else slot.with_quantity(left)
                remaining -= removed
        self._slots = replacement
        self._revision += 1
        return True

    def remove_from_slot(self, index: int, quantity: int) -> bool:
        index = _slot_index(index)
        quantity = _quantity(quantity)
        stack = self._slots[index]
        if not isinstance(stack, ItemStack) or stack.quantity < quantity:
            return False
        left = stack.quantity - quantity
        self._slots[index] = None if left == 0 else stack.with_quantity(left)
        self._revision += 1
        return True

    def set_slot(self, index: int, slot: InventorySlot) -> bool:
        index = _slot_index(index)
        if slot is not None and not isinstance(slot, (ItemStack, ToolInstance)):
            raise TypeError("slot must be an inventory value or None.")
        if self._slots[index] == slot:
            return False
        self._slots[index] = slot
        self._revision += 1
        return True

    def contains(self, item: ItemType, quantity: int) -> bool:
        return self.total_quantity(item) >= _quantity(quantity)

    def total_quantity(self, item: ItemType) -> int:
        if not item_policy(item).stackable:
            return 0
        return sum(
            slot.quantity
            for slot in self._slots
            if isinstance(slot, ItemStack) and slot.item is item
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
        if all(slot is None for slot in self._slots):
            return False
        self._slots = [None] * INVENTORY_CAPACITY
        self._revision += 1
        return True

    @property
    def occupied_slots(self) -> int:
        return sum(slot is not None for slot in self._slots)

    @property
    def total_items(self) -> int:
        return sum(
            slot.quantity if isinstance(slot, ItemStack) else 1
            for slot in self._slots
            if slot is not None
        )


def create_bootstrap_inventory(*, enabled: bool = True) -> PlayerInventory:
    """Create the deterministic survival hotbar."""
    inventory = PlayerInventory()
    if enabled:
        inventory.add_tool(ToolInstance.create(ItemType.WOODEN_PICKAXE), slot=0)
        inventory.add_tool(ToolInstance.create(ItemType.WOODEN_SHOVEL), slot=1)
        for index, item, quantity in (
            (2, ItemType.GRASS_BLOCK, 8),
            (3, ItemType.DIRT_BLOCK, 8),
            (4, ItemType.STONE_BLOCK, 8),
            (5, ItemType.SAND_BLOCK, 4),
            (6, ItemType.SNOW_BLOCK, 4),
        ):
            inventory.set_slot(index, ItemStack(item=item, quantity=quantity))
    return inventory
