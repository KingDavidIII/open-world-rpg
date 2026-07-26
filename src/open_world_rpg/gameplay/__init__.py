"""Stable gameplay-domain contracts."""

from .drops import (
    MAX_ACTIVE_DROPS,
    DroppedItem,
    DroppedItemManager,
    DroppedItemSnapshot,
    PickupResult,
)
from .inventory import (
    HOTBAR_SIZE,
    INVENTORY_CAPACITY,
    InventoryAddResult,
    PlayerInventory,
    PlayerInventorySnapshot,
    create_bootstrap_inventory,
)
from .items import (
    MAX_STACK_SIZE,
    ItemStack,
    ItemType,
    item_for_material,
    material_for_item,
)

__all__ = [
    "HOTBAR_SIZE",
    "INVENTORY_CAPACITY",
    "MAX_ACTIVE_DROPS",
    "MAX_STACK_SIZE",
    "DroppedItem",
    "DroppedItemManager",
    "DroppedItemSnapshot",
    "InventoryAddResult",
    "ItemStack",
    "ItemType",
    "PickupResult",
    "PlayerInventory",
    "PlayerInventorySnapshot",
    "create_bootstrap_inventory",
    "item_for_material",
    "material_for_item",
]
