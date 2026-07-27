"""Dropped-item simulation and pickup policy."""

from dataclasses import FrozenInstanceError

import pytest

from open_world_rpg.gameplay import (
    MAX_ACTIVE_DROPS,
    DroppedItem,
    DroppedItemManager,
    DroppedItemSnapshot,
    ItemStack,
    ItemType,
    PickupResult,
    PlayerInventory,
)


def test_drop_creation_identity_snapshot_and_validation() -> None:
    manager = DroppedItemManager()
    assert manager.pickup_radius == pytest.approx(2.25)
    first = manager.spawn(item=ItemType.STONE_BLOCK, quantity=1, position=(-1.5, 4.5, -2.5))
    second = manager.spawn(item=ItemType.DIRT_BLOCK, quantity=2, position=(0.5, 2.0, 0.5))
    assert (first.identifier, second.identifier) == (1, 2)
    assert manager.revision == 2
    assert manager.items() == (first, second)
    snapshot = manager.snapshot()
    assert DroppedItemManager.from_snapshot(snapshot).snapshot() == snapshot
    with pytest.raises(FrozenInstanceError):
        first.age = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        DroppedItem(identifier=True, item=ItemType.STONE_BLOCK, quantity=1, position=(0, 0, 0))
    with pytest.raises(ValueError):
        DroppedItem(identifier=0, item=ItemType.STONE_BLOCK, quantity=1, position=(0, 0, 0))
    with pytest.raises(TypeError):
        DroppedItem(identifier=1, item="stone", quantity=1, position=(0, 0, 0))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DroppedItem(identifier=1, item=ItemType.STONE_BLOCK, quantity=1, position=[0, 0, 0])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DroppedItem(
            identifier=1,
            item=ItemType.STONE_BLOCK,
            quantity=1,
            position=(float("nan"), 0, 0),
        )
    with pytest.raises(TypeError):
        DroppedItem(
            identifier=1,
            item=ItemType.STONE_BLOCK,
            quantity=1,
            position=(0, "bad", 0),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        DroppedItem(
            identifier=1,
            item=ItemType.STONE_BLOCK,
            quantity=1,
            position=(0, 0, 0),
            age=True,
        )
    with pytest.raises(ValueError):
        DroppedItem(
            identifier=1,
            item=ItemType.STONE_BLOCK,
            quantity=1,
            position=(0, 0, 0),
            pickup_delay=-1,
        )
    with pytest.raises(TypeError):
        DroppedItem(
            identifier=1,
            item=ItemType.STONE_BLOCK,
            quantity=1,
            position=(0, 0, 0),
            settled=1,  # type: ignore[arg-type]
        )
    assert PickupResult(accepted=1).changed
    assert not PickupResult().changed


def test_drop_physics_settling_despawn_and_validation() -> None:
    manager = DroppedItemManager(despawn_seconds=1)
    manager.spawn(item=ItemType.GRASS_BLOCK, quantity=1, position=(0.5, 1.4, 0.5))
    assert not manager.update(0, solid_at=lambda _x, _y, _z: True)
    assert manager.update(0.5, solid_at=lambda _x, y, _z: y == 0)
    assert manager.items()[0].settled
    assert manager.items()[0].position[1] == pytest.approx(1.13)
    assert manager.update(0.6, solid_at=lambda _x, y, _z: y == 0)
    assert not manager.items()
    for value in (True, "1"):
        with pytest.raises(TypeError):
            manager.update(value, solid_at=lambda _x, _y, _z: False)  # type: ignore[arg-type]
    for value in (-1, float("inf")):
        with pytest.raises(ValueError):
            manager.update(value, solid_at=lambda _x, _y, _z: False)
    with pytest.raises(TypeError):
        manager.update(1, solid_at=None)  # type: ignore[arg-type]


def test_pickup_delay_partial_full_inventory_and_order() -> None:
    manager = DroppedItemManager()
    manager.spawn(item=ItemType.STONE_BLOCK, quantity=4, position=(0, 0, 0))
    manager.spawn(item=ItemType.DIRT_BLOCK, quantity=1, position=(0, 0, 0))
    inventory = PlayerInventory()
    assert manager.pickup_near(position=(0, 0, 0), inventory=inventory) == ()
    manager.update(0.3, solid_at=lambda _x, _y, _z: False)
    results = manager.pickup_near(position=(0, 0, 0), inventory=inventory)
    assert [result.item for result in results] == [ItemType.STONE_BLOCK, ItemType.DIRT_BLOCK]
    assert len(manager) == 0

    for index in range(27):
        inventory.set_slot(index, ItemStack(item=ItemType.SAND_BLOCK, quantity=64))
    manager.spawn(item=ItemType.SNOW_BLOCK, quantity=2, position=(0, 0, 0))
    manager.update(0.3, solid_at=lambda _x, _y, _z: False)
    assert manager.pickup_near(position=(0, 0, 0), inventory=inventory) == ()
    inventory.set_slot(0, ItemStack(item=ItemType.SNOW_BLOCK, quantity=63))
    result = manager.pickup_near(position=(0, 0, 0), inventory=inventory)[0]
    assert (result.accepted, result.remainder) == (1, 1)
    assert manager.items()[0].quantity == 1
    assert manager.nearest_distance((0, 0, 0)) is not None
    inventory.set_slot(1, None)
    assert manager.pickup_near(position=(20, 0, 0), inventory=inventory) == ()
    assert manager.pickup_near(position=(0, 0, 0), inventory=inventory)[0].accepted == 1
    assert manager.nearest_distance((0, 0, 0)) is None


def test_drop_cap_discards_oldest_deterministically() -> None:
    manager = DroppedItemManager()
    for index in range(MAX_ACTIVE_DROPS + 1):
        manager.spawn(item=ItemType.DIRT_BLOCK, quantity=1, position=(index, 0, 0))
    assert len(manager) == MAX_ACTIVE_DROPS
    assert manager.items()[0].identifier == 2


def test_drop_snapshot_and_manager_validation() -> None:
    item = DroppedItem(identifier=1, item=ItemType.DIRT_BLOCK, quantity=1, position=(0, 0, 0))
    with pytest.raises(TypeError):
        DroppedItemSnapshot(revision=True, next_identifier=2, items=(item,))
    with pytest.raises(ValueError):
        DroppedItemSnapshot(revision=-1, next_identifier=2, items=(item,))
    with pytest.raises(TypeError):
        DroppedItemSnapshot(revision=0, next_identifier=True, items=(item,))
    with pytest.raises(ValueError):
        DroppedItemSnapshot(revision=0, next_identifier=0, items=(item,))
    with pytest.raises(TypeError):
        DroppedItemSnapshot(revision=0, next_identifier=2, items=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DroppedItemSnapshot(revision=0, next_identifier=2, items=("bad",))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DroppedItemSnapshot(revision=0, next_identifier=2, items=(item, item))
    with pytest.raises(ValueError):
        DroppedItemSnapshot(revision=0, next_identifier=1, items=(item,))
    many = tuple(
        DroppedItem(
            identifier=index + 1,
            item=ItemType.DIRT_BLOCK,
            quantity=1,
            position=(0, 0, 0),
        )
        for index in range(MAX_ACTIVE_DROPS + 1)
    )
    with pytest.raises(ValueError):
        DroppedItemSnapshot(
            revision=0,
            next_identifier=MAX_ACTIVE_DROPS + 2,
            items=many,
        )
    with pytest.raises(TypeError):
        DroppedItemManager.from_snapshot("bad")  # type: ignore[arg-type]
    for value in (True, "1"):
        with pytest.raises(TypeError):
            DroppedItemManager(pickup_radius=value)  # type: ignore[arg-type]
    for value in (0, -1, float("inf")):
        with pytest.raises(ValueError):
            DroppedItemManager(despawn_seconds=value)
    with pytest.raises(TypeError):
        DroppedItemManager().pickup_near(position=(0, 0, 0), inventory="bad")  # type: ignore[arg-type]
