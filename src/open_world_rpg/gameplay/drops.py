"""Deterministic lightweight dropped-item simulation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from .inventory import PlayerInventory
from .items import ItemStack, ItemType

MAX_ACTIVE_DROPS: Final = 1024
Position = tuple[float, float, float]
SolidLookup = Callable[[int, int, int], bool]


def _vector(name: str, value: object) -> Position:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{name} must be a three-value tuple.")
    if any(
        isinstance(component, bool) or not isinstance(component, int | float) for component in value
    ):
        raise TypeError(f"{name} components must be numbers.")
    result = tuple(float(component) for component in value)
    if any(not math.isfinite(component) for component in result):
        raise ValueError(f"{name} components must be finite.")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class DroppedItem:
    identifier: int
    item: ItemType
    quantity: int
    position: Position
    velocity: Position = (0.0, 1.5, 0.0)
    age: float = 0.0
    pickup_delay: float = 0.3
    settled: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.identifier, bool) or not isinstance(self.identifier, int):
            raise TypeError("identifier must be an integer.")
        if self.identifier <= 0:
            raise ValueError("identifier must be positive.")
        if not isinstance(self.item, ItemType):
            raise TypeError("item must be an ItemType.")
        ItemStack(item=self.item, quantity=self.quantity)
        object.__setattr__(self, "position", _vector("position", self.position))
        object.__setattr__(self, "velocity", _vector("velocity", self.velocity))
        for name, value in (("age", self.age), ("pickup_delay", self.pickup_delay)):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a number.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.settled, bool):
            raise TypeError("settled must be a boolean.")


@dataclass(frozen=True, slots=True, kw_only=True)
class DroppedItemSnapshot:
    revision: int
    next_identifier: int
    items: tuple[DroppedItem, ...]

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer.")
        if self.revision < 0:
            raise ValueError("revision must be non-negative.")
        if isinstance(self.next_identifier, bool) or not isinstance(self.next_identifier, int):
            raise TypeError("next_identifier must be an integer.")
        if self.next_identifier <= 0:
            raise ValueError("next_identifier must be positive.")
        if not isinstance(self.items, tuple):
            raise TypeError("items must be a tuple.")
        if any(not isinstance(item, DroppedItem) for item in self.items):
            raise TypeError("items must contain DroppedItem values.")
        identifiers = tuple(item.identifier for item in self.items)
        if identifiers != tuple(sorted(identifiers)) or len(set(identifiers)) != len(identifiers):
            raise ValueError("items must use unique ascending identifiers.")
        if identifiers and self.next_identifier <= identifiers[-1]:
            raise ValueError("next_identifier must exceed saved identifiers.")
        if len(self.items) > MAX_ACTIVE_DROPS:
            raise ValueError("too many active dropped items.")


@dataclass(frozen=True, slots=True, kw_only=True)
class PickupResult:
    item: ItemType | None = None
    accepted: int = 0
    remainder: int = 0

    @property
    def changed(self) -> bool:
        return self.accepted > 0


class DroppedItemManager:
    """Ordered drop ownership, bounded physics, pickup, and despawning."""

    def __init__(self, *, pickup_radius: float = 1.5, despawn_seconds: float = 300.0) -> None:
        for name, value in (("pickup_radius", pickup_radius), ("despawn_seconds", despawn_seconds)):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a number.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.pickup_radius = float(pickup_radius)
        self.despawn_seconds = float(despawn_seconds)
        self._items: dict[int, DroppedItem] = {}
        self._next_identifier = 1
        self._revision = 0

    @classmethod
    def from_snapshot(cls, snapshot: DroppedItemSnapshot) -> DroppedItemManager:
        if not isinstance(snapshot, DroppedItemSnapshot):
            raise TypeError("snapshot must be a DroppedItemSnapshot.")
        manager = cls()
        manager._revision = snapshot.revision
        manager._next_identifier = snapshot.next_identifier
        manager._items = {item.identifier: item for item in snapshot.items}
        return manager

    @property
    def revision(self) -> int:
        return self._revision

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> tuple[DroppedItem, ...]:
        return tuple(self._items[identifier] for identifier in sorted(self._items))

    def spawn(self, *, item: ItemType, quantity: int, position: Position) -> DroppedItem:
        if len(self._items) == MAX_ACTIVE_DROPS:
            del self._items[min(self._items)]
        drop = DroppedItem(
            identifier=self._next_identifier,
            item=item,
            quantity=quantity,
            position=position,
        )
        self._items[drop.identifier] = drop
        self._next_identifier += 1
        self._revision += 1
        return drop

    def update(self, delta_seconds: float, *, solid_at: SolidLookup) -> bool:
        if isinstance(delta_seconds, bool) or not isinstance(delta_seconds, int | float):
            raise TypeError("delta_seconds must be a number.")
        if not math.isfinite(delta_seconds) or delta_seconds < 0:
            raise ValueError("delta_seconds must be finite and non-negative.")
        if not callable(solid_at):
            raise TypeError("solid_at must be callable.")
        if delta_seconds == 0 or not self._items:
            return False
        changed = False
        steps = max(1, math.ceil(delta_seconds / 0.05))
        step = delta_seconds / steps
        replacement: dict[int, DroppedItem] = {}
        for identifier, original in self._items.items():
            if original.settled:
                drop = DroppedItem(
                    identifier=original.identifier,
                    item=original.item,
                    quantity=original.quantity,
                    position=original.position,
                    velocity=original.velocity,
                    age=original.age + delta_seconds,
                    pickup_delay=original.pickup_delay,
                    settled=True,
                )
            else:
                drop = original
                for _ in range(steps):
                    x, y, z = drop.position
                    vx, vy, vz = drop.velocity
                    age = drop.age + step
                    vy = max(-18.0, vy - 18.0 * step)
                    candidate_y = y + vy * step
                    ground = math.floor(candidate_y - 0.12)
                    settled = solid_at(math.floor(x), ground, math.floor(z)) and vy <= 0
                    if settled:
                        candidate_y = ground + 1.13
                        vy = 0.0
                    drop = DroppedItem(
                        identifier=drop.identifier,
                        item=drop.item,
                        quantity=drop.quantity,
                        position=(x + vx * step, candidate_y, z + vz * step),
                        velocity=(vx * 0.85, vy, vz * 0.85),
                        age=age,
                        pickup_delay=drop.pickup_delay,
                        settled=settled,
                    )
            if drop.age < self.despawn_seconds:
                replacement[identifier] = drop
            changed = changed or drop != original
        if len(replacement) != len(self._items):
            changed = True
        self._items = replacement
        self._revision += 1
        return True

    def pickup_near(
        self, *, position: Position, inventory: PlayerInventory
    ) -> tuple[PickupResult, ...]:
        player = _vector("position", position)
        if not isinstance(inventory, PlayerInventory):
            raise TypeError("inventory must be a PlayerInventory.")
        results: list[PickupResult] = []
        changed = False
        radius_squared = self.pickup_radius**2
        for identifier in tuple(sorted(self._items)):
            drop = self._items[identifier]
            if drop.age < drop.pickup_delay:
                continue
            distance_squared = sum((a - b) ** 2 for a, b in zip(drop.position, player, strict=True))
            if distance_squared > radius_squared:
                continue
            added = inventory.add(drop.item, drop.quantity)
            if not added.accepted:
                continue
            changed = True
            results.append(
                PickupResult(item=drop.item, accepted=added.accepted, remainder=added.remainder)
            )
            if added.remainder:
                self._items[identifier] = DroppedItem(
                    identifier=drop.identifier,
                    item=drop.item,
                    quantity=added.remainder,
                    position=drop.position,
                    velocity=drop.velocity,
                    age=drop.age,
                    pickup_delay=drop.pickup_delay,
                    settled=drop.settled,
                )
            else:
                del self._items[identifier]
        if changed:
            self._revision += 1
        return tuple(results)

    def nearest_distance(self, position: Position) -> float | None:
        point = _vector("position", position)
        if not self._items:
            return None
        return min(
            math.sqrt(sum((a - b) ** 2 for a, b in zip(item.position, point, strict=True)))
            for item in self._items.values()
        )

    def snapshot(self) -> DroppedItemSnapshot:
        return DroppedItemSnapshot(
            revision=self._revision,
            next_identifier=self._next_identifier,
            items=self.items(),
        )
