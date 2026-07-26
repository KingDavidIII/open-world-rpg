"""Deterministic fixed-point player health, stamina, and fall state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

VITAL_SCALE: Final = 1_000
MAXIMUM_HEALTH: Final = 100
MAXIMUM_STAMINA: Final = 100
SPRINT_DRAIN_PER_SECOND: Final = 18
JUMP_STAMINA_COST: Final = 12
STAMINA_REGEN_PER_SECOND: Final = 14
STAMINA_REGEN_DELAY_MICROSECONDS: Final = 1_000_000
MINIMUM_SPRINT_STAMINA: Final = 5
SAFE_FALL_DISTANCE_MILLI: Final = 3_000
FALL_DAMAGE_PER_BLOCK: Final = 8


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerVitalsSnapshot:
    health_milli: int = MAXIMUM_HEALTH * VITAL_SCALE
    maximum_health_milli: int = MAXIMUM_HEALTH * VITAL_SCALE
    stamina_milli: int = MAXIMUM_STAMINA * VITAL_SCALE
    maximum_stamina_milli: int = MAXIMUM_STAMINA * VITAL_SCALE
    grounded: bool = True
    accumulated_fall_milli: int = 0
    regeneration_delay_microseconds: int = 0
    death_count: int = 0
    revision: int = 0
    last_fall_distance_milli: int = 0
    last_fall_damage: int = 0

    def __post_init__(self) -> None:
        for name in (
            "health_milli",
            "maximum_health_milli",
            "stamina_milli",
            "maximum_stamina_milli",
            "accumulated_fall_milli",
            "regeneration_delay_microseconds",
            "death_count",
            "revision",
            "last_fall_distance_milli",
            "last_fall_damage",
        ):
            _integer(name, getattr(self, name))
        if not isinstance(self.grounded, bool):
            raise TypeError("grounded must be a boolean.")
        if self.maximum_health_milli == 0 or self.maximum_stamina_milli == 0:
            raise ValueError("maximum vitals must be positive.")
        if self.health_milli > self.maximum_health_milli:
            raise ValueError("health cannot exceed maximum health.")
        if self.stamina_milli > self.maximum_stamina_milli:
            raise ValueError("stamina cannot exceed maximum stamina.")

    @property
    def health(self) -> float:
        return self.health_milli / VITAL_SCALE

    @property
    def stamina(self) -> float:
        return self.stamina_milli / VITAL_SCALE


class PlayerVitals:
    """Controlled mutable owner of immutable fixed-point snapshots."""

    def __init__(self, snapshot: PlayerVitalsSnapshot | None = None) -> None:
        self._state = snapshot or PlayerVitalsSnapshot()

    @property
    def snapshot(self) -> PlayerVitalsSnapshot:
        return self._state

    @property
    def can_sprint(self) -> bool:
        return self._state.stamina_milli >= MINIMUM_SPRINT_STAMINA * VITAL_SCALE

    def restore(self, snapshot: PlayerVitalsSnapshot) -> bool:
        if not isinstance(snapshot, PlayerVitalsSnapshot):
            raise TypeError("snapshot must be a PlayerVitalsSnapshot.")
        changed = snapshot != self._state
        self._state = snapshot
        return changed

    def _replace(self, **changes: int | bool) -> bool:
        values = {field: getattr(self._state, field) for field in self._state.__dataclass_fields__}
        values.update(changes)
        candidate = PlayerVitalsSnapshot(**values)
        if candidate == self._state:
            return False
        values["revision"] = self._state.revision + 1
        self._state = PlayerVitalsSnapshot(**values)
        return True

    def update_stamina(self, microseconds: int, *, sprinting: bool, active: bool = True) -> bool:
        _integer("microseconds", microseconds)
        if not isinstance(sprinting, bool) or not isinstance(active, bool):
            raise TypeError("sprinting and active must be booleans.")
        if microseconds == 0 or not active:
            return False
        state = self._state
        if sprinting and state.stamina_milli:
            drain = SPRINT_DRAIN_PER_SECOND * microseconds * VITAL_SCALE // 1_000_000
            return self._replace(
                stamina_milli=max(0, state.stamina_milli - drain),
                regeneration_delay_microseconds=STAMINA_REGEN_DELAY_MICROSECONDS,
            )
        delay = max(0, state.regeneration_delay_microseconds - microseconds)
        regen_time = max(0, microseconds - state.regeneration_delay_microseconds)
        gain = STAMINA_REGEN_PER_SECOND * regen_time * VITAL_SCALE // 1_000_000
        return self._replace(
            stamina_milli=min(state.maximum_stamina_milli, state.stamina_milli + gain),
            regeneration_delay_microseconds=delay,
        )

    def jump(self) -> bool:
        cost = JUMP_STAMINA_COST * VITAL_SCALE
        if self._state.stamina_milli < cost:
            return False
        self._replace(
            stamina_milli=self._state.stamina_milli - cost,
            regeneration_delay_microseconds=STAMINA_REGEN_DELAY_MICROSECONDS,
        )
        return True

    def damage(self, amount: int) -> bool:
        _integer("amount", amount)
        if amount == 0 or self._state.health_milli == 0:
            return False
        return self._replace(health_milli=max(0, self._state.health_milli - amount * VITAL_SCALE))

    def record_airborne_descent(self, distance_milli: int) -> bool:
        _integer("distance_milli", distance_milli)
        return self._replace(
            grounded=False,
            accumulated_fall_milli=self._state.accumulated_fall_milli + distance_milli,
        )

    def land(self, *, in_water: bool = False, immune: bool = False) -> int:
        if not isinstance(in_water, bool) or not isinstance(immune, bool):
            raise TypeError("landing flags must be booleans.")
        distance = self._state.accumulated_fall_milli
        excess = max(0, distance - SAFE_FALL_DISTANCE_MILLI)
        damage = 0 if in_water or immune else excess * FALL_DAMAGE_PER_BLOCK // VITAL_SCALE
        health = max(0, self._state.health_milli - damage * VITAL_SCALE)
        self._replace(
            health_milli=health,
            grounded=True,
            accumulated_fall_milli=0,
            last_fall_distance_milli=distance,
            last_fall_damage=damage,
        )
        return damage

    def reset_fall(self) -> bool:
        return self._replace(grounded=True, accumulated_fall_milli=0)

    def respawn(self) -> bool:
        return self._replace(
            health_milli=self._state.maximum_health_milli,
            stamina_milli=self._state.maximum_stamina_milli,
            grounded=True,
            accumulated_fall_milli=0,
            regeneration_delay_microseconds=0,
            death_count=self._state.death_count + 1,
        )
