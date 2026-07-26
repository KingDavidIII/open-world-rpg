"""Deterministic renderer-independent timed block mining."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from open_world_rpg.world import BlockMaterial, WorldBlockCoordinate

from .items import ToolClassification, ToolInstance, ToolTier, item_policy

MICROSECONDS_PER_SECOND: Final = 1_000_000
SPEED_SCALE: Final = 100
_HARDNESS_MICROSECONDS: Final = {
    BlockMaterial.SNOW: 300_000,
    BlockMaterial.SAND: 450_000,
    BlockMaterial.DIRT: 550_000,
    BlockMaterial.GRASS: 650_000,
    BlockMaterial.STONE: 2_000_000,
}
_SHOVEL_MATERIALS: Final = frozenset(
    (BlockMaterial.GRASS, BlockMaterial.DIRT, BlockMaterial.SAND, BlockMaterial.SNOW)
)


class MiningStatus(StrEnum):
    IDLE = "idle"
    ACTIVE = "active"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


def hardness_microseconds(material: BlockMaterial) -> int | None:
    if not isinstance(material, BlockMaterial):
        raise TypeError("material must be a BlockMaterial.")
    return _HARDNESS_MICROSECONDS.get(material)


def tool_speed_multiplier(tool: ToolInstance | None, material: BlockMaterial) -> int:
    """Return the authoritative mining speed in hundredths."""
    hardness = hardness_microseconds(material)
    if hardness is None:
        return 0
    if tool is None:
        return 100
    policy = item_policy(tool.item)
    classification = policy.tool_classification
    effective = classification is ToolClassification.PICKAXE and material is BlockMaterial.STONE
    effective = effective or (
        classification is ToolClassification.SHOVEL and material in _SHOVEL_MATERIALS
    )
    if effective:
        return 225 if policy.tool_tier is ToolTier.WOOD else 375
    return 75 if policy.tool_tier is ToolTier.WOOD else 85


def mining_duration_microseconds(material: BlockMaterial, tool: ToolInstance | None) -> int | None:
    hardness = hardness_microseconds(material)
    if hardness is None:
        return None
    speed = tool_speed_multiplier(tool, material)
    return (hardness * SPEED_SCALE + speed - 1) // speed


@dataclass(frozen=True, slots=True, kw_only=True)
class MiningSnapshot:
    target: WorldBlockCoordinate | None
    target_material: BlockMaterial | None
    selected_tool: ToolInstance | None
    elapsed_microseconds: int
    required_microseconds: int
    status: MiningStatus
    last_cancellation_reason: str | None

    @property
    def normalised_progress(self) -> float:
        if self.required_microseconds == 0:
            return 0.0
        return min(1.0, self.elapsed_microseconds / self.required_microseconds)


class TimedMiningController:
    """Explicit exactly-once mining state machine."""

    def __init__(self) -> None:
        self._target: WorldBlockCoordinate | None = None
        self._material: BlockMaterial | None = None
        self._tool: ToolInstance | None = None
        self._elapsed = 0
        self._required = 0
        self._status = MiningStatus.IDLE
        self._reason: str | None = None

    @property
    def snapshot(self) -> MiningSnapshot:
        return MiningSnapshot(
            target=self._target,
            target_material=self._material,
            selected_tool=self._tool,
            elapsed_microseconds=self._elapsed,
            required_microseconds=self._required,
            status=self._status,
            last_cancellation_reason=self._reason,
        )

    def begin(
        self,
        *,
        target: WorldBlockCoordinate,
        material: BlockMaterial,
        tool: ToolInstance | None,
    ) -> bool:
        if not isinstance(target, WorldBlockCoordinate):
            raise TypeError("target must be a WorldBlockCoordinate.")
        if tool is not None and not isinstance(tool, ToolInstance):
            raise TypeError("tool must be a ToolInstance or None.")
        required = mining_duration_microseconds(material, tool)
        if required is None:
            self.cancel("target is not mineable")
            return False
        if (
            self._status is MiningStatus.ACTIVE
            and target == self._target
            and material is self._material
            and tool == self._tool
        ):
            return False
        self._target = target
        self._material = material
        self._tool = tool
        self._elapsed = 0
        self._required = required
        self._status = MiningStatus.ACTIVE
        self._reason = None
        return True

    def advance(self, microseconds: int) -> MiningSnapshot:
        if isinstance(microseconds, bool) or not isinstance(microseconds, int):
            raise TypeError("microseconds must be an integer.")
        if microseconds < 0:
            raise ValueError("microseconds must be non-negative.")
        if self._status is not MiningStatus.ACTIVE or microseconds == 0:
            return self.snapshot
        self._elapsed = min(self._required, self._elapsed + microseconds)
        if self._elapsed == self._required:
            self._status = MiningStatus.COMPLETED
        return self.snapshot

    def cancel(self, reason: str) -> bool:
        if not isinstance(reason, str):
            raise TypeError("reason must be a string.")
        if not reason.strip():
            raise ValueError("reason must not be blank.")
        if self._status not in (MiningStatus.ACTIVE, MiningStatus.COMPLETED):
            return False
        self._status = MiningStatus.CANCELLED
        self._reason = reason
        return True

    def reset(self) -> bool:
        if self._status is MiningStatus.IDLE:
            return False
        self._target = None
        self._material = None
        self._tool = None
        self._elapsed = 0
        self._required = 0
        self._status = MiningStatus.IDLE
        self._reason = None
        return True
