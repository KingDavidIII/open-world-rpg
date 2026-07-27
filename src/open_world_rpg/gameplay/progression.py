"""Deterministic playable-survival progression and first-run guidance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from .inventory import PlayerInventory
from .items import ItemType, ToolInstance

WOOD_LOG_TARGET: Final = 3
STONE_BLOCK_TARGET: Final = 3


class ProgressionStage(StrEnum):
    """Stable ordered stages for the v0.9.0 survival loop."""

    COLLECT_WOOD = "collect wood"
    CRAFT_PLANKS = "craft planks"
    CRAFT_STICKS = "craft sticks"
    CRAFT_WOODEN_PICKAXE = "craft wooden pickaxe"
    COLLECT_STONE = "collect stone"
    CRAFT_STONE_PICKAXE = "craft stone pickaxe"
    COMPLETE = "complete"


_STAGE_ORDER: Final = tuple(ProgressionStage)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProgressionObjective:
    """One renderer-independent objective presented to the player."""

    title: str
    instruction: str
    progress: str

    def __post_init__(self) -> None:
        for name, value in (
            ("title", self.title),
            ("instruction", self.instruction),
            ("progress", self.progress),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string.")
            if not value.strip():
                raise ValueError(f"{name} must not be blank.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SurvivalProgressionSnapshot:
    """Persistence-safe progression state."""

    stage: ProgressionStage = ProgressionStage.COLLECT_WOOD
    guide_completed: bool = False
    revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ProgressionStage):
            raise TypeError("stage must be a ProgressionStage.")
        if not isinstance(self.guide_completed, bool):
            raise TypeError("guide_completed must be a boolean.")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer.")
        if self.revision < 0:
            raise ValueError("revision must be non-negative.")


GUIDE_PAGES: Final[tuple[tuple[str, str], ...]] = (
    (
        "WELCOME TO THE SURVIVAL PROTOTYPE",
        "Move with WASD, look with the mouse, jump with Space, and sprint with Shift.",
    ),
    (
        "BUILD YOUR FIRST TOOL",
        (
            "Break the starter tree directly ahead and collect three logs. "
            "Press E, craft planks and sticks, then craft a wooden pickaxe."
        ),
    ),
    (
        "REACH THE STONE AGE",
        (
            "Mine three stone blocks with the wooden pickaxe, craft a stone pickaxe, "
            "then save from Escape."
        ),
    ),
)


class SurvivalProgression:
    """Advance the compact wood-to-stone playable loop exactly once."""

    def __init__(self, snapshot: SurvivalProgressionSnapshot | None = None) -> None:
        state = SurvivalProgressionSnapshot() if snapshot is None else snapshot
        if not isinstance(state, SurvivalProgressionSnapshot):
            raise TypeError("snapshot must be a SurvivalProgressionSnapshot or None.")
        self._stage = state.stage
        self._guide_completed = state.guide_completed
        self._revision = state.revision
        self._guide_page_index = 0

    @property
    def snapshot(self) -> SurvivalProgressionSnapshot:
        return SurvivalProgressionSnapshot(
            stage=self._stage,
            guide_completed=self._guide_completed,
            revision=self._revision,
        )

    @property
    def stage(self) -> ProgressionStage:
        return self._stage

    @property
    def completed(self) -> bool:
        return self._stage is ProgressionStage.COMPLETE

    @property
    def guide_completed(self) -> bool:
        return self._guide_completed

    @property
    def guide_page_index(self) -> int:
        return self._guide_page_index

    @property
    def guide_page(self) -> tuple[str, str]:
        return GUIDE_PAGES[self._guide_page_index]

    def next_guide_page(self) -> bool:
        """Advance one page; return True only when the guide closes."""
        if self._guide_completed:
            return True
        if self._guide_page_index + 1 < len(GUIDE_PAGES):
            self._guide_page_index += 1
            return False
        self._guide_completed = True
        self._revision += 1
        return True

    def dismiss_guide(self) -> bool:
        if self._guide_completed:
            return False
        self._guide_completed = True
        self._revision += 1
        return True

    def objective(self, inventory: PlayerInventory) -> ProgressionObjective:
        if not isinstance(inventory, PlayerInventory):
            raise TypeError("inventory must be a PlayerInventory.")
        logs = inventory.total_quantity(ItemType.WOOD_LOG)
        planks = inventory.total_quantity(ItemType.WOOD_PLANK)
        sticks = inventory.total_quantity(ItemType.STICK)
        stone = inventory.total_quantity(ItemType.STONE_BLOCK)
        objectives = {
            ProgressionStage.COLLECT_WOOD: ProgressionObjective(
                title="Gather wood",
                instruction="Break the starter tree directly ahead and collect three logs.",
                progress=f"{min(logs, WOOD_LOG_TARGET)}/{WOOD_LOG_TARGET} logs",
            ),
            ProgressionStage.CRAFT_PLANKS: ProgressionObjective(
                title="Craft wood planks",
                instruction="Open the inventory with E and craft Wood Planks.",
                progress=f"{planks} planks available",
            ),
            ProgressionStage.CRAFT_STICKS: ProgressionObjective(
                title="Craft sticks",
                instruction="Craft Sticks from two wood planks.",
                progress=f"{sticks} sticks available",
            ),
            ProgressionStage.CRAFT_WOODEN_PICKAXE: ProgressionObjective(
                title="Craft a wooden pickaxe",
                instruction="Use three planks and two sticks to craft a Wooden Pickaxe.",
                progress="Tool required",
            ),
            ProgressionStage.COLLECT_STONE: ProgressionObjective(
                title="Mine stone",
                instruction="Select the wooden pickaxe and collect three Stone Blocks.",
                progress=f"{min(stone, STONE_BLOCK_TARGET)}/{STONE_BLOCK_TARGET} stone",
            ),
            ProgressionStage.CRAFT_STONE_PICKAXE: ProgressionObjective(
                title="Craft a stone pickaxe",
                instruction="Use three stone and two sticks to craft a Stone Pickaxe.",
                progress="Final objective",
            ),
            ProgressionStage.COMPLETE: ProgressionObjective(
                title="Stone Age reached",
                instruction=(
                    "The playable survival loop is complete. Continue building or save and quit."
                ),
                progress="Objective complete",
            ),
        }
        return objectives[self._stage]

    def record_pickup(self, item: ItemType, inventory: PlayerInventory) -> bool:
        if not isinstance(item, ItemType):
            raise TypeError("item must be an ItemType.")
        if not isinstance(inventory, PlayerInventory):
            raise TypeError("inventory must be a PlayerInventory.")
        if (
            self._stage is ProgressionStage.COLLECT_WOOD
            and item is ItemType.WOOD_LOG
            and inventory.total_quantity(ItemType.WOOD_LOG) >= WOOD_LOG_TARGET
        ):
            return self._advance_to(ProgressionStage.CRAFT_PLANKS)
        if (
            self._stage is ProgressionStage.COLLECT_STONE
            and item is ItemType.STONE_BLOCK
            and inventory.total_quantity(ItemType.STONE_BLOCK) >= STONE_BLOCK_TARGET
        ):
            return self._advance_to(ProgressionStage.CRAFT_STONE_PICKAXE)
        return False

    def record_craft(self, output: ItemType, inventory: PlayerInventory) -> bool:
        if not isinstance(output, ItemType):
            raise TypeError("output must be an ItemType.")
        if not isinstance(inventory, PlayerInventory):
            raise TypeError("inventory must be a PlayerInventory.")
        expected = {
            ProgressionStage.CRAFT_PLANKS: (ItemType.WOOD_PLANK, ProgressionStage.CRAFT_STICKS),
            ProgressionStage.CRAFT_STICKS: (
                ItemType.STICK,
                ProgressionStage.CRAFT_WOODEN_PICKAXE,
            ),
            ProgressionStage.CRAFT_WOODEN_PICKAXE: (
                ItemType.WOODEN_PICKAXE,
                ProgressionStage.COLLECT_STONE,
            ),
            ProgressionStage.CRAFT_STONE_PICKAXE: (
                ItemType.STONE_PICKAXE,
                ProgressionStage.COMPLETE,
            ),
        }
        transition = expected.get(self._stage)
        if transition is None or output is not transition[0]:
            return False
        if output in (ItemType.WOODEN_PICKAXE, ItemType.STONE_PICKAXE) and not self._has_tool(
            inventory, output
        ):
            return False
        return self._advance_to(transition[1])

    def recipe_unlocked(self, identifier: str) -> bool:
        if not isinstance(identifier, str):
            raise TypeError("identifier must be a string.")
        if identifier.startswith("stone_"):
            return _STAGE_ORDER.index(self._stage) >= _STAGE_ORDER.index(
                ProgressionStage.COLLECT_STONE
            )
        return True

    @classmethod
    def infer_from_inventory(
        cls,
        inventory: PlayerInventory,
        *,
        guide_completed: bool = True,
    ) -> SurvivalProgression:
        """Construct a safe legacy progression state without mutating inventory."""
        if not isinstance(inventory, PlayerInventory):
            raise TypeError("inventory must be a PlayerInventory.")
        if not isinstance(guide_completed, bool):
            raise TypeError("guide_completed must be a boolean.")
        if cls._has_tool(inventory, ItemType.STONE_PICKAXE):
            stage = ProgressionStage.COMPLETE
        elif cls._has_tool(inventory, ItemType.WOODEN_PICKAXE):
            stage = (
                ProgressionStage.CRAFT_STONE_PICKAXE
                if inventory.total_quantity(ItemType.STONE_BLOCK) >= STONE_BLOCK_TARGET
                else ProgressionStage.COLLECT_STONE
            )
        elif inventory.total_quantity(ItemType.STICK) >= 2:
            stage = ProgressionStage.CRAFT_WOODEN_PICKAXE
        elif inventory.total_quantity(ItemType.WOOD_PLANK) >= 2:
            stage = ProgressionStage.CRAFT_STICKS
        elif inventory.total_quantity(ItemType.WOOD_LOG) >= WOOD_LOG_TARGET:
            stage = ProgressionStage.CRAFT_PLANKS
        else:
            stage = ProgressionStage.COLLECT_WOOD
        return cls(
            SurvivalProgressionSnapshot(
                stage=stage,
                guide_completed=guide_completed,
                revision=0,
            )
        )

    def _advance_to(self, stage: ProgressionStage) -> bool:
        if _STAGE_ORDER.index(stage) <= _STAGE_ORDER.index(self._stage):
            return False
        self._stage = stage
        self._revision += 1
        return True

    @staticmethod
    def _has_tool(inventory: PlayerInventory, item: ItemType) -> bool:
        return any(
            isinstance(slot, ToolInstance) and slot.item is item for slot in inventory.slots()
        )
