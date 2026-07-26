"""Immutable creative hotbar selection policy."""

from __future__ import annotations

from dataclasses import dataclass

from open_world_rpg.world import BlockMaterial

HOTBAR_SIZE = 9
DEFAULT_HOTBAR_SLOTS: tuple[BlockMaterial | None, ...] = (
    BlockMaterial.GRASS,
    BlockMaterial.DIRT,
    BlockMaterial.STONE,
    BlockMaterial.SAND,
    BlockMaterial.SNOW,
    None,
    None,
    None,
    None,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class VoxelHotbar:
    """Nine creative slots with deterministic wrapping selection."""

    slots: tuple[BlockMaterial | None, ...] = DEFAULT_HOTBAR_SLOTS
    selected_index: int = 0

    def __post_init__(self) -> None:
        if len(self.slots) != HOTBAR_SIZE:
            raise ValueError("hotbar must contain exactly nine slots.")
        if any(item is not None and not isinstance(item, BlockMaterial) for item in self.slots):
            raise TypeError("hotbar slots must contain BlockMaterial values or None.")
        if isinstance(self.selected_index, bool) or not isinstance(self.selected_index, int):
            raise TypeError("selected_index must be an integer.")
        if not 0 <= self.selected_index < HOTBAR_SIZE:
            raise ValueError("selected_index must be between zero and eight.")

    @property
    def selected_material(self) -> BlockMaterial | None:
        return self.slots[self.selected_index]

    def select(self, slot_number: int) -> VoxelHotbar:
        if isinstance(slot_number, bool) or not isinstance(slot_number, int):
            raise TypeError("slot_number must be an integer.")
        if not 1 <= slot_number <= HOTBAR_SIZE:
            raise ValueError("slot_number must be between one and nine.")
        return VoxelHotbar(slots=self.slots, selected_index=slot_number - 1)

    def cycle(self, steps: int) -> VoxelHotbar:
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer.")
        return VoxelHotbar(
            slots=self.slots,
            selected_index=(self.selected_index - steps) % HOTBAR_SIZE,
        )
