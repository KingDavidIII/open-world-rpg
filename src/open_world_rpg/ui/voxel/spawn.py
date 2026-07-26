"""Deterministic selection of a safe, visually useful voxel spawn."""

from __future__ import annotations

from collections.abc import Callable

from .blocks import BlockColumn, BlockType

ColumnLookup = Callable[[int, int], BlockColumn]
BlockedLookup = Callable[[int, int], bool]


def select_spawn(
    *,
    column_at: ColumnLookup,
    origin_x: int = 8,
    origin_z: int = 8,
    radius: int = 24,
    step: int = 4,
    blocked_at: BlockedLookup | None = None,
) -> tuple[int, int]:
    """Prefer dry grass with nearby water and visible local relief."""
    if radius < 0 or step <= 0:
        raise ValueError("spawn radius must be non-negative and step must be positive.")
    best: tuple[int, int] | None = None
    best_score = -10_000
    for z in range(origin_z - radius, origin_z + radius + 1, step):
        for x in range(origin_x - radius, origin_x + radius + 1, step):
            centre = column_at(x, z)
            if centre.water is not None or centre.surface is not BlockType.GRASS:
                continue
            if blocked_at is not None and blocked_at(x, z):
                continue
            immediate = (
                column_at(x - 1, z),
                column_at(x + 1, z),
                column_at(x, z - 1),
                column_at(x, z + 1),
            )
            if max(abs(item.ground_height - centre.ground_height) for item in immediate) > 1:
                continue
            neighbours = (
                column_at(x - 6, z),
                column_at(x + 6, z),
                column_at(x, z - 6),
                column_at(x, z + 6),
            )
            water = sum(item.water is not None for item in neighbours)
            relief = max(abs(item.ground_height - centre.ground_height) for item in neighbours)
            distance = abs(x - origin_x) + abs(z - origin_z)
            score = water * 40 + min(relief, 8) * 5 - distance
            if score > best_score:
                best = (x, z)
                best_score = score
    if best is None:
        raise ValueError("No safe voxel spawn was found within the search area.")
    return best
