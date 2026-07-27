"""Frame-time and block-aligned natural-feature contracts."""

from __future__ import annotations

import math

import pytest

from open_world_rpg.ui.voxel import (
    BlockColumn,
    BlockType,
    FrameTimeTracker,
    natural_blocks_in_area,
    tree_shape,
)
from open_world_rpg.world import BlockMaterial, WorldBlockCoordinate


def grass_column(_x: int, _z: int) -> BlockColumn:
    return BlockColumn(
        ground_height=12,
        surface_height=12,
        surface=BlockType.GRASS,
        subsurface=BlockType.DIRT,
    )


def test_frame_time_tracker_validates_and_reports_percentiles_and_stalls() -> None:
    with pytest.raises(TypeError):
        FrameTimeTracker(maximum_samples=True)
    with pytest.raises(ValueError):
        FrameTimeTracker(maximum_samples=9)

    tracker = FrameTimeTracker(maximum_samples=10)
    assert tracker.snapshot.sample_count == 0
    for invalid in (True, "0.1"):
        with pytest.raises(TypeError):
            tracker.record(invalid)  # type: ignore[arg-type]
    for invalid in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            tracker.record(invalid)

    for value in (0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022, 0.023, 0.1, 0.25):
        tracker.record(value)
    snapshot = tracker.snapshot
    assert snapshot.sample_count == 10
    assert snapshot.average_fps > 0
    assert snapshot.one_percent_low_fps == pytest.approx(4.0)
    assert snapshot.p95_frame_ms == pytest.approx(250.0)
    assert snapshot.worst_frame_ms == pytest.approx(250.0)
    assert snapshot.stall_count == 2
    assert snapshot.severe_stall_count == 1

    tracker.record(0.016)
    assert tracker.snapshot.sample_count == 10


def test_tree_shape_is_block_aligned_and_natural_area_resolves_tree_materials() -> None:
    shape = tree_shape(world_x=4, ground_y=12, world_z=-3)
    assert shape.trunk == tuple(WorldBlockCoordinate(x=4, y=y, z=-3) for y in range(13, 17))
    assert not set(shape.trunk) & set(shape.leaves)
    assert shape.blocks[shape.trunk[0]] is BlockMaterial.WOOD
    assert all(shape.blocks[item] is BlockMaterial.LEAVES for item in shape.leaves)

    with pytest.raises(ValueError):
        natural_blocks_in_area(
            minimum_x=1,
            maximum_x=0,
            minimum_z=0,
            maximum_z=0,
            column_at=grass_column,
            terrain_seed_at=lambda _x, _z: 7,
        )

    blocks = natural_blocks_in_area(
        minimum_x=-449,
        maximum_x=-447,
        minimum_z=-2,
        maximum_z=0,
        column_at=grass_column,
        terrain_seed_at=lambda _x, _z: 7,
    )
    assert WorldBlockCoordinate(x=-448, y=13, z=-1) in blocks
    assert blocks[WorldBlockCoordinate(x=-448, y=13, z=-1)] is BlockMaterial.WOOD
    assert blocks[WorldBlockCoordinate(x=-449, y=16, z=-2)] is BlockMaterial.LEAVES
