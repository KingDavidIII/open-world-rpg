"""Stable block material, coordinate, edit, and store contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from open_world_rpg.world import (
    BlockEdit,
    BlockEditStore,
    BlockMaterial,
    ChunkCoordinate,
    WorldBlockCoordinate,
)


def test_block_material_values_and_solidity_are_stable() -> None:
    assert tuple(item.value for item in BlockMaterial) == (
        "air",
        "grass",
        "dirt",
        "stone",
        "sand",
        "snow",
        "wood",
        "leaves",
        "water",
    )
    assert not BlockMaterial.AIR.is_solid
    assert not BlockMaterial.WATER.is_solid
    assert all(
        material.is_solid
        for material in BlockMaterial
        if material not in (BlockMaterial.AIR, BlockMaterial.WATER)
    )


def test_world_block_coordinate_is_validated_ordered_and_negative_safe() -> None:
    coordinate = WorldBlockCoordinate(x=-17, y=2**100, z=16)
    assert coordinate.chunk_coordinate == ChunkCoordinate(x=-2, y=1)
    assert (coordinate.local_coordinate.x, coordinate.local_coordinate.y) == (15, 0)
    assert coordinate.offset(x=1, y=-2, z=3) == WorldBlockCoordinate(x=-16, y=2**100 - 2, z=19)
    assert (
        sorted(
            (
                WorldBlockCoordinate(x=1, y=0, z=0),
                WorldBlockCoordinate(x=-1, y=9, z=0),
            )
        )[0].x
        == -1
    )
    with pytest.raises(FrozenInstanceError):
        coordinate.x = 0  # type: ignore[misc]
    for kwargs in (
        {"x": True, "y": 0, "z": 0},
        {"x": 0, "y": 1.5, "z": 0},
        {"x": 0, "y": 0, "z": "0"},
    ):
        with pytest.raises(TypeError):
            WorldBlockCoordinate(**cast(Any, kwargs))
    with pytest.raises(TypeError):
        coordinate.offset(x=True)


def test_block_edit_validation_and_immutability() -> None:
    coordinate = WorldBlockCoordinate(x=0, y=1, z=2)
    edit = BlockEdit(coordinate=coordinate, material=BlockMaterial.STONE, revision=1)
    with pytest.raises(FrozenInstanceError):
        edit.revision = 2  # type: ignore[misc]
    for kwargs, error in (
        (
            {"coordinate": cast(Any, (0, 1, 2)), "material": BlockMaterial.STONE, "revision": 1},
            TypeError,
        ),
        ({"coordinate": coordinate, "material": cast(Any, "stone"), "revision": 1}, TypeError),
        ({"coordinate": coordinate, "material": BlockMaterial.STONE, "revision": True}, TypeError),
        ({"coordinate": coordinate, "material": BlockMaterial.STONE, "revision": 0}, ValueError),
    ):
        with pytest.raises(error):
            BlockEdit(**kwargs)  # type: ignore[arg-type]


def test_edit_store_revision_noops_chunk_index_snapshot_and_clear() -> None:
    store = BlockEditStore()
    west = WorldBlockCoordinate(x=-1, y=4, z=0)
    east = WorldBlockCoordinate(x=16, y=5, z=15)
    assert store.revision == 0
    assert not store.contains(west)
    assert store.get(west) is None
    assert not store.remove_override(west)
    assert not store.clear()

    first = store.set_block(east, BlockMaterial.DIRT)
    assert first.revision == 1
    assert store.set_block(east, BlockMaterial.DIRT) is first
    replaced = store.set_block(east, BlockMaterial.STONE)
    assert replaced.revision == 2
    store.set_block(west, BlockMaterial.AIR)
    same_chunk = WorldBlockCoordinate(x=-2, y=5, z=0)
    store.set_block(same_chunk, BlockMaterial.SAND)
    assert store.coordinates() == (same_chunk, west, east)
    assert store.edits_for_chunk(ChunkCoordinate(x=-1, y=0))[0].coordinate == same_chunk
    assert store.edits_for_chunk(ChunkCoordinate(x=1, y=0))[0].coordinate == east
    assert store.edits_for_chunk(ChunkCoordinate(x=99, y=99)) == ()
    snapshot = store.snapshot()
    assert snapshot.revision == 4
    assert snapshot.edits == tuple(store.get(item) for item in store.coordinates())
    assert store.remove_override(west)
    assert store.revision == 5
    assert store.edits_for_chunk(ChunkCoordinate(x=-1, y=0))[0].coordinate == same_chunk
    assert store.remove_override(east)
    assert store.revision == 6
    assert store.clear()
    assert store.revision == 7
    assert len(store) == 0

    for operation in (
        lambda: store.get(cast(Any, (0, 0, 0))),
        lambda: store.contains(cast(Any, (0, 0, 0))),
        lambda: store.set_block(cast(Any, (0, 0, 0)), BlockMaterial.DIRT),
        lambda: store.set_block(east, cast(Any, "dirt")),
        lambda: store.remove_override(cast(Any, (0, 0, 0))),
        lambda: store.edits_for_chunk(cast(Any, (0, 0))),
    ):
        with pytest.raises(TypeError):
            operation()
