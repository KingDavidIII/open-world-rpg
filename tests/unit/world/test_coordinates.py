"""Tests for deterministic world coordinate value objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from open_world_rpg.world import (
    CHUNK_SIZE,
    REGION_SIZE_IN_CHUNKS,
    REGION_SIZE_IN_TILES,
    ChunkCoordinate,
    LocalTileCoordinate,
    RegionCoordinate,
    WorldPosition,
)


def test_spatial_constants_define_nested_square_grids() -> None:
    assert CHUNK_SIZE == 16
    assert REGION_SIZE_IN_CHUNKS == 16
    assert REGION_SIZE_IN_TILES == 256


@pytest.mark.parametrize(
    ("coordinate_type", "field_name", "value"),
    [
        (WorldPosition, "x", True),
        (WorldPosition, "y", False),
        (WorldPosition, "x", 1.5),
        (ChunkCoordinate, "x", True),
        (ChunkCoordinate, "y", "0"),
        (RegionCoordinate, "x", False),
        (RegionCoordinate, "y", None),
        (LocalTileCoordinate, "x", True),
        (LocalTileCoordinate, "y", 1.5),
    ],
)
def test_coordinates_reject_non_integer_values(
    coordinate_type: type[WorldPosition | ChunkCoordinate | RegionCoordinate | LocalTileCoordinate],
    field_name: str,
    value: object,
) -> None:
    values: dict[str, Any] = {"x": 0, "y": 0}
    values[field_name] = value

    with pytest.raises(TypeError, match=rf"{field_name} must be an integer"):
        coordinate_type(**values)


@pytest.mark.parametrize(
    ("x", "y", "field_name"),
    [
        (-1, 0, "x"),
        (CHUNK_SIZE, 0, "x"),
        (0, -1, "y"),
        (0, CHUNK_SIZE, "y"),
    ],
)
def test_local_tile_rejects_values_outside_chunk(
    x: int,
    y: int,
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{field_name} must be between"):
        LocalTileCoordinate(x=x, y=y)


def test_coordinates_are_immutable_and_value_comparable() -> None:
    position = WorldPosition(x=4, y=-7)

    assert position == WorldPosition(x=4, y=-7)
    assert hash(position) == hash(WorldPosition(x=4, y=-7))

    with pytest.raises(FrozenInstanceError):
        position.x = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("world", "expected_chunk", "expected_local"),
    [
        ((0, 0), (0, 0), (0, 0)),
        ((15, 15), (0, 0), (15, 15)),
        ((16, 16), (1, 1), (0, 0)),
        ((-1, -1), (-1, -1), (15, 15)),
        ((-16, -16), (-1, -1), (0, 0)),
        ((-17, -17), (-2, -2), (15, 15)),
        ((16, -17), (1, -2), (0, 15)),
    ],
)
def test_world_to_chunk_and_local_uses_floor_division(
    world: tuple[int, int],
    expected_chunk: tuple[int, int],
    expected_local: tuple[int, int],
) -> None:
    position = WorldPosition(x=world[0], y=world[1])

    assert position.to_chunk() == ChunkCoordinate(
        x=expected_chunk[0],
        y=expected_chunk[1],
    )
    assert ChunkCoordinate.from_world(position) == position.to_chunk()
    assert position.to_local_tile() == LocalTileCoordinate(
        x=expected_local[0],
        y=expected_local[1],
    )
    assert LocalTileCoordinate.from_world(position) == position.to_local_tile()


@pytest.mark.parametrize(
    ("world", "expected_region"),
    [
        ((0, 0), (0, 0)),
        ((REGION_SIZE_IN_TILES - 1, REGION_SIZE_IN_TILES - 1), (0, 0)),
        ((REGION_SIZE_IN_TILES, REGION_SIZE_IN_TILES), (1, 1)),
        ((-1, -1), (-1, -1)),
        ((-REGION_SIZE_IN_TILES, -REGION_SIZE_IN_TILES), (-1, -1)),
        ((-REGION_SIZE_IN_TILES - 1, -REGION_SIZE_IN_TILES - 1), (-2, -2)),
    ],
)
def test_world_to_region_uses_floor_division(
    world: tuple[int, int],
    expected_region: tuple[int, int],
) -> None:
    position = WorldPosition(x=world[0], y=world[1])
    expected = RegionCoordinate(x=expected_region[0], y=expected_region[1])

    assert position.to_region() == expected
    assert RegionCoordinate.from_world(position) == expected


@pytest.mark.parametrize(
    ("chunk", "expected_region"),
    [
        ((0, 0), (0, 0)),
        ((REGION_SIZE_IN_CHUNKS - 1, REGION_SIZE_IN_CHUNKS - 1), (0, 0)),
        ((REGION_SIZE_IN_CHUNKS, REGION_SIZE_IN_CHUNKS), (1, 1)),
        ((-1, -1), (-1, -1)),
        ((-REGION_SIZE_IN_CHUNKS, -REGION_SIZE_IN_CHUNKS), (-1, -1)),
        ((-REGION_SIZE_IN_CHUNKS - 1, -REGION_SIZE_IN_CHUNKS - 1), (-2, -2)),
    ],
)
def test_chunk_to_region_uses_floor_division(
    chunk: tuple[int, int],
    expected_region: tuple[int, int],
) -> None:
    coordinate = ChunkCoordinate(x=chunk[0], y=chunk[1])
    expected = RegionCoordinate(x=expected_region[0], y=expected_region[1])

    assert coordinate.to_region() == expected
    assert RegionCoordinate.from_chunk(coordinate) == expected


@pytest.mark.parametrize(
    ("chunk", "expected_world"),
    [
        ((0, 0), (0, 0)),
        ((2, 3), (32, 48)),
        ((-1, -2), (-16, -32)),
    ],
)
def test_chunk_to_world_origin(
    chunk: tuple[int, int],
    expected_world: tuple[int, int],
) -> None:
    assert ChunkCoordinate(x=chunk[0], y=chunk[1]).to_world_origin() == WorldPosition(
        x=expected_world[0],
        y=expected_world[1],
    )


@pytest.mark.parametrize(
    ("region", "expected_chunk", "expected_world"),
    [
        ((0, 0), (0, 0), (0, 0)),
        ((2, 3), (32, 48), (512, 768)),
        ((-1, -2), (-16, -32), (-256, -512)),
    ],
)
def test_region_origins(
    region: tuple[int, int],
    expected_chunk: tuple[int, int],
    expected_world: tuple[int, int],
) -> None:
    coordinate = RegionCoordinate(x=region[0], y=region[1])

    assert coordinate.to_chunk_origin() == ChunkCoordinate(
        x=expected_chunk[0],
        y=expected_chunk[1],
    )
    assert coordinate.to_world_origin() == WorldPosition(
        x=expected_world[0],
        y=expected_world[1],
    )


@pytest.mark.parametrize(
    ("chunk", "local", "expected_world"),
    [
        ((0, 0), (0, 0), (0, 0)),
        ((1, 2), (15, 7), (31, 39)),
        ((-1, -2), (15, 0), (-1, -32)),
        ((-2, -1), (0, 15), (-32, -1)),
    ],
)
def test_chunk_and_local_round_trip_to_world(
    chunk: tuple[int, int],
    local: tuple[int, int],
    expected_world: tuple[int, int],
) -> None:
    position = WorldPosition.from_chunk_and_local(
        chunk=ChunkCoordinate(x=chunk[0], y=chunk[1]),
        local=LocalTileCoordinate(x=local[0], y=local[1]),
    )

    assert position == WorldPosition(x=expected_world[0], y=expected_world[1])
    assert position.to_chunk() == ChunkCoordinate(x=chunk[0], y=chunk[1])
    assert position.to_local_tile() == LocalTileCoordinate(x=local[0], y=local[1])


@pytest.mark.parametrize(
    ("operation", "argument_name"),
    [
        (
            lambda: ChunkCoordinate.from_world(cast(Any, object())),
            "position",
        ),
        (
            lambda: RegionCoordinate.from_world(cast(Any, object())),
            "position",
        ),
        (
            lambda: RegionCoordinate.from_chunk(cast(Any, object())),
            "chunk",
        ),
        (
            lambda: LocalTileCoordinate.from_world(cast(Any, object())),
            "position",
        ),
        (
            lambda: WorldPosition.from_chunk_and_local(
                chunk=cast(Any, object()),
                local=LocalTileCoordinate(x=0, y=0),
            ),
            "chunk",
        ),
        (
            lambda: WorldPosition.from_chunk_and_local(
                chunk=ChunkCoordinate(x=0, y=0),
                local=cast(Any, object()),
            ),
            "local",
        ),
    ],
)
def test_conversion_methods_reject_incorrect_value_object_types(
    operation: Any,
    argument_name: str,
) -> None:
    with pytest.raises(TypeError, match=rf"{argument_name} must be"):
        operation()
