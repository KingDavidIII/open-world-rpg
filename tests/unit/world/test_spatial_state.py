"""Tests for immutable region and chunk state contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from open_world_rpg.world import (
    MAX_DERIVED_SEED,
    REGION_SIZE_IN_CHUNKS,
    ChunkCoordinate,
    ChunkGenerationKey,
    ChunkMetadata,
    ChunkSnapshot,
    ChunkState,
    ChunkTransitionError,
    RegionChunkIndex,
    RegionCoordinate,
    RegionGenerationKey,
    RegionLocalChunkCoordinate,
    RegionMetadata,
    RegionSnapshot,
    RegionState,
    RegionTransitionError,
    WorldGenerationStage,
    WorldSeed,
)

WORLD_SEED = WorldSeed(value=42)


def test_chunk_and_region_state_values_are_explicit() -> None:
    assert [state.value for state in ChunkState] == [
        "declared",
        "generating",
        "ready",
        "active",
        "suspended",
        "unloaded",
        "failed",
    ]
    assert [state.value for state in RegionState] == [
        "declared",
        "indexing",
        "ready",
        "active",
        "suspended",
        "unloaded",
        "failed",
    ]


def test_chunk_factory_derives_every_stage_seed() -> None:
    coordinate = ChunkCoordinate(x=-17, y=16)

    metadata = ChunkMetadata.create(
        world_seed=WORLD_SEED,
        coordinate=coordinate,
    )

    assert metadata.region_coordinate == RegionCoordinate(x=-2, y=1)
    assert metadata.state is ChunkState.DECLARED
    assert metadata.revision == 0
    assert (
        metadata.terrain_seed,
        metadata.climate_seed,
        metadata.biome_seed,
        metadata.feature_seed,
        metadata.resource_seed,
        metadata.structure_seed,
        metadata.entity_seed,
    ) == tuple(
        ChunkGenerationKey(
            world_seed=WORLD_SEED,
            coordinate=coordinate,
            stage=stage,
        ).derived_seed
        for stage in WorldGenerationStage
    )


def test_region_factory_derives_every_stage_seed() -> None:
    coordinate = RegionCoordinate(x=-17, y=16)

    metadata = RegionMetadata.create(
        world_seed=WORLD_SEED,
        coordinate=coordinate,
    )

    assert metadata.state is RegionState.DECLARED
    assert metadata.revision == 0
    assert (
        metadata.terrain_seed,
        metadata.climate_seed,
        metadata.biome_seed,
        metadata.feature_seed,
        metadata.resource_seed,
        metadata.structure_seed,
        metadata.entity_seed,
    ) == tuple(
        RegionGenerationKey(
            world_seed=WORLD_SEED,
            coordinate=coordinate,
            stage=stage,
        ).derived_seed
        for stage in WorldGenerationStage
    )


def test_factories_preserve_known_generation_vectors() -> None:
    seed = WorldSeed(value=0)

    chunk = ChunkMetadata.create(
        world_seed=seed,
        coordinate=ChunkCoordinate(x=0, y=0),
    )
    region = RegionMetadata.create(
        world_seed=seed,
        coordinate=RegionCoordinate(x=0, y=0),
    )

    assert chunk.terrain_seed == 13_651_306_222_367_028_357
    assert region.terrain_seed == 11_875_099_115_532_928_456


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: ChunkMetadata.create(
                world_seed=cast(Any, 42),
                coordinate=ChunkCoordinate(x=0, y=0),
            ),
            "world_seed must be a WorldSeed",
        ),
        (
            lambda: ChunkMetadata.create(
                world_seed=WORLD_SEED,
                coordinate=cast(Any, object()),
            ),
            "coordinate must be a ChunkCoordinate",
        ),
        (
            lambda: RegionMetadata.create(
                world_seed=cast(Any, 42),
                coordinate=RegionCoordinate(x=0, y=0),
            ),
            "world_seed must be a WorldSeed",
        ),
        (
            lambda: RegionMetadata.create(
                world_seed=WORLD_SEED,
                coordinate=cast(Any, object()),
            ),
            "coordinate must be a RegionCoordinate",
        ),
    ],
)
def test_metadata_factories_reject_invalid_dependencies(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        factory()


@pytest.mark.parametrize(
    ("kind", "field_name", "value", "message"),
    [
        ("chunk", "world_seed", 42, "world_seed must be a WorldSeed"),
        ("chunk", "coordinate", object(), "coordinate must be a ChunkCoordinate"),
        (
            "chunk",
            "region_coordinate",
            object(),
            "region_coordinate must be a RegionCoordinate",
        ),
        ("chunk", "state", "declared", "state must be a ChunkState"),
        ("region", "world_seed", 42, "world_seed must be a WorldSeed"),
        ("region", "coordinate", object(), "coordinate must be a RegionCoordinate"),
        ("region", "state", "declared", "state must be a RegionState"),
    ],
)
def test_metadata_rejects_invalid_component_types(
    kind: str,
    field_name: str,
    value: object,
    message: str,
) -> None:
    metadata: ChunkMetadata | RegionMetadata = (
        ChunkMetadata.create(
            world_seed=WORLD_SEED,
            coordinate=ChunkCoordinate(x=0, y=0),
        )
        if kind == "chunk"
        else RegionMetadata.create(
            world_seed=WORLD_SEED,
            coordinate=RegionCoordinate(x=0, y=0),
        )
    )

    with pytest.raises(TypeError, match=message):
        replace(metadata, **{field_name: value})


def test_chunk_rejects_inconsistent_region_coordinate() -> None:
    metadata = ChunkMetadata.create(
        world_seed=WORLD_SEED,
        coordinate=ChunkCoordinate(x=-1, y=-1),
    )

    with pytest.raises(ValueError, match="region_coordinate must contain"):
        replace(metadata, region_coordinate=RegionCoordinate(x=0, y=0))


SEED_FIELDS = [
    "terrain_seed",
    "climate_seed",
    "biome_seed",
    "feature_seed",
    "resource_seed",
    "structure_seed",
    "entity_seed",
]


@pytest.mark.parametrize("kind", ["chunk", "region"])
@pytest.mark.parametrize("field_name", SEED_FIELDS)
@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (True, TypeError, "must be an integer"),
        ("1", TypeError, "must be an integer"),
        (-1, ValueError, "must be between"),
        (MAX_DERIVED_SEED + 1, ValueError, "must be between"),
    ],
)
def test_all_generation_seed_fields_validate_range_and_type(
    kind: str,
    field_name: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    metadata: ChunkMetadata | RegionMetadata = (
        ChunkMetadata.create(
            world_seed=WORLD_SEED,
            coordinate=ChunkCoordinate(x=0, y=0),
        )
        if kind == "chunk"
        else RegionMetadata.create(
            world_seed=WORLD_SEED,
            coordinate=RegionCoordinate(x=0, y=0),
        )
    )

    with pytest.raises(error_type, match=field_name + " " + message):
        replace(metadata, **{field_name: value})


@pytest.mark.parametrize("kind", ["chunk", "region"])
@pytest.mark.parametrize("field_name", SEED_FIELDS)
def test_all_generation_seed_fields_must_match_derivation(
    kind: str,
    field_name: str,
) -> None:
    metadata: ChunkMetadata | RegionMetadata = (
        ChunkMetadata.create(
            world_seed=WORLD_SEED,
            coordinate=ChunkCoordinate(x=3, y=4),
        )
        if kind == "chunk"
        else RegionMetadata.create(
            world_seed=WORLD_SEED,
            coordinate=RegionCoordinate(x=3, y=4),
        )
    )
    current = cast(int, getattr(metadata, field_name))
    mismatched = current + 1 if current < MAX_DERIVED_SEED else current - 1

    with pytest.raises(ValueError, match=field_name + " must match"):
        replace(metadata, **{field_name: mismatched})


@pytest.mark.parametrize("kind", ["chunk", "region"])
@pytest.mark.parametrize(
    ("revision", "error_type", "message"),
    [
        (True, TypeError, "revision must be an integer"),
        ("1", TypeError, "revision must be an integer"),
        (-1, ValueError, "revision must be greater than or equal to zero"),
    ],
)
def test_metadata_revision_validation(
    kind: str,
    revision: object,
    error_type: type[Exception],
    message: str,
) -> None:
    metadata: ChunkMetadata | RegionMetadata = (
        ChunkMetadata.create(
            world_seed=WORLD_SEED,
            coordinate=ChunkCoordinate(x=0, y=0),
        )
        if kind == "chunk"
        else RegionMetadata.create(
            world_seed=WORLD_SEED,
            coordinate=RegionCoordinate(x=0, y=0),
        )
    )

    with pytest.raises(error_type, match=message):
        replace(metadata, revision=revision)


CHUNK_VALID_TRANSITIONS = [
    (ChunkState.DECLARED, ChunkState.GENERATING),
    (ChunkState.DECLARED, ChunkState.FAILED),
    (ChunkState.GENERATING, ChunkState.READY),
    (ChunkState.GENERATING, ChunkState.FAILED),
    (ChunkState.READY, ChunkState.ACTIVE),
    (ChunkState.READY, ChunkState.UNLOADED),
    (ChunkState.READY, ChunkState.FAILED),
    (ChunkState.ACTIVE, ChunkState.SUSPENDED),
    (ChunkState.ACTIVE, ChunkState.FAILED),
    (ChunkState.SUSPENDED, ChunkState.ACTIVE),
    (ChunkState.SUSPENDED, ChunkState.UNLOADED),
    (ChunkState.SUSPENDED, ChunkState.FAILED),
    (ChunkState.UNLOADED, ChunkState.GENERATING),
    (ChunkState.UNLOADED, ChunkState.FAILED),
]
REGION_VALID_TRANSITIONS = [
    (RegionState.DECLARED, RegionState.INDEXING),
    (RegionState.DECLARED, RegionState.FAILED),
    (RegionState.INDEXING, RegionState.READY),
    (RegionState.INDEXING, RegionState.FAILED),
    (RegionState.READY, RegionState.ACTIVE),
    (RegionState.READY, RegionState.UNLOADED),
    (RegionState.READY, RegionState.FAILED),
    (RegionState.ACTIVE, RegionState.SUSPENDED),
    (RegionState.ACTIVE, RegionState.FAILED),
    (RegionState.SUSPENDED, RegionState.ACTIVE),
    (RegionState.SUSPENDED, RegionState.UNLOADED),
    (RegionState.SUSPENDED, RegionState.FAILED),
    (RegionState.UNLOADED, RegionState.INDEXING),
    (RegionState.UNLOADED, RegionState.FAILED),
]


@pytest.mark.parametrize(("start", "target"), CHUNK_VALID_TRANSITIONS)
def test_every_valid_chunk_transition_increments_revision_once(
    start: ChunkState,
    target: ChunkState,
) -> None:
    original = replace(
        ChunkMetadata.create(
            world_seed=WORLD_SEED,
            coordinate=ChunkCoordinate(x=0, y=0),
        ),
        state=start,
        revision=8,
    )

    transitioned = original.transition_to(target)

    assert transitioned is not original
    assert transitioned.state is target
    assert transitioned.revision == 9
    assert original.state is start
    assert original.revision == 8


@pytest.mark.parametrize(("start", "target"), REGION_VALID_TRANSITIONS)
def test_every_valid_region_transition_increments_revision_once(
    start: RegionState,
    target: RegionState,
) -> None:
    original = replace(
        RegionMetadata.create(
            world_seed=WORLD_SEED,
            coordinate=RegionCoordinate(x=0, y=0),
        ),
        state=start,
        revision=8,
    )

    transitioned = original.transition_to(target)

    assert transitioned is not original
    assert transitioned.state is target
    assert transitioned.revision == 9
    assert original.state is start
    assert original.revision == 8


CHUNK_INVALID_TRANSITIONS = [
    (start, target)
    for start in ChunkState
    for target in ChunkState
    if start is not target and (start, target) not in CHUNK_VALID_TRANSITIONS
]
REGION_INVALID_TRANSITIONS = [
    (start, target)
    for start in RegionState
    for target in RegionState
    if start is not target and (start, target) not in REGION_VALID_TRANSITIONS
]


@pytest.mark.parametrize(("start", "target"), CHUNK_INVALID_TRANSITIONS)
def test_every_invalid_chunk_transition_preserves_original(
    start: ChunkState,
    target: ChunkState,
) -> None:
    original = replace(
        ChunkMetadata.create(
            world_seed=WORLD_SEED,
            coordinate=ChunkCoordinate(x=0, y=0),
        ),
        state=start,
    )

    with pytest.raises(ChunkTransitionError, match="Cannot transition chunk"):
        original.transition_to(target)

    assert original.state is start
    assert original.revision == 0


@pytest.mark.parametrize(("start", "target"), REGION_INVALID_TRANSITIONS)
def test_every_invalid_region_transition_preserves_original(
    start: RegionState,
    target: RegionState,
) -> None:
    original = replace(
        RegionMetadata.create(
            world_seed=WORLD_SEED,
            coordinate=RegionCoordinate(x=0, y=0),
        ),
        state=start,
    )

    with pytest.raises(RegionTransitionError, match="Cannot transition region"):
        original.transition_to(target)

    assert original.state is start
    assert original.revision == 0


def test_transition_validation_and_same_state_no_op() -> None:
    chunk = ChunkMetadata.create(
        world_seed=WORLD_SEED,
        coordinate=ChunkCoordinate(x=0, y=0),
    )
    region = RegionMetadata.create(
        world_seed=WORLD_SEED,
        coordinate=RegionCoordinate(x=0, y=0),
    )

    assert chunk.transition_to(ChunkState.DECLARED) is chunk
    assert region.transition_to(RegionState.DECLARED) is region

    with pytest.raises(TypeError, match="state must be a ChunkState"):
        chunk.transition_to(cast(Any, "generating"))

    with pytest.raises(TypeError, match="state must be a RegionState"):
        region.transition_to(cast(Any, "indexing"))


def test_snapshots_are_complete_and_immutable() -> None:
    chunk = ChunkMetadata.create(
        world_seed=WORLD_SEED,
        coordinate=ChunkCoordinate(x=-1, y=-1),
    ).transition_to(ChunkState.GENERATING)
    region = RegionMetadata.create(
        world_seed=WORLD_SEED,
        coordinate=RegionCoordinate(x=-1, y=-1),
    ).transition_to(RegionState.INDEXING)

    chunk_snapshot = chunk.snapshot()
    region_snapshot = region.snapshot()

    assert isinstance(chunk_snapshot, ChunkSnapshot)
    assert chunk_snapshot.world_seed is WORLD_SEED
    assert chunk_snapshot.coordinate == chunk.coordinate
    assert chunk_snapshot.region_coordinate == chunk.region_coordinate
    assert chunk_snapshot.terrain_seed == chunk.terrain_seed
    assert chunk_snapshot.entity_seed == chunk.entity_seed
    assert chunk_snapshot.state is ChunkState.GENERATING
    assert chunk_snapshot.revision == 1
    assert isinstance(region_snapshot, RegionSnapshot)
    assert region_snapshot.world_seed is WORLD_SEED
    assert region_snapshot.coordinate == region.coordinate
    assert region_snapshot.terrain_seed == region.terrain_seed
    assert region_snapshot.entity_seed == region.entity_seed
    assert region_snapshot.state is RegionState.INDEXING
    assert region_snapshot.revision == 1

    with pytest.raises(FrozenInstanceError):
        chunk_snapshot.revision = 2  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        region_snapshot.revision = 2  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["x", "y"])
@pytest.mark.parametrize("value", [True, False, "0", 1.5])
def test_region_local_chunk_coordinate_rejects_non_integers(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, Any] = {"x": 0, "y": 0}
    values[field_name] = value

    with pytest.raises(TypeError, match=field_name + " must be an integer"):
        RegionLocalChunkCoordinate(**values)


@pytest.mark.parametrize(
    ("x", "y", "field_name"),
    [
        (-1, 0, "x"),
        (REGION_SIZE_IN_CHUNKS, 0, "x"),
        (0, -1, "y"),
        (0, REGION_SIZE_IN_CHUNKS, "y"),
    ],
)
def test_region_local_chunk_coordinate_rejects_out_of_range(
    x: int,
    y: int,
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name + " must be between"):
        RegionLocalChunkCoordinate(x=x, y=y)


@pytest.mark.parametrize(
    ("region", "minimum", "maximum"),
    [
        ((-1, -1), (-16, -16), (-1, -1)),
        ((0, 0), (0, 0), (15, 15)),
        ((1, 1), (16, 16), (31, 31)),
        (
            (10**100, -(10**100)),
            (16 * 10**100, -(16 * 10**100)),
            (16 * 10**100 + 15, -(16 * 10**100) + 15),
        ),
    ],
)
def test_region_index_boundaries_support_negative_and_extreme_regions(
    region: tuple[int, int],
    minimum: tuple[int, int],
    maximum: tuple[int, int],
) -> None:
    index = RegionChunkIndex(region_coordinate=RegionCoordinate(x=region[0], y=region[1]))

    assert index.minimum == ChunkCoordinate(x=minimum[0], y=minimum[1])
    assert index.maximum == ChunkCoordinate(x=maximum[0], y=maximum[1])
    assert index.chunk_count == REGION_SIZE_IN_CHUNKS**2


def test_region_index_iterates_exact_row_major_order() -> None:
    index = RegionChunkIndex(region_coordinate=RegionCoordinate(x=0, y=0))
    coordinates = tuple(index)

    assert len(coordinates) == REGION_SIZE_IN_CHUNKS**2
    assert len(set(coordinates)) == REGION_SIZE_IN_CHUNKS**2
    assert coordinates[:17] == (
        ChunkCoordinate(x=0, y=0),
        ChunkCoordinate(x=1, y=0),
        ChunkCoordinate(x=2, y=0),
        ChunkCoordinate(x=3, y=0),
        ChunkCoordinate(x=4, y=0),
        ChunkCoordinate(x=5, y=0),
        ChunkCoordinate(x=6, y=0),
        ChunkCoordinate(x=7, y=0),
        ChunkCoordinate(x=8, y=0),
        ChunkCoordinate(x=9, y=0),
        ChunkCoordinate(x=10, y=0),
        ChunkCoordinate(x=11, y=0),
        ChunkCoordinate(x=12, y=0),
        ChunkCoordinate(x=13, y=0),
        ChunkCoordinate(x=14, y=0),
        ChunkCoordinate(x=15, y=0),
        ChunkCoordinate(x=0, y=1),
    )
    assert coordinates[-1] == ChunkCoordinate(x=15, y=15)


def test_region_index_contains_and_round_trips_local_lookup() -> None:
    index = RegionChunkIndex(region_coordinate=RegionCoordinate(x=-2, y=3))
    local = RegionLocalChunkCoordinate(x=15, y=7)
    chunk = index.chunk_at(local)

    assert chunk == ChunkCoordinate(x=-17, y=55)
    assert index.contains(index.minimum) is True
    assert index.contains(index.maximum) is True
    assert index.contains(chunk) is True
    assert index.contains(ChunkCoordinate(x=-33, y=48)) is False
    assert index.contains(ChunkCoordinate(x=-16, y=48)) is False
    assert index.local_coordinate(chunk) == local


def test_region_index_rejects_invalid_inputs_and_outside_lookup() -> None:
    with pytest.raises(TypeError, match="region_coordinate"):
        RegionChunkIndex(region_coordinate=cast(Any, object()))

    index = RegionChunkIndex(region_coordinate=RegionCoordinate(x=0, y=0))

    with pytest.raises(TypeError, match="coordinate must be a ChunkCoordinate"):
        index.contains(cast(Any, object()))

    with pytest.raises(ValueError, match="does not belong"):
        index.local_coordinate(ChunkCoordinate(x=-1, y=0))

    with pytest.raises(TypeError, match="local must be"):
        index.chunk_at(cast(Any, object()))


def test_metadata_and_index_values_are_immutable() -> None:
    chunk = ChunkMetadata.create(
        world_seed=WORLD_SEED,
        coordinate=ChunkCoordinate(x=0, y=0),
    )
    region = RegionMetadata.create(
        world_seed=WORLD_SEED,
        coordinate=RegionCoordinate(x=0, y=0),
    )
    local = RegionLocalChunkCoordinate(x=0, y=0)
    index = RegionChunkIndex(region_coordinate=RegionCoordinate(x=0, y=0))

    with pytest.raises(FrozenInstanceError):
        chunk.revision = 1  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        region.revision = 1  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        local.x = 1  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        index.region_coordinate = RegionCoordinate(x=1, y=1)  # type: ignore[misc]
