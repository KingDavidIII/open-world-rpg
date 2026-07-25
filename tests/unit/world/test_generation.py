"""Tests for deterministic world-generation identities."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from open_world_rpg.core.config import MAX_WORLD_SEED
from open_world_rpg.world import (
    DERIVATION_NAMESPACE,
    DERIVATION_VERSION,
    DERIVED_SEED_BITS,
    MAX_DERIVED_SEED,
    ChunkCoordinate,
    ChunkGenerationKey,
    RegionCoordinate,
    RegionGenerationKey,
    WorldGenerationStage,
    WorldSeed,
)


def test_generation_stages_have_stable_explicit_values() -> None:
    assert WorldGenerationStage.TERRAIN.value == "terrain"
    assert WorldGenerationStage.CLIMATE.value == "climate"
    assert WorldGenerationStage.BIOMES.value == "biomes"
    assert WorldGenerationStage.FEATURES.value == "features"
    assert WorldGenerationStage.RESOURCES.value == "resources"
    assert WorldGenerationStage.STRUCTURES.value == "structures"
    assert WorldGenerationStage.ENTITIES.value == "entities"


def test_derivation_contract_constants_are_public() -> None:
    assert DERIVATION_NAMESPACE == b"open-world-rpg/world-generation"
    assert DERIVATION_VERSION == b"v1"
    assert DERIVED_SEED_BITS == 64
    assert MAX_DERIVED_SEED == (1 << 64) - 1


@pytest.mark.parametrize("value", [True, False, 1.5, "1", None])
def test_world_seed_rejects_booleans_and_non_integers(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an integer"):
        WorldSeed(value=cast(Any, value))


@pytest.mark.parametrize("value", [-1, MAX_WORLD_SEED + 1])
def test_world_seed_rejects_values_outside_project_range(value: int) -> None:
    with pytest.raises(ValueError, match="value must be between"):
        WorldSeed(value=value)


def test_world_seed_accepts_range_boundaries_and_is_immutable() -> None:
    minimum = WorldSeed(value=0)
    maximum = WorldSeed(value=MAX_WORLD_SEED)

    assert minimum.value == 0
    assert maximum.value == MAX_WORLD_SEED
    assert minimum == WorldSeed(value=0)
    assert hash(minimum) == hash(WorldSeed(value=0))

    with pytest.raises(FrozenInstanceError):
        minimum.value = 1  # type: ignore[misc]


def test_world_seed_builds_region_and_chunk_keys() -> None:
    seed = WorldSeed(value=42)
    region = RegionCoordinate(x=-2, y=3)
    chunk = ChunkCoordinate(x=4, y=-5)

    assert seed.for_region(
        coordinate=region,
        stage=WorldGenerationStage.CLIMATE,
    ) == RegionGenerationKey(
        world_seed=seed,
        coordinate=region,
        stage=WorldGenerationStage.CLIMATE,
    )
    assert seed.for_chunk(
        coordinate=chunk,
        stage=WorldGenerationStage.BIOMES,
    ) == ChunkGenerationKey(
        world_seed=seed,
        coordinate=chunk,
        stage=WorldGenerationStage.BIOMES,
    )


KeyFactory = Callable[[], RegionGenerationKey | ChunkGenerationKey]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: RegionGenerationKey(
                world_seed=cast(Any, 1),
                coordinate=RegionCoordinate(x=0, y=0),
                stage=WorldGenerationStage.TERRAIN,
            ),
            "world_seed must be a WorldSeed",
        ),
        (
            lambda: RegionGenerationKey(
                world_seed=WorldSeed(value=1),
                coordinate=cast(Any, ChunkCoordinate(x=0, y=0)),
                stage=WorldGenerationStage.TERRAIN,
            ),
            "coordinate must be a RegionCoordinate",
        ),
        (
            lambda: RegionGenerationKey(
                world_seed=WorldSeed(value=1),
                coordinate=RegionCoordinate(x=0, y=0),
                stage=cast(Any, "terrain"),
            ),
            "stage must be a WorldGenerationStage",
        ),
        (
            lambda: ChunkGenerationKey(
                world_seed=cast(Any, 1),
                coordinate=ChunkCoordinate(x=0, y=0),
                stage=WorldGenerationStage.TERRAIN,
            ),
            "world_seed must be a WorldSeed",
        ),
        (
            lambda: ChunkGenerationKey(
                world_seed=WorldSeed(value=1),
                coordinate=cast(Any, RegionCoordinate(x=0, y=0)),
                stage=WorldGenerationStage.TERRAIN,
            ),
            "coordinate must be a ChunkCoordinate",
        ),
        (
            lambda: ChunkGenerationKey(
                world_seed=WorldSeed(value=1),
                coordinate=ChunkCoordinate(x=0, y=0),
                stage=cast(Any, "terrain"),
            ),
            "stage must be a WorldGenerationStage",
        ),
    ],
)
def test_generation_keys_reject_invalid_dependencies(
    factory: KeyFactory,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        factory()


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (
            lambda: RegionGenerationKey(
                world_seed=WorldSeed(value=0),
                coordinate=RegionCoordinate(x=0, y=0),
                stage=WorldGenerationStage.TERRAIN,
            ),
            11_875_099_115_532_928_456,
        ),
        (
            lambda: RegionGenerationKey(
                world_seed=WorldSeed(value=42),
                coordinate=RegionCoordinate(x=-17, y=16),
                stage=WorldGenerationStage.CLIMATE,
            ),
            322_912_376_958_628_278,
        ),
        (
            lambda: ChunkGenerationKey(
                world_seed=WorldSeed(value=0),
                coordinate=ChunkCoordinate(x=0, y=0),
                stage=WorldGenerationStage.TERRAIN,
            ),
            13_651_306_222_367_028_357,
        ),
        (
            lambda: ChunkGenerationKey(
                world_seed=WorldSeed(value=MAX_WORLD_SEED),
                coordinate=ChunkCoordinate(x=-(10**100), y=10**100),
                stage=WorldGenerationStage.ENTITIES,
            ),
            10_075_197_146_656_503_219,
        ),
    ],
)
def test_known_derivation_vectors(
    factory: KeyFactory,
    expected: int,
) -> None:
    assert factory().derived_seed == expected


def test_identical_inputs_are_reproducible_and_within_output_range() -> None:
    key = ChunkGenerationKey(
        world_seed=WorldSeed(value=123),
        coordinate=ChunkCoordinate(x=-1, y=16),
        stage=WorldGenerationStage.RESOURCES,
    )
    duplicate = ChunkGenerationKey(
        world_seed=WorldSeed(value=123),
        coordinate=ChunkCoordinate(x=-1, y=16),
        stage=WorldGenerationStage.RESOURCES,
    )

    assert key.derived_seed == duplicate.derived_seed
    assert 0 <= key.derived_seed <= MAX_DERIVED_SEED


def test_coordinates_stages_and_world_seeds_are_separated() -> None:
    values = {
        ChunkGenerationKey(
            world_seed=WorldSeed(value=1),
            coordinate=ChunkCoordinate(x=15, y=16),
            stage=WorldGenerationStage.TERRAIN,
        ).derived_seed,
        ChunkGenerationKey(
            world_seed=WorldSeed(value=1),
            coordinate=ChunkCoordinate(x=16, y=16),
            stage=WorldGenerationStage.TERRAIN,
        ).derived_seed,
        ChunkGenerationKey(
            world_seed=WorldSeed(value=1),
            coordinate=ChunkCoordinate(x=15, y=16),
            stage=WorldGenerationStage.CLIMATE,
        ).derived_seed,
        ChunkGenerationKey(
            world_seed=WorldSeed(value=2),
            coordinate=ChunkCoordinate(x=15, y=16),
            stage=WorldGenerationStage.TERRAIN,
        ).derived_seed,
    }

    assert len(values) == 4


def test_chunk_and_region_namespaces_are_separated() -> None:
    seed = WorldSeed(value=9)
    region_seed = RegionGenerationKey(
        world_seed=seed,
        coordinate=RegionCoordinate(x=-20, y=30),
        stage=WorldGenerationStage.STRUCTURES,
    ).derived_seed
    chunk_seed = ChunkGenerationKey(
        world_seed=seed,
        coordinate=ChunkCoordinate(x=-20, y=30),
        stage=WorldGenerationStage.STRUCTURES,
    ).derived_seed

    assert region_seed != chunk_seed


def test_generation_keys_support_extreme_coordinates_and_are_immutable() -> None:
    coordinate = ChunkCoordinate(x=-(10**1000), y=10**1000)
    key = ChunkGenerationKey(
        world_seed=WorldSeed(value=MAX_WORLD_SEED),
        coordinate=coordinate,
        stage=WorldGenerationStage.FEATURES,
    )

    assert 0 <= key.derived_seed <= MAX_DERIVED_SEED

    with pytest.raises(FrozenInstanceError):
        key.coordinate = ChunkCoordinate(x=0, y=0)  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RegionGenerationKey(
            world_seed=WorldSeed(value=7),
            coordinate=RegionCoordinate(x=-3, y=4),
            stage=WorldGenerationStage.BIOMES,
        ),
        lambda: ChunkGenerationKey(
            world_seed=WorldSeed(value=7),
            coordinate=ChunkCoordinate(x=-3, y=4),
            stage=WorldGenerationStage.BIOMES,
        ),
    ],
)
def test_rng_instances_are_independent_with_reproducible_sequences(
    factory: KeyFactory,
) -> None:
    key = factory()
    first = key.create_rng()
    second = key.create_rng()

    assert first is not second
    assert [first.randrange(1_000_000) for _ in range(10)] == [
        second.randrange(1_000_000) for _ in range(10)
    ]

    first.randrange(1_000_000)
    assert first.getstate() != second.getstate()


def test_derivation_is_independent_of_python_hash_seed() -> None:
    script = (
        "from open_world_rpg.world import "
        "ChunkCoordinate, ChunkGenerationKey, WorldGenerationStage, WorldSeed; "
        "print(ChunkGenerationKey("
        "world_seed=WorldSeed(value=42), "
        "coordinate=ChunkCoordinate(x=-17, y=16), "
        "stage=WorldGenerationStage.CLIMATE"
        ").derived_seed)"
    )
    outputs = []

    for hash_seed in ("1", "987654321"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(result.stdout.strip())

    assert outputs[0] == outputs[1]
