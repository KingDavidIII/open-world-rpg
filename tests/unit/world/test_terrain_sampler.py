"""Tests for the production deterministic fixed-point terrain sampler."""

from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError
from itertools import pairwise
from typing import Any, cast

import pytest

import open_world_rpg.world as world
from open_world_rpg.core.config import MAX_WORLD_SEED
from open_world_rpg.world import (
    CHUNK_SIZE,
    REGION_SIZE_IN_TILES,
    TERRAIN_SAMPLER_DIGEST_BITS,
    TERRAIN_SAMPLER_FIXED_POINT_BITS,
    TERRAIN_SAMPLER_FIXED_POINT_ONE,
    TERRAIN_SAMPLER_NAMESPACE,
    TERRAIN_SAMPLER_VERSION,
    ChunkCoordinate,
    DeterministicTerrainSampler,
    LocalTileCoordinate,
    NormalizedTerrainSample,
    TerrainGenerationConfig,
    TerrainSampleCoordinate,
    TerrainSampler,
    WorldSeed,
    WorldSpecification,
)


def create_specification(seed: int = 42) -> WorldSpecification:
    return WorldSpecification(name="Sampler World", seed=WorldSeed(value=seed))


def sample_at(
    *,
    seed: int = 42,
    x: int,
    y: int,
    config: TerrainGenerationConfig | None = None,
) -> int:
    return (
        DeterministicTerrainSampler()
        .sample(
            specification=create_specification(seed),
            coordinate=TerrainSampleCoordinate(x=x, y=y),
            config=TerrainGenerationConfig() if config is None else config,
        )
        .value
    )


def test_sampler_contract_constants_are_explicit_and_public() -> None:
    assert TERRAIN_SAMPLER_NAMESPACE == b"open-world-rpg/terrain-sampler"
    assert TERRAIN_SAMPLER_VERSION == b"v1"
    assert TERRAIN_SAMPLER_DIGEST_BITS == 64
    assert TERRAIN_SAMPLER_FIXED_POINT_BITS == 32
    assert TERRAIN_SAMPLER_FIXED_POINT_ONE == 1 << 32


@pytest.mark.parametrize(
    ("seed", "x", "y", "config", "expected"),
    [
        (0, 0, 0, TerrainGenerationConfig(), 370_537),
        (42, -17, 16, TerrainGenerationConfig(), -121_181),
        (42, 16, 256, TerrainGenerationConfig(), -537_052),
        (
            MAX_WORLD_SEED,
            -(10**100),
            10**100,
            TerrainGenerationConfig(),
            139_642,
        ),
        (7, 3, -9, TerrainGenerationConfig(octave_count=1), -795_327),
        (
            7,
            3,
            -9,
            TerrainGenerationConfig(
                octave_count=3,
                persistence_numerator=2,
                persistence_denominator=3,
                lacunarity_numerator=3,
                lacunarity_denominator=2,
                sampling_scale_numerator=5,
                sampling_scale_denominator=17,
            ),
            -158_183,
        ),
    ],
)
def test_known_sampler_vectors(
    seed: int,
    x: int,
    y: int,
    config: TerrainGenerationConfig,
    expected: int,
) -> None:
    assert sample_at(seed=seed, x=x, y=y, config=config) == expected


def test_sampler_conforms_to_protocol_and_is_repeatable() -> None:
    sampler = DeterministicTerrainSampler()
    arguments = {
        "specification": create_specification(),
        "coordinate": TerrainSampleCoordinate(x=-999, y=1_001),
        "config": TerrainGenerationConfig(),
    }

    assert isinstance(sampler, TerrainSampler)
    assert sampler.sample(**arguments) == sampler.sample(**arguments)
    assert isinstance(sampler.sample(**arguments), NormalizedTerrainSample)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("specification", object(), "specification must be"),
        ("coordinate", object(), "coordinate must be"),
        ("config", object(), "config must be"),
    ],
)
def test_sampler_validates_protocol_inputs(
    field_name: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "specification": create_specification(),
        "coordinate": TerrainSampleCoordinate(x=0, y=0),
        "config": TerrainGenerationConfig(),
        field_name: value,
    }

    with pytest.raises(TypeError, match=message):
        DeterministicTerrainSampler().sample(**cast(Any, arguments))


def test_world_seed_and_coordinate_separation() -> None:
    values = {
        sample_at(seed=1, x=100, y=-200),
        sample_at(seed=2, x=100, y=-200),
        sample_at(seed=1, x=101, y=-200),
        sample_at(seed=1, x=100, y=-199),
    }

    assert len(values) == 4


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (0, 0),
        (-1, -1),
        (-CHUNK_SIZE, CHUNK_SIZE),
        (-(10**1000), 10**1000),
    ],
)
def test_output_is_always_a_valid_normalized_sample(x: int, y: int) -> None:
    result = sample_at(seed=MAX_WORLD_SEED, x=x, y=y)

    assert -1_000_000 <= result <= 1_000_000


@pytest.mark.parametrize(
    "config",
    [
        TerrainGenerationConfig(octave_count=1),
        TerrainGenerationConfig(octave_count=7),
        TerrainGenerationConfig(persistence_numerator=1, persistence_denominator=10),
        TerrainGenerationConfig(lacunarity_numerator=1, lacunarity_denominator=3),
        TerrainGenerationConfig(lacunarity_numerator=7, lacunarity_denominator=2),
        TerrainGenerationConfig(
            sampling_scale_numerator=1,
            sampling_scale_denominator=1,
        ),
        TerrainGenerationConfig(
            sampling_scale_numerator=999,
            sampling_scale_denominator=1_001,
        ),
    ],
)
def test_every_supported_octave_parameter_is_used_deterministically(
    config: TerrainGenerationConfig,
) -> None:
    first = sample_at(x=-123_456_789, y=987_654_321, config=config)
    second = sample_at(x=-123_456_789, y=987_654_321, config=config)

    assert first == second
    assert -1_000_000 <= first <= 1_000_000


def test_fraction_rounding_carries_exactly_into_next_lattice_point() -> None:
    denominator = (2 * TERRAIN_SAMPLER_FIXED_POINT_ONE) + 1
    config = TerrainGenerationConfig(
        octave_count=1,
        sampling_scale_numerator=1,
        sampling_scale_denominator=denominator,
    )

    assert sample_at(x=-1, y=-1, config=config) == sample_at(
        x=0,
        y=0,
        config=config,
    )


@pytest.mark.parametrize(
    "world_x",
    [
        -REGION_SIZE_IN_TILES,
        -CHUNK_SIZE,
        0,
        CHUNK_SIZE,
        REGION_SIZE_IN_TILES,
    ],
)
def test_chunk_and_region_boundaries_use_only_absolute_world_coordinates(
    world_x: int,
) -> None:
    direct = TerrainSampleCoordinate(x=world_x, y=-world_x - 1)
    reconstructed = TerrainSampleCoordinate.from_chunk_and_local(
        chunk=direct.to_chunk(),
        local=direct.to_local_tile(),
    )
    sampler = DeterministicTerrainSampler()

    assert reconstructed == direct
    assert sampler.sample(
        specification=create_specification(),
        coordinate=direct,
        config=TerrainGenerationConfig(),
    ) == sampler.sample(
        specification=create_specification(),
        coordinate=reconstructed,
        config=TerrainGenerationConfig(),
    )


def test_adjacent_samples_cross_chunk_and_region_boundaries_without_reset() -> None:
    chunk_values = [sample_at(x=x, y=7) for x in (CHUNK_SIZE - 1, CHUNK_SIZE, CHUNK_SIZE + 1)]
    region_values = [
        sample_at(x=x, y=-11)
        for x in (
            REGION_SIZE_IN_TILES - 1,
            REGION_SIZE_IN_TILES,
            REGION_SIZE_IN_TILES + 1,
        )
    ]

    assert len(set(chunk_values)) > 1
    assert len(set(region_values)) > 1
    assert all(abs(right - left) < 1_000_000 for left, right in pairwise(chunk_values))
    assert all(abs(right - left) < 1_000_000 for left, right in pairwise(region_values))


def test_same_coordinate_from_distinct_chunk_local_routes_is_identical() -> None:
    coordinate = TerrainSampleCoordinate.from_chunk_and_local(
        chunk=ChunkCoordinate(x=-2, y=3),
        local=LocalTileCoordinate(x=15, y=0),
    )

    assert coordinate == TerrainSampleCoordinate(x=-17, y=48)
    assert sample_at(x=coordinate.x, y=coordinate.y) == sample_at(x=-17, y=48)


def test_sampler_does_not_read_or_mutate_global_random_state() -> None:
    random.seed(98_765)
    before = random.getstate()

    sample_at(x=123, y=456)

    assert random.getstate() == before


def test_sampler_is_independent_of_python_hash_seed_across_processes() -> None:
    script = (
        "from open_world_rpg.world import "
        "DeterministicTerrainSampler, TerrainGenerationConfig, "
        "TerrainSampleCoordinate, WorldSeed, WorldSpecification; "
        "print(DeterministicTerrainSampler().sample("
        "specification=WorldSpecification(name='Subprocess', seed=WorldSeed(value=42)), "
        "coordinate=TerrainSampleCoordinate(x=-17, y=16), "
        "config=TerrainGenerationConfig()).value)"
    )
    outputs: list[str] = []

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

    assert outputs == ["-121181", "-121181"]


def test_sampler_is_immutable_and_publicly_exported() -> None:
    sampler = DeterministicTerrainSampler()

    with pytest.raises((FrozenInstanceError, TypeError)):
        sampler.state = 1  # type: ignore[attr-defined]

    names = {
        "DeterministicTerrainSampler",
        "TERRAIN_SAMPLER_DIGEST_BITS",
        "TERRAIN_SAMPLER_FIXED_POINT_BITS",
        "TERRAIN_SAMPLER_FIXED_POINT_ONE",
        "TERRAIN_SAMPLER_NAMESPACE",
        "TERRAIN_SAMPLER_VERSION",
    }
    assert names <= set(world.__all__)
    assert all(hasattr(world, name) for name in names)
