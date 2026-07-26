"""Tests for deterministic complete chunk terrain generation."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from hashlib import blake2b
from typing import Any, cast

import pytest

import open_world_rpg.world as world
from open_world_rpg.core.config import MAX_WORLD_SEED
from open_world_rpg.world import (
    CHUNK_SIZE,
    ChunkCoordinate,
    ChunkGenerationKey,
    ChunkTerrain,
    DeterministicTerrainGenerator,
    DeterministicTerrainSampler,
    IncompatibleTerrainDimensionsError,
    InvalidTerrainPayloadError,
    LocalTileCoordinate,
    NormalizedTerrainSample,
    TerrainGenerationConfig,
    TerrainGenerator,
    TerrainGeneratorExecutionError,
    TerrainSampleCoordinate,
    TerrainType,
    WorldGenerationStage,
    WorldSeed,
    WorldSpecification,
)


def create_specification(seed: int = 42) -> WorldSpecification:
    return WorldSpecification(name="Generated World", seed=WorldSeed(value=seed))


def terrain_fingerprint(terrain: ChunkTerrain) -> str:
    """Return the test-vector BLAKE2b-256 canonical row-major fingerprint."""
    digest = blake2b(digest_size=32)
    for tile in terrain:
        terrain_type = tile.terrain_type.value.encode("ascii")
        digest.update(tile.coordinate.x.to_bytes(2, byteorder="big", signed=False))
        digest.update(tile.coordinate.y.to_bytes(2, byteorder="big", signed=False))
        digest.update(tile.elevation.metres.to_bytes(4, byteorder="big", signed=True))
        digest.update(len(terrain_type).to_bytes(1, byteorder="big", signed=False))
        digest.update(terrain_type)
        digest.update(tile.revision.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


class ConstantSampler:
    def __init__(self, value: int) -> None:
        self.value = value
        self.calls: list[TerrainSampleCoordinate] = []

    def sample(
        self,
        *,
        specification: WorldSpecification,
        coordinate: TerrainSampleCoordinate,
        config: TerrainGenerationConfig,
    ) -> NormalizedTerrainSample:
        del specification, config
        self.calls.append(coordinate)
        return NormalizedTerrainSample(value=self.value)


class CategorySampler:
    _VALUES = (-100_000, -33_333, 0, 33_333, 200_000, 500_000)

    def sample(
        self,
        *,
        specification: WorldSpecification,
        coordinate: TerrainSampleCoordinate,
        config: TerrainGenerationConfig,
    ) -> NormalizedTerrainSample:
        del specification, config
        return NormalizedTerrainSample(value=self._VALUES[coordinate.x % len(self._VALUES)])


def test_default_construction_is_immutable_and_conforms_to_protocol() -> None:
    generator = DeterministicTerrainGenerator()

    assert isinstance(generator, TerrainGenerator)
    assert isinstance(generator.sampler, DeterministicTerrainSampler)
    assert generator.config == TerrainGenerationConfig()
    with pytest.raises(FrozenInstanceError):
        generator.config = TerrainGenerationConfig(octave_count=1)  # type: ignore[misc]


def test_injected_sampler_and_configuration_are_used_without_mutation() -> None:
    sampler = ConstantSampler(500_000)
    config = TerrainGenerationConfig(
        base_elevation_metres=100,
        elevation_amplitude_metres=2_000,
        octave_count=1,
    )
    specification = create_specification()
    generator = DeterministicTerrainGenerator(sampler=sampler, config=config)

    terrain = generator.generate(
        specification=specification,
        coordinate=ChunkCoordinate(x=-1, y=2),
    )

    assert len(sampler.calls) == CHUNK_SIZE**2
    assert all(tile.elevation.metres == 1_100 for tile in terrain)
    assert all(tile.terrain_type is TerrainType.HILLS for tile in terrain)
    assert generator.config is config
    assert specification == create_specification()


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("sampler", object(), "sampler must implement"),
        (
            "sampler",
            cast(Any, type("BadSampler", (), {"sample": None})()),
            "sampler must implement",
        ),
        ("config", object(), "config must be"),
    ],
)
def test_constructor_rejects_invalid_dependencies(
    field_name: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        DeterministicTerrainGenerator(**cast(Any, {field_name: value}))


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("specification", object(), "specification must be"),
        ("coordinate", object(), "coordinate must be"),
    ],
)
def test_generate_rejects_invalid_inputs(
    field_name: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "specification": create_specification(),
        "coordinate": ChunkCoordinate(x=0, y=0),
        field_name: value,
    }

    with pytest.raises(TypeError, match=message):
        DeterministicTerrainGenerator().generate(**cast(Any, arguments))


def test_generation_has_exact_row_major_coverage_and_zero_revisions() -> None:
    coordinate = ChunkCoordinate(x=-17, y=16)
    terrain = DeterministicTerrainGenerator().generate(
        specification=create_specification(),
        coordinate=coordinate,
    )
    expected_coordinates = tuple(
        LocalTileCoordinate(x=x, y=y) for y in range(CHUNK_SIZE) for x in range(CHUNK_SIZE)
    )

    assert len(terrain) == CHUNK_SIZE**2 == 256
    assert tuple(tile.coordinate for tile in terrain) == expected_coordinates
    assert len(set(expected_coordinates)) == 256
    assert terrain.revision == 0
    assert all(tile.revision == 0 for tile in terrain)
    assert terrain.width == terrain.height == CHUNK_SIZE


@pytest.mark.parametrize(
    "coordinate",
    [
        ChunkCoordinate(x=-1, y=-1),
        ChunkCoordinate(x=0, y=0),
        ChunkCoordinate(x=15, y=15),
        ChunkCoordinate(x=16, y=16),
        ChunkCoordinate(x=-(10**100), y=10**100),
    ],
)
def test_negative_boundary_and_extreme_chunks_preserve_identity(
    coordinate: ChunkCoordinate,
) -> None:
    specification = create_specification(MAX_WORLD_SEED)
    terrain = DeterministicTerrainGenerator().generate(
        specification=specification,
        coordinate=coordinate,
    )
    expected_seed = ChunkGenerationKey(
        world_seed=specification.seed,
        coordinate=coordinate,
        stage=WorldGenerationStage.TERRAIN,
    ).derived_seed

    assert terrain.chunk_coordinate == coordinate
    assert terrain.world_seed == specification.seed
    assert terrain.terrain_seed == expected_seed
    assert len(terrain) == 256


def test_injected_deterministic_fixture_reaches_every_terrain_type() -> None:
    terrain = DeterministicTerrainGenerator(sampler=CategorySampler()).generate(
        specification=create_specification(),
        coordinate=ChunkCoordinate(x=0, y=0),
    )

    assert {tile.terrain_type for tile in terrain} == set(TerrainType)
    assert all(count > 0 for count in terrain.terrain_type_counts.values())


CUSTOM_CONFIG = TerrainGenerationConfig(
    base_elevation_metres=250,
    elevation_amplitude_metres=5_000,
    octave_count=3,
    persistence_numerator=2,
    persistence_denominator=3,
    lacunarity_numerator=3,
    lacunarity_denominator=2,
    sampling_scale_numerator=5,
    sampling_scale_denominator=97,
)


@pytest.mark.parametrize(
    (
        "seed",
        "coordinate",
        "config",
        "expected_seed",
        "expected_minimum",
        "expected_maximum",
        "expected_counts",
        "expected_tiles",
        "expected_fingerprint",
    ),
    [
        (
            0,
            ChunkCoordinate(x=0, y=0),
            TerrainGenerationConfig(),
            13_651_306_222_367_028_357,
            425,
            1_112,
            {TerrainType.HILLS: 256},
            (
                (0, 0, 1_112, TerrainType.HILLS),
                (7, 11, 843, TerrainType.HILLS),
                (15, 15, 656, TerrainType.HILLS),
            ),
            "a59f286df0b061a299fba72204dccad4c6d00f2603ca42327491a9e49f202443",
        ),
        (
            42,
            ChunkCoordinate(x=-17, y=16),
            TerrainGenerationConfig(),
            12_075_653_072_142_629_653,
            -340,
            870,
            {
                TerrainType.DEEP_WATER: 8,
                TerrainType.SHALLOW_WATER: 18,
                TerrainType.COAST: 2,
                TerrainType.PLAINS: 56,
                TerrainType.HILLS: 172,
            },
            (
                (0, 0, -340, TerrainType.DEEP_WATER),
                (7, 11, 535, TerrainType.HILLS),
                (15, 15, 665, TerrainType.HILLS),
            ),
            "f87fcf86e6255b633e57208b3ade21765db23884fa28a9d828695a76556bbad8",
        ),
        (
            7,
            ChunkCoordinate(x=3, y=-9),
            CUSTOM_CONFIG,
            16_805_337_723_853_242_028,
            -1_942,
            334,
            {
                TerrainType.DEEP_WATER: 231,
                TerrainType.SHALLOW_WATER: 14,
                TerrainType.COAST: 1,
                TerrainType.PLAINS: 9,
                TerrainType.HILLS: 1,
            },
            (
                (0, 0, 334, TerrainType.HILLS),
                (7, 11, -1_317, TerrainType.DEEP_WATER),
                (15, 15, -1_181, TerrainType.DEEP_WATER),
            ),
            "a6ccfa19d1b863e3830a9362e71e6522cd1f5222682d06f1b5863f795e3c6d01",
        ),
        (
            MAX_WORLD_SEED,
            ChunkCoordinate(x=-(10**100), y=10**100),
            TerrainGenerationConfig(),
            4_461_314_968_234_582_812,
            -2_448,
            -1_139,
            {TerrainType.DEEP_WATER: 256},
            (
                (0, 0, -2_441, TerrainType.DEEP_WATER),
                (7, 11, -1_707, TerrainType.DEEP_WATER),
                (15, 15, -1_139, TerrainType.DEEP_WATER),
            ),
            "555a9c1a6987c1448dddce518a730bfa64140fde592e5f66535a7ab87118208b",
        ),
    ],
)
def test_stable_known_chunk_vectors(
    seed: int,
    coordinate: ChunkCoordinate,
    config: TerrainGenerationConfig,
    expected_seed: int,
    expected_minimum: int,
    expected_maximum: int,
    expected_counts: Mapping[TerrainType, int],
    expected_tiles: tuple[tuple[int, int, int, TerrainType], ...],
    expected_fingerprint: str,
) -> None:
    terrain = DeterministicTerrainGenerator(config=config).generate(
        specification=create_specification(seed),
        coordinate=coordinate,
    )

    assert terrain.terrain_seed == expected_seed
    assert terrain.minimum_elevation.metres == expected_minimum
    assert terrain.maximum_elevation.metres == expected_maximum
    assert {
        key: value for key, value in terrain.terrain_type_counts.items() if value
    } == expected_counts
    assert (
        tuple(
            (
                x,
                y,
                terrain.tile_at(LocalTileCoordinate(x=x, y=y)).elevation.metres,
                terrain.tile_at(LocalTileCoordinate(x=x, y=y)).terrain_type,
            )
            for x, y, _, _ in expected_tiles
        )
        == expected_tiles
    )
    assert terrain_fingerprint(terrain) == expected_fingerprint


def test_equivalent_generators_are_repeatable_and_leave_inputs_unchanged() -> None:
    specification = create_specification(123)
    coordinate = ChunkCoordinate(x=-5, y=8)
    config = TerrainGenerationConfig(octave_count=2)
    first_generator = DeterministicTerrainGenerator(config=config)
    second_generator = DeterministicTerrainGenerator(config=config)

    first = first_generator.generate(specification=specification, coordinate=coordinate)
    second = second_generator.generate(specification=specification, coordinate=coordinate)

    assert first == second
    assert terrain_fingerprint(first) == terrain_fingerprint(second)
    assert first_generator == second_generator
    assert specification == create_specification(123)
    assert config == TerrainGenerationConfig(octave_count=2)


class FailingSampler:
    def __init__(self, error: Exception, fail_after: int = 0) -> None:
        self.error = error
        self.fail_after = fail_after
        self.calls = 0

    def sample(
        self,
        *,
        specification: WorldSpecification,
        coordinate: TerrainSampleCoordinate,
        config: TerrainGenerationConfig,
    ) -> NormalizedTerrainSample:
        del specification, coordinate, config
        self.calls += 1
        if self.calls > self.fail_after:
            raise self.error
        return NormalizedTerrainSample(value=0)


def test_unexpected_sampler_failure_is_wrapped_with_chunk_context_and_cause() -> None:
    original = ValueError("sampler exploded")
    sampler = FailingSampler(original, fail_after=3)

    with pytest.raises(
        TerrainGeneratorExecutionError,
        match=r"chunk \(-4, 9\)",
    ) as caught:
        DeterministicTerrainGenerator(sampler=sampler).generate(
            specification=create_specification(),
            coordinate=ChunkCoordinate(x=-4, y=9),
        )

    assert caught.value.__cause__ is original
    assert sampler.calls == 4


def test_existing_terrain_domain_errors_remain_identifiable() -> None:
    original = InvalidTerrainPayloadError("invalid sampled payload")

    with pytest.raises(InvalidTerrainPayloadError) as caught:
        DeterministicTerrainGenerator(sampler=FailingSampler(original)).generate(
            specification=create_specification(),
            coordinate=ChunkCoordinate(x=1, y=2),
        )

    assert caught.value is original


def test_incompatible_dimensions_and_versions_fail_before_sampling() -> None:
    sampler = ConstantSampler(0)
    specification = create_specification()
    object.__setattr__(specification, "chunk_size_tiles", CHUNK_SIZE + 1)

    with pytest.raises(IncompatibleTerrainDimensionsError, match="chunk size"):
        DeterministicTerrainGenerator(sampler=sampler).generate(
            specification=specification,
            coordinate=ChunkCoordinate(x=0, y=0),
        )
    assert sampler.calls == []

    specification = create_specification()
    object.__setattr__(specification, "generation_format_version", "v2")
    with pytest.raises(InvalidTerrainPayloadError, match="formats must match"):
        DeterministicTerrainGenerator(sampler=sampler).generate(
            specification=specification,
            coordinate=ChunkCoordinate(x=0, y=0),
        )
    assert sampler.calls == []

    specification = create_specification()
    config = TerrainGenerationConfig()
    object.__setattr__(config, "generation_format_version", "v2")
    with pytest.raises(InvalidTerrainPayloadError, match="formats must match"):
        DeterministicTerrainGenerator(sampler=sampler, config=config).generate(
            specification=specification,
            coordinate=ChunkCoordinate(x=0, y=0),
        )
    assert sampler.calls == []


def test_chunk_fingerprint_is_independent_of_python_hash_seed() -> None:
    script = """
from hashlib import blake2b
from open_world_rpg.world import (
    ChunkCoordinate,
    DeterministicTerrainGenerator,
    WorldSeed,
    WorldSpecification,
)
terrain = DeterministicTerrainGenerator().generate(
    specification=WorldSpecification(name="Subprocess", seed=WorldSeed(value=42)),
    coordinate=ChunkCoordinate(x=-17, y=16),
)
digest = blake2b(digest_size=32)
for tile in terrain:
    kind = tile.terrain_type.value.encode("ascii")
    digest.update(tile.coordinate.x.to_bytes(2, "big"))
    digest.update(tile.coordinate.y.to_bytes(2, "big"))
    digest.update(tile.elevation.metres.to_bytes(4, "big", signed=True))
    digest.update(len(kind).to_bytes(1, "big"))
    digest.update(kind)
    digest.update(tile.revision.to_bytes(8, "big"))
print(digest.hexdigest())
"""
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

    assert outputs == [
        "f87fcf86e6255b633e57208b3ade21765db23884fa28a9d828695a76556bbad8",
        "f87fcf86e6255b633e57208b3ade21765db23884fa28a9d828695a76556bbad8",
    ]


def test_generator_is_completely_exported() -> None:
    assert "DeterministicTerrainGenerator" in world.__all__
    assert world.DeterministicTerrainGenerator is DeterministicTerrainGenerator
