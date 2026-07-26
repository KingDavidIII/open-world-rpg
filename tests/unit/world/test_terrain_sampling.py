"""Tests for deterministic terrain sampling and classification contracts."""

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

import open_world_rpg.world as world
from open_world_rpg.world import (
    MAX_NORMALIZED_TERRAIN_SAMPLE,
    MAX_TERRAIN_ELEVATION,
    MIN_NORMALIZED_TERRAIN_SAMPLE,
    MIN_TERRAIN_ELEVATION,
    NEUTRAL_NORMALIZED_TERRAIN_SAMPLE,
    ChunkCoordinate,
    LocalTileCoordinate,
    NormalizedTerrainSample,
    TerrainClassifier,
    TerrainElevation,
    TerrainGenerationConfig,
    TerrainSampleCoordinate,
    TerrainSampler,
    TerrainSamplerExecutionError,
    TerrainSamplingError,
    TerrainType,
    UnsupportedTerrainGenerationConfigError,
    WorldSeed,
    WorldSpecification,
)


def test_default_generation_config_is_v1_integer_compatibility_policy() -> None:
    config = TerrainGenerationConfig()

    assert config == TerrainGenerationConfig()
    assert config.generation_format_version == "v1"
    assert config.base_elevation_metres == 0
    assert config.elevation_amplitude_metres == 3_000
    assert (
        config.persistence_numerator,
        config.persistence_denominator,
        config.lacunarity_numerator,
        config.lacunarity_denominator,
        config.sampling_scale_numerator,
        config.sampling_scale_denominator,
    ) == (1, 2, 2, 1, 1, 64)


@pytest.mark.parametrize(
    "field_name",
    [
        "sea_level_metres",
        "deep_water_max_metres",
        "shallow_water_max_metres",
        "coast_max_metres",
        "plains_max_metres",
        "hills_max_metres",
        "mountain_min_metres",
        "base_elevation_metres",
        "elevation_amplitude_metres",
        "octave_count",
        "persistence_numerator",
        "persistence_denominator",
        "lacunarity_numerator",
        "lacunarity_denominator",
        "sampling_scale_numerator",
        "sampling_scale_denominator",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_generation_config_rejects_non_integer_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match=f"{field_name} must be an integer"):
        replace(TerrainGenerationConfig(), **{field_name: value})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"deep_water_max_metres": -1}, "strictly ordered"),
        ({"shallow_water_max_metres": 0}, "strictly ordered"),
        ({"sea_level_metres": 20}, "strictly ordered"),
        ({"coast_max_metres": 300}, "strictly ordered"),
        ({"plains_max_metres": 1_200}, "strictly ordered"),
        ({"hills_max_metres": MAX_TERRAIN_ELEVATION}, "strictly ordered"),
        ({"deep_water_max_metres": MIN_TERRAIN_ELEVATION - 1}, "strictly ordered"),
        ({"mountain_min_metres": 1_202}, "immediately follow"),
        ({"elevation_amplitude_metres": 0}, "greater than zero"),
        (
            {
                "base_elevation_metres": MAX_TERRAIN_ELEVATION,
                "elevation_amplitude_metres": 1,
            },
            "output exceeds",
        ),
        (
            {
                "base_elevation_metres": MIN_TERRAIN_ELEVATION,
                "elevation_amplitude_metres": 1,
            },
            "output exceeds",
        ),
        ({"octave_count": 0}, "octave_count"),
        ({"persistence_numerator": 0}, "persistence_numerator"),
        ({"persistence_denominator": 0}, "persistence_denominator"),
        ({"persistence_numerator": 3}, "less than or equal"),
        ({"lacunarity_numerator": 0}, "lacunarity_numerator"),
        ({"lacunarity_denominator": 0}, "lacunarity_denominator"),
        ({"sampling_scale_numerator": 0}, "sampling_scale_numerator"),
        ({"sampling_scale_denominator": 0}, "sampling_scale_denominator"),
        ({"generation_format_version": "v2"}, "supported v1"),
        ({"generation_format_version": cast(Any, b"v1")}, "supported v1"),
    ],
)
def test_generation_config_rejects_incompatible_values(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(UnsupportedTerrainGenerationConfigError, match=message):
        replace(TerrainGenerationConfig(), **changes)


def test_custom_generation_config_accepts_exact_rational_values() -> None:
    config = TerrainGenerationConfig(
        base_elevation_metres=100,
        elevation_amplitude_metres=4_000,
        octave_count=8,
        persistence_numerator=3,
        persistence_denominator=4,
        lacunarity_numerator=5,
        lacunarity_denominator=2,
        sampling_scale_numerator=3,
        sampling_scale_denominator=128,
    )

    assert config.persistence_numerator == 3
    assert config.lacunarity_denominator == 2
    assert config.sampling_scale_denominator == 128


@pytest.mark.parametrize(
    "value",
    [
        MIN_NORMALIZED_TERRAIN_SAMPLE,
        NEUTRAL_NORMALIZED_TERRAIN_SAMPLE,
        MAX_NORMALIZED_TERRAIN_SAMPLE,
    ],
)
def test_normalized_sample_accepts_inclusive_boundaries(value: int) -> None:
    assert NormalizedTerrainSample(value=value).value == value


@pytest.mark.parametrize("value", [True, False, 0.0, "0"])
def test_normalized_sample_rejects_non_integers(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an integer"):
        NormalizedTerrainSample(value=cast(Any, value))
    with pytest.raises(TypeError, match="value must be an integer"):
        NormalizedTerrainSample.clamp(cast(Any, value))


@pytest.mark.parametrize(
    "value",
    [MIN_NORMALIZED_TERRAIN_SAMPLE - 1, MAX_NORMALIZED_TERRAIN_SAMPLE + 1],
)
def test_normalized_sample_rejects_out_of_range(value: int) -> None:
    with pytest.raises(ValueError, match="value must be between"):
        NormalizedTerrainSample(value=value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-(10**100), MIN_NORMALIZED_TERRAIN_SAMPLE),
        (MIN_NORMALIZED_TERRAIN_SAMPLE, MIN_NORMALIZED_TERRAIN_SAMPLE),
        (123, 123),
        (MAX_NORMALIZED_TERRAIN_SAMPLE, MAX_NORMALIZED_TERRAIN_SAMPLE),
        (10**100, MAX_NORMALIZED_TERRAIN_SAMPLE),
    ],
)
def test_normalized_sample_clamp_is_explicit_and_deterministic(
    value: int,
    expected: int,
) -> None:
    assert NormalizedTerrainSample.clamp(value).value == expected


def test_normalized_sample_scales_with_integer_rounding_away_from_zero() -> None:
    config = TerrainGenerationConfig(
        base_elevation_metres=10,
        elevation_amplitude_metres=1,
    )

    assert NormalizedTerrainSample(value=499_999).to_elevation(config).metres == 10
    assert NormalizedTerrainSample(value=500_000).to_elevation(config).metres == 11
    assert NormalizedTerrainSample(value=-499_999).to_elevation(config).metres == 10
    assert NormalizedTerrainSample(value=-500_000).to_elevation(config).metres == 9


def test_normalized_sample_scaling_has_exact_endpoints_and_no_float_contract() -> None:
    config = TerrainGenerationConfig(
        base_elevation_metres=-100,
        elevation_amplitude_metres=12_345,
    )

    assert NormalizedTerrainSample(value=-1_000_000).to_elevation(config) == TerrainElevation(
        metres=-12_445
    )
    assert NormalizedTerrainSample(value=0).to_elevation(config) == TerrainElevation(metres=-100)
    assert NormalizedTerrainSample(value=1_000_000).to_elevation(config) == TerrainElevation(
        metres=12_245
    )
    with pytest.raises(TypeError, match="TerrainGenerationConfig"):
        NormalizedTerrainSample(value=0).to_elevation(cast(Any, object()))


@pytest.mark.parametrize(
    ("metres", "expected"),
    [
        (MIN_TERRAIN_ELEVATION, TerrainType.DEEP_WATER),
        (-200, TerrainType.DEEP_WATER),
        (-199, TerrainType.SHALLOW_WATER),
        (-1, TerrainType.SHALLOW_WATER),
        (0, TerrainType.COAST),
        (20, TerrainType.COAST),
        (21, TerrainType.PLAINS),
        (300, TerrainType.PLAINS),
        (301, TerrainType.HILLS),
        (1_200, TerrainType.HILLS),
        (1_201, TerrainType.MOUNTAINS),
        (MAX_TERRAIN_ELEVATION, TerrainType.MOUNTAINS),
    ],
)
def test_classifier_is_gap_free_at_every_default_boundary(
    metres: int,
    expected: TerrainType,
) -> None:
    assert TerrainClassifier().classify(TerrainElevation(metres=metres)) is expected


def test_classifier_uses_custom_config_and_validates_types() -> None:
    classifier = TerrainClassifier(
        config=TerrainGenerationConfig(
            deep_water_max_metres=-500,
            shallow_water_max_metres=-101,
            sea_level_metres=-100,
            coast_max_metres=50,
            plains_max_metres=500,
            hills_max_metres=2_000,
            mountain_min_metres=2_001,
        )
    )

    assert classifier.classify(TerrainElevation(metres=-500)) is TerrainType.DEEP_WATER
    assert classifier.classify(TerrainElevation(metres=-499)) is TerrainType.SHALLOW_WATER
    with pytest.raises(TypeError, match="config must be"):
        TerrainClassifier(config=cast(Any, object()))
    with pytest.raises(TypeError, match="elevation must be"):
        classifier.classify(cast(Any, 0))


@pytest.mark.parametrize("field_name", ["x", "y"])
@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_sample_coordinate_rejects_non_integer_axes(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {"x": 0, "y": 0, field_name: value}
    with pytest.raises(TypeError, match=f"{field_name} must be an integer"):
        TerrainSampleCoordinate(**cast(Any, values))


@pytest.mark.parametrize(
    ("chunk", "local", "expected"),
    [
        (
            ChunkCoordinate(x=0, y=0),
            LocalTileCoordinate(x=0, y=0),
            TerrainSampleCoordinate(x=0, y=0),
        ),
        (
            ChunkCoordinate(x=-1, y=-1),
            LocalTileCoordinate(x=15, y=15),
            TerrainSampleCoordinate(x=-1, y=-1),
        ),
        (
            ChunkCoordinate(x=-1, y=2),
            LocalTileCoordinate(x=0, y=7),
            TerrainSampleCoordinate(x=-16, y=39),
        ),
        (
            ChunkCoordinate(x=10**100, y=-(10**100)),
            LocalTileCoordinate(x=15, y=0),
            TerrainSampleCoordinate(x=(16 * 10**100) + 15, y=-(16 * 10**100)),
        ),
    ],
)
def test_sample_coordinate_chunk_local_round_trip(
    chunk: ChunkCoordinate,
    local: LocalTileCoordinate,
    expected: TerrainSampleCoordinate,
) -> None:
    coordinate = TerrainSampleCoordinate.from_chunk_and_local(chunk=chunk, local=local)

    assert coordinate == expected
    assert coordinate.to_chunk() == chunk
    assert coordinate.to_local_tile() == local
    assert (coordinate.to_world_position().x, coordinate.to_world_position().y) == (
        expected.x,
        expected.y,
    )


def test_sample_coordinate_factory_validates_component_types() -> None:
    with pytest.raises(TypeError, match="chunk must be"):
        TerrainSampleCoordinate.from_chunk_and_local(
            chunk=cast(Any, object()),
            local=LocalTileCoordinate(x=0, y=0),
        )
    with pytest.raises(TypeError, match="local must be"):
        TerrainSampleCoordinate.from_chunk_and_local(
            chunk=ChunkCoordinate(x=0, y=0),
            local=cast(Any, object()),
        )


class ExampleSampler:
    def sample(
        self,
        *,
        specification: WorldSpecification,
        coordinate: TerrainSampleCoordinate,
        config: TerrainGenerationConfig,
    ) -> NormalizedTerrainSample:
        del specification, config
        return NormalizedTerrainSample.clamp(coordinate.x + coordinate.y)


class NotASampler:
    pass


def test_runtime_checkable_sampler_is_distinct_from_generator_contract() -> None:
    sampler = ExampleSampler()
    specification = WorldSpecification(
        name="Sampling World",
        seed=WorldSeed(value=42),
    )

    assert isinstance(sampler, TerrainSampler)
    assert not isinstance(NotASampler(), TerrainSampler)
    assert sampler.sample(
        specification=specification,
        coordinate=TerrainSampleCoordinate(x=-4, y=7),
        config=TerrainGenerationConfig(),
    ) == NormalizedTerrainSample(value=3)


def test_sampling_models_are_immutable() -> None:
    values: list[tuple[object, str]] = [
        (TerrainGenerationConfig(), "octave_count"),
        (NormalizedTerrainSample(value=0), "value"),
        (TerrainClassifier(), "config"),
        (TerrainSampleCoordinate(x=0, y=0), "x"),
    ]

    for value, field_name in values:
        with pytest.raises(FrozenInstanceError):
            setattr(value, field_name, 99)


def test_sampling_errors_and_complete_public_exports() -> None:
    assert issubclass(UnsupportedTerrainGenerationConfigError, TerrainSamplingError)
    assert issubclass(TerrainSamplerExecutionError, TerrainSamplingError)
    names = {
        "MAX_NORMALIZED_TERRAIN_SAMPLE",
        "MIN_NORMALIZED_TERRAIN_SAMPLE",
        "NEUTRAL_NORMALIZED_TERRAIN_SAMPLE",
        "NormalizedTerrainSample",
        "TerrainClassifier",
        "TerrainGenerationConfig",
        "TerrainSampleCoordinate",
        "TerrainSampler",
        "TerrainSamplerExecutionError",
        "TerrainSamplingError",
        "UnsupportedTerrainGenerationConfigError",
    }

    assert names <= set(world.__all__)
    assert all(hasattr(world, name) for name in names)
