"""Deterministic terrain sampling and classification contracts.

All persisted and exchanged values are integers. Rational configuration values
are stored as explicit numerator/denominator pairs, making this module a
generation-format compatibility boundary independent of platform floating
point behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from open_world_rpg.world.coordinates import (
    ChunkCoordinate,
    LocalTileCoordinate,
    WorldPosition,
)
from open_world_rpg.world.model import (
    SUPPORTED_GENERATION_FORMAT_VERSION,
    WorldSpecification,
)
from open_world_rpg.world.terrain import (
    MAX_TERRAIN_ELEVATION,
    MIN_TERRAIN_ELEVATION,
    TerrainElevation,
    TerrainType,
)

MIN_NORMALIZED_TERRAIN_SAMPLE: Final = -1_000_000
NEUTRAL_NORMALIZED_TERRAIN_SAMPLE: Final = 0
MAX_NORMALIZED_TERRAIN_SAMPLE: Final = 1_000_000


class TerrainSamplingError(RuntimeError):
    """Base error for terrain sampling failures."""


class UnsupportedTerrainGenerationConfigError(TerrainSamplingError):
    """Raised when terrain generation configuration is incompatible."""


class TerrainSamplerExecutionError(TerrainSamplingError):
    """Raised when a terrain sampler cannot produce a valid sample."""


def _require_integer(*, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainGenerationConfig:
    """Immutable v1 terrain sampling, scaling, and classification policy."""

    sea_level_metres: int = 0
    deep_water_max_metres: int = -200
    shallow_water_max_metres: int = -1
    coast_max_metres: int = 20
    plains_max_metres: int = 300
    hills_max_metres: int = 1_200
    mountain_min_metres: int = 1_201
    base_elevation_metres: int = 0
    elevation_amplitude_metres: int = 3_000
    octave_count: int = 4
    persistence_numerator: int = 1
    persistence_denominator: int = 2
    lacunarity_numerator: int = 2
    lacunarity_denominator: int = 1
    sampling_scale_numerator: int = 1
    sampling_scale_denominator: int = 64
    generation_format_version: str = SUPPORTED_GENERATION_FORMAT_VERSION

    def __post_init__(self) -> None:
        integer_fields = (
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
        )
        for name in integer_fields:
            _require_integer(name=name, value=getattr(self, name))

        if not (
            MIN_TERRAIN_ELEVATION
            <= self.deep_water_max_metres
            < self.shallow_water_max_metres
            < self.sea_level_metres
            < self.coast_max_metres
            < self.plains_max_metres
            < self.hills_max_metres
            < MAX_TERRAIN_ELEVATION
        ):
            raise UnsupportedTerrainGenerationConfigError(
                "terrain thresholds must be strictly ordered within the supported elevation range."
            )
        if self.mountain_min_metres != self.hills_max_metres + 1:
            raise UnsupportedTerrainGenerationConfigError(
                "mountain_min_metres must immediately follow hills_max_metres."
            )
        if self.elevation_amplitude_metres <= 0:
            raise UnsupportedTerrainGenerationConfigError(
                "elevation_amplitude_metres must be greater than zero."
            )
        if (
            self.base_elevation_metres - self.elevation_amplitude_metres < MIN_TERRAIN_ELEVATION
            or self.base_elevation_metres + self.elevation_amplitude_metres > MAX_TERRAIN_ELEVATION
        ):
            raise UnsupportedTerrainGenerationConfigError(
                "configured elevation output exceeds the supported terrain elevation range."
            )
        if self.octave_count <= 0:
            raise UnsupportedTerrainGenerationConfigError("octave_count must be greater than zero.")
        self._validate_ratio(
            name="persistence",
            numerator=self.persistence_numerator,
            denominator=self.persistence_denominator,
            maximum_one=True,
        )
        self._validate_ratio(
            name="lacunarity",
            numerator=self.lacunarity_numerator,
            denominator=self.lacunarity_denominator,
        )
        self._validate_ratio(
            name="sampling_scale",
            numerator=self.sampling_scale_numerator,
            denominator=self.sampling_scale_denominator,
        )
        if (
            not isinstance(self.generation_format_version, str)
            or self.generation_format_version != SUPPORTED_GENERATION_FORMAT_VERSION
        ):
            raise UnsupportedTerrainGenerationConfigError(
                "generation_format_version must match the supported v1 generation contract."
            )

    @staticmethod
    def _validate_ratio(
        *,
        name: str,
        numerator: int,
        denominator: int,
        maximum_one: bool = False,
    ) -> None:
        if numerator <= 0:
            raise UnsupportedTerrainGenerationConfigError(
                f"{name}_numerator must be greater than zero."
            )
        if denominator <= 0:
            raise UnsupportedTerrainGenerationConfigError(
                f"{name}_denominator must be greater than zero."
            )
        if maximum_one and numerator > denominator:
            raise UnsupportedTerrainGenerationConfigError(
                "persistence must be less than or equal to one."
            )


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class NormalizedTerrainSample:
    """Signed fixed-point sample in the inclusive range ±1,000,000."""

    value: int

    def __post_init__(self) -> None:
        _require_integer(name="value", value=self.value)
        if not MIN_NORMALIZED_TERRAIN_SAMPLE <= self.value <= MAX_NORMALIZED_TERRAIN_SAMPLE:
            raise ValueError(
                "value must be between "
                f"{MIN_NORMALIZED_TERRAIN_SAMPLE} and {MAX_NORMALIZED_TERRAIN_SAMPLE}."
            )

    @classmethod
    def clamp(cls, value: int) -> NormalizedTerrainSample:
        """Clamp an integer to the normalized sample range."""
        _require_integer(name="value", value=value)
        return cls(
            value=max(
                MIN_NORMALIZED_TERRAIN_SAMPLE,
                min(MAX_NORMALIZED_TERRAIN_SAMPLE, value),
            )
        )

    def to_elevation(self, config: TerrainGenerationConfig) -> TerrainElevation:
        """Scale exactly, rounding half values away from zero."""
        if not isinstance(config, TerrainGenerationConfig):
            raise TypeError("config must be a TerrainGenerationConfig.")
        product = self.value * config.elevation_amplitude_metres
        magnitude = (abs(product) + (MAX_NORMALIZED_TERRAIN_SAMPLE // 2)) // (
            MAX_NORMALIZED_TERRAIN_SAMPLE
        )
        delta = magnitude if product >= 0 else -magnitude
        return TerrainElevation(metres=config.base_elevation_metres + delta)


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainClassifier:
    """Gap-free elevation classifier governed by one immutable config."""

    config: TerrainGenerationConfig = field(default_factory=TerrainGenerationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.config, TerrainGenerationConfig):
            raise TypeError("config must be a TerrainGenerationConfig.")

    def classify(self, elevation: TerrainElevation) -> TerrainType:
        """Classify every supported elevation into one foundational type."""
        if not isinstance(elevation, TerrainElevation):
            raise TypeError("elevation must be a TerrainElevation.")
        metres = elevation.metres
        if metres <= self.config.deep_water_max_metres:
            return TerrainType.DEEP_WATER
        if metres <= self.config.shallow_water_max_metres:
            return TerrainType.SHALLOW_WATER
        if metres <= self.config.coast_max_metres:
            return TerrainType.COAST
        if metres <= self.config.plains_max_metres:
            return TerrainType.PLAINS
        if metres <= self.config.hills_max_metres:
            return TerrainType.HILLS
        return TerrainType.MOUNTAINS


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainSampleCoordinate:
    """Absolute integer tile coordinate supplied to a terrain sampler."""

    x: int
    y: int

    def __post_init__(self) -> None:
        _require_integer(name="x", value=self.x)
        _require_integer(name="y", value=self.y)

    @classmethod
    def from_chunk_and_local(
        cls,
        *,
        chunk: ChunkCoordinate,
        local: LocalTileCoordinate,
    ) -> TerrainSampleCoordinate:
        """Combine a chunk and local tile using the world coordinate contract."""
        if not isinstance(chunk, ChunkCoordinate):
            raise TypeError("chunk must be a ChunkCoordinate.")
        if not isinstance(local, LocalTileCoordinate):
            raise TypeError("local must be a LocalTileCoordinate.")
        position = WorldPosition.from_chunk_and_local(chunk=chunk, local=local)
        return cls(x=position.x, y=position.y)

    def to_world_position(self) -> WorldPosition:
        """Return the equivalent absolute world tile position."""
        return WorldPosition(x=self.x, y=self.y)

    def to_chunk(self) -> ChunkCoordinate:
        """Return the containing chunk using floor division."""
        return self.to_world_position().to_chunk()

    def to_local_tile(self) -> LocalTileCoordinate:
        """Return the non-negative local offset using modulo."""
        return self.to_world_position().to_local_tile()


@runtime_checkable
class TerrainSampler(Protocol):
    """Contract for deterministic normalized terrain sampling, not generation."""

    def sample(
        self,
        *,
        specification: WorldSpecification,
        coordinate: TerrainSampleCoordinate,
        config: TerrainGenerationConfig,
    ) -> NormalizedTerrainSample:
        """Return one deterministic normalized sample."""
