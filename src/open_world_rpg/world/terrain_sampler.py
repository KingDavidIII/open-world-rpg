"""Stable integer-only production terrain sampling.

Sampler format v1 hashes lattice points with BLAKE2b using an 8-byte digest.
The canonical payload contains, in order:

* the 4-byte-length-prefixed ASCII namespace
  ``open-world-rpg/terrain-sampler``;
* the 4-byte-length-prefixed ASCII sampler version ``v1``;
* the world seed as an unsigned 8-byte big-endian integer;
* the octave as an unsigned 4-byte big-endian integer;
* lattice x and y as a sign byte followed by a 4-byte big-endian magnitude
  length and a minimal unsigned big-endian magnitude.

The digest is interpreted as an unsigned big-endian integer and reduced modulo
2,000,001, then shifted by -1,000,000. Interpolation uses unsigned Q32
fractions, the smoothstep curve ``3t² - 2t³``, and signed half-away-from-zero
rounding. Absolute coordinates are scaled as exact rationals; floor division
and its non-negative remainder therefore apply identically to negative values.
Octaves are combined as one exact persistence-weighted integer ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Final

from open_world_rpg.world.model import WorldSpecification
from open_world_rpg.world.terrain_sampling import (
    MAX_NORMALIZED_TERRAIN_SAMPLE,
    MIN_NORMALIZED_TERRAIN_SAMPLE,
    NormalizedTerrainSample,
    TerrainGenerationConfig,
    TerrainSampleCoordinate,
)

TERRAIN_SAMPLER_NAMESPACE: Final = b"open-world-rpg/terrain-sampler"
TERRAIN_SAMPLER_VERSION: Final = b"v1"
TERRAIN_SAMPLER_DIGEST_BITS: Final = 64
TERRAIN_SAMPLER_FIXED_POINT_BITS: Final = 32
TERRAIN_SAMPLER_FIXED_POINT_ONE: Final = 1 << TERRAIN_SAMPLER_FIXED_POINT_BITS

_DIGEST_SIZE: Final = TERRAIN_SAMPLER_DIGEST_BITS // 8
_SAMPLE_VALUE_COUNT: Final = MAX_NORMALIZED_TERRAIN_SAMPLE - MIN_NORMALIZED_TERRAIN_SAMPLE + 1


def _encode_field(value: bytes) -> bytes:
    return len(value).to_bytes(4, byteorder="big", signed=False) + value


def _encode_signed_integer(value: int) -> bytes:
    sign = b"\x01" if value < 0 else b"\x00"
    magnitude = abs(value)
    byte_count = max(1, (magnitude.bit_length() + 7) // 8)
    encoded = magnitude.to_bytes(byte_count, byteorder="big", signed=False)
    return sign + _encode_field(encoded)


def _round_ratio(*, numerator: int, denominator: int) -> int:
    """Divide with deterministic half-away-from-zero rounding."""
    magnitude = (abs(numerator) + (denominator // 2)) // denominator
    return magnitude if numerator >= 0 else -magnitude


def _lattice_value(
    *,
    world_seed: int,
    octave: int,
    lattice_x: int,
    lattice_y: int,
) -> int:
    payload = b"".join(
        (
            _encode_field(TERRAIN_SAMPLER_NAMESPACE),
            _encode_field(TERRAIN_SAMPLER_VERSION),
            world_seed.to_bytes(8, byteorder="big", signed=False),
            octave.to_bytes(4, byteorder="big", signed=False),
            _encode_signed_integer(lattice_x),
            _encode_signed_integer(lattice_y),
        )
    )
    digest = blake2b(payload, digest_size=_DIGEST_SIZE).digest()
    unsigned_value = int.from_bytes(digest, byteorder="big", signed=False)
    return MIN_NORMALIZED_TERRAIN_SAMPLE + (unsigned_value % _SAMPLE_VALUE_COUNT)


def _fraction_to_fixed(*, remainder: int, denominator: int) -> tuple[int, int]:
    fixed = _round_ratio(
        numerator=remainder * TERRAIN_SAMPLER_FIXED_POINT_ONE,
        denominator=denominator,
    )
    if fixed == TERRAIN_SAMPLER_FIXED_POINT_ONE:
        return 1, 0
    return 0, fixed


def _smoothstep(value: int) -> int:
    squared = _round_ratio(
        numerator=value * value,
        denominator=TERRAIN_SAMPLER_FIXED_POINT_ONE,
    )
    factor = (3 * TERRAIN_SAMPLER_FIXED_POINT_ONE) - (2 * value)
    return _round_ratio(
        numerator=squared * factor,
        denominator=TERRAIN_SAMPLER_FIXED_POINT_ONE,
    )


def _interpolate(*, start: int, end: int, fraction: int) -> int:
    return start + _round_ratio(
        numerator=(end - start) * fraction,
        denominator=TERRAIN_SAMPLER_FIXED_POINT_ONE,
    )


def _sample_octave(
    *,
    world_seed: int,
    octave: int,
    coordinate: TerrainSampleCoordinate,
    scale_numerator: int,
    scale_denominator: int,
) -> int:
    scaled_x = coordinate.x * scale_numerator
    scaled_y = coordinate.y * scale_numerator
    lattice_x, remainder_x = divmod(scaled_x, scale_denominator)
    lattice_y, remainder_y = divmod(scaled_y, scale_denominator)
    carry_x, fraction_x = _fraction_to_fixed(
        remainder=remainder_x,
        denominator=scale_denominator,
    )
    carry_y, fraction_y = _fraction_to_fixed(
        remainder=remainder_y,
        denominator=scale_denominator,
    )
    lattice_x += carry_x
    lattice_y += carry_y
    fade_x = _smoothstep(fraction_x)
    fade_y = _smoothstep(fraction_y)

    lower = _interpolate(
        start=_lattice_value(
            world_seed=world_seed,
            octave=octave,
            lattice_x=lattice_x,
            lattice_y=lattice_y,
        ),
        end=_lattice_value(
            world_seed=world_seed,
            octave=octave,
            lattice_x=lattice_x + 1,
            lattice_y=lattice_y,
        ),
        fraction=fade_x,
    )
    upper = _interpolate(
        start=_lattice_value(
            world_seed=world_seed,
            octave=octave,
            lattice_x=lattice_x,
            lattice_y=lattice_y + 1,
        ),
        end=_lattice_value(
            world_seed=world_seed,
            octave=octave,
            lattice_x=lattice_x + 1,
            lattice_y=lattice_y + 1,
        ),
        fraction=fade_x,
    )
    return _interpolate(start=lower, end=upper, fraction=fade_y)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeterministicTerrainSampler:
    """Stateless v1 fixed-point terrain sampler."""

    def sample(
        self,
        *,
        specification: WorldSpecification,
        coordinate: TerrainSampleCoordinate,
        config: TerrainGenerationConfig,
    ) -> NormalizedTerrainSample:
        """Return a stable multi-octave sample for one absolute tile."""
        if not isinstance(specification, WorldSpecification):
            raise TypeError("specification must be a WorldSpecification.")
        if not isinstance(coordinate, TerrainSampleCoordinate):
            raise TypeError("coordinate must be a TerrainSampleCoordinate.")
        if not isinstance(config, TerrainGenerationConfig):
            raise TypeError("config must be a TerrainGenerationConfig.")

        weighted_sum = 0
        weight_sum = 0
        frequency_numerator = config.sampling_scale_numerator
        frequency_denominator = config.sampling_scale_denominator

        for octave in range(config.octave_count):
            value = _sample_octave(
                world_seed=specification.seed.value,
                octave=octave,
                coordinate=coordinate,
                scale_numerator=frequency_numerator,
                scale_denominator=frequency_denominator,
            )
            weight = config.persistence_numerator**octave * config.persistence_denominator ** (
                config.octave_count - 1 - octave
            )
            weighted_sum += value * weight
            weight_sum += weight
            frequency_numerator *= config.lacunarity_numerator
            frequency_denominator *= config.lacunarity_denominator

        result = _round_ratio(numerator=weighted_sum, denominator=weight_sum)
        return NormalizedTerrainSample.clamp(result)
