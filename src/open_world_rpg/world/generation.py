"""Stable generation identities for deterministic world content.

Seed derivation is a persistence and compatibility boundary. Version 1 hashes
a canonical payload with BLAKE2b using an 8-byte digest. The payload contains,
in order:

* the length-prefixed ASCII namespace marker
  ``open-world-rpg/world-generation``;
* the length-prefixed ASCII format version ``v1``;
* the length-prefixed key namespace, either ``region`` or ``chunk``;
* the world seed as an unsigned 8-byte big-endian integer;
* x and y as canonical signed integers, each encoded as a sign byte followed
  by a 4-byte big-endian magnitude length and a minimal big-endian magnitude;
* the generation stage as a length-prefixed ASCII enum value.

Every variable-length field uses a 4-byte unsigned big-endian length. No
platform byte order, object representation, process state, or Python hash
randomisation participates in this format.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import blake2b
from random import Random
from typing import Final

from open_world_rpg.core.config import MAX_WORLD_SEED, MIN_WORLD_SEED
from open_world_rpg.world.coordinates import ChunkCoordinate, RegionCoordinate

DERIVATION_NAMESPACE: Final = b"open-world-rpg/world-generation"
DERIVATION_VERSION: Final = b"v1"
DERIVED_SEED_BITS: Final = 64
MAX_DERIVED_SEED: Final = (1 << DERIVED_SEED_BITS) - 1
_DIGEST_SIZE: Final = DERIVED_SEED_BITS // 8
_REGION_NAMESPACE: Final = b"region"
_CHUNK_NAMESPACE: Final = b"chunk"


class WorldGenerationStage(StrEnum):
    """Independent deterministic stages of future world generation."""

    TERRAIN = "terrain"
    CLIMATE = "climate"
    BIOMES = "biomes"
    FEATURES = "features"
    RESOURCES = "resources"
    STRUCTURES = "structures"
    ENTITIES = "entities"


def _encode_field(value: bytes) -> bytes:
    return len(value).to_bytes(4, byteorder="big", signed=False) + value


def _encode_signed_integer(value: int) -> bytes:
    sign = b"\x01" if value < 0 else b"\x00"
    magnitude = abs(value)
    byte_count = max(1, (magnitude.bit_length() + 7) // 8)
    encoded_magnitude = magnitude.to_bytes(
        byte_count,
        byteorder="big",
        signed=False,
    )
    return sign + _encode_field(encoded_magnitude)


def _derive_seed(
    *,
    namespace: bytes,
    world_seed: int,
    x: int,
    y: int,
    stage: WorldGenerationStage,
) -> int:
    payload = b"".join(
        (
            _encode_field(DERIVATION_NAMESPACE),
            _encode_field(DERIVATION_VERSION),
            _encode_field(namespace),
            world_seed.to_bytes(8, byteorder="big", signed=False),
            _encode_signed_integer(x),
            _encode_signed_integer(y),
            _encode_field(stage.value.encode("ascii")),
        )
    )
    digest = blake2b(payload, digest_size=_DIGEST_SIZE).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldSeed:
    """Validated root seed from which generation identities are derived."""

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("value must be an integer.")

        if self.value < MIN_WORLD_SEED or self.value > MAX_WORLD_SEED:
            raise ValueError(f"value must be between {MIN_WORLD_SEED} and {MAX_WORLD_SEED}.")

    def for_region(
        self,
        *,
        coordinate: RegionCoordinate,
        stage: WorldGenerationStage,
    ) -> RegionGenerationKey:
        """Return a deterministic generation key for a region and stage."""
        return RegionGenerationKey(
            world_seed=self,
            coordinate=coordinate,
            stage=stage,
        )

    def for_chunk(
        self,
        *,
        coordinate: ChunkCoordinate,
        stage: WorldGenerationStage,
    ) -> ChunkGenerationKey:
        """Return a deterministic generation key for a chunk and stage."""
        return ChunkGenerationKey(
            world_seed=self,
            coordinate=coordinate,
            stage=stage,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RegionGenerationKey:
    """Stable generation identity for one region and generation stage."""

    world_seed: WorldSeed
    coordinate: RegionCoordinate
    stage: WorldGenerationStage

    def __post_init__(self) -> None:
        if not isinstance(self.world_seed, WorldSeed):
            raise TypeError("world_seed must be a WorldSeed.")

        if not isinstance(self.coordinate, RegionCoordinate):
            raise TypeError("coordinate must be a RegionCoordinate.")

        if not isinstance(self.stage, WorldGenerationStage):
            raise TypeError("stage must be a WorldGenerationStage.")

    @property
    def derived_seed(self) -> int:
        """Return the stable unsigned 64-bit seed for this generation key."""
        return _derive_seed(
            namespace=_REGION_NAMESPACE,
            world_seed=self.world_seed.value,
            x=self.coordinate.x,
            y=self.coordinate.y,
            stage=self.stage,
        )

    def create_rng(self) -> Random:
        """Return an independent RNG seeded from this generation identity."""
        return Random(self.derived_seed)


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkGenerationKey:
    """Stable generation identity for one chunk and generation stage."""

    world_seed: WorldSeed
    coordinate: ChunkCoordinate
    stage: WorldGenerationStage

    def __post_init__(self) -> None:
        if not isinstance(self.world_seed, WorldSeed):
            raise TypeError("world_seed must be a WorldSeed.")

        if not isinstance(self.coordinate, ChunkCoordinate):
            raise TypeError("coordinate must be a ChunkCoordinate.")

        if not isinstance(self.stage, WorldGenerationStage):
            raise TypeError("stage must be a WorldGenerationStage.")

    @property
    def derived_seed(self) -> int:
        """Return the stable unsigned 64-bit seed for this generation key."""
        return _derive_seed(
            namespace=_CHUNK_NAMESPACE,
            world_seed=self.world_seed.value,
            x=self.coordinate.x,
            y=self.coordinate.y,
            stage=self.stage,
        )

    def create_rng(self) -> Random:
        """Return an independent RNG seeded from this generation identity."""
        return Random(self.derived_seed)
