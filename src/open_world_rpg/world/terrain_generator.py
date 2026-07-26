"""Deterministic assembly of complete immutable chunk terrain payloads."""

from __future__ import annotations

from dataclasses import dataclass, field

from open_world_rpg.world.coordinates import (
    CHUNK_SIZE,
    ChunkCoordinate,
    LocalTileCoordinate,
)
from open_world_rpg.world.generation import ChunkGenerationKey, WorldGenerationStage
from open_world_rpg.world.model import (
    SUPPORTED_GENERATION_FORMAT_VERSION,
    WorldSpecification,
)
from open_world_rpg.world.terrain import (
    ChunkTerrain,
    IncompatibleTerrainDimensionsError,
    InvalidTerrainPayloadError,
    TerrainGenerationError,
    TerrainGeneratorExecutionError,
    TerrainTile,
)
from open_world_rpg.world.terrain_sampler import DeterministicTerrainSampler
from open_world_rpg.world.terrain_sampling import (
    TerrainClassifier,
    TerrainGenerationConfig,
    TerrainSampleCoordinate,
    TerrainSampler,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeterministicTerrainGenerator:
    """Stateless deterministic pipeline assembling one complete chunk."""

    sampler: TerrainSampler = field(default_factory=DeterministicTerrainSampler)
    config: TerrainGenerationConfig = field(default_factory=TerrainGenerationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.sampler, TerrainSampler) or not callable(
            getattr(self.sampler, "sample", None)
        ):
            raise TypeError("sampler must implement TerrainSampler.")
        if not isinstance(self.config, TerrainGenerationConfig):
            raise TypeError("config must be a TerrainGenerationConfig.")

    def generate(
        self,
        *,
        specification: WorldSpecification,
        coordinate: ChunkCoordinate,
    ) -> ChunkTerrain:
        """Generate all local tiles in increasing y, then increasing x."""
        self._validate_compatibility(
            specification=specification,
            coordinate=coordinate,
        )
        classifier = TerrainClassifier(config=self.config)

        try:
            tiles = tuple(
                self._generate_tile(
                    specification=specification,
                    chunk_coordinate=coordinate,
                    local_coordinate=LocalTileCoordinate(x=x, y=y),
                    classifier=classifier,
                )
                for y in range(CHUNK_SIZE)
                for x in range(CHUNK_SIZE)
            )
            terrain_seed = ChunkGenerationKey(
                world_seed=specification.seed,
                coordinate=coordinate,
                stage=WorldGenerationStage.TERRAIN,
            ).derived_seed
            return ChunkTerrain(
                world_seed=specification.seed,
                chunk_coordinate=coordinate,
                terrain_seed=terrain_seed,
                width=specification.chunk_size_tiles,
                height=specification.chunk_size_tiles,
                tiles=tiles,
                revision=0,
                generation_format_version=self.config.generation_format_version,
            )
        except TerrainGenerationError:
            raise
        except Exception as error:
            raise TerrainGeneratorExecutionError(
                f"Terrain generation failed for chunk ({coordinate.x}, {coordinate.y})."
            ) from error

    def _generate_tile(
        self,
        *,
        specification: WorldSpecification,
        chunk_coordinate: ChunkCoordinate,
        local_coordinate: LocalTileCoordinate,
        classifier: TerrainClassifier,
    ) -> TerrainTile:
        sample_coordinate = TerrainSampleCoordinate.from_chunk_and_local(
            chunk=chunk_coordinate,
            local=local_coordinate,
        )
        sample = self.sampler.sample(
            specification=specification,
            coordinate=sample_coordinate,
            config=self.config,
        )
        elevation = sample.to_elevation(self.config)
        return TerrainTile(
            coordinate=local_coordinate,
            elevation=elevation,
            terrain_type=classifier.classify(elevation),
            revision=0,
        )

    def _validate_compatibility(
        self,
        *,
        specification: WorldSpecification,
        coordinate: ChunkCoordinate,
    ) -> None:
        if not isinstance(specification, WorldSpecification):
            raise TypeError("specification must be a WorldSpecification.")
        if not isinstance(coordinate, ChunkCoordinate):
            raise TypeError("coordinate must be a ChunkCoordinate.")
        if specification.chunk_size_tiles != CHUNK_SIZE:
            raise IncompatibleTerrainDimensionsError(
                f"World specification chunk size must be {CHUNK_SIZE}."
            )
        if (
            specification.generation_format_version != SUPPORTED_GENERATION_FORMAT_VERSION
            or self.config.generation_format_version != specification.generation_format_version
        ):
            raise InvalidTerrainPayloadError(
                "Terrain configuration and world specification generation formats must match."
            )
