"""Controlled terrain generation and in-memory caching service."""

from __future__ import annotations

from dataclasses import dataclass

from open_world_rpg.world.coordinates import ChunkCoordinate
from open_world_rpg.world.model import WorldSpecification
from open_world_rpg.world.terrain import (
    ChunkTerrain,
    TerrainGenerator,
)
from open_world_rpg.world.terrain_generator import DeterministicTerrainGenerator
from open_world_rpg.world.terrain_repository import (
    IncompatibleTerrainRepositoryScopeError,
    InMemoryTerrainRepository,
    TerrainRepository,
    TerrainRepositoryError,
    TerrainRepositoryScope,
)
from open_world_rpg.world.terrain_sampling import TerrainGenerationConfig


class TerrainGenerationServiceError(RuntimeError):
    """Raised when coordinated generation cannot publish terrain."""


class TerrainAlreadyGeneratedError(TerrainGenerationServiceError):
    """Raised when generate_new targets an occupied coordinate."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainGenerationServiceSnapshot:
    """Immutable diagnostics; counter-only changes never alter repository revision."""

    repository_revision: int
    cached_chunk_count: int
    cache_hits: int
    cache_misses: int
    successful_generations: int
    failed_generations: int
    evictions: int

    def __post_init__(self) -> None:
        for name in (
            "repository_revision",
            "cached_chunk_count",
            "cache_hits",
            "cache_misses",
            "successful_generations",
            "failed_generations",
            "evictions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be greater than or equal to zero.")


class TerrainGenerationService:
    """Coordinate one world's deterministic generation and terrain cache."""

    __slots__ = (
        "_cache_hits",
        "_cache_misses",
        "_evictions",
        "_failed_generations",
        "_successful_generations",
        "config",
        "generator",
        "repository",
        "specification",
    )

    def __init__(
        self,
        *,
        specification: WorldSpecification,
        config: TerrainGenerationConfig,
        generator: TerrainGenerator | None = None,
        repository: TerrainRepository | None = None,
    ) -> None:
        if not isinstance(specification, WorldSpecification):
            raise TypeError("specification must be a WorldSpecification.")
        if not isinstance(config, TerrainGenerationConfig):
            raise TypeError("config must be a TerrainGenerationConfig.")

        expected_scope = TerrainRepositoryScope(
            world_seed=specification.seed,
            chunk_size_tiles=specification.chunk_size_tiles,
            generation_format_version=specification.generation_format_version,
            terrain_config=config,
        )
        resolved_generator: TerrainGenerator = (
            DeterministicTerrainGenerator(config=config) if generator is None else generator
        )
        if not isinstance(resolved_generator, TerrainGenerator) or not callable(
            getattr(resolved_generator, "generate", None)
        ):
            raise TypeError("generator must implement TerrainGenerator.")
        generator_config = getattr(resolved_generator, "config", config)
        if generator_config != config:
            raise IncompatibleTerrainRepositoryScopeError(
                "Generator configuration must match the service configuration."
            )

        resolved_repository: TerrainRepository = (
            InMemoryTerrainRepository(scope=expected_scope) if repository is None else repository
        )
        if not isinstance(resolved_repository, TerrainRepository):
            raise TypeError("repository must implement TerrainRepository.")
        if resolved_repository.scope != expected_scope:
            raise IncompatibleTerrainRepositoryScopeError(
                "Repository scope must match the service world and terrain configuration."
            )

        self.specification = specification
        self.config = config
        self.generator = resolved_generator
        self.repository = resolved_repository
        self._cache_hits = 0
        self._cache_misses = 0
        self._successful_generations = 0
        self._failed_generations = 0
        self._evictions = 0

    def get(self, coordinate: ChunkCoordinate) -> ChunkTerrain:
        """Return cached terrain without changing service counters."""
        self._validate_coordinate(coordinate)
        return self.repository.get(coordinate)

    def contains(self, coordinate: ChunkCoordinate) -> bool:
        """Return cache membership without changing service counters."""
        self._validate_coordinate(coordinate)
        return self.repository.contains(coordinate)

    def get_or_generate(self, coordinate: ChunkCoordinate) -> ChunkTerrain:
        """Return a cache hit or generate, validate, and publish one miss."""
        self._validate_coordinate(coordinate)
        if self.repository.contains(coordinate):
            self._cache_hits += 1
            return self.repository.get(coordinate)
        self._cache_misses += 1
        return self._generate_and_store(coordinate)

    def generate_new(self, coordinate: ChunkCoordinate) -> ChunkTerrain:
        """Generate only when no terrain is already cached."""
        self._validate_coordinate(coordinate)
        if self.repository.contains(coordinate):
            raise TerrainAlreadyGeneratedError(
                f"Terrain for chunk ({coordinate.x}, {coordinate.y}) is already generated."
            )
        return self._generate_and_store(coordinate)

    def evict(self, coordinate: ChunkCoordinate) -> None:
        """Evict one cached chunk, counting only successful removals."""
        self._validate_coordinate(coordinate)
        if not self.repository.contains(coordinate):
            return
        self.repository.remove(coordinate)
        self._evictions += 1

    def clear(self) -> None:
        """Clear cached terrain without changing generation counters."""
        self.repository.clear()

    def snapshot(self) -> TerrainGenerationServiceSnapshot:
        """Return immutable service counters and current cache state."""
        repository_snapshot = self.repository.snapshot()
        return TerrainGenerationServiceSnapshot(
            repository_revision=repository_snapshot.revision,
            cached_chunk_count=repository_snapshot.chunk_count,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            successful_generations=self._successful_generations,
            failed_generations=self._failed_generations,
            evictions=self._evictions,
        )

    def _generate_and_store(self, coordinate: ChunkCoordinate) -> ChunkTerrain:
        try:
            terrain = self.generator.generate(
                specification=self.specification,
                coordinate=coordinate,
            )
        except Exception as error:
            self._failed_generations += 1
            raise TerrainGenerationServiceError(
                f"Terrain generation failed for chunk ({coordinate.x}, {coordinate.y})."
            ) from error

        if not isinstance(terrain, ChunkTerrain) or terrain.chunk_coordinate != coordinate:
            self._failed_generations += 1
            incompatibility = IncompatibleTerrainRepositoryScopeError(
                "Generated terrain must match the requested chunk coordinate."
            )
            raise TerrainGenerationServiceError(
                f"Terrain generation failed for chunk ({coordinate.x}, {coordinate.y})."
            ) from incompatibility

        try:
            self.repository.store(terrain)
        except TerrainRepositoryError as error:
            raise TerrainGenerationServiceError(
                f"Generated terrain could not be stored for chunk ({coordinate.x}, {coordinate.y})."
            ) from error

        self._successful_generations += 1
        return terrain

    @staticmethod
    def _validate_coordinate(coordinate: object) -> None:
        if not isinstance(coordinate, ChunkCoordinate):
            raise TypeError("coordinate must be a ChunkCoordinate.")
