"""Tests for controlled terrain generation and cache coordination."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, NoReturn, cast

import pytest

import open_world_rpg.world as world
from open_world_rpg.world import (
    CHUNK_SIZE,
    ChunkCoordinate,
    ChunkTerrain,
    DeterministicTerrainGenerator,
    IncompatibleTerrainRepositoryScopeError,
    InMemoryTerrainRepository,
    TerrainAlreadyGeneratedError,
    TerrainGenerationConfig,
    TerrainGenerationError,
    TerrainGenerationService,
    TerrainGenerationServiceError,
    TerrainGenerationServiceSnapshot,
    TerrainGenerator,
    TerrainRepositoryConflictError,
    TerrainRepositoryScope,
    WorldSeed,
    WorldSpecification,
)

CONFIG = TerrainGenerationConfig(octave_count=1)
SPECIFICATION = WorldSpecification(name="Service World", seed=WorldSeed(value=42))


def create_scope(
    *,
    specification: WorldSpecification = SPECIFICATION,
    config: TerrainGenerationConfig = CONFIG,
) -> TerrainRepositoryScope:
    return TerrainRepositoryScope(
        world_seed=specification.seed,
        chunk_size_tiles=specification.chunk_size_tiles,
        generation_format_version=specification.generation_format_version,
        terrain_config=config,
    )


def create_service(
    *,
    generator: TerrainGenerator | None = None,
    repository: InMemoryTerrainRepository | None = None,
    config: TerrainGenerationConfig = CONFIG,
    specification: WorldSpecification = SPECIFICATION,
) -> TerrainGenerationService:
    return TerrainGenerationService(
        specification=specification,
        config=config,
        generator=generator,
        repository=repository,
    )


def test_default_service_constructs_compatible_dependencies() -> None:
    service = create_service()

    assert service.specification is SPECIFICATION
    assert service.config is CONFIG
    assert isinstance(service.generator, DeterministicTerrainGenerator)
    assert service.generator.config is CONFIG
    assert isinstance(service.repository, InMemoryTerrainRepository)
    assert service.repository.scope == create_scope()
    assert service.snapshot() == TerrainGenerationServiceSnapshot(
        repository_revision=0,
        cached_chunk_count=0,
        cache_hits=0,
        cache_misses=0,
        successful_generations=0,
        failed_generations=0,
        evictions=0,
    )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("specification", object(), "specification must be"),
        ("config", object(), "config must be"),
        ("generator", object(), "generator must implement"),
        ("repository", object(), "repository must implement"),
    ],
)
def test_service_constructor_validates_dependencies(
    field_name: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "specification": SPECIFICATION,
        "config": CONFIG,
        field_name: value,
    }

    with pytest.raises(TypeError, match=message):
        TerrainGenerationService(**cast(Any, values))


def test_service_rejects_generator_configuration_mismatch() -> None:
    generator = DeterministicTerrainGenerator(config=TerrainGenerationConfig(octave_count=2))

    with pytest.raises(IncompatibleTerrainRepositoryScopeError, match="Generator configuration"):
        create_service(generator=generator)


@pytest.mark.parametrize("mismatch", ["seed", "dimensions", "format", "config"])
def test_service_rejects_repository_scope_mismatch(mismatch: str) -> None:
    specification = SPECIFICATION
    config = CONFIG
    scope = create_scope()
    if mismatch == "seed":
        scope = TerrainRepositoryScope(
            world_seed=WorldSeed(value=43),
            chunk_size_tiles=CHUNK_SIZE,
            generation_format_version="v1",
            terrain_config=config,
        )
    elif mismatch == "dimensions":
        specification = WorldSpecification(name="Mutated", seed=SPECIFICATION.seed)
        object.__setattr__(specification, "chunk_size_tiles", CHUNK_SIZE + 1)
    elif mismatch == "format":
        specification = WorldSpecification(name="Mutated", seed=SPECIFICATION.seed)
        object.__setattr__(specification, "generation_format_version", "v2")
    else:
        scope = create_scope(config=TerrainGenerationConfig(octave_count=2))

    repository = InMemoryTerrainRepository(scope=scope)
    with pytest.raises(IncompatibleTerrainRepositoryScopeError):
        create_service(
            repository=repository,
            specification=specification,
            config=config,
        )


@pytest.mark.parametrize(
    "method_name",
    ["get", "contains", "get_or_generate", "generate_new", "evict"],
)
def test_service_coordinate_operations_validate_types(method_name: str) -> None:
    service = create_service()

    with pytest.raises(TypeError, match="coordinate must be"):
        getattr(service, method_name)(cast(Any, object()))


def test_get_contains_and_get_or_generate_cache_counter_policy() -> None:
    service = create_service()
    coordinate = ChunkCoordinate(x=-3, y=4)

    assert not service.contains(coordinate)
    terrain = service.get_or_generate(coordinate)
    after_miss = service.snapshot()

    assert service.contains(coordinate)
    assert service.get(coordinate) is terrain
    assert after_miss.cache_misses == 1
    assert after_miss.successful_generations == 1
    assert after_miss.cache_hits == 0
    assert after_miss.repository_revision == 1

    assert service.get_or_generate(coordinate) is terrain
    after_hit = service.snapshot()
    assert after_hit.cache_hits == 1
    assert after_hit.cache_misses == 1
    assert after_hit.successful_generations == 1
    assert after_hit.repository_revision == 1


def test_generate_new_success_and_existing_rejection_counters() -> None:
    service = create_service()
    coordinate = ChunkCoordinate(x=10**100, y=-(10**100))

    terrain = service.generate_new(coordinate)
    snapshot = service.snapshot()

    assert terrain.chunk_coordinate == coordinate
    assert snapshot.successful_generations == 1
    assert snapshot.cache_hits == snapshot.cache_misses == 0

    with pytest.raises(TerrainAlreadyGeneratedError, match="already generated"):
        service.generate_new(coordinate)

    assert service.snapshot() == snapshot


class FailingGenerator:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def generate(
        self,
        *,
        specification: WorldSpecification,
        coordinate: ChunkCoordinate,
    ) -> ChunkTerrain:
        del specification, coordinate
        self.calls += 1
        raise self.error


class WrongCoordinateGenerator:
    def generate(
        self,
        *,
        specification: WorldSpecification,
        coordinate: ChunkCoordinate,
    ) -> ChunkTerrain:
        return DeterministicTerrainGenerator(config=CONFIG).generate(
            specification=specification,
            coordinate=ChunkCoordinate(x=coordinate.x + 1, y=coordinate.y),
        )


class InvalidPayloadGenerator:
    def generate(
        self,
        *,
        specification: WorldSpecification,
        coordinate: ChunkCoordinate,
    ) -> ChunkTerrain:
        del specification, coordinate
        return cast(Any, object())


@pytest.mark.parametrize(
    "error",
    [TerrainGenerationError("domain failure"), ValueError("unexpected failure")],
)
def test_generator_failure_is_chained_counted_and_repository_unchanged(
    error: Exception,
) -> None:
    generator = FailingGenerator(error)
    service = create_service(generator=generator)
    coordinate = ChunkCoordinate(x=1, y=1)

    with pytest.raises(TerrainGenerationServiceError, match="generation failed") as caught:
        service.get_or_generate(coordinate)

    assert caught.value.__cause__ is error
    assert generator.calls == 1
    assert len(service.repository) == 0
    assert service.snapshot() == TerrainGenerationServiceSnapshot(
        repository_revision=0,
        cached_chunk_count=0,
        cache_hits=0,
        cache_misses=1,
        successful_generations=0,
        failed_generations=1,
        evictions=0,
    )


class RejectingRepository(InMemoryTerrainRepository):
    def store(self, terrain: ChunkTerrain) -> NoReturn:
        del terrain
        raise TerrainRepositoryConflictError("repository rejected terrain")


def test_repository_failure_after_generation_is_atomic_and_not_successful() -> None:
    repository = RejectingRepository(scope=create_scope())
    service = create_service(repository=repository)
    coordinate = ChunkCoordinate(x=2, y=-5)

    with pytest.raises(TerrainGenerationServiceError, match="could not be stored") as caught:
        service.get_or_generate(coordinate)

    assert isinstance(caught.value.__cause__, TerrainRepositoryConflictError)
    assert len(repository) == 0
    assert repository.revision == 0
    assert service.snapshot() == TerrainGenerationServiceSnapshot(
        repository_revision=0,
        cached_chunk_count=0,
        cache_hits=0,
        cache_misses=1,
        successful_generations=0,
        failed_generations=0,
        evictions=0,
    )


def test_wrong_generated_coordinate_is_rejected_before_publication() -> None:
    service = create_service(generator=WrongCoordinateGenerator())
    coordinate = ChunkCoordinate(x=-6, y=8)

    with pytest.raises(TerrainGenerationServiceError) as caught:
        service.generate_new(coordinate)

    assert isinstance(caught.value.__cause__, IncompatibleTerrainRepositoryScopeError)
    assert len(service.repository) == 0
    assert service.snapshot().failed_generations == 1


def test_non_terrain_generator_result_is_rejected_before_publication() -> None:
    service = create_service(generator=InvalidPayloadGenerator())

    with pytest.raises(TerrainGenerationServiceError):
        service.generate_new(ChunkCoordinate(x=0, y=0))

    assert len(service.repository) == 0
    assert service.snapshot().failed_generations == 1


def test_eviction_and_clear_have_exact_counter_and_revision_semantics() -> None:
    service = create_service()
    first = ChunkCoordinate(x=0, y=0)
    second = ChunkCoordinate(x=1, y=0)
    service.generate_new(first)
    service.generate_new(second)

    service.evict(ChunkCoordinate(x=99, y=99))
    assert service.snapshot().evictions == 0

    service.evict(first)
    after_evict = service.snapshot()
    assert after_evict.evictions == 1
    assert after_evict.cached_chunk_count == 1
    assert after_evict.repository_revision == 3

    service.clear()
    after_clear = service.snapshot()
    assert after_clear.cached_chunk_count == 0
    assert after_clear.repository_revision == 4
    assert after_clear.evictions == 1
    service.clear()
    assert service.snapshot() == after_clear


def test_equivalent_services_produce_equivalent_terrain() -> None:
    coordinate = ChunkCoordinate(x=-8, y=9)
    first = create_service()
    second = create_service()

    assert first.get_or_generate(coordinate) == second.get_or_generate(coordinate)
    assert first.snapshot() == second.snapshot()


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("repository_revision", True, TypeError),
        ("cached_chunk_count", "0", TypeError),
        ("cache_hits", -1, ValueError),
        ("cache_misses", -1, ValueError),
        ("successful_generations", -1, ValueError),
        ("failed_generations", -1, ValueError),
        ("evictions", -1, ValueError),
    ],
)
def test_service_snapshot_validation(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, object] = {
        "repository_revision": 0,
        "cached_chunk_count": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "successful_generations": 0,
        "failed_generations": 0,
        "evictions": 0,
        field_name: value,
    }

    with pytest.raises(error_type):
        TerrainGenerationServiceSnapshot(**cast(Any, values))


def test_service_snapshot_is_immutable_and_exports_are_complete() -> None:
    snapshot = create_service().snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.cache_hits = 1  # type: ignore[misc]
    assert issubclass(TerrainAlreadyGeneratedError, TerrainGenerationServiceError)
    names = {
        "TerrainAlreadyGeneratedError",
        "TerrainGenerationService",
        "TerrainGenerationServiceError",
        "TerrainGenerationServiceSnapshot",
    }
    assert names <= set(world.__all__)
    assert all(hasattr(world, name) for name in names)
