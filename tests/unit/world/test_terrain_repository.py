"""Tests for controlled in-memory terrain repository contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

import open_world_rpg.world as world
from open_world_rpg.world import (
    CHUNK_SIZE,
    ChunkCoordinate,
    ChunkTerrain,
    DeterministicTerrainGenerator,
    IncompatibleTerrainRepositoryScopeError,
    InMemoryTerrainRepository,
    TerrainGenerationConfig,
    TerrainMissingError,
    TerrainRepository,
    TerrainRepositoryAccessError,
    TerrainRepositoryConflictError,
    TerrainRepositoryError,
    TerrainRepositoryScope,
    TerrainRepositorySnapshot,
    WorldSeed,
    WorldSpecification,
)

CONFIG = TerrainGenerationConfig(octave_count=1)
WORLD_SEED = WorldSeed(value=42)


def create_scope(
    *,
    world_seed: WorldSeed = WORLD_SEED,
    config: TerrainGenerationConfig = CONFIG,
) -> TerrainRepositoryScope:
    return TerrainRepositoryScope(
        world_seed=world_seed,
        chunk_size_tiles=CHUNK_SIZE,
        generation_format_version="v1",
        terrain_config=config,
    )


def create_terrain(
    coordinate: ChunkCoordinate,
    *,
    seed: int = 42,
    config: TerrainGenerationConfig = CONFIG,
) -> ChunkTerrain:
    return DeterministicTerrainGenerator(config=config).generate(
        specification=WorldSpecification(
            name="Repository World",
            seed=WorldSeed(value=seed),
        ),
        coordinate=coordinate,
    )


@pytest.mark.parametrize(
    ("field_name", "value", "error_type", "message"),
    [
        ("world_seed", 42, TypeError, "world_seed must be"),
        ("chunk_size_tiles", True, TypeError, "chunk_size_tiles must be an integer"),
        ("chunk_size_tiles", 16.0, TypeError, "chunk_size_tiles must be an integer"),
        (
            "chunk_size_tiles",
            CHUNK_SIZE + 1,
            IncompatibleTerrainRepositoryScopeError,
            "supported value",
        ),
        ("generation_format_version", b"v1", TypeError, "must be a string"),
        ("terrain_config", object(), TypeError, "terrain_config must be"),
        (
            "generation_format_version",
            "v2",
            IncompatibleTerrainRepositoryScopeError,
            "formats must agree",
        ),
    ],
)
def test_repository_scope_validation(
    field_name: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {
        "world_seed": WORLD_SEED,
        "chunk_size_tiles": CHUNK_SIZE,
        "generation_format_version": "v1",
        "terrain_config": CONFIG,
        field_name: value,
    }

    with pytest.raises(error_type, match=message):
        TerrainRepositoryScope(**cast(Any, values))


def test_repository_scope_rejects_mismatched_config_format() -> None:
    config = TerrainGenerationConfig()
    object.__setattr__(config, "generation_format_version", "v2")

    with pytest.raises(IncompatibleTerrainRepositoryScopeError, match="formats must agree"):
        create_scope(config=config)


def test_scope_is_immutable_and_repository_conforms_to_protocol() -> None:
    scope = create_scope()
    repository = InMemoryTerrainRepository(scope=scope)

    assert isinstance(repository, TerrainRepository)
    assert repository.scope is scope
    assert repository.revision == 0
    assert len(repository) == 0
    with pytest.raises(FrozenInstanceError):
        scope.chunk_size_tiles = 32  # type: ignore[misc]
    with pytest.raises(TypeError, match="scope must be"):
        InMemoryTerrainRepository(scope=cast(Any, object()))


@pytest.mark.parametrize("method_name", ["get", "contains", "remove"])
def test_repository_coordinate_operations_validate_types(method_name: str) -> None:
    repository = InMemoryTerrainRepository(scope=create_scope())

    with pytest.raises(TypeError, match="coordinate must be"):
        getattr(repository, method_name)(cast(Any, object()))


def test_store_validates_payload_type() -> None:
    repository = InMemoryTerrainRepository(scope=create_scope())

    with pytest.raises(TypeError, match="terrain must be"):
        repository.store(cast(Any, object()))


def test_storage_retrieval_and_reads_do_not_mutate_revision() -> None:
    repository = InMemoryTerrainRepository(scope=create_scope())
    coordinate = ChunkCoordinate(x=-2, y=3)
    terrain = create_terrain(coordinate)

    repository.store(terrain)
    revision = repository.revision

    assert repository.contains(coordinate)
    assert repository.get(coordinate) is terrain
    assert len(repository) == 1
    assert repository.coordinates() == (coordinate,)
    assert repository.snapshot().revision == revision
    assert repository.revision == revision == 1


def test_missing_retrieval_raises_explicit_error_with_key_error_cause() -> None:
    repository = InMemoryTerrainRepository(scope=create_scope())

    with pytest.raises(TerrainMissingError, match=r"chunk \(-1, 7\)") as caught:
        repository.get(ChunkCoordinate(x=-1, y=7))

    assert isinstance(caught.value.__cause__, KeyError)
    assert repository.revision == 0


def test_idempotent_store_is_no_op_and_conflicting_store_is_atomic() -> None:
    repository = InMemoryTerrainRepository(scope=create_scope())
    coordinate = ChunkCoordinate(x=1, y=2)
    terrain = create_terrain(coordinate)
    repository.store(terrain)

    repository.store(terrain)
    assert repository.revision == 1

    conflicting = replace(terrain, revision=1)
    with pytest.raises(TerrainRepositoryConflictError, match="Different terrain"):
        repository.store(conflicting)

    assert repository.get(coordinate) is terrain
    assert repository.revision == 1


def test_remove_and_clear_revision_policy() -> None:
    repository = InMemoryTerrainRepository(scope=create_scope())
    first = create_terrain(ChunkCoordinate(x=0, y=0))
    second = create_terrain(ChunkCoordinate(x=1, y=0))

    repository.remove(first.chunk_coordinate)
    repository.clear()
    assert repository.revision == 0

    repository.store(first)
    repository.store(second)
    assert repository.revision == 2

    repository.remove(first.chunk_coordinate)
    assert repository.revision == 3
    repository.remove(first.chunk_coordinate)
    assert repository.revision == 3

    repository.clear()
    assert repository.revision == 4
    repository.clear()
    assert repository.revision == 4
    assert len(repository) == 0


def test_coordinates_are_deterministic_increasing_y_then_x() -> None:
    repository = InMemoryTerrainRepository(scope=create_scope())
    coordinates = (
        ChunkCoordinate(x=10**100, y=0),
        ChunkCoordinate(x=-5, y=2),
        ChunkCoordinate(x=4, y=-3),
        ChunkCoordinate(x=-(10**100), y=0),
        ChunkCoordinate(x=3, y=-3),
    )
    for coordinate in coordinates:
        repository.store(create_terrain(coordinate))

    assert repository.coordinates() == (
        ChunkCoordinate(x=3, y=-3),
        ChunkCoordinate(x=4, y=-3),
        ChunkCoordinate(x=-(10**100), y=0),
        ChunkCoordinate(x=10**100, y=0),
        ChunkCoordinate(x=-5, y=2),
    )


def test_snapshot_is_immutable_and_excludes_payloads() -> None:
    repository = InMemoryTerrainRepository(scope=create_scope())
    coordinate = ChunkCoordinate(x=-1, y=-1)
    repository.store(create_terrain(coordinate))

    snapshot = repository.snapshot()

    assert snapshot == TerrainRepositorySnapshot(
        scope=repository.scope,
        revision=1,
        chunk_count=1,
        coordinates=(coordinate,),
    )
    assert not hasattr(snapshot, "terrain")
    assert not hasattr(snapshot, "tiles")
    with pytest.raises(FrozenInstanceError):
        snapshot.revision = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type", "message"),
    [
        ("scope", object(), TypeError, "scope must be"),
        ("revision", True, TypeError, "revision must be an integer"),
        ("revision", -1, ValueError, "revision must be greater"),
        ("chunk_count", "1", TypeError, "chunk_count must be an integer"),
        ("chunk_count", -1, ValueError, "chunk_count must be greater"),
        ("coordinates", [], TypeError, "coordinates must be a tuple"),
        ("coordinates", (object(),), TypeError, "coordinates must be a tuple"),
        ("chunk_count", 2, ValueError, "coordinate count must match"),
    ],
)
def test_snapshot_validation(
    field_name: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {
        "scope": create_scope(),
        "revision": 0,
        "chunk_count": 0,
        "coordinates": (),
        field_name: value,
    }

    with pytest.raises(error_type, match=message):
        TerrainRepositorySnapshot(**cast(Any, values))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("world_seed", "world seed"),
        ("dimensions", "dimensions"),
        ("format", "generation format"),
        ("terrain_seed", "TERRAIN-stage"),
    ],
)
def test_store_rejects_scope_mismatch_atomically(
    mutation: str,
    message: str,
) -> None:
    repository = InMemoryTerrainRepository(scope=create_scope())
    terrain = create_terrain(ChunkCoordinate(x=5, y=-7))

    if mutation == "world_seed":
        terrain = create_terrain(terrain.chunk_coordinate, seed=43)
    elif mutation == "dimensions":
        object.__setattr__(terrain, "width", CHUNK_SIZE + 1)
    elif mutation == "format":
        object.__setattr__(terrain, "generation_format_version", "v2")
    else:
        object.__setattr__(terrain, "terrain_seed", terrain.terrain_seed + 1)

    with pytest.raises(IncompatibleTerrainRepositoryScopeError, match=message):
        repository.store(terrain)

    assert len(repository) == 0
    assert repository.revision == 0


def test_repository_errors_and_exports_are_public() -> None:
    assert issubclass(TerrainRepositoryAccessError, TerrainRepositoryError)
    assert issubclass(TerrainMissingError, TerrainRepositoryAccessError)
    assert issubclass(IncompatibleTerrainRepositoryScopeError, TerrainRepositoryError)
    assert issubclass(TerrainRepositoryConflictError, TerrainRepositoryError)
    names = {
        "InMemoryTerrainRepository",
        "IncompatibleTerrainRepositoryScopeError",
        "TerrainMissingError",
        "TerrainRepository",
        "TerrainRepositoryAccessError",
        "TerrainRepositoryConflictError",
        "TerrainRepositoryError",
        "TerrainRepositoryScope",
        "TerrainRepositorySnapshot",
    }
    assert names <= set(world.__all__)
    assert all(hasattr(world, name) for name in names)
