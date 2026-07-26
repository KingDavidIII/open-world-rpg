"""Tests for immutable terrain contracts and generation interface."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from open_world_rpg.world import (
    CHUNK_SIZE,
    MAX_DERIVED_SEED,
    MAX_TERRAIN_ELEVATION,
    MIN_TERRAIN_ELEVATION,
    ChunkCoordinate,
    ChunkGenerationKey,
    ChunkTerrain,
    ChunkTerrainSnapshot,
    DuplicateTerrainCoordinateError,
    IncompatibleTerrainDimensionsError,
    IncompleteTerrainCoverageError,
    InvalidTerrainPayloadError,
    LocalTileCoordinate,
    TerrainElevation,
    TerrainGenerationError,
    TerrainGenerator,
    TerrainGeneratorExecutionError,
    TerrainTile,
    TerrainType,
    WorldGenerationStage,
    WorldSeed,
    WorldSpecification,
)

WORLD_SEED = WorldSeed(value=42)
CHUNK_COORDINATE = ChunkCoordinate(x=-17, y=16)


def terrain_seed(
    *,
    world_seed: WorldSeed = WORLD_SEED,
    coordinate: ChunkCoordinate = CHUNK_COORDINATE,
) -> int:
    return ChunkGenerationKey(
        world_seed=world_seed,
        coordinate=coordinate,
        stage=WorldGenerationStage.TERRAIN,
    ).derived_seed


def create_tiles() -> tuple[TerrainTile, ...]:
    terrain_types = tuple(TerrainType)
    return tuple(
        TerrainTile(
            coordinate=LocalTileCoordinate(x=x, y=y),
            elevation=TerrainElevation(metres=(y * CHUNK_SIZE) + x - 128),
            terrain_type=terrain_types[((y * CHUNK_SIZE) + x) % len(terrain_types)],
            revision=(y * CHUNK_SIZE) + x,
        )
        for y in range(CHUNK_SIZE)
        for x in range(CHUNK_SIZE)
    )


def create_chunk_terrain(
    *,
    tiles: tuple[TerrainTile, ...] | None = None,
    coordinate: ChunkCoordinate = CHUNK_COORDINATE,
    world_seed: WorldSeed = WORLD_SEED,
) -> ChunkTerrain:
    return ChunkTerrain(
        world_seed=world_seed,
        chunk_coordinate=coordinate,
        terrain_seed=terrain_seed(
            world_seed=world_seed,
            coordinate=coordinate,
        ),
        width=CHUNK_SIZE,
        height=CHUNK_SIZE,
        tiles=create_tiles() if tiles is None else tiles,
    )


def test_elevation_boundaries_ordering_and_units_policy() -> None:
    minimum = TerrainElevation(metres=MIN_TERRAIN_ELEVATION)
    sea_level = TerrainElevation(metres=0)
    maximum = TerrainElevation(metres=MAX_TERRAIN_ELEVATION)

    assert MIN_TERRAIN_ELEVATION == -32_768
    assert MAX_TERRAIN_ELEVATION == 32_767
    assert minimum < sea_level < maximum
    assert sorted((maximum, minimum, sea_level)) == [minimum, sea_level, maximum]


@pytest.mark.parametrize("value", [True, False, 1.5, "1", None])
def test_elevation_rejects_booleans_and_non_integers(value: object) -> None:
    with pytest.raises(TypeError, match="metres must be an integer"):
        TerrainElevation(metres=cast(Any, value))


@pytest.mark.parametrize(
    "value",
    [MIN_TERRAIN_ELEVATION - 1, MAX_TERRAIN_ELEVATION + 1],
)
def test_elevation_rejects_values_outside_supported_range(value: int) -> None:
    with pytest.raises(ValueError, match="metres must be between"):
        TerrainElevation(metres=value)


def test_terrain_type_values_are_foundational_and_explicit() -> None:
    assert [terrain_type.value for terrain_type in TerrainType] == [
        "deep_water",
        "shallow_water",
        "coast",
        "plains",
        "hills",
        "mountains",
    ]


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("coordinate", object(), "coordinate must be a LocalTileCoordinate"),
        ("elevation", 0, "elevation must be a TerrainElevation"),
        ("terrain_type", "plains", "terrain_type must be a TerrainType"),
    ],
)
def test_terrain_tile_rejects_invalid_field_types(
    field_name: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, Any] = {
        "coordinate": LocalTileCoordinate(x=0, y=0),
        "elevation": TerrainElevation(metres=0),
        "terrain_type": TerrainType.PLAINS,
    }
    values[field_name] = value

    with pytest.raises(TypeError, match=message):
        TerrainTile(**values)


@pytest.mark.parametrize(
    ("revision", "error_type", "message"),
    [
        (True, TypeError, "revision must be an integer"),
        ("1", TypeError, "revision must be an integer"),
        (-1, ValueError, "revision must be greater than or equal to zero"),
    ],
)
def test_terrain_tile_revision_validation(
    revision: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        TerrainTile(
            coordinate=LocalTileCoordinate(x=0, y=0),
            elevation=TerrainElevation(metres=0),
            terrain_type=TerrainType.PLAINS,
            revision=cast(Any, revision),
        )


def test_terrain_tile_is_immutable_and_has_no_rendering_data() -> None:
    tile = TerrainTile(
        coordinate=LocalTileCoordinate(x=0, y=0),
        elevation=TerrainElevation(metres=0),
        terrain_type=TerrainType.COAST,
    )

    assert not hasattr(tile, "sprite")
    assert not hasattr(tile, "entity")

    with pytest.raises(FrozenInstanceError):
        tile.revision = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("world_seed", 42, "world_seed must be a WorldSeed"),
        ("chunk_coordinate", object(), "chunk_coordinate must be a ChunkCoordinate"),
    ],
)
def test_chunk_terrain_rejects_invalid_identity_types(
    field_name: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, Any] = {
        "world_seed": WORLD_SEED,
        "chunk_coordinate": CHUNK_COORDINATE,
        "terrain_seed": terrain_seed(),
        "width": CHUNK_SIZE,
        "height": CHUNK_SIZE,
        "tiles": create_tiles(),
    }
    values[field_name] = value

    with pytest.raises(TypeError, match=message):
        ChunkTerrain(**values)


@pytest.mark.parametrize("field_name", ["width", "height"])
@pytest.mark.parametrize("value", [True, False, "16", 16.0, 15, 17])
def test_chunk_terrain_rejects_incompatible_dimensions(
    field_name: str,
    value: object,
) -> None:
    terrain = create_chunk_terrain()

    with pytest.raises(IncompatibleTerrainDimensionsError, match=field_name):
        replace(terrain, **{field_name: value})


@pytest.mark.parametrize("value", [True, False, "1", 1.5])
def test_chunk_terrain_seed_rejects_non_integers(value: object) -> None:
    terrain = create_chunk_terrain()

    with pytest.raises(TypeError, match="terrain_seed must be an integer"):
        replace(terrain, terrain_seed=value)


@pytest.mark.parametrize("value", [-1, MAX_DERIVED_SEED + 1])
def test_chunk_terrain_seed_rejects_out_of_range_values(value: int) -> None:
    terrain = create_chunk_terrain()

    with pytest.raises(ValueError, match="terrain_seed must be between"):
        replace(terrain, terrain_seed=value)


def test_chunk_terrain_seed_must_match_deterministic_key() -> None:
    terrain = create_chunk_terrain()
    mismatched = (
        terrain.terrain_seed + 1
        if terrain.terrain_seed < MAX_DERIVED_SEED
        else terrain.terrain_seed - 1
    )

    with pytest.raises(InvalidTerrainPayloadError, match="TERRAIN generation key"):
        replace(terrain, terrain_seed=mismatched)


@pytest.mark.parametrize(
    ("revision", "error_type", "message"),
    [
        (True, TypeError, "revision must be an integer"),
        ("1", TypeError, "revision must be an integer"),
        (-1, ValueError, "revision must be greater than or equal to zero"),
    ],
)
def test_chunk_terrain_revision_validation(
    revision: object,
    error_type: type[Exception],
    message: str,
) -> None:
    terrain = create_chunk_terrain()

    with pytest.raises(error_type, match=message):
        replace(terrain, revision=revision)


@pytest.mark.parametrize("value", [b"v1", "v0", "v2"])
def test_chunk_terrain_rejects_incompatible_generation_format(value: object) -> None:
    terrain = create_chunk_terrain()

    with pytest.raises(InvalidTerrainPayloadError, match="generation_format_version"):
        replace(terrain, generation_format_version=value)


def test_chunk_terrain_requires_immutable_typed_tile_tuple() -> None:
    terrain = create_chunk_terrain()

    with pytest.raises(TypeError, match="tiles must be a tuple"):
        replace(terrain, tiles=cast(Any, list(terrain.tiles)))

    invalid_tiles = cast(tuple[TerrainTile, ...], (object(), *terrain.tiles[1:]))
    with pytest.raises(TypeError, match="only TerrainTile"):
        replace(terrain, tiles=invalid_tiles)


def test_chunk_terrain_rejects_missing_and_duplicate_coordinates() -> None:
    tiles = create_tiles()

    with pytest.raises(IncompleteTerrainCoverageError, match="exactly 256"):
        create_chunk_terrain(tiles=tiles[:-1])

    duplicate_tiles = (*tiles[:-1], tiles[0])
    with pytest.raises(DuplicateTerrainCoordinateError, match="Duplicate"):
        create_chunk_terrain(tiles=duplicate_tiles)


def test_local_coordinate_model_rejects_out_of_range_terrain_coordinates() -> None:
    with pytest.raises(ValueError, match="x must be between"):
        LocalTileCoordinate(x=CHUNK_SIZE, y=0)


def test_chunk_terrain_sorts_and_iterates_exact_row_major_coverage() -> None:
    reversed_tiles = tuple(reversed(create_tiles()))

    terrain = create_chunk_terrain(tiles=reversed_tiles)
    coordinates = tuple(tile.coordinate for tile in terrain)

    assert len(terrain) == CHUNK_SIZE**2 == 256
    assert len(set(coordinates)) == 256
    assert coordinates[:17] == (
        *(LocalTileCoordinate(x=x, y=0) for x in range(CHUNK_SIZE)),
        LocalTileCoordinate(x=0, y=1),
    )
    assert coordinates[-1] == LocalTileCoordinate(x=15, y=15)


def test_negative_chunk_identity_and_tile_lookup() -> None:
    terrain = create_chunk_terrain()
    coordinate = LocalTileCoordinate(x=7, y=11)

    tile = terrain.tile_at(coordinate)

    assert terrain.chunk_coordinate == ChunkCoordinate(x=-17, y=16)
    assert tile.coordinate == coordinate
    assert tile is terrain.tiles[(11 * CHUNK_SIZE) + 7]

    with pytest.raises(TypeError, match="coordinate must be a LocalTileCoordinate"):
        terrain.tile_at(cast(Any, object()))


def test_chunk_terrain_elevation_extrema_and_type_counts_are_immutable() -> None:
    terrain = create_chunk_terrain()

    assert terrain.minimum_elevation == TerrainElevation(metres=-128)
    assert terrain.maximum_elevation == TerrainElevation(metres=127)
    assert tuple(terrain.terrain_type_counts) == tuple(TerrainType)
    assert sum(terrain.terrain_type_counts.values()) == 256
    assert terrain.terrain_type_counts[TerrainType.DEEP_WATER] == 43
    assert terrain.terrain_type_counts[TerrainType.MOUNTAINS] == 42

    with pytest.raises(TypeError):
        terrain.terrain_type_counts[TerrainType.PLAINS] = 0  # type: ignore[index]


def test_chunk_terrain_snapshot_is_complete_and_immutable() -> None:
    terrain = replace(create_chunk_terrain(), revision=9)

    snapshot = terrain.snapshot()

    assert snapshot == ChunkTerrainSnapshot(
        world_seed=terrain.world_seed,
        chunk_coordinate=terrain.chunk_coordinate,
        terrain_seed=terrain.terrain_seed,
        width=CHUNK_SIZE,
        height=CHUNK_SIZE,
        tiles=terrain.tiles,
        revision=9,
        generation_format_version="v1",
    )
    assert snapshot.tiles is terrain.tiles

    with pytest.raises(FrozenInstanceError):
        terrain.revision = 10  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        snapshot.revision = 10  # type: ignore[misc]


class ExampleTerrainGenerator:
    def generate(
        self,
        *,
        specification: WorldSpecification,
        coordinate: ChunkCoordinate,
    ) -> ChunkTerrain:
        return create_chunk_terrain(
            world_seed=specification.seed,
            coordinate=coordinate,
        )


class NotATerrainGenerator:
    pass


def test_runtime_checkable_generator_protocol_and_error_hierarchy() -> None:
    generator = ExampleTerrainGenerator()
    specification = WorldSpecification(
        name="Terrain World",
        seed=WORLD_SEED,
    )

    assert isinstance(generator, TerrainGenerator)
    assert not isinstance(NotATerrainGenerator(), TerrainGenerator)
    assert (
        generator.generate(
            specification=specification,
            coordinate=CHUNK_COORDINATE,
        )
        == create_chunk_terrain()
    )
    assert issubclass(InvalidTerrainPayloadError, TerrainGenerationError)
    assert issubclass(IncompleteTerrainCoverageError, InvalidTerrainPayloadError)
    assert issubclass(DuplicateTerrainCoordinateError, InvalidTerrainPayloadError)
    assert issubclass(IncompatibleTerrainDimensionsError, InvalidTerrainPayloadError)
    assert issubclass(TerrainGeneratorExecutionError, TerrainGenerationError)
