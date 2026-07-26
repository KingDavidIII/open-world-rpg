"""Tests for renderer-independent terrain viewport and HUD calculations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from open_world_rpg.application import create_terrain_runtime
from open_world_rpg.ui import (
    TERRAIN_PALETTE,
    CameraState,
    TerrainHudSnapshot,
    TerrainViewport,
    terrain_surface_cache_key,
)
from open_world_rpg.world import (
    CHUNK_SIZE,
    ChunkCoordinate,
    TerrainGenerationConfig,
    TerrainType,
    WorldPosition,
    WorldSeed,
    WorldSpecification,
)


@pytest.mark.parametrize(
    ("field_name", "value", "error_type", "message"),
    [
        ("x_tiles", True, TypeError, "must be a number"),
        ("y_tiles", "0", TypeError, "must be a number"),
        ("x_tiles", float("inf"), ValueError, "must be finite"),
        ("y_tiles", float("nan"), ValueError, "must be finite"),
        ("movement_speed_tiles_per_second", 0.0, ValueError, "greater than zero"),
        ("fast_multiplier", -1.0, ValueError, "greater than zero"),
    ],
)
def test_camera_validation(
    field_name: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        CameraState(**cast(Any, {field_name: value}))


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("width_pixels", True, TypeError),
        ("width_pixels", 0, ValueError),
        ("height_pixels", 1.5, TypeError),
        ("height_pixels", -1, ValueError),
        ("tile_size_pixels", "16", TypeError),
        ("tile_size_pixels", 0, ValueError),
    ],
)
def test_viewport_validation(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, object] = {
        "width_pixels": 320,
        "height_pixels": 180,
        "tile_size_pixels": 10,
        field_name: value,
    }
    with pytest.raises(error_type):
        TerrainViewport(**cast(Any, values))


def test_camera_movement_is_frame_rate_independent_and_diagonal_normalised() -> None:
    camera = CameraState(movement_speed_tiles_per_second=20.0, fast_multiplier=4.0)

    one_frame = camera.moved(horizontal=1, vertical=0, delta_seconds=1.0)
    two_frames = camera.moved(
        horizontal=1,
        vertical=0,
        delta_seconds=0.5,
    ).moved(horizontal=1, vertical=0, delta_seconds=0.5)
    diagonal = camera.moved(horizontal=1, vertical=1, delta_seconds=1.0)
    fast = camera.moved(horizontal=-1, vertical=0, delta_seconds=1.0, fast=True)

    assert one_frame.x_tiles == two_frames.x_tiles == 20.0
    assert one_frame.y_tiles == two_frames.y_tiles == 0.0
    assert diagonal.x_tiles**2 + diagonal.y_tiles**2 == pytest.approx(20.0**2)
    assert fast.x_tiles == -80.0
    assert camera.moved(horizontal=0, vertical=0, delta_seconds=999.0) is camera


@pytest.mark.parametrize(
    ("arguments", "error_type", "message"),
    [
        ({"horizontal": True, "vertical": 0, "delta_seconds": 1.0}, TypeError, "horizontal"),
        ({"horizontal": 2, "vertical": 0, "delta_seconds": 1.0}, ValueError, "horizontal"),
        ({"horizontal": 0, "vertical": "1", "delta_seconds": 1.0}, TypeError, "vertical"),
        ({"horizontal": 0, "vertical": -2, "delta_seconds": 1.0}, ValueError, "vertical"),
        ({"horizontal": 1, "vertical": 0, "delta_seconds": True}, TypeError, "delta_seconds"),
        ({"horizontal": 1, "vertical": 0, "delta_seconds": -1.0}, ValueError, "delta_seconds"),
        (
            {"horizontal": 1, "vertical": 0, "delta_seconds": float("inf")},
            ValueError,
            "delta_seconds",
        ),
        (
            {"horizontal": 1, "vertical": 0, "delta_seconds": 1.0, "fast": 1},
            TypeError,
            "fast",
        ),
    ],
)
def test_camera_movement_validation(
    arguments: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        CameraState().moved(**cast(Any, arguments))


def test_negative_camera_tile_uses_floor_and_crosses_chunk_region_boundaries() -> None:
    camera = CameraState(x_tiles=-0.01, y_tiles=-(CHUNK_SIZE * 16) - 0.01)

    assert camera.world_tile == WorldPosition(x=-1, y=-257)
    assert camera.world_tile.to_chunk() == ChunkCoordinate(x=-1, y=-17)
    assert camera.world_tile.to_chunk().to_region().x == -1
    assert camera.world_tile.to_chunk().to_region().y == -2


def test_visible_chunks_are_deterministic_row_major_with_preload() -> None:
    viewport = TerrainViewport(
        width_pixels=CHUNK_SIZE * 10,
        height_pixels=CHUNK_SIZE * 10,
        tile_size_pixels=10,
    )
    camera = CameraState(x_tiles=8.0, y_tiles=8.0)

    assert viewport.visible_chunks(camera=camera) == (ChunkCoordinate(x=0, y=0),)
    assert viewport.visible_chunks(camera=camera, preload_margin_chunks=1) == tuple(
        ChunkCoordinate(x=x, y=y) for y in (-1, 0, 1) for x in (-1, 0, 1)
    )


def test_visible_chunks_support_negative_and_large_camera_coordinates() -> None:
    viewport = TerrainViewport(width_pixels=16, height_pixels=16, tile_size_pixels=16)

    negative = viewport.visible_chunks(camera=CameraState(x_tiles=-0.5, y_tiles=-0.5))
    huge = viewport.visible_chunks(
        camera=CameraState(x_tiles=float(10**12), y_tiles=float(-(10**12)))
    )

    assert negative == (ChunkCoordinate(x=-1, y=-1),)
    assert huge == tuple(
        ChunkCoordinate(x=x, y=y)
        for y in (-62_500_000_001, -62_500_000_000)
        for x in (62_499_999_999, 62_500_000_000)
    )


@pytest.mark.parametrize(
    ("camera", "margin", "error_type", "message"),
    [
        (object(), 0, TypeError, "camera"),
        (CameraState(), True, TypeError, "preload_margin_chunks"),
        (CameraState(), -1, ValueError, "non-negative"),
    ],
)
def test_visible_chunk_validation(
    camera: object,
    margin: object,
    error_type: type[Exception],
    message: str,
) -> None:
    viewport = TerrainViewport(width_pixels=100, height_pixels=100, tile_size_pixels=10)
    with pytest.raises(error_type, match=message):
        viewport.visible_chunks(
            camera=cast(Any, camera),
            preload_margin_chunks=cast(Any, margin),
        )


def test_screen_world_conversion_round_trip_including_negative_tiles() -> None:
    viewport = TerrainViewport(width_pixels=320, height_pixels=180, tile_size_pixels=10)
    camera = CameraState(x_tiles=-17.25, y_tiles=31.75)
    position = WorldPosition(x=-20, y=29)

    screen = viewport.world_to_screen(
        camera=camera,
        world_x=position.x,
        world_y=position.y,
    )

    assert (
        viewport.screen_to_world(
            camera=camera,
            screen_x=screen[0] + 1,
            screen_y=screen[1] + 1,
        )
        == position
    )


@pytest.mark.parametrize(
    ("method", "arguments", "message"),
    [
        ("world_to_screen", {"camera": object(), "world_x": 0, "world_y": 0}, "camera"),
        (
            "screen_to_world",
            {"camera": object(), "screen_x": 0, "screen_y": 0},
            "camera",
        ),
        (
            "screen_to_world",
            {"camera": CameraState(), "screen_x": True, "screen_y": 0},
            "screen_x",
        ),
        (
            "screen_to_world",
            {"camera": CameraState(), "screen_x": 0, "screen_y": 1.5},
            "screen_y",
        ),
    ],
)
def test_coordinate_conversion_validation(
    method: str,
    arguments: dict[str, object],
    message: str,
) -> None:
    viewport = TerrainViewport(width_pixels=320, height_pixels=180, tile_size_pixels=10)
    with pytest.raises(TypeError, match=message):
        getattr(viewport, method)(**cast(Any, arguments))


def test_palette_is_complete_and_distinct_in_ui_layer() -> None:
    assert tuple(TERRAIN_PALETTE) == tuple(TerrainType)
    assert len(set(TERRAIN_PALETTE.values())) == len(TerrainType)
    assert all(
        len(colour) == 3 and all(0 <= component <= 255 for component in colour)
        for colour in TERRAIN_PALETTE.values()
    )


def create_runtime_and_terrain() -> tuple[Any, Any]:
    specification = WorldSpecification(name="HUD", seed=WorldSeed(value=7))
    config = TerrainGenerationConfig(octave_count=1)
    runtime = create_terrain_runtime(world=specification, config=config)
    coordinate = ChunkCoordinate(x=-1, y=2)
    terrain = runtime.get_or_generate(coordinate)
    return runtime, terrain


def test_hud_snapshot_calculates_runtime_and_spatial_values() -> None:
    runtime, _ = create_runtime_and_terrain()
    camera = CameraState(x_tiles=-0.1, y_tiles=32.5)

    hud = TerrainHudSnapshot.from_runtime(
        camera=camera,
        runtime=runtime,
        visible_chunk_count=3,
    )

    assert hud.camera_tile == WorldPosition(x=-1, y=32)
    assert hud.chunk_coordinate == ChunkCoordinate(x=-1, y=2)
    assert hud.region_coordinate == ChunkCoordinate(x=-1, y=2).to_region()
    assert hud.world_seed == 7
    assert hud.visible_chunk_count == 3
    assert hud.cached_chunk_count == 1
    assert hud.terrain_runtime_revision == 3
    assert hud.repository_revision == 1
    assert hud.successful_generations == 1


@pytest.mark.parametrize(
    ("camera", "runtime", "count", "error_type"),
    [
        (object(), None, 0, TypeError),
        (CameraState(), object(), 0, TypeError),
        (CameraState(), None, True, TypeError),
        (CameraState(), None, -1, ValueError),
    ],
)
def test_hud_snapshot_validation(
    camera: object,
    runtime: object,
    count: object,
    error_type: type[Exception],
) -> None:
    valid_runtime, _ = create_runtime_and_terrain()
    resolved_runtime = valid_runtime if runtime is None else runtime
    with pytest.raises(error_type):
        TerrainHudSnapshot.from_runtime(
            camera=cast(Any, camera),
            runtime=cast(Any, resolved_runtime),
            visible_chunk_count=cast(Any, count),
        )


def test_surface_cache_key_changes_only_for_render_relevant_payload_fields() -> None:
    _, terrain = create_runtime_and_terrain()
    key = terrain_surface_cache_key(terrain, tile_size_pixels=16)

    assert key == terrain_surface_cache_key(terrain, tile_size_pixels=16)
    assert key != terrain_surface_cache_key(terrain, tile_size_pixels=17)
    assert key != terrain_surface_cache_key(replace(terrain, revision=1), tile_size_pixels=16)
    with pytest.raises(TypeError, match="terrain must be"):
        terrain_surface_cache_key(cast(Any, object()), tile_size_pixels=16)
    with pytest.raises(TypeError, match="tile_size_pixels must be"):
        terrain_surface_cache_key(terrain, tile_size_pixels=cast(Any, True))
    with pytest.raises(ValueError, match="greater than zero"):
        terrain_surface_cache_key(terrain, tile_size_pixels=0)


def test_camera_and_viewport_are_immutable() -> None:
    camera = CameraState()
    viewport = TerrainViewport(width_pixels=100, height_pixels=100, tile_size_pixels=10)

    with pytest.raises(FrozenInstanceError):
        camera.x_tiles = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        viewport.width_pixels = 200  # type: ignore[misc]
