"""Tests for deterministic natural-world presentation calculations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from open_world_rpg.ui import (
    TERRAIN_GRADIENTS,
    TERRAIN_PALETTE,
    VisualDetail,
    ZoomState,
    slope_light,
    stable_visual_value,
    terrain_colour,
    transition_mask,
    visual_details,
    water_wave_phase,
)
from open_world_rpg.world import TerrainType


def test_natural_gradients_cover_every_terrain_and_vary_with_elevation() -> None:
    assert tuple(TERRAIN_GRADIENTS) == tuple(TerrainType)
    assert tuple(TERRAIN_PALETTE) == tuple(TerrainType)
    assert terrain_colour(terrain_type=TerrainType.DEEP_WATER, elevation=-3_000) != terrain_colour(
        terrain_type=TerrainType.DEEP_WATER, elevation=-200
    )
    assert terrain_colour(terrain_type=TerrainType.MOUNTAINS, elevation=1_201) != terrain_colour(
        terrain_type=TerrainType.MOUNTAINS, elevation=3_000
    )
    assert terrain_colour(terrain_type=TerrainType.PLAINS, elevation=-99_999, light=-999) == (
        0,
        0,
        0,
    )
    assert terrain_colour(terrain_type=TerrainType.PLAINS, elevation=99_999, light=999) == (
        255,
        255,
        255,
    )


def test_colour_and_slope_validation_and_directional_light() -> None:
    assert slope_light(centre=0, west=100, east=0, north=100, south=0) > 0
    assert slope_light(centre=0, west=0, east=100, north=0, south=100) < 0
    assert slope_light(centre=0, west=10_000, east=0, north=0, south=0) == 36
    with pytest.raises(TypeError):
        terrain_colour(terrain_type=cast(Any, "plains"), elevation=0)
    with pytest.raises(TypeError):
        terrain_colour(terrain_type=TerrainType.PLAINS, elevation=True)
    with pytest.raises(TypeError):
        terrain_colour(terrain_type=TerrainType.PLAINS, elevation=0, light=True)
    with pytest.raises(TypeError):
        slope_light(centre=True, west=0, east=0, north=0, south=0)
    with pytest.raises(ValueError):
        slope_light(centre=0, west=0, east=0, north=0, south=0, strength=0)


def test_details_masks_and_hashing_are_stable_and_coordinate_sensitive() -> None:
    first = stable_visual_value(seed=4, world_x=-1, world_y=2, channel="grass")
    assert first == stable_visual_value(seed=4, world_x=-1, world_y=2, channel="grass")
    assert first != stable_visual_value(seed=4, world_x=0, world_y=2, channel="grass")
    assert not transition_mask(
        seed=4,
        world_x=1,
        world_y=2,
        terrain_type=TerrainType.PLAINS,
        neighbour_type=TerrainType.PLAINS,
    )
    mask = transition_mask(
        seed=4,
        world_x=1,
        world_y=2,
        terrain_type=TerrainType.PLAINS,
        neighbour_type=TerrainType.HILLS,
    )
    assert mask is transition_mask(
        seed=4,
        world_x=1,
        world_y=2,
        terrain_type=TerrainType.PLAINS,
        neighbour_type=TerrainType.HILLS,
    )
    with pytest.raises(TypeError):
        transition_mask(
            seed=4,
            world_x=1,
            world_y=2,
            terrain_type=cast(Any, "plains"),
            neighbour_type=TerrainType.HILLS,
        )


@pytest.mark.parametrize("terrain_type", list(TerrainType))
def test_visual_details_are_deterministic_for_every_category(
    terrain_type: TerrainType,
) -> None:
    results = [
        visual_details(seed=8, world_x=x, world_y=-3, terrain_type=terrain_type) for x in range(30)
    ]
    assert results == [
        visual_details(seed=8, world_x=x, world_y=-3, terrain_type=terrain_type) for x in range(30)
    ]
    assert any(results)
    assert all(isinstance(detail, VisualDetail) for details in results for detail in details)
    with pytest.raises(FrozenInstanceError):
        next(detail for details in results for detail in details).kind = "changed"  # type: ignore[misc]


def test_detail_and_water_validation_and_phase() -> None:
    with pytest.raises(TypeError):
        visual_details(
            seed=1,
            world_x=0,
            world_y=0,
            terrain_type=cast(Any, "water"),
        )
    assert (
        water_wave_phase(world_x=-2, world_y=5, animation_tick=9) == ((-2 * 3) + (5 * 5) + 9) % 24
    )
    with pytest.raises(TypeError):
        water_wave_phase(world_x=True, world_y=0, animation_tick=0)


def test_zoom_clamps_and_preserves_immutable_policy() -> None:
    zoom = ZoomState(tile_size_pixels=20)
    assert zoom.changed(steps=1).tile_size_pixels == 22
    assert zoom.changed(steps=999).tile_size_pixels == 48
    assert zoom.changed(steps=-999).tile_size_pixels == 6
    assert ZoomState(tile_size_pixels=6).changed(steps=-1) is not None
    with pytest.raises(TypeError):
        zoom.changed(steps=cast(Any, True))
    with pytest.raises(TypeError):
        ZoomState(tile_size_pixels=cast(Any, True))
    with pytest.raises(ValueError):
        ZoomState(tile_size_pixels=5)
    with pytest.raises(ValueError):
        ZoomState(
            tile_size_pixels=10,
            minimum_tile_size_pixels=20,
            maximum_tile_size_pixels=10,
        )
