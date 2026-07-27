"""Pure batched dropped-item geometry using the existing voxel atlas."""

from __future__ import annotations

import math
from array import array

from open_world_rpg.gameplay import DroppedItem, ItemType

from .texture_atlas import FaceTexture, atlas_uv

_TEXTURE = {
    ItemType.GRASS_BLOCK: FaceTexture.GRASS_TOP,
    ItemType.DIRT_BLOCK: FaceTexture.DIRT,
    ItemType.STONE_BLOCK: FaceTexture.STONE,
    ItemType.SAND_BLOCK: FaceTexture.SAND,
    ItemType.SNOW_BLOCK: FaceTexture.SNOW_TOP,
    ItemType.WOOD_LOG: FaceTexture.SAND,
    ItemType.WOOD_PLANK: FaceTexture.SAND,
    ItemType.STICK: FaceTexture.GRASS_SIDE,
}


def build_dropped_item_vertices(items: tuple[DroppedItem, ...]) -> bytes:
    """Build one deterministic crossed-quad batch; settled items gently bob."""
    if not isinstance(items, tuple):
        raise TypeError("items must be a tuple.")
    if any(not isinstance(item, DroppedItem) for item in items):
        raise TypeError("items must contain DroppedItem values.")
    vertices = array("f")
    for item in items:
        x, y, z = item.position
        y += math.sin(item.age * 2.5 + item.identifier) * 0.04 if item.settled else 0.0
        starter_resource = item.item is ItemType.WOOD_LOG
        if starter_resource:
            y += 0.12
        radius = 0.34 if starter_resource else 0.18
        shade = 1.15 if starter_resource else 0.9
        corners = (
            (x - radius, y - radius, z - radius),
            (x + radius, y - radius, z + radius),
            (x + radius, y + radius, z + radius),
            (x - radius, y + radius, z - radius),
            (x - radius, y - radius, z + radius),
            (x + radius, y - radius, z - radius),
            (x + radius, y + radius, z - radius),
            (x - radius, y + radius, z + radius),
        )
        u0, v0, u1, v1 = atlas_uv(_TEXTURE[item.item])
        uv = ((u0, v1), (u1, v1), (u1, v0), (u0, v1), (u1, v0), (u0, v0))
        for indices in ((0, 1, 2, 0, 2, 3), (4, 5, 6, 4, 6, 7)):
            for vertex_index, uv_value in zip(indices, uv, strict=True):
                vertices.extend((*corners[vertex_index], *uv_value, shade))
    return vertices.tobytes()
