"""Original deterministic pixel-art atlas generation and UV policy."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Final

ATLAS_TILE_SIZE: Final = 16
ATLAS_COLUMNS: Final = 3
ATLAS_ROWS: Final = 3
ATLAS_SIZE: Final = ATLAS_TILE_SIZE * ATLAS_COLUMNS


class FaceTexture(StrEnum):
    """Nine stable atlas cells for terrain face roles."""

    GRASS_TOP = "grass_top"
    GRASS_SIDE = "grass_side"
    DIRT = "dirt"
    STONE = "stone"
    SAND = "sand"
    SNOW_TOP = "snow_top"
    SNOW_SIDE = "snow_side"
    DEEP_WATER = "deep_water"
    SHALLOW_WATER = "shallow_water"


_BASES: Final[dict[FaceTexture, tuple[int, int, int, int]]] = {
    FaceTexture.GRASS_TOP: (76, 151, 54, 255),
    FaceTexture.GRASS_SIDE: (108, 104, 55, 255),
    FaceTexture.DIRT: (111, 75, 44, 255),
    FaceTexture.STONE: (111, 116, 119, 255),
    FaceTexture.SAND: (205, 184, 119, 255),
    FaceTexture.SNOW_TOP: (225, 237, 241, 255),
    FaceTexture.SNOW_SIDE: (188, 207, 215, 255),
    FaceTexture.DEEP_WATER: (25, 74, 128, 180),
    FaceTexture.SHALLOW_WATER: (39, 137, 171, 168),
}


def _noise(texture: FaceTexture, x: int, y: int) -> int:
    payload = f"open-world-rpg/voxel-atlas/v1:{texture.value}:{x}:{y}".encode()
    return hashlib.blake2b(payload, digest_size=1).digest()[0]


def generate_texture_atlas() -> bytes:
    """Generate one deterministic RGBA atlas with crisp original textures."""
    pixels = bytearray(ATLAS_SIZE * ATLAS_SIZE * 4)
    for index, texture in enumerate(FaceTexture):
        cell_x = index % ATLAS_COLUMNS
        cell_y = index // ATLAS_COLUMNS
        base = _BASES[texture]
        for y in range(ATLAS_TILE_SIZE):
            for x in range(ATLAS_TILE_SIZE):
                noise = _noise(texture, x, y)
                delta = (noise % 19) - 9
                red, green, blue, alpha = base
                if texture is FaceTexture.GRASS_SIDE and y < 4 + (noise % 3):
                    red, green, blue = _BASES[FaceTexture.GRASS_TOP][:3]
                elif texture is FaceTexture.SNOW_SIDE and y < 3 + (noise % 2):
                    red, green, blue = _BASES[FaceTexture.SNOW_TOP][:3]
                elif (
                    texture in (FaceTexture.DEEP_WATER, FaceTexture.SHALLOW_WATER)
                    and (x + y + noise) % 11 == 0
                ):
                    delta += 18
                offset = (
                    ((cell_y * ATLAS_TILE_SIZE + y) * ATLAS_SIZE) + cell_x * ATLAS_TILE_SIZE + x
                ) * 4
                pixels[offset : offset + 4] = bytes(
                    (
                        max(0, min(255, red + delta)),
                        max(0, min(255, green + delta)),
                        max(0, min(255, blue + delta)),
                        alpha,
                    )
                )
    return bytes(pixels)


def atlas_uv(texture: FaceTexture) -> tuple[float, float, float, float]:
    """Return UV bounds inset by half a texel to prevent atlas bleeding."""
    if not isinstance(texture, FaceTexture):
        raise TypeError("texture must be a FaceTexture.")
    index = tuple(FaceTexture).index(texture)
    x = index % ATLAS_COLUMNS
    y = index // ATLAS_COLUMNS
    inset = 0.5 / ATLAS_SIZE
    return (
        x / ATLAS_COLUMNS + inset,
        y / ATLAS_ROWS + inset,
        (x + 1) / ATLAS_COLUMNS - inset,
        (y + 1) / ATLAS_ROWS - inset,
    )
