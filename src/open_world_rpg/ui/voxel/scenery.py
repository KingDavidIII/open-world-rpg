"""Deterministic UI-only voxel scenery placement."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from open_world_rpg.world import CHUNK_SIZE, ChunkCoordinate

from .blocks import BlockColumn, BlockType


class SceneryKind(StrEnum):
    """Lightweight decorative mesh identities."""

    TREE = "tree"
    GRASS_TUFT = "grass_tuft"
    ROCK = "rock"
    SHRUB = "shrub"


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneryPlacement:
    """A coordinate-owned decoration that never becomes a gameplay entity."""

    kind: SceneryKind
    world_x: int
    world_z: int
    height: int

    @property
    def owner(self) -> ChunkCoordinate:
        return ChunkCoordinate(
            x=self.world_x // CHUNK_SIZE,
            y=self.world_z // CHUNK_SIZE,
        )


def scenery_at(
    *,
    seed: int,
    world_x: int,
    world_z: int,
    column: BlockColumn,
    slope: int,
) -> SceneryPlacement | None:
    """Place restrained scenery only on dry, gently sloped grass."""
    if column.water is not None or column.surface is not BlockType.GRASS or slope > 1:
        return None
    payload = f"open-world-rpg/voxel-scenery/v1:{seed}:{world_x}:{world_z}".encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    if value % 97 == 0:
        kind = SceneryKind.TREE
    elif value % 29 == 0:
        kind = SceneryKind.GRASS_TUFT
    elif value % 71 == 0:
        kind = SceneryKind.ROCK
    elif value % 53 == 0:
        kind = SceneryKind.SHRUB
    else:
        return None
    return SceneryPlacement(
        kind=kind,
        world_x=world_x,
        world_z=world_z,
        height=column.ground_height,
    )
