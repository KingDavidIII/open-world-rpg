"""Pure deterministic chunk streaming selection."""

from __future__ import annotations

from open_world_rpg.world import CHUNK_SIZE, ChunkCoordinate


def streaming_chunks(
    *, world_x: float, world_z: float, render_distance: int
) -> tuple[ChunkCoordinate, ...]:
    """Return a square render set ordered by increasing z then x."""
    if isinstance(render_distance, bool) or not isinstance(render_distance, int):
        raise TypeError("render_distance must be an integer.")
    if render_distance < 0:
        raise ValueError("render_distance must be non-negative.")
    centre_x = int(world_x // CHUNK_SIZE)
    centre_z = int(world_z // CHUNK_SIZE)
    return tuple(
        ChunkCoordinate(x=x, y=z)
        for z in range(centre_z - render_distance, centre_z + render_distance + 1)
        for x in range(centre_x - render_distance, centre_x + render_distance + 1)
    )
