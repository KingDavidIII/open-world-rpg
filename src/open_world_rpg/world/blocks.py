"""Stable editable-block identities and deterministic in-memory edit overlays."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from open_world_rpg.world.coordinates import CHUNK_SIZE, ChunkCoordinate, LocalTileCoordinate


class BlockMaterial(StrEnum):
    """Persistence-safe material identities independent of presentation."""

    AIR = "air"
    GRASS = "grass"
    DIRT = "dirt"
    STONE = "stone"
    SAND = "sand"
    SNOW = "snow"
    WOOD = "wood"
    LEAVES = "leaves"
    WATER = "water"

    @property
    def is_solid(self) -> bool:
        """Return whether the material participates in player collision."""
        return self not in (BlockMaterial.AIR, BlockMaterial.WATER)


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class WorldBlockCoordinate:
    """Absolute integer voxel coordinate ordered by x, then y, then z."""

    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")

    @property
    def chunk_coordinate(self) -> ChunkCoordinate:
        """Return the owning horizontal chunk using floor division."""
        return ChunkCoordinate(x=self.x // CHUNK_SIZE, y=self.z // CHUNK_SIZE)

    @property
    def local_coordinate(self) -> LocalTileCoordinate:
        """Return the validated local horizontal coordinate."""
        return LocalTileCoordinate(x=self.x % CHUNK_SIZE, y=self.z % CHUNK_SIZE)

    def offset(self, *, x: int = 0, y: int = 0, z: int = 0) -> WorldBlockCoordinate:
        """Return a coordinate offset by validated integer components."""
        for name, value in (("x", x), ("y", y), ("z", z)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} offset must be an integer.")
        return WorldBlockCoordinate(x=self.x + x, y=self.y + y, z=self.z + z)


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockEdit:
    """One immutable override of generated voxel content."""

    coordinate: WorldBlockCoordinate
    material: BlockMaterial
    revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate, WorldBlockCoordinate):
            raise TypeError("coordinate must be a WorldBlockCoordinate.")
        if not isinstance(self.material, BlockMaterial):
            raise TypeError("material must be a BlockMaterial.")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer.")
        if self.revision <= 0:
            raise ValueError("revision must be positive.")


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockEditStoreSnapshot:
    """Immutable deterministic projection of an edit store."""

    revision: int
    edits: tuple[BlockEdit, ...]

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer.")
        if self.revision < 0:
            raise ValueError("revision must be non-negative.")
        if not isinstance(self.edits, tuple):
            raise TypeError("edits must be a tuple.")
        coordinates: set[WorldBlockCoordinate] = set()
        for edit in self.edits:
            if not isinstance(edit, BlockEdit):
                raise TypeError("edits must contain BlockEdit values.")
            if edit.coordinate in coordinates:
                raise ValueError("edits must not contain duplicate coordinates.")
            if edit.revision > self.revision:
                raise ValueError("edit revision cannot exceed store revision.")
            coordinates.add(edit.coordinate)
        if self.edits != tuple(sorted(self.edits, key=lambda item: item.coordinate)):
            raise ValueError("edits must use deterministic coordinate ordering.")


class BlockEditStore:
    """Controlled mutable overlay indexed by absolute and chunk coordinates."""

    def __init__(self) -> None:
        self._revision = 0
        self._edits: dict[WorldBlockCoordinate, BlockEdit] = {}
        self._chunks: dict[ChunkCoordinate, set[WorldBlockCoordinate]] = {}

    @classmethod
    def from_snapshot(cls, snapshot: BlockEditStoreSnapshot) -> BlockEditStore:
        """Atomically construct a store from a fully validated snapshot."""
        if not isinstance(snapshot, BlockEditStoreSnapshot):
            raise TypeError("snapshot must be a BlockEditStoreSnapshot.")
        store = cls()
        store._revision = snapshot.revision
        store._edits = {edit.coordinate: edit for edit in snapshot.edits}
        for coordinate in store._edits:
            store._chunks.setdefault(coordinate.chunk_coordinate, set()).add(coordinate)
        return store

    @property
    def revision(self) -> int:
        return self._revision

    def __len__(self) -> int:
        return len(self._edits)

    def get(self, coordinate: WorldBlockCoordinate) -> BlockEdit | None:
        self._require_coordinate(coordinate)
        return self._edits.get(coordinate)

    def contains(self, coordinate: WorldBlockCoordinate) -> bool:
        self._require_coordinate(coordinate)
        return coordinate in self._edits

    def set_block(self, coordinate: WorldBlockCoordinate, material: BlockMaterial) -> BlockEdit:
        self._require_coordinate(coordinate)
        if not isinstance(material, BlockMaterial):
            raise TypeError("material must be a BlockMaterial.")
        current = self._edits.get(coordinate)
        if current is not None and current.material is material:
            return current
        self._revision += 1
        edit = BlockEdit(coordinate=coordinate, material=material, revision=self._revision)
        self._edits[coordinate] = edit
        self._chunks.setdefault(coordinate.chunk_coordinate, set()).add(coordinate)
        return edit

    def remove_override(self, coordinate: WorldBlockCoordinate) -> bool:
        self._require_coordinate(coordinate)
        if coordinate not in self._edits:
            return False
        del self._edits[coordinate]
        chunk = coordinate.chunk_coordinate
        coordinates = self._chunks[chunk]
        coordinates.remove(coordinate)
        if not coordinates:
            del self._chunks[chunk]
        self._revision += 1
        return True

    def coordinates(self) -> tuple[WorldBlockCoordinate, ...]:
        return tuple(sorted(self._edits))

    def edits_for_chunk(self, coordinate: ChunkCoordinate) -> tuple[BlockEdit, ...]:
        if not isinstance(coordinate, ChunkCoordinate):
            raise TypeError("coordinate must be a ChunkCoordinate.")
        return tuple(self._edits[item] for item in sorted(self._chunks.get(coordinate, ())))

    def snapshot(self) -> BlockEditStoreSnapshot:
        return BlockEditStoreSnapshot(
            revision=self._revision,
            edits=tuple(self._edits[item] for item in sorted(self._edits)),
        )

    def clear(self) -> bool:
        if not self._edits:
            return False
        self._edits.clear()
        self._chunks.clear()
        self._revision += 1
        return True

    @staticmethod
    def _require_coordinate(coordinate: WorldBlockCoordinate) -> None:
        if not isinstance(coordinate, WorldBlockCoordinate):
            raise TypeError("coordinate must be a WorldBlockCoordinate.")
