"""Persisted block-edit schema, compatibility, corruption, and restoration."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from open_world_rpg.application.save_service import (
    BlockEditRestoreError,
    GameSaveService,
)
from open_world_rpg.application.session import GameMode, RuntimeContext
from open_world_rpg.core import ProjectPaths
from open_world_rpg.persistence import (
    PersistedBlockEdit,
    PersistedBlockEditOverlay,
    RuntimeStorage,
    SaveCorruptionError,
    SaveDocument,
    SaveRepository,
    SaveSlot,
)
from open_world_rpg.world import (
    BlockEdit,
    BlockEditStore,
    BlockEditStoreSnapshot,
    BlockMaterial,
    WorldBlockCoordinate,
)

WORLD_ID = UUID("00000000-0000-0000-0000-000000000001")


def context(*, session_id: UUID = WORLD_ID, seed: int = 7) -> RuntimeContext:
    value = RuntimeContext.create(
        session_id=session_id,
        game_mode=GameMode.NEW_GAME,
        world_seed=seed,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    value.start()
    return value


def overlay() -> PersistedBlockEditOverlay:
    return PersistedBlockEditOverlay(
        revision=4,
        edits=(
            PersistedBlockEdit(
                x=-(2**100),
                y=3,
                z=15,
                material=BlockMaterial.AIR,
                revision=1,
            ),
            PersistedBlockEdit(
                x=16,
                y=64,
                z=2**100,
                material=BlockMaterial.SNOW,
                revision=4,
            ),
        ),
    )


def document(*, block_edits: PersistedBlockEditOverlay | None = None) -> SaveDocument:
    return SaveDocument.from_runtime_context(
        context=context(),
        saved_at=datetime(2026, 1, 2, tzinfo=UTC),
        block_edits=block_edits,
    )


def service(tmp_path: Path, *, session_id: UUID = WORLD_ID, seed: int = 7) -> GameSaveService:
    paths = ProjectPaths(
        project_root=tmp_path,
        save_directory=tmp_path,
        log_directory=tmp_path / "logs",
    )
    return GameSaveService(
        repository=SaveRepository(storage=RuntimeStorage(paths=paths)),
        context=context(session_id=session_id, seed=seed),
        logger=logging.getLogger("test.block-edits"),
    )


def test_persisted_record_and_overlay_round_trip_are_canonical() -> None:
    persisted = overlay()
    text = document(block_edits=persisted).to_json()
    assert text.index(str(-(2**100))) < text.index('"x": 16')
    restored = SaveDocument.from_json(text)
    assert restored.block_edits == persisted
    snapshot = persisted.to_snapshot()
    assert PersistedBlockEditOverlay.from_snapshot(snapshot) == persisted
    assert snapshot.edits[0].material is BlockMaterial.AIR
    assert PersistedBlockEdit.from_block_edit(snapshot.edits[0]) == persisted.edits[0]
    assert persisted.edits[0].to_block_edit() == snapshot.edits[0]
    with pytest.raises(TypeError):
        SaveDocument(
            schema_version=document().schema_version,
            saved_at=datetime(2026, 1, 2, tzinfo=UTC),
            session=document().session,
            block_edits=cast(Any, object()),
        )


def test_old_schema_one_document_without_edits_loads_as_empty_overlay() -> None:
    old_text = document().to_json()
    assert '"block_edits"' not in old_text
    restored = SaveDocument.from_json(old_text)
    assert restored.block_edits is None
    restored_store = service(Path.cwd()).restore_block_edits(
        restored,
        expected_world_id=WORLD_ID,
        expected_world_seed=7,
    )
    assert restored_store.snapshot() == BlockEditStoreSnapshot(revision=0, edits=())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("x", True),
        ("y", 1.5),
        ("z", "0"),
        ("revision", False),
        ("revision", 0),
    ],
)
def test_persisted_record_rejects_invalid_integer_fields(field: str, value: object) -> None:
    values: dict[str, object] = {
        "x": 0,
        "y": 1,
        "z": 2,
        "material": BlockMaterial.STONE,
        "revision": 1,
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        PersistedBlockEdit(**cast(Any, values))
    with pytest.raises(TypeError):
        PersistedBlockEdit(
            x=0,
            y=1,
            z=2,
            material=cast(Any, "stone"),
            revision=1,
        )


def test_overlay_rejects_revision_duplicates_order_and_invalid_members() -> None:
    first, second = overlay().edits
    with pytest.raises(TypeError):
        PersistedBlockEditOverlay(revision=True)
    with pytest.raises(ValueError):
        PersistedBlockEditOverlay(revision=-1)
    with pytest.raises(TypeError):
        PersistedBlockEditOverlay(revision=1, edits=cast(Any, []))
    with pytest.raises(TypeError):
        PersistedBlockEditOverlay(revision=1, edits=cast(Any, (object(),)))
    with pytest.raises(ValueError, match="ordering"):
        PersistedBlockEditOverlay(revision=4, edits=(second, first))
    with pytest.raises(ValueError, match="duplicate"):
        PersistedBlockEditOverlay(revision=4, edits=(first, first))
    with pytest.raises(ValueError, match="exceed"):
        PersistedBlockEditOverlay(revision=0, edits=(first,))
    with pytest.raises(TypeError):
        PersistedBlockEditOverlay.from_snapshot(cast(Any, object()))
    with pytest.raises(TypeError):
        PersistedBlockEdit.from_block_edit(cast(Any, object()))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ((), [], "JSON object"),
        (("revision",), True, "revision must be an integer"),
        (("edits",), {}, "JSON array"),
        (("edits",), [1], "must be a JSON object"),
        (("edits",), [{"x": 0}], "missing required fields"),
        (
            ("edits",),
            [{"x": 0, "y": 1, "z": 2, "material": 3, "revision": 1}],
            "material must be a string",
        ),
        (
            ("edits",),
            [{"x": 0, "y": 1, "z": 2, "material": "lava", "revision": 1}],
            "unsupported material",
        ),
        (
            ("edits",),
            [{"x": True, "y": 1, "z": 2, "material": "stone", "revision": 1}],
            "overlay is invalid",
        ),
    ],
)
def test_json_corruption_is_rejected(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    raw = json.loads(document(block_edits=overlay()).to_json())
    if not path:
        raw["block_edits"] = value
    else:
        raw["block_edits"][path[0]] = value
    with pytest.raises(SaveCorruptionError, match=message):
        SaveDocument.from_json(json.dumps(raw))


def test_duplicate_and_store_revision_corruption_are_rejected() -> None:
    raw = json.loads(document(block_edits=overlay()).to_json())
    raw["block_edits"]["edits"].append(dict(raw["block_edits"]["edits"][0]))
    with pytest.raises(SaveCorruptionError, match="invalid"):
        SaveDocument.from_json(json.dumps(raw))
    raw = json.loads(document(block_edits=overlay()).to_json())
    raw["block_edits"]["revision"] = 0
    with pytest.raises(SaveCorruptionError, match="invalid"):
        SaveDocument.from_json(json.dumps(raw))


def test_save_service_restores_atomically_and_validates_world_scope(tmp_path: Path) -> None:
    save_service = service(tmp_path)
    store = BlockEditStore.from_snapshot(overlay().to_snapshot())
    slot = SaveSlot("voxel")
    save_service.save(slot=slot, block_edits=store.snapshot())
    loaded = save_service.load(slot)
    restored = save_service.restore_block_edits(
        loaded,
        expected_world_id=WORLD_ID,
        expected_world_seed=7,
    )
    assert restored.snapshot() == store.snapshot()
    with pytest.raises(BlockEditRestoreError, match="identity"):
        save_service.restore_block_edits(
            loaded,
            expected_world_id=UUID(int=2),
            expected_world_seed=7,
        )
    with pytest.raises(BlockEditRestoreError, match="seed"):
        save_service.restore_block_edits(
            loaded,
            expected_world_id=WORLD_ID,
            expected_world_seed=8,
        )
    with pytest.raises(TypeError):
        save_service.restore_block_edits(
            cast(Any, object()),
            expected_world_id=WORLD_ID,
            expected_world_seed=7,
        )
    with pytest.raises(TypeError):
        save_service.restore_block_edits(
            loaded,
            expected_world_id=cast(Any, "id"),
            expected_world_seed=7,
        )
    with pytest.raises(TypeError):
        save_service.restore_block_edits(
            loaded,
            expected_world_id=WORLD_ID,
            expected_world_seed=True,
        )


def test_block_edit_store_snapshot_restoration_is_atomic_and_indexed() -> None:
    snapshot = overlay().to_snapshot()
    store = BlockEditStore.from_snapshot(snapshot)
    assert store.snapshot() == snapshot
    assert store.edits_for_chunk(snapshot.edits[0].coordinate.chunk_coordinate)
    with pytest.raises(TypeError):
        BlockEditStore.from_snapshot(cast(Any, object()))
    first = snapshot.edits[0]
    with pytest.raises(ValueError, match="duplicate"):
        BlockEditStoreSnapshot(revision=4, edits=(first, first))
    with pytest.raises(ValueError, match="ordering"):
        BlockEditStoreSnapshot(revision=4, edits=tuple(reversed(snapshot.edits)))
    with pytest.raises(ValueError, match="exceed"):
        BlockEditStoreSnapshot(revision=0, edits=(first,))
    with pytest.raises(TypeError):
        BlockEditStoreSnapshot(revision=True, edits=())
    with pytest.raises(ValueError):
        BlockEditStoreSnapshot(revision=-1, edits=())
    with pytest.raises(TypeError):
        BlockEditStoreSnapshot(revision=0, edits=cast(Any, []))
    with pytest.raises(TypeError):
        BlockEditStoreSnapshot(revision=0, edits=cast(Any, (object(),)))
    edit = BlockEdit(
        coordinate=WorldBlockCoordinate(x=0, y=1, z=2),
        material=BlockMaterial.AIR,
        revision=1,
    )
    assert (
        BlockEditStore.from_snapshot(BlockEditStoreSnapshot(revision=3, edits=(edit,))).revision
        == 3
    )
