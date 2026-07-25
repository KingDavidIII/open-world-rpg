"""Tests for the versioned save-game repository."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, cast
from uuid import UUID

import pytest

from open_world_rpg.application.session import GameMode, SessionState
from open_world_rpg.core import ProjectPaths
from open_world_rpg.persistence.document import (
    CURRENT_SAVE_SCHEMA_VERSION,
    SaveCompatibilityError,
    SaveCorruptionError,
    SaveDocument,
    SessionSnapshot,
)
from open_world_rpg.persistence.repository import (
    SaveReadError,
    SaveRepository,
    SaveSerialisationError,
    SaveSlotNotFoundError,
    SaveWriteError,
)
from open_world_rpg.persistence.storage import (
    RuntimeStorage,
    SaveSlot,
    StorageWriteError,
)

SESSION_ID = UUID("12345678-1234-5678-1234-567812345678")
SAVED_AT = datetime(2026, 7, 25, 10, 5, tzinfo=UTC)


def create_repository(tmp_path: Path) -> SaveRepository:
    return SaveRepository(storage=RuntimeStorage(paths=ProjectPaths.from_project_root(tmp_path)))


def create_document() -> SaveDocument:
    return SaveDocument(
        schema_version=CURRENT_SAVE_SCHEMA_VERSION,
        saved_at=SAVED_AT,
        session=SessionSnapshot(
            session_id=SESSION_ID,
            game_mode=GameMode.NEW_GAME,
            world_seed=42,
            state=SessionState.PAUSED,
        ),
        payload={
            "player": {
                "name": "Ọlá",
                "level": 9,
                "inventory": [
                    "iron-sword",
                    "healing-potion",
                ],
            }
        },
    )


def test_repository_rejects_invalid_storage() -> None:
    with pytest.raises(TypeError, match="storage"):
        SaveRepository(storage=cast(Any, object()))


def test_save_persists_canonical_document(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    slot = SaveSlot("campaign-01")
    document = create_document()

    path = repository.save(
        slot=slot,
        document=document,
    )

    assert path == repository.storage.save_path(slot)
    assert path.read_text(encoding="utf-8") == document.to_json()


def test_saved_document_can_be_loaded(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    slot = SaveSlot("campaign-01")
    document = create_document()
    repository.save(slot=slot, document=document)

    loaded = repository.load(slot)

    assert loaded == document


def test_save_replaces_existing_document(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    slot = SaveSlot("autosave")

    original = create_document()
    replacement = SaveDocument(
        schema_version=CURRENT_SAVE_SCHEMA_VERSION,
        saved_at=datetime(2026, 7, 25, 11, 0, tzinfo=UTC),
        session=original.session,
        payload={"player": {"level": 10}},
    )

    repository.save(slot=slot, document=original)
    repository.save(slot=slot, document=replacement)

    assert repository.load(slot) == replacement


def test_save_rejects_invalid_arguments(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    document = create_document()

    with pytest.raises(TypeError, match="slot"):
        repository.save(
            slot=cast(Any, "campaign-01"),
            document=document,
        )

    with pytest.raises(TypeError, match="document"):
        repository.save(
            slot=SaveSlot("campaign-01"),
            document=cast(Any, object()),
        )


def test_load_rejects_invalid_slot(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)

    with pytest.raises(TypeError, match="slot"):
        repository.load(cast(Any, "campaign-01"))


def test_save_wraps_storage_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = create_repository(tmp_path)
    slot = SaveSlot("campaign-01")

    def raise_storage_error(
        self: RuntimeStorage,
        *,
        slot: SaveSlot,
        content: str,
    ) -> NoReturn:
        del self, slot, content
        raise StorageWriteError("disk unavailable")

    monkeypatch.setattr(
        RuntimeStorage,
        "write_save_text",
        raise_storage_error,
    )

    with pytest.raises(
        SaveWriteError,
        match="Could not persist save slot",
    ) as error:
        repository.save(
            slot=slot,
            document=create_document(),
        )

    assert isinstance(error.value.__cause__, StorageWriteError)


def test_save_wraps_serialisation_failure(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    document = create_document()

    mutable_payload = cast(dict[str, Any], document.payload)
    mutable_payload["invalid"] = b"bytes"

    with pytest.raises(
        SaveSerialisationError,
        match="Could not serialise save slot",
    ) as error:
        repository.save(
            slot=SaveSlot("campaign-01"),
            document=document,
        )

    assert isinstance(error.value.__cause__, TypeError)


def test_load_reports_missing_slot(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)

    with pytest.raises(
        SaveSlotNotFoundError,
        match="'missing' does not exist",
    ) as error:
        repository.load(SaveSlot("missing"))

    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_load_wraps_filesystem_read_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = create_repository(tmp_path)

    def raise_read_error(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> NoReturn:
        del self, encoding, errors
        raise PermissionError("access denied")

    monkeypatch.setattr(Path, "read_text", raise_read_error)

    with pytest.raises(
        SaveReadError,
        match="Could not read save slot",
    ) as error:
        repository.load(SaveSlot("campaign-01"))

    assert isinstance(error.value.__cause__, PermissionError)


def test_load_wraps_invalid_utf8(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    slot = SaveSlot("campaign-01")

    repository.storage.prepare()
    repository.storage.save_path(slot).write_bytes(b"\xff\xfe")

    with pytest.raises(
        SaveReadError,
        match="Could not read save slot",
    ) as error:
        repository.load(slot)

    assert isinstance(error.value.__cause__, UnicodeDecodeError)


def test_load_preserves_corruption_error(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    slot = SaveSlot("corrupt")

    repository.storage.prepare()
    repository.storage.save_path(slot).write_text(
        "{invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        SaveCorruptionError,
        match="not valid JSON",
    ):
        repository.load(slot)


def test_load_preserves_compatibility_error(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    slot = SaveSlot("future-version")

    raw_document = {
        "schema_version": 99,
        "saved_at": SAVED_AT.isoformat(),
        "session": {
            "session_id": str(SESSION_ID),
            "game_mode": "new_game",
            "world_seed": 42,
            "state": "paused",
        },
        "payload": {},
    }

    repository.storage.prepare()
    repository.storage.save_path(slot).write_text(
        json.dumps(raw_document),
        encoding="utf-8",
    )

    with pytest.raises(
        SaveCompatibilityError,
        match="Unsupported save schema version 99",
    ):
        repository.load(slot)
