"""Tests for safe paths and atomic runtime storage operations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest

import open_world_rpg.persistence.storage as storage_module
from open_world_rpg.core import ProjectPaths
from open_world_rpg.persistence.storage import (
    MAX_SAVE_SLOT_LENGTH,
    RuntimeStorage,
    SaveSlot,
    StoragePreparationError,
    StorageRecoveryError,
    StorageWriteError,
)


def create_storage(tmp_path: Path) -> RuntimeStorage:
    return RuntimeStorage(paths=ProjectPaths.from_project_root(tmp_path))


def test_save_slot_normalises_name() -> None:
    slot = SaveSlot("  Player_One-01  ")

    assert slot.name == "player_one-01"
    assert slot.file_name == "player_one-01.json"


@pytest.mark.parametrize(
    "name",
    [
        "a",
        "slot-01",
        "slot_01",
        "player1",
        "a" * MAX_SAVE_SLOT_LENGTH,
    ],
)
def test_save_slot_accepts_safe_names(name: str) -> None:
    assert SaveSlot(name).name == name


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (100, TypeError, "name must be a string"),
        ("", ValueError, "cannot be empty"),
        ("   ", ValueError, "cannot be empty"),
        ("a" * (MAX_SAVE_SLOT_LENGTH + 1), ValueError, "cannot exceed"),
        ("-slot", ValueError, "must start and end"),
        ("slot-", ValueError, "must start and end"),
        ("_slot", ValueError, "must start and end"),
        ("slot_", ValueError, "must start and end"),
        ("slot name", ValueError, "may contain"),
        ("slot.json", ValueError, "may contain"),
        ("../slot", ValueError, "may contain"),
        ("con", ValueError, "reserved"),
        ("COM1", ValueError, "reserved"),
        ("lpt9", ValueError, "reserved"),
    ],
)
def test_save_slot_rejects_unsafe_names(
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        SaveSlot(cast(Any, value))


def test_runtime_storage_rejects_invalid_paths() -> None:
    with pytest.raises(TypeError, match="paths"):
        RuntimeStorage(paths=cast(Any, object()))


def test_prepare_creates_runtime_directories(tmp_path: Path) -> None:
    storage = create_storage(tmp_path)

    storage.prepare()

    assert storage.paths.save_directory.is_dir()
    assert storage.paths.log_directory.is_dir()


def test_prepare_is_idempotent(tmp_path: Path) -> None:
    storage = create_storage(tmp_path)

    storage.prepare()
    storage.prepare()

    assert storage.paths.save_directory.is_dir()
    assert storage.paths.log_directory.is_dir()


def test_prepare_wraps_filesystem_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = create_storage(tmp_path)

    def raise_directory_error(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> NoReturn:
        del self, mode, parents, exist_ok
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "mkdir", raise_directory_error)

    with pytest.raises(
        StoragePreparationError,
        match="Could not prepare runtime directory",
    ) as error:
        storage.prepare()

    assert isinstance(error.value.__cause__, OSError)


def test_save_path_returns_canonical_location(tmp_path: Path) -> None:
    storage = create_storage(tmp_path)

    path = storage.save_path(SaveSlot("campaign-01"))

    assert path == storage.paths.save_directory / "campaign-01.json"


def test_save_path_rejects_invalid_slot(tmp_path: Path) -> None:
    storage = create_storage(tmp_path)

    with pytest.raises(TypeError, match="slot"):
        storage.save_path(cast(Any, "campaign-01"))


def test_write_save_text_creates_utf8_file(tmp_path: Path) -> None:
    storage = create_storage(tmp_path)
    content = '{"hero": "Ọlá", "level": 4}\n'

    destination = storage.write_save_text(
        slot=SaveSlot("campaign-01"),
        content=content,
    )

    assert destination == storage.save_path(SaveSlot("campaign-01"))
    assert destination.read_text(encoding="utf-8") == content
    assert storage.paths.log_directory.is_dir()


def test_write_save_text_replaces_existing_file(tmp_path: Path) -> None:
    storage = create_storage(tmp_path)
    slot = SaveSlot("autosave")

    destination = storage.write_save_text(
        slot=slot,
        content="old state",
    )
    returned_path = storage.write_save_text(
        slot=slot,
        content="new state",
    )

    assert returned_path == destination
    assert destination.read_text(encoding="utf-8") == "new state"


def test_write_save_text_rejects_non_string_content(
    tmp_path: Path,
) -> None:
    storage = create_storage(tmp_path)

    with pytest.raises(TypeError, match="content"):
        storage.write_save_text(
            slot=SaveSlot("slot"),
            content=cast(Any, b"data"),
        )


def test_replace_failure_is_wrapped_and_temp_file_is_removed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = create_storage(tmp_path)
    slot = SaveSlot("slot")

    def raise_replace_error(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
    ) -> NoReturn:
        del source, destination
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", raise_replace_error)

    with pytest.raises(
        StorageWriteError,
        match="Could not atomically write",
    ) as error:
        storage.write_save_text(
            slot=slot,
            content="save data",
        )

    assert isinstance(error.value.__cause__, OSError)
    assert storage.save_path(slot).exists() is False
    assert list(storage.paths.save_directory.glob("*.tmp")) == []


def test_temporary_file_creation_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = create_storage(tmp_path)

    def raise_temporary_file_error(
        *args: object,
        **kwargs: object,
    ) -> NoReturn:
        del args, kwargs
        raise OSError("temporary file unavailable")

    monkeypatch.setattr(
        storage_module,
        "NamedTemporaryFile",
        raise_temporary_file_error,
    )

    with pytest.raises(
        StorageWriteError,
        match="Could not atomically write",
    ) as error:
        storage.write_save_text(
            slot=SaveSlot("slot"),
            content="save data",
        )

    assert isinstance(error.value.__cause__, OSError)


def test_cleanup_failure_does_not_hide_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = create_storage(tmp_path)

    def raise_replace_error(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
    ) -> NoReturn:
        del source, destination
        raise OSError("replace failed")

    def raise_cleanup_error(
        self: Path,
        missing_ok: bool = False,
    ) -> NoReturn:
        del self, missing_ok
        raise OSError("cleanup failed")

    monkeypatch.setattr(os, "replace", raise_replace_error)
    monkeypatch.setattr(Path, "unlink", raise_cleanup_error)

    with pytest.raises(StorageWriteError) as error:
        storage.write_save_text(
            slot=SaveSlot("slot"),
            content="save data",
        )

    assert str(error.value.__cause__) == "replace failed"


def test_backup_path_and_file_name_are_canonical(tmp_path: Path) -> None:
    storage = create_storage(tmp_path)
    slot = SaveSlot("Campaign-01")

    assert slot.backup_file_name == "campaign-01.backup.json"
    assert storage.backup_path(slot) == storage.paths.save_directory / slot.backup_file_name

    with pytest.raises(TypeError, match="slot"):
        storage.backup_path(cast(Any, "campaign-01"))


def test_replacing_save_retains_previous_primary_as_backup(tmp_path: Path) -> None:
    storage = create_storage(tmp_path)
    slot = SaveSlot("autosave")

    storage.write_save_text(slot=slot, content="first")
    storage.write_save_text(slot=slot, content="second")

    assert storage.save_path(slot).read_text(encoding="utf-8") == "second"
    assert storage.backup_path(slot).read_text(encoding="utf-8") == "first"


def test_existing_primary_rotation_failure_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = create_storage(tmp_path)
    slot = SaveSlot("autosave")
    storage.write_save_text(slot=slot, content="stable")
    real_replace = os.replace

    def reject_backup_rotation(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
    ) -> None:
        if Path(source) == storage.save_path(slot):
            raise OSError("backup unavailable")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", reject_backup_rotation)

    with pytest.raises(StorageWriteError, match="atomically write"):
        storage.write_save_text(slot=slot, content="new")

    assert storage.save_path(slot).read_text(encoding="utf-8") == "stable"
    assert list(storage.paths.save_directory.glob("*.tmp")) == []


def test_restore_backup_recreates_primary_without_consuming_backup(tmp_path: Path) -> None:
    storage = create_storage(tmp_path)
    slot = SaveSlot("autosave")
    storage.write_save_text(slot=slot, content="stable")
    storage.write_save_text(slot=slot, content="new")
    storage.save_path(slot).write_text("corrupt", encoding="utf-8")

    restored = storage.restore_backup(slot)

    assert restored == storage.save_path(slot)
    assert restored.read_text(encoding="utf-8") == "stable"
    assert storage.backup_path(slot).read_text(encoding="utf-8") == "stable"


def test_restore_backup_rejects_invalid_slot(tmp_path: Path) -> None:
    storage = create_storage(tmp_path)

    with pytest.raises(TypeError, match="slot"):
        storage.restore_backup(cast(Any, "autosave"))


def test_restore_backup_wraps_missing_or_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = create_storage(tmp_path)
    slot = SaveSlot("autosave")

    with pytest.raises(StorageRecoveryError, match="restore backup") as missing:
        storage.restore_backup(slot)
    assert isinstance(missing.value.__cause__, FileNotFoundError)

    storage.write_save_text(slot=slot, content="stable")
    storage.write_save_text(slot=slot, content="new")

    def reject_restore(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
    ) -> NoReturn:
        del source, destination
        raise OSError("restore failed")

    monkeypatch.setattr(os, "replace", reject_restore)

    with pytest.raises(StorageRecoveryError, match="restore backup") as failed:
        storage.restore_backup(slot)

    assert isinstance(failed.value.__cause__, OSError)
    assert list(storage.paths.save_directory.glob("*.tmp")) == []
