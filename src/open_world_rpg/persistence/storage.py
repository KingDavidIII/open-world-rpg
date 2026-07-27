"""Safe runtime storage paths and atomic save-file operations."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final

from open_world_rpg.core import ProjectPaths

MAX_SAVE_SLOT_LENGTH: Final = 64
SAVE_FILE_SUFFIX: Final = ".json"
SAVE_BACKUP_SUFFIX: Final = ".backup.json"

_SAVE_SLOT_PATTERN: Final = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")

_RESERVED_WINDOWS_NAMES: Final = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)


class StorageError(RuntimeError):
    """Base exception for runtime storage failures."""


class StoragePreparationError(StorageError):
    """Raised when required runtime directories cannot be prepared."""


class StorageWriteError(StorageError):
    """Raised when a save file cannot be written atomically."""


class StorageRecoveryError(StorageError):
    """Raised when a valid backup cannot be restored atomically."""


@dataclass(frozen=True, slots=True)
class SaveSlot:
    """Validated filesystem-safe identity for one save slot."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string.")

        normalised = self.name.strip().lower()

        if not normalised:
            raise ValueError("name cannot be empty.")

        if len(normalised) > MAX_SAVE_SLOT_LENGTH:
            raise ValueError(f"name cannot exceed {MAX_SAVE_SLOT_LENGTH} characters.")

        if _SAVE_SLOT_PATTERN.fullmatch(normalised) is None:
            raise ValueError(
                "name may contain lowercase letters, numbers, hyphens, and "
                "underscores, and must start and end with a letter or number."
            )

        if normalised in _RESERVED_WINDOWS_NAMES:
            raise ValueError("name is reserved by the operating system.")

        object.__setattr__(self, "name", normalised)

    @property
    def file_name(self) -> str:
        """Return the canonical JSON file name for this slot."""
        return f"{self.name}{SAVE_FILE_SUFFIX}"

    @property
    def backup_file_name(self) -> str:
        """Return the canonical rotating-backup file name for this slot."""
        return f"{self.name}{SAVE_BACKUP_SUFFIX}"


@dataclass(frozen=True, slots=True)
class RuntimeStorage:
    """Manage canonical runtime directories and atomic save writes."""

    paths: ProjectPaths

    def __post_init__(self) -> None:
        if not isinstance(self.paths, ProjectPaths):
            raise TypeError("paths must be a ProjectPaths.")

    def prepare(self) -> None:
        """Create all required runtime directories."""
        for directory in (
            self.paths.save_directory,
            self.paths.log_directory,
        ):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise StoragePreparationError(
                    f"Could not prepare runtime directory: {directory}"
                ) from exc

    def save_path(self, slot: SaveSlot) -> Path:
        """Return the canonical path of a save slot."""
        if not isinstance(slot, SaveSlot):
            raise TypeError("slot must be a SaveSlot.")

        return self.paths.save_directory / slot.file_name

    def backup_path(self, slot: SaveSlot) -> Path:
        """Return the canonical rotating-backup path of a save slot."""
        if not isinstance(slot, SaveSlot):
            raise TypeError("slot must be a SaveSlot.")

        return self.paths.save_directory / slot.backup_file_name

    def write_save_text(
        self,
        *,
        slot: SaveSlot,
        content: str,
    ) -> Path:
        """Atomically write UTF-8 text while retaining the previous valid file."""
        if not isinstance(content, str):
            raise TypeError("content must be a string.")

        destination = self.save_path(slot)
        backup = self.backup_path(slot)
        self.prepare()

        temporary_path: Path | None = None

        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            if destination.exists():
                os.replace(destination, backup)

            os.replace(temporary_path, destination)
        except OSError as exc:
            if temporary_path is not None:
                self._discard_temporary_file(temporary_path)

            raise StorageWriteError(f"Could not atomically write save slot {slot.name!r}.") from exc

        return destination

    def restore_backup(self, slot: SaveSlot) -> Path:
        """Restore a backup to the primary slot without consuming the backup."""
        if not isinstance(slot, SaveSlot):
            raise TypeError("slot must be a SaveSlot.")

        destination = self.save_path(slot)
        backup = self.backup_path(slot)
        self.prepare()
        temporary_path: Path | None = None

        try:
            content = backup.read_bytes()
            with NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.recovery.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
        except OSError as exc:
            if temporary_path is not None:
                self._discard_temporary_file(temporary_path)
            raise StorageRecoveryError(
                f"Could not restore backup for save slot {slot.name!r}."
            ) from exc

        return destination

    @staticmethod
    def _discard_temporary_file(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return
