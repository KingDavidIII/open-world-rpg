"""Repository boundary for versioned save-game documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from open_world_rpg.persistence.document import SaveDocument
from open_world_rpg.persistence.storage import (
    RuntimeStorage,
    SaveSlot,
    StorageError,
)


class SaveRepositoryError(RuntimeError):
    """Base exception for save-repository operations."""


class SaveSlotNotFoundError(SaveRepositoryError):
    """Raised when a requested save slot does not exist."""


class SaveReadError(SaveRepositoryError):
    """Raised when save content cannot be read from storage."""


class SaveWriteError(SaveRepositoryError):
    """Raised when save content cannot be written to storage."""


class SaveSerialisationError(SaveRepositoryError):
    """Raised when a save document cannot be serialised."""


@dataclass(frozen=True, slots=True)
class SaveRepository:
    """Persist and restore strictly validated save documents."""

    storage: RuntimeStorage

    def __post_init__(self) -> None:
        if not isinstance(self.storage, RuntimeStorage):
            raise TypeError("storage must be a RuntimeStorage.")

    def save(
        self,
        *,
        slot: SaveSlot,
        document: SaveDocument,
    ) -> Path:
        """Serialise and atomically persist a save document."""
        if not isinstance(slot, SaveSlot):
            raise TypeError("slot must be a SaveSlot.")

        if not isinstance(document, SaveDocument):
            raise TypeError("document must be a SaveDocument.")

        try:
            content = document.to_json()
        except (TypeError, ValueError, OverflowError) as exc:
            raise SaveSerialisationError(f"Could not serialise save slot {slot.name!r}.") from exc

        try:
            return self.storage.write_save_text(
                slot=slot,
                content=content,
            )
        except StorageError as exc:
            raise SaveWriteError(f"Could not persist save slot {slot.name!r}.") from exc

    def load(self, slot: SaveSlot) -> SaveDocument:
        """Read and strictly validate a save document."""
        if not isinstance(slot, SaveSlot):
            raise TypeError("slot must be a SaveSlot.")

        path = self.storage.save_path(slot)

        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SaveSlotNotFoundError(f"Save slot {slot.name!r} does not exist.") from exc
        except (OSError, UnicodeError) as exc:
            raise SaveReadError(f"Could not read save slot {slot.name!r}.") from exc

        return SaveDocument.from_json(content)
