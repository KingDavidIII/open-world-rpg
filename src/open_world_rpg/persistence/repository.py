"""Repository boundary for versioned save-game documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from open_world_rpg.persistence.document import (
    SaveCorruptionError,
    SaveDocument,
)
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


class SaveRecoveryError(SaveRepositoryError):
    """Raised when a valid backup cannot be restored to the primary slot."""


@dataclass(frozen=True, slots=True)
class SaveLoadResult:
    """One validated load result with explicit backup-recovery status."""

    document: SaveDocument
    recovered_from_backup: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.document, SaveDocument):
            raise TypeError("document must be a SaveDocument.")
        if not isinstance(self.recovered_from_backup, bool):
            raise TypeError("recovered_from_backup must be a boolean.")


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
        """Read a slot, recovering its primary file from a valid backup when needed."""
        return self.load_with_status(slot).document

    def load_with_status(self, slot: SaveSlot) -> SaveLoadResult:
        """Read a slot and report whether automatic backup recovery was required."""
        if not isinstance(slot, SaveSlot):
            raise TypeError("slot must be a SaveSlot.")

        try:
            return SaveLoadResult(document=self._load_path(self.storage.save_path(slot), slot=slot))
        except (SaveSlotNotFoundError, SaveReadError, SaveCorruptionError) as primary_error:
            try:
                backup_document = self._load_path(
                    self.storage.backup_path(slot),
                    slot=slot,
                    source_label="backup",
                )
            except (
                SaveSlotNotFoundError,
                SaveReadError,
                SaveCorruptionError,
            ):
                raise primary_error from primary_error.__cause__

            try:
                self.storage.restore_backup(slot)
            except StorageError as exc:
                raise SaveRecoveryError(
                    f"Could not restore the valid backup for save slot {slot.name!r}."
                ) from exc

            return SaveLoadResult(
                document=backup_document,
                recovered_from_backup=True,
            )

    @staticmethod
    def _load_path(
        path: Path,
        *,
        slot: SaveSlot,
        source_label: str = "primary",
    ) -> SaveDocument:
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            if source_label == "primary":
                message = f"Save slot {slot.name!r} does not exist."
            else:
                message = f"Backup for save slot {slot.name!r} does not exist."
            raise SaveSlotNotFoundError(message) from exc
        except (OSError, UnicodeError) as exc:
            message = (
                f"Could not read save slot {slot.name!r}."
                if source_label == "primary"
                else f"Could not read backup save slot {slot.name!r}."
            )
            raise SaveReadError(message) from exc

        return SaveDocument.from_json(content)
