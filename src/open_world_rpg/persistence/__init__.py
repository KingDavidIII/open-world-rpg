"""Save-game persistence and runtime storage infrastructure."""

from open_world_rpg.persistence.document import (
    CURRENT_SAVE_SCHEMA_VERSION,
    JsonValue,
    PersistedBlockEdit,
    PersistedBlockEditOverlay,
    SaveCompatibilityError,
    SaveCorruptionError,
    SaveDocument,
    SaveDocumentError,
    SessionSnapshot,
)
from open_world_rpg.persistence.repository import (
    SaveReadError,
    SaveRepository,
    SaveRepositoryError,
    SaveSerialisationError,
    SaveSlotNotFoundError,
    SaveWriteError,
)
from open_world_rpg.persistence.storage import (
    MAX_SAVE_SLOT_LENGTH,
    SAVE_FILE_SUFFIX,
    RuntimeStorage,
    SaveSlot,
    StorageError,
    StoragePreparationError,
    StorageWriteError,
)

__all__ = [
    "CURRENT_SAVE_SCHEMA_VERSION",
    "MAX_SAVE_SLOT_LENGTH",
    "SAVE_FILE_SUFFIX",
    "JsonValue",
    "PersistedBlockEdit",
    "PersistedBlockEditOverlay",
    "RuntimeStorage",
    "SaveCompatibilityError",
    "SaveCorruptionError",
    "SaveDocument",
    "SaveDocumentError",
    "SaveReadError",
    "SaveRepository",
    "SaveRepositoryError",
    "SaveSerialisationError",
    "SaveSlot",
    "SaveSlotNotFoundError",
    "SaveWriteError",
    "SessionSnapshot",
    "StorageError",
    "StoragePreparationError",
    "StorageWriteError",
]
