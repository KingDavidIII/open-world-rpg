"""Save-game persistence and runtime storage infrastructure."""

from open_world_rpg.persistence.document import (
    CURRENT_SAVE_SCHEMA_VERSION,
    JsonValue,
    SaveCompatibilityError,
    SaveCorruptionError,
    SaveDocument,
    SaveDocumentError,
    SessionSnapshot,
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
    "RuntimeStorage",
    "SaveCompatibilityError",
    "SaveCorruptionError",
    "SaveDocument",
    "SaveDocumentError",
    "SaveSlot",
    "SessionSnapshot",
    "StorageError",
    "StoragePreparationError",
    "StorageWriteError",
]
