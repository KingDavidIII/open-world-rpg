"""Save-game persistence and runtime storage infrastructure."""

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
    "MAX_SAVE_SLOT_LENGTH",
    "SAVE_FILE_SUFFIX",
    "RuntimeStorage",
    "SaveSlot",
    "StorageError",
    "StoragePreparationError",
    "StorageWriteError",
]
