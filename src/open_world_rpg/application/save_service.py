"""Application service for saving and loading game sessions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from open_world_rpg.application.session import RuntimeContext
from open_world_rpg.persistence.document import JsonValue, SaveDocument
from open_world_rpg.persistence.repository import SaveRepository
from open_world_rpg.persistence.storage import SaveSlot


@dataclass(frozen=True, slots=True)
class GameSaveService:
    """Coordinate runtime sessions with persistent save documents."""

    repository: SaveRepository
    context: RuntimeContext
    logger: logging.Logger

    def __post_init__(self) -> None:
        if not isinstance(self.repository, SaveRepository):
            raise TypeError("repository must be a SaveRepository.")

        if not isinstance(self.context, RuntimeContext):
            raise TypeError("context must be a RuntimeContext.")

        if not isinstance(self.logger, logging.Logger):
            raise TypeError("logger must be a logging.Logger.")

    def save(
        self,
        *,
        slot: SaveSlot,
        payload: dict[str, JsonValue] | None = None,
        saved_at: datetime | None = None,
    ) -> Path:
        """Create and persist a document for the current session."""
        if not isinstance(slot, SaveSlot):
            raise TypeError("slot must be a SaveSlot.")

        document: SaveDocument | None = None

        try:
            document = SaveDocument.from_runtime_context(
                context=self.context,
                payload=payload,
                saved_at=saved_at,
            )
            path = self.repository.save(
                slot=slot,
                document=document,
            )
        except Exception:
            self.logger.exception(
                "Game session could not be saved.",
                extra=self._diagnostic_context(
                    event="persistence.save_failed",
                    slot=slot,
                    document=document,
                ),
            )
            raise

        self.logger.info(
            "Game session saved.",
            extra=self._diagnostic_context(
                event="persistence.save_succeeded",
                slot=slot,
                document=document,
            ),
        )
        return path

    def load(self, slot: SaveSlot) -> SaveDocument:
        """Load and validate a document from a named save slot."""
        if not isinstance(slot, SaveSlot):
            raise TypeError("slot must be a SaveSlot.")

        try:
            document = self.repository.load(slot)
        except Exception:
            self.logger.exception(
                "Game session could not be loaded.",
                extra=self._diagnostic_context(
                    event="persistence.load_failed",
                    slot=slot,
                ),
            )
            raise

        self.logger.info(
            "Game session loaded.",
            extra=self._diagnostic_context(
                event="persistence.load_succeeded",
                slot=slot,
                document=document,
            ),
        )
        return document

    def _diagnostic_context(
        self,
        *,
        event: str,
        slot: SaveSlot,
        document: SaveDocument | None = None,
    ) -> dict[str, object]:
        context: dict[str, object] = {
            "event": event,
            "session_id": str(self.context.session_id),
            "world_seed": self.context.world_seed,
            "session_state": self.context.state.value,
            "save_slot": slot.name,
        }

        if document is not None:
            context.update(
                {
                    "schema_version": document.schema_version,
                    "saved_session_id": str(document.session.session_id),
                    "saved_session_state": document.session.state.value,
                }
            )

        return context
