"""Restore persisted save documents into resumable runtime sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from open_world_rpg.application.session import Clock, RuntimeContext
from open_world_rpg.core import GameConfig
from open_world_rpg.persistence.document import JsonValue, SaveDocument


class SessionRestoreError(RuntimeError):
    """Base exception for runtime-session restoration failures."""


class SessionConfigurationMismatchError(SessionRestoreError):
    """Raised when the current configuration cannot load a save."""


@dataclass(frozen=True, slots=True)
class RestoredGameSession:
    """A restored runtime context paired with its source document."""

    context: RuntimeContext
    document: SaveDocument

    def __post_init__(self) -> None:
        if not isinstance(self.context, RuntimeContext):
            raise TypeError("context must be a RuntimeContext.")

        if not isinstance(self.document, SaveDocument):
            raise TypeError("document must be a SaveDocument.")

        snapshot = self.document.session

        if self.context.session_id != snapshot.session_id:
            raise ValueError("Restored context session identity does not match the save.")

        if self.context.game_mode is not snapshot.game_mode:
            raise ValueError("Restored context game mode does not match the save.")

        if self.context.world_seed != snapshot.world_seed:
            raise ValueError("Restored context world seed does not match the save.")

        if self.context.state is not snapshot.state:
            raise ValueError("Restored context state does not match the save.")

    @property
    def payload(self) -> dict[str, JsonValue]:
        """Return an isolated copy of the restored gameplay payload."""
        return deepcopy(self.document.payload)


def restore_game_session(
    *,
    document: SaveDocument,
    config: GameConfig,
    clock: Clock | None = None,
) -> RestoredGameSession:
    """Restore a strictly compatible save document into a runtime context."""
    if not isinstance(document, SaveDocument):
        raise TypeError("document must be a SaveDocument.")

    if not isinstance(config, GameConfig):
        raise TypeError("config must be a GameConfig.")

    saved_seed = document.session.world_seed
    configured_seed = config.simulation.world_seed

    if saved_seed != configured_seed:
        raise SessionConfigurationMismatchError(
            "Saved world seed does not match the configured world seed: "
            f"saved={saved_seed}, configured={configured_seed}."
        )

    context = RuntimeContext.restore(
        session_id=document.session.session_id,
        game_mode=document.session.game_mode,
        world_seed=document.session.world_seed,
        state=document.session.state,
        clock=clock,
    )

    return RestoredGameSession(
        context=context,
        document=document,
    )
