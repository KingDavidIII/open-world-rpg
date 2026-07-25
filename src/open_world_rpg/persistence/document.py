"""Versioned save-game documents and deterministic JSON serialisation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final, NoReturn, TypeAlias
from uuid import UUID

from open_world_rpg.application.session import (
    GameMode,
    RuntimeContext,
    SessionState,
)
from open_world_rpg.core.config import MAX_WORLD_SEED, MIN_WORLD_SEED

CURRENT_SAVE_SCHEMA_VERSION: Final = 1

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None

_RESUMABLE_SESSION_STATES: Final = frozenset(
    {
        SessionState.ACTIVE,
        SessionState.PAUSED,
    }
)

_DOCUMENT_KEYS: Final = frozenset(
    {
        "schema_version",
        "saved_at",
        "session",
        "payload",
    }
)

_SESSION_KEYS: Final = frozenset(
    {
        "session_id",
        "game_mode",
        "world_seed",
        "state",
    }
)


class SaveDocumentError(RuntimeError):
    """Base exception for save-document failures."""


class SaveCorruptionError(SaveDocumentError):
    """Raised when save content is malformed or semantically invalid."""


class SaveCompatibilityError(SaveDocumentError):
    """Raised when a save uses an unsupported schema version."""


def _normalise_timestamp(*, name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")

    return value.astimezone(UTC)


def _validate_world_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("world_seed must be an integer.")

    if value < MIN_WORLD_SEED or value > MAX_WORLD_SEED:
        raise ValueError(f"world_seed must be between {MIN_WORLD_SEED} and {MAX_WORLD_SEED}.")

    return value


def _normalise_json_value(
    value: object,
    *,
    path: str,
) -> JsonValue:
    if value is None or isinstance(value, str | bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers.")

        return value

    if isinstance(value, list):
        return [
            _normalise_json_value(
                item,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]

    if isinstance(value, dict):
        normalised: dict[str, JsonValue] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings.")

            normalised[key] = _normalise_json_value(
                item,
                path=f"{path}.{key}",
            )

        return normalised

    raise TypeError(f"{path} contains an unsupported value of type {type(value).__name__!r}.")


def _normalise_payload(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError("payload must be a dictionary.")

    normalised = _normalise_json_value(value, path="payload")

    if not isinstance(normalised, dict):  # pragma: no cover
        raise TypeError("payload must be a dictionary.")

    return normalised


def _require_exact_keys(
    mapping: dict[str, object],
    *,
    expected: frozenset[str],
    location: str,
) -> None:
    actual = frozenset(mapping)

    missing = sorted(expected - actual)
    if missing:
        raise SaveCorruptionError(f"{location} is missing required fields: {', '.join(missing)}.")

    unexpected = sorted(actual - expected)
    if unexpected:
        raise SaveCorruptionError(
            f"{location} contains unexpected fields: {', '.join(unexpected)}."
        )


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"Invalid JSON numeric constant: {value}")


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionSnapshot:
    """Resumable session metadata stored inside a save document."""

    session_id: UUID
    game_mode: GameMode
    world_seed: int
    state: SessionState

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, UUID):
            raise TypeError("session_id must be a UUID.")

        if not isinstance(self.game_mode, GameMode):
            raise TypeError("game_mode must be a GameMode.")

        object.__setattr__(
            self,
            "world_seed",
            _validate_world_seed(self.world_seed),
        )

        if not isinstance(self.state, SessionState):
            raise TypeError("state must be a SessionState.")

        if self.state not in _RESUMABLE_SESSION_STATES:
            raise ValueError("state must represent an active or paused session.")

    @classmethod
    def from_runtime_context(
        cls,
        context: RuntimeContext,
    ) -> SessionSnapshot:
        """Capture resumable metadata from a runtime context."""
        if not isinstance(context, RuntimeContext):
            raise TypeError("context must be a RuntimeContext.")

        return cls(
            session_id=context.session_id,
            game_mode=context.game_mode,
            world_seed=context.world_seed,
            state=context.state,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SaveDocument:
    """Strict, versioned envelope for one save-game payload."""

    schema_version: int
    saved_at: datetime
    session: SessionSnapshot
    payload: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer.")

        if self.schema_version != CURRENT_SAVE_SCHEMA_VERSION:
            raise SaveCompatibilityError(
                f"Unsupported save schema version "
                f"{self.schema_version}; expected "
                f"{CURRENT_SAVE_SCHEMA_VERSION}."
            )

        object.__setattr__(
            self,
            "saved_at",
            _normalise_timestamp(
                name="saved_at",
                value=self.saved_at,
            ),
        )

        if not isinstance(self.session, SessionSnapshot):
            raise TypeError("session must be a SessionSnapshot.")

        object.__setattr__(
            self,
            "payload",
            _normalise_payload(self.payload),
        )

    @classmethod
    def from_runtime_context(
        cls,
        *,
        context: RuntimeContext,
        payload: dict[str, JsonValue] | None = None,
        saved_at: datetime | None = None,
    ) -> SaveDocument:
        """Create a document from a resumable runtime context."""
        if not isinstance(context, RuntimeContext):
            raise TypeError("context must be a RuntimeContext.")

        timestamp = datetime.now(UTC) if saved_at is None else saved_at

        return cls(
            schema_version=CURRENT_SAVE_SCHEMA_VERSION,
            saved_at=timestamp,
            session=SessionSnapshot.from_runtime_context(context),
            payload={} if payload is None else payload,
        )

    def to_json(self) -> str:
        """Serialise the document into canonical UTF-8 JSON text."""
        document = {
            "schema_version": self.schema_version,
            "saved_at": self.saved_at.isoformat(),
            "session": {
                "session_id": str(self.session.session_id),
                "game_mode": self.session.game_mode.value,
                "world_seed": self.session.world_seed,
                "state": self.session.state.value,
            },
            "payload": self.payload,
        }

        return (
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, text: str) -> SaveDocument:
        """Parse and strictly validate a save document."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        try:
            raw_document = json.loads(
                text,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise SaveCorruptionError("Save document is not valid JSON.") from exc

        if not isinstance(raw_document, dict):
            raise SaveCorruptionError("Save document root must be a JSON object.")

        document = dict(raw_document)
        _require_exact_keys(
            document,
            expected=_DOCUMENT_KEYS,
            location="Save document",
        )

        schema_version = document["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise SaveCorruptionError("schema_version must be an integer.")

        if schema_version != CURRENT_SAVE_SCHEMA_VERSION:
            raise SaveCompatibilityError(
                f"Unsupported save schema version "
                f"{schema_version}; expected "
                f"{CURRENT_SAVE_SCHEMA_VERSION}."
            )

        saved_at = cls._parse_timestamp(document["saved_at"])
        session = cls._parse_session(document["session"])

        try:
            payload = _normalise_payload(document["payload"])
        except (TypeError, ValueError) as exc:
            raise SaveCorruptionError("Save payload is invalid.") from exc

        return cls(
            schema_version=schema_version,
            saved_at=saved_at,
            session=session,
            payload=payload,
        )

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        if not isinstance(value, str):
            raise SaveCorruptionError("saved_at must be an ISO-8601 string.")

        try:
            timestamp = datetime.fromisoformat(value)
            return _normalise_timestamp(
                name="saved_at",
                value=timestamp,
            )
        except ValueError as exc:
            raise SaveCorruptionError(
                "saved_at must be a timezone-aware ISO-8601 timestamp."
            ) from exc

    @staticmethod
    def _parse_session(value: object) -> SessionSnapshot:
        if not isinstance(value, dict):
            raise SaveCorruptionError("session must be a JSON object.")

        session_data = dict(value)
        _require_exact_keys(
            session_data,
            expected=_SESSION_KEYS,
            location="Session metadata",
        )

        session_id_value = session_data["session_id"]
        game_mode_value = session_data["game_mode"]
        world_seed = session_data["world_seed"]
        state_value = session_data["state"]

        if not isinstance(session_id_value, str):
            raise SaveCorruptionError("session_id must be a UUID string.")

        if not isinstance(game_mode_value, str):
            raise SaveCorruptionError("game_mode must be a string.")

        if not isinstance(state_value, str):
            raise SaveCorruptionError("state must be a string.")

        try:
            session_id = UUID(session_id_value)
        except ValueError as exc:
            raise SaveCorruptionError("session_id must be a valid UUID.") from exc

        try:
            game_mode = GameMode(game_mode_value)
        except ValueError as exc:
            raise SaveCorruptionError(f"Unsupported game mode: {game_mode_value!r}.") from exc

        try:
            state = SessionState(state_value)
        except ValueError as exc:
            raise SaveCorruptionError(f"Unsupported session state: {state_value!r}.") from exc

        try:
            return SessionSnapshot(
                session_id=session_id,
                game_mode=game_mode,
                world_seed=world_seed,
                state=state,
            )
        except (TypeError, ValueError) as exc:
            raise SaveCorruptionError("Session metadata is invalid.") from exc
