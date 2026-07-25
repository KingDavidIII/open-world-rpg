"""Tests for strict save-document JSON serialisation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import pytest

from open_world_rpg.application.session import GameMode, SessionState
from open_world_rpg.persistence.document import (
    CURRENT_SAVE_SCHEMA_VERSION,
    SaveCompatibilityError,
    SaveCorruptionError,
    SaveDocument,
    SessionSnapshot,
)

SESSION_ID = UUID("12345678-1234-5678-1234-567812345678")
SAVED_AT = datetime(2026, 7, 25, 10, 5, tzinfo=UTC)


def create_document() -> SaveDocument:
    return SaveDocument(
        schema_version=CURRENT_SAVE_SCHEMA_VERSION,
        saved_at=SAVED_AT,
        session=SessionSnapshot(
            session_id=SESSION_ID,
            game_mode=GameMode.NEW_GAME,
            world_seed=42,
            state=SessionState.PAUSED,
        ),
        payload={
            "player": {
                "name": "Ọlá",
                "alive": True,
                "health": 87.5,
                "companion": None,
                "inventory": [
                    "iron-sword",
                    "healing-potion",
                ],
            }
        },
    )


def create_raw_document() -> dict[str, object]:
    return {
        "schema_version": CURRENT_SAVE_SCHEMA_VERSION,
        "saved_at": SAVED_AT.isoformat(),
        "session": {
            "session_id": str(SESSION_ID),
            "game_mode": "new_game",
            "world_seed": 42,
            "state": "paused",
        },
        "payload": {
            "player": {
                "name": "Ọlá",
            }
        },
    }


def test_to_json_is_deterministic_and_unicode_safe() -> None:
    document = create_document()

    first = document.to_json()
    second = document.to_json()

    assert first == second
    assert first.endswith("\n")
    assert "Ọlá" in first
    assert '"schema_version": 1' in first


def test_json_round_trip_preserves_document() -> None:
    document = create_document()

    restored = SaveDocument.from_json(document.to_json())

    assert restored == document


def test_from_json_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="text"):
        SaveDocument.from_json(cast(Any, b"{}"))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{",
        '{"value": NaN}',
        '{"value": Infinity}',
    ],
)
def test_from_json_rejects_malformed_json(text: str) -> None:
    with pytest.raises(
        SaveCorruptionError,
        match="not valid JSON",
    ):
        SaveDocument.from_json(text)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "document",
        42,
    ],
)
def test_from_json_requires_object_root(value: object) -> None:
    with pytest.raises(
        SaveCorruptionError,
        match="root must be a JSON object",
    ):
        SaveDocument.from_json(json.dumps(value))


def test_from_json_rejects_missing_document_field() -> None:
    raw = create_raw_document()
    del raw["payload"]

    with pytest.raises(
        SaveCorruptionError,
        match="missing required fields: payload",
    ):
        SaveDocument.from_json(json.dumps(raw))


def test_from_json_rejects_unexpected_document_field() -> None:
    raw = create_raw_document()
    raw["unexpected"] = True

    with pytest.raises(
        SaveCorruptionError,
        match="unexpected fields: unexpected",
    ):
        SaveDocument.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    "version",
    [
        "1",
        True,
    ],
)
def test_from_json_rejects_invalid_schema_type(
    version: object,
) -> None:
    raw = create_raw_document()
    raw["schema_version"] = version

    with pytest.raises(
        SaveCorruptionError,
        match="schema_version must be an integer",
    ):
        SaveDocument.from_json(json.dumps(raw))


def test_from_json_rejects_unsupported_schema() -> None:
    raw = create_raw_document()
    raw["schema_version"] = 99

    with pytest.raises(
        SaveCompatibilityError,
        match="Unsupported save schema version 99",
    ):
        SaveDocument.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (100, "ISO-8601 string"),
        ("not-a-date", "timezone-aware ISO-8601"),
        ("2026-07-25T10:00:00", "timezone-aware ISO-8601"),
    ],
)
def test_from_json_rejects_invalid_saved_at(
    value: object,
    message: str,
) -> None:
    raw = create_raw_document()
    raw["saved_at"] = value

    with pytest.raises(SaveCorruptionError, match=message):
        SaveDocument.from_json(json.dumps(raw))


def test_from_json_requires_session_object() -> None:
    raw = create_raw_document()
    raw["session"] = "invalid"

    with pytest.raises(
        SaveCorruptionError,
        match="session must be a JSON object",
    ):
        SaveDocument.from_json(json.dumps(raw))


def test_from_json_rejects_missing_session_field() -> None:
    raw = create_raw_document()
    session = cast(dict[str, object], raw["session"])
    del session["world_seed"]

    with pytest.raises(
        SaveCorruptionError,
        match="missing required fields: world_seed",
    ):
        SaveDocument.from_json(json.dumps(raw))


def test_from_json_rejects_unexpected_session_field() -> None:
    raw = create_raw_document()
    session = cast(dict[str, object], raw["session"])
    session["unexpected"] = True

    with pytest.raises(
        SaveCorruptionError,
        match="unexpected fields: unexpected",
    ):
        SaveDocument.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("session_id", 100, "UUID string"),
        ("session_id", "invalid", "valid UUID"),
        ("game_mode", 100, "game_mode must be a string"),
        ("game_mode", "unknown", "Unsupported game mode"),
        ("state", 100, "state must be a string"),
        ("state", "unknown", "Unsupported session state"),
        ("state", "terminated", "Session metadata is invalid"),
        ("world_seed", "42", "Session metadata is invalid"),
        ("world_seed", -1, "Session metadata is invalid"),
    ],
)
def test_from_json_rejects_invalid_session_metadata(
    field: str,
    value: object,
    message: str,
) -> None:
    raw = create_raw_document()
    session = cast(dict[str, object], raw["session"])
    session[field] = value

    with pytest.raises(SaveCorruptionError, match=message):
        SaveDocument.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    "payload",
    [
        [],
        "invalid",
        42,
        None,
    ],
)
def test_from_json_requires_object_payload(
    payload: object,
) -> None:
    raw = create_raw_document()
    raw["payload"] = payload

    with pytest.raises(
        SaveCorruptionError,
        match="Save payload is invalid",
    ):
        SaveDocument.from_json(json.dumps(raw))
