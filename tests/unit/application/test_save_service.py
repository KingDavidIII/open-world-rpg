"""Tests for the application save-game service."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, NoReturn, cast
from uuid import UUID

import pytest

from open_world_rpg.application.save_service import GameSaveService
from open_world_rpg.application.session import (
    GameMode,
    RuntimeContext,
    SessionState,
)
from open_world_rpg.core import JsonLogFormatter, ProjectPaths
from open_world_rpg.persistence.document import (
    CURRENT_SAVE_SCHEMA_VERSION,
    SaveDocument,
    SessionSnapshot,
)
from open_world_rpg.persistence.repository import (
    SaveRepository,
    SaveSlotNotFoundError,
    SaveWriteError,
)
from open_world_rpg.persistence.storage import RuntimeStorage, SaveSlot

SESSION_ID = UUID("12345678-1234-5678-1234-567812345678")
CREATED_AT = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
SAVED_AT = datetime(2026, 7, 25, 10, 5, tzinfo=UTC)


def create_logger(stream: StringIO) -> logging.Logger:
    logger = logging.Logger(
        "test.open_world_rpg.save_service",
        level=logging.DEBUG,
    )
    logger.propagate = False

    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)

    return logger


def create_active_context() -> RuntimeContext:
    timestamps = iter(
        [
            CREATED_AT,
            CREATED_AT + timedelta(seconds=1),
        ]
    )
    context = RuntimeContext.create(
        game_mode=GameMode.NEW_GAME,
        world_seed=42,
        clock=lambda: next(timestamps),
        session_id=SESSION_ID,
    )
    context.start()
    return context


def create_repository(tmp_path: Path) -> SaveRepository:
    return SaveRepository(storage=RuntimeStorage(paths=ProjectPaths.from_project_root(tmp_path)))


def create_service(
    tmp_path: Path,
    *,
    stream: StringIO | None = None,
    context: RuntimeContext | None = None,
) -> GameSaveService:
    resolved_stream = StringIO() if stream is None else stream

    return GameSaveService(
        repository=create_repository(tmp_path),
        context=create_active_context() if context is None else context,
        logger=create_logger(resolved_stream),
    )


def read_payloads(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def create_document() -> SaveDocument:
    return SaveDocument(
        schema_version=CURRENT_SAVE_SCHEMA_VERSION,
        saved_at=SAVED_AT,
        session=SessionSnapshot(
            session_id=SESSION_ID,
            game_mode=GameMode.NEW_GAME,
            world_seed=42,
            state=SessionState.ACTIVE,
        ),
        payload={"player": {"level": 7}},
    )


def test_service_rejects_invalid_dependencies(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    context = create_active_context()
    logger = logging.Logger("test")

    with pytest.raises(TypeError, match="repository"):
        GameSaveService(
            repository=cast(Any, object()),
            context=context,
            logger=logger,
        )

    with pytest.raises(TypeError, match="context"):
        GameSaveService(
            repository=repository,
            context=cast(Any, object()),
            logger=logger,
        )

    with pytest.raises(TypeError, match="logger"):
        GameSaveService(
            repository=repository,
            context=context,
            logger=cast(Any, object()),
        )


def test_save_creates_document_for_current_session(
    tmp_path: Path,
) -> None:
    stream = StringIO()
    service = create_service(tmp_path, stream=stream)
    slot = SaveSlot("campaign-01")

    path = service.save(
        slot=slot,
        saved_at=SAVED_AT,
        payload={
            "player": {
                "name": "Ọlá",
                "level": 7,
            }
        },
    )

    document = service.repository.load(slot)
    payload = read_payloads(stream)[-1]

    assert path == service.repository.storage.save_path(slot)
    assert document.saved_at == SAVED_AT
    assert document.session.session_id == SESSION_ID
    assert document.session.state is SessionState.ACTIVE
    assert document.payload["player"] == {
        "name": "Ọlá",
        "level": 7,
    }

    assert payload["event"] == "persistence.save_succeeded"
    assert payload["save_slot"] == "campaign-01"
    assert payload["schema_version"] == CURRENT_SAVE_SCHEMA_VERSION
    assert payload["saved_session_id"] == str(SESSION_ID)
    assert payload["saved_session_state"] == "active"
    assert payload["session_state"] == "active"


def test_save_uses_empty_payload_by_default(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    slot = SaveSlot("autosave")

    service.save(
        slot=slot,
        saved_at=SAVED_AT,
    )

    assert service.repository.load(slot).payload == {}


def test_load_returns_document_and_logs_metadata(
    tmp_path: Path,
) -> None:
    stream = StringIO()
    service = create_service(tmp_path, stream=stream)
    slot = SaveSlot("campaign-01")
    expected = create_document()
    service.repository.save(slot=slot, document=expected)

    loaded = service.load(slot)
    payload = read_payloads(stream)[-1]

    assert loaded == expected
    assert payload["event"] == "persistence.load_succeeded"
    assert payload["save_slot"] == "campaign-01"
    assert payload["schema_version"] == CURRENT_SAVE_SCHEMA_VERSION
    assert payload["saved_session_id"] == str(SESSION_ID)
    assert payload["saved_session_state"] == "active"


def test_save_rejects_invalid_slot(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    with pytest.raises(TypeError, match="slot"):
        service.save(slot=cast(Any, "campaign-01"))


def test_load_rejects_invalid_slot(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    with pytest.raises(TypeError, match="slot"):
        service.load(cast(Any, "campaign-01"))


def test_non_resumable_session_save_is_logged(
    tmp_path: Path,
) -> None:
    stream = StringIO()
    context = RuntimeContext.create(
        game_mode=GameMode.NEW_GAME,
        world_seed=42,
        clock=lambda: CREATED_AT,
        session_id=SESSION_ID,
    )
    service = create_service(
        tmp_path,
        stream=stream,
        context=context,
    )

    with pytest.raises(ValueError, match="active or paused"):
        service.save(
            slot=SaveSlot("campaign-01"),
            saved_at=SAVED_AT,
        )

    payload = read_payloads(stream)[-1]

    assert payload["event"] == "persistence.save_failed"
    assert payload["save_slot"] == "campaign-01"
    assert payload["session_state"] == "created"
    assert "schema_version" not in payload
    assert "ValueError" in payload["exception"]


def test_repository_save_failure_is_logged_and_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stream = StringIO()
    service = create_service(tmp_path, stream=stream)

    def raise_save_error(
        self: SaveRepository,
        *,
        slot: SaveSlot,
        document: SaveDocument,
    ) -> NoReturn:
        del self, slot, document
        raise SaveWriteError("disk unavailable")

    monkeypatch.setattr(SaveRepository, "save", raise_save_error)

    with pytest.raises(SaveWriteError, match="disk unavailable"):
        service.save(
            slot=SaveSlot("campaign-01"),
            saved_at=SAVED_AT,
        )

    payload = read_payloads(stream)[-1]

    assert payload["event"] == "persistence.save_failed"
    assert payload["schema_version"] == CURRENT_SAVE_SCHEMA_VERSION
    assert payload["saved_session_id"] == str(SESSION_ID)
    assert "SaveWriteError: disk unavailable" in payload["exception"]


def test_missing_slot_load_is_logged_and_preserved(
    tmp_path: Path,
) -> None:
    stream = StringIO()
    service = create_service(tmp_path, stream=stream)

    with pytest.raises(
        SaveSlotNotFoundError,
        match="'missing' does not exist",
    ):
        service.load(SaveSlot("missing"))

    payload = read_payloads(stream)[-1]

    assert payload["event"] == "persistence.load_failed"
    assert payload["save_slot"] == "missing"
    assert "schema_version" not in payload
    assert "SaveSlotNotFoundError" in payload["exception"]
