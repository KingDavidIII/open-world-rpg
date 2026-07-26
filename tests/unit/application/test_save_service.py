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

from open_world_rpg.application.save_service import GameSaveService, ResourceStateRestoreError
from open_world_rpg.application.session import (
    GameMode,
    RuntimeContext,
    SessionState,
)
from open_world_rpg.core import JsonLogFormatter, ProjectPaths
from open_world_rpg.gameplay import (
    DroppedItemManager,
    ItemType,
    PlayerVitals,
    create_bootstrap_inventory,
)
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


def test_gameplay_resources_round_trip_and_legacy_policy(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    inventory = create_bootstrap_inventory()
    inventory.select_hotbar(2)
    drops = DroppedItemManager()
    drops.spawn(item=ItemType.GRASS_BLOCK, quantity=1, position=(-1.5, 2.5, 3.5))
    service.save(
        slot=SaveSlot("resources"),
        inventory=inventory.snapshot(),
        dropped_items=drops.snapshot(),
    )
    document = service.load(SaveSlot("resources"))
    restored_inventory, restored_drops = service.restore_resources(
        document,
        expected_world_id=SESSION_ID,
        expected_world_seed=42,
        legacy_inventory=create_bootstrap_inventory(enabled=False).snapshot(),
    )
    assert restored_inventory.snapshot() == inventory.snapshot()
    assert restored_drops.snapshot() == drops.snapshot()

    legacy = SaveDocument.from_runtime_context(context=service.context)
    legacy_inventory, legacy_drops = service.restore_resources(
        legacy,
        expected_world_id=SESSION_ID,
        expected_world_seed=42,
        legacy_inventory=inventory.snapshot(),
    )
    assert legacy_inventory.snapshot() == inventory.snapshot()
    assert len(legacy_drops) == 0


def test_survival_vitals_round_trip_defaults_and_validation(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    vitals = PlayerVitals()
    vitals.update_stamina(1_000_000, sprinting=True)
    vitals.damage(7)
    service.save(slot=SaveSlot("vitals"), vitals=vitals.snapshot)
    document = service.load(SaveSlot("vitals"))
    restored = service.restore_vitals(
        document,
        expected_world_id=SESSION_ID,
        expected_world_seed=42,
    ).snapshot
    assert (restored.health, restored.stamina, restored.death_count, restored.revision) == (
        vitals.snapshot.health,
        vitals.snapshot.stamina,
        vitals.snapshot.death_count,
        vitals.snapshot.revision,
    )
    assert restored.regeneration_delay_microseconds == 0
    legacy = SaveDocument.from_runtime_context(context=service.context)
    assert (
        service.restore_vitals(
            legacy,
            expected_world_id=SESSION_ID,
            expected_world_seed=42,
        ).snapshot.health
        == 100
    )
    malformed = SaveDocument.from_runtime_context(
        context=service.context,
        payload={"vitals": {"health_milli": -1}},
    )
    with pytest.raises(ResourceStateRestoreError):
        service.restore_vitals(
            malformed,
            expected_world_id=SESSION_ID,
            expected_world_seed=42,
        )
    with pytest.raises(TypeError):
        service._vitals_payload("bad")  # type: ignore[arg-type]
    invalid_slot = SaveDocument.from_runtime_context(
        context=service.context,
        payload={
            "inventory": {
                "revision": 0,
                "selected_hotbar_index": 0,
                "slots": [{"kind": "unknown"}] + [None] * 26,
            }
        },
    )
    with pytest.raises(ResourceStateRestoreError):
        service.restore_resources(
            invalid_slot,
            expected_world_id=SESSION_ID,
            expected_world_seed=42,
            legacy_inventory=create_bootstrap_inventory().snapshot(),
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"inventory": []},
        {"inventory": {"revision": 0, "selected_hotbar_index": 0, "slots": []}},
        {
            "dropped_items": {
                "revision": 0,
                "next_identifier": 1,
                "items": [{"identifier": 1}],
            }
        },
    ],
)
def test_malformed_gameplay_resources_fail_atomically(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    service = create_service(tmp_path)
    document = SaveDocument.from_runtime_context(
        context=service.context,
        payload=cast(Any, payload),
    )
    with pytest.raises(ResourceStateRestoreError):
        service.restore_resources(
            document,
            expected_world_id=SESSION_ID,
            expected_world_seed=42,
            legacy_inventory=create_bootstrap_inventory().snapshot(),
        )


def test_gameplay_resource_codec_validation_branches(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    with pytest.raises(TypeError):
        service._inventory_payload("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        service._drops_payload("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        service.restore_resources(
            SaveDocument.from_runtime_context(context=service.context),
            expected_world_id=SESSION_ID,
            expected_world_seed=42,
            legacy_inventory="bad",  # type: ignore[arg-type]
        )
    invalid_values: tuple[object, ...] = (
        {"revision": 0, "selected_hotbar_index": 0, "slots": "bad"},
        {"revision": 0, "selected_hotbar_index": 0, "slots": ["bad"]},
    )
    for value in invalid_values:
        with pytest.raises((TypeError, ValueError)):
            service._parse_inventory(cast(Any, value))
    with pytest.raises(ValueError):
        service._parse_drops(cast(Any, []))
    with pytest.raises(TypeError):
        service._parse_drops(cast(Any, {"revision": 0, "next_identifier": 1, "items": "bad"}))
    complete = {
        "identifier": 1,
        "item": "stone_block",
        "quantity": 1,
        "position": "bad",
        "velocity": [0, 0, 0],
        "age": 0,
        "pickup_delay": 0.3,
        "settled": False,
    }
    with pytest.raises(TypeError):
        service._parse_drops(
            cast(
                Any,
                {"revision": 0, "next_identifier": 2, "items": [complete]},
            )
        )


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
