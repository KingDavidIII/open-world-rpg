"""Application service for saving and loading game sessions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from open_world_rpg.application.session import RuntimeContext
from open_world_rpg.gameplay import (
    DroppedItem,
    DroppedItemManager,
    DroppedItemSnapshot,
    ItemStack,
    ItemType,
    PlayerInventory,
    PlayerInventorySnapshot,
)
from open_world_rpg.persistence.document import (
    JsonValue,
    PersistedBlockEditOverlay,
    SaveDocument,
)
from open_world_rpg.persistence.repository import SaveRepository
from open_world_rpg.persistence.storage import SaveSlot
from open_world_rpg.world import BlockEditStore, BlockEditStoreSnapshot


class BlockEditRestoreError(RuntimeError):
    """Raised when a save cannot be applied to the current voxel world."""


class ResourceStateRestoreError(RuntimeError):
    """Raised when persisted inventory or world items are malformed."""


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
        block_edits: BlockEditStoreSnapshot | None = None,
        inventory: PlayerInventorySnapshot | None = None,
        dropped_items: DroppedItemSnapshot | None = None,
    ) -> Path:
        """Create and persist a document for the current session."""
        if not isinstance(slot, SaveSlot):
            raise TypeError("slot must be a SaveSlot.")

        document: SaveDocument | None = None

        try:
            resource_payload = {} if payload is None else dict(payload)
            if inventory is not None:
                resource_payload["inventory"] = self._inventory_payload(inventory)
            if dropped_items is not None:
                resource_payload["dropped_items"] = self._drops_payload(dropped_items)
            document = SaveDocument.from_runtime_context(
                context=self.context,
                payload=resource_payload,
                saved_at=saved_at,
                block_edits=(
                    None
                    if block_edits is None
                    else PersistedBlockEditOverlay.from_snapshot(block_edits)
                ),
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

    def restore_block_edits(
        self,
        document: SaveDocument,
        *,
        expected_world_id: UUID,
        expected_world_seed: int,
    ) -> BlockEditStore:
        """Validate identity and atomically construct the saved edit overlay."""
        if not isinstance(document, SaveDocument):
            raise TypeError("document must be a SaveDocument.")
        if not isinstance(expected_world_id, UUID):
            raise TypeError("expected_world_id must be a UUID.")
        if isinstance(expected_world_seed, bool) or not isinstance(expected_world_seed, int):
            raise TypeError("expected_world_seed must be an integer.")
        if document.session.session_id != expected_world_id:
            raise BlockEditRestoreError("Saved world identity does not match the current world.")
        if document.session.world_seed != expected_world_seed:
            raise BlockEditRestoreError("Saved world seed does not match the current world.")
        overlay = document.block_edits
        return (
            BlockEditStore()
            if overlay is None
            else BlockEditStore.from_snapshot(overlay.to_snapshot())
        )

    def restore_resources(
        self,
        document: SaveDocument,
        *,
        expected_world_id: UUID,
        expected_world_seed: int,
        legacy_inventory: PlayerInventorySnapshot,
    ) -> tuple[PlayerInventory, DroppedItemManager]:
        """Atomically construct gameplay resources after world-scope validation."""
        self.restore_block_edits(
            document,
            expected_world_id=expected_world_id,
            expected_world_seed=expected_world_seed,
        )
        if not isinstance(legacy_inventory, PlayerInventorySnapshot):
            raise TypeError("legacy_inventory must be a PlayerInventorySnapshot.")
        try:
            inventory_value = document.payload.get("inventory")
            drops_value = document.payload.get("dropped_items")
            inventory = (
                PlayerInventory.from_snapshot(legacy_inventory)
                if inventory_value is None
                else PlayerInventory.from_snapshot(self._parse_inventory(inventory_value))
            )
            drops = (
                DroppedItemManager()
                if drops_value is None
                else DroppedItemManager.from_snapshot(self._parse_drops(drops_value))
            )
            return inventory, drops
        except (TypeError, ValueError, KeyError) as exc:
            raise ResourceStateRestoreError("Saved gameplay resource state is invalid.") from exc

    @staticmethod
    def _inventory_payload(snapshot: PlayerInventorySnapshot) -> dict[str, JsonValue]:
        if not isinstance(snapshot, PlayerInventorySnapshot):
            raise TypeError("inventory must be a PlayerInventorySnapshot.")
        return {
            "revision": snapshot.revision,
            "selected_hotbar_index": snapshot.selected_hotbar_index,
            "slots": [
                None if stack is None else {"item": stack.item.value, "quantity": stack.quantity}
                for stack in snapshot.slots
            ],
        }

    @staticmethod
    def _drops_payload(snapshot: DroppedItemSnapshot) -> dict[str, JsonValue]:
        if not isinstance(snapshot, DroppedItemSnapshot):
            raise TypeError("dropped_items must be a DroppedItemSnapshot.")
        return {
            "revision": snapshot.revision,
            "next_identifier": snapshot.next_identifier,
            "items": [
                {
                    "identifier": item.identifier,
                    "item": item.item.value,
                    "quantity": item.quantity,
                    "position": list(item.position),
                    "velocity": list(item.velocity),
                    "age": item.age,
                    "pickup_delay": item.pickup_delay,
                    "settled": item.settled,
                }
                for item in snapshot.items
            ],
        }

    @staticmethod
    def _parse_inventory(value: JsonValue) -> PlayerInventorySnapshot:
        if not isinstance(value, dict) or set(value) != {
            "revision",
            "selected_hotbar_index",
            "slots",
        }:
            raise ValueError("inventory must contain canonical fields.")
        data = cast(dict[str, Any], value)
        raw_slots = data["slots"]
        if not isinstance(raw_slots, list):
            raise TypeError("inventory slots must be a list.")
        slots: list[ItemStack | None] = []
        for raw_stack in raw_slots:
            if raw_stack is None:
                slots.append(None)
                continue
            if not isinstance(raw_stack, dict) or set(raw_stack) != {"item", "quantity"}:
                raise ValueError("inventory stack must contain item and quantity.")
            slots.append(
                ItemStack(
                    item=ItemType(raw_stack["item"]),
                    quantity=raw_stack["quantity"],
                )
            )
        return PlayerInventorySnapshot(
            revision=data["revision"],
            selected_hotbar_index=data["selected_hotbar_index"],
            slots=tuple(slots),
        )

    @staticmethod
    def _parse_drops(value: JsonValue) -> DroppedItemSnapshot:
        if not isinstance(value, dict) or set(value) != {
            "revision",
            "next_identifier",
            "items",
        }:
            raise ValueError("dropped_items must contain canonical fields.")
        data = cast(dict[str, Any], value)
        raw_items = data["items"]
        if not isinstance(raw_items, list):
            raise TypeError("dropped_items items must be a list.")
        expected = {
            "identifier",
            "item",
            "quantity",
            "position",
            "velocity",
            "age",
            "pickup_delay",
            "settled",
        }
        items: list[DroppedItem] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict) or set(raw_item) != expected:
                raise ValueError("dropped item must contain canonical fields.")
            position = raw_item["position"]
            velocity = raw_item["velocity"]
            if not isinstance(position, list) or not isinstance(velocity, list):
                raise TypeError("dropped item vectors must be lists.")
            items.append(
                DroppedItem(
                    identifier=raw_item["identifier"],
                    item=ItemType(raw_item["item"]),
                    quantity=raw_item["quantity"],
                    position=tuple(position),
                    velocity=tuple(velocity),
                    age=raw_item["age"],
                    pickup_delay=raw_item["pickup_delay"],
                    settled=raw_item["settled"],
                )
            )
        return DroppedItemSnapshot(
            revision=data["revision"],
            next_identifier=data["next_identifier"],
            items=tuple(items),
        )

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
