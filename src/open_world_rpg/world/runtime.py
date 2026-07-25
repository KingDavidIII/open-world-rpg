"""Controlled mutable runtime boundary for immutable world models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from open_world_rpg.engine.events import EventBus
from open_world_rpg.world.metadata import WorldId, WorldState
from open_world_rpg.world.model import WorldModel, WorldSnapshot
from open_world_rpg.world.time import WorldInstant


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldStateChanged:
    """Event published after a successful world lifecycle transition."""

    world_id: WorldId
    previous_state: WorldState
    current_state: WorldState
    revision: int


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldTimeAdvanced:
    """Event published after a successful positive world-time advance."""

    world_id: WorldId
    previous_tick: int
    current_tick: int
    advanced_ticks: int
    revision: int


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldRuntimeSnapshot:
    """Immutable revisioned projection of the current world."""

    revision: int
    world: WorldSnapshot


_ModelOperation = Callable[[WorldModel], WorldModel]


class WorldRuntime:
    """Own and replace the current immutable world model."""

    __slots__ = ("_event_bus", "_model", "_revision")

    def __init__(
        self,
        *,
        model: WorldModel,
        event_bus: EventBus | None = None,
    ) -> None:
        if not isinstance(model, WorldModel):
            raise TypeError("model must be a WorldModel.")

        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None.")

        self._model = model
        self._event_bus = event_bus
        self._revision = 0

    @property
    def model(self) -> WorldModel:
        """Return the current immutable world model."""
        return self._model

    @property
    def revision(self) -> int:
        """Return the successful state-change revision."""
        return self._revision

    @property
    def snapshot(self) -> WorldRuntimeSnapshot:
        """Return an immutable snapshot of revision and world state."""
        return WorldRuntimeSnapshot(
            revision=self._revision,
            world=self._model.snapshot(),
        )

    def initialise(self) -> None:
        """Initialise the current world."""
        self._apply_state_transition(WorldModel.initialise)

    def activate(self) -> None:
        """Activate the current world."""
        self._apply_state_transition(WorldModel.activate)

    def pause(self) -> None:
        """Pause the current world."""
        self._apply_state_transition(WorldModel.pause)

    def resume(self) -> None:
        """Resume the current world."""
        self._apply_state_transition(WorldModel.resume)

    def close(self) -> None:
        """Close the current world."""
        self._apply_state_transition(WorldModel.close)

    def fail(self) -> None:
        """Fail the current world."""
        self._apply_state_transition(WorldModel.fail)

    def advance_tick(self) -> None:
        """Advance the active world by exactly one tick."""
        self.advance_ticks(1)

    def advance_ticks(self, ticks: int) -> None:
        """Advance the active world by a validated tick count."""
        if isinstance(ticks, bool) or not isinstance(ticks, int):
            raise TypeError("ticks must be an integer.")

        if ticks < 0:
            raise ValueError("ticks must be greater than or equal to zero.")

        previous = self._model
        candidate = previous.advance_ticks(ticks)
        if candidate == previous:
            return

        self._model = candidate
        self._revision += 1
        self._publish(
            WorldTimeAdvanced(
                world_id=candidate.metadata.world_id,
                previous_tick=previous.clock.current.tick,
                current_tick=candidate.clock.current.tick,
                advanced_ticks=ticks,
                revision=self._revision,
            )
        )

    def reset_clock(
        self,
        *,
        instant: WorldInstant | None = None,
    ) -> None:
        """Reset world time through the aggregate's lifecycle policy."""
        previous = self._model
        candidate = previous.reset_clock(instant=instant)
        if candidate == previous:
            return

        self._model = candidate
        self._revision += 1

    def _apply_state_transition(self, operation: _ModelOperation) -> None:
        previous = self._model
        candidate = operation(previous)
        self._model = candidate
        self._revision += 1
        self._publish(
            WorldStateChanged(
                world_id=candidate.metadata.world_id,
                previous_state=previous.metadata.state,
                current_state=candidate.metadata.state,
                revision=self._revision,
            )
        )

    def _publish(self, event: object) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(event)
