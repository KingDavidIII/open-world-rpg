"""Engine subsystem coordinating deterministic world lifecycle and time."""

from __future__ import annotations

import logging
import math
from typing import cast

from open_world_rpg.core import GameConfig
from open_world_rpg.engine.dependencies import EngineSubsystemBase
from open_world_rpg.engine.services import EngineContext
from open_world_rpg.world.metadata import WorldState
from open_world_rpg.world.runtime import WorldRuntime


class WorldSubsystemError(RuntimeError):
    """Base error for world subsystem lifecycle and configuration failures."""


class WorldSubsystemConfigurationError(WorldSubsystemError):
    """Raised when application and world rules are inconsistent."""


class WorldSubsystemStateError(WorldSubsystemError):
    """Raised when subsystem lifecycle state forbids an operation."""


class WorldSubsystem(EngineSubsystemBase):
    """Coordinate one world runtime with engine fixed updates."""

    __slots__ = ("_config", "_runtime", "_started")

    def __init__(self) -> None:
        super().__init__(name="world")
        self._runtime: WorldRuntime | None = None
        self._config: GameConfig | None = None
        self._started = False

    @property
    def started(self) -> bool:
        """Return whether this subsystem has started."""
        return self._started

    @property
    def runtime(self) -> WorldRuntime:
        """Return the bound world runtime."""
        if self._runtime is None:
            raise WorldSubsystemStateError("World subsystem services are not bound.")
        return self._runtime

    @property
    def config(self) -> GameConfig:
        """Return the bound game configuration."""
        if self._config is None:
            raise WorldSubsystemStateError("World subsystem services are not bound.")
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """Return the logger supplied by the bound engine context."""
        return self.context.logger

    def bind_services(self, context: EngineContext) -> None:
        """Bind and resolve required world/application services."""
        super().bind_services(context)
        self._runtime = cast(WorldRuntime, self.require_service(WorldRuntime))
        self._config = cast(GameConfig, self.require_service(GameConfig))

    def start(self) -> None:
        """Validate configuration and map world state into engine startup."""
        runtime = self.runtime
        config = self.config
        self.logger.info(
            "World subsystem startup requested.",
            extra=self._diagnostic_context(event="world.subsystem_start_requested"),
        )

        try:
            if self._started:
                raise WorldSubsystemStateError("World subsystem is already started.")

            model = runtime.model
            if model.specification.seed.value != config.simulation.world_seed:
                raise WorldSubsystemConfigurationError(
                    "World seed must match GameConfig simulation world_seed: "
                    f"world={model.specification.seed.value}, "
                    f"configuration={config.simulation.world_seed}."
                )

            if model.specification.time_config.ticks_per_second != config.simulation.tick_rate:
                raise WorldSubsystemConfigurationError(
                    "World ticks_per_second must match GameConfig simulation tick_rate: "
                    f"world={model.specification.time_config.ticks_per_second}, "
                    f"configuration={config.simulation.tick_rate}."
                )

            state = model.metadata.state
            if state is WorldState.CREATED:
                runtime.initialise()
                runtime.activate()
            elif state is WorldState.INITIALISED:
                runtime.activate()
            elif state is WorldState.PAUSED:
                runtime.resume()
            elif state is not WorldState.ACTIVE:
                raise WorldSubsystemStateError(
                    f"Cannot start world subsystem while world is {state.value!r}."
                )
        except Exception:
            self.logger.error(
                "World subsystem startup rejected.",
                extra=self._diagnostic_context(event="world.subsystem_startup_rejected"),
            )
            raise

        self._started = True
        self.logger.info(
            "World subsystem started.",
            extra=self._diagnostic_context(event="world.subsystem_started"),
        )

    def update(self, fixed_delta_seconds: float) -> None:
        """Advance exactly one world tick for one engine fixed update."""
        if not self._started:
            raise WorldSubsystemStateError("World subsystem is not started.")

        if type(fixed_delta_seconds) is not float:
            raise TypeError("fixed_delta_seconds must be a float.")

        if not math.isfinite(fixed_delta_seconds):
            raise ValueError("fixed_delta_seconds must be finite.")

        if fixed_delta_seconds <= 0.0:
            raise ValueError("fixed_delta_seconds must be greater than zero.")

        self.runtime.advance_tick()
        self.logger.debug(
            "World subsystem fixed update completed.",
            extra=self._diagnostic_context(event="world.subsystem_fixed_update_completed"),
        )

    def render(self, interpolation_alpha: float) -> None:
        """Perform no world rendering."""
        del interpolation_alpha

    def stop(self) -> None:
        """Pause an active world without closing it."""
        if not self._started:
            return

        self.logger.info(
            "World subsystem stopping.",
            extra=self._diagnostic_context(event="world.subsystem_stopping"),
        )
        if self.runtime.model.metadata.state is WorldState.ACTIVE:
            self.runtime.pause()

        self._started = False
        self.logger.info(
            "World subsystem stopped.",
            extra=self._diagnostic_context(event="world.subsystem_stopped"),
        )

    def _diagnostic_context(self, *, event: str) -> dict[str, object]:
        snapshot = self.runtime.snapshot
        world = snapshot.world
        date_time = world.date_time
        return {
            "event": event,
            "world_id": str(world.world_id),
            "world_name": world.name,
            "world_state": world.state.value,
            "world_revision": snapshot.revision,
            "world_tick": world.absolute_world_tick,
            "world_year": date_time.year,
            "world_day_of_year": date_time.day_of_year,
            "world_hour": date_time.hour,
            "world_minute": date_time.minute,
            "world_second": date_time.second,
            "world_tick_within_second": date_time.tick_within_second,
            "world_seed": world.seed.value,
            "world_tick_rate": self.runtime.model.specification.time_config.ticks_per_second,
        }
