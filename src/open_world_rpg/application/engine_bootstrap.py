"""Application-level construction and coordination of the game engine."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from open_world_rpg.application.runtime import GameApplication
from open_world_rpg.core import GameConfig
from open_world_rpg.engine import (
    EngineClock,
    EngineRuntime,
    EngineRuntimeSnapshot,
    EngineSubsystem,
    FixedStepConfig,
    FixedStepScheduler,
    SubsystemRegistry,
)


class EngineApplicationError(RuntimeError):
    """Base exception for application and engine coordination failures."""


class EngineApplicationExecutionError(EngineApplicationError):
    """Raised when an application-engine lifecycle operation fails."""

    def __init__(
        self,
        *,
        operation: str,
        cause: Exception,
        cleanup_error: Exception | None = None,
    ) -> None:
        self.operation = operation
        self.cause = cause
        self.cleanup_error = cleanup_error

        message = f"Application-engine execution failed during {operation}."

        if cleanup_error is not None:
            message += " Application cleanup also failed."

        super().__init__(message)


def fixed_step_config_from_game_config(
    config: GameConfig,
) -> FixedStepConfig:
    """Map simulation configuration into deterministic engine timing."""
    if not isinstance(config, GameConfig):
        raise TypeError("config must be a GameConfig.")

    return FixedStepConfig(
        tick_rate_hz=config.simulation.tick_rate,
        max_updates_per_frame=(config.simulation.max_frame_skip),
    )


def create_engine_runtime(
    *,
    application: GameApplication,
    subsystems: Iterable[EngineSubsystem] = (),
    clock: EngineClock | None = None,
    logger: logging.Logger | None = None,
) -> EngineRuntime:
    """Construct an engine runtime from application configuration."""
    if not isinstance(application, GameApplication):
        raise TypeError("application must be a GameApplication.")

    runtime_logger = application.logger if logger is None else logger

    return EngineRuntime(
        registry=SubsystemRegistry(subsystems),
        scheduler=FixedStepScheduler(fixed_step_config_from_game_config(application.config)),
        clock=clock,
        logger=runtime_logger,
    )


def run_engine_application(
    *,
    application: GameApplication,
    engine: EngineRuntime,
    max_frames: int | None = None,
) -> EngineRuntimeSnapshot:
    """Run an engine within a coordinated application lifecycle."""
    if not isinstance(application, GameApplication):
        raise TypeError("application must be a GameApplication.")

    if not isinstance(engine, EngineRuntime):
        raise TypeError("engine must be an EngineRuntime.")

    _validate_frame_limit(
        max_frames,
        name="max_frames",
        allow_none=True,
    )

    try:
        application.start()
    except Exception as exc:
        raise EngineApplicationExecutionError(
            operation="application startup",
            cause=exc,
        ) from exc

    try:
        snapshot = engine.run(max_frames=max_frames)
    except Exception as exc:
        cleanup_error: Exception | None = None

        try:
            application.fail()
        except Exception as application_error:
            cleanup_error = application_error

        raise EngineApplicationExecutionError(
            operation="engine execution",
            cause=exc,
            cleanup_error=cleanup_error,
        ) from exc

    try:
        application.stop()
    except Exception as exc:
        raise EngineApplicationExecutionError(
            operation="application shutdown",
            cause=exc,
        ) from exc

    return snapshot


def run_engine_smoke_test(
    *,
    application: GameApplication,
    engine: EngineRuntime,
    frame_count: int = 1,
) -> EngineRuntimeSnapshot:
    """Run a bounded engine smoke test with clean application shutdown."""
    _validate_frame_limit(
        frame_count,
        name="frame_count",
        allow_none=False,
    )

    return run_engine_application(
        application=application,
        engine=engine,
        max_frames=frame_count,
    )


def _validate_frame_limit(
    value: int | None,
    *,
    name: str,
    allow_none: bool,
) -> None:
    if value is None:
        if allow_none:
            return

        raise TypeError(f"{name} must be an integer.")

    if type(value) is not int:
        suffix = " or None" if allow_none else ""
        raise TypeError(f"{name} must be an integer{suffix}.")

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
