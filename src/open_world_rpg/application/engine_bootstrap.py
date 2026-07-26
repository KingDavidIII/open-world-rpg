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
    EngineServiceRegistration,
    EngineSubsystem,
    EventBus,
    FixedStepConfig,
    FixedStepScheduler,
    SubsystemRegistry,
    create_engine_context,
    resolve_subsystem_order,
)
from open_world_rpg.world import (
    TerrainGenerationConfig,
    TerrainGenerationService,
    TerrainRuntime,
    WorldId,
    WorldModel,
    WorldRuntime,
    WorldSeed,
    WorldSpecification,
    WorldSubsystem,
    WorldTimeConfig,
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
    event_bus: EventBus | None = None,
    service_registrations: Iterable[EngineServiceRegistration] = (),
) -> EngineRuntime:
    """Construct an engine runtime from application configuration."""
    if not isinstance(application, GameApplication):
        raise TypeError("application must be a GameApplication.")

    runtime_logger = application.logger if logger is None else logger
    resolved_event_bus = EventBus() if event_bus is None else event_bus
    context = create_engine_context(
        logger=runtime_logger,
        event_bus=resolved_event_bus,
        registrations=(
            EngineServiceRegistration(
                GameConfig,
                application.config,
            ),
            *tuple(service_registrations),
        ),
    )

    return EngineRuntime(
        registry=SubsystemRegistry(resolve_subsystem_order(subsystems)),
        scheduler=FixedStepScheduler(fixed_step_config_from_game_config(application.config)),
        clock=clock,
        context=context,
    )


def create_world_model(
    *,
    application: GameApplication,
    name: str | None = None,
) -> WorldModel:
    """Create a default world using application identity and simulation rules."""
    if not isinstance(application, GameApplication):
        raise TypeError("application must be a GameApplication.")

    world_name = application.config.title if name is None else name
    specification = WorldSpecification(
        name=world_name,
        seed=WorldSeed(value=application.config.simulation.world_seed),
        time_config=WorldTimeConfig(
            ticks_per_second=application.config.simulation.tick_rate,
        ),
    )
    return WorldModel.create(
        specification=specification,
        created_at=application.context.created_at,
        world_id=WorldId(value=application.context.session_id),
    )


def create_world_runtime(
    *,
    application: GameApplication,
    name: str | None = None,
    event_bus: EventBus | None = None,
    logger: logging.Logger | None = None,
) -> WorldRuntime:
    """Create a controlled world runtime from an application."""
    if event_bus is not None and not isinstance(event_bus, EventBus):
        raise TypeError("event_bus must be an EventBus or None.")

    if logger is not None and not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger or None.")

    model = create_world_model(
        application=application,
        name=name,
    )
    return WorldRuntime(
        model=model,
        event_bus=event_bus,
        logger=application.logger if logger is None else logger,
    )


def create_terrain_generation_service(
    *,
    world: WorldModel | WorldSpecification,
    config: TerrainGenerationConfig | None = None,
) -> TerrainGenerationService:
    """Create a compatible terrain service from a world model or specification."""
    specification = _world_specification(world)
    resolved_config = TerrainGenerationConfig() if config is None else config
    if not isinstance(resolved_config, TerrainGenerationConfig):
        raise TypeError("config must be a TerrainGenerationConfig or None.")
    return TerrainGenerationService(
        specification=specification,
        config=resolved_config,
    )


def create_terrain_runtime(
    *,
    world: WorldModel | WorldSpecification,
    config: TerrainGenerationConfig | None = None,
    service: TerrainGenerationService | None = None,
    event_bus: EventBus | None = None,
    logger: logging.Logger | None = None,
) -> TerrainRuntime:
    """Create a compatible terrain runtime without engine registration."""
    specification = _world_specification(world)
    if event_bus is not None and not isinstance(event_bus, EventBus):
        raise TypeError("event_bus must be an EventBus or None.")
    if logger is not None and not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger or None.")
    resolved_service = (
        create_terrain_generation_service(world=specification, config=config)
        if service is None
        else service
    )
    if not isinstance(resolved_service, TerrainGenerationService):
        raise TypeError("service must be a TerrainGenerationService or None.")
    return TerrainRuntime(
        specification=specification,
        service=resolved_service,
        event_bus=event_bus,
        logger=logger,
    )


def create_application_terrain_runtime(
    *,
    application: GameApplication,
    world: WorldModel | WorldSpecification | None = None,
    config: TerrainGenerationConfig | None = None,
    event_bus: EventBus | None = None,
) -> TerrainRuntime:
    """Create terrain infrastructure using the application logger."""
    if not isinstance(application, GameApplication):
        raise TypeError("application must be a GameApplication.")
    resolved_world = create_world_model(application=application) if world is None else world
    return create_terrain_runtime(
        world=resolved_world,
        config=config,
        event_bus=event_bus,
        logger=application.logger,
    )


def _world_specification(
    world: WorldModel | WorldSpecification,
) -> WorldSpecification:
    if isinstance(world, WorldModel):
        return world.specification
    if isinstance(world, WorldSpecification):
        return world
    raise TypeError("world must be a WorldModel or WorldSpecification.")


def create_world_engine_runtime(
    *,
    application: GameApplication,
    name: str | None = None,
    subsystems: Iterable[EngineSubsystem] = (),
    clock: EngineClock | None = None,
    logger: logging.Logger | None = None,
    event_bus: EventBus | None = None,
    service_registrations: Iterable[EngineServiceRegistration] = (),
) -> EngineRuntime:
    """Create an engine with a wired world runtime and world subsystem."""
    if not isinstance(application, GameApplication):
        raise TypeError("application must be a GameApplication.")

    resolved_event_bus = EventBus() if event_bus is None else event_bus
    resolved_logger = application.logger if logger is None else logger
    world_runtime = create_world_runtime(
        application=application,
        name=name,
        event_bus=resolved_event_bus,
        logger=resolved_logger,
    )
    return create_engine_runtime(
        application=application,
        subsystems=(WorldSubsystem(), *tuple(subsystems)),
        clock=clock,
        logger=resolved_logger,
        event_bus=resolved_event_bus,
        service_registrations=(
            EngineServiceRegistration(WorldRuntime, world_runtime),
            *tuple(service_registrations),
        ),
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
