"""End-to-end acceptance coverage for the completed World Foundation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from open_world_rpg.application import (
    ApplicationState,
    GameApplication,
    GameMode,
    RuntimeContext,
    SessionState,
    create_world_engine_runtime,
    run_engine_application,
)
from open_world_rpg.core import (
    GameConfig,
    ProjectPaths,
    RuntimeEnvironment,
    SimulationConfig,
)
from open_world_rpg.engine import EngineRuntimeState, EventBus
from open_world_rpg.world import (
    WorldRuntime,
    WorldState,
    WorldStateChanged,
    WorldTimeAdvanced,
)

SESSION_ID = UUID("12345678-1234-5678-1234-567812345678")
CREATED_AT = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)


class DeterministicFrameClock:
    def __init__(self, *timestamps_ns: int) -> None:
        self._timestamps = iter(timestamps_ns)

    def now_ns(self) -> int:
        return next(self._timestamps)


class RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def create_test_application(
    tmp_path: Path,
    *,
    logger: logging.Logger,
) -> GameApplication:
    config = GameConfig(
        title="Open World RPG",
        environment=RuntimeEnvironment.TEST,
        simulation=SimulationConfig(
            world_seed=77,
            tick_rate=20,
            max_frame_skip=5,
        ),
        paths=ProjectPaths.from_project_root(tmp_path),
    )
    context = RuntimeContext.create(
        game_mode=GameMode.NEW_GAME,
        world_seed=77,
        session_id=SESSION_ID,
        clock=lambda: CREATED_AT,
    )
    return GameApplication(
        config=config,
        context=context,
        logger=logger,
    )


def test_complete_application_engine_world_acceptance(tmp_path: Path) -> None:
    logger = logging.Logger("test.integration.world_foundation")
    logger.setLevel(logging.DEBUG)
    log_handler = RecordingHandler()
    logger.addHandler(log_handler)
    application = create_test_application(tmp_path, logger=logger)
    event_bus = EventBus()
    step_ns = 50_000_000
    engine = create_world_engine_runtime(
        application=application,
        event_bus=event_bus,
        clock=DeterministicFrameClock(
            0,
            step_ns,
            step_ns * 2,
            step_ns * 3,
        ),
    )
    world_runtime = engine.context.resolve(WorldRuntime)
    world_events: list[WorldStateChanged | WorldTimeAdvanced] = []
    event_bus.subscribe(WorldStateChanged, world_events.append)
    event_bus.subscribe(WorldTimeAdvanced, world_events.append)

    assert world_runtime.model.metadata.state is WorldState.CREATED
    assert world_runtime.revision == 0

    engine_snapshot = run_engine_application(
        application=application,
        engine=engine,
        max_frames=4,
    )
    event_bus.dispatch_pending()

    world = world_runtime.model
    assert engine_snapshot.state is EngineRuntimeState.STOPPED
    assert engine_snapshot.frame_count == 4
    assert engine_snapshot.update_count == 3
    assert application.state is ApplicationState.STOPPED
    assert application.context.state is SessionState.TERMINATED
    assert world.metadata.state is WorldState.PAUSED
    assert world.clock.current.tick == 3
    assert world_runtime.revision == 6
    assert world.metadata.world_id.value == application.context.session_id
    assert world.metadata.created_at == application.context.created_at == CREATED_AT
    assert world.metadata.seed == application.config.simulation.world_seed == 77
    assert (
        world.specification.time_config.ticks_per_second
        == application.config.simulation.tick_rate
        == 20
    )
    assert [
        (
            type(event),
            event.revision,
        )
        for event in world_events
    ] == [
        (WorldStateChanged, 1),
        (WorldStateChanged, 2),
        (WorldTimeAdvanced, 3),
        (WorldTimeAdvanced, 4),
        (WorldTimeAdvanced, 5),
        (WorldStateChanged, 6),
    ]
    assert [
        getattr(record, "event", None)
        for record in log_handler.records
        if getattr(record, "event", "").startswith("world.")
    ] == [
        "world.subsystem_start_requested",
        "world.initialised",
        "world.activated",
        "world.subsystem_started",
        "world.time_advanced",
        "world.subsystem_fixed_update_completed",
        "world.time_advanced",
        "world.subsystem_fixed_update_completed",
        "world.time_advanced",
        "world.subsystem_fixed_update_completed",
        "world.subsystem_stopping",
        "world.paused",
        "world.subsystem_stopped",
    ]
