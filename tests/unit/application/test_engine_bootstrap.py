"""Tests for application-level engine construction and coordination."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest

from open_world_rpg.application import (
    ApplicationState,
    EngineApplicationExecutionError,
    GameApplication,
    GameMode,
    RuntimeContext,
    SessionState,
    create_engine_runtime,
    fixed_step_config_from_game_config,
    run_engine_application,
    run_engine_smoke_test,
)
from open_world_rpg.core import (
    GameConfig,
    ProjectPaths,
    RuntimeEnvironment,
    SimulationConfig,
)
from open_world_rpg.engine import (
    EngineRuntime,
    EngineRuntimeExecutionError,
    EngineRuntimeState,
    EngineSubsystem,
)


class SequenceClock:
    """Deterministic engine clock."""

    def __init__(self, *timestamps: int) -> None:
        self._timestamps = iter(timestamps)

    def now_ns(self) -> int:
        return next(self._timestamps)


class RecordingSubsystem:
    """Configurable subsystem used by bootstrap tests."""

    name = "world"

    def __init__(
        self,
        events: list[str],
        *,
        failures: set[str] | None = None,
    ) -> None:
        self.events = events
        self.failures = set() if failures is None else failures

    def start(self) -> None:
        self.events.append("start")
        self._fail_if_requested("start")

    def update(self, fixed_delta_seconds: float) -> None:
        self.events.append(f"update:{fixed_delta_seconds}")
        self._fail_if_requested("update")

    def render(self, interpolation_alpha: float) -> None:
        self.events.append(f"render:{interpolation_alpha}")
        self._fail_if_requested("render")

    def stop(self) -> None:
        self.events.append("stop")
        self._fail_if_requested("stop")

    def _fail_if_requested(self, operation: str) -> None:
        if operation in self.failures:
            raise RuntimeError(f"{operation} failure")


def create_test_application(
    tmp_path: Path,
    *,
    tick_rate: int = 20,
    max_frame_skip: int = 3,
    logger: logging.Logger | None = None,
) -> GameApplication:
    config = GameConfig(
        environment=RuntimeEnvironment.TEST,
        simulation=SimulationConfig(
            world_seed=42,
            tick_rate=tick_rate,
            max_frame_skip=max_frame_skip,
        ),
        paths=ProjectPaths.from_project_root(tmp_path),
    )
    context = RuntimeContext.create(
        game_mode=GameMode.NEW_GAME,
        world_seed=config.simulation.world_seed,
    )

    return GameApplication(
        config=config,
        context=context,
        logger=(logging.Logger("test.application") if logger is None else logger),
    )


def test_fixed_step_config_maps_simulation_settings(
    tmp_path: Path,
) -> None:
    application = create_test_application(
        tmp_path,
        tick_rate=30,
        max_frame_skip=4,
    )

    timing = fixed_step_config_from_game_config(application.config)

    assert timing.tick_rate_hz == 30
    assert timing.max_updates_per_frame == 4
    assert timing.max_frame_duration_ns == 250_000_000


def test_fixed_step_config_rejects_invalid_config() -> None:
    with pytest.raises(TypeError, match="GameConfig"):
        fixed_step_config_from_game_config(cast(Any, object()))


def test_create_engine_runtime_maps_application_configuration(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    logger = logging.Logger("test.engine")
    application = create_test_application(
        tmp_path,
        tick_rate=25,
        max_frame_skip=6,
    )
    subsystem = RecordingSubsystem(events)
    clock = SequenceClock(0)

    engine = create_engine_runtime(
        application=application,
        subsystems=[subsystem],
        clock=clock,
        logger=logger,
    )

    assert isinstance(subsystem, EngineSubsystem)
    assert engine.scheduler.config.tick_rate_hz == 25
    assert engine.scheduler.config.max_updates_per_frame == 6
    assert engine.registry.subsystem_names == ("world",)
    assert engine.clock is clock
    assert engine.logger is logger


def test_create_engine_runtime_uses_application_logger(
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)

    engine = create_engine_runtime(
        application=application,
        clock=SequenceClock(0),
    )

    assert engine.logger is application.logger


def test_create_engine_runtime_rejects_invalid_application() -> None:
    with pytest.raises(TypeError, match="GameApplication"):
        create_engine_runtime(application=cast(Any, object()))


def test_run_engine_application_coordinates_successful_lifecycle(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    application = create_test_application(
        tmp_path,
        tick_rate=20,
    )
    engine = create_engine_runtime(
        application=application,
        subsystems=[RecordingSubsystem(events)],
        clock=SequenceClock(
            0,
            50_000_000,
            100_000_000,
        ),
    )

    snapshot = run_engine_application(
        application=application,
        engine=engine,
        max_frames=3,
    )

    assert snapshot.state is EngineRuntimeState.STOPPED
    assert snapshot.frame_count == 3
    assert snapshot.update_count == 2
    assert snapshot.stop_reason == "frame_limit"
    assert engine.state is EngineRuntimeState.STOPPED
    assert application.state is ApplicationState.STOPPED
    assert application.context.state is SessionState.TERMINATED
    assert events == [
        "start",
        "render:0.0",
        "update:0.05",
        "render:0.0",
        "update:0.05",
        "render:0.0",
        "stop",
    ]


def test_run_engine_application_marks_application_failed(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    application = create_test_application(tmp_path)
    engine = create_engine_runtime(
        application=application,
        subsystems=[
            RecordingSubsystem(
                events,
                failures={"render"},
            )
        ],
        clock=SequenceClock(0),
    )

    with pytest.raises(
        EngineApplicationExecutionError,
        match="engine execution",
    ) as error:
        run_engine_application(
            application=application,
            engine=engine,
            max_frames=1,
        )

    assert isinstance(
        error.value.cause,
        EngineRuntimeExecutionError,
    )
    assert error.value.cleanup_error is None
    assert engine.state is EngineRuntimeState.FAILED
    assert application.state is ApplicationState.FAILED
    assert application.context.state is SessionState.FAILED
    assert events == [
        "start",
        "render:0.0",
        "stop",
    ]


def test_engine_failure_reports_application_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)
    engine = create_engine_runtime(
        application=application,
        subsystems=[
            RecordingSubsystem(
                [],
                failures={"render"},
            )
        ],
        clock=SequenceClock(0),
    )

    def raise_cleanup_error(
        self: GameApplication,
    ) -> NoReturn:
        del self
        raise RuntimeError("application cleanup failure")

    monkeypatch.setattr(
        GameApplication,
        "fail",
        raise_cleanup_error,
    )

    with pytest.raises(
        EngineApplicationExecutionError,
        match="cleanup also failed",
    ) as error:
        run_engine_application(
            application=application,
            engine=engine,
            max_frames=1,
        )

    assert isinstance(
        error.value.cleanup_error,
        RuntimeError,
    )


def test_application_start_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)
    engine = create_engine_runtime(
        application=application,
        clock=SequenceClock(0),
    )

    def raise_start_error(
        self: GameApplication,
    ) -> NoReturn:
        del self
        raise RuntimeError("startup failure")

    monkeypatch.setattr(
        GameApplication,
        "start",
        raise_start_error,
    )

    with pytest.raises(
        EngineApplicationExecutionError,
        match="application startup",
    ) as error:
        run_engine_application(
            application=application,
            engine=engine,
            max_frames=1,
        )

    assert isinstance(error.value.cause, RuntimeError)
    assert engine.state is EngineRuntimeState.CREATED


def test_application_shutdown_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)
    engine = create_engine_runtime(
        application=application,
        clock=SequenceClock(0),
    )

    def raise_stop_error(
        self: GameApplication,
    ) -> NoReturn:
        del self
        raise RuntimeError("shutdown failure")

    monkeypatch.setattr(
        GameApplication,
        "stop",
        raise_stop_error,
    )

    with pytest.raises(
        EngineApplicationExecutionError,
        match="application shutdown",
    ) as error:
        run_engine_application(
            application=application,
            engine=engine,
            max_frames=1,
        )

    assert isinstance(error.value.cause, RuntimeError)
    assert engine.state is EngineRuntimeState.STOPPED


def test_run_engine_smoke_test_executes_bounded_frame(
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)
    engine = create_engine_runtime(
        application=application,
        clock=SequenceClock(0),
    )

    snapshot = run_engine_smoke_test(
        application=application,
        engine=engine,
    )

    assert snapshot.frame_count == 1
    assert snapshot.stop_reason == "frame_limit"
    assert application.state is ApplicationState.STOPPED


@pytest.mark.parametrize(
    "application",
    [object(), "application"],
)
def test_run_rejects_invalid_application(
    application: object,
) -> None:
    with pytest.raises(TypeError, match="GameApplication"):
        run_engine_application(
            application=cast(Any, application),
            engine=cast(Any, object()),
            max_frames=1,
        )


def test_run_rejects_invalid_engine(
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)

    with pytest.raises(TypeError, match="EngineRuntime"):
        run_engine_application(
            application=application,
            engine=cast(Any, object()),
            max_frames=1,
        )


@pytest.mark.parametrize(
    "value",
    [True, 1.5, "3"],
)
def test_run_rejects_invalid_frame_limit_type(
    tmp_path: Path,
    value: object,
) -> None:
    application = create_test_application(tmp_path)
    engine = create_engine_runtime(
        application=application,
        clock=SequenceClock(0),
    )

    with pytest.raises(TypeError, match="max_frames"):
        run_engine_application(
            application=application,
            engine=engine,
            max_frames=cast(Any, value),
        )

    assert application.state is ApplicationState.CREATED


@pytest.mark.parametrize("value", [0, -1])
def test_run_rejects_non_positive_frame_limit(
    tmp_path: Path,
    value: int,
) -> None:
    application = create_test_application(tmp_path)
    engine = create_engine_runtime(
        application=application,
        clock=SequenceClock(0),
    )

    with pytest.raises(ValueError, match="greater than zero"):
        run_engine_application(
            application=application,
            engine=engine,
            max_frames=value,
        )


@pytest.mark.parametrize(
    "value",
    [None, True, 1.5, "1"],
)
def test_smoke_test_rejects_invalid_frame_count(
    tmp_path: Path,
    value: object,
) -> None:
    application = create_test_application(tmp_path)
    engine = create_engine_runtime(
        application=application,
        clock=SequenceClock(0),
    )

    with pytest.raises(TypeError, match="frame_count"):
        run_engine_smoke_test(
            application=application,
            engine=engine,
            frame_count=cast(Any, value),
        )


@pytest.mark.parametrize("value", [0, -1])
def test_smoke_test_rejects_non_positive_frame_count(
    tmp_path: Path,
    value: int,
) -> None:
    application = create_test_application(tmp_path)
    engine = create_engine_runtime(
        application=application,
        clock=SequenceClock(0),
    )

    with pytest.raises(ValueError, match="greater than zero"):
        run_engine_smoke_test(
            application=application,
            engine=engine,
            frame_count=value,
        )


class StopRequestClock:
    """Clock that requests a controlled stop during its first frame."""

    def __init__(self) -> None:
        self.engine: EngineRuntime | None = None

    def now_ns(self) -> int:
        assert self.engine is not None
        self.engine.request_stop("test_completed")
        return 0


def test_run_engine_application_allows_unbounded_execution(
    tmp_path: Path,
) -> None:
    application = create_test_application(tmp_path)
    clock = StopRequestClock()
    engine = create_engine_runtime(
        application=application,
        clock=clock,
    )
    clock.engine = engine

    snapshot = run_engine_application(
        application=application,
        engine=engine,
    )

    assert snapshot.state is EngineRuntimeState.STOPPED
    assert snapshot.frame_count == 1
    assert snapshot.update_count == 0
    assert snapshot.stop_reason == "test_completed"
    assert application.state is ApplicationState.STOPPED
    assert application.context.state is SessionState.TERMINATED


class DependencyOrderedSubsystem:
    """Subsystem exposing explicit startup dependencies."""

    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        dependencies: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.dependencies = dependencies
        self.events = events

    def start(self) -> None:
        self.events.append(f"{self.name}:start")

    def update(self, fixed_delta_seconds: float) -> None:
        del fixed_delta_seconds

    def render(self, interpolation_alpha: float) -> None:
        del interpolation_alpha

    def stop(self) -> None:
        self.events.append(f"{self.name}:stop")


def test_create_engine_runtime_resolves_subsystem_dependencies(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    application = create_test_application(tmp_path)
    render_system = DependencyOrderedSubsystem(
        "render",
        events,
        dependencies=("world",),
    )
    input_system = DependencyOrderedSubsystem(
        "input",
        events,
    )
    world_system = DependencyOrderedSubsystem(
        "world",
        events,
        dependencies=("input",),
    )

    engine = create_engine_runtime(
        application=application,
        subsystems=[
            render_system,
            input_system,
            world_system,
        ],
        clock=SequenceClock(0),
    )

    assert engine.registry.subsystem_names == (
        "input",
        "world",
        "render",
    )

    engine.start()
    engine.shutdown()

    assert events == [
        "input:start",
        "world:start",
        "render:start",
        "render:stop",
        "world:stop",
        "input:stop",
    ]
