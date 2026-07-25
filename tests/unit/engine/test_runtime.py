"""Tests for deterministic engine runtime orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from open_world_rpg.engine import (
    ClockRegressionError,
    EngineClock,
    EngineRuntime,
    EngineRuntimeExecutionError,
    EngineRuntimeState,
    EngineRuntimeStateError,
    FixedStepConfig,
    FixedStepScheduler,
    FrameSchedule,
    SubsystemExecutionError,
    SubsystemRegistry,
    SubsystemRegistryState,
    SubsystemShutdownError,
)


class SequenceClock:
    """Deterministic clock backed by predefined timestamps."""

    def __init__(self, *timestamps: int) -> None:
        self._timestamps = iter(timestamps)

    def now_ns(self) -> int:
        return next(self._timestamps)


class StoppingClock:
    """Clock that requests an engine stop on a chosen sample."""

    def __init__(
        self,
        *timestamps: int,
        stop_on_call: int,
    ) -> None:
        self._timestamps = iter(timestamps)
        self._stop_on_call = stop_on_call
        self._call_count = 0
        self.runtime: EngineRuntime | None = None

    def now_ns(self) -> int:
        self._call_count += 1

        if self._call_count == self._stop_on_call:
            assert self.runtime is not None
            self.runtime.request_stop("clock_requested")

        return next(self._timestamps)


class InvalidClock:
    """Object that does not implement EngineClock."""


class RecordingSubsystem:
    """Configurable engine subsystem test double."""

    def __init__(
        self,
        events: list[str],
        *,
        failures: set[str] | None = None,
        on_update: Callable[[], None] | None = None,
    ) -> None:
        self.name = "world"
        self.events = events
        self.failures = set() if failures is None else failures
        self.on_update = on_update

    def start(self) -> None:
        self.events.append("start")
        self._fail_if_requested("start")

    def update(self, fixed_delta_seconds: float) -> None:
        self.events.append(f"update:{fixed_delta_seconds}")

        if self.on_update is not None:
            self.on_update()

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


def create_runtime(
    *timestamps: int,
    tick_rate_hz: int = 10,
    max_frame_duration_ns: int = 1_000_000_000,
    max_updates_per_frame: int = 8,
    failures: set[str] | None = None,
    clock: EngineClock | None = None,
) -> tuple[EngineRuntime, list[str]]:
    events: list[str] = []
    registry = SubsystemRegistry(
        [
            RecordingSubsystem(
                events,
                failures=failures,
            )
        ]
    )
    scheduler = FixedStepScheduler(
        FixedStepConfig(
            tick_rate_hz=tick_rate_hz,
            max_frame_duration_ns=max_frame_duration_ns,
            max_updates_per_frame=max_updates_per_frame,
        )
    )
    resolved_clock = SequenceClock(*timestamps) if clock is None else clock

    runtime = EngineRuntime(
        registry=registry,
        scheduler=scheduler,
        clock=resolved_clock,
    )
    return runtime, events


def test_sequence_clock_implements_engine_clock() -> None:
    assert isinstance(SequenceClock(0), EngineClock)


def test_runtime_uses_default_scheduler_and_clock() -> None:
    runtime = EngineRuntime(registry=SubsystemRegistry())

    assert isinstance(runtime.scheduler, FixedStepScheduler)
    assert isinstance(runtime.clock, EngineClock)


def test_runtime_rejects_invalid_registry() -> None:
    with pytest.raises(TypeError, match="registry"):
        EngineRuntime(registry=cast(Any, object()))


def test_runtime_rejects_invalid_scheduler() -> None:
    with pytest.raises(TypeError, match="scheduler"):
        EngineRuntime(
            registry=SubsystemRegistry(),
            scheduler=cast(Any, object()),
        )


def test_runtime_rejects_invalid_clock() -> None:
    with pytest.raises(
        TypeError,
        match="implement EngineClock",
    ):
        EngineRuntime(
            registry=SubsystemRegistry(),
            clock=cast(Any, InvalidClock()),
        )


def test_runtime_initial_state() -> None:
    runtime, _ = create_runtime(0)

    assert runtime.state is EngineRuntimeState.CREATED
    assert runtime.frame_count == 0
    assert runtime.update_count == 0
    assert runtime.dropped_update_count == 0
    assert runtime.stop_reason is None
    assert runtime.last_schedule is None
    assert runtime.snapshot.state is EngineRuntimeState.CREATED


def test_start_initialises_subsystems_and_scheduler() -> None:
    runtime, events = create_runtime(0)
    runtime.scheduler.advance(999)

    runtime.start()

    assert runtime.state is EngineRuntimeState.RUNNING
    assert runtime.registry.state is SubsystemRegistryState.STARTED
    assert runtime.scheduler.started is False
    assert runtime.scheduler.frame_count == 0
    assert events == ["start"]


def test_start_rejects_invalid_state() -> None:
    runtime, _ = create_runtime(0)
    runtime.start()

    with pytest.raises(
        EngineRuntimeStateError,
        match="runtime is running",
    ):
        runtime.start()


def test_start_failure_is_wrapped_and_cleaned_up() -> None:
    runtime, events = create_runtime(
        0,
        failures={"start"},
    )

    with pytest.raises(
        EngineRuntimeExecutionError,
        match="failed during start",
    ) as error:
        runtime.start()

    assert error.value.operation == "start"
    assert isinstance(
        error.value.cause,
        SubsystemExecutionError,
    )
    assert error.value.cleanup_error is None
    assert runtime.state is EngineRuntimeState.FAILED
    assert runtime.registry.state is SubsystemRegistryState.STOPPED
    assert events == ["start"]


def test_first_frame_renders_without_updates() -> None:
    runtime, events = create_runtime(0)
    runtime.start()
    events.clear()

    schedule = runtime.run_frame()

    assert schedule.frame_index == 0
    assert schedule.elapsed_ns == 0
    assert schedule.update_count == 0
    assert runtime.frame_count == 1
    assert runtime.update_count == 0
    assert runtime.last_schedule is schedule
    assert events == ["render:0.0"]


def test_runtime_executes_fixed_updates_before_render() -> None:
    runtime, events = create_runtime(
        0,
        100_000_000,
    )
    runtime.start()
    runtime.run_frame()
    events.clear()

    schedule = runtime.run_frame()

    assert schedule.update_count == 1
    assert runtime.frame_count == 2
    assert runtime.update_count == 1
    assert runtime.dropped_update_count == 0
    assert events == [
        "update:0.1",
        "render:0.0",
    ]


def test_runtime_executes_multiple_updates() -> None:
    runtime, events = create_runtime(
        0,
        350_000_000,
    )
    runtime.start()
    runtime.run_frame()
    events.clear()

    schedule = runtime.run_frame()

    assert schedule.update_count == 3
    assert runtime.update_count == 3
    assert events == [
        "update:0.1",
        "update:0.1",
        "update:0.1",
        "render:0.5",
    ]


def test_runtime_tracks_dropped_updates() -> None:
    runtime, _ = create_runtime(
        0,
        1_000_000_000,
        max_updates_per_frame=2,
    )
    runtime.start()
    runtime.run_frame()

    schedule = runtime.run_frame()

    assert schedule.update_count == 2
    assert schedule.dropped_update_count == 8
    assert runtime.update_count == 2
    assert runtime.dropped_update_count == 8


@pytest.mark.parametrize(
    "state",
    [
        EngineRuntimeState.CREATED,
        EngineRuntimeState.STOP_REQUESTED,
        EngineRuntimeState.STOPPED,
    ],
)
def test_run_frame_requires_running_state(
    state: EngineRuntimeState,
) -> None:
    runtime, _ = create_runtime(0)

    if state is EngineRuntimeState.STOP_REQUESTED:
        runtime.start()
        runtime.request_stop()
    elif state is EngineRuntimeState.STOPPED:
        runtime.shutdown()

    with pytest.raises(
        EngineRuntimeStateError,
        match=f"runtime is {state.value}",
    ):
        runtime.run_frame()


def test_clock_regression_fails_runtime_and_cleans_up() -> None:
    runtime, events = create_runtime(100, 99)
    runtime.start()
    runtime.run_frame()
    events.clear()

    with pytest.raises(
        EngineRuntimeExecutionError,
        match="frame execution",
    ) as error:
        runtime.run_frame()

    assert isinstance(
        error.value.cause,
        ClockRegressionError,
    )
    assert runtime.state is EngineRuntimeState.FAILED
    assert runtime.registry.state is SubsystemRegistryState.STOPPED
    assert events == ["stop"]


def test_update_failure_preserves_completed_statistics() -> None:
    runtime, events = create_runtime(
        0,
        100_000_000,
        failures={"update"},
    )
    runtime.start()
    runtime.run_frame()
    events.clear()

    with pytest.raises(
        EngineRuntimeExecutionError,
        match="frame execution",
    ) as error:
        runtime.run_frame()

    assert isinstance(
        error.value.cause,
        SubsystemExecutionError,
    )
    assert runtime.frame_count == 1
    assert runtime.update_count == 0
    assert runtime.last_schedule is not None
    assert events == ["update:0.1", "stop"]


def test_render_failure_counts_completed_updates_only() -> None:
    runtime, events = create_runtime(
        0,
        100_000_000,
        failures={"render"},
    )
    runtime.start()

    with pytest.raises(EngineRuntimeExecutionError):
        runtime.run_frame()

    assert runtime.frame_count == 0
    assert runtime.update_count == 0
    assert runtime.last_schedule is None
    assert events == ["start", "render:0.0", "stop"]


def test_cleanup_failure_is_attached_to_frame_error() -> None:
    runtime, _ = create_runtime(
        0,
        100_000_000,
        failures={"update", "stop"},
    )
    runtime.start()
    runtime.run_frame()

    with pytest.raises(
        EngineRuntimeExecutionError,
        match="cleanup also failed",
    ) as error:
        runtime.run_frame()

    assert isinstance(
        error.value.cause,
        SubsystemExecutionError,
    )
    assert isinstance(
        error.value.cleanup_error,
        SubsystemShutdownError,
    )
    assert runtime.state is EngineRuntimeState.FAILED


@pytest.mark.parametrize("reason", [123, None])
def test_request_stop_rejects_invalid_reason_type(
    reason: object,
) -> None:
    runtime, _ = create_runtime(0)
    runtime.start()

    with pytest.raises(TypeError, match="must be a string"):
        runtime.request_stop(cast(Any, reason))


@pytest.mark.parametrize("reason", ["", " ", "\t"])
def test_request_stop_rejects_empty_reason(
    reason: str,
) -> None:
    runtime, _ = create_runtime(0)
    runtime.start()

    with pytest.raises(ValueError, match="cannot be empty"):
        runtime.request_stop(reason)


@pytest.mark.parametrize("reason", [" stop", "stop ", " stop "])
def test_request_stop_rejects_surrounding_whitespace(
    reason: str,
) -> None:
    runtime, _ = create_runtime(0)
    runtime.start()

    with pytest.raises(
        ValueError,
        match="surrounding whitespace",
    ):
        runtime.request_stop(reason)


def test_request_stop_transitions_once() -> None:
    runtime, _ = create_runtime(0)
    runtime.start()

    runtime.request_stop("user_exit")
    runtime.request_stop("second_request")

    assert runtime.state is EngineRuntimeState.STOP_REQUESTED
    assert runtime.stop_reason == "user_exit"


@pytest.mark.parametrize(
    "prepare",
    [
        "created",
        "stopped",
    ],
)
def test_request_stop_requires_running_state(
    prepare: str,
) -> None:
    runtime, _ = create_runtime(0)

    if prepare == "stopped":
        runtime.shutdown()

    with pytest.raises(
        EngineRuntimeStateError,
        match=f"runtime is {prepare}",
    ):
        runtime.request_stop()


def test_shutdown_from_created_state() -> None:
    runtime, events = create_runtime(0)

    runtime.shutdown()

    assert runtime.state is EngineRuntimeState.STOPPED
    assert runtime.stop_reason == "shutdown"
    assert events == []


def test_shutdown_stops_running_runtime() -> None:
    runtime, events = create_runtime(0)
    runtime.start()
    events.clear()

    runtime.shutdown()

    assert runtime.state is EngineRuntimeState.STOPPED
    assert runtime.stop_reason == "shutdown"
    assert events == ["stop"]


def test_shutdown_preserves_requested_stop_reason() -> None:
    runtime, _ = create_runtime(0)
    runtime.start()
    runtime.request_stop("user_exit")

    runtime.shutdown()

    assert runtime.state is EngineRuntimeState.STOPPED
    assert runtime.stop_reason == "user_exit"


def test_shutdown_is_idempotent() -> None:
    runtime, events = create_runtime(0)
    runtime.start()
    events.clear()

    runtime.shutdown()
    runtime.shutdown()

    assert events == ["stop"]


def test_shutdown_failure_is_wrapped() -> None:
    runtime, _ = create_runtime(
        0,
        failures={"stop"},
    )
    runtime.start()

    with pytest.raises(
        EngineRuntimeExecutionError,
        match="failed during shutdown",
    ) as error:
        runtime.shutdown()

    assert isinstance(
        error.value.cause,
        SubsystemShutdownError,
    )
    assert error.value.cleanup_error is None
    assert runtime.state is EngineRuntimeState.FAILED

    runtime.shutdown()
    assert runtime.state is EngineRuntimeState.STOPPED


def test_run_stops_at_frame_limit() -> None:
    runtime, events = create_runtime(
        0,
        100_000_000,
        200_000_000,
    )

    snapshot = runtime.run(max_frames=3)

    assert snapshot.state is EngineRuntimeState.STOPPED
    assert snapshot.frame_count == 3
    assert snapshot.update_count == 2
    assert snapshot.dropped_update_count == 0
    assert snapshot.stop_reason == "frame_limit"
    assert runtime.state is EngineRuntimeState.STOPPED
    assert events == [
        "start",
        "render:0.0",
        "update:0.1",
        "render:0.0",
        "update:0.1",
        "render:0.0",
        "stop",
    ]


def test_run_without_frame_limit_honours_stop_request() -> None:
    clock = StoppingClock(
        0,
        100_000_000,
        stop_on_call=2,
    )
    runtime, events = create_runtime(
        clock=clock,
    )
    clock.runtime = runtime

    snapshot = runtime.run()

    assert snapshot.state is EngineRuntimeState.STOPPED
    assert snapshot.frame_count == 2
    assert snapshot.update_count == 1
    assert snapshot.stop_reason == "clock_requested"
    assert events[-1] == "stop"


@pytest.mark.parametrize("value", [True, 1.5, "3"])
def test_run_rejects_invalid_frame_limit_type(
    value: object,
) -> None:
    runtime, _ = create_runtime(0)

    with pytest.raises(TypeError, match="max_frames"):
        runtime.run(max_frames=cast(Any, value))


@pytest.mark.parametrize("value", [0, -1])
def test_run_rejects_non_positive_frame_limit(
    value: int,
) -> None:
    runtime, _ = create_runtime(0)

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        runtime.run(max_frames=value)


def test_run_rejects_already_started_runtime() -> None:
    runtime, _ = create_runtime(0)
    runtime.start()

    with pytest.raises(EngineRuntimeStateError):
        runtime.run(max_frames=1)


def test_run_propagates_frame_failure_after_cleanup() -> None:
    runtime, events = create_runtime(
        0,
        failures={"render"},
    )

    with pytest.raises(EngineRuntimeExecutionError):
        runtime.run(max_frames=1)

    assert runtime.state is EngineRuntimeState.FAILED
    assert runtime.registry.state is SubsystemRegistryState.STOPPED
    assert events == ["start", "render:0.0", "stop"]


class ShutdownAfterFrameRuntime(EngineRuntime):
    """Runtime that completes a frame and then stops itself."""

    def run_frame(self) -> FrameSchedule:
        schedule = super().run_frame()
        self.shutdown()
        return schedule


def test_run_returns_when_runtime_stops_during_frame() -> None:
    events: list[str] = []
    runtime = ShutdownAfterFrameRuntime(
        registry=SubsystemRegistry([RecordingSubsystem(events)]),
        scheduler=FixedStepScheduler(FixedStepConfig(tick_rate_hz=10)),
        clock=SequenceClock(0),
    )

    snapshot = runtime.run()

    assert snapshot.state is EngineRuntimeState.STOPPED
    assert snapshot.frame_count == 1
    assert snapshot.update_count == 0
    assert snapshot.stop_reason == "shutdown"
    assert events == [
        "start",
        "render:0.0",
        "stop",
    ]
