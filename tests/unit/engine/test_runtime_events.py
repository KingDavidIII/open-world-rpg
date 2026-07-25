"""Tests for managed event dispatch within engine frame phases."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from io import StringIO
from typing import Any, cast

import pytest

from open_world_rpg.core import JsonLogFormatter
from open_world_rpg.engine import (
    EngineEventPhase,
    EngineEventPhaseError,
    EngineRuntime,
    EngineRuntimeExecutionError,
    EngineRuntimeState,
    EventBus,
    EventDispatchError,
    FixedStepConfig,
    FixedStepScheduler,
    SubsystemRegistry,
)


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    name: str


class SequenceClock:
    """Deterministic clock backed by predefined timestamps."""

    def __init__(self, *timestamps: int) -> None:
        self._timestamps = iter(timestamps)

    def now_ns(self) -> int:
        return next(self._timestamps)


class EventAwareSubsystem:
    """Subsystem that can publish update and render events."""

    name = "world"

    def __init__(
        self,
        *,
        event_bus: EventBus,
        calls: list[str],
        publish_update: bool = False,
        publish_render: bool = False,
    ) -> None:
        self.event_bus = event_bus
        self.calls = calls
        self.publish_update = publish_update
        self.publish_render = publish_render

    def start(self) -> None:
        self.calls.append("start")

    def update(self, fixed_delta_seconds: float) -> None:
        self.calls.append(f"update:{fixed_delta_seconds}")

        if self.publish_update:
            self.event_bus.publish(RuntimeEvent("from_update"))

    def render(self, interpolation_alpha: float) -> None:
        self.calls.append(f"render:{interpolation_alpha}")

        if self.publish_render:
            self.event_bus.publish(RuntimeEvent("from_render"))

    def stop(self) -> None:
        self.calls.append("stop")


def create_logger(stream: StringIO) -> logging.Logger:
    logger = logging.Logger(
        "test.engine.runtime.events",
        level=logging.DEBUG,
    )
    logger.propagate = False

    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)

    return logger


def create_runtime(
    *,
    timestamps: tuple[int, ...],
    event_bus: EventBus | None = None,
    subsystem: EventAwareSubsystem | None = None,
    logger: logging.Logger | None = None,
) -> EngineRuntime:
    resolved_bus = EventBus() if event_bus is None else event_bus
    registry = SubsystemRegistry([] if subsystem is None else [subsystem])

    return EngineRuntime(
        registry=registry,
        scheduler=FixedStepScheduler(FixedStepConfig(tick_rate_hz=10)),
        clock=SequenceClock(*timestamps),
        logger=logger,
        event_bus=resolved_bus,
    )


def test_runtime_uses_default_event_bus() -> None:
    runtime = EngineRuntime(
        registry=SubsystemRegistry(),
        clock=SequenceClock(0),
    )

    assert isinstance(runtime.event_bus, EventBus)
    assert runtime.last_event_dispatches == ()
    assert runtime.snapshot.event_dispatches == ()
    assert runtime.snapshot.pending_event_count == 0


def test_runtime_rejects_invalid_event_bus() -> None:
    with pytest.raises(TypeError, match="EventBus"):
        EngineRuntime(
            registry=SubsystemRegistry(),
            event_bus=cast(Any, object()),
        )


def test_first_frame_dispatches_preloaded_event_before_render() -> None:
    event_bus = EventBus()
    calls: list[str] = []
    event_bus.subscribe(
        RuntimeEvent,
        lambda event: calls.append(f"event:{event.name}"),
    )
    event_bus.publish(RuntimeEvent("preloaded"))

    subsystem = EventAwareSubsystem(
        event_bus=event_bus,
        calls=calls,
    )
    runtime = create_runtime(
        timestamps=(0,),
        event_bus=event_bus,
        subsystem=subsystem,
    )
    runtime.start()
    calls.clear()

    runtime.run_frame()

    assert calls == [
        "event:preloaded",
        "render:0.0",
    ]
    assert [dispatch.phase for dispatch in runtime.last_event_dispatches] == [
        EngineEventPhase.BEFORE_RENDER,
        EngineEventPhase.AFTER_RENDER,
    ]
    assert runtime.last_event_dispatches[0].report.events_dispatched == 1
    assert runtime.last_event_dispatches[1].report.events_dispatched == 0


def test_event_publication_is_deferred_to_next_phase() -> None:
    event_bus = EventBus()
    calls: list[str] = []

    def handler(event: RuntimeEvent) -> None:
        calls.append(f"event:{event.name}")

        follow_ups = {
            "before_update": "after_update",
            "after_update": "before_render",
            "before_render": "after_render",
            "after_render": "next_frame",
        }

        follow_up = follow_ups.get(event.name)
        if follow_up is not None:
            event_bus.publish(RuntimeEvent(follow_up))

    event_bus.subscribe(RuntimeEvent, handler)

    subsystem = EventAwareSubsystem(
        event_bus=event_bus,
        calls=calls,
    )
    runtime = create_runtime(
        timestamps=(
            0,
            100_000_000,
            200_000_000,
        ),
        event_bus=event_bus,
        subsystem=subsystem,
    )
    runtime.start()
    runtime.run_frame()
    calls.clear()

    event_bus.publish(RuntimeEvent("before_update"))
    runtime.run_frame()

    assert calls == [
        "event:before_update",
        "update:0.1",
        "event:after_update",
        "event:before_render",
        "render:0.0",
        "event:after_render",
    ]
    assert event_bus.pending_event_count == 1
    assert [dispatch.phase for dispatch in runtime.last_event_dispatches] == [
        EngineEventPhase.BEFORE_UPDATE,
        EngineEventPhase.AFTER_UPDATE,
        EngineEventPhase.BEFORE_RENDER,
        EngineEventPhase.AFTER_RENDER,
    ]
    assert [dispatch.update_index for dispatch in runtime.last_event_dispatches] == [
        0,
        0,
        None,
        None,
    ]

    calls.clear()
    runtime.run_frame()

    assert calls[0] == "event:next_frame"


def test_subsystem_events_dispatch_after_update_and_render() -> None:
    event_bus = EventBus()
    calls: list[str] = []
    event_bus.subscribe(
        RuntimeEvent,
        lambda event: calls.append(f"event:{event.name}"),
    )

    subsystem = EventAwareSubsystem(
        event_bus=event_bus,
        calls=calls,
        publish_update=True,
        publish_render=True,
    )
    runtime = create_runtime(
        timestamps=(0, 100_000_000),
        event_bus=event_bus,
        subsystem=subsystem,
    )
    runtime.start()
    runtime.run_frame()
    calls.clear()

    runtime.run_frame()

    assert calls == [
        "update:0.1",
        "event:from_update",
        "render:0.0",
        "event:from_render",
    ]
    assert event_bus.pending_event_count == 0


def test_render_phase_dispatch_failure_fails_runtime() -> None:
    event_bus = EventBus()
    calls: list[str] = []

    def failing_handler(event: RuntimeEvent) -> None:
        raise RuntimeError(event.name)

    event_bus.subscribe(RuntimeEvent, failing_handler)
    event_bus.publish(RuntimeEvent("failure"))

    subsystem = EventAwareSubsystem(
        event_bus=event_bus,
        calls=calls,
    )
    runtime = create_runtime(
        timestamps=(0,),
        event_bus=event_bus,
        subsystem=subsystem,
    )
    runtime.start()
    calls.clear()

    with pytest.raises(
        EngineRuntimeExecutionError,
    ) as error:
        runtime.run_frame()

    assert isinstance(
        error.value.cause,
        EngineEventPhaseError,
    )
    phase_error = error.value.cause
    assert phase_error.phase is EngineEventPhase.BEFORE_RENDER
    assert phase_error.update_index is None
    assert isinstance(
        phase_error.cause,
        EventDispatchError,
    )
    assert phase_error.dispatch.report.failure_count == 1
    assert runtime.state is EngineRuntimeState.FAILED
    assert runtime.last_event_dispatches == (phase_error.dispatch,)
    assert calls == ["stop"]


def test_update_phase_failure_reports_update_index() -> None:
    event_bus = EventBus()
    calls: list[str] = []

    def failing_handler(event: RuntimeEvent) -> None:
        raise RuntimeError(event.name)

    event_bus.subscribe(RuntimeEvent, failing_handler)

    subsystem = EventAwareSubsystem(
        event_bus=event_bus,
        calls=calls,
    )
    runtime = create_runtime(
        timestamps=(0, 100_000_000),
        event_bus=event_bus,
        subsystem=subsystem,
    )
    runtime.start()
    runtime.run_frame()
    calls.clear()
    event_bus.publish(RuntimeEvent("failure"))

    with pytest.raises(
        EngineRuntimeExecutionError,
    ) as error:
        runtime.run_frame()

    phase_error = cast(
        EngineEventPhaseError,
        error.value.cause,
    )
    assert isinstance(
        phase_error,
        EngineEventPhaseError,
    )
    assert phase_error.phase is EngineEventPhase.BEFORE_UPDATE
    assert phase_error.update_index == 0
    assert "fixed update 0" in str(phase_error)
    assert not any(call.startswith("update:") for call in calls)


def test_successful_event_dispatch_emits_diagnostics() -> None:
    stream = StringIO()
    event_bus = EventBus()
    event_bus.subscribe(
        RuntimeEvent,
        lambda event: None,
    )
    event_bus.publish(RuntimeEvent("diagnostic"))

    runtime = create_runtime(
        timestamps=(0,),
        event_bus=event_bus,
        logger=create_logger(stream),
    )
    runtime.start()
    runtime.run_frame()

    payloads = [json.loads(line) for line in stream.getvalue().splitlines()]
    payload = next(item for item in payloads if item["event"] == "engine.events_dispatched")

    assert payload["event_phase"] == "before_render"
    assert payload["event_dispatch_count"] == 1
    assert payload["event_handler_invocation_count"] == 1
    assert payload["event_dispatch_failure_count"] == 0
    assert payload["pending_event_count"] == 0
    assert "event_phase_update_index" not in payload


def test_failed_update_dispatch_emits_phase_diagnostics() -> None:
    stream = StringIO()
    event_bus = EventBus()

    def failing_handler(event: RuntimeEvent) -> None:
        raise RuntimeError(event.name)

    event_bus.subscribe(RuntimeEvent, failing_handler)

    runtime = create_runtime(
        timestamps=(0, 100_000_000),
        event_bus=event_bus,
        logger=create_logger(stream),
    )
    runtime.start()
    runtime.run_frame()
    event_bus.publish(RuntimeEvent("failure"))

    with pytest.raises(EngineRuntimeExecutionError):
        runtime.run_frame()

    payloads = [json.loads(line) for line in stream.getvalue().splitlines()]
    payload = next(item for item in reversed(payloads) if item["event"] == "engine.frame_failed")

    assert payload["event_phase"] == "before_update"
    assert payload["event_phase_update_index"] == 0
    assert payload["event_dispatch_count"] == 1
    assert payload["event_handler_invocation_count"] == 1
    assert payload["event_dispatch_failure_count"] == 1
