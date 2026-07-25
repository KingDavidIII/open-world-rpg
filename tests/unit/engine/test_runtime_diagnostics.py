"""Tests for structured engine runtime diagnostics."""

from __future__ import annotations

import json
import logging
from io import StringIO
from typing import Any, cast

import pytest

from open_world_rpg.core import JsonLogFormatter
from open_world_rpg.engine import (
    EngineRuntime,
    EngineRuntimeExecutionError,
    EngineRuntimeState,
    FixedStepConfig,
    FixedStepScheduler,
    SubsystemRegistry,
)


class SequenceClock:
    """Deterministic engine clock for diagnostic tests."""

    def __init__(self, *timestamps: int) -> None:
        self._timestamps = iter(timestamps)

    def now_ns(self) -> int:
        return next(self._timestamps)


class DiagnosticSubsystem:
    """Configurable subsystem used by diagnostic tests."""

    name = "world"

    def __init__(
        self,
        *,
        failures: set[str] | None = None,
    ) -> None:
        self.failures = set() if failures is None else failures

    def start(self) -> None:
        self._fail_if_requested("start")

    def update(self, fixed_delta_seconds: float) -> None:
        del fixed_delta_seconds
        self._fail_if_requested("update")

    def render(self, interpolation_alpha: float) -> None:
        del interpolation_alpha
        self._fail_if_requested("render")

    def stop(self) -> None:
        self._fail_if_requested("stop")

    def _fail_if_requested(self, operation: str) -> None:
        if operation in self.failures:
            raise RuntimeError(f"{operation} failure")


def create_logger(stream: StringIO) -> logging.Logger:
    logger = logging.Logger(
        "test.open_world_rpg.engine.runtime",
        level=logging.DEBUG,
    )
    logger.propagate = False

    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)

    return logger


def create_runtime(
    stream: StringIO,
    *timestamps: int,
    failures: set[str] | None = None,
    max_updates_per_frame: int = 8,
) -> EngineRuntime:
    return EngineRuntime(
        registry=SubsystemRegistry([DiagnosticSubsystem(failures=failures)]),
        scheduler=FixedStepScheduler(
            FixedStepConfig(
                tick_rate_hz=10,
                max_frame_duration_ns=1_000_000_000,
                max_updates_per_frame=max_updates_per_frame,
            )
        ),
        clock=SequenceClock(*timestamps),
        logger=create_logger(stream),
    )


def read_payloads(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def payload_for(
    stream: StringIO,
    event: str,
) -> dict[str, object]:
    return next(payload for payload in reversed(read_payloads(stream)) if payload["event"] == event)


def test_runtime_rejects_invalid_logger() -> None:
    with pytest.raises(TypeError, match="logger"):
        EngineRuntime(
            registry=SubsystemRegistry(),
            logger=cast(Any, object()),
        )


def test_start_emits_structured_lifecycle_events() -> None:
    stream = StringIO()
    runtime = create_runtime(stream, 0)

    assert runtime.logger.name == ("test.open_world_rpg.engine.runtime")

    runtime.start()

    payloads = read_payloads(stream)

    assert [payload["event"] for payload in payloads] == [
        "engine.starting",
        "engine.started",
    ]
    assert payloads[0]["engine_state"] == "created"
    assert payloads[1]["engine_state"] == "running"
    assert payloads[1]["subsystem_count"] == 1
    assert payloads[1]["cumulative_frame_count"] == 0


def test_completed_frame_emits_schedule_statistics() -> None:
    stream = StringIO()
    runtime = create_runtime(
        stream,
        0,
        100_000_000,
    )
    runtime.start()
    runtime.run_frame()
    runtime.run_frame()

    payload = payload_for(
        stream,
        "engine.frame_completed",
    )

    assert payload["engine_state"] == "running"
    assert payload["frame_index"] == 1
    assert payload["frame_elapsed_ns"] == 100_000_000
    assert payload["frame_simulated_elapsed_ns"] == 100_000_000
    assert payload["frame_update_count"] == 1
    assert payload["frame_dropped_update_count"] == 0
    assert payload["interpolation_alpha"] == 0.0
    assert payload["cumulative_frame_count"] == 2
    assert payload["cumulative_update_count"] == 1


def test_dropped_updates_emit_warning_diagnostics() -> None:
    stream = StringIO()
    runtime = create_runtime(
        stream,
        0,
        1_000_000_000,
        max_updates_per_frame=2,
    )
    runtime.start()
    runtime.run_frame()
    runtime.run_frame()

    payload = payload_for(
        stream,
        "engine.updates_dropped",
    )

    assert payload["frame_index"] == 1
    assert payload["frame_update_count"] == 2
    assert payload["frame_dropped_update_count"] == 8
    assert payload["cumulative_update_count"] == 2
    assert payload["cumulative_dropped_update_count"] == 8


def test_stop_request_and_shutdown_emit_reason() -> None:
    stream = StringIO()
    runtime = create_runtime(stream, 0)
    runtime.start()

    runtime.request_stop("user_exit")
    runtime.shutdown()

    requested = payload_for(
        stream,
        "engine.stop_requested",
    )
    stopped = payload_for(
        stream,
        "engine.stopped",
    )

    assert requested["engine_state"] == "stop_requested"
    assert requested["stop_reason"] == "user_exit"
    assert stopped["engine_state"] == "stopped"
    assert stopped["stop_reason"] == "user_exit"


def test_start_failure_emits_exception_diagnostics() -> None:
    stream = StringIO()
    runtime = create_runtime(
        stream,
        0,
        failures={"start"},
    )

    with pytest.raises(EngineRuntimeExecutionError):
        runtime.start()

    payload = payload_for(
        stream,
        "engine.start_failed",
    )

    assert runtime.state is EngineRuntimeState.FAILED
    assert payload["engine_state"] == "failed"
    assert payload["engine_operation"] == "start"
    assert "cleanup_failed" not in payload
    assert "SubsystemExecutionError" in payload["exception"]


def test_frame_failure_reports_schedule_and_cleanup_failure() -> None:
    stream = StringIO()
    runtime = create_runtime(
        stream,
        0,
        100_000_000,
        failures={"update", "stop"},
    )
    runtime.start()
    runtime.run_frame()

    with pytest.raises(EngineRuntimeExecutionError):
        runtime.run_frame()

    payload = payload_for(
        stream,
        "engine.frame_failed",
    )

    assert payload["engine_state"] == "failed"
    assert payload["engine_operation"] == "frame_execution"
    assert payload["frame_index"] == 1
    assert payload["frame_update_count"] == 1
    assert payload["cleanup_failed"] is True
    assert "SubsystemExecutionError" in payload["exception"]


def test_shutdown_failure_emits_exception_diagnostics() -> None:
    stream = StringIO()
    runtime = create_runtime(
        stream,
        0,
        failures={"stop"},
    )
    runtime.start()

    with pytest.raises(EngineRuntimeExecutionError):
        runtime.shutdown()

    payload = payload_for(
        stream,
        "engine.shutdown_failed",
    )

    assert payload["engine_state"] == "failed"
    assert payload["engine_operation"] == "shutdown"
    assert payload["cleanup_failed"] is True
    assert "SubsystemShutdownError" in payload["exception"]
