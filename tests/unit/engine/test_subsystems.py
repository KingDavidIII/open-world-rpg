"""Tests for ordered engine subsystem lifecycle management."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest

from open_world_rpg.engine import (
    DuplicateSubsystemError,
    EngineSubsystem,
    SubsystemExecutionError,
    SubsystemRegistry,
    SubsystemRegistryState,
    SubsystemRegistryStateError,
    SubsystemShutdownError,
)


class RecordingSubsystem:
    """Configurable subsystem test double."""

    def __init__(
        self,
        name: object,
        events: list[str],
        *,
        failures: set[str] | None = None,
    ) -> None:
        self.name = cast(Any, name)
        self.events = events
        self.failures = set() if failures is None else failures

    def start(self) -> None:
        self.events.append(f"{self.name}:start")
        self._fail_if_requested("start")

    def update(self, fixed_delta_seconds: float) -> None:
        self.events.append(f"{self.name}:update:{fixed_delta_seconds}")
        self._fail_if_requested("update")

    def render(self, interpolation_alpha: float) -> None:
        self.events.append(f"{self.name}:render:{interpolation_alpha}")
        self._fail_if_requested("render")

    def stop(self) -> None:
        self.events.append(f"{self.name}:stop")
        self._fail_if_requested("stop")

    def _fail_if_requested(self, operation: str) -> None:
        if operation in self.failures:
            raise RuntimeError(f"{self.name} {operation} failure")


class IncompleteSubsystem:
    """Object that does not satisfy EngineSubsystem."""

    name = "incomplete"


def create_registry(
    *,
    failures: dict[str, set[str]] | None = None,
) -> tuple[SubsystemRegistry, list[str]]:
    events: list[str] = []
    resolved_failures = {} if failures is None else failures

    registry = SubsystemRegistry(
        [
            RecordingSubsystem(
                "input",
                events,
                failures=resolved_failures.get("input"),
            ),
            RecordingSubsystem(
                "world",
                events,
                failures=resolved_failures.get("world"),
            ),
            RecordingSubsystem(
                "render",
                events,
                failures=resolved_failures.get("render"),
            ),
        ]
    )
    return registry, events


def test_recording_subsystem_implements_contract() -> None:
    subsystem = RecordingSubsystem("world", [])

    assert isinstance(subsystem, EngineSubsystem)


def test_registry_starts_empty_in_created_state() -> None:
    registry = SubsystemRegistry()

    assert registry.state is SubsystemRegistryState.CREATED
    assert registry.subsystem_count == 0
    assert registry.subsystem_names == ()
    assert registry.started_subsystem_names == ()


def test_constructor_registers_subsystems_in_order() -> None:
    events: list[str] = []
    registry = SubsystemRegistry(
        [
            RecordingSubsystem("input", events),
            RecordingSubsystem("world", events),
        ]
    )

    assert registry.subsystem_count == 2
    assert registry.subsystem_names == ("input", "world")


def test_register_adds_subsystem() -> None:
    registry = SubsystemRegistry()
    registry.register(RecordingSubsystem("world", []))

    assert registry.subsystem_names == ("world",)


def test_register_rejects_incomplete_subsystem() -> None:
    registry = SubsystemRegistry()

    with pytest.raises(
        TypeError,
        match="implement EngineSubsystem",
    ):
        registry.register(cast(Any, IncompleteSubsystem()))


def test_register_rejects_non_string_name() -> None:
    registry = SubsystemRegistry()

    with pytest.raises(TypeError, match="name must be a string"):
        registry.register(RecordingSubsystem(123, []))


@pytest.mark.parametrize("name", ["", " ", "\t"])
def test_register_rejects_empty_name(name: str) -> None:
    registry = SubsystemRegistry()

    with pytest.raises(ValueError, match="cannot be empty"):
        registry.register(RecordingSubsystem(name, []))


@pytest.mark.parametrize("name", [" world", "world ", " world "])
def test_register_rejects_surrounding_whitespace(
    name: str,
) -> None:
    registry = SubsystemRegistry()

    with pytest.raises(
        ValueError,
        match="surrounding whitespace",
    ):
        registry.register(RecordingSubsystem(name, []))


def test_register_rejects_duplicate_name() -> None:
    registry = SubsystemRegistry([RecordingSubsystem("world", [])])

    with pytest.raises(
        DuplicateSubsystemError,
        match="'world' is already registered",
    ):
        registry.register(RecordingSubsystem("world", []))


def test_start_runs_in_registration_order() -> None:
    registry, events = create_registry()

    registry.start()

    assert registry.state is SubsystemRegistryState.STARTED
    assert registry.started_subsystem_names == (
        "input",
        "world",
        "render",
    )
    assert events == [
        "input:start",
        "world:start",
        "render:start",
    ]


def test_empty_registry_can_start_and_shutdown() -> None:
    registry = SubsystemRegistry()

    registry.start()
    assert registry.state is SubsystemRegistryState.STARTED

    registry.shutdown()
    assert registry.state is SubsystemRegistryState.STOPPED


@pytest.mark.parametrize(
    "operation",
    ["register", "start"],
)
def test_created_only_operations_reject_started_registry(
    operation: str,
) -> None:
    registry = SubsystemRegistry()
    registry.start()

    with pytest.raises(
        SubsystemRegistryStateError,
        match="registry is started",
    ):
        if operation == "register":
            registry.register(RecordingSubsystem("world", []))
        else:
            registry.start()


def test_update_runs_in_registration_order() -> None:
    registry, events = create_registry()
    registry.start()
    events.clear()

    registry.update(1.0 / 60.0)

    assert events == [
        f"input:update:{1.0 / 60.0}",
        f"world:update:{1.0 / 60.0}",
        f"render:update:{1.0 / 60.0}",
    ]


def test_render_runs_in_registration_order() -> None:
    registry, events = create_registry()
    registry.start()
    events.clear()

    registry.render(0.25)

    assert events == [
        "input:render:0.25",
        "world:render:0.25",
        "render:render:0.25",
    ]


@pytest.mark.parametrize("operation", ["update", "render"])
@pytest.mark.parametrize(
    "state",
    [
        SubsystemRegistryState.CREATED,
        SubsystemRegistryState.STOPPED,
    ],
)
def test_execution_requires_started_state(
    operation: str,
    state: SubsystemRegistryState,
) -> None:
    registry = SubsystemRegistry()

    if state is SubsystemRegistryState.STOPPED:
        registry.shutdown()

    with pytest.raises(
        SubsystemRegistryStateError,
        match=f"registry is {state.value}",
    ):
        if operation == "update":
            registry.update(0.1)
        else:
            registry.render(0.5)


@pytest.mark.parametrize("value", [True, 1, "0.1"])
def test_update_rejects_invalid_delta_type(
    value: object,
) -> None:
    registry = SubsystemRegistry()
    registry.start()

    with pytest.raises(
        TypeError,
        match="fixed_delta_seconds",
    ):
        registry.update(cast(Any, value))


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf],
)
def test_update_rejects_non_finite_delta(value: float) -> None:
    registry = SubsystemRegistry()
    registry.start()

    with pytest.raises(ValueError, match="must be finite"):
        registry.update(value)


@pytest.mark.parametrize("value", [0.0, -0.1])
def test_update_rejects_non_positive_delta(
    value: float,
) -> None:
    registry = SubsystemRegistry()
    registry.start()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        registry.update(value)


@pytest.mark.parametrize("value", [True, 0, "0.5"])
def test_render_rejects_invalid_alpha_type(
    value: object,
) -> None:
    registry = SubsystemRegistry()
    registry.start()

    with pytest.raises(
        TypeError,
        match="interpolation_alpha",
    ):
        registry.render(cast(Any, value))


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf],
)
def test_render_rejects_non_finite_alpha(
    value: float,
) -> None:
    registry = SubsystemRegistry()
    registry.start()

    with pytest.raises(ValueError, match="must be finite"):
        registry.render(value)


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.5])
def test_render_rejects_out_of_range_alpha(
    value: float,
) -> None:
    registry = SubsystemRegistry()
    registry.start()

    with pytest.raises(ValueError, match="less than one"):
        registry.render(value)


def test_start_failure_rolls_back_started_subsystems() -> None:
    registry, events = create_registry(failures={"render": {"start"}})

    with pytest.raises(
        SubsystemExecutionError,
        match="'render' failed during start",
    ) as error:
        registry.start()

    assert error.value.operation == "start"
    assert error.value.subsystem_name == "render"
    assert isinstance(error.value.cause, RuntimeError)
    assert error.value.cleanup_failures == ()
    assert registry.state is SubsystemRegistryState.FAILED
    assert registry.started_subsystem_names == ()
    assert events == [
        "input:start",
        "world:start",
        "render:start",
        "world:stop",
        "input:stop",
    ]


def test_start_failure_reports_rollback_failures() -> None:
    registry, events = create_registry(
        failures={
            "input": {"stop"},
            "world": {"stop"},
            "render": {"start"},
        }
    )

    with pytest.raises(
        SubsystemExecutionError,
        match="Cleanup also failed for: world, input",
    ) as error:
        registry.start()

    assert [failure.subsystem_name for failure in error.value.cleanup_failures] == [
        "world",
        "input",
    ]
    assert all(isinstance(failure.error, RuntimeError) for failure in error.value.cleanup_failures)
    assert events[-2:] == ["world:stop", "input:stop"]


def test_update_failure_marks_registry_failed() -> None:
    registry, events = create_registry(failures={"world": {"update"}})
    registry.start()
    events.clear()

    with pytest.raises(
        SubsystemExecutionError,
        match="'world' failed during update",
    ) as error:
        registry.update(0.1)

    assert error.value.operation == "update"
    assert error.value.cleanup_failures == ()
    assert registry.state is SubsystemRegistryState.FAILED
    assert events == [
        "input:update:0.1",
        "world:update:0.1",
    ]


def test_render_failure_marks_registry_failed() -> None:
    registry, events = create_registry(failures={"world": {"render"}})
    registry.start()
    events.clear()

    with pytest.raises(
        SubsystemExecutionError,
        match="'world' failed during render",
    ) as error:
        registry.render(0.5)

    assert error.value.operation == "render"
    assert registry.state is SubsystemRegistryState.FAILED
    assert events == [
        "input:render:0.5",
        "world:render:0.5",
    ]


def test_shutdown_stops_in_reverse_startup_order() -> None:
    registry, events = create_registry()
    registry.start()
    events.clear()

    registry.shutdown()

    assert registry.state is SubsystemRegistryState.STOPPED
    assert registry.started_subsystem_names == ()
    assert events == [
        "render:stop",
        "world:stop",
        "input:stop",
    ]


def test_shutdown_is_idempotent() -> None:
    registry, events = create_registry()
    registry.start()
    events.clear()

    registry.shutdown()
    registry.shutdown()

    assert events == [
        "render:stop",
        "world:stop",
        "input:stop",
    ]


def test_shutdown_from_created_stops_without_callbacks() -> None:
    registry, events = create_registry()

    registry.shutdown()

    assert registry.state is SubsystemRegistryState.STOPPED
    assert events == []


def test_shutdown_after_execution_failure_cleans_up() -> None:
    registry, events = create_registry(failures={"world": {"update"}})
    registry.start()
    events.clear()

    with pytest.raises(SubsystemExecutionError):
        registry.update(0.1)

    registry.shutdown()

    assert registry.state is SubsystemRegistryState.STOPPED
    assert events[-3:] == [
        "render:stop",
        "world:stop",
        "input:stop",
    ]


def test_shutdown_collects_all_failures() -> None:
    registry, events = create_registry(
        failures={
            "input": {"stop"},
            "render": {"stop"},
        }
    )
    registry.start()
    events.clear()

    with pytest.raises(
        SubsystemShutdownError,
        match="render, input",
    ) as error:
        registry.shutdown()

    assert [failure.subsystem_name for failure in error.value.failures] == ["render", "input"]
    assert registry.state is SubsystemRegistryState.FAILED
    assert registry.started_subsystem_names == ()
    assert events == [
        "render:stop",
        "world:stop",
        "input:stop",
    ]

    registry.shutdown()
    assert registry.state is SubsystemRegistryState.STOPPED


def test_failed_registry_rejects_update_and_render() -> None:
    registry, _ = create_registry(failures={"world": {"update"}})
    registry.start()

    with pytest.raises(SubsystemExecutionError):
        registry.update(0.1)

    with pytest.raises(
        SubsystemRegistryStateError,
        match="registry is failed",
    ):
        registry.update(0.1)

    with pytest.raises(
        SubsystemRegistryStateError,
        match="registry is failed",
    ):
        registry.render(0.5)
