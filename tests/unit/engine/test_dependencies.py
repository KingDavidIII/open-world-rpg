"""Tests for dependency-aware subsystem construction."""

from __future__ import annotations

from typing import Any, cast

import pytest

from open_world_rpg.engine import (
    DuplicateSubsystemError,
    EngineSubsystem,
    EngineSubsystemBase,
    MissingSubsystemDependencyError,
    SubsystemDependencyCycleError,
    SubsystemRegistry,
    resolve_subsystem_order,
)


class RecordingSubsystem:
    """Configurable dependency-aware subsystem test double."""

    def __init__(
        self,
        name: object,
        events: list[str],
        *,
        dependencies: object = (),
    ) -> None:
        self.name = cast(Any, name)
        self.dependencies = cast(Any, dependencies)
        self.events = events

    def start(self) -> None:
        self.events.append(f"{self.name}:start")

    def update(self, fixed_delta_seconds: float) -> None:
        self.events.append(f"{self.name}:update:{fixed_delta_seconds}")

    def render(self, interpolation_alpha: float) -> None:
        self.events.append(f"{self.name}:render:{interpolation_alpha}")

    def stop(self) -> None:
        self.events.append(f"{self.name}:stop")


class PlainSubsystem:
    """Subsystem without an explicit dependencies attribute."""

    def __init__(
        self,
        name: str,
        events: list[str],
    ) -> None:
        self.name = name
        self.events = events

    def start(self) -> None:
        self.events.append(f"{self.name}:start")

    def update(self, fixed_delta_seconds: float) -> None:
        self.events.append(f"{self.name}:update:{fixed_delta_seconds}")

    def render(self, interpolation_alpha: float) -> None:
        self.events.append(f"{self.name}:render:{interpolation_alpha}")

    def stop(self) -> None:
        self.events.append(f"{self.name}:stop")


class IncompleteSubsystem:
    """Object that does not implement EngineSubsystem."""

    name = "incomplete"


def subsystem_names(
    subsystems: tuple[EngineSubsystem, ...],
) -> tuple[str, ...]:
    return tuple(subsystem.name for subsystem in subsystems)


def test_base_subsystem_implements_engine_contract() -> None:
    subsystem = EngineSubsystemBase(name="world")

    assert isinstance(subsystem, EngineSubsystem)
    assert subsystem.name == "world"
    assert subsystem.dependencies == ()


def test_base_subsystem_accepts_dependencies() -> None:
    subsystem = EngineSubsystemBase(
        name="render",
        dependencies=["world", "camera"],
    )

    assert subsystem.dependencies == ("world", "camera")


def test_base_subsystem_lifecycle_defaults_are_no_ops() -> None:
    subsystem = EngineSubsystemBase(name="world")

    subsystem.start()
    subsystem.update(1.0 / 60.0)
    subsystem.render(0.5)
    subsystem.stop()


@pytest.mark.parametrize("name", [123, None, object()])
def test_base_rejects_non_string_name(
    name: object,
) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        EngineSubsystemBase(name=cast(Any, name))


@pytest.mark.parametrize("name", ["", " ", "\t"])
def test_base_rejects_empty_name(name: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        EngineSubsystemBase(name=name)


@pytest.mark.parametrize(
    "name",
    [" world", "world ", " world "],
)
def test_base_rejects_surrounding_name_whitespace(
    name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="surrounding whitespace",
    ):
        EngineSubsystemBase(name=name)


@pytest.mark.parametrize(
    "dependencies",
    ["world", b"world", 123, None],
)
def test_base_rejects_invalid_dependency_collection(
    dependencies: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="iterable of strings",
    ):
        EngineSubsystemBase(
            name="render",
            dependencies=cast(Any, dependencies),
        )


@pytest.mark.parametrize(
    "dependency",
    [123, None, object()],
)
def test_base_rejects_non_string_dependency(
    dependency: object,
) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        EngineSubsystemBase(
            name="render",
            dependencies=[cast(Any, dependency)],
        )


@pytest.mark.parametrize("dependency", ["", " ", "\t"])
def test_base_rejects_empty_dependency(
    dependency: str,
) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        EngineSubsystemBase(
            name="render",
            dependencies=[dependency],
        )


@pytest.mark.parametrize(
    "dependency",
    [" world", "world ", " world "],
)
def test_base_rejects_dependency_whitespace(
    dependency: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="surrounding whitespace",
    ):
        EngineSubsystemBase(
            name="render",
            dependencies=[dependency],
        )


def test_base_rejects_duplicate_dependency() -> None:
    with pytest.raises(
        ValueError,
        match="declared more than once",
    ):
        EngineSubsystemBase(
            name="render",
            dependencies=["world", "world"],
        )


def test_resolver_accepts_empty_collection() -> None:
    assert resolve_subsystem_order(()) == ()


def test_plain_subsystems_preserve_registration_order() -> None:
    events: list[str] = []
    input_system = PlainSubsystem("input", events)
    world_system = PlainSubsystem("world", events)
    render_system = PlainSubsystem("render", events)

    resolved = resolve_subsystem_order([input_system, world_system, render_system])

    assert resolved == (
        input_system,
        world_system,
        render_system,
    )


def test_resolver_orders_declared_dependencies() -> None:
    events: list[str] = []
    render_system = RecordingSubsystem(
        "render",
        events,
        dependencies=("world",),
    )
    input_system = RecordingSubsystem("input", events)
    world_system = RecordingSubsystem(
        "world",
        events,
        dependencies=("input",),
    )

    resolved = resolve_subsystem_order([render_system, input_system, world_system])

    assert subsystem_names(resolved) == (
        "input",
        "world",
        "render",
    )


def test_resolver_handles_transitive_dependencies() -> None:
    events: list[str] = []
    ui_system = RecordingSubsystem(
        "ui",
        events,
        dependencies=("render",),
    )
    render_system = RecordingSubsystem(
        "render",
        events,
        dependencies=("world",),
    )
    world_system = RecordingSubsystem(
        "world",
        events,
        dependencies=("physics",),
    )
    physics_system = RecordingSubsystem("physics", events)

    resolved = resolve_subsystem_order(
        [
            ui_system,
            render_system,
            world_system,
            physics_system,
        ]
    )

    assert subsystem_names(resolved) == (
        "physics",
        "world",
        "render",
        "ui",
    )


def test_resolver_preserves_unrelated_relative_order() -> None:
    events: list[str] = []
    audio_system = RecordingSubsystem("audio", events)
    render_system = RecordingSubsystem(
        "render",
        events,
        dependencies=("world",),
    )
    input_system = RecordingSubsystem("input", events)
    world_system = RecordingSubsystem("world", events)

    resolved = resolve_subsystem_order(
        [
            audio_system,
            render_system,
            input_system,
            world_system,
        ]
    )

    assert subsystem_names(resolved) == (
        "audio",
        "input",
        "world",
        "render",
    )


def test_resolver_accepts_generator_input() -> None:
    events: list[str] = []
    systems = (
        subsystem
        for subsystem in [
            RecordingSubsystem("input", events),
            RecordingSubsystem(
                "world",
                events,
                dependencies=("input",),
            ),
        ]
    )

    resolved = resolve_subsystem_order(systems)

    assert subsystem_names(resolved) == (
        "input",
        "world",
    )


def test_resolver_rejects_incomplete_subsystem() -> None:
    with pytest.raises(
        TypeError,
        match="implement EngineSubsystem",
    ):
        resolve_subsystem_order([cast(Any, IncompleteSubsystem())])


def test_resolver_rejects_invalid_subsystem_name() -> None:
    with pytest.raises(
        TypeError,
        match="subsystem name must be a string",
    ):
        resolve_subsystem_order([RecordingSubsystem(123, [])])


def test_resolver_rejects_duplicate_name() -> None:
    with pytest.raises(
        DuplicateSubsystemError,
        match="'world' is already registered",
    ):
        resolve_subsystem_order(
            [
                RecordingSubsystem("world", []),
                RecordingSubsystem("world", []),
            ]
        )


def test_resolver_reports_missing_dependency() -> None:
    subsystem = RecordingSubsystem(
        "render",
        [],
        dependencies=("world",),
    )

    with pytest.raises(
        MissingSubsystemDependencyError,
        match="'world'",
    ) as error:
        resolve_subsystem_order([subsystem])

    assert error.value.subsystem_name == "render"
    assert error.value.dependency_name == "world"


def test_resolver_reports_self_dependency_cycle() -> None:
    subsystem = RecordingSubsystem(
        "world",
        [],
        dependencies=("world",),
    )

    with pytest.raises(
        SubsystemDependencyCycleError,
        match="world -> world",
    ) as error:
        resolve_subsystem_order([subsystem])

    assert error.value.cycle == ("world", "world")


def test_resolver_reports_multi_subsystem_cycle() -> None:
    input_system = RecordingSubsystem(
        "input",
        [],
        dependencies=("render",),
    )
    render_system = RecordingSubsystem(
        "render",
        [],
        dependencies=("world",),
    )
    world_system = RecordingSubsystem(
        "world",
        [],
        dependencies=("input",),
    )

    with pytest.raises(
        SubsystemDependencyCycleError,
        match="input -> render -> world -> input",
    ) as error:
        resolve_subsystem_order([input_system, render_system, world_system])

    assert error.value.cycle == (
        "input",
        "render",
        "world",
        "input",
    )


def test_cycle_detection_excludes_blocked_non_cycle_node() -> None:
    observer = RecordingSubsystem(
        "observer",
        [],
        dependencies=("input",),
    )
    input_system = RecordingSubsystem(
        "input",
        [],
        dependencies=("render",),
    )
    render_system = RecordingSubsystem(
        "render",
        [],
        dependencies=("input",),
    )

    with pytest.raises(
        SubsystemDependencyCycleError,
    ) as error:
        resolve_subsystem_order([observer, input_system, render_system])

    assert error.value.cycle == (
        "input",
        "render",
        "input",
    )


def test_resolved_order_drives_registry_lifecycle() -> None:
    events: list[str] = []
    render_system = RecordingSubsystem(
        "render",
        events,
        dependencies=("world",),
    )
    input_system = RecordingSubsystem("input", events)
    world_system = RecordingSubsystem(
        "world",
        events,
        dependencies=("input",),
    )
    registry = SubsystemRegistry(
        resolve_subsystem_order([render_system, input_system, world_system])
    )

    registry.start()
    registry.update(0.1)
    registry.render(0.5)
    registry.shutdown()

    assert events == [
        "input:start",
        "world:start",
        "render:start",
        "input:update:0.1",
        "world:update:0.1",
        "render:update:0.1",
        "input:render:0.5",
        "world:render:0.5",
        "render:render:0.5",
        "render:stop",
        "world:stop",
        "input:stop",
    ]
