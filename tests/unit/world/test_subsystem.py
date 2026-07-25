"""Tests for the concrete world engine subsystem."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from open_world_rpg.core import (
    GameConfig,
    ProjectPaths,
    RuntimeEnvironment,
    SimulationConfig,
)
from open_world_rpg.engine import (
    EngineServiceRegistration,
    EventBus,
    create_engine_context,
)
from open_world_rpg.world import (
    WorldClock,
    WorldId,
    WorldInstant,
    WorldMetadata,
    WorldModel,
    WorldRuntime,
    WorldSeed,
    WorldSpecification,
    WorldState,
    WorldSubsystem,
    WorldSubsystemConfigurationError,
    WorldSubsystemError,
    WorldSubsystemStateError,
    WorldTimeConfig,
)

CREATED_AT = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)


def create_config(
    tmp_path: Path,
    *,
    seed: int = 42,
    tick_rate: int = 60,
) -> GameConfig:
    return GameConfig(
        environment=RuntimeEnvironment.TEST,
        simulation=SimulationConfig(
            world_seed=seed,
            tick_rate=tick_rate,
        ),
        paths=ProjectPaths.from_project_root(tmp_path),
    )


def create_runtime(
    *,
    state: WorldState = WorldState.CREATED,
    seed: int = 42,
    ticks_per_second: int = 60,
    tick: int = 0,
) -> WorldRuntime:
    specification = WorldSpecification(
        name="Subsystem World",
        seed=WorldSeed(value=seed),
        time_config=WorldTimeConfig(ticks_per_second=ticks_per_second),
    )
    return WorldRuntime(
        model=WorldModel(
            metadata=WorldMetadata(
                world_id=WorldId.create(),
                name=specification.name,
                seed=specification.seed.value,
                created_at=CREATED_AT,
                state=state,
            ),
            specification=specification,
            clock=WorldClock(
                config=specification.time_config,
                current=WorldInstant(tick=tick),
            ),
        )
    )


def bind_subsystem(
    subsystem: WorldSubsystem,
    *,
    runtime: WorldRuntime,
    config: GameConfig,
) -> None:
    event_bus = EventBus()
    context = create_engine_context(
        logger=logging.Logger("test.world.subsystem"),
        event_bus=event_bus,
        registrations=(
            EngineServiceRegistration(WorldRuntime, runtime),
            EngineServiceRegistration(GameConfig, config),
        ),
    )
    subsystem.bind_services(context)


def test_subsystem_has_stable_identity_and_requires_bound_services() -> None:
    subsystem = WorldSubsystem()

    assert subsystem.name == "world"
    assert subsystem.dependencies == ()
    assert subsystem.started is False
    assert issubclass(WorldSubsystemConfigurationError, WorldSubsystemError)
    assert issubclass(WorldSubsystemStateError, WorldSubsystemError)

    with pytest.raises(WorldSubsystemStateError, match="services are not bound"):
        _ = subsystem.runtime

    with pytest.raises(WorldSubsystemStateError, match="services are not bound"):
        _ = subsystem.config


def test_subsystem_binds_required_services(tmp_path: Path) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime()
    config = create_config(tmp_path)

    bind_subsystem(subsystem, runtime=runtime, config=config)

    assert subsystem.runtime is runtime
    assert subsystem.config is config
    assert subsystem.services_bound is True


@pytest.mark.parametrize(
    ("state", "expected_revision"),
    [
        (WorldState.CREATED, 2),
        (WorldState.INITIALISED, 1),
        (WorldState.ACTIVE, 0),
        (WorldState.PAUSED, 1),
    ],
)
def test_subsystem_maps_supported_startup_states(
    tmp_path: Path,
    state: WorldState,
    expected_revision: int,
) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime(state=state)
    bind_subsystem(
        subsystem,
        runtime=runtime,
        config=create_config(tmp_path),
    )

    subsystem.start()

    assert subsystem.started is True
    assert runtime.model.metadata.state is WorldState.ACTIVE
    assert runtime.revision == expected_revision


@pytest.mark.parametrize("state", [WorldState.CLOSED, WorldState.FAILED])
def test_subsystem_rejects_terminal_world_startup(
    tmp_path: Path,
    state: WorldState,
) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime(state=state)
    bind_subsystem(
        subsystem,
        runtime=runtime,
        config=create_config(tmp_path),
    )

    with pytest.raises(WorldSubsystemStateError, match="Cannot start"):
        subsystem.start()

    assert subsystem.started is False
    assert runtime.model.metadata.state is state
    assert runtime.revision == 0


def test_subsystem_rejects_seed_mismatch(tmp_path: Path) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime(seed=41)
    bind_subsystem(
        subsystem,
        runtime=runtime,
        config=create_config(tmp_path, seed=42),
    )

    with pytest.raises(WorldSubsystemConfigurationError, match="seed must match"):
        subsystem.start()

    assert runtime.revision == 0


def test_subsystem_rejects_tick_rate_mismatch(tmp_path: Path) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime(ticks_per_second=59)
    bind_subsystem(
        subsystem,
        runtime=runtime,
        config=create_config(tmp_path, tick_rate=60),
    )

    with pytest.raises(WorldSubsystemConfigurationError, match="ticks_per_second"):
        subsystem.start()

    assert runtime.revision == 0


def test_each_fixed_update_advances_exactly_one_tick(tmp_path: Path) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime(state=WorldState.ACTIVE, tick=100)
    bind_subsystem(
        subsystem,
        runtime=runtime,
        config=create_config(tmp_path),
    )
    subsystem.start()

    subsystem.update(0.000_001)
    subsystem.update(1000.0)

    assert runtime.model.clock.current.tick == 102
    assert runtime.revision == 2


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_update_rejects_non_positive_or_non_finite_delta(
    tmp_path: Path,
    value: float,
) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime(state=WorldState.ACTIVE)
    bind_subsystem(
        subsystem,
        runtime=runtime,
        config=create_config(tmp_path),
    )
    subsystem.start()

    with pytest.raises(ValueError, match="fixed_delta_seconds"):
        subsystem.update(value)

    assert runtime.model.clock.current.tick == 0


def test_update_rejects_invalid_delta_type_and_unstarted_state(
    tmp_path: Path,
) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime(state=WorldState.ACTIVE)
    bind_subsystem(
        subsystem,
        runtime=runtime,
        config=create_config(tmp_path),
    )

    with pytest.raises(WorldSubsystemStateError, match="not started"):
        subsystem.update(1.0)

    subsystem.start()
    with pytest.raises(TypeError, match="must be a float"):
        subsystem.update(1)  # type: ignore[arg-type]

    assert runtime.model.clock.current.tick == 0


def test_render_is_a_no_op(tmp_path: Path) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime(state=WorldState.ACTIVE, tick=5)
    bind_subsystem(
        subsystem,
        runtime=runtime,
        config=create_config(tmp_path),
    )

    subsystem.render(float("nan"))

    assert runtime.model.clock.current.tick == 5


def test_stop_pauses_active_world_and_repeated_stop_is_no_op(tmp_path: Path) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime(state=WorldState.ACTIVE)
    bind_subsystem(
        subsystem,
        runtime=runtime,
        config=create_config(tmp_path),
    )
    subsystem.start()

    subsystem.stop()
    revision_after_stop = runtime.revision
    subsystem.stop()

    assert subsystem.started is False
    assert runtime.model.metadata.state is WorldState.PAUSED
    assert revision_after_stop == 1
    assert runtime.revision == revision_after_stop


def test_stop_leaves_externally_paused_or_terminal_world_unchanged(
    tmp_path: Path,
) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime(state=WorldState.ACTIVE)
    bind_subsystem(
        subsystem,
        runtime=runtime,
        config=create_config(tmp_path),
    )
    subsystem.start()
    runtime.pause()
    revision = runtime.revision

    subsystem.stop()

    assert runtime.model.metadata.state is WorldState.PAUSED
    assert runtime.revision == revision


@pytest.mark.parametrize("terminal_operation", ["close", "fail"])
def test_stop_leaves_externally_terminal_world_unchanged(
    tmp_path: Path,
    terminal_operation: str,
) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime(state=WorldState.ACTIVE)
    bind_subsystem(
        subsystem,
        runtime=runtime,
        config=create_config(tmp_path),
    )
    subsystem.start()
    getattr(runtime, terminal_operation)()
    revision = runtime.revision

    subsystem.stop()

    assert runtime.model.metadata.state in {WorldState.CLOSED, WorldState.FAILED}
    assert runtime.revision == revision


def test_repeated_start_rejected_and_restart_after_stop_resumes(
    tmp_path: Path,
) -> None:
    subsystem = WorldSubsystem()
    runtime = create_runtime(state=WorldState.ACTIVE)
    bind_subsystem(
        subsystem,
        runtime=runtime,
        config=create_config(tmp_path),
    )
    subsystem.start()

    with pytest.raises(WorldSubsystemStateError, match="already started"):
        subsystem.start()

    subsystem.stop()
    subsystem.start()

    assert subsystem.started is True
    assert runtime.model.metadata.state is WorldState.ACTIVE
