"""Tests for deterministic engine timing primitives."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest

from open_world_rpg.engine.timing import (
    NANOSECONDS_PER_SECOND,
    ClockRegressionError,
    EngineClock,
    FixedStepConfig,
    FixedStepScheduler,
    FrameSchedule,
    MonotonicClock,
)


def create_schedule(
    **overrides: object,
) -> FrameSchedule:
    values: dict[str, object] = {
        "frame_index": 0,
        "timestamp_ns": 100,
        "elapsed_ns": 50,
        "simulated_elapsed_ns": 50,
        "update_count": 1,
        "dropped_update_count": 0,
        "interpolation_alpha": 0.5,
    }
    values.update(overrides)
    return FrameSchedule(**cast(Any, values))


def create_scheduler(
    *,
    tick_rate_hz: int = 10,
    max_frame_duration_ns: int = NANOSECONDS_PER_SECOND,
    max_updates_per_frame: int = 8,
) -> FixedStepScheduler:
    return FixedStepScheduler(
        FixedStepConfig(
            tick_rate_hz=tick_rate_hz,
            max_frame_duration_ns=max_frame_duration_ns,
            max_updates_per_frame=max_updates_per_frame,
        )
    )


def test_monotonic_clock_implements_clock_contract() -> None:
    clock = MonotonicClock()

    first = clock.now_ns()
    second = clock.now_ns()

    assert isinstance(clock, EngineClock)
    assert type(first) is int
    assert first >= 0
    assert second >= first


def test_fixed_step_config_defaults() -> None:
    config = FixedStepConfig()

    assert config.tick_rate_hz == 60
    assert config.max_frame_duration_ns == 250_000_000
    assert config.max_updates_per_frame == 8
    assert config.fixed_step_seconds == pytest.approx(1.0 / 60.0)


@pytest.mark.parametrize(
    "value",
    [
        True,
        60.0,
        "60",
    ],
)
def test_fixed_step_config_rejects_invalid_tick_rate_type(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="tick_rate_hz"):
        FixedStepConfig(tick_rate_hz=cast(Any, value))


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_fixed_step_config_rejects_non_positive_tick_rate(
    value: int,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        FixedStepConfig(tick_rate_hz=value)


def test_fixed_step_config_rejects_unrepresentable_tick_rate() -> None:
    with pytest.raises(
        ValueError,
        match="nanosecond clock resolution",
    ):
        FixedStepConfig(tick_rate_hz=NANOSECONDS_PER_SECOND + 1)


@pytest.mark.parametrize(
    "value",
    [
        True,
        250.0,
        "250",
    ],
)
def test_fixed_step_config_rejects_invalid_frame_duration_type(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="max_frame_duration_ns"):
        FixedStepConfig(max_frame_duration_ns=cast(Any, value))


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_fixed_step_config_rejects_non_positive_frame_duration(
    value: int,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        FixedStepConfig(max_frame_duration_ns=value)


@pytest.mark.parametrize(
    "value",
    [
        True,
        8.0,
        "8",
    ],
)
def test_fixed_step_config_rejects_invalid_update_limit_type(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="max_updates_per_frame"):
        FixedStepConfig(max_updates_per_frame=cast(Any, value))


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_fixed_step_config_rejects_non_positive_update_limit(
    value: int,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        FixedStepConfig(max_updates_per_frame=value)


def test_frame_schedule_accepts_valid_values() -> None:
    schedule = create_schedule()

    assert schedule.frame_index == 0
    assert schedule.timestamp_ns == 100
    assert schedule.elapsed_ns == 50
    assert schedule.simulated_elapsed_ns == 50
    assert schedule.update_count == 1
    assert schedule.dropped_update_count == 0
    assert schedule.interpolation_alpha == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame_index", True),
        ("timestamp_ns", 1.0),
        ("elapsed_ns", "50"),
        ("simulated_elapsed_ns", None),
        ("update_count", 1.0),
        ("dropped_update_count", False),
    ],
)
def test_frame_schedule_rejects_invalid_integer_types(
    field: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match=field):
        create_schedule(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "frame_index",
        "timestamp_ns",
        "elapsed_ns",
        "simulated_elapsed_ns",
        "update_count",
        "dropped_update_count",
    ],
)
def test_frame_schedule_rejects_negative_integer_values(
    field: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to zero",
    ):
        create_schedule(**{field: -1})


def test_frame_schedule_rejects_excess_simulated_duration() -> None:
    with pytest.raises(
        ValueError,
        match="cannot exceed elapsed_ns",
    ):
        create_schedule(
            elapsed_ns=49,
            simulated_elapsed_ns=50,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        True,
        "0.5",
    ],
)
def test_frame_schedule_rejects_invalid_alpha_type(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="interpolation_alpha"):
        create_schedule(
            interpolation_alpha=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_frame_schedule_rejects_non_finite_alpha(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        create_schedule(interpolation_alpha=value)


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.0,
        1.5,
    ],
)
def test_frame_schedule_rejects_out_of_range_alpha(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="less than one",
    ):
        create_schedule(interpolation_alpha=value)


def test_scheduler_uses_default_configuration() -> None:
    scheduler = FixedStepScheduler()

    assert scheduler.config == FixedStepConfig()
    assert scheduler.started is False
    assert scheduler.frame_count == 0
    assert scheduler.last_timestamp_ns is None
    assert scheduler.interpolation_alpha == 0.0


def test_scheduler_rejects_invalid_configuration() -> None:
    with pytest.raises(TypeError, match="config"):
        FixedStepScheduler(cast(Any, object()))


def test_first_sample_initialises_scheduler() -> None:
    scheduler = create_scheduler()

    schedule = scheduler.advance(500)

    assert schedule == FrameSchedule(
        frame_index=0,
        timestamp_ns=500,
        elapsed_ns=0,
        simulated_elapsed_ns=0,
        update_count=0,
        dropped_update_count=0,
        interpolation_alpha=0.0,
    )
    assert scheduler.started is True
    assert scheduler.frame_count == 1
    assert scheduler.last_timestamp_ns == 500


def test_scheduler_produces_exact_fixed_update() -> None:
    scheduler = create_scheduler(tick_rate_hz=10)
    scheduler.advance(0)

    schedule = scheduler.advance(100_000_000)

    assert schedule.update_count == 1
    assert schedule.dropped_update_count == 0
    assert schedule.interpolation_alpha == 0.0


def test_scheduler_accumulates_fractional_frame_time() -> None:
    scheduler = create_scheduler(tick_rate_hz=10)
    scheduler.advance(0)

    first = scheduler.advance(25_000_000)
    second = scheduler.advance(50_000_000)
    third = scheduler.advance(100_000_000)

    assert first.update_count == 0
    assert first.interpolation_alpha == pytest.approx(0.25)

    assert second.update_count == 0
    assert second.interpolation_alpha == pytest.approx(0.5)

    assert third.update_count == 1
    assert third.interpolation_alpha == 0.0


def test_scheduler_produces_multiple_updates() -> None:
    scheduler = create_scheduler(tick_rate_hz=10)
    scheduler.advance(0)

    schedule = scheduler.advance(350_000_000)

    assert schedule.update_count == 3
    assert schedule.dropped_update_count == 0
    assert schedule.interpolation_alpha == pytest.approx(0.5)


def test_scheduler_clamps_long_frame_duration() -> None:
    scheduler = create_scheduler(
        tick_rate_hz=10,
        max_frame_duration_ns=250_000_000,
    )
    scheduler.advance(0)

    schedule = scheduler.advance(NANOSECONDS_PER_SECOND)

    assert schedule.elapsed_ns == NANOSECONDS_PER_SECOND
    assert schedule.simulated_elapsed_ns == 250_000_000
    assert schedule.update_count == 2
    assert schedule.interpolation_alpha == pytest.approx(0.5)


def test_scheduler_drops_updates_above_frame_limit() -> None:
    scheduler = create_scheduler(
        tick_rate_hz=10,
        max_updates_per_frame=3,
    )
    scheduler.advance(0)

    schedule = scheduler.advance(NANOSECONDS_PER_SECOND)

    assert schedule.update_count == 3
    assert schedule.dropped_update_count == 7
    assert schedule.interpolation_alpha == 0.0


def test_dropped_updates_do_not_create_permanent_backlog() -> None:
    scheduler = create_scheduler(
        tick_rate_hz=10,
        max_updates_per_frame=2,
    )
    scheduler.advance(0)

    overloaded = scheduler.advance(NANOSECONDS_PER_SECOND)
    next_frame = scheduler.advance(NANOSECONDS_PER_SECOND + 50_000_000)

    assert overloaded.update_count == 2
    assert overloaded.dropped_update_count == 8
    assert next_frame.update_count == 0
    assert next_frame.dropped_update_count == 0
    assert next_frame.interpolation_alpha == pytest.approx(0.5)


def test_scheduler_accepts_identical_timestamp() -> None:
    scheduler = create_scheduler()
    scheduler.advance(100)

    schedule = scheduler.advance(100)

    assert schedule.elapsed_ns == 0
    assert schedule.simulated_elapsed_ns == 0
    assert schedule.update_count == 0


def test_scheduler_rejects_clock_regression_without_mutation() -> None:
    scheduler = create_scheduler()
    scheduler.advance(100)

    with pytest.raises(
        ClockRegressionError,
        match="previous=100, current=99",
    ):
        scheduler.advance(99)

    assert scheduler.last_timestamp_ns == 100
    assert scheduler.frame_count == 1

    recovered = scheduler.advance(101)

    assert recovered.frame_index == 1
    assert recovered.elapsed_ns == 1


@pytest.mark.parametrize(
    "value",
    [
        True,
        100.0,
        "100",
    ],
)
def test_scheduler_rejects_invalid_timestamp_type(
    value: object,
) -> None:
    scheduler = create_scheduler()

    with pytest.raises(TypeError, match="timestamp_ns"):
        scheduler.advance(cast(Any, value))


def test_scheduler_rejects_negative_timestamp() -> None:
    scheduler = create_scheduler()

    with pytest.raises(
        ValueError,
        match="greater than or equal to zero",
    ):
        scheduler.advance(-1)


def test_scheduler_reset_clears_all_accumulated_state() -> None:
    scheduler = create_scheduler(tick_rate_hz=10)
    scheduler.advance(100)
    scheduler.advance(150_000_100)

    assert scheduler.started is True
    assert scheduler.frame_count == 2
    assert scheduler.interpolation_alpha == pytest.approx(0.5)

    scheduler.reset()

    assert scheduler.started is False
    assert scheduler.frame_count == 0
    assert scheduler.last_timestamp_ns is None
    assert scheduler.interpolation_alpha == 0.0

    restarted = scheduler.advance(900)

    assert restarted.frame_index == 0
    assert restarted.elapsed_ns == 0
