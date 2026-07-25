"""Tests for deterministic world time and calendar values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from open_world_rpg.world import (
    WorldClock,
    WorldClockSnapshot,
    WorldDateTime,
    WorldInstant,
    WorldTimeConfig,
)


def test_default_time_configuration_and_derived_units() -> None:
    config = WorldTimeConfig()

    assert config.ticks_per_second == 60
    assert config.seconds_per_minute == 60
    assert config.minutes_per_hour == 60
    assert config.hours_per_day == 24
    assert config.days_per_year == 365
    assert config.ticks_per_minute == 3_600
    assert config.ticks_per_hour == 216_000
    assert config.ticks_per_day == 5_184_000
    assert config.ticks_per_year == 1_892_160_000


def test_custom_time_configuration_and_derived_units() -> None:
    config = WorldTimeConfig(
        ticks_per_second=4,
        seconds_per_minute=5,
        minutes_per_hour=6,
        hours_per_day=7,
        days_per_year=8,
    )

    assert config.ticks_per_minute == 20
    assert config.ticks_per_hour == 120
    assert config.ticks_per_day == 840
    assert config.ticks_per_year == 6_720


@pytest.mark.parametrize(
    "field_name",
    [
        "ticks_per_second",
        "seconds_per_minute",
        "minutes_per_hour",
        "hours_per_day",
        "days_per_year",
    ],
)
@pytest.mark.parametrize("value", [True, False, 1.5, "1", None])
def test_time_configuration_rejects_non_integer_values(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, Any] = {
        "ticks_per_second": 1,
        "seconds_per_minute": 1,
        "minutes_per_hour": 1,
        "hours_per_day": 1,
        "days_per_year": 1,
    }
    values[field_name] = value

    with pytest.raises(TypeError, match=rf"{field_name} must be an integer"):
        WorldTimeConfig(**values)


@pytest.mark.parametrize(
    "field_name",
    [
        "ticks_per_second",
        "seconds_per_minute",
        "minutes_per_hour",
        "hours_per_day",
        "days_per_year",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_time_configuration_rejects_non_positive_values(
    field_name: str,
    value: int,
) -> None:
    values = {
        "ticks_per_second": 1,
        "seconds_per_minute": 1,
        "minutes_per_hour": 1,
        "hours_per_day": 1,
        "days_per_year": 1,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=rf"{field_name} must be greater than zero"):
        WorldTimeConfig(**values)


def test_zero_instant_and_advancement_are_immutable() -> None:
    instant = WorldInstant()

    assert instant.tick == 0
    assert instant.advance(0) == instant

    advanced = instant.advance(25)

    assert advanced == WorldInstant(tick=25)
    assert advanced is not instant
    assert instant.tick == 0

    with pytest.raises(FrozenInstanceError):
        instant.tick = 1  # type: ignore[misc]


@pytest.mark.parametrize("value", [True, False, 1.5, "1", None])
def test_world_instant_rejects_non_integer_tick(value: object) -> None:
    with pytest.raises(TypeError, match="tick must be an integer"):
        WorldInstant(tick=cast(Any, value))


def test_world_instant_rejects_negative_tick() -> None:
    with pytest.raises(ValueError, match="tick must be greater than or equal to zero"):
        WorldInstant(tick=-1)


@pytest.mark.parametrize("value", [True, False, 1.5, "1", None])
def test_world_instant_advance_rejects_non_integer_ticks(value: object) -> None:
    with pytest.raises(TypeError, match="ticks must be an integer"):
        WorldInstant().advance(cast(Any, value))


def test_world_instant_advance_rejects_negative_ticks() -> None:
    with pytest.raises(ValueError, match="ticks must be greater than or equal to zero"):
        WorldInstant().advance(-1)


@pytest.mark.parametrize(
    ("tick", "expected"),
    [
        (0, (1, 1, 0, 0, 0, 0)),
        (59, (1, 1, 0, 0, 0, 59)),
        (60, (1, 1, 0, 0, 1, 0)),
        (3_599, (1, 1, 0, 0, 59, 59)),
        (3_600, (1, 1, 0, 1, 0, 0)),
        (215_999, (1, 1, 0, 59, 59, 59)),
        (216_000, (1, 1, 1, 0, 0, 0)),
        (5_183_999, (1, 1, 23, 59, 59, 59)),
        (5_184_000, (1, 2, 0, 0, 0, 0)),
        (1_892_159_999, (1, 365, 23, 59, 59, 59)),
        (1_892_160_000, (2, 1, 0, 0, 0, 0)),
        (5_676_480_000, (4, 1, 0, 0, 0, 0)),
    ],
)
def test_default_calendar_boundaries(
    tick: int,
    expected: tuple[int, int, int, int, int, int],
) -> None:
    date_time = WorldDateTime.from_instant(
        instant=WorldInstant(tick=tick),
        config=WorldTimeConfig(),
    )

    assert (
        date_time.year,
        date_time.day_of_year,
        date_time.hour,
        date_time.minute,
        date_time.second,
        date_time.tick_within_second,
    ) == expected
    assert date_time.to_instant() == WorldInstant(tick=tick)


def test_custom_calendar_multiple_year_conversion_and_round_trip() -> None:
    config = WorldTimeConfig(
        ticks_per_second=3,
        seconds_per_minute=4,
        minutes_per_hour=5,
        hours_per_day=6,
        days_per_year=7,
    )
    tick = (config.ticks_per_year * 12) + (config.ticks_per_day * 5) + 359
    instant = WorldInstant(tick=tick)

    date_time = WorldDateTime.from_instant(instant=instant, config=config)

    assert date_time.year == 13
    assert date_time.day_of_year == 6
    assert date_time.hour == 5
    assert date_time.minute == 4
    assert date_time.second == 3
    assert date_time.tick_within_second == 2
    assert date_time.to_instant() is instant


def test_extremely_large_tick_round_trip_has_no_float_drift() -> None:
    tick = (10**100) + 12_345
    instant = WorldInstant(tick=tick)
    date_time = WorldDateTime(
        instant=instant,
        config=WorldTimeConfig(),
    )

    assert date_time.to_instant().tick == tick
    assert isinstance(date_time.year, int)
    assert isinstance(date_time.tick_within_second, int)


@pytest.mark.parametrize(
    ("instant", "config", "message"),
    [
        (object(), WorldTimeConfig(), "instant must be a WorldInstant"),
        (WorldInstant(), object(), "config must be a WorldTimeConfig"),
    ],
)
def test_world_datetime_rejects_invalid_dependencies(
    instant: object,
    config: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        WorldDateTime(
            instant=cast(Any, instant),
            config=cast(Any, config),
        )


def test_default_clock_exposes_date_time_and_snapshot() -> None:
    clock = WorldClock()

    assert clock.current == WorldInstant()
    assert clock.config == WorldTimeConfig()
    assert clock.date_time.year == 1
    assert clock.date_time.day_of_year == 1

    snapshot = clock.snapshot()

    assert snapshot == WorldClockSnapshot(
        config=clock.config,
        instant=clock.current,
    )
    assert snapshot.date_time == clock.date_time


def test_clock_advances_ticks_and_seconds_without_mutating_original() -> None:
    config = WorldTimeConfig(ticks_per_second=7)
    original = WorldClock(config=config, current=WorldInstant(tick=3))

    by_ticks = original.advance_ticks(11)
    by_seconds = by_ticks.advance_seconds(13)

    assert original.current.tick == 3
    assert by_ticks.current.tick == 14
    assert by_seconds.current.tick == 105
    assert by_seconds.config is config


def test_repeated_second_advancement_has_no_floating_point_drift() -> None:
    clock = WorldClock(config=WorldTimeConfig(ticks_per_second=7))

    for _ in range(10_000):
        clock = clock.advance_seconds(1)

    assert clock.current.tick == 70_000
    assert clock.date_time.tick_within_second == 0


@pytest.mark.parametrize("value", [True, False, 1.5, "1", None])
def test_clock_advance_seconds_rejects_non_integer_values(value: object) -> None:
    with pytest.raises(TypeError, match="seconds must be an integer"):
        WorldClock().advance_seconds(cast(Any, value))


def test_clock_advance_seconds_rejects_negative_value() -> None:
    with pytest.raises(ValueError, match="seconds must be greater than or equal to zero"):
        WorldClock().advance_seconds(-1)


def test_clock_reset_is_explicit_validated_and_immutable() -> None:
    original = WorldClock(current=WorldInstant(tick=500))

    zeroed = original.reset()
    repositioned = original.reset(instant=WorldInstant(tick=75))

    assert zeroed.current == WorldInstant()
    assert repositioned.current == WorldInstant(tick=75)
    assert original.current == WorldInstant(tick=500)

    with pytest.raises(TypeError, match="instant must be a WorldInstant"):
        original.reset(instant=cast(Any, 10))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: WorldClock(config=cast(Any, object())),
            "config must be a WorldTimeConfig",
        ),
        (
            lambda: WorldClock(current=cast(Any, object())),
            "current must be a WorldInstant",
        ),
        (
            lambda: WorldClockSnapshot(
                config=cast(Any, object()),
                instant=WorldInstant(),
            ),
            "config must be a WorldTimeConfig",
        ),
        (
            lambda: WorldClockSnapshot(
                config=WorldTimeConfig(),
                instant=cast(Any, object()),
            ),
            "instant must be a WorldInstant",
        ),
    ],
)
def test_clock_models_reject_invalid_dependencies(
    factory: Any,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        factory()


def test_clock_and_snapshot_are_immutable() -> None:
    clock = WorldClock()
    snapshot = clock.snapshot()

    with pytest.raises(FrozenInstanceError):
        clock.current = WorldInstant(tick=1)  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        snapshot.instant = WorldInstant(tick=1)  # type: ignore[misc]
