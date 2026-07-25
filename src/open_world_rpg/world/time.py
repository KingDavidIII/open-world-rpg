"""Deterministic integer-based world calendar and clock values.

Public calendar indexing follows familiar display conventions: ``year`` and
``day_of_year`` are one-based. ``hour``, ``minute``, ``second``, and
``tick_within_second`` are zero-based. Absolute ticks are also zero-based, with
tick zero representing year 1, day 1, 00:00:00 at tick zero of that second.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


def _require_positive_integer(*, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _require_non_negative_integer(*, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero.")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldTimeConfig:
    """Integer unit sizes for the deterministic world calendar."""

    ticks_per_second: int = 60
    seconds_per_minute: int = 60
    minutes_per_hour: int = 60
    hours_per_day: int = 24
    days_per_year: int = 365

    def __post_init__(self) -> None:
        _require_positive_integer(
            name="ticks_per_second",
            value=self.ticks_per_second,
        )
        _require_positive_integer(
            name="seconds_per_minute",
            value=self.seconds_per_minute,
        )
        _require_positive_integer(
            name="minutes_per_hour",
            value=self.minutes_per_hour,
        )
        _require_positive_integer(
            name="hours_per_day",
            value=self.hours_per_day,
        )
        _require_positive_integer(
            name="days_per_year",
            value=self.days_per_year,
        )

    @property
    def ticks_per_minute(self) -> int:
        """Return the exact number of ticks in one world minute."""
        return self.ticks_per_second * self.seconds_per_minute

    @property
    def ticks_per_hour(self) -> int:
        """Return the exact number of ticks in one world hour."""
        return self.ticks_per_minute * self.minutes_per_hour

    @property
    def ticks_per_day(self) -> int:
        """Return the exact number of ticks in one world day."""
        return self.ticks_per_hour * self.hours_per_day

    @property
    def ticks_per_year(self) -> int:
        """Return the exact number of ticks in one world year."""
        return self.ticks_per_day * self.days_per_year


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldInstant:
    """Absolute non-negative position on the world timeline."""

    tick: int = 0

    def __post_init__(self) -> None:
        _require_non_negative_integer(name="tick", value=self.tick)

    def advance(self, ticks: int) -> WorldInstant:
        """Return a new instant advanced by a non-negative tick count."""
        _require_non_negative_integer(name="ticks", value=ticks)
        return WorldInstant(tick=self.tick + ticks)


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldDateTime:
    """Calendar projection of an absolute world instant.

    Years and days of year are one-based. All time-of-day fields and
    ``tick_within_second`` are zero-based.
    """

    instant: WorldInstant
    config: WorldTimeConfig

    def __post_init__(self) -> None:
        if not isinstance(self.instant, WorldInstant):
            raise TypeError("instant must be a WorldInstant.")

        if not isinstance(self.config, WorldTimeConfig):
            raise TypeError("config must be a WorldTimeConfig.")

    @classmethod
    def from_instant(
        cls,
        *,
        instant: WorldInstant,
        config: WorldTimeConfig,
    ) -> WorldDateTime:
        """Project an absolute instant into the configured calendar."""
        return cls(instant=instant, config=config)

    @property
    def year(self) -> int:
        """Return the one-based world year."""
        return (self.instant.tick // self.config.ticks_per_year) + 1

    @property
    def day_of_year(self) -> int:
        """Return the one-based day within the current world year."""
        ticks_within_year = self.instant.tick % self.config.ticks_per_year
        return (ticks_within_year // self.config.ticks_per_day) + 1

    @property
    def hour(self) -> int:
        """Return the zero-based hour within the current world day."""
        ticks_within_day = self.instant.tick % self.config.ticks_per_day
        return ticks_within_day // self.config.ticks_per_hour

    @property
    def minute(self) -> int:
        """Return the zero-based minute within the current world hour."""
        ticks_within_hour = self.instant.tick % self.config.ticks_per_hour
        return ticks_within_hour // self.config.ticks_per_minute

    @property
    def second(self) -> int:
        """Return the zero-based second within the current world minute."""
        ticks_within_minute = self.instant.tick % self.config.ticks_per_minute
        return ticks_within_minute // self.config.ticks_per_second

    @property
    def tick_within_second(self) -> int:
        """Return the zero-based tick within the current world second."""
        return self.instant.tick % self.config.ticks_per_second

    def to_instant(self) -> WorldInstant:
        """Return the exact absolute instant represented by this projection."""
        return self.instant


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldClockSnapshot:
    """Immutable capture of a world clock's configuration and instant."""

    config: WorldTimeConfig
    instant: WorldInstant

    def __post_init__(self) -> None:
        if not isinstance(self.config, WorldTimeConfig):
            raise TypeError("config must be a WorldTimeConfig.")

        if not isinstance(self.instant, WorldInstant):
            raise TypeError("instant must be a WorldInstant.")

    @property
    def date_time(self) -> WorldDateTime:
        """Return the calendar projection captured by this snapshot."""
        return WorldDateTime.from_instant(
            instant=self.instant,
            config=self.config,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldClock:
    """Immutable deterministic clock advanced only by explicit integer input."""

    config: WorldTimeConfig = WorldTimeConfig()
    current: WorldInstant = WorldInstant()

    def __post_init__(self) -> None:
        if not isinstance(self.config, WorldTimeConfig):
            raise TypeError("config must be a WorldTimeConfig.")

        if not isinstance(self.current, WorldInstant):
            raise TypeError("current must be a WorldInstant.")

    @property
    def date_time(self) -> WorldDateTime:
        """Return the calendar projection of the current instant."""
        return WorldDateTime.from_instant(
            instant=self.current,
            config=self.config,
        )

    def advance_ticks(self, ticks: int) -> WorldClock:
        """Return a clock advanced by a non-negative number of ticks."""
        return replace(self, current=self.current.advance(ticks))

    def advance_seconds(self, seconds: int) -> WorldClock:
        """Return a clock advanced by whole seconds using integer arithmetic."""
        _require_non_negative_integer(name="seconds", value=seconds)
        return self.advance_ticks(seconds * self.config.ticks_per_second)

    def snapshot(self) -> WorldClockSnapshot:
        """Capture the clock's current immutable state."""
        return WorldClockSnapshot(
            config=self.config,
            instant=self.current,
        )

    def reset(self, *, instant: WorldInstant | None = None) -> WorldClock:
        """Return a clock reset explicitly to zero or a supplied valid instant."""
        resolved = WorldInstant() if instant is None else instant
        if not isinstance(resolved, WorldInstant):
            raise TypeError("instant must be a WorldInstant.")

        return replace(self, current=resolved)
