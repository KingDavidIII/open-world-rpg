"""Deterministic fixed-step timing primitives for the game engine."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

NANOSECONDS_PER_SECOND: Final = 1_000_000_000


class EngineTimingError(RuntimeError):
    """Base exception for engine timing failures."""


class ClockRegressionError(EngineTimingError):
    """Raised when a clock produces a timestamp earlier than its previous value."""


@runtime_checkable
class EngineClock(Protocol):
    """Clock contract used by engine timing infrastructure."""

    def now_ns(self) -> int:
        """Return the current monotonic timestamp in nanoseconds."""


@dataclass(frozen=True, slots=True)
class MonotonicClock:
    """Production clock backed by Python's monotonic performance counter."""

    def now_ns(self) -> int:
        """Return a monotonic high-resolution timestamp."""
        return time.perf_counter_ns()


@dataclass(frozen=True, slots=True)
class FixedStepConfig:
    """Validated configuration for fixed-step simulation scheduling."""

    tick_rate_hz: int = 60
    max_frame_duration_ns: int = 250_000_000
    max_updates_per_frame: int = 8

    def __post_init__(self) -> None:
        if type(self.tick_rate_hz) is not int:
            raise TypeError("tick_rate_hz must be an integer.")

        if self.tick_rate_hz <= 0:
            raise ValueError("tick_rate_hz must be greater than zero.")

        if self.tick_rate_hz > NANOSECONDS_PER_SECOND:
            raise ValueError("tick_rate_hz cannot exceed nanosecond clock resolution.")

        if type(self.max_frame_duration_ns) is not int:
            raise TypeError("max_frame_duration_ns must be an integer.")

        if self.max_frame_duration_ns <= 0:
            raise ValueError("max_frame_duration_ns must be greater than zero.")

        if type(self.max_updates_per_frame) is not int:
            raise TypeError("max_updates_per_frame must be an integer.")

        if self.max_updates_per_frame <= 0:
            raise ValueError("max_updates_per_frame must be greater than zero.")

    @property
    def fixed_step_seconds(self) -> float:
        """Return the duration of one simulation update in seconds."""
        return 1.0 / self.tick_rate_hz


@dataclass(frozen=True, slots=True)
class FrameSchedule:
    """Immutable simulation schedule calculated for one rendered frame."""

    frame_index: int
    timestamp_ns: int
    elapsed_ns: int
    simulated_elapsed_ns: int
    update_count: int
    dropped_update_count: int
    interpolation_alpha: float

    def __post_init__(self) -> None:
        self._validate_non_negative_integer(
            "frame_index",
            self.frame_index,
        )
        self._validate_non_negative_integer(
            "timestamp_ns",
            self.timestamp_ns,
        )
        self._validate_non_negative_integer(
            "elapsed_ns",
            self.elapsed_ns,
        )
        self._validate_non_negative_integer(
            "simulated_elapsed_ns",
            self.simulated_elapsed_ns,
        )
        self._validate_non_negative_integer(
            "update_count",
            self.update_count,
        )
        self._validate_non_negative_integer(
            "dropped_update_count",
            self.dropped_update_count,
        )

        if self.simulated_elapsed_ns > self.elapsed_ns:
            raise ValueError("simulated_elapsed_ns cannot exceed elapsed_ns.")

        if type(self.interpolation_alpha) is not float:
            raise TypeError("interpolation_alpha must be a float.")

        if not math.isfinite(self.interpolation_alpha):
            raise ValueError("interpolation_alpha must be finite.")

        if not 0.0 <= self.interpolation_alpha < 1.0:
            raise ValueError(
                "interpolation_alpha must be greater than or equal to zero and less than one."
            )

    @staticmethod
    def _validate_non_negative_integer(
        name: str,
        value: int,
    ) -> None:
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer.")

        if value < 0:
            raise ValueError(f"{name} must be greater than or equal to zero.")


class FixedStepScheduler:
    """Convert monotonic clock samples into deterministic update schedules."""

    __slots__ = (
        "_accumulator_units",
        "_config",
        "_frame_count",
        "_last_timestamp_ns",
    )

    def __init__(
        self,
        config: FixedStepConfig | None = None,
    ) -> None:
        resolved_config = FixedStepConfig() if config is None else config

        if not isinstance(resolved_config, FixedStepConfig):
            raise TypeError("config must be a FixedStepConfig.")

        self._config = resolved_config
        self._last_timestamp_ns: int | None = None
        self._frame_count = 0
        self._accumulator_units = 0

    @property
    def config(self) -> FixedStepConfig:
        """Return the scheduler configuration."""
        return self._config

    @property
    def started(self) -> bool:
        """Return whether at least one clock sample has been accepted."""
        return self._last_timestamp_ns is not None

    @property
    def frame_count(self) -> int:
        """Return the number of schedules produced."""
        return self._frame_count

    @property
    def last_timestamp_ns(self) -> int | None:
        """Return the most recently accepted timestamp."""
        return self._last_timestamp_ns

    @property
    def interpolation_alpha(self) -> float:
        """Return the current fractional progress towards the next update."""
        return self._accumulator_units / NANOSECONDS_PER_SECOND

    def reset(self) -> None:
        """Return the scheduler to its initial unstarted state."""
        self._last_timestamp_ns = None
        self._frame_count = 0
        self._accumulator_units = 0

    def advance(self, timestamp_ns: int) -> FrameSchedule:
        """Accept a clock sample and calculate the next frame schedule."""
        if type(timestamp_ns) is not int:
            raise TypeError("timestamp_ns must be an integer.")

        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be greater than or equal to zero.")

        if self._last_timestamp_ns is None:
            return self._start(timestamp_ns)

        previous_timestamp_ns = self._last_timestamp_ns

        if timestamp_ns < previous_timestamp_ns:
            raise ClockRegressionError(
                "Engine clock moved backwards: "
                f"previous={previous_timestamp_ns}, "
                f"current={timestamp_ns}."
            )

        elapsed_ns = timestamp_ns - previous_timestamp_ns
        simulated_elapsed_ns = min(
            elapsed_ns,
            self._config.max_frame_duration_ns,
        )

        self._last_timestamp_ns = timestamp_ns
        self._accumulator_units += simulated_elapsed_ns * self._config.tick_rate_hz

        available_updates = self._accumulator_units // NANOSECONDS_PER_SECOND
        update_count = min(
            available_updates,
            self._config.max_updates_per_frame,
        )
        dropped_update_count = available_updates - update_count

        self._accumulator_units -= available_updates * NANOSECONDS_PER_SECOND

        schedule = FrameSchedule(
            frame_index=self._frame_count,
            timestamp_ns=timestamp_ns,
            elapsed_ns=elapsed_ns,
            simulated_elapsed_ns=simulated_elapsed_ns,
            update_count=update_count,
            dropped_update_count=dropped_update_count,
            interpolation_alpha=self.interpolation_alpha,
        )
        self._frame_count += 1

        return schedule

    def _start(self, timestamp_ns: int) -> FrameSchedule:
        self._last_timestamp_ns = timestamp_ns

        schedule = FrameSchedule(
            frame_index=self._frame_count,
            timestamp_ns=timestamp_ns,
            elapsed_ns=0,
            simulated_elapsed_ns=0,
            update_count=0,
            dropped_update_count=0,
            interpolation_alpha=0.0,
        )
        self._frame_count += 1

        return schedule
