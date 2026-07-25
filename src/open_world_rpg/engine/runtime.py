"""Deterministic engine runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from open_world_rpg.engine.subsystems import SubsystemRegistry
from open_world_rpg.engine.timing import (
    EngineClock,
    FixedStepScheduler,
    FrameSchedule,
    MonotonicClock,
)


class EngineRuntimeError(RuntimeError):
    """Base exception for engine runtime failures."""


class EngineRuntimeStateError(EngineRuntimeError):
    """Raised when an operation is invalid for the runtime state."""


class EngineRuntimeExecutionError(EngineRuntimeError):
    """Raised when startup, frame execution, or shutdown fails."""

    def __init__(
        self,
        *,
        operation: str,
        cause: Exception,
        cleanup_error: Exception | None = None,
    ) -> None:
        self.operation = operation
        self.cause = cause
        self.cleanup_error = cleanup_error

        message = f"Engine runtime failed during {operation}."

        if cleanup_error is not None:
            message += " Subsystem cleanup also failed."

        super().__init__(message)


class EngineRuntimeState(StrEnum):
    """Lifecycle state of the deterministic engine runtime."""

    CREATED = "created"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EngineRuntimeSnapshot:
    """Immutable engine runtime statistics and lifecycle state."""

    state: EngineRuntimeState
    frame_count: int
    update_count: int
    dropped_update_count: int
    stop_reason: str | None
    last_schedule: FrameSchedule | None


class EngineRuntime:
    """Coordinate timing and subsystem execution for the game engine."""

    __slots__ = (
        "_clock",
        "_dropped_update_count",
        "_frame_count",
        "_last_schedule",
        "_registry",
        "_scheduler",
        "_state",
        "_stop_reason",
        "_update_count",
    )

    def __init__(
        self,
        *,
        registry: SubsystemRegistry,
        scheduler: FixedStepScheduler | None = None,
        clock: EngineClock | None = None,
    ) -> None:
        if not isinstance(registry, SubsystemRegistry):
            raise TypeError("registry must be a SubsystemRegistry.")

        resolved_scheduler = FixedStepScheduler() if scheduler is None else scheduler
        if not isinstance(
            resolved_scheduler,
            FixedStepScheduler,
        ):
            raise TypeError("scheduler must be a FixedStepScheduler.")

        resolved_clock = MonotonicClock() if clock is None else clock
        if not isinstance(resolved_clock, EngineClock):
            raise TypeError("clock must implement EngineClock.")

        self._registry = registry
        self._scheduler = resolved_scheduler
        self._clock = resolved_clock
        self._state = EngineRuntimeState.CREATED
        self._frame_count = 0
        self._update_count = 0
        self._dropped_update_count = 0
        self._stop_reason: str | None = None
        self._last_schedule: FrameSchedule | None = None

    @property
    def state(self) -> EngineRuntimeState:
        """Return the current engine runtime state."""
        return self._state

    @property
    def registry(self) -> SubsystemRegistry:
        """Return the managed subsystem registry."""
        return self._registry

    @property
    def scheduler(self) -> FixedStepScheduler:
        """Return the fixed-step scheduler."""
        return self._scheduler

    @property
    def clock(self) -> EngineClock:
        """Return the engine clock."""
        return self._clock

    @property
    def frame_count(self) -> int:
        """Return the number of successfully rendered frames."""
        return self._frame_count

    @property
    def update_count(self) -> int:
        """Return the number of completed fixed updates."""
        return self._update_count

    @property
    def dropped_update_count(self) -> int:
        """Return the number of updates dropped by frame limits."""
        return self._dropped_update_count

    @property
    def stop_reason(self) -> str | None:
        """Return the reason the engine was asked to stop."""
        return self._stop_reason

    @property
    def last_schedule(self) -> FrameSchedule | None:
        """Return the most recent successfully rendered schedule."""
        return self._last_schedule

    @property
    def snapshot(self) -> EngineRuntimeSnapshot:
        """Return an immutable runtime-state snapshot."""
        return EngineRuntimeSnapshot(
            state=self._state,
            frame_count=self._frame_count,
            update_count=self._update_count,
            dropped_update_count=self._dropped_update_count,
            stop_reason=self._stop_reason,
            last_schedule=self._last_schedule,
        )

    def start(self) -> None:
        """Start registered subsystems and initialise runtime state."""
        self._require_state(
            EngineRuntimeState.CREATED,
            operation="start",
        )
        self._scheduler.reset()

        try:
            self._registry.start()
        except Exception as exc:
            cleanup_error = self._cleanup_registry()
            self._state = EngineRuntimeState.FAILED

            raise EngineRuntimeExecutionError(
                operation="start",
                cause=exc,
                cleanup_error=cleanup_error,
            ) from exc

        self._state = EngineRuntimeState.RUNNING

    def run_frame(self) -> FrameSchedule:
        """Execute one scheduled engine frame."""
        self._require_state(
            EngineRuntimeState.RUNNING,
            operation="run a frame",
        )

        try:
            timestamp_ns = self._clock.now_ns()
            schedule = self._scheduler.advance(timestamp_ns)

            for _ in range(schedule.update_count):
                self._registry.update(self._scheduler.config.fixed_step_seconds)
                self._update_count += 1

            self._registry.render(schedule.interpolation_alpha)
        except Exception as exc:
            cleanup_error = self._cleanup_registry()
            self._state = EngineRuntimeState.FAILED

            raise EngineRuntimeExecutionError(
                operation="frame execution",
                cause=exc,
                cleanup_error=cleanup_error,
            ) from exc

        self._frame_count += 1
        self._dropped_update_count += schedule.dropped_update_count
        self._last_schedule = schedule

        return schedule

    def request_stop(
        self,
        reason: str = "requested",
    ) -> None:
        """Request a controlled stop after the current frame."""
        self._validate_stop_reason(reason)

        if self._state is EngineRuntimeState.STOP_REQUESTED:
            return

        self._require_state(
            EngineRuntimeState.RUNNING,
            operation="request a stop",
        )

        self._stop_reason = reason
        self._state = EngineRuntimeState.STOP_REQUESTED

    def shutdown(self) -> None:
        """Stop all managed subsystems."""
        if self._state is EngineRuntimeState.STOPPED:
            return

        try:
            self._registry.shutdown()
        except Exception as exc:
            self._state = EngineRuntimeState.FAILED
            raise EngineRuntimeExecutionError(
                operation="shutdown",
                cause=exc,
            ) from exc

        if self._stop_reason is None:
            self._stop_reason = "shutdown"

        self._state = EngineRuntimeState.STOPPED

    def run(
        self,
        *,
        max_frames: int | None = None,
    ) -> EngineRuntimeSnapshot:
        """Run frames until stopped or an optional limit is reached."""
        self._validate_max_frames(max_frames)
        self.start()

        executed_frames = 0

        try:
            while self._state is EngineRuntimeState.RUNNING:
                self.run_frame()
                executed_frames += 1

                if max_frames is not None and executed_frames >= max_frames:
                    self.request_stop("frame_limit")
        finally:
            if self._state in {
                EngineRuntimeState.RUNNING,
                EngineRuntimeState.STOP_REQUESTED,
            }:
                self.shutdown()

        return self.snapshot

    def _cleanup_registry(self) -> Exception | None:
        try:
            self._registry.shutdown()
        except Exception as exc:
            return exc

        return None

    def _require_state(
        self,
        expected: EngineRuntimeState,
        *,
        operation: str,
    ) -> None:
        if self._state is not expected:
            raise EngineRuntimeStateError(
                f"Cannot {operation} while engine runtime is {self._state.value}."
            )

    @staticmethod
    def _validate_stop_reason(reason: str) -> None:
        if not isinstance(reason, str):
            raise TypeError("stop reason must be a string.")

        if not reason.strip():
            raise ValueError("stop reason cannot be empty.")

        if reason != reason.strip():
            raise ValueError("stop reason cannot contain surrounding whitespace.")

    @staticmethod
    def _validate_max_frames(
        max_frames: int | None,
    ) -> None:
        if max_frames is None:
            return

        if type(max_frames) is not int:
            raise TypeError("max_frames must be an integer or None.")

        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero.")
