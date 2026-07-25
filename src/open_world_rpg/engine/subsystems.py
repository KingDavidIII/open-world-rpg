"""Ordered lifecycle management for engine subsystems."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class SubsystemRegistryError(RuntimeError):
    """Base exception for subsystem registry failures."""


class DuplicateSubsystemError(SubsystemRegistryError):
    """Raised when two registered subsystems use the same name."""


class SubsystemRegistryStateError(SubsystemRegistryError):
    """Raised when an operation is invalid for the registry state."""


@dataclass(frozen=True, slots=True)
class SubsystemFailure:
    """A subsystem failure captured during lifecycle cleanup."""

    subsystem_name: str
    error: Exception


class SubsystemExecutionError(SubsystemRegistryError):
    """Raised when one subsystem fails during a lifecycle operation."""

    def __init__(
        self,
        *,
        operation: str,
        subsystem_name: str,
        cause: Exception,
        cleanup_failures: tuple[SubsystemFailure, ...] = (),
    ) -> None:
        self.operation = operation
        self.subsystem_name = subsystem_name
        self.cause = cause
        self.cleanup_failures = cleanup_failures

        message = f"Subsystem {subsystem_name!r} failed during {operation}."

        if cleanup_failures:
            failed_names = ", ".join(failure.subsystem_name for failure in cleanup_failures)
            message += f" Cleanup also failed for: {failed_names}."

        super().__init__(message)


class SubsystemShutdownError(SubsystemRegistryError):
    """Raised after one or more subsystems fail during shutdown."""

    def __init__(
        self,
        failures: tuple[SubsystemFailure, ...],
    ) -> None:
        self.failures = failures
        failed_names = ", ".join(failure.subsystem_name for failure in failures)
        super().__init__(f"Subsystem shutdown failed for: {failed_names}.")


class SubsystemRegistryState(StrEnum):
    """Lifecycle state of an engine subsystem registry."""

    CREATED = "created"
    STARTED = "started"
    STOPPED = "stopped"
    FAILED = "failed"


@runtime_checkable
class EngineSubsystem(Protocol):
    """Contract implemented by an engine subsystem."""

    @property
    def name(self) -> str:
        """Return the unique subsystem name."""

    def start(self) -> None:
        """Acquire resources and initialise the subsystem."""

    def update(self, fixed_delta_seconds: float) -> None:
        """Advance deterministic simulation state."""

    def render(self, interpolation_alpha: float) -> None:
        """Render interpolated presentation state."""

    def stop(self) -> None:
        """Release resources owned by the subsystem."""


class SubsystemRegistry:
    """Manage ordered startup, execution, and shutdown of subsystems."""

    __slots__ = (
        "_by_name",
        "_started",
        "_state",
        "_subsystems",
    )

    def __init__(
        self,
        subsystems: Iterable[EngineSubsystem] = (),
    ) -> None:
        self._subsystems: list[EngineSubsystem] = []
        self._by_name: dict[str, EngineSubsystem] = {}
        self._started: list[EngineSubsystem] = []
        self._state = SubsystemRegistryState.CREATED

        for subsystem in subsystems:
            self.register(subsystem)

    @property
    def state(self) -> SubsystemRegistryState:
        """Return the current registry lifecycle state."""
        return self._state

    @property
    def subsystem_names(self) -> tuple[str, ...]:
        """Return subsystem names in registration order."""
        return tuple(subsystem.name for subsystem in self._subsystems)

    @property
    def started_subsystem_names(self) -> tuple[str, ...]:
        """Return successfully started subsystems in startup order."""
        return tuple(subsystem.name for subsystem in self._started)

    @property
    def subsystem_count(self) -> int:
        """Return the number of registered subsystems."""
        return len(self._subsystems)

    def register(
        self,
        subsystem: EngineSubsystem,
    ) -> None:
        """Register a subsystem before registry startup."""
        self._require_state(
            SubsystemRegistryState.CREATED,
            operation="register",
        )

        if not isinstance(subsystem, EngineSubsystem):
            raise TypeError("subsystem must implement EngineSubsystem.")

        name = subsystem.name

        if not isinstance(name, str):
            raise TypeError("subsystem name must be a string.")

        if not name.strip():
            raise ValueError("subsystem name cannot be empty.")

        if name != name.strip():
            raise ValueError("subsystem name cannot contain surrounding whitespace.")

        if name in self._by_name:
            raise DuplicateSubsystemError(f"Subsystem {name!r} is already registered.")

        self._subsystems.append(subsystem)
        self._by_name[name] = subsystem

    def start(self) -> None:
        """Start every subsystem in registration order."""
        self._require_state(
            SubsystemRegistryState.CREATED,
            operation="start",
        )

        for subsystem in self._subsystems:
            try:
                subsystem.start()
            except Exception as exc:
                cleanup_failures = self._rollback_started()
                self._state = SubsystemRegistryState.FAILED

                raise SubsystemExecutionError(
                    operation="start",
                    subsystem_name=subsystem.name,
                    cause=exc,
                    cleanup_failures=cleanup_failures,
                ) from exc

            self._started.append(subsystem)

        self._state = SubsystemRegistryState.STARTED

    def update(self, fixed_delta_seconds: float) -> None:
        """Update every started subsystem in registration order."""
        self._require_state(
            SubsystemRegistryState.STARTED,
            operation="update",
        )
        self._validate_fixed_delta(fixed_delta_seconds)

        for subsystem in self._started:
            try:
                subsystem.update(fixed_delta_seconds)
            except Exception as exc:
                self._state = SubsystemRegistryState.FAILED
                raise SubsystemExecutionError(
                    operation="update",
                    subsystem_name=subsystem.name,
                    cause=exc,
                ) from exc

    def render(self, interpolation_alpha: float) -> None:
        """Render every started subsystem in registration order."""
        self._require_state(
            SubsystemRegistryState.STARTED,
            operation="render",
        )
        self._validate_interpolation_alpha(interpolation_alpha)

        for subsystem in self._started:
            try:
                subsystem.render(interpolation_alpha)
            except Exception as exc:
                self._state = SubsystemRegistryState.FAILED
                raise SubsystemExecutionError(
                    operation="render",
                    subsystem_name=subsystem.name,
                    cause=exc,
                ) from exc

    def shutdown(self) -> None:
        """Stop started subsystems in reverse startup order."""
        if self._state is SubsystemRegistryState.STOPPED:
            return

        if self._state is SubsystemRegistryState.CREATED:
            self._state = SubsystemRegistryState.STOPPED
            return

        failures: list[SubsystemFailure] = []

        for subsystem in reversed(self._started):
            try:
                subsystem.stop()
            except Exception as exc:
                failures.append(
                    SubsystemFailure(
                        subsystem_name=subsystem.name,
                        error=exc,
                    )
                )

        self._started.clear()

        if failures:
            self._state = SubsystemRegistryState.FAILED
            raise SubsystemShutdownError(tuple(failures))

        self._state = SubsystemRegistryState.STOPPED

    def _rollback_started(
        self,
    ) -> tuple[SubsystemFailure, ...]:
        failures: list[SubsystemFailure] = []

        for subsystem in reversed(self._started):
            try:
                subsystem.stop()
            except Exception as exc:
                failures.append(
                    SubsystemFailure(
                        subsystem_name=subsystem.name,
                        error=exc,
                    )
                )

        self._started.clear()
        return tuple(failures)

    def _require_state(
        self,
        expected: SubsystemRegistryState,
        *,
        operation: str,
    ) -> None:
        if self._state is not expected:
            raise SubsystemRegistryStateError(
                f"Cannot {operation} subsystems while registry is {self._state.value}."
            )

    @staticmethod
    def _validate_fixed_delta(
        fixed_delta_seconds: float,
    ) -> None:
        if type(fixed_delta_seconds) is not float:
            raise TypeError("fixed_delta_seconds must be a float.")

        if not math.isfinite(fixed_delta_seconds):
            raise ValueError("fixed_delta_seconds must be finite.")

        if fixed_delta_seconds <= 0.0:
            raise ValueError("fixed_delta_seconds must be greater than zero.")

    @staticmethod
    def _validate_interpolation_alpha(
        interpolation_alpha: float,
    ) -> None:
        if type(interpolation_alpha) is not float:
            raise TypeError("interpolation_alpha must be a float.")

        if not math.isfinite(interpolation_alpha):
            raise ValueError("interpolation_alpha must be finite.")

        if not 0.0 <= interpolation_alpha < 1.0:
            raise ValueError(
                "interpolation_alpha must be greater than or equal to zero and less than one."
            )
