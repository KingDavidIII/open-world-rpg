"""Application runtime lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from open_world_rpg.core import GameConfig


class ApplicationState(StrEnum):
    """Possible lifecycle states of a game application."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ApplicationLifecycleError(RuntimeError):
    """Raised when an invalid lifecycle operation is attempted."""


@dataclass(slots=True)
class GameApplication:
    """Own the configuration and lifecycle of one game process."""

    config: GameConfig
    _state: ApplicationState = field(
        default=ApplicationState.CREATED,
        init=False,
        repr=False,
    )

    @property
    def state(self) -> ApplicationState:
        """Return the application's current lifecycle state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Return whether the application is currently running."""
        return self._state is ApplicationState.RUNNING

    def start(self) -> None:
        """Prepare runtime resources and start the application."""
        self._require_state(
            expected=ApplicationState.CREATED,
            operation="start",
        )
        self._state = ApplicationState.STARTING

        try:
            self._create_runtime_directories()
        except OSError:
            self._state = ApplicationState.FAILED
            raise

        self._state = ApplicationState.RUNNING

    def stop(self) -> None:
        """Stop a running application.

        Repeated calls after a successful stop are intentionally harmless.
        """
        if self._state is ApplicationState.STOPPED:
            return

        self._require_state(
            expected=ApplicationState.RUNNING,
            operation="stop",
        )
        self._state = ApplicationState.STOPPING
        self._state = ApplicationState.STOPPED

    def fail(self) -> None:
        """Mark the application as failed."""
        if self._state is ApplicationState.STOPPED:
            raise ApplicationLifecycleError("A stopped application cannot be marked as failed.")

        self._state = ApplicationState.FAILED

    def _create_runtime_directories(self) -> None:
        for directory in (
            self.config.paths.save_directory,
            self.config.paths.log_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _require_state(
        self,
        *,
        expected: ApplicationState,
        operation: str,
    ) -> None:
        if self._state is not expected:
            raise ApplicationLifecycleError(
                f"Cannot {operation} application while state is "
                f"{self._state.value!r}; expected {expected.value!r}."
            )
