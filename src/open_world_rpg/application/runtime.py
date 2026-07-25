"""Application runtime lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from open_world_rpg.application.session import RuntimeContext
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
    """Own the configuration, session context, and process lifecycle."""

    config: GameConfig
    context: RuntimeContext
    _state: ApplicationState = field(
        default=ApplicationState.CREATED,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.config, GameConfig):
            raise TypeError("config must be a GameConfig.")

        if not isinstance(self.context, RuntimeContext):
            raise TypeError("context must be a RuntimeContext.")

        if self.context.world_seed != self.config.simulation.world_seed:
            raise ValueError("Runtime context seed must match the configured world seed.")

    @property
    def state(self) -> ApplicationState:
        """Return the application's current lifecycle state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Return whether the application process is currently running."""
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
            self.context.start()
        except Exception:
            self._state = ApplicationState.FAILED
            raise

        self._state = ApplicationState.RUNNING

    def pause(self) -> None:
        """Pause gameplay while keeping the application process running."""
        self._require_state(
            expected=ApplicationState.RUNNING,
            operation="pause",
        )
        self.context.pause()

    def resume(self) -> None:
        """Resume a paused game session."""
        self._require_state(
            expected=ApplicationState.RUNNING,
            operation="resume",
        )
        self.context.resume()

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

        try:
            self.context.terminate()
        except Exception:
            self._state = ApplicationState.FAILED
            raise

        self._state = ApplicationState.STOPPED

    def fail(self) -> None:
        """Mark the application and its active session as failed."""
        if self._state is ApplicationState.STOPPED:
            raise ApplicationLifecycleError("A stopped application cannot be marked as failed.")

        self.context.fail()
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
