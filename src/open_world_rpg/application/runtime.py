"""Application runtime lifecycle management."""

from __future__ import annotations

import logging
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
    """Own configuration, session context, diagnostics, and process lifecycle."""

    config: GameConfig
    context: RuntimeContext
    logger: logging.Logger
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

        if not isinstance(self.logger, logging.Logger):
            raise TypeError("logger must be a logging.Logger.")

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
        self._log_event(
            level=logging.INFO,
            event="application.starting",
            message="Application startup initiated.",
        )

        try:
            self._create_runtime_directories()
            self.context.start()
        except Exception:
            self._state = ApplicationState.FAILED
            self.logger.exception(
                "Application startup failed.",
                extra=self._diagnostic_context(event="application.start_failed"),
            )
            raise

        self._log_event(
            level=logging.INFO,
            event="session.activated",
            message="Game session activated.",
        )

        self._state = ApplicationState.RUNNING
        self._log_event(
            level=logging.INFO,
            event="application.running",
            message="Application runtime started.",
        )

    def pause(self) -> None:
        """Pause gameplay while keeping the process running."""
        self._require_state(
            expected=ApplicationState.RUNNING,
            operation="pause",
        )

        try:
            self.context.pause()
        except Exception:
            self.logger.exception(
                "Game session could not be paused.",
                extra=self._diagnostic_context(event="session.pause_failed"),
            )
            raise

        self._log_event(
            level=logging.INFO,
            event="session.paused",
            message="Game session paused.",
        )

    def resume(self) -> None:
        """Resume a paused game session."""
        self._require_state(
            expected=ApplicationState.RUNNING,
            operation="resume",
        )

        try:
            self.context.resume()
        except Exception:
            self.logger.exception(
                "Game session could not be resumed.",
                extra=self._diagnostic_context(event="session.resume_failed"),
            )
            raise

        self._log_event(
            level=logging.INFO,
            event="session.resumed",
            message="Game session resumed.",
        )

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
        self._log_event(
            level=logging.INFO,
            event="application.stopping",
            message="Application shutdown initiated.",
        )

        try:
            self.context.terminate()
        except Exception:
            self._state = ApplicationState.FAILED
            self.logger.exception(
                "Application shutdown failed.",
                extra=self._diagnostic_context(event="application.stop_failed"),
            )
            raise

        self._log_event(
            level=logging.INFO,
            event="session.terminated",
            message="Game session terminated.",
        )

        self._state = ApplicationState.STOPPED
        self._log_event(
            level=logging.INFO,
            event="application.stopped",
            message="Application runtime stopped.",
        )

    def fail(self) -> None:
        """Mark the application and its active session as failed."""
        if self._state is ApplicationState.STOPPED:
            raise ApplicationLifecycleError("A stopped application cannot be marked as failed.")

        self.context.fail()
        self._log_event(
            level=logging.ERROR,
            event="session.failed",
            message="Game session entered the failed state.",
        )

        self._state = ApplicationState.FAILED
        self._log_event(
            level=logging.ERROR,
            event="application.failed",
            message="Application entered the failed state.",
        )

    def _create_runtime_directories(self) -> None:
        for directory in (
            self.config.paths.save_directory,
            self.config.paths.log_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _diagnostic_context(self, *, event: str) -> dict[str, object]:
        return {
            "event": event,
            "session_id": str(self.context.session_id),
            "world_seed": self.context.world_seed,
            "application_state": self._state.value,
            "session_state": self.context.state.value,
        }

    def _log_event(
        self,
        *,
        level: int,
        event: str,
        message: str,
    ) -> None:
        self.logger.log(
            level,
            message,
            extra=self._diagnostic_context(event=event),
        )

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
