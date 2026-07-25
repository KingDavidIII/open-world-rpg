"""Runtime session identity and controlled game-state transitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeAlias
from uuid import UUID, uuid4

from open_world_rpg.core.config import MAX_WORLD_SEED, MIN_WORLD_SEED

Clock: TypeAlias = Callable[[], datetime]


class GameMode(StrEnum):
    """Supported ways to initialise a game session."""

    NEW_GAME = "new_game"
    LOADED_GAME = "loaded_game"


class SessionState(StrEnum):
    """Possible lifecycle states of a game session."""

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"
    FAILED = "failed"


class SessionTransitionError(RuntimeError):
    """Raised when a session transition is invalid."""


class SessionClockError(RuntimeError):
    """Raised when the runtime clock moves backwards."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalise_timestamp(*, name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")

    return value.astimezone(UTC)


def _validate_world_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("world_seed must be an integer.")

    if value < MIN_WORLD_SEED or value > MAX_WORLD_SEED:
        raise ValueError(f"world_seed must be between {MIN_WORLD_SEED} and {MAX_WORLD_SEED}.")

    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeContext:
    """Identity, timing, and mutable lifecycle state for one game session."""

    session_id: UUID
    game_mode: GameMode
    world_seed: int
    created_at: datetime
    _clock: Clock = field(default=_utc_now, repr=False, compare=False)
    _state: SessionState = field(
        default=SessionState.CREATED,
        init=False,
        repr=False,
    )
    _started_at: datetime | None = field(default=None, init=False, repr=False)
    _paused_at: datetime | None = field(default=None, init=False, repr=False)
    _resumed_at: datetime | None = field(default=None, init=False, repr=False)
    _terminated_at: datetime | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _failed_at: datetime | None = field(default=None, init=False, repr=False)
    _last_transition_at: datetime = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, UUID):
            raise TypeError("session_id must be a UUID.")

        if not isinstance(self.game_mode, GameMode):
            raise TypeError("game_mode must be a GameMode.")

        if not callable(self._clock):
            raise TypeError("clock must be callable.")

        object.__setattr__(
            self,
            "world_seed",
            _validate_world_seed(self.world_seed),
        )

        created_at = _normalise_timestamp(
            name="created_at",
            value=self.created_at,
        )
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "_last_transition_at", created_at)

    @classmethod
    def create(
        cls,
        *,
        game_mode: GameMode,
        world_seed: int,
        clock: Clock | None = None,
        session_id: UUID | None = None,
    ) -> RuntimeContext:
        """Create a new session using injectable identity and time providers."""
        resolved_clock = _utc_now if clock is None else clock

        if not callable(resolved_clock):
            raise TypeError("clock must be callable.")

        created_at = _normalise_timestamp(
            name="clock result",
            value=resolved_clock(),
        )

        return cls(
            session_id=uuid4() if session_id is None else session_id,
            game_mode=game_mode,
            world_seed=world_seed,
            created_at=created_at,
            _clock=resolved_clock,
        )

    @classmethod
    def restore(
        cls,
        *,
        session_id: UUID,
        game_mode: GameMode,
        world_seed: int,
        state: SessionState,
        clock: Clock | None = None,
    ) -> RuntimeContext:
        """Restore a previously saved resumable session."""
        if not isinstance(state, SessionState):
            raise TypeError("state must be a SessionState.")

        if state not in {
            SessionState.ACTIVE,
            SessionState.PAUSED,
        }:
            raise ValueError("state must represent an active or paused session.")

        context = cls.create(
            session_id=session_id,
            game_mode=game_mode,
            world_seed=world_seed,
            clock=clock,
        )
        context.start()

        if state is SessionState.PAUSED:
            context.pause()

        return context

    @property
    def state(self) -> SessionState:
        """Return the current session state."""
        return self._state

    @property
    def started_at(self) -> datetime | None:
        """Return when the session first became active."""
        return self._started_at

    @property
    def paused_at(self) -> datetime | None:
        """Return when the session was most recently paused."""
        return self._paused_at

    @property
    def resumed_at(self) -> datetime | None:
        """Return when the session was most recently resumed."""
        return self._resumed_at

    @property
    def terminated_at(self) -> datetime | None:
        """Return when the session terminated normally."""
        return self._terminated_at

    @property
    def failed_at(self) -> datetime | None:
        """Return when the session entered the failed state."""
        return self._failed_at

    @property
    def last_transition_at(self) -> datetime:
        """Return the timestamp of the most recent state transition."""
        return self._last_transition_at

    @property
    def is_active(self) -> bool:
        """Return whether gameplay is currently active."""
        return self._state is SessionState.ACTIVE

    @property
    def is_paused(self) -> bool:
        """Return whether gameplay is currently paused."""
        return self._state is SessionState.PAUSED

    @property
    def is_terminal(self) -> bool:
        """Return whether no further valid gameplay transitions remain."""
        return self._state in {
            SessionState.TERMINATED,
            SessionState.FAILED,
        }

    def start(self) -> None:
        """Transition a newly created session into active gameplay."""
        self._require_state(
            allowed=(SessionState.CREATED,),
            operation="start",
        )
        timestamp = self._next_transition_timestamp()

        object.__setattr__(self, "_started_at", timestamp)
        object.__setattr__(self, "_state", SessionState.ACTIVE)

    def pause(self) -> None:
        """Pause an active game session."""
        self._require_state(
            allowed=(SessionState.ACTIVE,),
            operation="pause",
        )
        timestamp = self._next_transition_timestamp()

        object.__setattr__(self, "_paused_at", timestamp)
        object.__setattr__(self, "_state", SessionState.PAUSED)

    def resume(self) -> None:
        """Resume a paused game session."""
        self._require_state(
            allowed=(SessionState.PAUSED,),
            operation="resume",
        )
        timestamp = self._next_transition_timestamp()

        object.__setattr__(self, "_resumed_at", timestamp)
        object.__setattr__(self, "_state", SessionState.ACTIVE)

    def terminate(self) -> None:
        """Terminate an active or paused session normally."""
        self._require_state(
            allowed=(
                SessionState.ACTIVE,
                SessionState.PAUSED,
            ),
            operation="terminate",
        )
        timestamp = self._next_transition_timestamp()

        object.__setattr__(self, "_terminated_at", timestamp)
        object.__setattr__(self, "_state", SessionState.TERMINATED)

    def fail(self) -> None:
        """Move a non-terminated session into the failed state."""
        if self._state is SessionState.TERMINATED:
            raise SessionTransitionError("A terminated session cannot be marked as failed.")

        if self._state is SessionState.FAILED:
            return

        timestamp = self._next_transition_timestamp()

        object.__setattr__(self, "_failed_at", timestamp)
        object.__setattr__(self, "_state", SessionState.FAILED)

    def _next_transition_timestamp(self) -> datetime:
        timestamp = _normalise_timestamp(
            name="clock result",
            value=self._clock(),
        )

        if timestamp < self._last_transition_at:
            raise SessionClockError("Runtime clock moved backwards during a session transition.")

        object.__setattr__(self, "_last_transition_at", timestamp)
        return timestamp

    def _require_state(
        self,
        *,
        allowed: tuple[SessionState, ...],
        operation: str,
    ) -> None:
        if self._state in allowed:
            return

        expected = ", ".join(repr(state.value) for state in allowed)
        raise SessionTransitionError(
            f"Cannot {operation} session while state is {self._state.value!r}; expected {expected}."
        )
