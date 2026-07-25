"""Deterministic queued event dispatch for engine services."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar, cast

TEvent = TypeVar("TEvent")
_ObjectEventHandler = Callable[[object], None]


class EventBusError(RuntimeError):
    """Base exception for engine event bus failures."""


class EventSubscriptionError(EventBusError):
    """Raised when an event subscription token is invalid."""


class EventDispatchStateError(EventBusError):
    """Raised when an event dispatch operation is re-entered."""


@dataclass(frozen=True, slots=True)
class EventSubscription:
    """Opaque token representing one event handler subscription."""

    event_type: type[object]
    subscription_id: int
    _owner: object = field(repr=False)


@dataclass(frozen=True, slots=True)
class EventDispatchFailure:
    """A handler failure captured during event dispatch."""

    event: object
    event_sequence: int
    subscription: EventSubscription
    error: Exception


@dataclass(frozen=True, slots=True)
class EventDispatchReport:
    """Summary of one event queue dispatch operation."""

    events_dispatched: int
    handler_invocations: int
    failure_count: int
    pending_event_count: int


class EventDispatchError(EventBusError):
    """Raised after one or more event handlers fail."""

    def __init__(
        self,
        *,
        failures: tuple[EventDispatchFailure, ...],
        report: EventDispatchReport,
    ) -> None:
        self.failures = failures
        self.report = report

        failed_handlers = ", ".join(
            (
                f"{failure.subscription.event_type.__qualname__}"
                f"#{failure.subscription.subscription_id}"
            )
            for failure in failures
        )

        super().__init__(
            f"Event dispatch failed for {len(failures)} handler invocation(s): {failed_handlers}."
        )


@dataclass(frozen=True, slots=True)
class _HandlerRegistration:
    subscription: EventSubscription
    handler: _ObjectEventHandler


class EventBus:
    """Queue and dispatch engine events in deterministic FIFO order."""

    __slots__ = (
        "_dispatching",
        "_handlers",
        "_next_subscription_id",
        "_owner",
        "_pending",
    )

    def __init__(self) -> None:
        self._handlers: dict[
            type[object],
            list[_HandlerRegistration],
        ] = {}
        self._pending: deque[object] = deque()
        self._next_subscription_id = 1
        self._owner = object()
        self._dispatching = False

    @property
    def subscription_count(self) -> int:
        """Return the number of active handler subscriptions."""
        return sum(len(registrations) for registrations in self._handlers.values())

    @property
    def subscribed_event_types(
        self,
    ) -> tuple[type[object], ...]:
        """Return subscribed event types in first-subscription order."""
        return tuple(self._handlers)

    @property
    def pending_event_count(self) -> int:
        """Return the number of events waiting for dispatch."""
        return len(self._pending)

    @property
    def is_dispatching(self) -> bool:
        """Return whether the event queue is currently dispatching."""
        return self._dispatching

    def subscribe(
        self,
        event_type: type[TEvent],
        handler: Callable[[TEvent], None],
    ) -> EventSubscription:
        """Subscribe a handler to one exact event type."""
        resolved_event_type = _validate_event_type(event_type)

        if not callable(handler):
            raise TypeError("event handler must be callable.")

        subscription = EventSubscription(
            event_type=resolved_event_type,
            subscription_id=self._next_subscription_id,
            _owner=self._owner,
        )
        self._next_subscription_id += 1

        registration = _HandlerRegistration(
            subscription=subscription,
            handler=cast(_ObjectEventHandler, handler),
        )

        self._handlers.setdefault(
            resolved_event_type,
            [],
        ).append(registration)

        return subscription

    def unsubscribe(
        self,
        subscription: EventSubscription,
    ) -> bool:
        """Remove a subscription, returning whether it was active."""
        if not isinstance(subscription, EventSubscription):
            raise TypeError("subscription must be an EventSubscription.")

        if subscription._owner is not self._owner:
            raise EventSubscriptionError("Subscription belongs to a different event bus.")

        registrations = self._handlers.get(subscription.event_type)
        if registrations is None:
            return False

        for index, registration in enumerate(registrations):
            if registration.subscription.subscription_id != subscription.subscription_id:
                continue

            del registrations[index]

            if not registrations:
                del self._handlers[subscription.event_type]

            return True

        return False

    def publish(self, event: object) -> None:
        """Append an event instance to the deterministic queue."""
        if event is None or isinstance(event, type):
            raise TypeError("event must be a non-null event instance.")

        self._pending.append(event)

    def dispatch_pending(
        self,
        *,
        max_events: int | None = None,
    ) -> EventDispatchReport:
        """Dispatch queued events in FIFO order."""
        if self._dispatching:
            raise EventDispatchStateError("Event dispatch cannot be re-entered.")

        _validate_max_events(max_events)

        events_dispatched = 0
        handler_invocations = 0
        failures: list[EventDispatchFailure] = []

        self._dispatching = True

        try:
            while self._pending and (max_events is None or events_dispatched < max_events):
                event = self._pending.popleft()
                event_sequence = events_dispatched + 1
                registrations = tuple(self._handlers.get(type(event), []))

                for registration in registrations:
                    handler_invocations += 1

                    try:
                        registration.handler(event)
                    except Exception as exc:
                        failures.append(
                            EventDispatchFailure(
                                event=event,
                                event_sequence=event_sequence,
                                subscription=(registration.subscription),
                                error=exc,
                            )
                        )

                events_dispatched += 1
        finally:
            self._dispatching = False

        report = EventDispatchReport(
            events_dispatched=events_dispatched,
            handler_invocations=handler_invocations,
            failure_count=len(failures),
            pending_event_count=len(self._pending),
        )

        if failures:
            raise EventDispatchError(
                failures=tuple(failures),
                report=report,
            )

        return report

    def clear_pending(self) -> int:
        """Discard queued events and return the removed count."""
        removed_count = len(self._pending)
        self._pending.clear()
        return removed_count


def _validate_event_type(
    event_type: object,
) -> type[object]:
    if not isinstance(event_type, type):
        raise TypeError("event_type must be a type.")

    return cast(type[object], event_type)


def _validate_max_events(
    max_events: int | None,
) -> None:
    if max_events is None:
        return

    if type(max_events) is not int:
        raise TypeError("max_events must be an integer or None.")

    if max_events <= 0:
        raise ValueError("max_events must be greater than zero.")
