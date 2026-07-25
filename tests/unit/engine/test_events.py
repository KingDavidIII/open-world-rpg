"""Tests for deterministic queued engine event dispatch."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import pytest

from open_world_rpg.engine import (
    EventBus,
    EventDispatchError,
    EventDispatchStateError,
    EventSubscription,
    EventSubscriptionError,
)


@dataclass(frozen=True, slots=True)
class InputEvent:
    key: str


@dataclass(frozen=True, slots=True)
class DerivedInputEvent(InputEvent):
    source: str


@dataclass(frozen=True, slots=True)
class WorldEvent:
    action: str


class CallableHandler:
    """Callable event handler test double."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def __call__(self, event: InputEvent) -> None:
        self.calls.append(event.key)


def test_bus_starts_empty_and_idle() -> None:
    bus = EventBus()

    assert bus.subscription_count == 0
    assert bus.subscribed_event_types == ()
    assert bus.pending_event_count == 0
    assert bus.is_dispatching is False


def test_subscribe_returns_owned_token() -> None:
    bus = EventBus()

    subscription = bus.subscribe(
        InputEvent,
        lambda event: None,
    )

    assert isinstance(subscription, EventSubscription)
    assert subscription.event_type is InputEvent
    assert subscription.subscription_id == 1
    assert bus.subscription_count == 1
    assert bus.subscribed_event_types == (InputEvent,)


def test_subscription_ids_are_monotonic() -> None:
    bus = EventBus()

    first = bus.subscribe(
        InputEvent,
        lambda event: None,
    )
    second = bus.subscribe(
        InputEvent,
        lambda event: None,
    )

    assert first.subscription_id == 1
    assert second.subscription_id == 2


@pytest.mark.parametrize(
    "event_type",
    [None, "InputEvent", object()],
)
def test_subscribe_rejects_invalid_event_type(
    event_type: object,
) -> None:
    bus = EventBus()

    with pytest.raises(TypeError, match="event_type"):
        bus.subscribe(
            cast(Any, event_type),
            lambda event: None,
        )


@pytest.mark.parametrize(
    "handler",
    [None, 123, object()],
)
def test_subscribe_rejects_non_callable_handler(
    handler: object,
) -> None:
    bus = EventBus()

    with pytest.raises(TypeError, match="callable"):
        bus.subscribe(
            InputEvent,
            cast(Any, handler),
        )


def test_callable_object_can_subscribe() -> None:
    bus = EventBus()
    calls: list[str] = []

    bus.subscribe(InputEvent, CallableHandler(calls))
    bus.publish(InputEvent("jump"))

    report = bus.dispatch_pending()

    assert calls == ["jump"]
    assert report.handler_invocations == 1


@pytest.mark.parametrize(
    "event",
    [None, InputEvent],
)
def test_publish_rejects_invalid_event_instance(
    event: object,
) -> None:
    bus = EventBus()

    with pytest.raises(
        TypeError,
        match="non-null event instance",
    ):
        bus.publish(cast(Any, event))


def test_publish_queues_event() -> None:
    bus = EventBus()

    bus.publish(InputEvent("jump"))

    assert bus.pending_event_count == 1


def test_dispatch_without_subscribers_consumes_event() -> None:
    bus = EventBus()
    bus.publish(InputEvent("jump"))

    report = bus.dispatch_pending()

    assert report.events_dispatched == 1
    assert report.handler_invocations == 0
    assert report.failure_count == 0
    assert report.pending_event_count == 0
    assert bus.pending_event_count == 0


def test_empty_dispatch_returns_empty_report() -> None:
    bus = EventBus()

    report = bus.dispatch_pending()

    assert report.events_dispatched == 0
    assert report.handler_invocations == 0
    assert report.failure_count == 0
    assert report.pending_event_count == 0


def test_handlers_run_in_subscription_order() -> None:
    bus = EventBus()
    calls: list[str] = []

    bus.subscribe(
        InputEvent,
        lambda event: calls.append("first"),
    )
    bus.subscribe(
        InputEvent,
        lambda event: calls.append("second"),
    )
    bus.publish(InputEvent("jump"))

    report = bus.dispatch_pending()

    assert calls == ["first", "second"]
    assert report.events_dispatched == 1
    assert report.handler_invocations == 2


def test_events_dispatch_in_fifo_order() -> None:
    bus = EventBus()
    calls: list[str] = []

    bus.subscribe(
        InputEvent,
        lambda event: calls.append(f"input:{event.key}"),
    )
    bus.subscribe(
        WorldEvent,
        lambda event: calls.append(f"world:{event.action}"),
    )

    bus.publish(InputEvent("jump"))
    bus.publish(WorldEvent("spawn"))
    bus.publish(InputEvent("interact"))

    report = bus.dispatch_pending()

    assert calls == [
        "input:jump",
        "world:spawn",
        "input:interact",
    ]
    assert report.events_dispatched == 3
    assert report.handler_invocations == 3


def test_dispatch_uses_exact_runtime_event_type() -> None:
    bus = EventBus()
    calls: list[str] = []

    bus.subscribe(
        InputEvent,
        lambda event: calls.append("base"),
    )
    bus.subscribe(
        DerivedInputEvent,
        lambda event: calls.append("derived"),
    )

    bus.publish(
        DerivedInputEvent(
            key="jump",
            source="keyboard",
        )
    )
    bus.dispatch_pending()

    assert calls == ["derived"]


@pytest.mark.parametrize(
    "value",
    [True, 1.5, "1"],
)
def test_dispatch_rejects_invalid_limit_type(
    value: object,
) -> None:
    bus = EventBus()

    with pytest.raises(TypeError, match="max_events"):
        bus.dispatch_pending(max_events=cast(Any, value))


@pytest.mark.parametrize("value", [0, -1])
def test_dispatch_rejects_non_positive_limit(
    value: int,
) -> None:
    bus = EventBus()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        bus.dispatch_pending(max_events=value)


def test_bounded_dispatch_leaves_remaining_events() -> None:
    bus = EventBus()
    calls: list[str] = []

    bus.subscribe(
        InputEvent,
        lambda event: calls.append(event.key),
    )

    bus.publish(InputEvent("one"))
    bus.publish(InputEvent("two"))
    bus.publish(InputEvent("three"))

    report = bus.dispatch_pending(max_events=2)

    assert calls == ["one", "two"]
    assert report.events_dispatched == 2
    assert report.pending_event_count == 1
    assert bus.pending_event_count == 1

    second_report = bus.dispatch_pending()

    assert calls == ["one", "two", "three"]
    assert second_report.events_dispatched == 1
    assert second_report.pending_event_count == 0


def test_clear_pending_returns_removed_count() -> None:
    bus = EventBus()
    bus.publish(InputEvent("one"))
    bus.publish(InputEvent("two"))

    removed_count = bus.clear_pending()

    assert removed_count == 2
    assert bus.pending_event_count == 0


def test_clear_empty_queue_returns_zero() -> None:
    bus = EventBus()

    assert bus.clear_pending() == 0


def test_unsubscribe_rejects_invalid_token() -> None:
    bus = EventBus()

    with pytest.raises(
        TypeError,
        match="EventSubscription",
    ):
        bus.unsubscribe(cast(Any, object()))


def test_unsubscribe_rejects_foreign_subscription() -> None:
    first_bus = EventBus()
    second_bus = EventBus()

    subscription = first_bus.subscribe(
        InputEvent,
        lambda event: None,
    )

    with pytest.raises(
        EventSubscriptionError,
        match="different event bus",
    ):
        second_bus.unsubscribe(subscription)


def test_unsubscribe_removes_active_handler() -> None:
    bus = EventBus()
    calls: list[str] = []

    subscription = bus.subscribe(
        InputEvent,
        lambda event: calls.append(event.key),
    )

    assert bus.unsubscribe(subscription) is True
    assert bus.subscription_count == 0
    assert bus.subscribed_event_types == ()

    bus.publish(InputEvent("jump"))
    bus.dispatch_pending()

    assert calls == []


def test_unsubscribe_is_idempotent() -> None:
    bus = EventBus()

    subscription = bus.subscribe(
        InputEvent,
        lambda event: None,
    )

    assert bus.unsubscribe(subscription) is True
    assert bus.unsubscribe(subscription) is False


def test_unsubscribe_unknown_owned_id_returns_false() -> None:
    bus = EventBus()

    subscription = bus.subscribe(
        InputEvent,
        lambda event: None,
    )
    unknown_subscription = replace(
        subscription,
        subscription_id=999,
    )

    assert bus.unsubscribe(unknown_subscription) is False
    assert bus.subscription_count == 1


def test_unsubscribe_preserves_other_handlers() -> None:
    bus = EventBus()
    calls: list[str] = []

    first = bus.subscribe(
        InputEvent,
        lambda event: calls.append("first"),
    )
    bus.subscribe(
        InputEvent,
        lambda event: calls.append("second"),
    )

    assert bus.unsubscribe(first) is True
    assert bus.subscription_count == 1
    assert bus.subscribed_event_types == (InputEvent,)

    bus.publish(InputEvent("jump"))
    bus.dispatch_pending()

    assert calls == ["second"]


def test_event_type_order_tracks_first_subscription() -> None:
    bus = EventBus()

    input_subscription = bus.subscribe(
        InputEvent,
        lambda event: None,
    )
    bus.subscribe(
        WorldEvent,
        lambda event: None,
    )

    assert bus.subscribed_event_types == (
        InputEvent,
        WorldEvent,
    )

    bus.unsubscribe(input_subscription)

    assert bus.subscribed_event_types == (WorldEvent,)


def test_unsubscribe_during_dispatch_uses_event_snapshot() -> None:
    bus = EventBus()
    calls: list[str] = []
    second_subscription: EventSubscription | None = None

    def first_handler(event: InputEvent) -> None:
        calls.append(f"first:{event.key}")

        if event.key == "one":
            assert second_subscription is not None
            assert bus.unsubscribe(second_subscription)

    def second_handler(event: InputEvent) -> None:
        calls.append(f"second:{event.key}")

    bus.subscribe(InputEvent, first_handler)
    second_subscription = bus.subscribe(
        InputEvent,
        second_handler,
    )

    bus.publish(InputEvent("one"))
    bus.publish(InputEvent("two"))
    bus.dispatch_pending()

    assert calls == [
        "first:one",
        "second:one",
        "first:two",
    ]


def test_subscribe_during_dispatch_applies_to_later_event() -> None:
    bus = EventBus()
    calls: list[str] = []

    def late_handler(event: InputEvent) -> None:
        calls.append(f"late:{event.key}")

    def first_handler(event: InputEvent) -> None:
        calls.append(f"first:{event.key}")

        if event.key == "one":
            bus.subscribe(InputEvent, late_handler)

    bus.subscribe(InputEvent, first_handler)
    bus.publish(InputEvent("one"))
    bus.publish(InputEvent("two"))

    bus.dispatch_pending()

    assert calls == [
        "first:one",
        "first:two",
        "late:two",
    ]


def test_handler_can_publish_follow_up_event() -> None:
    bus = EventBus()
    calls: list[str] = []

    def handler(event: InputEvent) -> None:
        calls.append(event.key)

        if event.key == "one":
            bus.publish(InputEvent("two"))

    bus.subscribe(InputEvent, handler)
    bus.publish(InputEvent("one"))

    report = bus.dispatch_pending()

    assert calls == ["one", "two"]
    assert report.events_dispatched == 2
    assert report.handler_invocations == 2
    assert report.pending_event_count == 0


def test_nested_dispatch_is_rejected() -> None:
    bus = EventBus()
    calls: list[str] = []

    def handler(event: InputEvent) -> None:
        calls.append(event.key)

        with pytest.raises(
            EventDispatchStateError,
            match="cannot be re-entered",
        ):
            bus.dispatch_pending()

    bus.subscribe(InputEvent, handler)
    bus.publish(InputEvent("jump"))

    report = bus.dispatch_pending()

    assert calls == ["jump"]
    assert report.failure_count == 0
    assert bus.is_dispatching is False


def test_handler_failures_are_aggregated_after_dispatch() -> None:
    bus = EventBus()
    calls: list[str] = []

    def failing_handler(event: InputEvent) -> None:
        calls.append(f"fail:{event.key}")
        raise RuntimeError(f"{event.key} failure")

    def successful_handler(event: InputEvent) -> None:
        calls.append(f"success:{event.key}")

    first_subscription = bus.subscribe(
        InputEvent,
        failing_handler,
    )
    bus.subscribe(InputEvent, successful_handler)

    bus.publish(InputEvent("one"))
    bus.publish(InputEvent("two"))

    with pytest.raises(
        EventDispatchError,
        match="2 handler invocation",
    ) as error:
        bus.dispatch_pending()

    assert calls == [
        "fail:one",
        "success:one",
        "fail:two",
        "success:two",
    ]
    assert len(error.value.failures) == 2
    assert all(failure.subscription == first_subscription for failure in error.value.failures)
    assert [failure.event_sequence for failure in error.value.failures] == [1, 2]
    assert all(isinstance(failure.error, RuntimeError) for failure in error.value.failures)
    assert error.value.report.events_dispatched == 2
    assert error.value.report.handler_invocations == 4
    assert error.value.report.failure_count == 2
    assert error.value.report.pending_event_count == 0
    assert bus.pending_event_count == 0
    assert bus.is_dispatching is False


def test_failed_bounded_dispatch_preserves_remaining_queue() -> None:
    bus = EventBus()

    def failing_handler(event: InputEvent) -> None:
        raise RuntimeError(event.key)

    bus.subscribe(InputEvent, failing_handler)
    bus.publish(InputEvent("one"))
    bus.publish(InputEvent("two"))
    bus.publish(InputEvent("three"))

    with pytest.raises(EventDispatchError) as error:
        bus.dispatch_pending(max_events=2)

    assert error.value.report.events_dispatched == 2
    assert error.value.report.handler_invocations == 2
    assert error.value.report.failure_count == 2
    assert error.value.report.pending_event_count == 1
    assert bus.pending_event_count == 1
    assert bus.is_dispatching is False


def test_dispatch_recovers_after_handler_failure() -> None:
    bus = EventBus()
    calls: list[str] = []

    def failing_handler(event: InputEvent) -> None:
        raise RuntimeError(event.key)

    subscription = bus.subscribe(
        InputEvent,
        failing_handler,
    )
    bus.publish(InputEvent("one"))

    with pytest.raises(EventDispatchError):
        bus.dispatch_pending()

    assert bus.unsubscribe(subscription)
    bus.subscribe(
        InputEvent,
        lambda event: calls.append(event.key),
    )
    bus.publish(InputEvent("two"))

    report = bus.dispatch_pending()

    assert calls == ["two"]
    assert report.events_dispatched == 1
    assert report.handler_invocations == 1
    assert report.failure_count == 0
