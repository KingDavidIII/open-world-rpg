"""Tests for typed engine services and subsystem context binding."""

from __future__ import annotations

import logging
from typing import Any, cast

import pytest

from open_world_rpg.engine import (
    DuplicateEngineServiceError,
    EngineContext,
    EngineContextBindingError,
    EngineContextUnavailableError,
    EngineRuntime,
    EngineServiceConsumer,
    EngineServiceRegistration,
    EngineServices,
    EngineServicesFrozenError,
    EngineSubsystemBase,
    EventBus,
    MissingEngineServiceError,
    SubsystemRegistry,
    SubsystemServiceBindingError,
    bind_subsystem_services,
    create_engine_context,
)


class WorldService:
    """Example world-state service."""


class DerivedWorldService(WorldService):
    """Concrete world-state service."""


class PlainSubsystem:
    """Subsystem that does not consume engine services."""

    name = "plain"

    def start(self) -> None:
        pass

    def update(self, fixed_delta_seconds: float) -> None:
        del fixed_delta_seconds

    def render(self, interpolation_alpha: float) -> None:
        del interpolation_alpha

    def stop(self) -> None:
        pass


class RecordingConsumer:
    """Context-aware subsystem test double."""

    def __init__(
        self,
        name: str,
        calls: list[str],
        *,
        failure: Exception | None = None,
    ) -> None:
        self.name = name
        self.calls = calls
        self.failure = failure
        self.context: EngineContext | None = None

    def bind_services(
        self,
        context: EngineContext,
    ) -> None:
        self.calls.append(self.name)

        if self.failure is not None:
            raise self.failure

        self.context = context

    def start(self) -> None:
        pass

    def update(self, fixed_delta_seconds: float) -> None:
        del fixed_delta_seconds

    def render(self, interpolation_alpha: float) -> None:
        del interpolation_alpha

    def stop(self) -> None:
        pass


def create_context(
    *registrations: EngineServiceRegistration,
) -> EngineContext:
    return create_engine_context(
        logger=logging.Logger("test.engine.services"),
        event_bus=EventBus(),
        registrations=registrations,
    )


def test_services_start_empty_and_mutable() -> None:
    services = EngineServices()

    assert services.service_count == 0
    assert services.registered_types == ()
    assert services.is_frozen is False


def test_services_register_and_resolve_exact_type() -> None:
    services = EngineServices()
    world = DerivedWorldService()

    services.register(WorldService, world)

    assert services.service_count == 1
    assert services.registered_types == (WorldService,)
    assert services.resolve(WorldService) is world
    assert services.try_resolve(WorldService) is world


def test_services_do_not_resolve_by_concrete_subclass() -> None:
    services = EngineServices()
    world = DerivedWorldService()
    services.register(WorldService, world)

    assert services.try_resolve(DerivedWorldService) is None

    with pytest.raises(
        MissingEngineServiceError,
        match="DerivedWorldService",
    ) as error:
        services.resolve(DerivedWorldService)

    assert error.value.service_type is DerivedWorldService


@pytest.mark.parametrize(
    "service_type",
    [None, "WorldService", object()],
)
def test_services_reject_invalid_service_type(
    service_type: object,
) -> None:
    services = EngineServices()

    with pytest.raises(TypeError, match="must be a type"):
        services.register(
            cast(Any, service_type),
            cast(Any, WorldService()),
        )


def test_services_reject_mismatched_instance() -> None:
    services = EngineServices()

    with pytest.raises(
        TypeError,
        match="instance of WorldService",
    ):
        services.register(
            WorldService,
            cast(Any, object()),
        )


def test_services_reject_duplicate_type() -> None:
    services = EngineServices()
    services.register(
        WorldService,
        DerivedWorldService(),
    )

    with pytest.raises(
        DuplicateEngineServiceError,
        match="already registered",
    ):
        services.register(
            WorldService,
            DerivedWorldService(),
        )


def test_services_freeze_is_idempotent() -> None:
    services = EngineServices()

    services.freeze()
    services.freeze()

    assert services.is_frozen is True


def test_frozen_services_reject_registration() -> None:
    services = EngineServices()
    services.freeze()

    with pytest.raises(
        EngineServicesFrozenError,
        match="frozen",
    ):
        services.register(
            WorldService,
            DerivedWorldService(),
        )


def test_registration_validates_service_pair() -> None:
    world = DerivedWorldService()
    registration = EngineServiceRegistration(
        WorldService,
        world,
    )

    assert registration.service_type is WorldService
    assert registration.service is world


def test_registration_rejects_invalid_type() -> None:
    with pytest.raises(TypeError, match="must be a type"):
        EngineServiceRegistration(
            cast(Any, "WorldService"),
            WorldService(),
        )


def test_registration_rejects_invalid_instance() -> None:
    with pytest.raises(
        TypeError,
        match="instance of WorldService",
    ):
        EngineServiceRegistration(
            WorldService,
            object(),
        )


def test_create_context_registers_builtin_services() -> None:
    logger = logging.Logger("test.context")
    event_bus = EventBus()

    context = create_engine_context(
        logger=logger,
        event_bus=event_bus,
    )

    assert context.logger is logger
    assert context.event_bus is event_bus
    assert context.services.is_frozen is True
    assert context.resolve(logging.Logger) is logger
    assert context.resolve(EventBus) is event_bus
    assert context.try_resolve(WorldService) is None


def test_create_context_registers_custom_service() -> None:
    world = DerivedWorldService()

    context = create_context(
        EngineServiceRegistration(
            WorldService,
            world,
        )
    )

    assert context.resolve(WorldService) is world
    assert context.services.registered_types == (
        logging.Logger,
        EventBus,
        WorldService,
    )


@pytest.mark.parametrize(
    "logger",
    [None, object()],
)
def test_create_context_rejects_invalid_logger(
    logger: object,
) -> None:
    with pytest.raises(TypeError, match="logger"):
        create_engine_context(
            logger=cast(Any, logger),
            event_bus=EventBus(),
        )


def test_create_context_rejects_invalid_event_bus() -> None:
    with pytest.raises(TypeError, match="event_bus"):
        create_engine_context(
            logger=logging.Logger("test"),
            event_bus=cast(Any, object()),
        )


def test_create_context_rejects_invalid_registration() -> None:
    with pytest.raises(
        TypeError,
        match="EngineServiceRegistration",
    ):
        create_engine_context(
            logger=logging.Logger("test"),
            event_bus=EventBus(),
            registrations=[cast(Any, object())],
        )


def test_context_rejects_invalid_logger() -> None:
    services = EngineServices()
    event_bus = EventBus()
    services.register(EventBus, event_bus)
    services.freeze()

    with pytest.raises(TypeError, match="context logger"):
        EngineContext(
            logger=cast(Any, object()),
            event_bus=event_bus,
            services=services,
        )


def test_context_rejects_invalid_event_bus() -> None:
    services = EngineServices()
    logger = logging.Logger("test")
    services.register(logging.Logger, logger)
    services.freeze()

    with pytest.raises(TypeError, match="context event_bus"):
        EngineContext(
            logger=logger,
            event_bus=cast(Any, object()),
            services=services,
        )


def test_context_rejects_invalid_services_object() -> None:
    with pytest.raises(TypeError, match="EngineServices"):
        EngineContext(
            logger=logging.Logger("test"),
            event_bus=EventBus(),
            services=cast(Any, object()),
        )


def test_context_requires_registered_logger() -> None:
    services = EngineServices()
    event_bus = EventBus()
    services.register(EventBus, event_bus)
    services.freeze()

    with pytest.raises(ValueError, match="logger"):
        EngineContext(
            logger=logging.Logger("test"),
            event_bus=event_bus,
            services=services,
        )


def test_context_requires_registered_event_bus() -> None:
    services = EngineServices()
    logger = logging.Logger("test")
    services.register(logging.Logger, logger)
    services.freeze()

    with pytest.raises(ValueError, match="event bus"):
        EngineContext(
            logger=logger,
            event_bus=EventBus(),
            services=services,
        )


def test_base_subsystem_starts_unbound() -> None:
    subsystem = EngineSubsystemBase(name="world")

    assert subsystem.services_bound is False

    with pytest.raises(
        EngineContextUnavailableError,
        match="no engine context",
    ):
        _ = subsystem.context


def test_base_subsystem_binds_and_resolves_service() -> None:
    world = DerivedWorldService()
    context = create_context(
        EngineServiceRegistration(
            WorldService,
            world,
        )
    )
    subsystem = EngineSubsystemBase(name="world")

    subsystem.bind_services(context)

    assert subsystem.services_bound is True
    assert subsystem.context is context
    assert subsystem.require_service(WorldService) is world


def test_base_subsystem_rejects_invalid_context() -> None:
    subsystem = EngineSubsystemBase(name="world")

    with pytest.raises(TypeError, match="EngineContext"):
        subsystem.bind_services(cast(Any, object()))


def test_base_subsystem_allows_same_context_again() -> None:
    context = create_context()
    subsystem = EngineSubsystemBase(name="world")

    subsystem.bind_services(context)
    subsystem.bind_services(context)

    assert subsystem.context is context


def test_base_subsystem_rejects_different_context() -> None:
    subsystem = EngineSubsystemBase(name="world")
    subsystem.bind_services(create_context())

    with pytest.raises(
        EngineContextBindingError,
        match="another engine context",
    ):
        subsystem.bind_services(create_context())


def test_consumer_protocol_is_runtime_checkable() -> None:
    consumer = RecordingConsumer("world", [])

    assert isinstance(consumer, EngineServiceConsumer)
    assert not isinstance(
        PlainSubsystem(),
        EngineServiceConsumer,
    )


def test_bind_subsystems_preserves_order_and_skips_plain() -> None:
    calls: list[str] = []
    context = create_context()
    first = RecordingConsumer("first", calls)
    second = RecordingConsumer("second", calls)

    bound = bind_subsystem_services(
        [
            first,
            PlainSubsystem(),
            second,
        ],
        context,
    )

    assert bound == ("first", "second")
    assert calls == ["first", "second"]
    assert first.context is context
    assert second.context is context


def test_bind_subsystems_rejects_invalid_context() -> None:
    with pytest.raises(TypeError, match="EngineContext"):
        bind_subsystem_services(
            [],
            cast(Any, object()),
        )


def test_bind_subsystems_rejects_invalid_subsystem() -> None:
    with pytest.raises(
        TypeError,
        match="implement EngineSubsystem",
    ):
        bind_subsystem_services(
            [cast(Any, object())],
            create_context(),
        )


def test_binding_failure_is_wrapped() -> None:
    failure = RuntimeError("binding failure")
    subsystem = RecordingConsumer(
        "world",
        [],
        failure=failure,
    )

    with pytest.raises(
        SubsystemServiceBindingError,
        match="'world'",
    ) as error:
        bind_subsystem_services(
            [subsystem],
            create_context(),
        )

    assert error.value.subsystem_name == "world"
    assert error.value.cause is failure


def test_registry_exposes_subsystems_in_order() -> None:
    first = PlainSubsystem()
    second = RecordingConsumer("second", [])
    registry = SubsystemRegistry([first, second])

    assert registry.subsystems == (first, second)


def test_runtime_creates_context_and_binds_base_subsystem() -> None:
    subsystem = EngineSubsystemBase(name="world")
    logger = logging.Logger("test.runtime")
    event_bus = EventBus()

    runtime = EngineRuntime(
        registry=SubsystemRegistry([subsystem]),
        logger=logger,
        event_bus=event_bus,
    )

    assert runtime.context.logger is logger
    assert runtime.context.event_bus is event_bus
    assert runtime.context.services.is_frozen
    assert subsystem.context is runtime.context


def test_runtime_accepts_explicit_context() -> None:
    context = create_context()

    runtime = EngineRuntime(
        registry=SubsystemRegistry(),
        context=context,
    )

    assert runtime.context is context
    assert runtime.logger is context.logger
    assert runtime.event_bus is context.event_bus


def test_runtime_rejects_invalid_context() -> None:
    with pytest.raises(TypeError, match="EngineContext"):
        EngineRuntime(
            registry=SubsystemRegistry(),
            context=cast(Any, object()),
        )


def test_runtime_rejects_logger_context_mismatch() -> None:
    context = create_context()

    with pytest.raises(
        ValueError,
        match=r"context\.logger",
    ):
        EngineRuntime(
            registry=SubsystemRegistry(),
            context=context,
            logger=logging.Logger("other"),
        )


def test_runtime_rejects_event_bus_context_mismatch() -> None:
    context = create_context()

    with pytest.raises(
        ValueError,
        match=r"context\.event_bus",
    ):
        EngineRuntime(
            registry=SubsystemRegistry(),
            context=context,
            event_bus=EventBus(),
        )
