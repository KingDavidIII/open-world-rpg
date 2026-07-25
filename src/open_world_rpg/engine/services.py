"""Typed shared services and subsystem runtime context."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast, runtime_checkable

from open_world_rpg.engine.events import EventBus
from open_world_rpg.engine.subsystems import EngineSubsystem

TService = TypeVar("TService")


class EngineServiceError(RuntimeError):
    """Base exception for engine service failures."""


class DuplicateEngineServiceError(EngineServiceError):
    """Raised when a service type is registered more than once."""


class MissingEngineServiceError(EngineServiceError):
    """Raised when a required service has not been registered."""

    def __init__(self, service_type: type[object]) -> None:
        self.service_type = service_type
        super().__init__(f"Engine service {service_type.__qualname__!r} is not registered.")


class EngineServicesFrozenError(EngineServiceError):
    """Raised when a frozen service collection is modified."""


class EngineContextUnavailableError(EngineServiceError):
    """Raised when a subsystem accesses services before binding."""


class EngineContextBindingError(EngineServiceError):
    """Raised when a subsystem is rebound to another context."""


class SubsystemServiceBindingError(EngineServiceError):
    """Raised when a subsystem rejects its engine context."""

    def __init__(
        self,
        *,
        subsystem_name: str,
        cause: Exception,
    ) -> None:
        self.subsystem_name = subsystem_name
        self.cause = cause

        super().__init__(f"Subsystem {subsystem_name!r} could not bind engine services.")


@dataclass(frozen=True, slots=True)
class EngineServiceRegistration:
    """One explicitly typed engine service registration."""

    service_type: type[object]
    service: object

    def __post_init__(self) -> None:
        resolved_type = _validate_service_type(self.service_type)
        _validate_service_instance(
            resolved_type,
            self.service,
        )


class EngineServices:
    """Mutable-then-frozen collection of exact-type services."""

    __slots__ = ("_frozen", "_services")

    def __init__(self) -> None:
        self._services: dict[type[object], object] = {}
        self._frozen = False

    @property
    def service_count(self) -> int:
        """Return the number of registered services."""
        return len(self._services)

    @property
    def registered_types(self) -> tuple[type[object], ...]:
        """Return service types in registration order."""
        return tuple(self._services)

    @property
    def is_frozen(self) -> bool:
        """Return whether further registration is prohibited."""
        return self._frozen

    def register(
        self,
        service_type: type[TService],
        service: TService,
    ) -> None:
        """Register one service under an exact lookup type."""
        if self._frozen:
            raise EngineServicesFrozenError("Engine services are frozen.")

        resolved_type = _validate_service_type(service_type)
        _validate_service_instance(
            resolved_type,
            service,
        )

        if resolved_type in self._services:
            raise DuplicateEngineServiceError(
                f"Engine service {resolved_type.__qualname__!r} is already registered."
            )

        self._services[resolved_type] = service

    def resolve(
        self,
        service_type: type[TService],
    ) -> TService:
        """Resolve a required service by its exact type."""
        resolved_type = _validate_service_type(service_type)

        try:
            service = self._services[resolved_type]
        except KeyError as exc:
            raise MissingEngineServiceError(resolved_type) from exc

        return cast(TService, service)

    def try_resolve(
        self,
        service_type: type[TService],
    ) -> TService | None:
        """Resolve an optional service by its exact type."""
        resolved_type = _validate_service_type(service_type)
        return cast(
            TService | None,
            self._services.get(resolved_type),
        )

    def freeze(self) -> None:
        """Prevent further service registration."""
        self._frozen = True


@dataclass(frozen=True, slots=True)
class EngineContext:
    """Shared runtime services available to engine subsystems."""

    logger: logging.Logger
    event_bus: EventBus
    services: EngineServices

    def __post_init__(self) -> None:
        if not isinstance(self.logger, logging.Logger):
            raise TypeError("context logger must be a logging.Logger.")

        if not isinstance(self.event_bus, EventBus):
            raise TypeError("context event_bus must be an EventBus.")

        if not isinstance(self.services, EngineServices):
            raise TypeError("context services must be EngineServices.")

        if self.services.try_resolve(logging.Logger) is not self.logger:
            raise ValueError("context services must contain its logger.")

        if self.services.try_resolve(EventBus) is not self.event_bus:
            raise ValueError("context services must contain its event bus.")

    def resolve(
        self,
        service_type: type[TService],
    ) -> TService:
        """Resolve a required shared service."""
        return self.services.resolve(service_type)

    def try_resolve(
        self,
        service_type: type[TService],
    ) -> TService | None:
        """Resolve an optional shared service."""
        return self.services.try_resolve(service_type)


@runtime_checkable
class EngineServiceConsumer(Protocol):
    """Optional contract for context-aware engine subsystems."""

    def bind_services(
        self,
        context: EngineContext,
    ) -> None:
        """Bind the immutable engine runtime context."""


def create_engine_context(
    *,
    logger: logging.Logger,
    event_bus: EventBus,
    registrations: Iterable[EngineServiceRegistration] = (),
) -> EngineContext:
    """Create and freeze an engine runtime context."""
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger.")

    if not isinstance(event_bus, EventBus):
        raise TypeError("event_bus must be an EventBus.")

    services = EngineServices()
    services.register(logging.Logger, logger)
    services.register(EventBus, event_bus)

    for registration in registrations:
        if not isinstance(
            registration,
            EngineServiceRegistration,
        ):
            raise TypeError("registrations must contain EngineServiceRegistration values.")

        services.register(
            registration.service_type,
            registration.service,
        )

    services.freeze()

    return EngineContext(
        logger=logger,
        event_bus=event_bus,
        services=services,
    )


def bind_subsystem_services(
    subsystems: Iterable[EngineSubsystem],
    context: EngineContext,
) -> tuple[str, ...]:
    """Bind context-aware subsystems in supplied order."""
    if not isinstance(context, EngineContext):
        raise TypeError("context must be an EngineContext.")

    bound_names: list[str] = []

    for subsystem in subsystems:
        if not isinstance(subsystem, EngineSubsystem):
            raise TypeError("subsystem must implement EngineSubsystem.")

        if not isinstance(
            subsystem,
            EngineServiceConsumer,
        ):
            continue

        try:
            subsystem.bind_services(context)
        except Exception as exc:
            raise SubsystemServiceBindingError(
                subsystem_name=subsystem.name,
                cause=exc,
            ) from exc

        bound_names.append(subsystem.name)

    return tuple(bound_names)


def _validate_service_type(
    service_type: object,
) -> type[object]:
    if not isinstance(service_type, type):
        raise TypeError("service_type must be a type.")

    return service_type


def _validate_service_instance(
    service_type: type[object],
    service: object,
) -> None:
    if not isinstance(service, service_type):
        raise TypeError(f"service must be an instance of {service_type.__qualname__}.")
