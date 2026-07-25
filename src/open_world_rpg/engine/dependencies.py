"""Dependency-aware engine subsystem construction."""

from __future__ import annotations

from collections.abc import Iterable

from open_world_rpg.engine.services import (
    EngineContext,
    EngineContextBindingError,
    EngineContextUnavailableError,
)
from open_world_rpg.engine.subsystems import (
    DuplicateSubsystemError,
    EngineSubsystem,
    SubsystemRegistryError,
)


class SubsystemDependencyError(SubsystemRegistryError):
    """Base exception for subsystem dependency failures."""


class MissingSubsystemDependencyError(SubsystemDependencyError):
    """Raised when a subsystem requires an unregistered dependency."""

    def __init__(
        self,
        *,
        subsystem_name: str,
        dependency_name: str,
    ) -> None:
        self.subsystem_name = subsystem_name
        self.dependency_name = dependency_name

        super().__init__(
            f"Subsystem {subsystem_name!r} requires unregistered dependency {dependency_name!r}."
        )


class SubsystemDependencyCycleError(SubsystemDependencyError):
    """Raised when subsystem dependencies contain a cycle."""

    def __init__(
        self,
        cycle: tuple[str, ...],
    ) -> None:
        self.cycle = cycle
        rendered_cycle = " -> ".join(cycle)

        super().__init__(f"Subsystem dependency cycle detected: {rendered_cycle}.")


class EngineSubsystemBase:
    """Reusable no-op subsystem with validated identity and dependencies."""

    __slots__ = ("_context", "_dependencies", "_name")

    def __init__(
        self,
        *,
        name: str,
        dependencies: Iterable[str] = (),
    ) -> None:
        self._name = _validate_name(
            name,
            description="subsystem name",
        )
        self._dependencies = _normalise_dependencies(dependencies)
        self._context: EngineContext | None = None

    @property
    def name(self) -> str:
        """Return the unique subsystem name."""
        return self._name

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Return required subsystem names in declaration order."""
        return self._dependencies

    @property
    def services_bound(self) -> bool:
        """Return whether an engine context has been bound."""
        return self._context is not None

    @property
    def context(self) -> EngineContext:
        """Return the bound engine context."""
        if self._context is None:
            raise EngineContextUnavailableError(f"Subsystem {self.name!r} has no engine context.")

        return self._context

    def bind_services(
        self,
        context: EngineContext,
    ) -> None:
        """Bind this subsystem to one engine context."""
        if not isinstance(context, EngineContext):
            raise TypeError("context must be an EngineContext.")

        if self._context is None:
            self._context = context
            return

        if self._context is context:
            return

        raise EngineContextBindingError(
            f"Subsystem {self.name!r} is already bound to another engine context."
        )

    def require_service(
        self,
        service_type: type[object],
    ) -> object:
        """Resolve a required service from the bound context."""
        return self.context.resolve(service_type)

    def start(self) -> None:
        """Acquire resources and initialise the subsystem."""

    def update(self, fixed_delta_seconds: float) -> None:
        """Advance deterministic simulation state."""
        del fixed_delta_seconds

    def render(self, interpolation_alpha: float) -> None:
        """Render interpolated presentation state."""
        del interpolation_alpha

    def stop(self) -> None:
        """Release resources owned by the subsystem."""


def resolve_subsystem_order(
    subsystems: Iterable[EngineSubsystem],
) -> tuple[EngineSubsystem, ...]:
    """Resolve subsystems into stable dependency order."""
    ordered_subsystems: list[EngineSubsystem] = []
    by_name: dict[str, EngineSubsystem] = {}
    dependencies_by_name: dict[str, tuple[str, ...]] = {}

    for subsystem in subsystems:
        if not isinstance(subsystem, EngineSubsystem):
            raise TypeError("subsystem must implement EngineSubsystem.")

        name = _validate_name(
            subsystem.name,
            description="subsystem name",
        )

        if name in by_name:
            raise DuplicateSubsystemError(f"Subsystem {name!r} is already registered.")

        dependencies = _normalise_dependencies(getattr(subsystem, "dependencies", ()))

        ordered_subsystems.append(subsystem)
        by_name[name] = subsystem
        dependencies_by_name[name] = dependencies

    for subsystem_name, dependencies in dependencies_by_name.items():
        for dependency_name in dependencies:
            if dependency_name not in by_name:
                raise MissingSubsystemDependencyError(
                    subsystem_name=subsystem_name,
                    dependency_name=dependency_name,
                )

    registration_order = tuple(subsystem.name for subsystem in ordered_subsystems)
    unresolved = set(registration_order)
    resolved_names: set[str] = set()
    resolved_subsystems: list[EngineSubsystem] = []

    while unresolved:
        ready_names = tuple(
            name
            for name in registration_order
            if name in unresolved
            and all(dependency in resolved_names for dependency in dependencies_by_name[name])
        )

        if not ready_names:
            cycle = _find_dependency_cycle(
                start_name=next(name for name in registration_order if name in unresolved),
                unresolved=unresolved,
                dependencies_by_name=dependencies_by_name,
            )
            raise SubsystemDependencyCycleError(cycle)

        for name in ready_names:
            resolved_subsystems.append(by_name[name])
            resolved_names.add(name)
            unresolved.remove(name)

    return tuple(resolved_subsystems)


def _find_dependency_cycle(
    *,
    start_name: str,
    unresolved: set[str],
    dependencies_by_name: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    path: list[str] = []
    positions: dict[str, int] = {}
    current_name = start_name

    while current_name not in positions:
        positions[current_name] = len(path)
        path.append(current_name)
        current_name = next(
            dependency
            for dependency in dependencies_by_name[current_name]
            if dependency in unresolved
        )

    cycle_start = positions[current_name]
    return (
        *path[cycle_start:],
        current_name,
    )


def _normalise_dependencies(
    dependencies: object,
) -> tuple[str, ...]:
    if isinstance(dependencies, (str, bytes)):
        raise TypeError("subsystem dependencies must be an iterable of strings.")

    if not isinstance(dependencies, Iterable):
        raise TypeError("subsystem dependencies must be an iterable of strings.")

    normalised: list[str] = []
    seen: set[str] = set()

    for dependency in dependencies:
        dependency_name = _validate_name(
            dependency,
            description="subsystem dependency",
        )

        if dependency_name in seen:
            raise ValueError(
                f"Subsystem dependency {dependency_name!r} is declared more than once."
            )

        normalised.append(dependency_name)
        seen.add(dependency_name)

    return tuple(normalised)


def _validate_name(
    value: object,
    *,
    description: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string.")

    if not value.strip():
        raise ValueError(f"{description} cannot be empty.")

    if value != value.strip():
        raise ValueError(f"{description} cannot contain surrounding whitespace.")

    return value
