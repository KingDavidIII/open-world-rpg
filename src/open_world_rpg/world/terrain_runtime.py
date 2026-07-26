"""Controlled terrain lifecycle orchestration over immutable chunk metadata."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NoReturn

from open_world_rpg.core.diagnostics import LOGGER_NAME
from open_world_rpg.engine.events import EventBus
from open_world_rpg.world.coordinates import ChunkCoordinate, RegionCoordinate
from open_world_rpg.world.model import WorldSpecification
from open_world_rpg.world.spatial_state import (
    ChunkMetadata,
    ChunkState,
    ChunkTransitionError,
)
from open_world_rpg.world.terrain import ChunkTerrain, TerrainElevation, TerrainType
from open_world_rpg.world.terrain_service import (
    TerrainGenerationService,
    TerrainGenerationServiceSnapshot,
)


class TerrainRuntimeError(RuntimeError):
    """Base error for controlled terrain lifecycle operations."""


class TerrainRuntimeMissingChunkError(TerrainRuntimeError):
    """Raised when a coordinate is not tracked by the runtime."""


class TerrainRuntimeInvalidOperationError(TerrainRuntimeError):
    """Raised when an operation is invalid for the current chunk state."""


class TerrainGenerationInProgressError(TerrainRuntimeInvalidOperationError):
    """Raised when generation is already in progress for a chunk."""


class IncompatibleTerrainRuntimeError(TerrainRuntimeError):
    """Raised when runtime construction dependencies are incompatible."""


class TerrainUnavailableError(TerrainRuntimeError):
    """Raised when tracked chunk terrain is not cached."""


class TerrainRuntimeGenerationError(TerrainRuntimeError):
    """Raised after a failed generation transitions metadata to FAILED."""


class TerrainRuntimeRepositoryError(TerrainRuntimeError):
    """Raised when eviction cannot be completed atomically."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainChunkDeclared:
    """Event published after new chunk metadata is declared."""

    coordinate: ChunkCoordinate
    region_coordinate: RegionCoordinate
    previous_state: ChunkState | None
    current_state: ChunkState
    runtime_revision: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainChunkGenerationStarted:
    """Event published after metadata enters GENERATING."""

    coordinate: ChunkCoordinate
    terrain_seed: int
    runtime_revision: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainChunkGenerated:
    """Event published after terrain is cached and metadata becomes READY."""

    coordinate: ChunkCoordinate
    terrain_seed: int
    minimum_elevation: TerrainElevation
    maximum_elevation: TerrainElevation
    terrain_type_counts: Mapping[TerrainType, int]
    runtime_revision: int
    repository_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terrain_type_counts",
            MappingProxyType(dict(self.terrain_type_counts)),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainChunkActivated:
    """Event published after chunk activation."""

    coordinate: ChunkCoordinate
    previous_state: ChunkState
    current_state: ChunkState
    runtime_revision: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainChunkSuspended:
    """Event published after active chunk suspension."""

    coordinate: ChunkCoordinate
    previous_state: ChunkState
    current_state: ChunkState
    runtime_revision: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainChunkUnloaded:
    """Event published after eviction and transition to UNLOADED."""

    coordinate: ChunkCoordinate
    previous_state: ChunkState
    current_state: ChunkState
    runtime_revision: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainChunkFailed:
    """Event published after transition to FAILED without exception objects."""

    coordinate: ChunkCoordinate
    previous_state: ChunkState
    current_state: ChunkState
    runtime_revision: int
    error_type_name: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TerrainRuntimeSnapshot:
    """Immutable runtime, lifecycle-count, and terrain-service projection."""

    revision: int
    tracked_chunk_count: int
    coordinates: tuple[ChunkCoordinate, ...]
    state_counts: Mapping[ChunkState, int]
    terrain_service: TerrainGenerationServiceSnapshot

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer.")
        if self.revision < 0:
            raise ValueError("revision must be greater than or equal to zero.")
        if isinstance(self.tracked_chunk_count, bool) or not isinstance(
            self.tracked_chunk_count, int
        ):
            raise TypeError("tracked_chunk_count must be an integer.")
        if self.tracked_chunk_count < 0:
            raise ValueError("tracked_chunk_count must be greater than or equal to zero.")
        if not isinstance(self.coordinates, tuple) or not all(
            isinstance(coordinate, ChunkCoordinate) for coordinate in self.coordinates
        ):
            raise TypeError("coordinates must be a tuple of ChunkCoordinate values.")
        if len(self.coordinates) != self.tracked_chunk_count:
            raise ValueError("coordinate count must match tracked_chunk_count.")
        counts = dict(self.state_counts)
        if tuple(counts) != tuple(ChunkState):
            raise ValueError("state_counts must contain every ChunkState in declaration order.")
        for count in counts.values():
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("state_counts values must be integers.")
            if count < 0:
                raise ValueError("state_counts values must be greater than or equal to zero.")
        if sum(counts.values()) != self.tracked_chunk_count:
            raise ValueError("state count sum must match tracked_chunk_count.")
        if not isinstance(self.terrain_service, TerrainGenerationServiceSnapshot):
            raise TypeError("terrain_service must be a TerrainGenerationServiceSnapshot.")
        object.__setattr__(self, "state_counts", MappingProxyType(counts))


class TerrainRuntime:
    """Coordinate chunk lifecycle, terrain cache access, and immutable events."""

    __slots__ = (
        "_event_bus",
        "_logger",
        "_metadata",
        "_revision",
        "_service",
        "_specification",
    )

    def __init__(
        self,
        *,
        specification: WorldSpecification,
        service: TerrainGenerationService,
        event_bus: EventBus | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(specification, WorldSpecification):
            raise TypeError("specification must be a WorldSpecification.")
        if not isinstance(service, TerrainGenerationService):
            raise TypeError("service must be a TerrainGenerationService.")
        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None.")
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("logger must be a logging.Logger or None.")
        if (
            service.specification != specification
            or service.config.generation_format_version != specification.generation_format_version
        ):
            raise IncompatibleTerrainRuntimeError(
                "Terrain service must match the runtime world specification."
            )
        self._specification = specification
        self._service = service
        self._event_bus = event_bus
        self._logger = logging.getLogger(LOGGER_NAME) if logger is None else logger
        self._metadata: dict[ChunkCoordinate, ChunkMetadata] = {}
        self._revision = 0

    @property
    def specification(self) -> WorldSpecification:
        """Return the immutable world specification."""
        return self._specification

    @property
    def service(self) -> TerrainGenerationService:
        """Return the controlled terrain generation service."""
        return self._service

    @property
    def revision(self) -> int:
        """Return the number of committed metadata state changes."""
        return self._revision

    @property
    def logger(self) -> logging.Logger:
        """Return the logger used for structured terrain diagnostics."""
        return self._logger

    def declare(self, coordinate: ChunkCoordinate) -> ChunkMetadata:
        """Declare missing metadata; an existing coordinate is a no-op."""
        self._validate_coordinate(coordinate)
        existing = self._metadata.get(coordinate)
        if existing is not None:
            return existing
        metadata = ChunkMetadata.create(
            world_seed=self._specification.seed,
            coordinate=coordinate,
        )
        self._metadata[coordinate] = metadata
        self._revision += 1
        self._publish(
            TerrainChunkDeclared(
                coordinate=coordinate,
                region_coordinate=metadata.region_coordinate,
                previous_state=None,
                current_state=metadata.state,
                runtime_revision=self._revision,
            )
        )
        self._log(
            level=logging.INFO,
            event="terrain.chunk_declared",
            message="Terrain chunk declared.",
            metadata=metadata,
        )
        return metadata

    def generate(self, coordinate: ChunkCoordinate) -> ChunkTerrain:
        """Generate tracked DECLARED or UNLOADED terrain."""
        metadata = self.metadata_at(coordinate)
        self._require_generation_state(metadata)
        generating = self._transition(metadata, ChunkState.GENERATING)
        self._publish(
            TerrainChunkGenerationStarted(
                coordinate=coordinate,
                terrain_seed=generating.terrain_seed,
                runtime_revision=self._revision,
            )
        )
        self._log(
            level=logging.DEBUG,
            event="terrain.generation_started",
            message="Terrain generation started.",
            metadata=generating,
            previous_state=metadata.state,
        )
        try:
            terrain = self._service.generate_new(coordinate)
        except Exception as error:
            failed = self._transition(generating, ChunkState.FAILED)
            self._publish(
                TerrainChunkFailed(
                    coordinate=coordinate,
                    previous_state=generating.state,
                    current_state=failed.state,
                    runtime_revision=self._revision,
                    error_type_name=type(error).__name__,
                )
            )
            self._log(
                level=logging.ERROR,
                event="terrain.chunk_failed",
                message="Terrain chunk generation failed.",
                metadata=failed,
                previous_state=generating.state,
            )
            raise TerrainRuntimeGenerationError(
                f"Terrain generation failed for chunk ({coordinate.x}, {coordinate.y})."
            ) from error

        ready = self._transition(generating, ChunkState.READY)
        self._publish(
            TerrainChunkGenerated(
                coordinate=coordinate,
                terrain_seed=ready.terrain_seed,
                minimum_elevation=terrain.minimum_elevation,
                maximum_elevation=terrain.maximum_elevation,
                terrain_type_counts=terrain.terrain_type_counts,
                runtime_revision=self._revision,
                repository_revision=self._service.snapshot().repository_revision,
            )
        )
        self._log(
            level=logging.INFO,
            event="terrain.chunk_generated",
            message="Terrain chunk generated.",
            metadata=ready,
            previous_state=generating.state,
            terrain=terrain,
        )
        return terrain

    def get_or_generate(self, coordinate: ChunkCoordinate) -> ChunkTerrain:
        """Declare missing chunks, then return cache or follow generation."""
        self._validate_coordinate(coordinate)
        metadata = self._metadata.get(coordinate)
        if metadata is None:
            metadata = self.declare(coordinate)
        if metadata.state in (ChunkState.READY, ChunkState.ACTIVE, ChunkState.SUSPENDED):
            return self.terrain_at(coordinate)
        if metadata.state in (ChunkState.DECLARED, ChunkState.UNLOADED):
            return self.generate(coordinate)
        if metadata.state is ChunkState.GENERATING:
            raise TerrainGenerationInProgressError(
                f"Terrain generation is already in progress for chunk "
                f"({metadata.coordinate.x}, {metadata.coordinate.y})."
            )
        self._raise_invalid(operation="generate", metadata=metadata)

    def activate(self, coordinate: ChunkCoordinate) -> ChunkMetadata:
        """Activate READY or SUSPENDED metadata; ACTIVE is a no-op."""
        metadata = self.metadata_at(coordinate)
        if metadata.state is ChunkState.ACTIVE:
            return metadata
        if metadata.state not in (ChunkState.READY, ChunkState.SUSPENDED):
            self._raise_invalid(operation="activate", metadata=metadata)
        transitioned = self._transition(metadata, ChunkState.ACTIVE)
        self._publish_lifecycle(TerrainChunkActivated, metadata, transitioned)
        self._log(
            level=logging.INFO,
            event="terrain.chunk_activated",
            message="Terrain chunk activated.",
            metadata=transitioned,
            previous_state=metadata.state,
        )
        return transitioned

    def suspend(self, coordinate: ChunkCoordinate) -> ChunkMetadata:
        """Suspend ACTIVE metadata; SUSPENDED is a no-op."""
        metadata = self.metadata_at(coordinate)
        if metadata.state is ChunkState.SUSPENDED:
            return metadata
        if metadata.state is not ChunkState.ACTIVE:
            self._raise_invalid(operation="suspend", metadata=metadata)
        transitioned = self._transition(metadata, ChunkState.SUSPENDED)
        self._publish_lifecycle(TerrainChunkSuspended, metadata, transitioned)
        self._log(
            level=logging.INFO,
            event="terrain.chunk_suspended",
            message="Terrain chunk suspended.",
            metadata=transitioned,
            previous_state=metadata.state,
        )
        return transitioned

    def unload(self, coordinate: ChunkCoordinate) -> ChunkMetadata:
        """Evict first, then atomically publish the UNLOADED metadata state."""
        metadata = self.metadata_at(coordinate)
        if metadata.state is ChunkState.UNLOADED:
            return metadata
        if metadata.state not in (ChunkState.READY, ChunkState.SUSPENDED):
            self._raise_invalid(operation="unload", metadata=metadata)
        try:
            self._service.evict(coordinate)
        except Exception as error:
            raise TerrainRuntimeRepositoryError(
                f"Terrain eviction failed for chunk ({coordinate.x}, {coordinate.y})."
            ) from error
        transitioned = self._transition(metadata, ChunkState.UNLOADED)
        self._publish_lifecycle(TerrainChunkUnloaded, metadata, transitioned)
        self._log(
            level=logging.INFO,
            event="terrain.chunk_unloaded",
            message="Terrain chunk unloaded.",
            metadata=transitioned,
            previous_state=metadata.state,
        )
        return transitioned

    def fail(self, coordinate: ChunkCoordinate) -> ChunkMetadata:
        """Transition any non-terminal tracked chunk to FAILED."""
        metadata = self.metadata_at(coordinate)
        if metadata.state is ChunkState.FAILED:
            return metadata
        transitioned = self._transition(metadata, ChunkState.FAILED)
        self._publish(
            TerrainChunkFailed(
                coordinate=coordinate,
                previous_state=metadata.state,
                current_state=transitioned.state,
                runtime_revision=self._revision,
                error_type_name="ExplicitTerrainFailure",
            )
        )
        self._log(
            level=logging.ERROR,
            event="terrain.chunk_failed",
            message="Terrain chunk failed.",
            metadata=transitioned,
            previous_state=metadata.state,
        )
        return transitioned

    def terrain_at(self, coordinate: ChunkCoordinate) -> ChunkTerrain:
        """Return cached terrain for a tracked coordinate."""
        self.metadata_at(coordinate)
        if not self._service.contains(coordinate):
            raise TerrainUnavailableError(
                f"Terrain for chunk ({coordinate.x}, {coordinate.y}) is unavailable."
            )
        return self._service.get(coordinate)

    def metadata_at(self, coordinate: ChunkCoordinate) -> ChunkMetadata:
        """Return tracked immutable metadata or raise an explicit error."""
        self._validate_coordinate(coordinate)
        try:
            return self._metadata[coordinate]
        except KeyError as error:
            raise TerrainRuntimeMissingChunkError(
                f"Chunk ({coordinate.x}, {coordinate.y}) is not tracked."
            ) from error

    def contains(self, coordinate: ChunkCoordinate) -> bool:
        """Return whether metadata is tracked without mutation."""
        self._validate_coordinate(coordinate)
        return coordinate in self._metadata

    def coordinates(self) -> tuple[ChunkCoordinate, ...]:
        """Return tracked coordinates by increasing y, then x."""
        return tuple(sorted(self._metadata, key=lambda coordinate: (coordinate.y, coordinate.x)))

    def snapshot(self) -> TerrainRuntimeSnapshot:
        """Return an immutable projection without mutating runtime or service."""
        coordinates = self.coordinates()
        counts = dict.fromkeys(ChunkState, 0)
        for metadata in self._metadata.values():
            counts[metadata.state] += 1
        return TerrainRuntimeSnapshot(
            revision=self._revision,
            tracked_chunk_count=len(coordinates),
            coordinates=coordinates,
            state_counts=counts,
            terrain_service=self._service.snapshot(),
        )

    def _transition(self, metadata: ChunkMetadata, state: ChunkState) -> ChunkMetadata:
        try:
            transitioned = metadata.transition_to(state)
        except ChunkTransitionError as error:
            raise TerrainRuntimeInvalidOperationError(str(error)) from error
        self._metadata[metadata.coordinate] = transitioned
        self._revision += 1
        return transitioned

    def _require_generation_state(self, metadata: ChunkMetadata) -> None:
        if metadata.state in (ChunkState.DECLARED, ChunkState.UNLOADED):
            return
        if metadata.state is ChunkState.GENERATING:
            raise TerrainGenerationInProgressError(
                f"Terrain generation is already in progress for chunk "
                f"({metadata.coordinate.x}, {metadata.coordinate.y})."
            )
        self._raise_invalid(operation="generate", metadata=metadata)

    @staticmethod
    def _raise_invalid(*, operation: str, metadata: ChunkMetadata) -> NoReturn:
        raise TerrainRuntimeInvalidOperationError(
            f"Cannot {operation} chunk ({metadata.coordinate.x}, {metadata.coordinate.y}) "
            f"while state is {metadata.state.value!r}."
        )

    def _publish_lifecycle(
        self,
        event_type: type[TerrainChunkActivated | TerrainChunkSuspended | TerrainChunkUnloaded],
        previous: ChunkMetadata,
        current: ChunkMetadata,
    ) -> None:
        self._publish(
            event_type(
                coordinate=current.coordinate,
                previous_state=previous.state,
                current_state=current.state,
                runtime_revision=self._revision,
            )
        )

    def _publish(self, event: object) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(event)

    def _log(
        self,
        *,
        level: int,
        event: str,
        message: str,
        metadata: ChunkMetadata,
        previous_state: ChunkState | None = None,
        terrain: ChunkTerrain | None = None,
    ) -> None:
        service = self._service.snapshot()
        context: dict[str, object] = {
            "event": event,
            "chunk_x": metadata.coordinate.x,
            "chunk_y": metadata.coordinate.y,
            "region_x": metadata.region_coordinate.x,
            "region_y": metadata.region_coordinate.y,
            "chunk_state": metadata.state.value,
            "terrain_runtime_revision": self._revision,
            "terrain_repository_revision": service.repository_revision,
            "terrain_seed": metadata.terrain_seed,
            "terrain_cache_hits": service.cache_hits,
            "terrain_cache_misses": service.cache_misses,
            "terrain_successful_generations": service.successful_generations,
            "terrain_failed_generations": service.failed_generations,
            "terrain_evictions": service.evictions,
        }
        if previous_state is not None:
            context["previous_chunk_state"] = previous_state.value
        if terrain is not None:
            context.update(
                {
                    "terrain_min_elevation": terrain.minimum_elevation.metres,
                    "terrain_max_elevation": terrain.maximum_elevation.metres,
                    "terrain_tile_count": len(terrain),
                }
            )
        self._logger.log(level, message, extra=context)

    @staticmethod
    def _validate_coordinate(coordinate: object) -> None:
        if not isinstance(coordinate, ChunkCoordinate):
            raise TypeError("coordinate must be a ChunkCoordinate.")
