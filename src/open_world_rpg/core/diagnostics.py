"""Structured logging configuration for runtime diagnostics."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, TextIO

LOGGER_NAME: Final = "open_world_rpg"

_CONTEXT_FIELDS: Final = (
    "event",
    "session_id",
    "world_id",
    "world_name",
    "world_state",
    "previous_world_state",
    "world_revision",
    "world_tick",
    "previous_world_tick",
    "advanced_world_ticks",
    "world_year",
    "world_day_of_year",
    "world_hour",
    "world_minute",
    "world_second",
    "world_tick_within_second",
    "world_seed",
    "world_tick_rate",
    "application_state",
    "session_state",
    "save_slot",
    "schema_version",
    "saved_session_id",
    "saved_session_state",
    "engine_state",
    "engine_operation",
    "subsystem_count",
    "frame_index",
    "frame_elapsed_ns",
    "frame_simulated_elapsed_ns",
    "frame_update_count",
    "frame_dropped_update_count",
    "interpolation_alpha",
    "cumulative_frame_count",
    "cumulative_update_count",
    "cumulative_dropped_update_count",
    "stop_reason",
    "cleanup_failed",
    "event_phase",
    "event_phase_update_index",
    "event_dispatch_count",
    "event_handler_invocation_count",
    "event_dispatch_failure_count",
    "pending_event_count",
    "chunk_x",
    "chunk_y",
    "region_x",
    "region_y",
    "chunk_state",
    "previous_chunk_state",
    "terrain_runtime_revision",
    "terrain_repository_revision",
    "terrain_seed",
    "terrain_min_elevation",
    "terrain_max_elevation",
    "terrain_tile_count",
    "terrain_cache_hits",
    "terrain_cache_misses",
    "terrain_successful_generations",
    "terrain_failed_generations",
    "terrain_evictions",
)


class LogLevel(StrEnum):
    """Supported runtime logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def numeric_value(self) -> int:
        """Return the standard-library logging value."""
        return {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }[self]


@dataclass(frozen=True, slots=True, kw_only=True)
class LoggingConfig:
    """Immutable configuration for runtime log handlers."""

    level: LogLevel = LogLevel.INFO
    console_enabled: bool = True
    file_enabled: bool = True
    file_name: str = "open-world-rpg.log"

    def __post_init__(self) -> None:
        if not isinstance(self.level, LogLevel):
            raise TypeError("level must be a LogLevel.")

        if not isinstance(self.console_enabled, bool):
            raise TypeError("console_enabled must be a boolean.")

        if not isinstance(self.file_enabled, bool):
            raise TypeError("file_enabled must be a boolean.")

        if not self.console_enabled and not self.file_enabled:
            raise ValueError("At least one log handler must be enabled.")

        if not isinstance(self.file_name, str):
            raise TypeError("file_name must be a string.")

        normalised_name = self.file_name.strip()
        if not normalised_name:
            raise ValueError("file_name cannot be empty.")

        path = Path(normalised_name)
        if path.name != normalised_name or normalised_name in {".", ".."}:
            raise ValueError("file_name must contain only a file name.")

        object.__setattr__(self, "file_name", normalised_name)


class JsonLogFormatter(logging.Formatter):
    """Format diagnostic records as stable JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=UTC,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field_name in _CONTEXT_FIELDS:
            field_value = getattr(record, field_name, None)
            if field_value is not None:
                payload[field_name] = field_value

        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )


def configure_runtime_logging(
    *,
    config: LoggingConfig,
    log_directory: Path,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure and return the project runtime logger."""
    if not isinstance(config, LoggingConfig):
        raise TypeError("config must be a LoggingConfig.")

    if not isinstance(log_directory, Path):
        raise TypeError("log_directory must be a pathlib.Path instance.")

    logger = logging.getLogger(LOGGER_NAME)
    reset_runtime_logging(logger)

    logger.setLevel(config.level.numeric_value)
    logger.propagate = False

    formatter = JsonLogFormatter()

    if config.console_enabled:
        console_handler = logging.StreamHandler(sys.stderr if stream is None else stream)
        console_handler.setLevel(config.level.numeric_value)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.file_enabled:
        resolved_directory = log_directory.expanduser().resolve(strict=False)
        resolved_directory.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            resolved_directory / config.file_name,
            encoding="utf-8",
        )
        file_handler.setLevel(config.level.numeric_value)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def reset_runtime_logging(
    logger: logging.Logger | None = None,
) -> None:
    """Remove and close every handler owned by the runtime logger."""
    target = logging.getLogger(LOGGER_NAME) if logger is None else logger

    for handler in tuple(target.handlers):
        target.removeHandler(handler)
        handler.close()
