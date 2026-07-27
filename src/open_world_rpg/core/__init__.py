"""Core domain primitives and runtime configuration."""

from open_world_rpg.core.config import (
    DEFAULT_WORLD_SEED,
    MAX_WORLD_SEED,
    DisplayConfig,
    GameConfig,
    ProjectPaths,
    RuntimeEnvironment,
    SimulationConfig,
)
from open_world_rpg.core.crash_reports import (
    CRASH_REPORT_SCHEMA_VERSION,
    CrashContextValue,
    CrashReportError,
    CrashReportWriteError,
    write_crash_report,
)
from open_world_rpg.core.diagnostics import (
    LOGGER_NAME,
    JsonLogFormatter,
    LoggingConfig,
    LogLevel,
    configure_runtime_logging,
    reset_runtime_logging,
)

__all__ = [
    "CRASH_REPORT_SCHEMA_VERSION",
    "DEFAULT_WORLD_SEED",
    "LOGGER_NAME",
    "MAX_WORLD_SEED",
    "CrashContextValue",
    "CrashReportError",
    "CrashReportWriteError",
    "DisplayConfig",
    "GameConfig",
    "JsonLogFormatter",
    "LogLevel",
    "LoggingConfig",
    "ProjectPaths",
    "RuntimeEnvironment",
    "SimulationConfig",
    "configure_runtime_logging",
    "reset_runtime_logging",
    "write_crash_report",
]
