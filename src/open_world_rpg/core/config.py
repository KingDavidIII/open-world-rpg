"""Validated configuration models for the game runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

MIN_WINDOW_WIDTH: Final = 640
MAX_WINDOW_WIDTH: Final = 7680
MIN_WINDOW_HEIGHT: Final = 360
MAX_WINDOW_HEIGHT: Final = 4320

MIN_TARGET_FPS: Final = 30
MAX_TARGET_FPS: Final = 360

MIN_TICK_RATE: Final = 10
MAX_TICK_RATE: Final = 240
MAX_FRAME_SKIP: Final = 10

MIN_WORLD_SEED: Final = 0
MAX_WORLD_SEED: Final = (1 << 63) - 1
DEFAULT_WORLD_SEED: Final = 0


class RuntimeEnvironment(StrEnum):
    """Supported application runtime environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"

    @property
    def is_debug(self) -> bool:
        """Return whether development diagnostics should be enabled."""
        return self is not RuntimeEnvironment.PRODUCTION


def _require_int(
    *,
    name: str,
    value: object,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")

    return value


def _require_bool(*, name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")

    return value


def _require_non_empty_text(*, name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    normalised = value.strip()
    if not normalised:
        raise ValueError(f"{name} cannot be empty.")

    return normalised


def _normalise_path(*, name: str, value: object) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path instance.")

    return value.expanduser().resolve(strict=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class DisplayConfig:
    """Window and rendering configuration."""

    width: int = 1280
    height: int = 720
    fullscreen: bool = False
    vsync: bool = True
    target_fps: int = 60

    def __post_init__(self) -> None:
        _require_int(
            name="width",
            value=self.width,
            minimum=MIN_WINDOW_WIDTH,
            maximum=MAX_WINDOW_WIDTH,
        )
        _require_int(
            name="height",
            value=self.height,
            minimum=MIN_WINDOW_HEIGHT,
            maximum=MAX_WINDOW_HEIGHT,
        )
        _require_bool(name="fullscreen", value=self.fullscreen)
        _require_bool(name="vsync", value=self.vsync)
        _require_int(
            name="target_fps",
            value=self.target_fps,
            minimum=MIN_TARGET_FPS,
            maximum=MAX_TARGET_FPS,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationConfig:
    """Deterministic simulation and update-loop configuration."""

    world_seed: int = DEFAULT_WORLD_SEED
    tick_rate: int = 60
    max_frame_skip: int = 5

    def __post_init__(self) -> None:
        _require_int(
            name="world_seed",
            value=self.world_seed,
            minimum=MIN_WORLD_SEED,
            maximum=MAX_WORLD_SEED,
        )
        _require_int(
            name="tick_rate",
            value=self.tick_rate,
            minimum=MIN_TICK_RATE,
            maximum=MAX_TICK_RATE,
        )
        _require_int(
            name="max_frame_skip",
            value=self.max_frame_skip,
            minimum=0,
            maximum=MAX_FRAME_SKIP,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectPaths:
    """Canonical filesystem locations used by the application."""

    project_root: Path
    save_directory: Path
    log_directory: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "project_root",
            _normalise_path(name="project_root", value=self.project_root),
        )
        object.__setattr__(
            self,
            "save_directory",
            _normalise_path(name="save_directory", value=self.save_directory),
        )
        object.__setattr__(
            self,
            "log_directory",
            _normalise_path(name="log_directory", value=self.log_directory),
        )

    @classmethod
    def from_project_root(cls, project_root: Path) -> ProjectPaths:
        """Construct conventional project paths from a root directory."""
        root = _normalise_path(name="project_root", value=project_root)

        return cls(
            project_root=root,
            save_directory=root / "saves",
            log_directory=root / "logs",
        )


def _default_project_paths() -> ProjectPaths:
    return ProjectPaths.from_project_root(Path.cwd())


@dataclass(frozen=True, slots=True, kw_only=True)
class GameConfig:
    """Top-level immutable configuration for a game process."""

    title: str = "Open World RPG"
    environment: RuntimeEnvironment = RuntimeEnvironment.DEVELOPMENT
    display: DisplayConfig = field(default_factory=DisplayConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    paths: ProjectPaths = field(default_factory=_default_project_paths)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "title",
            _require_non_empty_text(name="title", value=self.title),
        )

        if not isinstance(self.environment, RuntimeEnvironment):
            raise TypeError("environment must be a RuntimeEnvironment.")

        if not isinstance(self.display, DisplayConfig):
            raise TypeError("display must be a DisplayConfig.")

        if not isinstance(self.simulation, SimulationConfig):
            raise TypeError("simulation must be a SimulationConfig.")

        if not isinstance(self.paths, ProjectPaths):
            raise TypeError("paths must be a ProjectPaths.")

    @property
    def debug_enabled(self) -> bool:
        """Return whether debug-oriented behaviour is enabled."""
        return self.environment.is_debug

    @classmethod
    def create_default(
        cls,
        *,
        project_root: Path,
        environment: RuntimeEnvironment = RuntimeEnvironment.DEVELOPMENT,
    ) -> GameConfig:
        """Create a default configuration rooted at the supplied directory."""
        return cls(
            environment=environment,
            paths=ProjectPaths.from_project_root(project_root),
        )
