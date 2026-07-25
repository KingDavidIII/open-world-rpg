"""Tests for validated game configuration models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from open_world_rpg.core.config import (
    MAX_WORLD_SEED,
    DisplayConfig,
    GameConfig,
    ProjectPaths,
    RuntimeEnvironment,
    SimulationConfig,
)


def test_runtime_environment_values_and_debug_modes() -> None:
    assert RuntimeEnvironment.DEVELOPMENT.value == "development"
    assert RuntimeEnvironment.TEST.value == "test"
    assert RuntimeEnvironment.PRODUCTION.value == "production"

    assert RuntimeEnvironment.DEVELOPMENT.is_debug is True
    assert RuntimeEnvironment.TEST.is_debug is True
    assert RuntimeEnvironment.PRODUCTION.is_debug is False


def test_display_config_defaults() -> None:
    config = DisplayConfig()

    assert config.width == 1280
    assert config.height == 720
    assert config.fullscreen is False
    assert config.vsync is True
    assert config.target_fps == 60


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (639, ValueError),
        (7681, ValueError),
        ("1280", TypeError),
        (True, TypeError),
    ],
)
def test_display_config_rejects_invalid_width(
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="width"):
        DisplayConfig(width=cast(Any, value))


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (359, ValueError),
        (4321, ValueError),
        ("720", TypeError),
    ],
)
def test_display_config_rejects_invalid_height(
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="height"):
        DisplayConfig(height=cast(Any, value))


@pytest.mark.parametrize("field_name", ["fullscreen", "vsync"])
def test_display_config_rejects_non_boolean_flags(field_name: str) -> None:
    invalid = cast(Any, "yes")

    with pytest.raises(TypeError, match=field_name):
        if field_name == "fullscreen":
            DisplayConfig(fullscreen=invalid)
        else:
            DisplayConfig(vsync=invalid)


@pytest.mark.parametrize("value", [29, 361])
def test_display_config_rejects_invalid_target_fps(value: int) -> None:
    with pytest.raises(ValueError, match="target_fps"):
        DisplayConfig(target_fps=value)


def test_simulation_config_defaults() -> None:
    config = SimulationConfig()

    assert config.world_seed == 0
    assert config.tick_rate == 60
    assert config.max_frame_skip == 5


@pytest.mark.parametrize("value", [-1, MAX_WORLD_SEED + 1])
def test_simulation_config_rejects_out_of_range_seed(value: int) -> None:
    with pytest.raises(ValueError, match="world_seed"):
        SimulationConfig(world_seed=value)


def test_simulation_config_rejects_non_integer_seed() -> None:
    with pytest.raises(TypeError, match="world_seed"):
        SimulationConfig(world_seed=cast(Any, "seed"))


@pytest.mark.parametrize("value", [9, 241])
def test_simulation_config_rejects_invalid_tick_rate(value: int) -> None:
    with pytest.raises(ValueError, match="tick_rate"):
        SimulationConfig(tick_rate=value)


@pytest.mark.parametrize("value", [-1, 11])
def test_simulation_config_rejects_invalid_frame_skip(value: int) -> None:
    with pytest.raises(ValueError, match="max_frame_skip"):
        SimulationConfig(max_frame_skip=value)


def test_project_paths_are_normalised(tmp_path: Path) -> None:
    paths = ProjectPaths.from_project_root(tmp_path / ".." / tmp_path.name)

    assert paths.project_root == tmp_path.resolve()
    assert paths.save_directory == (tmp_path / "saves").resolve()
    assert paths.log_directory == (tmp_path / "logs").resolve()


def test_project_paths_reject_non_path_values(tmp_path: Path) -> None:
    invalid = cast(Any, str(tmp_path))

    with pytest.raises(TypeError, match="project_root"):
        ProjectPaths.from_project_root(invalid)

    with pytest.raises(TypeError, match="save_directory"):
        ProjectPaths(
            project_root=tmp_path,
            save_directory=invalid,
            log_directory=tmp_path / "logs",
        )

    with pytest.raises(TypeError, match="log_directory"):
        ProjectPaths(
            project_root=tmp_path,
            save_directory=tmp_path / "saves",
            log_directory=invalid,
        )


def test_game_config_create_default(tmp_path: Path) -> None:
    config = GameConfig.create_default(
        project_root=tmp_path,
        environment=RuntimeEnvironment.TEST,
    )

    assert config.title == "Open World RPG"
    assert config.environment is RuntimeEnvironment.TEST
    assert config.debug_enabled is True
    assert config.paths.project_root == tmp_path.resolve()
    assert config.paths.save_directory == (tmp_path / "saves").resolve()
    assert config.paths.log_directory == (tmp_path / "logs").resolve()


def test_game_config_normalises_title(tmp_path: Path) -> None:
    config = GameConfig(
        title="  Crown of the Open World  ",
        paths=ProjectPaths.from_project_root(tmp_path),
    )

    assert config.title == "Crown of the Open World"


@pytest.mark.parametrize("value", ["", "   "])
def test_game_config_rejects_empty_title(
    value: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="title"):
        GameConfig(
            title=value,
            paths=ProjectPaths.from_project_root(tmp_path),
        )


def test_game_config_rejects_non_string_title(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="title"):
        GameConfig(
            title=cast(Any, 100),
            paths=ProjectPaths.from_project_root(tmp_path),
        )


def test_game_config_rejects_invalid_nested_configuration(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.from_project_root(tmp_path)
    invalid = cast(Any, object())

    with pytest.raises(TypeError, match="environment"):
        GameConfig(environment=invalid, paths=paths)

    with pytest.raises(TypeError, match="display"):
        GameConfig(display=invalid, paths=paths)

    with pytest.raises(TypeError, match="simulation"):
        GameConfig(simulation=invalid, paths=paths)

    with pytest.raises(TypeError, match="paths"):
        GameConfig(paths=invalid)


def test_game_config_is_immutable(tmp_path: Path) -> None:
    config = GameConfig.create_default(project_root=tmp_path)

    with pytest.raises(FrozenInstanceError):
        config.title = "Changed"  # type: ignore[misc]


def test_production_configuration_disables_debug(tmp_path: Path) -> None:
    config = GameConfig.create_default(
        project_root=tmp_path,
        environment=RuntimeEnvironment.PRODUCTION,
    )

    assert config.debug_enabled is False


def test_game_config_uses_current_directory_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = GameConfig()

    assert config.paths.project_root == tmp_path.resolve()
    assert config.paths.save_directory == (tmp_path / "saves").resolve()
    assert config.paths.log_directory == (tmp_path / "logs").resolve()
