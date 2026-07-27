"""Tests for the primary command-line entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from open_world_rpg.__main__ import main


def test_main_delegates_to_playable_release(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[list[str]] = []
    voxel_module = ModuleType("open_world_rpg.ui.voxel_demo")

    def run_voxel_release(arguments: list[str]) -> int:
        received.append(arguments)
        return 7

    voxel_module.main = cast(Any, run_voxel_release)
    monkeypatch.setitem(sys.modules, "open_world_rpg.ui.voxel_demo", voxel_module)

    assert main(["--smoke-test", "--smoke-frames", "5"]) == 7
    assert received == [["--smoke-test", "--smoke-frames", "5"]]


def test_main_runtime_check_preserves_headless_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--runtime-check"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "Open World RPG v0.9.0 - runtime initialised.\n"
    assert (tmp_path / "saves").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_runtime_check_rejects_unrelated_arguments() -> None:
    with pytest.raises(SystemExit):
        main(["--runtime-check", "--smoke-test"])
