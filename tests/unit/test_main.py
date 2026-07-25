"""Tests for the command-line entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from open_world_rpg.__main__ import main


def test_main_bootstraps_application(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "Open World RPG v0.3.0 - runtime initialised.\n"
    assert (tmp_path / "saves").is_dir()
    assert (tmp_path / "logs").is_dir()
