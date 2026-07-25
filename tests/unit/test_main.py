"""Tests for the package entry point."""

from _pytest.capture import CaptureFixture

from open_world_rpg.__main__ import main


def test_main_returns_success(capsys: CaptureFixture[str]) -> None:
    """The application entry point should complete successfully."""
    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Open World RPG v0.1.0" in captured.out
