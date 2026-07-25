"""Command-line entry point for Open World RPG."""

from __future__ import annotations

from open_world_rpg.application import create_application, run_application


def main() -> int:
    """Construct and execute the game application."""
    return run_application(create_application())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
