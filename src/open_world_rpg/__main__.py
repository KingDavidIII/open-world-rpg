"""Primary command-line entry point for Open World RPG."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from open_world_rpg.application import create_application, run_application


def _run_runtime_check() -> int:
    """Execute the rendering-free application lifecycle acceptance check."""
    return run_application(create_application())


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the playable game, or run the explicit headless lifecycle check."""
    command = list(sys.argv[1:] if argv is None else argv)
    if "--runtime-check" in command:
        parser = argparse.ArgumentParser(description="Open World RPG runtime check")
        parser.add_argument(
            "--runtime-check",
            action="store_true",
            help="initialise and stop the rendering-free application runtime",
        )
        parser.parse_args(command)
        return _run_runtime_check()

    from open_world_rpg.ui.voxel_demo import main as run_voxel_release

    return run_voxel_release(command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
