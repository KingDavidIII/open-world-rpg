"""Module and installed entry point for the voxel terrain prototype."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from open_world_rpg.ui.voxel.application import (
    VoxelPrototypeApplication,
    VoxelPrototypeConfig,
)
from open_world_rpg.world import TerrainGenerationConfig


def main(argv: Sequence[str] | None = None) -> int:
    """Run interactively or execute a bounded hidden-window acceptance smoke."""
    parser = argparse.ArgumentParser(description="Open World RPG voxel terrain prototype")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="render three hidden frames and exit",
    )
    arguments = parser.parse_args(argv)
    config = (
        VoxelPrototypeConfig(
            width_pixels=320,
            height_pixels=180,
            target_fps=120,
            render_distance=0,
            hidden_window=True,
            terrain_config=TerrainGenerationConfig(octave_count=1),
        )
        if arguments.smoke_test
        else VoxelPrototypeConfig()
    )
    try:
        return VoxelPrototypeApplication(config=config).run(
            max_frames=3 if arguments.smoke_test else None
        )
    except Exception:
        logging.getLogger("open_world_rpg").exception("Voxel prototype failed.")
        return 1


if __name__ == "__main__":  # pragma: no cover - executed by python -m
    raise SystemExit(main())
