"""Module and installed entry point for the playable voxel survival release."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from open_world_rpg import __version__
from open_world_rpg.core import (
    LoggingConfig,
    ProjectPaths,
    configure_runtime_logging,
    reset_runtime_logging,
    write_crash_report,
)
from open_world_rpg.ui.voxel.application import (
    VoxelPrototypeApplication,
    VoxelPrototypeConfig,
)
from open_world_rpg.world import TerrainGenerationConfig


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _non_negative_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def _default_data_directory() -> Path:
    if bool(getattr(sys, "frozen", False)):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open World RPG playable voxel survival release")
    parser.add_argument(
        "--version",
        action="version",
        version=f"Open World RPG {__version__}",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="render a bounded hidden-window acceptance run and exit",
    )
    parser.add_argument(
        "--smoke-frames",
        type=_positive_integer,
        default=3,
        help="number of hidden frames rendered by --smoke-test (default: 3)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_directory(),
        help="root for default saves, logs, and crash reports",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        help="JSON save path used by the menu, F7/F8, and optional startup loading",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        dest="load_on_start",
        help="load the configured save before entering the world",
    )
    parser.add_argument(
        "--autosave",
        action="store_true",
        help="save modified world state after a clean shutdown",
    )
    parser.add_argument(
        "--direct-play",
        action="store_true",
        help="skip the main menu and progression flow",
    )
    parser.add_argument("--width", type=_positive_integer, help="window width in pixels")
    parser.add_argument("--height", type=_positive_integer, help="window height in pixels")
    parser.add_argument(
        "--target-fps",
        type=_positive_integer,
        help="render-loop frame-rate target",
    )
    parser.add_argument(
        "--vsync",
        action="store_true",
        help="request display synchronisation instead of the default responsive frame cap",
    )
    parser.add_argument(
        "--render-distance",
        type=_non_negative_integer,
        help="visible chunk radius",
    )
    parser.add_argument(
        "--world-seed",
        type=_non_negative_integer,
        default=0,
        help="deterministic non-negative world seed",
    )
    return parser


def _config_from_arguments(
    arguments: argparse.Namespace,
    *,
    data_paths: ProjectPaths,
) -> VoxelPrototypeConfig:
    save_path = arguments.save_path
    if not arguments.smoke_test and save_path is None:
        save_path = data_paths.save_directory / "voxel.json"
    if (arguments.load_on_start or arguments.autosave) and save_path is None:
        raise ValueError("--load and --autosave require --save-path during smoke tests")

    if arguments.smoke_test:
        return VoxelPrototypeConfig(
            width_pixels=320 if arguments.width is None else arguments.width,
            height_pixels=180 if arguments.height is None else arguments.height,
            target_fps=120 if arguments.target_fps is None else arguments.target_fps,
            render_distance=(0 if arguments.render_distance is None else arguments.render_distance),
            world_seed=arguments.world_seed,
            hidden_window=True,
            vsync_enabled=arguments.vsync,
            terrain_config=TerrainGenerationConfig(octave_count=1),
            save_path=save_path,
            load_on_start=arguments.load_on_start,
            autosave=arguments.autosave,
        )

    return VoxelPrototypeConfig(
        width_pixels=1280 if arguments.width is None else arguments.width,
        height_pixels=720 if arguments.height is None else arguments.height,
        target_fps=60 if arguments.target_fps is None else arguments.target_fps,
        render_distance=1 if arguments.render_distance is None else arguments.render_distance,
        vsync_enabled=arguments.vsync,
        world_seed=arguments.world_seed,
        save_path=save_path,
        load_on_start=arguments.load_on_start,
        autosave=arguments.autosave,
        game_flow_enabled=not arguments.direct_play,
        progression_enabled=not arguments.direct_play,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the playable release or execute a bounded hidden-window acceptance smoke."""
    command = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    arguments = parser.parse_args(command)
    data_root = arguments.data_dir.expanduser().resolve(strict=False)
    data_paths = ProjectPaths.from_project_root(data_root)
    try:
        config = _config_from_arguments(arguments, data_paths=data_paths)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    logger: logging.Logger | None = None
    try:
        logger = configure_runtime_logging(
            config=LoggingConfig(console_enabled=False),
            log_directory=data_paths.log_directory,
        )
        logger.info(
            "Voxel release launch started.",
            extra={
                "event": "voxel.launch_started",
                "world_seed": config.world_seed,
            },
        )
        exit_code = VoxelPrototypeApplication(config=config).run(
            max_frames=arguments.smoke_frames if arguments.smoke_test else None
        )
        logger.info(
            "Voxel release launch completed.",
            extra={
                "event": "voxel.launch_completed",
                "world_seed": config.world_seed,
            },
        )
        return exit_code
    except Exception as error:
        if logger is not None:
            logger.exception(
                "Voxel release failed.",
                extra={
                    "event": "voxel.launch_failed",
                    "world_seed": config.world_seed,
                },
            )
        report_path: Path | None = None
        try:
            report_path = write_crash_report(
                directory=data_root / "crash-reports",
                error=error,
                application_version=__version__,
                command=command,
                context={
                    "world_seed": config.world_seed,
                    "smoke_test": arguments.smoke_test,
                    "save_path": None if config.save_path is None else str(config.save_path),
                },
            )
        except Exception:
            if logger is not None:
                logger.exception(
                    "Crash report could not be written.",
                    extra={"event": "voxel.crash_report_failed"},
                )
        message = "Open World RPG could not start."
        if report_path is not None:
            message += f" Crash report: {report_path}"
        print(message, file=sys.stderr)
        return 1
    finally:
        if logger is not None:
            reset_runtime_logging(logger)


if __name__ == "__main__":  # pragma: no cover - executed by python -m
    raise SystemExit(main())
