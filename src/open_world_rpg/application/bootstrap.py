"""Application construction and process execution."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

from open_world_rpg import __version__
from open_world_rpg.application.runtime import GameApplication
from open_world_rpg.application.session import GameMode, RuntimeContext
from open_world_rpg.core import (
    GameConfig,
    LoggingConfig,
    RuntimeEnvironment,
    configure_runtime_logging,
)


def create_application(
    *,
    project_root: Path | None = None,
    environment: RuntimeEnvironment = RuntimeEnvironment.DEVELOPMENT,
    game_mode: GameMode = GameMode.NEW_GAME,
    logger: logging.Logger | None = None,
) -> GameApplication:
    """Construct a configured game application and runtime session."""
    root = Path.cwd() if project_root is None else project_root
    config = GameConfig.create_default(
        project_root=root,
        environment=environment,
    )
    context = RuntimeContext.create(
        game_mode=game_mode,
        world_seed=config.simulation.world_seed,
    )
    runtime_logger = (
        configure_runtime_logging(
            config=LoggingConfig(console_enabled=False),
            log_directory=config.paths.log_directory,
        )
        if logger is None
        else logger
    )

    return GameApplication(
        config=config,
        context=context,
        logger=runtime_logger,
    )


def run_application(
    application: GameApplication,
    *,
    output: TextIO | None = None,
) -> int:
    """Start, announce, and cleanly stop a game application."""
    stream = sys.stdout if output is None else output

    application.start()

    try:
        print(
            f"{application.config.title} v{__version__} - runtime initialised.",
            file=stream,
        )
    except Exception:
        application.fail()
        raise

    application.stop()
    return 0
