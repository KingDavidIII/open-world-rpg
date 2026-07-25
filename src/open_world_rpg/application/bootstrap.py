"""Application construction and process execution."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from open_world_rpg import __version__
from open_world_rpg.application.runtime import GameApplication
from open_world_rpg.core import GameConfig, RuntimeEnvironment


def create_application(
    *,
    project_root: Path | None = None,
    environment: RuntimeEnvironment = RuntimeEnvironment.DEVELOPMENT,
) -> GameApplication:
    """Construct a configured game application."""
    root = Path.cwd() if project_root is None else project_root

    return GameApplication(
        config=GameConfig.create_default(
            project_root=root,
            environment=environment,
        )
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
