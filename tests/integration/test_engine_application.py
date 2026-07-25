"""Integration tests for application and engine coordination."""

from __future__ import annotations

import logging
from pathlib import Path

from open_world_rpg.application import (
    ApplicationState,
    GameApplication,
    GameMode,
    RuntimeContext,
    SessionState,
    create_engine_runtime,
    run_engine_smoke_test,
)
from open_world_rpg.core import (
    GameConfig,
    ProjectPaths,
    RuntimeEnvironment,
    SimulationConfig,
)
from open_world_rpg.engine import EngineRuntimeState


class SingleSampleClock:
    """Clock providing one deterministic engine frame."""

    def now_ns(self) -> int:
        return 0


def test_application_engine_smoke_workflow(
    tmp_path: Path,
) -> None:
    config = GameConfig(
        environment=RuntimeEnvironment.TEST,
        simulation=SimulationConfig(
            world_seed=77,
            tick_rate=60,
            max_frame_skip=5,
        ),
        paths=ProjectPaths.from_project_root(tmp_path),
    )
    application = GameApplication(
        config=config,
        context=RuntimeContext.create(
            game_mode=GameMode.NEW_GAME,
            world_seed=77,
        ),
        logger=logging.Logger("test.integration.engine"),
    )
    engine = create_engine_runtime(
        application=application,
        clock=SingleSampleClock(),
    )

    snapshot = run_engine_smoke_test(
        application=application,
        engine=engine,
    )

    assert snapshot.state is EngineRuntimeState.STOPPED
    assert snapshot.frame_count == 1
    assert snapshot.update_count == 0
    assert snapshot.stop_reason == "frame_limit"
    assert application.state is ApplicationState.STOPPED
    assert application.context.state is SessionState.TERMINATED
    assert config.paths.save_directory.is_dir()
    assert config.paths.log_directory.is_dir()
