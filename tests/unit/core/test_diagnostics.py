"""Tests for structured runtime diagnostics."""

from __future__ import annotations

import json
import logging
import sys
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pytest

from open_world_rpg.core.diagnostics import (
    LOGGER_NAME,
    JsonLogFormatter,
    LoggingConfig,
    LogLevel,
    configure_runtime_logging,
    reset_runtime_logging,
)


@pytest.fixture(autouse=True)
def clean_runtime_logger() -> None:
    reset_runtime_logging()
    yield
    reset_runtime_logging()


def test_log_level_values_and_numeric_mappings() -> None:
    assert LogLevel.DEBUG.value == "debug"
    assert LogLevel.INFO.value == "info"
    assert LogLevel.WARNING.value == "warning"
    assert LogLevel.ERROR.value == "error"
    assert LogLevel.CRITICAL.value == "critical"

    assert LogLevel.DEBUG.numeric_value == logging.DEBUG
    assert LogLevel.INFO.numeric_value == logging.INFO
    assert LogLevel.WARNING.numeric_value == logging.WARNING
    assert LogLevel.ERROR.numeric_value == logging.ERROR
    assert LogLevel.CRITICAL.numeric_value == logging.CRITICAL


def test_logging_config_defaults() -> None:
    config = LoggingConfig()

    assert config.level is LogLevel.INFO
    assert config.console_enabled is True
    assert config.file_enabled is True
    assert config.file_name == "open-world-rpg.log"


def test_logging_config_normalises_file_name() -> None:
    config = LoggingConfig(file_name="  runtime.log  ")

    assert config.file_name == "runtime.log"


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("level", "info", TypeError),
        ("console_enabled", "yes", TypeError),
        ("file_enabled", 1, TypeError),
        ("file_name", 100, TypeError),
        ("file_name", "", ValueError),
        ("file_name", "   ", ValueError),
        ("file_name", "logs/runtime.log", ValueError),
        ("file_name", ".", ValueError),
        ("file_name", "..", ValueError),
    ],
)
def test_logging_config_rejects_invalid_values(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, Any] = {
        "level": LogLevel.INFO,
        "console_enabled": True,
        "file_enabled": True,
        "file_name": "runtime.log",
    }
    values[field_name] = value

    with pytest.raises(error_type, match=field_name):
        LoggingConfig(**values)


def test_logging_config_requires_a_handler() -> None:
    with pytest.raises(ValueError, match="At least one"):
        LoggingConfig(
            console_enabled=False,
            file_enabled=False,
        )


def test_json_formatter_emits_base_fields() -> None:
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name=LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Runtime ready",
        args=(),
        exc_info=None,
    )
    record.created = 0

    payload = json.loads(formatter.format(record))

    assert payload == {
        "level": "INFO",
        "logger": LOGGER_NAME,
        "message": "Runtime ready",
        "timestamp": "1970-01-01T00:00:00+00:00",
    }


def test_json_formatter_includes_runtime_context() -> None:
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name=LOGGER_NAME,
        level=logging.WARNING,
        pathname=__file__,
        lineno=20,
        msg="Session paused",
        args=(),
        exc_info=None,
    )
    record.event = "session.paused"
    record.session_id = "1234"
    record.world_seed = 42
    record.application_state = "running"
    record.session_state = "paused"

    payload = json.loads(formatter.format(record))

    assert payload["event"] == "session.paused"
    assert payload["session_id"] == "1234"
    assert payload["world_seed"] == 42
    assert payload["application_state"] == "running"
    assert payload["session_state"] == "paused"


def test_json_formatter_includes_exception_details() -> None:
    formatter = JsonLogFormatter()

    try:
        raise ValueError("invalid state")
    except ValueError:
        record = logging.LogRecord(
            name=LOGGER_NAME,
            level=logging.ERROR,
            pathname=__file__,
            lineno=30,
            msg="Runtime failed",
            args=(),
            exc_info=sys.exc_info(),
        )

    payload = json.loads(formatter.format(record))

    assert "ValueError: invalid state" in payload["exception"]


def test_configure_console_logging(tmp_path: Path) -> None:
    stream = StringIO()
    logger = configure_runtime_logging(
        config=LoggingConfig(file_enabled=False),
        log_directory=tmp_path,
        stream=stream,
    )

    logger.info(
        "Runtime ready",
        extra={"event": "application.ready"},
    )

    payload = json.loads(stream.getvalue())
    assert payload["message"] == "Runtime ready"
    assert payload["event"] == "application.ready"
    assert logger.name == LOGGER_NAME
    assert logger.propagate is False
    assert len(logger.handlers) == 1


def test_configure_console_logging_uses_standard_error(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    logger = configure_runtime_logging(
        config=LoggingConfig(file_enabled=False),
        log_directory=tmp_path,
    )

    logger.warning("Warning")

    captured = capsys.readouterr()
    assert json.loads(captured.err)["message"] == "Warning"


def test_configure_file_logging(tmp_path: Path) -> None:
    logger = configure_runtime_logging(
        config=LoggingConfig(
            console_enabled=False,
            file_name="runtime.log",
        ),
        log_directory=tmp_path,
    )

    logger.error("Failure", extra={"event": "application.failed"})

    for handler in logger.handlers:
        handler.flush()

    log_path = tmp_path / "runtime.log"
    payload = json.loads(log_path.read_text(encoding="utf-8"))

    assert payload["message"] == "Failure"
    assert payload["event"] == "application.failed"


def test_reconfiguration_replaces_existing_handlers(
    tmp_path: Path,
) -> None:
    first_stream = StringIO()
    second_stream = StringIO()

    configure_runtime_logging(
        config=LoggingConfig(file_enabled=False),
        log_directory=tmp_path,
        stream=first_stream,
    )
    logger = configure_runtime_logging(
        config=LoggingConfig(file_enabled=False),
        log_directory=tmp_path,
        stream=second_stream,
    )

    logger.info("Only second stream")

    assert first_stream.getvalue() == ""
    assert "Only second stream" in second_stream.getvalue()
    assert len(logger.handlers) == 1


def test_configuration_rejects_invalid_arguments(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="config"):
        configure_runtime_logging(
            config=cast(Any, object()),
            log_directory=tmp_path,
        )

    with pytest.raises(TypeError, match="log_directory"):
        configure_runtime_logging(
            config=LoggingConfig(),
            log_directory=cast(Any, str(tmp_path)),
        )


def test_reset_runtime_logging_accepts_explicit_logger(
    tmp_path: Path,
) -> None:
    logger = configure_runtime_logging(
        config=LoggingConfig(file_enabled=False),
        log_directory=tmp_path,
        stream=StringIO(),
    )

    reset_runtime_logging(logger)

    assert logger.handlers == []
