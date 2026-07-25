# Open World RPG

A production-quality open-world role-playing game written in Python.

The project is being developed incrementally with clean architecture,
deterministic world generation, explicit application lifecycles, testable
gameplay systems, and persistent save support.

## Current milestone

**v0.3.0 - Engine Foundation**

The engine foundation includes:

- Deterministic fixed-step simulation timing
- Monotonic clock validation and bounded frame catch-up
- Ordered subsystem lifecycle management
- Startup rollback and reverse-order subsystem shutdown
- Dependency-aware subsystem construction
- Deterministic engine runtime orchestration
- Structured engine lifecycle and frame diagnostics
- Application-to-engine bootstrap integration
- Deterministic queued event delivery
- Explicit before-update, after-update, before-render, and after-render phases
- Typed shared-service context and subsystem service binding
- 567 passing tests with 100% statement and branch coverage

## Requirements

- Python 3.11 or newer
- Git
- A Python virtual environment

## Installation

Create and activate a virtual environment:

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

## Running the project

Run the package module:

    python -m open_world_rpg

Or use the installed command:

    open-world-rpg

## Quality checks

    python -m ruff format .
    python -m ruff check .
    python -m ruff format --check .
    python -m mypy
    python -m pytest
    git diff --check

## Project structure

    OpenWorldRPG_Rebuild/
    ├── saves/
    ├── src/
    │   └── open_world_rpg/
    │       ├── application/
    │       ├── core/
    │       ├── engine/
    │       ├── entities/
    │       ├── gameplay/
    │       ├── persistence/
    │       ├── ui/
    │       └── world/
    ├── tests/
    │   ├── integration/
    │   └── unit/
    ├── LICENSE
    ├── README.md
    └── pyproject.toml

## Development status

The project is under active development. The current executable confirms that
the package, module entry point, installed CLI command, and development tooling
are functioning correctly.
