# Open World RPG

A production-quality open-world role-playing game written in Python.

The project is being developed incrementally with clean architecture,
deterministic world generation, explicit application lifecycles, testable
gameplay systems, and persistent save support.

## Current milestone

**v0.5.0 - Procedural Terrain Foundation**

The procedural terrain foundation includes:

- Immutable terrain elevation, tile and chunk payload contracts
- Deterministic terrain configuration and classification
- Fixed-point BLAKE2b terrain sampling
- Deterministic complete chunk generation
- Terrain repository and generation service
- Controlled TerrainRuntime lifecycle and events
- Terrain diagnostics and end-to-end acceptance coverage
- Retained top-down terrain debug map
- First-person ModernGL voxel prototype
- Chunk streaming and GPU mesh caching
- Hidden-face and chunk-boundary culling
- Procedural pixel-art texture atlas
- Grass, dirt, stone, sand, snow and water strata
- Deterministic trees, shrubs, rocks and grass details
- First-person camera, collision, jumping and flying
- Sky, fog, sun, water animation and in-window HUD
- Adjustable render distance
- 1,542 passing tests
- 100% statement and branch coverage

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
