# Open World RPG

A production-quality open-world role-playing game written in Python.

The project is being developed incrementally with clean architecture,
deterministic world generation, explicit application lifecycles, testable
gameplay systems, and persistent save support.

## Current milestone

**v0.6.0 - Player Interaction Foundation**

The player interaction foundation includes:

- Stable editable BlockMaterial domain
- Absolute WorldBlockCoordinate model
- Immutable BlockEdit records
- Revisioned BlockEditStore overlay
- Generated-terrain immutability
- Unified editable voxel block resolver
- First-person block targeting
- Left-click block breaking
- Right-click block placement
- Placement face-normal resolution
- Player-body placement rejection
- Interaction reach and cooldown policies
- Nine-slot creative hotbar
- Number-key and mouse-wheel selection
- Immediate mesh invalidation after edits
- Cross-chunk boundary invalidation
- Collision against placed and removed blocks
- Gravity after support removal
- Persistent block edits in save documents
- Backward-compatible legacy save loading
- Atomic save and restore
- World identity and seed validation
- Dirty-state tracking
- F7 save and F8 reload controls
- Optional `--save-path`, `--load` and `--autosave` CLI options
- Selective mesh reconciliation after loading
- 1,580 passing tests
- Zero skipped tests
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
