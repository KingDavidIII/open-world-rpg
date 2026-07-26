# Open World RPG

A production-quality open-world role-playing game written in Python.

The project is being developed incrementally with clean architecture,
deterministic world generation, explicit application lifecycles, testable
gameplay systems, and persistent save support.

## Current milestone

**v0.8.0 - Tools and Survival Foundation**

The tools and survival foundation includes:

- Central item and tool catalogue policy
- Wooden and stone pickaxes
- Wooden and stone shovels
- Pickaxe, shovel, wood-tier and stone-tier classification
- Non-stackable tool instances
- Wood durability of 64 and stone durability of 128
- Mixed ItemStack and ToolInstance inventory slots
- 27-slot inventory with a nine-slot hotbar
- Deterministic survival starter inventory
- Atomic tool replacement and destruction
- Block hardness and tool-effectiveness policies
- Hold-to-mine interaction
- Mining cancellation on release, target loss, range loss and selection changes
- Exactly-once block edits, resource drops and durability consumption
- Tool durability loss and tool-breaking feedback
- Player health and stamina
- Sprint stamina drain
- Jump stamina cost
- Delayed stamina regeneration
- Fall damage with flying and water immunity
- Death counting and safe respawn
- Inventory, block-edit and dropped-resource preservation after respawn
- Health, stamina, mining progress and durability HUD indicators
- Tool effectiveness and damage feedback
- Tool durability, vitals and death-count persistence
- Legacy schema-v1 stack-only save compatibility
- Atomic failed-load behaviour
- Active-mining cancellation after loading
- Frame-local terrain-column caching for collision queries
- Controlled mining and sprinting performance gates
- 1,618 passing tests
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
