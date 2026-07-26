# Open World RPG

A production-quality open-world role-playing game written in Python.

The project is being developed incrementally with clean architecture,
deterministic world generation, explicit application lifecycles, testable
gameplay systems, and persistent save support.

## Current milestone

**v0.7.0 - Inventory and Resource Gameplay**

The inventory and resource gameplay foundation includes:

- Stable ItemType domain
- Mappings between block materials and collectible items
- Immutable ItemStack values with a maximum stack size of 64
- 27-slot PlayerInventory with the first nine slots serving as the hotbar
- Deterministic first-fit stacking and partial-stack filling
- Atomic inventory removal and restoration
- Explicit inventory revision semantics
- Deterministic starter inventory
- Inventory-backed block placement with exact placement consumption
- Dropped-item spawning after block breaking
- One-block-to-one-item drop policy
- Deterministic dropped-item identifiers
- Dropped-item gravity and ground collision
- Pickup delays, radius and partial pickup
- Full-inventory pickup rejection
- 300-second despawn policy
- 1,024-item active-drop cap
- Batched procedural item rendering
- Shared GPU buffer and VAO reuse
- Settled-item physics optimisation
- Inventory and dropped-item persistence
- Additive schema-v1 save compatibility
- Deterministic bootstrap inventory for legacy saves
- Atomic inventory and dropped-item restoration
- Dirty-state integration
- Persisted selected hotbar slot
- End-to-end restart acceptance
- 1,604 passing tests
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
