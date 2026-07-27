# Open World RPG

A production-quality open-world role-playing game written in Python.

The project is developed incrementally with clean architecture, deterministic
world generation, explicit application lifecycles, testable gameplay systems,
and persistent save support.

## Current milestone

**v0.9.0 - Playable Survival Loop**

This release turns the voxel foundation into a guided, self-contained survival
prototype that can be launched and played without developer instructions.

The playable loop includes:

- Main menu with New World, Continue, and Quit
- First-launch three-page control and progression guide
- Empty new-world inventory with three starter logs placed near spawn
- Persistent objective tracking from wood gathering to the Stone Age
- Full 27-slot inventory with a nine-slot hotbar
- Keyboard and mouse inventory interaction
- Atomic slot movement, merging, swapping, and quick-move behaviour
- Logs, planks, sticks, and persistence-safe crafting resources
- Wooden and stone pickaxe and shovel recipes
- Recipe progression gates that require wooden tools before stone tools
- Hold-to-mine interaction with block hardness and tool effectiveness
- Stone mining restricted to an equipped pickaxe during progression play
- Block placement with reach and player-collision protection
- Valid and invalid interaction previews
- Wooden and stone tool durability and breakage
- Health, stamina, sprinting, jumping, fall damage, death, and safe respawn
- Pause, inventory/crafting, death, guide, and completion screens
- Save, Save & Quit, Continue, and Quit Without Saving flows
- Additive schema-v1 persistence for inventory, tools, vitals, deaths, world edits,
  dropped resources, and survival progression
- Safe progression inference for legacy schema-v1 saves
- Cached inventory atlas rendering and bounded progression checks
- Module and installed CLI smoke-test support
- 1,670 passing automated tests
- Zero skipped tests
- 100% statement and branch coverage

## Requirements

- Python 3.11 or 3.12
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

Run the playable voxel prototype:

    python -m open_world_rpg.ui.voxel_demo

Or use the installed voxel command:

    open-world-rpg-voxel-demo

The normal voxel launch opens the playable menu and progression flow. Use
`--direct-play` only for the developer-oriented direct world entry.

### Voxel controls

- `W A S D`: move
- `Mouse`: look
- `Shift`: sprint
- `Space`: jump or fly up
- `Ctrl`: fly down
- `Hold left mouse`: mine
- `Right mouse`: place the selected block
- `1-9` or mouse wheel: select a hotbar slot
- `E` or `Tab`: open inventory and crafting
- `Arrow keys`: navigate menus, inventory slots, and recipes
- `Enter`: activate a menu option or move a selected slot
- `Q`: quick-move the selected inventory slot
- `[` / `]` or `Page Up` / `Page Down`: select a recipe
- `C`: craft the selected recipe
- `Escape`: pause, close inventory, skip the guide, or resume
- `F1` or `H`: toggle the in-game control guide
- `F3`: toggle diagnostics

### Survival objective

    Collect three logs
    -> craft wood planks
    -> craft sticks
    -> craft a wooden pickaxe
    -> mine three stone blocks
    -> craft a stone pickaxe
    -> reach the Stone Age completion screen

## Quality checks

    python -m ruff format .
    python -m ruff check .
    python -m ruff format --check .
    python -m mypy
    python -m pytest
    git diff --check

## Project structure

    OpenWorldRPG_Rebuild/
    |-- docs/
    |-- saves/
    |-- src/
    |   `-- open_world_rpg/
    |       |-- application/
    |       |-- core/
    |       |-- engine/
    |       |-- entities/
    |       |-- gameplay/
    |       |-- persistence/
    |       |-- ui/
    |       `-- world/
    |-- tests/
    |   |-- integration/
    |   `-- unit/
    |-- LICENSE
    |-- README.md
    `-- pyproject.toml

## Development status

v0.9.0 is the final pre-1.0 feature milestone. The next phase is controlled
stabilisation, playtesting, balancing, performance tuning, packaging, and the
v1.0.0 foundation release.
