# Open World RPG

A production-quality open-world role-playing survival game written in Python.

The project is developed incrementally with clean architecture, deterministic
world generation, explicit application lifecycles, testable gameplay systems,
versioned persistence, and a playable first-person voxel interface.

## Current milestone

**v1.0 Release Candidate Stabilisation — package version 0.9.0**

The playable v0.9.0 survival loop is now under controlled v1.0 release-candidate
hardening. The package version remains `0.9.0` until the final release-preparation
commit, merge, and tag.

The current playable release includes:

- Main menu with New World, Continue, and Quit
- First-launch control and progression guide
- Empty new-world inventory with three starter logs near spawn
- Persistent objective tracking from wood gathering to the Stone Age
- Full 27-slot inventory with a nine-slot hotbar
- Keyboard and mouse inventory interaction
- Atomic stack movement, merging, swapping, and quick-move behaviour
- Logs, planks, sticks, and persistence-safe crafting resources
- Wooden and stone pickaxe and shovel recipes
- Recipe progression gates
- Hold-to-mine interaction with block hardness and tool effectiveness
- Stone mining restricted to an equipped pickaxe during progression play
- Block placement with reach and player-collision protection
- Valid and invalid interaction previews
- Wooden and stone tool durability and breakage
- Health, stamina, sprinting, jumping, fall damage, death, and safe respawn
- Pause, inventory/crafting, death, guide, and completion screens
- Save, Save & Quit, Continue, and Quit Without Saving flows
- Versioned persistence for inventory, tools, vitals, deaths, world edits,
  dropped resources, and survival progression
- Safe legacy-save progression inference
- Atomic rotating save backups and automatic corruption recovery
- JSON runtime logs and atomic crash reports
- Configurable release launcher and bounded graphical smoke tests
- Windows release validation and packaging scripts
- GitHub Actions quality matrix for Python 3.11 and 3.12
- 1,707 collected automated tests
- Zero-skipped-test and 100% statement/branch coverage release gates

## Requirements

- Python 3.11 or 3.12
- Git
- A Python virtual environment
- OpenGL 3.3-capable graphics support

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the optional release tooling when preparing distributable builds:

```powershell
python -m pip install -e ".[release]"
```

## Running the game

The primary package command now launches the playable survival release:

```powershell
python -m open_world_rpg
```

Or use the installed command:

```powershell
open-world-rpg
```

The dedicated voxel command remains available:

```powershell
python -m open_world_rpg.ui.voxel_demo
open-world-rpg-voxel-demo
```

The former rendering-free application lifecycle check is explicit:

```powershell
python -m open_world_rpg --runtime-check
```

Show the installed version:

```powershell
python -m open_world_rpg --version
open-world-rpg --version
```

### Release launcher options

```text
--data-dir PATH          Root for default saves, logs, and crash reports
--save-path PATH         Explicit JSON save path
--load                   Load the configured save on startup
--autosave               Save dirty world state after clean shutdown
--direct-play            Skip menu and progression flow
--width PIXELS           Window width
--height PIXELS          Window height
--target-fps FPS         Render-loop target
--render-distance RADIUS Visible chunk radius
--world-seed SEED        Deterministic non-negative world seed
--smoke-test             Hidden bounded renderer acceptance run
--smoke-frames COUNT     Hidden frames rendered during smoke test
```

Example isolated playtest:

```powershell
python -m open_world_rpg `
    --data-dir .\playtest-data `
    --world-seed 42 `
    --render-distance 2
```

### Runtime data

The default data directory is the current working directory for Python launches. A packaged Windows build uses the directory containing `OpenWorldRPG.exe`:

```text
saves/voxel.json
saves/voxel.backup.json
logs/open-world-rpg.log
crash-reports/open-world-rpg-crash-*.json
```

Launch packaged builds from a writable folder.

## Controls

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

## Survival objective

```text
Collect three logs
-> craft wood planks
-> craft sticks
-> craft a wooden pickaxe
-> mine three stone blocks
-> craft a stone pickaxe
-> reach the Stone Age completion screen
```

## Release validation

Run the complete controlled Windows release gate from the repository root:

```powershell
.\tools\validate_release.ps1 -ExpectedVersion 0.9.0
```

The gate enforces:

- Ruff lint and formatting
- strict mypy
- all automated tests
- zero skipped tests
- 100% statement and branch coverage
- both graphical import orders
- dependency integrity
- primary, voxel, and terrain smoke tests
- correct launcher versions
- no validation-induced working-tree changes

Manual playtesting is tracked in:

```text
docs/v1.0-manual-playtest.md
```

## Windows packaging

After release validation:

```powershell
.\tools\build_windows_release.ps1
```

The script produces a versioned ZIP under `dist/` containing the Windows executable,
licence, README, and launch notes.

## Project structure

```text
OpenWorldRPG_Rebuild/
|-- .github/workflows/
|-- docs/
|-- saves/
|-- src/open_world_rpg/
|   |-- application/
|   |-- core/
|   |-- engine/
|   |-- entities/
|   |-- gameplay/
|   |-- persistence/
|   |-- ui/
|   `-- world/
|-- tests/
|   |-- integration/
|   `-- unit/
|-- tools/
|-- LICENSE
|-- README.md
`-- pyproject.toml
```

## Development status

The v1.0 phase is release-candidate work only: manual playtesting, defect correction,
performance verification, recovery testing, packaging validation, and final release
polish. Large new gameplay systems remain outside the v1.0 scope.
