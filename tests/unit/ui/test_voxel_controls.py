"""Pure first-person control policy coverage."""

from __future__ import annotations

import math

import pytest

from open_world_rpg.ui.voxel import DEFAULT_CONTROL_HINTS, MovementAxes, normalise_movement_axes


def test_control_hints_are_compact_stable_and_cover_core_play() -> None:
    bindings = {hint.binding: hint.action for hint in DEFAULT_CONTROL_HINTS}
    assert bindings["W A S D"] == "Move"
    assert bindings["Hold LMB"] == "Mine target"
    assert bindings["RMB"] == "Place selected block"
    assert bindings["E / Tab"] == "Inventory & crafting"
    assert bindings["Esc"] == "Pause menu"


def test_movement_axes_preserve_cardinal_input_and_normalise_diagonals() -> None:
    idle = normalise_movement_axes(forward=0, sideways=0)
    forward = normalise_movement_axes(forward=1, sideways=0)
    diagonal = normalise_movement_axes(forward=1, sideways=1)

    assert idle == MovementAxes(forward=0.0, sideways=0.0)
    assert not idle.active
    assert forward == MovementAxes(forward=1.0, sideways=0.0)
    assert forward.active
    assert math.hypot(diagonal.forward, diagonal.sideways) == pytest.approx(1.0)
    assert diagonal.forward == diagonal.sideways


def test_movement_axes_reject_invalid_values() -> None:
    with pytest.raises(TypeError, match="forward"):
        normalise_movement_axes(forward=True, sideways=0)
    with pytest.raises(TypeError, match="sideways"):
        normalise_movement_axes(forward=0, sideways="right")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="forward"):
        normalise_movement_axes(forward=float("inf"), sideways=0)
    with pytest.raises(ValueError, match="sideways"):
        normalise_movement_axes(forward=0, sideways=float("nan"))
