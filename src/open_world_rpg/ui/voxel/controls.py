"""Renderer-independent first-person control and help policies."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class MovementAxes:
    """Normalised local movement axes with diagonal speed capped at one."""

    forward: float
    sideways: float

    @property
    def active(self) -> bool:
        return self.forward != 0.0 or self.sideways != 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class ControlHint:
    """One compact control description rendered by the in-game help panel."""

    binding: str
    action: str


DEFAULT_CONTROL_HINTS: tuple[ControlHint, ...] = (
    ControlHint(binding="W A S D", action="Move"),
    ControlHint(binding="Mouse", action="Look"),
    ControlHint(binding="Shift", action="Sprint"),
    ControlHint(binding="Space", action="Jump / fly up"),
    ControlHint(binding="Ctrl", action="Fly down"),
    ControlHint(binding="Hold LMB", action="Mine target"),
    ControlHint(binding="RMB", action="Place selected block"),
    ControlHint(binding="1-9 / Wheel", action="Select hotbar slot"),
    ControlHint(binding="Esc", action="Release mouse / quit"),
    ControlHint(binding="F1 / H", action="Toggle controls"),
    ControlHint(binding="F3", action="Toggle diagnostics"),
)


def normalise_movement_axes(*, forward: float, sideways: float) -> MovementAxes:
    """Return finite local axes without a diagonal movement-speed advantage."""
    for name, value in (("forward", forward), ("sideways", sideways)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number.")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    magnitude = math.hypot(forward, sideways)
    if magnitude <= 1.0:
        return MovementAxes(forward=float(forward), sideways=float(sideways))
    return MovementAxes(
        forward=float(forward) / magnitude,
        sideways=float(sideways) / magnitude,
    )
