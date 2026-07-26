"""Pure first-person camera and player state mathematics."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

Vector3 = tuple[float, float, float]


def camera_vectors(*, yaw_degrees: float, pitch_degrees: float) -> tuple[Vector3, Vector3]:
    """Return normalized forward and horizontal-right vectors."""
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    forward = (
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        -math.cos(pitch) * math.cos(yaw),
    )
    right = (math.cos(yaw), 0.0, math.sin(yaw))
    return forward, right


@dataclass(frozen=True, slots=True, kw_only=True)
class FirstPersonCamera:
    """Yaw/pitch camera with bounded pitch."""

    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    sensitivity: float = 0.12

    def looked(self, *, delta_x: float, delta_y: float) -> FirstPersonCamera:
        pitch = max(-89.0, min(89.0, self.pitch_degrees - delta_y * self.sensitivity))
        return replace(
            self,
            yaw_degrees=(self.yaw_degrees + delta_x * self.sensitivity) % 360.0,
            pitch_degrees=pitch,
        )

    @property
    def forward(self) -> Vector3:
        return camera_vectors(yaw_degrees=self.yaw_degrees, pitch_degrees=self.pitch_degrees)[0]

    @property
    def right(self) -> Vector3:
        return camera_vectors(yaw_degrees=self.yaw_degrees, pitch_degrees=self.pitch_degrees)[1]


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerState:
    """Immutable player kinematics used by collision and controls."""

    x: float
    y: float
    z: float
    vertical_velocity: float = 0.0
    grounded: bool = False
    flying: bool = False
