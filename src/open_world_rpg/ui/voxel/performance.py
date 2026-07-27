"""Frame-time sampling and release-gate diagnostics for the voxel runtime."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import fmean
from typing import Final

DEFAULT_SAMPLE_WINDOW: Final = 600
STALL_THRESHOLD_SECONDS: Final = 0.1
SEVERE_STALL_THRESHOLD_SECONDS: Final = 0.25


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameTimingSnapshot:
    """Stable frame-time projection used by diagnostics and playtest gates."""

    sample_count: int
    average_fps: float
    one_percent_low_fps: float
    p95_frame_ms: float
    worst_frame_ms: float
    stall_count: int
    severe_stall_count: int


class FrameTimeTracker:
    """Bounded rolling frame-time recorder with percentile-based FPS lows."""

    def __init__(self, *, maximum_samples: int = DEFAULT_SAMPLE_WINDOW) -> None:
        if isinstance(maximum_samples, bool) or not isinstance(maximum_samples, int):
            raise TypeError("maximum_samples must be an integer.")
        if maximum_samples < 10:
            raise ValueError("maximum_samples must be at least 10.")
        self._samples: deque[float] = deque(maxlen=maximum_samples)
        self._stall_count = 0
        self._severe_stall_count = 0

    def record(self, frame_seconds: float) -> None:
        if isinstance(frame_seconds, bool) or not isinstance(frame_seconds, (int, float)):
            raise TypeError("frame_seconds must be a number.")
        value = float(frame_seconds)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("frame_seconds must be finite and greater than zero.")
        self._samples.append(value)
        if value >= STALL_THRESHOLD_SECONDS:
            self._stall_count += 1
        if value >= SEVERE_STALL_THRESHOLD_SECONDS:
            self._severe_stall_count += 1

    @property
    def snapshot(self) -> FrameTimingSnapshot:
        if not self._samples:
            return FrameTimingSnapshot(
                sample_count=0,
                average_fps=0.0,
                one_percent_low_fps=0.0,
                p95_frame_ms=0.0,
                worst_frame_ms=0.0,
                stall_count=self._stall_count,
                severe_stall_count=self._severe_stall_count,
            )
        ordered = sorted(self._samples)
        one_percent_count = max(1, math.ceil(len(ordered) * 0.01))
        slowest = ordered[-one_percent_count:]
        p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
        return FrameTimingSnapshot(
            sample_count=len(ordered),
            average_fps=1.0 / fmean(ordered),
            one_percent_low_fps=1.0 / fmean(slowest),
            p95_frame_ms=ordered[p95_index] * 1_000.0,
            worst_frame_ms=ordered[-1] * 1_000.0,
            stall_count=self._stall_count,
            severe_stall_count=self._severe_stall_count,
        )
