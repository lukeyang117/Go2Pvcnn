"""Phase-signal gait helpers for the minimal Go2 planner core."""

from __future__ import annotations

import numpy as np

GAIT_PARAMS = {
    "trot": {
        "step_freq": 2.0,
        "duty_factor": 0.55,
        "offsets": np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float64),
    },
    "walk": {
        "step_freq": 1.0,
        "duty_factor": 0.75,
        "offsets": np.array([0.0, 0.5, 0.75, 0.25], dtype=np.float64),
    },
    "crawl": {
        "step_freq": 0.5,
        "duty_factor": 0.80,
        "offsets": np.array([0.0, 0.25, 0.75, 0.5], dtype=np.float64),
    },
    "pace": {
        "step_freq": 2.0,
        "duty_factor": 0.55,
        "offsets": np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float64),
    },
    "gallop": {
        "step_freq": 3.5,
        "duty_factor": 0.30,
        "offsets": np.array([0.0, 0.05, 0.4, 0.35], dtype=np.float64),
    },
}


def gait_schedule(
    t0: float,
    n_frames: int,
    dt: float,
    step_freq: float,
    duty_factor: float,
    phase_offsets: np.ndarray,
) -> np.ndarray:
    """Generate contact sequence. Returns [N, 4] float32 (1=stance, 0=swing)."""
    offsets = np.asarray(phase_offsets, dtype=np.float64).reshape(1, 4)
    t = t0 + np.arange(n_frames, dtype=np.float64)[:, None] * dt
    phase = np.mod(t * step_freq + offsets, 1.0)
    return (phase < duty_factor).astype(np.float32)


def foot_height_reference(
    t0: float,
    n_frames: int,
    dt: float,
    duty_ratio: float,
    cadence: float,
    amplitude: float,
    phases: np.ndarray,
) -> np.ndarray:
    """Continuous foot height reference with a bell-shaped swing profile."""
    out = np.zeros((n_frames, 4), dtype=np.float32)
    if duty_ratio >= 1.0:
        return out

    ph = np.asarray(phases, dtype=np.float64).reshape(1, 4)
    t = t0 + np.arange(n_frames, dtype=np.float64)[:, None] * dt
    inner = t * (2.0 * np.pi * cadence) + np.pi + 2.0 * np.pi * ph
    angle = np.mod(inner, 2.0 * np.pi) - np.pi
    scale = 0.5 / (1.0 - duty_ratio)
    angle = angle * scale
    clipped = np.clip(angle, -0.5 * np.pi, 0.5 * np.pi)
    value = np.abs(np.cos(clipped))
    return (amplitude * value).astype(np.float32)


def detect_swing_events(contact_seq: np.ndarray) -> dict:
    """Detect lift-off and touch-down frame indices per leg."""
    c = np.asarray(contact_seq, dtype=np.float64)
    d = np.diff(c, axis=0)
    lift_off: dict[int, list[int]] = {}
    touch_down: dict[int, list[int]] = {}
    for leg in range(4):
        lift_idx = np.where(d[:, leg] < -0.5)[0].astype(np.int64) + 1
        touch_idx = np.where(d[:, leg] > 0.5)[0].astype(np.int64) + 1
        lift_off[leg] = [int(x) for x in lift_idx.tolist()]
        touch_down[leg] = [int(x) for x in touch_idx.tolist()]
    return {"lift_off": lift_off, "touch_down": touch_down}


def stance_time(step_freq: float, duty_factor: float) -> float:
    """Stance duration in seconds."""
    return duty_factor / step_freq


def next_touchdown_times(step_freq: float, phase_offsets: np.ndarray) -> np.ndarray:
    """Return time-to-next-touchdown for each leg from cycle start."""
    offsets = np.mod(np.asarray(phase_offsets, dtype=np.float64).reshape(-1), 1.0)
    cycles = np.mod(1.0 - offsets, 1.0)
    cycles = np.where(cycles < 1e-9, 1.0, cycles)
    return cycles / float(step_freq)


__all__ = [
    "GAIT_PARAMS",
    "detect_swing_events",
    "foot_height_reference",
    "gait_schedule",
    "next_touchdown_times",
    "stance_time",
]
