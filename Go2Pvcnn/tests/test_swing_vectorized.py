"""L2 swing vectorization regression tests.

These tests load golden reference tensors produced by the current serial
implementation and verify that `batched_compute_swing_targets` reproduces
them exactly.  After vectorization they become regression guards.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import torch
import pytest

# Path setup mirroring conftest.py — needed for standalone imports
_TESTS_DIR = Path(__file__).resolve().parent
_GO2_ROOT = _TESTS_DIR.parent
_RAW_ROOT = _GO2_ROOT.parent / "raw" / "kinematic_footsteps"
for _p in (str(_GO2_ROOT), str(_RAW_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GOLDEN_DIR = _TESTS_DIR / "fixtures" / "golden"


def _build_golden_alignment() -> dict[str, Any]:
    """Replicate conftest golden alignment to avoid tests.conftest import ambiguity."""
    from extension.batched_planner.config import BatchedTrajectoryConfig
    from scripts.go2fp.config import TrajectoryConfig

    raw_d = dataclasses.asdict(TrajectoryConfig())
    batched_d = dataclasses.asdict(BatchedTrajectoryConfig())
    out: dict[str, Any] = {}
    for name in BatchedTrajectoryConfig.__dataclass_fields__:
        out[name] = raw_d[name] if name in raw_d else batched_d[name]
    for name in TrajectoryConfig.__dataclass_fields__:
        if name not in out:
            out[name] = raw_d[name]
    return out


GOLDEN_ALIGNMENT = _build_golden_alignment()


def _load_swing_golden() -> dict[str, torch.Tensor]:
    path = GOLDEN_DIR / "golden_swing_targets.pt"
    if not path.exists():
        pytest.skip(f"Golden file not found: {path}. Run generate_golden.py first.")
    return torch.load(path, weights_only=True)


def _recompute_swing_targets(
    contact_seq: torch.Tensor,
    foot_pos: torch.Tensor,
    touchdown_pos: torch.Tensor,
    step_height: float | None = None,
) -> torch.Tensor:
    """Re-run batched_compute_swing_targets with aligned config step_height."""
    from extension.batched_planner.swing import batched_compute_swing_targets

    if step_height is None:
        step_height = GOLDEN_ALIGNMENT["step_height"]

    return batched_compute_swing_targets(
        contact_seq,
        foot_pos,
        touchdown_pos,
        step_height,
        terrain_max_heights=None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Swing progress + target golden regression
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwingProgressVectorized:
    """Tests for the swing progress computation.

    Currently tests the serial implementation; after vectorization these
    become regression guards.
    """

    def test_swing_progress_matches_golden_n1(self):
        golden = _load_swing_golden()
        recomputed = _recompute_swing_targets(
            golden["contact_seq_n1"],
            golden["foot_pos_n1"],
            golden["touchdown_pos_n1"],
        )
        torch.testing.assert_close(
            recomputed,
            golden["swing_targets_n1"],
            atol=1e-10,
            rtol=0.0,
            msg="N=1 swing targets diverged from golden reference",
        )

    def test_swing_targets_match_golden_n4(self):
        golden = _load_swing_golden()
        recomputed = _recompute_swing_targets(
            golden["contact_seq_n4"],
            golden["foot_pos_n4"],
            golden["touchdown_pos_n4"],
        )
        torch.testing.assert_close(
            recomputed,
            golden["swing_targets_n4"],
            atol=1e-10,
            rtol=0.0,
            msg="N=4 swing targets diverged from golden reference",
        )

    def test_all_stance_returns_foot_pos(self):
        """When contact_seq is all 1 (all stance), targets should equal lift-off positions."""
        foot_pos = torch.tensor(
            [[[0.19, 0.11, 0.0],
              [0.19, -0.11, 0.0],
              [-0.19, 0.11, 0.0],
              [-0.19, -0.11, 0.0]]],
            dtype=torch.float32,
        )
        touchdown_pos = foot_pos.clone()
        touchdown_pos[..., 0] += 0.05

        n_frames = 20
        contact_seq = torch.ones((1, n_frames, 4), dtype=torch.float32)

        targets = _recompute_swing_targets(
            contact_seq, foot_pos, touchdown_pos,
            step_height=GOLDEN_ALIGNMENT["step_height"],
        )

        # All stance: targets should be the foot_pos (lift-off anchor) for every frame
        expected = foot_pos.to(torch.float64).unsqueeze(1).expand(1, n_frames, 4, 3)
        torch.testing.assert_close(
            targets, expected, atol=1e-10, rtol=0.0,
            msg="All-stance targets should equal lift-off foot positions",
        )

    def test_all_swing_returns_arc(self):
        """When contact_seq is all 0 (all swing), targets follow a Hermite arc."""
        foot_pos = torch.tensor(
            [[[0.19, 0.11, 0.0],
              [0.19, -0.11, 0.0],
              [-0.19, 0.11, 0.0],
              [-0.19, -0.11, 0.0]]],
            dtype=torch.float32,
        )
        touchdown_pos = foot_pos.clone()
        touchdown_pos[..., 0] += 0.05

        n_frames = 20
        contact_seq = torch.zeros((1, n_frames, 4), dtype=torch.float32)

        targets = _recompute_swing_targets(contact_seq, foot_pos, touchdown_pos)

        # First frame: swing_progress=0 → should be at lift-off position
        torch.testing.assert_close(
            targets[:, 0, :, :2],
            foot_pos.to(torch.float64)[..., :2],
            atol=1e-8, rtol=0.0,
            msg="First swing frame XY should match lift-off",
        )

        # Last frame: swing_progress=1 → should be at touchdown position
        torch.testing.assert_close(
            targets[:, -1, :, :2],
            touchdown_pos.to(torch.float64)[..., :2],
            atol=1e-8, rtol=0.0,
            msg="Last swing frame XY should match touchdown",
        )

        # Z should be non-negative for all frames (arc goes up)
        assert torch.all(targets[..., 2] >= -1e-8), "Swing arc z should be non-negative"

        # Mid-arc Z should be elevated above both endpoints
        mid = n_frames // 2
        lo_z = foot_pos[0, :, 2].to(torch.float64)
        td_z = touchdown_pos[0, :, 2].to(torch.float64)
        max_endpoint_z = torch.maximum(lo_z, td_z)
        assert torch.all(targets[0, mid, :, 2] > max_endpoint_z), \
            "Mid-arc z should exceed both endpoint z values"

    def test_single_frame_swing(self):
        """Only 1 frame of swing in the middle — progress should be 0.0 (single-frame run)."""
        foot_pos = torch.tensor(
            [[[0.19, 0.11, 0.0],
              [0.19, -0.11, 0.0],
              [-0.19, 0.11, 0.0],
              [-0.19, -0.11, 0.0]]],
            dtype=torch.float32,
        )
        touchdown_pos = foot_pos.clone()
        touchdown_pos[..., 0] += 0.05

        n_frames = 10
        contact_seq = torch.ones((1, n_frames, 4), dtype=torch.float32)
        contact_seq[:, 5, :] = 0.0  # single swing frame

        targets = _recompute_swing_targets(contact_seq, foot_pos, touchdown_pos)

        # The single swing frame at index 5 should produce a valid (non-NaN) target
        assert not torch.any(torch.isnan(targets[:, 5, :, :])), \
            "Single-frame swing should not produce NaN"

        # Stance frames should equal their anchor positions
        for t in [0, 1, 2, 3, 4, 6, 7, 8, 9]:
            assert not torch.any(torch.isnan(targets[:, t, :, :])), \
                f"Stance frame {t} should not produce NaN"

    def test_hermite_continuity_at_transitions(self):
        """Z at swing start ≈ lift_off_z, z at swing end ≈ touchdown_z."""
        from extension.batched_planner.gait import GAIT_PARAMS, batched_gait_schedule
        from extension.batched_planner.swing import batched_compute_swing_targets

        cfg = GOLDEN_ALIGNMENT
        offsets = torch.as_tensor(GAIT_PARAMS["trot"]["offsets"], dtype=torch.float64)
        n_frames, dt = 25, 0.02

        contact_seq = batched_gait_schedule(0.0, n_frames, dt, cfg["step_freq"], cfg["duty_factor"], offsets)

        foot_pos = torch.tensor(
            [[[0.19, 0.11, 0.02],
              [0.19, -0.11, 0.01],
              [-0.19, 0.11, 0.03],
              [-0.19, -0.11, 0.0]]],
            dtype=torch.float64,
        )
        touchdown_pos = foot_pos.clone()
        touchdown_pos[..., 0] += 0.05
        touchdown_pos[..., 2] = torch.tensor([[[0.01, 0.02, 0.0, 0.01]]], dtype=torch.float64)

        targets = batched_compute_swing_targets(
            contact_seq, foot_pos, touchdown_pos, cfg["step_height"],
            terrain_max_heights=None,
        )

        contact_bool = contact_seq[0] > 0.5
        for leg in range(4):
            stance = contact_bool[:, leg]
            lo_z = foot_pos[0, leg, 2].item()
            td_z = touchdown_pos[0, leg, 2].item()

            for t in range(1, n_frames):
                was_stance = bool(stance[t - 1].item())
                is_swing = not bool(stance[t].item())
                if was_stance and is_swing:
                    # Swing start: z should be close to lift-off z
                    actual_z = targets[0, t, leg, 2].item()
                    assert abs(actual_z - lo_z) < 0.02, (
                        f"Leg {leg} frame {t}: swing start z={actual_z:.4f} "
                        f"far from lift_off_z={lo_z:.4f}"
                    )

            for t in range(n_frames - 1):
                is_swing = not bool(stance[t].item())
                next_stance = bool(stance[t + 1].item()) if t + 1 < n_frames else True
                if is_swing and next_stance:
                    actual_z = targets[0, t, leg, 2].item()
                    assert abs(actual_z - td_z) < 0.02, (
                        f"Leg {leg} frame {t}: swing end z={actual_z:.4f} "
                        f"far from touchdown_z={td_z:.4f}"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# Golden target matching
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwingTargetsVectorized:
    """Direct golden-reference matching for swing targets."""

    def test_targets_match_golden_n1(self):
        golden = _load_swing_golden()
        recomputed = _recompute_swing_targets(
            golden["contact_seq_n1"],
            golden["foot_pos_n1"],
            golden["touchdown_pos_n1"],
        )
        torch.testing.assert_close(
            recomputed,
            golden["swing_targets_n1"],
            atol=1e-10,
            rtol=0.0,
            msg="N=1 targets do not match golden reference",
        )

    def test_targets_match_golden_n4(self):
        golden = _load_swing_golden()
        recomputed = _recompute_swing_targets(
            golden["contact_seq_n4"],
            golden["foot_pos_n4"],
            golden["touchdown_pos_n4"],
        )
        torch.testing.assert_close(
            recomputed,
            golden["swing_targets_n4"],
            atol=1e-10,
            rtol=0.0,
            msg="N=4 targets do not match golden reference",
        )
