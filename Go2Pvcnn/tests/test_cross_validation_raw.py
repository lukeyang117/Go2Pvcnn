"""L1 cross-validation: batched planner vs raw/kinematic_footsteps planner."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.go2fp.gait import (
    GAIT_PARAMS as RAW_GAIT_PARAMS,
    gait_schedule,
    next_touchdown_times,
    stance_time,
)
from scripts.go2fp.swing import compute_swing_targets
from scripts.go2fp.trajectory import (
    default_initial_state as raw_default_initial_state,
    generate_trajectory,
)
from scripts.go2fp.types import Command, RobotState

from extension.batched_planner.gait import (
    batched_gait_schedule,
    batched_next_touchdown_times,
    batched_stance_time,
)
from extension.batched_planner.swing import batched_compute_swing_targets
from extension.batched_planner.trajectory import batched_generate_trajectory
from extension.batched_planner.types import BatchedRobotState

from tests.fixtures.terrain_adapter import verify_terrain_height_at_consistency


# ── helpers ──────────────────────────────────────────────────────────────────


def _raw_state_to_batched(raw: RobotState) -> BatchedRobotState:
    """Build a single-env BatchedRobotState mirroring a raw RobotState."""
    return BatchedRobotState(
        root_pos=torch.as_tensor(
            np.asarray(raw.root_pos, dtype=np.float64),
        ).unsqueeze(0),
        root_quat=torch.as_tensor(
            np.asarray(raw.root_quat, dtype=np.float64),
        ).unsqueeze(0),
        joint_angles=torch.as_tensor(
            np.asarray(raw.joint_angles, dtype=np.float64),
        ).unsqueeze(0),
        foot_pos=torch.as_tensor(
            np.asarray(raw.foot_pos, dtype=np.float64).reshape(4, 3),
        ).unsqueeze(0),
    )


def _compare_field(
    name: str,
    raw_np: np.ndarray,
    batched_tensor: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Compare a raw numpy array against the batch-0 slice of a batched tensor."""
    raw_t = torch.as_tensor(np.asarray(raw_np, dtype=np.float64))
    batched_cpu = batched_tensor[0].to(dtype=torch.float64).cpu()
    torch.testing.assert_close(
        batched_cpu,
        raw_t,
        atol=atol,
        rtol=rtol,
        msg=f"Mismatch on field '{name}'",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Terrain bridge gate (prerequisite for terrain-dependent tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTerrainBridgeGate:
    """Prerequisite: flat-terrain sampling parity between heightmap and PlannerTerrain."""

    def test_flat_height_at_parity(self, flat_terrain_pair):
        heightmap_np, terrain, world_x_range, world_y_range = flat_terrain_pair
        verify_terrain_height_at_consistency(
            terrain,
            heightmap_np,
            world_x_range,
            world_y_range,
            atol_interior=1e-6,
            atol_boundary=1e-4,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Gait cross-validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestGaitCrossValidation:
    """Contact schedule, touchdown times, and stance time parity."""

    def test_contact_seq_trot(self, aligned_configs):
        raw_cfg, batched_cfg = aligned_configs
        offsets = RAW_GAIT_PARAMS["trot"]["offsets"]
        n_frames, dt = 25, 0.02

        raw_arr = gait_schedule(
            0.0, n_frames, dt, raw_cfg.step_freq, raw_cfg.duty_factor, offsets,
        )
        batched_tensor = batched_gait_schedule(
            0.0, n_frames, dt, batched_cfg.step_freq, batched_cfg.duty_factor, offsets,
        )

        expected = torch.tensor(raw_arr, dtype=torch.float32).unsqueeze(0)
        torch.testing.assert_close(batched_tensor, expected, atol=0.0, rtol=0.0)

    def test_touchdown_times_match(self, aligned_configs):
        raw_cfg, batched_cfg = aligned_configs
        offsets = RAW_GAIT_PARAMS["trot"]["offsets"]

        raw_td = next_touchdown_times(raw_cfg.step_freq, offsets)
        batched_td = batched_next_touchdown_times(batched_cfg.step_freq, offsets)

        expected = torch.tensor(raw_td, dtype=torch.float64).unsqueeze(0)
        torch.testing.assert_close(batched_td, expected, atol=1e-12, rtol=0.0)

    def test_stance_time_match(self, aligned_configs):
        raw_cfg, batched_cfg = aligned_configs

        raw_st = stance_time(raw_cfg.step_freq, raw_cfg.duty_factor)
        batched_st = batched_stance_time(batched_cfg.step_freq, batched_cfg.duty_factor)

        expected = torch.tensor([raw_st], dtype=torch.float64)
        torch.testing.assert_close(batched_st, expected, atol=1e-12, rtol=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Swing cross-validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwingCrossValidation:
    """Swing target computation parity on flat terrain."""

    def test_swing_targets_flat_forward(self, aligned_configs):
        raw_cfg, batched_cfg = aligned_configs
        offsets = RAW_GAIT_PARAMS["trot"]["offsets"]
        n_frames, dt = 25, 0.02

        contact_np = gait_schedule(
            0.0, n_frames, dt, raw_cfg.step_freq, raw_cfg.duty_factor, offsets,
        )
        contact_torch = torch.tensor(contact_np, dtype=torch.float32).unsqueeze(0)

        raw_state = raw_default_initial_state(terrain=None)
        lift_off = np.array(raw_state.foot_pos, dtype=np.float64).reshape(4, 3)
        touchdown = lift_off.copy()
        touchdown[:, 0] += 0.05

        terrain_max = np.zeros(4, dtype=np.float64)

        raw_targets = compute_swing_targets(
            contact_np, lift_off, touchdown, raw_cfg.step_height, terrain_max,
        )
        batched_targets = batched_compute_swing_targets(
            contact_torch,
            torch.as_tensor(lift_off).unsqueeze(0),
            torch.as_tensor(touchdown).unsqueeze(0),
            batched_cfg.step_height,
            terrain_max_heights=torch.as_tensor(terrain_max).unsqueeze(0),
        )

        expected = torch.as_tensor(raw_targets).unsqueeze(0)
        torch.testing.assert_close(batched_targets, expected, atol=1e-8, rtol=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. End-to-end trajectory cross-validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrajectoryEndToEnd:
    """Full pipeline parity between raw and batched planners."""

    @pytest.mark.parametrize(
        "cmd_values,label",
        [
            ([0.3, 0.0, 0.0], "forward"),
            ([0.0, 0.2, 0.0], "lateral"),
            ([0.0, 0.0, 0.5], "turn"),
            ([0.0, 0.0, 0.0], "standstill"),
        ],
    )
    def test_full_trajectory_flat(
        self, aligned_configs, flat_terrain_pair, cmd_values, label,
    ):
        raw_cfg, batched_cfg = aligned_configs
        _heightmap_np, batched_terrain, _wx, _wy = flat_terrain_pair

        raw_state = raw_default_initial_state(terrain=None)
        batched_state = _raw_state_to_batched(raw_state)

        raw_cmd = Command(vx=cmd_values[0], vy=cmd_values[1], yaw_rate=cmd_values[2])
        batched_cmd = torch.tensor([cmd_values], dtype=torch.float64)

        n_frames, dt = 25, 0.02

        raw_result = generate_trajectory(
            None, raw_state, raw_cmd, n_frames, dt, config=raw_cfg,
        )
        batched_result = batched_generate_trajectory(
            batched_terrain, batched_state, batched_cmd, n_frames, dt, cfg=batched_cfg,
        )

        raw_t = raw_result.root_pos_w.shape[0]
        batched_t = batched_result.root_pos_w.shape[1]
        assert raw_t == batched_t, (
            f"Frame count mismatch: raw={raw_t}, batched={batched_t}"
        )

        fields = [
            ("root_pos_w", 1e-8, 1e-6),
            ("root_quat_w", 1e-8, 1e-6),
            ("joint_angles", 1e-8, 1e-6),
            ("foot_pos_w", 1e-8, 1e-6),
            ("contact_state", 0.0, 0.0),
            ("planned_touchdown_w", 1e-8, 1e-6),
        ]
        for field_name, atol, rtol in fields:
            _compare_field(
                field_name,
                getattr(raw_result, field_name),
                getattr(batched_result, field_name),
                atol=atol,
                rtol=rtol,
            )
