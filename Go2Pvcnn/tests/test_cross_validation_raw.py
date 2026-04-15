"""L1 cross-validation: batched planner vs raw/kinematic_footsteps planner."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.go2fp.base_solver import integrate_base_planar
from scripts.go2fp.foothold import compute_footholds, compute_hip_positions
from scripts.go2fp.gait import (
    GAIT_PARAMS as RAW_GAIT_PARAMS,
    gait_schedule,
    next_touchdown_times,
    stance_time,
)
from scripts.go2fp.ik import (
    batch_forward_kinematics as raw_batch_fk,
    batch_inverse_kinematics as raw_batch_ik,
)
from scripts.go2fp.swing import compute_swing_targets
from scripts.go2fp.trajectory import (
    default_initial_state as raw_default_initial_state,
    generate_trajectory,
)
from scripts.go2fp.types import Command, RobotState

from extension.batched_planner.base_solver import batched_integrate_base_planar
from extension.batched_planner.foothold import batched_compute_footholds
from extension.batched_planner.gait import (
    batched_gait_schedule,
    batched_next_touchdown_times,
    batched_stance_time,
)
from extension.batched_planner.ik import (
    batch_forward_kinematics as batched_batch_fk,
    batch_inverse_kinematics as batched_batch_ik,
)
from extension.batched_planner.swing import batched_compute_swing_targets
from extension.batched_planner.trajectory import batched_generate_trajectory
from extension.batched_planner.types import BatchedRobotState

from tests.fixtures.terrain_adapter import (
    NumpyHeightmapTerrain,
    make_stairs_terrains,
    verify_terrain_height_at_consistency,
)


# ── helpers ──────────────────────────────────────────────────────────────────


class _FlatRawTerrain:
    """Flat z=0 terrain with scalar query interface for raw planner functions."""

    def height_at(self, x: float, y: float) -> float:
        return 0.0

    def roughness_at(self, x: float, y: float) -> float:
        return 0.0

    def max_height_along_segment(self, p0_xy, p1_xy) -> float:
        return 0.0


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

    def test_full_trajectory_stairs(self, aligned_configs):
        """E2E on a linear ramp (stairs surrogate) terrain.

        The raw side uses ``NumpyHeightmapTerrain`` wrapping the same heightmap
        so that ``height_at``, ``roughness_at`` and ``max_height_along_segment``
        match the batched ``PlannerTerrain`` via bilinear interpolation.

        Position and contact fields are checked tightly.  Orientation and joint
        angles use generous tolerance because the terrain estimator's roll-sign
        decision (``where(fb_z_mean < 0, ...)``) is discontinuous when height
        is y-invariant (ramp in x only).  A sub-ULP float32/float64 difference
        in foot-z can flip the roll sign, causing ~0.16 rad quaternion divergence
        while positions remain essentially identical.
        """
        from extension.batched_planner.terrain import PlannerTerrain

        raw_cfg, batched_cfg = aligned_configs

        heightmap_np, ray_hits, wx, wy = make_stairs_terrains()
        batched_terrain = PlannerTerrain.from_ray_hits(
            ray_hits, world_x_range=wx, world_y_range=wy,
        )
        raw_terrain = NumpyHeightmapTerrain(heightmap_np, wx, wy)

        raw_state = raw_default_initial_state(terrain=raw_terrain)
        batched_state = _raw_state_to_batched(raw_state)

        raw_cmd = Command(vx=0.5, vy=0.0, yaw_rate=0.0)
        batched_cmd = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)

        n_frames, dt = 25, 0.02

        raw_result = generate_trajectory(
            raw_terrain, raw_state, raw_cmd, n_frames, dt, config=raw_cfg,
        )
        batched_result = batched_generate_trajectory(
            batched_terrain, batched_state, batched_cmd, n_frames, dt, cfg=batched_cfg,
        )

        raw_t = raw_result.root_pos_w.shape[0]
        batched_t = batched_result.root_pos_w.shape[1]
        assert raw_t == batched_t, (
            f"Frame count mismatch: raw={raw_t}, batched={batched_t}"
        )

        tight_fields = [
            ("root_pos_w", 1e-4, 1e-4),
            ("foot_pos_w", 1e-4, 1e-4),
            ("contact_state", 0.0, 0.0),
            ("planned_touchdown_w", 1e-4, 1e-4),
        ]
        for field_name, atol, rtol in tight_fields:
            _compare_field(
                field_name,
                getattr(raw_result, field_name),
                getattr(batched_result, field_name),
                atol=atol,
                rtol=rtol,
            )

        relaxed_fields = [
            ("root_quat_w", 0.25, 0.0),
            ("joint_angles", 0.5, 0.0),
        ]
        for field_name, atol, rtol in relaxed_fields:
            _compare_field(
                field_name,
                getattr(raw_result, field_name),
                getattr(batched_result, field_name),
                atol=atol,
                rtol=rtol,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Foothold cross-validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestFootholdCrossValidation:
    """Foothold computation parity: Raibert heuristic + spiral search on flat terrain."""

    def test_footholds_flat_forward(self, aligned_configs, flat_terrain_pair):
        raw_cfg, batched_cfg = aligned_configs
        _, batched_terrain, _, _ = flat_terrain_pair

        raw_state = raw_default_initial_state(terrain=None)
        initial_yaw = 0.0

        offsets = RAW_GAIT_PARAMS["trot"]["offsets"]
        st = stance_time(raw_cfg.step_freq, raw_cfg.duty_factor)
        td_times = next_touchdown_times(raw_cfg.step_freq, offsets)
        hip_positions = compute_hip_positions(raw_state.root_pos, initial_yaw)

        ref_vel = np.array([0.3, 0.0], dtype=np.float64)
        foot_pos = np.asarray(raw_state.foot_pos, dtype=np.float64).reshape(4, 3)

        raw_footholds = compute_footholds(
            raw_state.root_pos,
            initial_yaw,
            ref_vel,
            ref_vel,
            hip_positions,
            st,
            raw_cfg.hip_height,
            _FlatRawTerrain(),
            foot_pos,
            td_times,
            yaw_rate=0.0,
            search_radius=raw_cfg.foothold_search_radius,
            search_step=raw_cfg.foothold_search_step,
            max_step_down=raw_cfg.max_foothold_step_down,
        )

        batched_footholds = batched_compute_footholds(
            base_pos=torch.as_tensor(raw_state.root_pos, dtype=torch.float64).unsqueeze(0),
            base_yaw=torch.tensor([initial_yaw], dtype=torch.float64),
            base_lin_vel_xy=torch.as_tensor(ref_vel, dtype=torch.float64).unsqueeze(0),
            ref_lin_vel_xy=torch.as_tensor(ref_vel, dtype=torch.float64).unsqueeze(0),
            hip_positions=torch.as_tensor(hip_positions, dtype=torch.float64).unsqueeze(0),
            stance_time=torch.tensor([st], dtype=torch.float64),
            com_height=torch.tensor([raw_cfg.hip_height], dtype=torch.float64),
            terrain=batched_terrain,
            previous_footholds=torch.as_tensor(foot_pos, dtype=torch.float64).unsqueeze(0),
            touchdown_times=torch.as_tensor(td_times, dtype=torch.float64).unsqueeze(0),
            yaw_rate=torch.tensor([0.0], dtype=torch.float64),
            search_radius=raw_cfg.foothold_search_radius,
            search_step=raw_cfg.foothold_search_step,
            max_step_down=raw_cfg.max_foothold_step_down,
        )

        _compare_field(
            "footholds", raw_footholds, batched_footholds,
            atol=1e-8, rtol=1e-6,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Base solver cross-validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaseSolverCrossValidation:
    """Planar base integration parity between raw and batched stacks."""

    @pytest.mark.parametrize(
        "vx,vy,yaw_rate,label",
        [
            (0.5, 0.0, 0.0, "forward"),
            (0.3, 0.1, 0.5, "turning"),
        ],
    )
    def test_integrate_planar(self, vx, vy, yaw_rate, label):
        initial_pos_xy = np.array([0.0, 0.0], dtype=np.float64)
        initial_yaw = 0.0
        n_frames, dt = 25, 0.02

        raw_pos_xy, raw_yaw = integrate_base_planar(
            initial_pos_xy, initial_yaw, vx, vy, yaw_rate, n_frames, dt,
        )

        batched_pos_xy, batched_yaw = batched_integrate_base_planar(
            torch.as_tensor(initial_pos_xy, dtype=torch.float64).unsqueeze(0),
            torch.tensor([initial_yaw], dtype=torch.float64),
            torch.tensor([vx], dtype=torch.float64),
            torch.tensor([vy], dtype=torch.float64),
            torch.tensor([yaw_rate], dtype=torch.float64),
            n_frames,
            dt,
        )

        torch.testing.assert_close(
            batched_pos_xy[0],
            torch.as_tensor(raw_pos_xy),
            atol=1e-8,
            rtol=1e-6,
            msg=f"pos_xy mismatch ({label})",
        )
        torch.testing.assert_close(
            batched_yaw[0],
            torch.as_tensor(raw_yaw),
            atol=1e-8,
            rtol=1e-6,
            msg=f"yaw mismatch ({label})",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. IK / FK cross-validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestIKCrossValidation:
    """Inverse and forward kinematics parity between raw numpy and batched torch."""

    def test_ik_default_pose(self):
        raw_state = raw_default_initial_state(terrain=None)
        root_pos = np.asarray(raw_state.root_pos, dtype=np.float64).reshape(1, 3)
        root_quat = np.asarray(raw_state.root_quat, dtype=np.float64).reshape(1, 4)
        foot_targets = np.asarray(raw_state.foot_pos, dtype=np.float64).reshape(1, 4, 3)

        raw_joints = raw_batch_ik(root_pos, root_quat, foot_targets)
        batched_joints = batched_batch_ik(
            torch.as_tensor(root_pos),
            torch.as_tensor(root_quat),
            torch.as_tensor(foot_targets),
        )

        torch.testing.assert_close(
            batched_joints,
            torch.as_tensor(raw_joints),
            atol=1e-8,
            rtol=1e-6,
        )

    def test_ik_offset_feet(self):
        """IK with feet raised slightly off the ground."""
        root_pos = np.array([[0.0, 0.0, 0.30]], dtype=np.float64)
        root_quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        foot_targets = np.array(
            [
                [
                    [0.19, 0.11, 0.02],
                    [0.19, -0.11, 0.02],
                    [-0.19, 0.11, 0.02],
                    [-0.19, -0.11, 0.02],
                ]
            ],
            dtype=np.float64,
        )

        raw_joints = raw_batch_ik(root_pos, root_quat, foot_targets)
        batched_joints = batched_batch_ik(
            torch.as_tensor(root_pos),
            torch.as_tensor(root_quat),
            torch.as_tensor(foot_targets),
        )

        torch.testing.assert_close(
            batched_joints,
            torch.as_tensor(raw_joints),
            atol=1e-8,
            rtol=1e-6,
        )

    def test_fk_default_pose(self):
        raw_state = raw_default_initial_state(terrain=None)
        root_pos = np.asarray(raw_state.root_pos, dtype=np.float64).reshape(1, 3)
        root_quat = np.asarray(raw_state.root_quat, dtype=np.float64).reshape(1, 4)
        joint_angles = np.asarray(raw_state.joint_angles, dtype=np.float64).reshape(1, 12)

        raw_body = raw_batch_fk(root_pos, root_quat, joint_angles)
        batched_body = batched_batch_fk(
            torch.as_tensor(root_pos),
            torch.as_tensor(root_quat),
            torch.as_tensor(joint_angles),
        )

        torch.testing.assert_close(
            batched_body,
            torch.as_tensor(raw_body),
            atol=1e-8,
            rtol=1e-6,
        )

    def test_ik_fk_round_trip(self):
        """IK → FK should recover the original foot targets."""
        raw_state = raw_default_initial_state(terrain=None)
        root_pos = torch.as_tensor(
            np.asarray(raw_state.root_pos, dtype=np.float64).reshape(1, 3),
        )
        root_quat = torch.as_tensor(
            np.asarray(raw_state.root_quat, dtype=np.float64).reshape(1, 4),
        )
        foot_targets = torch.as_tensor(
            np.asarray(raw_state.foot_pos, dtype=np.float64).reshape(1, 4, 3),
        )

        joints = batched_batch_ik(root_pos, root_quat, foot_targets)
        body_pos = batched_batch_fk(root_pos, root_quat, joints)
        recovered_feet = body_pos[:, 8:12, :]

        torch.testing.assert_close(
            recovered_feet,
            foot_targets,
            atol=1e-6,
            rtol=1e-6,
        )
