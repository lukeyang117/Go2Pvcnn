from __future__ import annotations

import torch
import pytest


def test_default_h16_covers_stance_swing_stance_for_every_leg() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule

    cfg = JointMpcRtiCfg()
    assert cfg.gait.half_cycle_steps == 8
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        torch.device("cpu"),
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )[0]

    assert contact.shape == (17, 4)
    for leg in range(4):
        transitions = contact[1:, leg].to(torch.int8) - contact[:-1, leg].to(torch.int8)
        assert torch.count_nonzero(transitions == -1) == 1
        assert torch.count_nonzero(transitions == 1) == 1


def test_go2_fk_returns_planner_leg_order_and_link_samples() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk

    root_pos = torch.zeros(2, 3)
    root_rpy = torch.zeros(2, 3)
    joint = torch.tensor([[0.0, 0.8, -1.5] * 4] * 2)

    geometry = go2_fk(root_pos, root_rpy, joint)

    assert geometry.foot_pos_w.shape == (2, 4, 3)
    assert geometry.knee_pos_w.shape == (2, 4, 3)
    assert geometry.shank_samples_w.shape == (2, 4, 3, 3)
    assert geometry.thigh_samples_w.shape == (2, 4, 3, 3)
    assert geometry.body_samples_w.shape[0] == 2
    assert geometry.body_samples_w.shape[-1] == 3
    assert geometry.foot_pos_w[0, 0, 1] > geometry.foot_pos_w[0, 1, 1]
    assert geometry.foot_pos_w[0, 2, 1] > geometry.foot_pos_w[0, 3, 1]


def test_go2_foot_only_fk_matches_full_geometry() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk, go2_foot_pos

    root_pos = torch.tensor([[0.1, -0.2, 0.35], [-0.3, 0.4, 0.31]])
    root_rpy = torch.tensor([[0.05, -0.1, 0.3], [-0.03, 0.08, -0.4]])
    joint = torch.tensor([[0.05, 0.7, -1.4] * 4, [-0.04, 0.9, -1.7] * 4])

    torch.testing.assert_close(
        go2_foot_pos(root_pos, root_rpy, joint),
        go2_fk(root_pos, root_rpy, joint).foot_pos_w,
    )


def test_go2_analytic_foot_jacobian_matches_central_difference() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import foot_jacobian_joint, go2_fk

    joint = torch.tensor([[0.05, 0.7, -1.4] * 4], dtype=torch.float64)
    root_pos = torch.zeros(1, 3, dtype=torch.float64)
    root_rpy = torch.tensor([[0.03, -0.04, 0.2]], dtype=torch.float64)
    jacobian = foot_jacobian_joint(root_pos, root_rpy, joint)
    epsilon = 1.0e-6
    joint_plus = joint.clone()
    joint_minus = joint.clone()
    joint_plus[0, 0] += epsilon
    joint_minus[0, 0] -= epsilon
    finite_difference = (
        go2_fk(root_pos, root_rpy, joint_plus).foot_pos_w
        - go2_fk(root_pos, root_rpy, joint_minus).foot_pos_w
    ) / (2.0 * epsilon)

    assert jacobian.shape == (1, 4, 3, 12)
    torch.testing.assert_close(jacobian[0, :, :, 0], finite_difference[0], atol=2.0e-5, rtol=2.0e-4)


def test_go2_local_leg_jacobian_matches_nonzero_blocks_of_full_joint_jacobian() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import foot_jacobian_joint, foot_jacobian_leg

    root_pos = torch.zeros(2, 3)
    root_rpy = torch.tensor([[0.03, -0.04, 0.2], [-0.02, 0.05, -0.3]])
    joint = torch.tensor([[0.05, 0.7, -1.4] * 4, [-0.04, 0.9, -1.7] * 4])

    local = foot_jacobian_leg(root_pos, root_rpy, joint)
    full = foot_jacobian_joint(root_pos, root_rpy, joint)

    assert local.shape == (2, 4, 3, 3)
    for leg in range(4):
        torch.testing.assert_close(local[:, leg], full[:, leg, :, 3 * leg : 3 * (leg + 1)])


@pytest.mark.parametrize("part", ("calf", "thigh"))
def test_link_sample_jacobian_matches_central_difference(part: str) -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk, link_sample_jacobians

    joint = torch.tensor([[0.05, 0.7, -1.4] * 4], dtype=torch.float64)
    root_pos = torch.zeros(1, 3, dtype=torch.float64)
    root_rpy = torch.tensor([[0.03, -0.04, 0.2]], dtype=torch.float64)
    jacobians = link_sample_jacobians(root_pos, root_rpy, joint)
    analytic = getattr(jacobians, f"{part}_samples")
    epsilon = 1.0e-6

    for local_joint in range(3):
        plus = joint.clone()
        minus = joint.clone()
        plus[0, local_joint] += epsilon
        minus[0, local_joint] -= epsilon
        plus_geometry = go2_fk(root_pos, root_rpy, plus)
        minus_geometry = go2_fk(root_pos, root_rpy, minus)
        attribute = "shank_samples_w" if part == "calf" else "thigh_samples_w"
        finite = (getattr(plus_geometry, attribute) - getattr(minus_geometry, attribute)) / (2.0 * epsilon)
        torch.testing.assert_close(
            analytic[0, 0, :, :, local_joint],
            finite[0, 0],
            atol=2.0e-5,
            rtol=2.0e-4,
        )
