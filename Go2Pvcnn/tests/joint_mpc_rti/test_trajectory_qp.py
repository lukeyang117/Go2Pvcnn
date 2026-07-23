from __future__ import annotations

import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.model.nominal import build_nominal
from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns
from extension.joint_mpc_rti.losses.objective import LossContext
from extension.joint_mpc_rti.solver.lq_problem import build_lq_problem
from extension.joint_mpc_rti.solver.trajectory_qp import solve_dense_qp
from extension.joint_mpc_rti.types import JointMpcRtiSolverState
from .helpers import make_command, make_state
from .test_perceptive_plan import _field, _warm
from .test_trajectory_losses import _flat_field


def _problem(*, batch: int = 1, dtype: torch.dtype = torch.float64):
    cfg = JointMpcRtiCfg()
    measured = make_state(batch, dtype=dtype)
    phase = torch.arange(batch, dtype=torch.long) % 24
    schedule = fixed_trot_schedule(phase)
    field = _field()
    if batch != 1:
        field = type(field)(
            **{
                name: (
                    value.expand(batch, *value.shape[1:]).clone()
                    if isinstance(value, torch.Tensor) and value.shape[0] == 1
                    else value
                )
                for name, value in vars(field).items()
            }
        )
    command = make_command(batch, vx=0.2).to(dtype)
    plan = select_touchdowns(
        measured,
        command,
        schedule,
        _warm(batch).to(dtype),
        field,
        cfg,
    )
    previous = JointMpcRtiSolverState(
        trajectory=_warm(batch).to(dtype),
        gait_phase=phase,
        initialized=torch.zeros(batch, dtype=torch.bool),
        stance_anchor_w=go2_fk(
            measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
        ).foot_pos_w,
    )
    nominal = build_nominal(
        measured,
        command,
        field,
        phase,
        perceptive_plan=plan,
        previous=previous,
        cfg=cfg,
    )
    context = LossContext(
        command_body=command,
        touchdown_reference_w=nominal.touchdown_reference_w,
        schedule=schedule,
        terrain=_flat_field(batch),
        stance_anchor_w=nominal.foot_reference_w,
        support_height=nominal.state.new_zeros(batch, 31),
        perceptive_field=field,
    )
    return build_lq_problem(nominal, context, cfg), nominal, context, cfg


def test_full_horizon_constraint_shapes_and_masks() -> None:
    problem, nominal, context, _ = _problem(batch=2)

    assert problem.z0_fixed.all()
    assert problem.lower.shape == (2, 31, 18)
    assert problem.upper.shape == (2, 31, 18)
    assert problem.rate_lower.shape == (2, 30, 15)
    assert problem.rate_upper.shape == (2, 30, 15)
    assert problem.stance_rows.shape == (2, 31, 4, 3, 18)
    assert problem.stance_active.shape == (2, 31, 4)
    assert torch.equal(problem.stance_active, context.schedule.stance)
    assert problem.touchdown_region_rows.shape == (2, 31, 4, 18, 4)
    assert problem.touchdown_region_rows.shape[-1] == 4
    assert problem.touchdown_plane_rows.shape == (2, 31, 4, 18)
    assert problem.clearance_rows.shape == (2, 31, 53, 18)
    assert problem.clearance_active.shape == (2, 31, 53)
    assert problem.clearance_active.all()
    torch.testing.assert_close(problem.lower[:, 0], torch.zeros_like(problem.lower[:, 0]))
    torch.testing.assert_close(problem.upper[:, 0], torch.zeros_like(problem.upper[:, 0]))
    assert torch.all(problem.lower <= problem.upper)
    assert nominal.perceptive_plan is not None


def test_every_stance_row_uses_complete_fk_jacobian() -> None:
    problem, nominal, context, _ = _problem()
    node, leg = torch.nonzero(context.schedule.stance[0], as_tuple=False)[5].tolist()
    direction = nominal.state.new_tensor(
        (0.01, -0.02, 0.03, 0.005, -0.004, 0.003) + (0.002,) * 12
    )
    epsilon = 1.0e-6
    before = go2_fk(
        nominal.state[:, node, :3],
        nominal.state[:, node, 3:6],
        nominal.state[:, node, 6:],
    ).foot_pos_w[:, leg]
    moved = nominal.state[:, node] + epsilon * direction
    after = go2_fk(moved[:, :3], moved[:, 3:6], moved[:, 6:]).foot_pos_w[:, leg]
    actual = (after - before) / epsilon
    predicted = torch.einsum(
        "bri,bi->br", problem.stance_rows[:, node, leg], direction[None]
    )

    torch.testing.assert_close(predicted, actual, atol=2.0e-5, rtol=2.0e-5)


def test_dense_reference_reports_scaled_kkt_and_typed_slack() -> None:
    problem, _, _, _ = _problem()

    solution = solve_dense_qp(problem)

    assert solution.direction.shape == (1, 31, 18)
    assert solution.kkt_primal_residual.max() <= 1.0e-4
    assert solution.kkt_dual_residual.max() <= 1.0e-4
    assert set(solution.slack_max) == {"collision", "region"}
    assert set(solution.active_constraint_count) == {
        "box",
        "rate",
        "stance",
        "touchdown_region",
        "touchdown_plane",
        "clearance",
    }
    assert torch.isfinite(solution.direction).all()


def test_joint_root_and_rate_bounds_are_relative_to_nominal() -> None:
    problem, nominal, _, cfg = _problem()
    maximum_joint_step = cfg.solver.joint_velocity_limit * cfg.runtime.dt
    nominal_joint_step = nominal.state[:, 1:, 6:] - nominal.state[:, :-1, 6:]

    torch.testing.assert_close(
        problem.rate_upper[..., 3:],
        torch.full_like(nominal_joint_step, maximum_joint_step) - nominal_joint_step,
    )
    torch.testing.assert_close(
        problem.rate_lower[..., 3:],
        -torch.full_like(nominal_joint_step, maximum_joint_step) - nominal_joint_step,
    )
    assert torch.all(problem.upper[:, 1:, :3] <= cfg.solver.root_position_trust + 1.0e-12)
    assert torch.all(problem.upper[:, 1:, 3:5] <= cfg.solver.root_roll_pitch_trust + 1.0e-12)
