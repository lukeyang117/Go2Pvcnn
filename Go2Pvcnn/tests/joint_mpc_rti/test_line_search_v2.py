from __future__ import annotations

from dataclasses import replace

import torch

from extension.joint_mpc_rti.solver.line_search import FILTER_NAMES, parallel_line_search
from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.nominal import build_nominal
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
from extension.joint_mpc_rti.solver.sqp_rti import published_stance_filter_mask
from extension.joint_mpc_rti.solver.trajectory_qp import JOINT_LOWER, JOINT_UPPER
from extension.joint_mpc_rti.terrain.query import query_world
from extension.joint_mpc_rti.types import JointMpcRtiSolverState
from .helpers import make_state
from .test_trajectory_losses import _flat_field


def _nominal(batch: int) -> torch.Tensor:
    state = torch.zeros(batch, 31, 18)
    state[..., 6:] = torch.tensor((0.0, 0.8, -1.5) * 4)
    return state


def _limits(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lower = state.new_tensor(JOINT_LOWER)
    upper = state.new_tensor(JOINT_UPPER)
    velocity = state.new_full((12,), 30.0)
    return lower, upper, velocity


def test_line_search_builds_five_state_candidates_and_selects_lowest_loss() -> None:
    nominal = _nominal(2)
    direction = torch.zeros_like(nominal)
    direction[..., 0] = 1.0
    lower, upper, velocity = _limits(nominal)

    def objective(state: torch.Tensor) -> torch.Tensor:
        return (state[:, :, 0] - 0.5).square().mean(dim=1)

    result = parallel_line_search(
        nominal,
        direction,
        objective,
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
        tie_tolerance=1.0e-7,
    )

    assert result.candidates.shape == (2, 5, 31, 18)
    assert result.alphas.tolist() == [1.0, 0.5, 0.25, 0.125, 0.0]
    torch.testing.assert_close(result.selected_loss, result.candidate_loss.min(dim=1).values)
    torch.testing.assert_close(result.alpha, torch.full((2,), 0.5))
    torch.testing.assert_close(result.state[..., 0], torch.full((2, 31), 0.5))


def test_line_search_evaluates_all_five_candidates_in_one_objective_call() -> None:
    nominal = _nominal(3)
    lower, upper, velocity = _limits(nominal)
    calls: list[tuple[int, ...]] = []

    def objective(state: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(state.shape))
        return state.square().mean(dim=(1, 2))

    parallel_line_search(
        nominal,
        torch.zeros_like(nominal),
        objective,
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
    )
    assert calls == [(15, 31, 18)]


def test_alpha_zero_candidate_preserves_nominal_when_direction_is_nonfinite() -> None:
    nominal = _nominal(1)
    lower, upper, velocity = _limits(nominal)

    result = parallel_line_search(
        nominal,
        torch.full_like(nominal, float("nan")),
        objective=lambda state: state.square().mean(dim=(1, 2)),
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
    )

    assert result.selected_index.item() == 4
    assert result.used_nominal.item()
    torch.testing.assert_close(result.candidates[:, 4], nominal)
    torch.testing.assert_close(result.state, nominal)


def test_line_search_filters_nonfinite_joint_limits_velocity_and_published_stance() -> None:
    assert FILTER_NAMES == (
        "finite",
        "joint_position",
        "joint_velocity",
        "published_kinematics",
    )


def test_published_stance_filter_rejects_candidates_over_anchor_tolerance() -> None:
    nominal = _nominal(1)
    direction = torch.zeros_like(nominal)
    direction[:, 1, 0] = 0.001
    lower, upper, velocity = _limits(nominal)
    anchor = go2_fk(
        nominal[:, 1, :3], nominal[:, 1, 3:6], nominal[:, 1, 6:]
    ).foot_pos_w
    stance = torch.tensor(((True, False, False, False),))

    result = parallel_line_search(
        nominal,
        direction,
        objective=lambda state: -state[:, 1, 0],
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        published_stance_anchor_w=anchor,
        published_stance_mask=stance,
        published_stance_tolerance=0.0005,
        dt=0.02,
    )

    assert result.valid.tolist() == [[False, True, True, True, True]]
    assert result.filter_valid.shape == (1, 5, 4)
    assert result.filter_valid[0, :, :3].all()
    assert result.filter_valid[0, :, 3].tolist() == [False, True, True, True, True]
    torch.testing.assert_close(result.valid, result.filter_valid.all(dim=-1))
    assert result.alpha.item() == 0.5


@torch.no_grad()
def test_published_stance_ground_filter_rejects_gap_and_penetration() -> None:
    for direction_z in (-0.0048, 0.0048):
        nominal = _nominal(1)
        foot = go2_fk(
            nominal[:, 1, :3], nominal[:, 1, 3:6], nominal[:, 1, 6:]
        ).foot_pos_w
        nominal[..., 2] += 0.022 - foot[:, :1, 2]
        direction = torch.zeros_like(nominal)
        direction[:, 1, 2] = direction_z
        lower, upper, velocity = _limits(nominal)

        result = parallel_line_search(
            nominal,
            direction,
            objective=lambda state: -direction_z * state[:, 1, 2],
            joint_lower=lower,
            joint_upper=upper,
            joint_velocity_limit=velocity,
            published_stance_anchor_w=go2_fk(
                nominal[:, 1, :3], nominal[:, 1, 3:6], nominal[:, 1, 6:]
            ).foot_pos_w,
            published_stance_mask=torch.zeros(1, 4, dtype=torch.bool),
            published_stance_ground_mask=torch.tensor(((True, False, False, False),)),
            published_stance_tolerance=0.0005,
            published_terrain_field=_flat_field(1),
            published_foot_contact_offset=0.022,
            dt=0.02,
        )

        assert result.filter_valid[0, :, 3].tolist() == [False, False, False, False, True]


def test_warm_manifold_makes_alpha_zero_published_stance_exact_fk_feasible() -> None:
    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    phase = torch.tensor((12,), dtype=torch.long)
    previous_trajectory = measured.as_vector()[:, None].expand(-1, 31, -1).clone()
    previous_trajectory[:, 2:, 0] += 0.006
    anchor = go2_fk(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_pos_w
    previous = JointMpcRtiSolverState(
        trajectory=previous_trajectory,
        gait_phase=phase,
        initialized=torch.ones(1, dtype=torch.bool),
        stance_anchor_w=anchor,
    )
    nominal = build_nominal(
        measured,
        torch.zeros(1, 3),
        _flat_field(1),
        phase,
        previous=previous,
        cfg=cfg,
    )
    lower, upper, velocity = _limits(nominal.state)
    schedule = fixed_trot_schedule(phase, horizon_steps=cfg.runtime.horizon_steps)

    result = parallel_line_search(
        nominal.state,
        torch.zeros_like(nominal.state),
        objective=lambda state: state.new_zeros(state.shape[0]),
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        published_stance_anchor_w=anchor,
        published_stance_mask=published_stance_filter_mask(schedule),
        published_stance_ground_mask=schedule.stance[:, 1],
        published_stance_tolerance=cfg.solver.published_stance_tolerance,
        published_terrain_field=_flat_field(1),
        published_foot_contact_offset=cfg.gait.foot_contact_offset,
        dt=cfg.runtime.dt,
    )

    assert result.filter_valid[0, 4, 3]


def test_published_stance_ground_filter_queries_each_candidate_xy_raw_height() -> None:
    nominal = _nominal(1)
    base_field = _flat_field(1)
    x = torch.arange(51, dtype=base_field.height_w.dtype) - 25.0
    ramp = (0.001 * x).view(1, 51, 1).expand(1, 51, 51)
    field = replace(base_field, height_w=ramp)
    foot = go2_fk(
        nominal[:, 1, :3], nominal[:, 1, 3:6], nominal[:, 1, 6:]
    ).foot_pos_w
    nominal_height = query_world(field, foot).height_w[:, 0]
    nominal[..., 2] += nominal_height[:, None] + 0.022 - foot[:, :1, 2]
    direction = torch.zeros_like(nominal)
    direction[:, 1, 0] = 0.02
    lower, upper, velocity = _limits(nominal)

    result = parallel_line_search(
        nominal,
        direction,
        objective=lambda state: -state[:, 1, 0],
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        published_stance_anchor_w=go2_fk(
            nominal[:, 1, :3], nominal[:, 1, 3:6], nominal[:, 1, 6:]
        ).foot_pos_w,
        published_stance_mask=torch.zeros(1, 4, dtype=torch.bool),
        published_stance_ground_mask=torch.tensor(((True, False, False, False),)),
        published_stance_tolerance=0.0005,
        published_terrain_field=field,
        published_foot_contact_offset=0.022,
        dt=0.02,
    )

    assert not result.filter_valid[0, 0, 3]
    assert result.filter_valid[0, 4, 3]
    assert result.filter_valid[0, :, 3].tolist() == [False, False, True, True, True]
    assert result.alpha.item() == 0.25


def test_published_stance_filter_mask_excludes_touchdown_onset() -> None:
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.solver import sqp_rti

    build = getattr(sqp_rti, "published_stance_filter_mask", None)
    assert callable(build)
    mask = build(fixed_trot_schedule(torch.tensor((0, 11))))

    assert mask.shape == (2, 4)
    assert mask[0].sum().item() == 2
    assert not mask[1].any()


def test_published_kinematics_filter_rejects_swing_below_candidate_surface() -> None:
    nominal = _nominal(1)
    foot = go2_fk(
        nominal[:, 1, :3], nominal[:, 1, 3:6], nominal[:, 1, 6:]
    ).foot_pos_w
    nominal[..., 2] += 0.020 - foot[:, :1, 2]
    direction = torch.zeros_like(nominal)
    direction[:, 1, 2] = 0.003
    lower, upper, velocity = _limits(nominal)

    result = parallel_line_search(
        nominal,
        direction,
        objective=lambda state: -state[:, 1, 2],
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        published_stance_anchor_w=go2_fk(
            nominal[:, 1, :3], nominal[:, 1, 3:6], nominal[:, 1, 6:]
        ).foot_pos_w,
        published_stance_mask=torch.zeros(1, 4, dtype=torch.bool),
        published_stance_tolerance=0.0005,
        published_swing_mask=torch.tensor(((False, True, False, True),)),
        published_terrain_field=_flat_field(1),
        published_foot_contact_offset=0.022,
        published_swing_clearance_buffer=0.0,
        dt=0.02,
    )

    assert result.filter_valid[0, :, 3].tolist() == [True, False, False, False, False]
    assert result.alpha.item() == 1.0


def test_published_swing_filter_queries_each_candidate_xy() -> None:
    nominal = _nominal(1)
    base_field = _flat_field(1)
    x = torch.arange(51, dtype=base_field.height_w.dtype) - 25.0
    ramp = (0.005 * x).view(1, 51, 1).expand(1, 51, 51)
    field = replace(base_field, height_w=ramp)
    foot = go2_fk(
        nominal[:, 1, :3], nominal[:, 1, 3:6], nominal[:, 1, 6:]
    ).foot_pos_w
    nominal_surface = query_world(field, foot).height_w[:, 1]
    nominal[..., 2] += nominal_surface[:, None] + 0.023 - foot[:, 1:2, 2]
    direction = torch.zeros_like(nominal)
    direction[:, 1, 0] = 0.02
    direction[:, 1, 2] = 0.003
    lower, upper, velocity = _limits(nominal)

    result = parallel_line_search(
        nominal,
        direction,
        objective=lambda state: -state[:, 1, 0],
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        published_stance_anchor_w=go2_fk(
            nominal[:, 1, :3], nominal[:, 1, 3:6], nominal[:, 1, 6:]
        ).foot_pos_w,
        published_stance_mask=torch.zeros(1, 4, dtype=torch.bool),
        published_swing_mask=torch.tensor(((False, True, False, False),)),
        published_terrain_field=field,
        published_foot_contact_offset=0.022,
        published_swing_clearance_buffer=0.0,
        published_h_wall=0.35,
        dt=0.02,
    )

    assert not result.filter_valid[0, 0, 3]
    assert result.filter_valid[0, 4, 3]
    assert result.published_swing_safe_z.shape == (1, 5, 4)
    assert result.published_swing_safe_z[0, 0, 1] > result.published_swing_safe_z[0, 4, 1]


def test_nonfinite_loss_is_not_selectable() -> None:
    nominal = _nominal(1)
    direction = torch.zeros_like(nominal)
    direction[..., 0] = 1.0
    lower, upper, velocity = _limits(nominal)

    def objective(state: torch.Tensor) -> torch.Tensor:
        loss = -state[:, :, 0].mean(dim=1)
        return torch.where(state[:, 0, 0] == 1.0, torch.full_like(loss, torch.nan), loss)

    result = parallel_line_search(
        nominal,
        direction,
        objective,
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
    )
    assert result.alpha.item() == 0.5


def test_joint_position_filter_selects_largest_valid_alpha() -> None:
    nominal = _nominal(1)
    direction = torch.zeros_like(nominal)
    direction[..., 6] = 3.0
    lower, upper, velocity = _limits(nominal)
    result = parallel_line_search(
        nominal,
        direction,
        objective=lambda state: -state[:, :, 6].mean(dim=1),
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
    )
    assert result.alpha.item() == 0.25


def test_joint_velocity_filter_selects_largest_valid_alpha() -> None:
    nominal = _nominal(1)
    direction = torch.zeros_like(nominal)
    direction[:, 1:, 6] = 1.0
    lower, upper, velocity = _limits(nominal)
    result = parallel_line_search(
        nominal,
        direction,
        objective=lambda state: -state[:, :, 6].mean(dim=1),
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
    )
    assert result.alpha.item() == 0.5


def test_equal_loss_prefers_larger_alpha() -> None:
    nominal = _nominal(1)
    lower, upper, velocity = _limits(nominal)
    result = parallel_line_search(
        nominal,
        torch.zeros_like(nominal),
        objective=lambda state: torch.zeros(state.shape[0], device=state.device),
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
        tie_tolerance=1.0e-7,
    )
    assert result.alpha.eq(1.0).all()


def test_all_filtered_candidates_report_selected_candidate_infeasible() -> None:
    nominal = _nominal(1)
    nominal[..., 6] = float(JOINT_UPPER[0]) + 0.1
    lower, upper, velocity = _limits(nominal)

    result = parallel_line_search(
        nominal,
        torch.zeros_like(nominal),
        objective=lambda state: state.square().mean(dim=(1, 2)),
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
    )

    assert result.alpha.item() == 0.0
    assert not result.valid.any()
    assert not result.selected_feasible.item()
