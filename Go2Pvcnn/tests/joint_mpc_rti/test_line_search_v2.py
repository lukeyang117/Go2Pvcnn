from __future__ import annotations

from dataclasses import replace

import torch

from extension.joint_mpc_rti import planner
from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.solver.line_search import (
    ALPHAS,
    HARD_FILTER_NAMES,
    hard_safe_line_search,
)
from .helpers import make_command, make_flat_field, make_state
from .test_trajectory_qp import _problem


def _objective(state: torch.Tensor) -> torch.Tensor:
    return state[..., 0].square().mean(dim=1)


def test_hard_line_search_builds_exactly_five_packed_candidates() -> None:
    problem, nominal, context, cfg = _problem(batch=2)
    direction = torch.zeros_like(nominal.state)

    result = hard_safe_line_search(
        nominal, direction, _objective, context, problem, cfg
    )

    assert result.candidates.shape == (2, 5, 31, 18)
    assert result.alphas.tolist() == list(ALPHAS)
    assert result.alpha_feasible.shape == (2, 5)
    assert result.alpha_reject_bits.shape == (2, 5, len(HARD_FILTER_NAMES))
    assert result.minimum_clearance_by_part.shape == (2, 5, 5)
    assert result.selected_index.shape == (2,)
    assert not result.stop.any()


def test_hard_line_search_applies_published_x1_joint_step_limit() -> None:
    problem, nominal, context, cfg = _problem()
    direction = torch.zeros_like(nominal.state)
    direction[:, 1, 6] = 0.50

    result = hard_safe_line_search(
        nominal, direction, _objective, context, problem, cfg
    )

    published_step = (
        result.candidates[:, :, 1, 6] - result.candidates[:, :, 0, 6]
    ).abs()
    assert (
        published_step[result.alpha_feasible]
        <= float(cfg.solver.published_joint_step_limit_rad) + 1.0e-6
    ).all()


def test_alpha_zero_preview_filter_matches_safe_nominal_event_window() -> None:
    cfg = JointMpcRtiCfg()
    cfg.nominal.command_scale = 0.65

    result = planner.step(
        make_state(1),
        make_command(1, vx=0.8),
        make_flat_field(1),
        None,
        cfg,
    )

    preview = HARD_FILTER_NAMES.index("preview_safety")
    assert result.diagnostics.nominal_safe.all()
    assert not result.diagnostics.alpha_reject_bits[:, 4, preview].any()
    assert result.full_trajectory.publish.all()


def test_alpha_zero_receives_same_filters_and_invalid_nominal_stops() -> None:
    problem, nominal, context, cfg = _problem()
    invalid_state = nominal.state.clone()
    invalid_state[..., 6] = 3.0
    invalid = replace(nominal, state=invalid_state)

    result = hard_safe_line_search(
        invalid,
        torch.zeros_like(invalid_state),
        _objective,
        context,
        problem,
        cfg,
    )

    assert not result.alpha_feasible.any()
    assert result.stop.all()
    assert not result.publish.any()
    joint_filter = HARD_FILTER_NAMES.index("joint")
    assert result.alpha_reject_bits[0, 4, joint_filter]


def test_current_field_refresh_mismatch_rejects_all_candidates() -> None:
    problem, nominal, context, cfg = _problem()
    current = context.perceptive_field
    assert current is not None
    stale = replace(current, refresh_id=current.refresh_id - 1)
    stale_context = replace(context, perceptive_field=stale)

    result = hard_safe_line_search(
        nominal,
        torch.zeros_like(nominal.state),
        _objective,
        stale_context,
        problem,
        cfg,
        expected_refresh_id=current.refresh_id,
    )

    assert not result.alpha_feasible.any()
    assert result.stop.all()
    freshness = HARD_FILTER_NAMES.index("fresh_field")
    assert result.alpha_reject_bits[..., freshness].all()


def test_exact_stance_region_plane_and_swept_filters_apply_to_alpha_zero() -> None:
    problem, nominal, context, cfg = _problem()
    bad_reference = nominal.foot_reference_w.clone()
    bad_reference[context.schedule.stance] += bad_reference.new_tensor((0.02, 0.0, 0.0))
    invalid = replace(nominal, foot_reference_w=bad_reference)

    result = hard_safe_line_search(
        invalid,
        torch.zeros_like(nominal.state),
        _objective,
        context,
        problem,
        cfg,
    )

    stance = HARD_FILTER_NAMES.index("stance_xyz")
    assert result.alpha_reject_bits[0, 4, stance]
    assert not result.alpha_feasible[0, 4]


def test_nonfinite_direction_keeps_exact_nominal_alpha_zero_candidate() -> None:
    problem, nominal, context, cfg = _problem()

    result = hard_safe_line_search(
        nominal,
        torch.full_like(nominal.state, float("nan")),
        _objective,
        context,
        problem,
        cfg,
    )

    torch.testing.assert_close(result.candidates[:, 4], nominal.state)
    assert result.alpha_reject_bits[0, :4, HARD_FILTER_NAMES.index("finite")].all()
