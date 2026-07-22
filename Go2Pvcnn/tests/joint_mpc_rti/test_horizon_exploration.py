from __future__ import annotations

import pytest


def test_production_horizon_is_fixed_h30_full_cycle() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()

    assert cfg.runtime.horizon_steps == 30
    assert cfg.gait.half_cycle_steps == 15
    assert cfg.runtime.horizon_steps == 2 * cfg.gait.half_cycle_steps
    assert cfg.gait.max_swing_extension_steps == 10
    assert cfg.solver.line_search_alphas == (1.0, 0.5, 0.25, 0.1)
    assert cfg.solver.root_lateral_offset_limit_m == pytest.approx(0.06)
    assert cfg.solver.root_lateral_velocity_error_limit_mps == pytest.approx(0.20)
    assert cfg.solver.root_roll_pitch_limit_rad == pytest.approx(0.1047197551)
    assert cfg.solver.root_roll_pitch_rate_limit_rps == pytest.approx(0.6)
    assert cfg.solver.root_yaw_error_limit_rad == pytest.approx(0.1745329252)
    assert cfg.solver.root_yaw_rate_error_limit_rps == pytest.approx(0.8)


def test_horizon_candidates_cover_one_full_fixed_trot_cycle() -> None:
    from .horizon_exploration import make_horizon_candidates

    candidates = make_horizon_candidates((16, 20, 24, 30, 40, 50))

    assert [(candidate.horizon_steps, candidate.half_cycle_steps) for candidate in candidates] == [
        (16, 8),
        (20, 10),
        (24, 12),
        (30, 15),
        (40, 20),
        (50, 25),
    ]
    assert all(candidate.horizon_steps == 2 * candidate.half_cycle_steps for candidate in candidates)


@pytest.mark.parametrize("horizon", (15, 17, 52))
def test_horizon_candidates_reject_invalid_full_cycles(horizon: int) -> None:
    from .horizon_exploration import make_horizon_candidates

    with pytest.raises(ValueError, match="even and within"):
        make_horizon_candidates((horizon,))


def test_selection_uses_shortest_candidate_passing_every_metric() -> None:
    from .acceptance_thresholds import evaluate_metric_cell
    from .horizon_exploration import HorizonReport, make_horizon_candidates, select_shortest_passing

    h16, h20, h24 = make_horizon_candidates((16, 20, 24))
    reports = (
        HorizonReport(h16, (evaluate_metric_cell(("cross",), {"foot_collision_frame_rate": 0.1}),)),
        HorizonReport(h24, (evaluate_metric_cell(("cross",), {"foot_collision_frame_rate": 0.0}),)),
        HorizonReport(h20, (evaluate_metric_cell(("cross",), {"foot_collision_frame_rate": 0.0}),)),
    )

    selected = select_shortest_passing(reports)

    assert selected.candidate.horizon_steps == 20


def test_selection_rejects_missing_or_failed_metric_cells() -> None:
    from .acceptance_thresholds import evaluate_metric_cell
    from .horizon_exploration import HorizonReport, make_horizon_candidates, select_shortest_passing

    h16, h20 = make_horizon_candidates((16, 20))
    reports = (
        HorizonReport(h16, ()),
        HorizonReport(h20, (evaluate_metric_cell(("cross",), {"calf_collision_frame_rate": 0.1}),)),
    )

    with pytest.raises(RuntimeError, match="no horizon candidate"):
        select_shortest_passing(reports)
