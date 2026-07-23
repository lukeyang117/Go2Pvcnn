from __future__ import annotations

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.types import JointMpcRtiTrajectory


def test_final_contract_is_h30_current_map_and_one_rti() -> None:
    cfg = JointMpcRtiCfg()

    assert cfg.runtime.horizon_steps == 30
    assert cfg.runtime.future_frames == 30
    assert cfg.runtime.state_nodes == 31
    assert cfg.runtime.dt == 0.02
    assert cfg.runtime.max_field_age_steps == 0
    assert cfg.runtime.sqp_iterations_per_step == 1
    assert cfg.gait.period_steps == 24
    assert cfg.gait.swing_steps == cfg.gait.stance_steps == 12
    assert cfg.solver.line_search_alphas == (1.0, 0.5, 0.25, 0.125, 0.0)


def test_trajectory_contract_separates_internal_nodes_from_future_frames() -> None:
    fields = JointMpcRtiTrajectory.__dataclass_fields__

    assert "state_nodes" in fields
    assert "future_state" in fields
