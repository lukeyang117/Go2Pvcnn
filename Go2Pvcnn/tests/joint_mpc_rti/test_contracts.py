from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_joint_mpc_rti_config_defaults_match_fixed_shape_contract() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    assert cfg.runtime.horizon_steps == 16
    assert cfg.runtime.dt == pytest.approx(0.02)
    assert cfg.runtime.sqp_iterations_per_step == 1
    assert tuple(cfg.solver.line_search_alphas) == (1.0, 0.25)


def test_factory_accepts_joint_mpc_rti_without_changing_mpc_default() -> None:
    from extension.trajectory_manager_factory import planner_backend_from_cfg

    assert planner_backend_from_cfg(SimpleNamespace()) == "mpc"
    assert planner_backend_from_cfg(SimpleNamespace(planner_backend="joint_mpc_rti")) == "joint_mpc_rti"


def test_state_contract_rejects_wrong_joint_shape() -> None:
    from extension.joint_mpc_rti.types import JointMpcRtiState

    with pytest.raises(ValueError, match="joint_pos"):
        JointMpcRtiState(
            root_pos_w=torch.zeros(2, 3),
            root_rpy_w=torch.zeros(2, 3),
            joint_pos=torch.zeros(2, 11),
            root_lin_vel_b=torch.zeros(2, 3),
            root_ang_vel_b=torch.zeros(2, 3),
            joint_vel=torch.zeros(2, 12),
        )
