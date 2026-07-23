from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.types import JointMpcRtiSolverState


def test_production_contract_is_h30_24_12_and_one_rti() -> None:
    cfg = JointMpcRtiCfg()

    assert cfg.runtime.horizon_steps == 30
    assert cfg.runtime.dt == 0.02
    assert cfg.runtime.sqp_iterations_per_step == 1
    assert cfg.gait.period_steps == 24
    assert cfg.gait.swing_steps == 12
    assert cfg.gait.stance_steps == 12
    assert cfg.solver.line_search_alphas == (1.0, 0.5, 0.25, 0.125, 0.0)


def test_lq_config_has_exactly_eight_residual_family_weights() -> None:
    cfg = JointMpcRtiCfg()

    assert set(cfg.lq_cost.family_names()) == {
        "velocity",
        "posture",
        "root",
        "swing",
        "touchdown",
        "smooth",
        "warm",
        "slack",
    }


def test_solver_state_has_only_warm_lifecycle_state() -> None:
    fields = set(JointMpcRtiSolverState.__dataclass_fields__)

    assert fields == {
        "trajectory",
        "gait_phase",
        "initialized",
        "stance_anchor_w",
        "preview_tail_state",
    }
