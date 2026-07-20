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


def test_loss_config_has_exactly_seven_top_level_weights() -> None:
    cfg = JointMpcRtiCfg()

    assert set(cfg.losses.weights()) == {
        "command",
        "step",
        "contact",
        "swing_speed",
        "terrain",
        "posture",
        "smooth",
    }


def test_solver_state_has_no_recovery_or_independent_control() -> None:
    fields = set(JointMpcRtiSolverState.__dataclass_fields__)

    assert fields == {"trajectory", "gait_phase", "valid"}
