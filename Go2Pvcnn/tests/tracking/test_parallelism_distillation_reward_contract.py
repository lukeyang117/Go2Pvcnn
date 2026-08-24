import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function not found: {name}")


def test_parallelism_collision_function_returns_raw_event_for_negative_weight() -> None:
    path = ROOT / "tracking/mdp/rewards.py"
    function = _function_node(path, "parallelism_geometry_collision_penalty")
    returns = [node for node in function.body if isinstance(node, ast.Return)]
    assert returns, "collision reward must return a value"
    returned = returns[-1].value
    assert isinstance(returned, ast.Name)
    assert returned.id == "event"


def test_parallelism_collision_configs_keep_negative_penalty_weight() -> None:
    source = (ROOT / "tracking/parallelism_small_obstacles_env_cfg.py").read_text()
    assert "parallelism_geometry_collision" in source
    assert "weight=-2.0" in source


def test_distillation_reward_contract_removes_reference_rewards() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_distillation_env_cfg.py").read_text()
    for term in (
        "track_root_pos",
        "track_root_rot",
        "reference_joint_pos",
        "reference_joint_vel",
        "reference_joint_max",
        "reference_foot_pos",
        "reference_active_swing_foot_max",
        "active_swing_foot_on_small_obstacle",
        "semantic_contact_collision",
    ):
        assert f"{term} = None" in source
    assert "undesired_contacts = RewTerm(" in source
    assert "func=isaac_mdp.undesired_contacts" in source
    assert '"threshold": 1.0' in source
    assert '".*_thigh"' in source
    assert '"std": 0.5' in source
    assert '"threshold"] = 0.5' in source


def test_distillation_script_uses_requested_teacher_and_ppo_dominance() -> None:
    source = (ROOT / "scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh").read_text()
    assert source.count("model_9899.pt") == 1
    assert "--ppo-coef 1.0" in source
    assert "--teacher-coef 0.1" in source
