from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2] / "extension/joint_mpc_rti"
FORBIDDEN_FILES = (
    "losses/semantic.py",
    "losses/terrain.py",
    "losses/step.py",
    "losses/swing_speed.py",
    "losses/contact.py",
    "losses/rollout_objective.py",
    "losses/objective.py",
    "losses/barriers.py",
    "losses/command.py",
    "losses/posture.py",
    "losses/smoothness.py",
    "losses/__init__.py",
    "terrain/cost_map.py",
    "runtime/reference_buffer.py",
    "solver/primal_dual_ilqr.py",
    "solver/associative_tvlqr.py",
    "solver/gauss_newton.py",
    "solver/linearization.py",
    "model/rollout.py",
)


def _production_imports() -> set[str]:
    roots = ("planner.py", "runtime/manager.py", "runtime/cuda_graph.py")
    pending = list(roots)
    visited: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in visited:
            continue
        visited.add(relative)
        tree = ast.parse((PACKAGE / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            prefix = "extension.joint_mpc_rti."
            if not node.module.startswith(prefix):
                continue
            imported = node.module[len(prefix) :].replace(".", "/") + ".py"
            if (PACKAGE / imported).is_file() and imported not in visited:
                pending.append(imported)
    return visited


def test_superseded_modules_are_deleted_not_merely_bypassed() -> None:
    assert not [relative for relative in FORBIDDEN_FILES if (PACKAGE / relative).exists()]


def test_production_import_graph_contains_no_superseded_module() -> None:
    imports = _production_imports()
    assert not (imports & set(FORBIDDEN_FILES))


def test_final_production_graph_has_no_dynamics_or_output_repair_symbol() -> None:
    source = "\n".join(
        (PACKAGE / relative).read_text(encoding="utf-8")
        for relative in _production_imports()
    )
    for forbidden in (
        "contact_force",
        "friction_cone",
        "rigid_body_dynamics",
        "fallback_to_cold",
        "post_publication_repair",
        "output_projection",
    ):
        assert forbidden not in source


def test_final_terrain_config_has_no_gaussian_occupancy_tuning() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiTerrainCfg

    fields = JointMpcRtiTerrainCfg.__dataclass_fields__
    for obsolete in (
        "small_sigma_m",
        "large_sigma_m",
        "small_gain",
        "large_gain",
        "kernel_radius_cells",
    ):
        assert obsolete not in fields


def test_final_config_exposes_lq_weights_not_legacy_seven_loss_weights() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    assert not hasattr(cfg, "losses")
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
