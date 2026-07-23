from __future__ import annotations

import ast
import inspect
import textwrap

import torch

from extension.joint_mpc_rti.model.analytic_ik import go2_analytic_ik
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk


def _reachable_targets(batch: int, nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    root_pos = torch.zeros(batch, nodes, 3, dtype=dtype)
    root_pos[..., 2] = 0.34
    root_pos[..., 0] = torch.linspace(-0.1, 0.2, nodes, dtype=dtype)
    root_rpy = torch.zeros_like(root_pos)
    root_rpy[..., 0] = 0.04
    root_rpy[..., 1] = -0.03
    root_rpy[..., 2] = torch.linspace(-0.2, 0.3, nodes, dtype=dtype)
    joint = torch.tensor((0.08, 0.75, -1.55) * 4, dtype=dtype).expand(batch, nodes, -1).clone()
    flat_fk = go2_fk(
        root_pos.reshape(-1, 3),
        root_rpy.reshape(-1, 3),
        joint.reshape(-1, 12),
    )
    return root_pos, root_rpy, flat_fk.foot_pos_w.reshape(batch, nodes, 4, 3)


def test_batched_analytic_ik_matches_fk_for_b31x4_targets() -> None:
    root_pos, root_rpy, foot_target = _reachable_targets(batch=8, nodes=31)

    q, reachable = go2_analytic_ik(root_pos, root_rpy, foot_target)
    fk = go2_fk(root_pos, root_rpy, q.reshape(8, 31, 12)).foot_pos_w

    assert q.shape == (8, 31, 4, 3)
    assert reachable.shape == (8, 31, 4)
    assert reachable.all()
    torch.testing.assert_close(fk, foot_target, atol=2e-5, rtol=2e-5)


def test_analytic_ik_reports_unreachable_without_silent_joint_clipping() -> None:
    root_pos = torch.zeros(2, 31, 3)
    root_rpy = torch.zeros_like(root_pos)
    foot_target = torch.zeros(2, 31, 4, 3)
    foot_target[..., 0] = 2.0

    q, reachable = go2_analytic_ik(root_pos, root_rpy, foot_target)

    assert torch.isfinite(q).all()
    assert not reachable.any()


def test_analytic_ik_source_has_no_python_time_or_leg_loop() -> None:
    source = inspect.getsource(go2_analytic_ik)
    tree = ast.parse(textwrap.dedent(source))

    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))


def test_selected_leg_ik_matches_all_leg_ik() -> None:
    from extension.joint_mpc_rti.model.analytic_ik import go2_analytic_ik_selected

    root_pos, root_rpy, foot_target = _reachable_targets(batch=2, nodes=1)
    all_joint, all_reachable = go2_analytic_ik(root_pos, root_rpy, foot_target)
    leg = torch.tensor(((0,), (3,)))
    batch = torch.arange(2)[:, None]
    node = torch.zeros(2, 1, dtype=torch.long)
    selected_joint, selected_reachable = go2_analytic_ik_selected(
        root_pos,
        root_rpy,
        foot_target[batch, node, leg],
        leg,
    )

    torch.testing.assert_close(selected_joint, all_joint[batch, node, leg])
    assert torch.equal(selected_reachable, all_reachable[batch, node, leg])
