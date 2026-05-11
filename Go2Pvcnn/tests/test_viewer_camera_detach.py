from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from extension.viz import go2_foostep_planner as viewer


class _FakeSim:
    def __init__(self):
        self.calls = []

    def set_camera_view(self, camera_position, target_position):
        self.calls.append((camera_position, target_position))


class _FakeEnv:
    def __init__(self):
        self.sim = _FakeSim()


def test_update_camera_accepts_requires_grad_tensors():
    env = _FakeEnv()
    root_pos = torch.tensor([[1.0, 2.0, 0.3]], dtype=torch.float32, requires_grad=True)
    root_yaw = torch.tensor([0.2], dtype=torch.float32, requires_grad=True)

    viewer._update_camera(env, root_pos=root_pos, root_yaw=root_yaw, distance=3.2, height=1.6)

    assert len(env.sim.calls) == 1


def test_adapt_mpc_result_detaches_autograd_graph():
    batch = 1
    horizon = 4
    result = SimpleNamespace(
        root_pos=torch.randn(batch, horizon, 3, dtype=torch.float32, requires_grad=True),
        root_rpy=torch.randn(batch, horizon, 3, dtype=torch.float32, requires_grad=True),
        foot_pos=torch.randn(batch, horizon, 4, 3, dtype=torch.float32, requires_grad=True),
        planned_touchdown_w=torch.randn(batch, horizon, 4, 3, dtype=torch.float32, requires_grad=True),
        joint_angles=torch.randn(batch, horizon, 12, dtype=torch.float32, requires_grad=True),
        contact_state=torch.ones(batch, horizon, 4, dtype=torch.bool),
        touchdown_seq=torch.randn(batch, 4, 2, 3, dtype=torch.float32, requires_grad=True),
        status=torch.zeros(batch, dtype=torch.long),
        feasible=torch.ones(batch, dtype=torch.bool),
        safe_fallback=torch.zeros(batch, dtype=torch.bool),
        hard_reason_mask=torch.zeros(batch, 4, dtype=torch.bool),
    )

    adapted = viewer._adapt_mpc_result_for_viewer(result)

    assert not adapted.root_pos_w.requires_grad
    assert not adapted.root_quat_w.requires_grad
    assert not adapted.foot_pos_w.requires_grad
    assert not adapted.joint_angles.requires_grad
