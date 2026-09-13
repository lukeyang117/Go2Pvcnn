from __future__ import annotations

import torch

from scripts.policy_benchmark import _contact_masks


class _Data:
    def __init__(self, force_matrix_w: torch.Tensor) -> None:
        self.force_matrix_w = force_matrix_w


class _Sensor:
    def __init__(self, force_matrix_w: torch.Tensor) -> None:
        self.data = _Data(force_matrix_w)


class _Scene:
    def __init__(self) -> None:
        self.sensors = {
            "semantic_contact_small": _Sensor(
                torch.tensor([[[[0.0, 0.0, 0.0]]], [[[0.0, 2.0, 0.0]]]])
            ),
            "semantic_contact_large": _Sensor(
                torch.tensor([[[[0.0, 3.0, 0.0]]], [[[0.0, 0.0, 0.0]]]])
            ),
        }


class _Env:
    num_envs = 2
    scene = _Scene()
    termination_manager = None


def test_contact_masks_read_semantic_contact_sensor_force_matrices() -> None:
    large, small, fallen = _contact_masks(_Env(), 2, torch.device("cpu"))

    assert large.tolist() == [True, False]
    assert small.tolist() == [False, True]
    assert fallen.tolist() == [False, False]
