from __future__ import annotations

import torch

from extension.joint_mpc_rti.tensor_constants import constant_like


def test_same_named_constant_tracks_changed_configuration_values() -> None:
    reference = torch.zeros(1)

    first = constant_like(reference, "test_tunable_constant", (0.01,))
    second = constant_like(reference, "test_tunable_constant", (0.15,))

    torch.testing.assert_close(first, torch.tensor([0.01]))
    torch.testing.assert_close(second, torch.tensor([0.15]))
