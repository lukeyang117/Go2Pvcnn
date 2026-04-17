from __future__ import annotations

import torch

from tests.fixtures.viewer_runtime_diagnostics import build_command_cases


def test_build_command_cases_includes_forward_command():
    cases = build_command_cases(device=torch.device("cpu"), num_envs=1)

    assert "forward" in cases
    assert cases["forward"].shape == (1, 3)
    assert torch.linalg.vector_norm(cases["forward"]).item() > 0
