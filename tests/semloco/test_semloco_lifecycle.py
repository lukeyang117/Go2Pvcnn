import torch

from semloco.semloco_lifecycle import SemlocoTargetCache


class Output:
    target_foothold_w = torch.ones(1, 4, 3)
    valid = torch.ones(1, 4, dtype=torch.bool)


def test_target_is_locked_during_swing():
    cache = SemlocoTargetCache(1, "cpu")
    cache.update_swing_targets(Output(), torch.tensor([[True, False, False, False]]))
    first = cache.target.clone()
    Output.target_foothold_w = torch.full((1, 4, 3), 2.0)
    cache.update_swing_targets(Output(), torch.tensor([[True, False, False, False]]))
    assert torch.equal(cache.target, first)
    cache.update_swing_targets(Output(), torch.tensor([[False, False, False, False]]))
    cache.update_swing_targets(Output(), torch.tensor([[True, False, False, False]]))
    assert torch.equal(cache.target[:, 0], torch.full((1, 3), 2.0))
