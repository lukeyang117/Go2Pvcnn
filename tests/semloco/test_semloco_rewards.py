import torch

from semloco.semloco_rewards import semantic_foothold_tracking_reward, semloco_clearance_penalty


def test_foothold_valid_mask_and_zero_valid():
    feet = torch.zeros(2, 4, 3); target = feet.clone(); swing = torch.ones(2, 4, dtype=torch.bool)
    valid = torch.tensor([[True, False, False, False], [False, False, False, False]])
    reward = semantic_foothold_tracking_reward(feet, target, swing, valid)
    assert torch.allclose(reward, torch.tensor([1.0, 0.0]))


def test_clearance_is_one_sided_and_masked():
    feet = torch.zeros(1, 4, 3); feet[..., 2] = 0.2
    phase = torch.full((1, 4), 0.5); mask = torch.ones(1, 4, dtype=torch.bool)
    assert torch.equal(semloco_clearance_penalty(feet, phase, mask), torch.zeros(1))
    feet[..., 2] = 0.0
    assert (semloco_clearance_penalty(feet, phase, mask) < 0).all()
