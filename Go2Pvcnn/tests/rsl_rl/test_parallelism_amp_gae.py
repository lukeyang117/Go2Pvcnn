import torch

from rsl_rl.storage.parallelism_amp_storage import ParallelismAMPStorage, combine_advantages


def test_amp_gae_cuts_at_inactive_boundary():
    storage = ParallelismAMPStorage(1, 8, [2], [2], [1], "cpu")
    storage.amp_rewards[:, 0, 0] = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1], dtype=torch.float)
    storage.amp_active[:, 0, 0] = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1], dtype=torch.float)
    storage.amp_values.zero_()
    storage.compute_returns(torch.zeros(1, 1), torch.zeros(1, 1), 0.99, 0.95)
    assert torch.equal(storage.amp_advantages[2:5], torch.zeros(3, 1, 1))
    assert storage.amp_advantages[1].abs() > 1e-6


def test_inactive_rows_keep_exact_base_actor_advantage():
    base = torch.tensor([[1.0], [2.0], [3.0]])
    amp = torch.tensor([[10.0], [20.0], [30.0]])
    mask = torch.tensor([[1.0], [0.0], [1.0]])
    combined = combine_advantages(base, amp, mask, 0.1)
    assert combined[1] == base[1]
