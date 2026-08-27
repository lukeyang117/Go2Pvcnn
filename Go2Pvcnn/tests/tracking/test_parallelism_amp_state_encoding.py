import torch

from tracking.managers.parallelism_amp_manager import ParallelismAmpManager


def test_reconstruct_and_encode_returns_936_finite_values():
    manager = ParallelismAmpManager(2, "cpu")
    deltas = torch.randn(2, 23, 39)
    encoded = manager.reconstruct_and_encode(torch.zeros(2, 39), deltas)
    assert encoded.shape == (2, 936)
    assert torch.isfinite(encoded).all()


def test_local_encoding_removes_common_root_translation():
    manager = ParallelismAmpManager(1, "cpu")
    deltas = torch.zeros(1, 23, 39)
    deltas[..., 24] = 0.1
    first = manager.reconstruct_and_encode(torch.zeros(1, 39), deltas)
    shifted_anchor = torch.zeros(1, 39)
    shifted_anchor[..., 24:27] = torch.tensor([7.0, -2.0, 1.0])
    second = manager.reconstruct_and_encode(shifted_anchor, deltas)
    assert torch.allclose(first, second, atol=1e-5)
