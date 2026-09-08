from types import SimpleNamespace

import pytest
import torch

from ame_baseline.ame_observations import downsampled_ame_scan


class _FakeCfg:
    name = "scanner"


class _FakeEnv:
    def __init__(self, num_rays=4):
        side = int(num_rays**0.5) if num_rays == 4 else 1
        hits = torch.zeros(1, num_rays, 3)
        if num_rays == 4:
            hits[0, :, 0] = torch.tensor([-1.0, 1.0, -1.0, 1.0])
            hits[0, :, 1] = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        semantic = torch.tensor([[[0, 1], [2, 0]]]) if num_rays == 4 else torch.zeros(1, 1, num_rays, dtype=torch.long)
        self.scene = SimpleNamespace(
            sensors={
                "scanner": SimpleNamespace(
                    data=SimpleNamespace(
                        ray_hits_w=hits,
                        pos_w=torch.zeros(1, 3),
                        quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                        semantic_map=semantic,
                    )
                )
            }
        )


def test_downsampled_ame_scan_has_xyz_and_one_hot_channels():
    out = downsampled_ame_scan(_FakeEnv(), _FakeCfg(), target_size=2)
    assert out.shape == (1, 6, 2, 2)
    assert torch.equal(out[:, 3:].sum(dim=1), torch.ones(1, 2, 2))
    assert torch.allclose(out[:, :3].mean(), torch.tensor(0.0))


def test_downsampled_ame_scan_rejects_non_square_scan():
    with pytest.raises(ValueError, match="perfect square"):
        downsampled_ame_scan(_FakeEnv(num_rays=3), _FakeCfg(), target_size=2)
