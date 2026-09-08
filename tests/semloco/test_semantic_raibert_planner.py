import torch

from semloco.semantic_raibert_planner import SemanticRaibertPlanner


def _inputs():
    root = torch.zeros(2, 3)
    quat = torch.zeros(2, 4); quat[:, 0] = 1
    feet = torch.tensor([[[0.3, 0.2, 0.0], [0.3, -0.2, 0.0], [-0.3, 0.2, 0.0], [-0.3, -0.2, 0.0]]]).repeat(2, 1, 1)
    cmd = torch.tensor([[1.0, 0.2, 0.5], [-1.0, -0.2, -0.5]])
    scan = {"height": torch.zeros(2, 21, 21), "semantic": torch.zeros(2, 21, 21, dtype=torch.long), "origin": torch.tensor([[-0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]]), "resolution": 0.05}
    return root, quat, feet, cmd, scan


def test_planner_shapes_and_finite():
    root, quat, feet, cmd, scan = _inputs()
    out = SemanticRaibertPlanner().plan(root, quat, torch.zeros(2, 3), torch.zeros(2, 3), feet, cmd, torch.zeros(2, 4), scan)
    assert out.target_foothold_w.shape == (2, 4, 3)
    assert out.valid.shape == (2, 4)
    assert torch.isfinite(out.target_foothold_w).all()


def test_obstacle_marks_only_blocked_candidates_invalid():
    root, quat, feet, cmd, scan = _inputs()
    scan["semantic"][0, 10, 10] = 2
    out = SemanticRaibertPlanner().plan(root, quat, torch.zeros(2, 3), torch.zeros(2, 3), feet, cmd, torch.zeros(2, 4), scan)
    assert out.valid.all()
    assert torch.linalg.vector_norm(out.target_foothold_w[0, :, :2] - torch.tensor([0.0, 0.0]), dim=-1).min() > 0.0
