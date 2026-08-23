import torch


def _context(**kwargs):
    from tracking.mdp.distillation import terrain_imitation_context_from_metadata

    return terrain_imitation_context_from_metadata(**kwargs)


def test_flat_weights_stay_one_and_complex_weights_reach_endpoints():
    context = _context(
        terrain_types=torch.tensor([0, 1, 2, 3]),
        terrain_levels=torch.tensor([0, 9, 0, 9]),
        terrain_column_names=("flat_dense_small_obstacles", "flat", "boxes", "pyramid_stairs"),
        end_multipliers={
            "flat_dense_small_obstacles": 1.0,
            "flat": 1.0,
            "boxes": 0.0,
            "pyramid_stairs": 0.0,
        },
        powers={
            "flat_dense_small_obstacles": 1.0,
            "flat": 1.0,
            "boxes": 1.5,
            "pyramid_stairs": 2.0,
        },
        num_rows=10,
        plan_valid=torch.ones(4),
    )
    assert torch.equal(context[:, 0], torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(context[:, 1], torch.ones(4))


def test_complex_weight_is_monotone_for_intermediate_levels():
    context = _context(
        terrain_types=torch.zeros(10, dtype=torch.long),
        terrain_levels=torch.arange(10),
        terrain_column_names=("pyramid_stairs",),
        end_multipliers={"pyramid_stairs": 0.0},
        powers={"pyramid_stairs": 2.0},
        num_rows=10,
        plan_valid=torch.ones(10),
    )
    assert torch.all(context[:-1, 0] >= context[1:, 0])
    assert context[0, 0] == 1.0
    assert context[-1, 0] == 0.0


def test_invalid_plan_zeroes_only_imitation_context():
    context = _context(
        terrain_types=torch.tensor([0, 0]),
        terrain_levels=torch.tensor([0, 0]),
        terrain_column_names=("flat",),
        end_multipliers={"flat": 1.0},
        powers={"flat": 1.0},
        num_rows=10,
        plan_valid=torch.tensor([1.0, 0.0]),
    )
    assert torch.equal(context[:, 0], torch.tensor([1.0, 0.0]))
    assert torch.equal(context[:, 1], torch.tensor([1.0, 0.0]))


def test_unknown_terrain_has_zero_context():
    context = _context(
        terrain_types=torch.tensor([4]),
        terrain_levels=torch.tensor([0]),
        terrain_column_names=("flat",),
        end_multipliers={"flat": 1.0},
        powers={"flat": 1.0},
        num_rows=10,
        plan_valid=torch.ones(1),
    )
    assert torch.equal(context, torch.zeros(1, 2))
