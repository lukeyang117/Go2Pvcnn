import json
import os
import sys
from pathlib import Path

import torch

REPO = Path("/mnt/mydisk/lhy/testPvcnnWithIsaacsim")
GO2 = REPO / "Go2Pvcnn"
for p in (str(REPO), str(GO2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Go2Pvcnn.tests.fixtures import viewer_runtime_diagnostics as viewer_diag
from Go2Pvcnn.tests.test_mpc_body_leg_collision_headless import (
    _min_root_distance_to_obstacle,
    _planned_collision_metrics,
    _stance_semantic_count,
)


def _row(runtime, *, name, apply_cfg):
    apply_cfg(runtime.mpc_planner_cfg)
    plan = runtime.plan_case_near_s4_anchor_command_relative(
        "large",
        command_name="forward",
        longitudinal_offset_m=-0.35,
        lateral_offset_m=0.0,
        z_clearance=0.65,
    )
    terrain = runtime._single_env_terrain()
    root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
    anchor = runtime.s4_semantic_course_anchor("large")
    obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=root.device)
    breakdown = getattr(plan.result, "loss_breakdown", {}) or {}
    metrics = _planned_collision_metrics(plan.result, terrain, runtime._viewer)
    return {
        "name": name,
        "min_dist": _min_root_distance_to_obstacle(root, obstacle_xy),
        "required_dist": 0.5 * float(anchor.target_diameter) + 0.08,
        "stance_semantic_count": _stance_semantic_count(plan.result, terrain),
        "root_start_xy": root[0, 0, :2].detach().cpu().tolist(),
        "root_end_xy": root[0, -1, :2].detach().cpu().tolist(),
        "root_min_x": float(root[0, :, 0].min().item()),
        "root_max_x": float(root[0, :, 0].max().item()),
        "root_min_y": float(root[0, :, 1].min().item()),
        "root_max_y": float(root[0, :, 1].max().item()),
        "summary": plan.summary,
        "metrics": metrics,
        "breakdown": {
            k: float(torch.as_tensor(v).reshape(-1)[0].item())
            for k, v in breakdown.items()
            if torch.as_tensor(v).numel() > 0
        },
    }


def _base(cfg):
    cfg.runtime.optimize_steps = 24
    cfg.runtime.lr = 1.0e-2
    cfg.runtime.grad_clip_norm = 10.0
    cfg.losses.high_obstacle_avoidance.weight = 250.0
    cfg.losses.high_obstacle_avoidance.lateral_clearance_m = 0.45
    cfg.losses.high_obstacle_avoidance.corridor_width_m = 0.40
    cfg.losses.high_obstacle_avoidance.longitudinal_influence_m = 0.55
    cfg.losses.semantic_obstacle.weight = 1.0
    cfg.losses.root_foot_center.weight = 1.0


def _variant(**kwargs):
    def apply(cfg):
        _base(cfg)
        for dotted, value in kwargs.items():
            obj = cfg
            parts = dotted.split("__")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

    return apply


def main():
    out_path = REPO / "tmp/t302_mpc_metric_tuning/large_forward_sweep.jsonl"
    rows = []
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        device=os.environ.get("MPC_TEST_DEVICE", "cuda:0"),
        warmup_steps=6,
    )
    variants = [
        ("baseline", _variant()),
        ("opt32", _variant(runtime__optimize_steps=32)),
        ("opt48", _variant(runtime__optimize_steps=48)),
        ("lr02", _variant(runtime__lr=2.0e-2)),
        ("clip50", _variant(runtime__grad_clip_norm=50.0)),
        ("avoid_w500", _variant(losses__high_obstacle_avoidance__weight=500.0)),
        ("avoid_w1000", _variant(losses__high_obstacle_avoidance__weight=1000.0)),
        ("avoid_clear060", _variant(losses__high_obstacle_avoidance__lateral_clearance_m=0.60)),
        ("avoid_influence100", _variant(losses__high_obstacle_avoidance__longitudinal_influence_m=1.0)),
        ("no_root_center", _variant(losses__root_foot_center__weight=0.0)),
        ("sem_w02", _variant(losses__semantic_obstacle__weight=0.2)),
        ("opt48_w1000", _variant(runtime__optimize_steps=48, losses__high_obstacle_avoidance__weight=1000.0)),
    ]
    try:
        for name, apply in variants:
            row = _row(runtime, name=name, apply_cfg=apply)
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
        out_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
