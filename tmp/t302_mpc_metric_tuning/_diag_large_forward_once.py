import json
import os
import sys
from pathlib import Path

import torch

REPO = Path('/mnt/mydisk/lhy/testPvcnnWithIsaacsim')
GO2 = REPO / 'Go2Pvcnn'
for p in (str(REPO), str(GO2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Go2Pvcnn.tests.fixtures import viewer_runtime_diagnostics as viewer_diag
from Go2Pvcnn.tests.test_mpc_body_leg_collision_headless import (
    _min_root_distance_to_obstacle,
    _stance_semantic_count,
    _planned_collision_metrics,
)

runtime = viewer_diag.make_real_runtime_fixture(
    num_envs=2,
    planner_backend='mpc',
    device=os.environ.get('MPC_TEST_DEVICE', 'cuda:0'),
    warmup_steps=6,
)
try:
    plan = runtime.plan_case_near_s4_anchor_command_relative(
        'large', command_name='forward', longitudinal_offset_m=-0.35, lateral_offset_m=0.0, z_clearance=0.65
    )
    terrain = runtime._single_env_terrain()
    root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
    anchor = runtime.s4_semantic_course_anchor('large')
    obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=root.device)
    breakdown = getattr(plan.result, 'loss_breakdown', {}) or {}
    row = {
        'min_dist': _min_root_distance_to_obstacle(root, obstacle_xy),
        'required_dist': 0.5 * float(anchor.target_diameter) + 0.08,
        'stance_semantic_count': _stance_semantic_count(plan.result, terrain),
        'summary': plan.summary,
        'anchor_world_xy': list(anchor.world_xy),
        'root_start_xy': root[0,0,:2].detach().cpu().tolist(),
        'root_end_xy': root[0,-1,:2].detach().cpu().tolist(),
        'root_min_y': float(root[0,:,1].min().item()),
        'root_max_y': float(root[0,:,1].max().item()),
        'root_min_x': float(root[0,:,0].min().item()),
        'root_max_x': float(root[0,:,0].max().item()),
        'breakdown': {k: float(torch.as_tensor(v).reshape(-1)[0].item()) for k, v in breakdown.items() if torch.as_tensor(v).numel() > 0},
        'metrics': _planned_collision_metrics(plan.result, terrain, runtime._viewer),
    }
    out = REPO / 'tmp/t302_mpc_metric_tuning/large_forward_once_diag.json'
    out.write_text(json.dumps(row, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(row, indent=2, sort_keys=True))
finally:
    runtime.close()
