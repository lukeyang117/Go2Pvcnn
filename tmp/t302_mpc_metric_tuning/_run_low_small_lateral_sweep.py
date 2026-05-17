from __future__ import annotations
import json
from pathlib import Path
import torch
from Go2Pvcnn.tests.fixtures import viewer_runtime_diagnostics as viewer_diag
from Go2Pvcnn.tests.test_mpc_body_leg_collision_headless import (
    _crosses_obstacle_along_command,
    _planned_collision_metrics,
    _stance_semantic_count,
    _stance_semantic_ratio,
)

out = Path('tmp/t302_mpc_metric_tuning/low_small_lateral_crossing_sweep.jsonl')
rows = []
runtime = viewer_diag.make_real_runtime_fixture(
    num_envs=2,
    planner_backend='mpc',
    device='cuda:0',
    warmup_steps=6,
    semantic_small_height_m=0.16,
)
try:
    runtime.mpc_planner_cfg.diagnostics.enabled = True
    candidates = [
        ('base', 8.0, 0.06, 24),
        ('w12_m06', 12.0, 0.06, 24),
        ('w16_m06', 16.0, 0.06, 24),
        ('w24_m06', 24.0, 0.06, 24),
        ('w16_m10', 16.0, 0.10, 24),
        ('w24_m10', 24.0, 0.10, 24),
        ('w32_m10', 32.0, 0.10, 24),
        ('w32_m14', 32.0, 0.14, 24),
        ('w32_m10_opt32', 32.0, 0.10, 32),
    ]
    for name, weight, margin, opt_steps in candidates:
        runtime.mpc_planner_cfg.losses.low_small_crossing.weight = float(weight)
        runtime.mpc_planner_cfg.losses.low_small_crossing.pass_margin_m = float(margin)
        runtime.mpc_planner_cfg.runtime.optimize_steps = int(opt_steps)
        for command_name in ('lateral_left', 'lateral_right'):
            plan = runtime.plan_case_near_s4_anchor_command_relative(
                'small',
                command_name=command_name,
                longitudinal_offset_m=-0.35,
                lateral_offset_m=0.0,
                z_clearance=0.65,
            )
            terrain = runtime._single_env_terrain()
            root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
            anchor = runtime.s4_semantic_course_anchor('small')
            obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=root.device)
            crossed, min_lateral = _crosses_obstacle_along_command(root, obstacle_xy, plan.command)
            direction = plan.command[0, :2].to(dtype=root.dtype, device=root.device)
            direction = direction / torch.linalg.vector_norm(direction).clamp_min(1.0e-6)
            along = ((root[0, :, :2] - obstacle_xy) * direction).sum(dim=-1)
            loss_breakdown = getattr(plan.result, 'loss_breakdown', None) or {}
            row = {
                'kind': 'sample',
                'name': name,
                'command': command_name,
                'crossed': bool(crossed),
                'along_start': float(along[0].item()),
                'along_end': float(along[-1].item()),
                'along_max': float(along.max().item()),
                'min_lateral': float(min_lateral),
                'root_dxy': [float(v) for v in (root[0, -1, :2] - root[0, 0, :2]).detach().cpu().tolist()],
                'stance_semantic_count': _stance_semantic_count(plan.result, terrain),
                'stance_semantic_ratio': _stance_semantic_ratio(plan.result, terrain),
                'metrics': _planned_collision_metrics(plan.result, terrain, runtime._viewer),
                'low_small_crossing_weight': float(weight),
                'pass_margin_m': float(margin),
                'optimize_steps': int(opt_steps),
            }
            for key in ('tracking', 'low_small_crossing', 'semantic_contact_avoid', 'stance_semantic', 'smoothness', 'root_foot_center'):
                if key in loss_breakdown:
                    row['loss_' + key] = float(torch.as_tensor(loss_breakdown[key]).reshape(-1)[0].item())
            rows.append(row)
            out.open('a', encoding='utf-8').write(json.dumps(row, sort_keys=True) + '\n')
finally:
    runtime.close()
summary = {
    'kind': 'summary',
    'sample_count': len(rows),
    'all_crossed': all(r['crossed'] for r in rows if r['kind'] == 'sample'),
    'min_along_end_by_candidate': {},
}
for name, *_ in candidates:
    vals = [r['along_end'] for r in rows if r['name'] == name]
    if vals:
        summary['min_along_end_by_candidate'][name] = min(vals)
out.open('a', encoding='utf-8').write(json.dumps(summary, sort_keys=True) + '\n')
