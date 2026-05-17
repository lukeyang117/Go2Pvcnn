from __future__ import annotations
import json
from pathlib import Path
import torch
from Go2Pvcnn.tests.fixtures import viewer_runtime_diagnostics as viewer_diag
from extension.batch_mpc_planner.losses.terrain_clearance import _terrain_grid_world_xy

out = Path('tmp/t302_mpc_metric_tuning/grid_vs_hits_xy_diag.jsonl')
out.write_text('', encoding='utf-8')
runtime = viewer_diag.make_real_runtime_fixture(num_envs=2, planner_backend='mpc', device='cuda:0', warmup_steps=6, semantic_small_height_m=0.16)
try:
    for command_name in ('forward','lateral_left','lateral_right'):
        plan = runtime.plan_case_near_s4_anchor_command_relative('small', command_name=command_name, longitudinal_offset_m=-0.35, lateral_offset_m=0.0, z_clearance=0.65)
        terrain, ray_hits = runtime._single_env_terrain_and_hits()
        grid_xy = _terrain_grid_world_xy(terrain, dtype=torch.float32, device=terrain.height_map.device)[0]
        hits_xy = torch.as_tensor(ray_hits, dtype=torch.float32, device=grid_xy.device)[:, :2]
        sem = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=grid_xy.device).reshape(-1)
        mask = sem == 1
        root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32, device=grid_xy.device)
        cmd = plan.command.to(dtype=torch.float32, device=grid_xy.device)
        heading = cmd[0,:2] / torch.linalg.vector_norm(cmd[0,:2]).clamp_min(1e-6)
        anchor = runtime.s4_semantic_course_anchor('small')
        anchor_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=grid_xy.device)
        row = {
            'command': command_name,
            'root0_xy': [float(v) for v in root[0,0,:2].detach().cpu()],
            'anchor_xy': [float(v) for v in anchor_xy.detach().cpu()],
            'sensor_pos_xy': [float(v) for v in torch.as_tensor(terrain.sensor_pos_w[0,:2]).detach().cpu()],
            'sensor_yaw': float(torch.as_tensor(terrain.sensor_yaw).reshape(-1)[0].detach().cpu()),
            'grid_hit_abs_diff_max': float((grid_xy - hits_xy).abs().amax().detach().cpu()),
            'grid_hit_abs_diff_mean': float((grid_xy - hits_xy).abs().mean().detach().cpu()),
            'sem_count': int(mask.sum().detach().cpu()),
        }
        for prefix, xy in (('grid', grid_xy), ('hits', hits_xy)):
            if bool(mask.any()):
                vals = xy[mask]
                delta = vals - root[0,0,:2]
                forward = (delta * heading).sum(-1)
                dist_anchor = torch.linalg.vector_norm(vals - anchor_xy, dim=-1)
                row.update({
                    f'{prefix}_sem_xy_mean': [float(v) for v in vals.mean(0).detach().cpu()],
                    f'{prefix}_sem_forward_min': float(forward.min().detach().cpu()),
                    f'{prefix}_sem_forward_max': float(forward.max().detach().cpu()),
                    f'{prefix}_sem_anchor_dist_min': float(dist_anchor.min().detach().cpu()),
                    f'{prefix}_sem_anchor_dist_mean': float(dist_anchor.mean().detach().cpu()),
                })
        out.open('a', encoding='utf-8').write(json.dumps(row, sort_keys=True)+'\n')
finally:
    runtime.close()
