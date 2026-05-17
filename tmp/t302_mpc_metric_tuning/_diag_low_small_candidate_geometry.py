from __future__ import annotations
import json
from pathlib import Path
import torch
from Go2Pvcnn.tests.fixtures import viewer_runtime_diagnostics as viewer_diag
from Go2Pvcnn.tests.test_mpc_body_leg_collision_headless import _crosses_obstacle_along_command
from extension.batch_mpc_planner.losses.terrain_clearance import _semantic_id_mask, _terrain_grid_world_xy, low_small_crossing_progress_loss
from extension.batch_mpc_planner.terrain import height_at

out = Path('tmp/t302_mpc_metric_tuning/low_small_candidate_geometry_diag.jsonl')
out.write_text('', encoding='utf-8')
runtime = viewer_diag.make_real_runtime_fixture(num_envs=2, planner_backend='mpc', device='cuda:0', warmup_steps=6, semantic_small_height_m=0.16)
try:
    runtime.mpc_planner_cfg.diagnostics.enabled = True
    for command_name in ('forward','lateral_left','lateral_right'):
        plan = runtime.plan_case_near_s4_anchor_command_relative('small', command_name=command_name, longitudinal_offset_m=-0.35, lateral_offset_m=0.0, z_clearance=0.65)
        terrain = runtime._single_env_terrain()
        root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
        rpy = runtime._viewer._quat_wxyz_to_rpy(torch.as_tensor(plan.result.root_quat_w, dtype=torch.float32).reshape(-1,4)).reshape_as(root)
        command = plan.command.to(dtype=root.dtype, device=root.device)
        anchor = runtime.s4_semantic_course_anchor('small')
        obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=root.device)
        crossed, min_lat = _crosses_obstacle_along_command(root, obstacle_xy, plan.command)
        grid_xy = _terrain_grid_world_xy(terrain, dtype=root.dtype, device=root.device)
        height = torch.as_tensor(terrain.height_map, dtype=root.dtype, device=root.device)
        sem = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=root.device)
        grid_z = height.reshape(1, -1)
        grid_sem = sem.reshape(1, -1)
        root0 = root[:, 0]
        root_end = root[:, -1]
        root_ground_z = height_at(terrain, root0[:, None, :2]).reshape(1).to(dtype=root.dtype, device=root.device)
        small = _semantic_id_mask(grid_sem, (1,))
        low_small = torch.logical_and(small, (grid_z - root_ground_z[:, None]) <= 0.30)
        cmd_xy = command[:, :2]
        heading = cmd_xy / torch.linalg.vector_norm(cmd_xy, dim=-1).clamp_min(1e-6).unsqueeze(-1)
        delta = grid_xy - root0[:, None, :2]
        forward = (delta * heading[:, None, :]).sum(dim=-1)
        lateral = delta[..., 0] * (-heading[:, 1]).view(1, 1) + delta[..., 1] * heading[:, 0].view(1, 1)
        candidate = low_small & (forward >= 0.0) & (forward <= 1.0) & (torch.abs(lateral) <= 0.28)
        desired = torch.where(candidate, forward + 0.24 + 0.06, torch.zeros_like(forward))
        required = desired.amax(dim=-1)
        progress = ((root_end[:, :2] - root0[:, :2]) * heading).sum(dim=-1)
        loss = low_small_crossing_progress_loss(terrain, root, rpy, command, small_ids=(1,), high_small_relative_height_m=0.30, corridor_width_m=0.28, forward_distance_m=1.0, pass_margin_m=0.06, obstacle_depth_m=0.24, linear_speed_eps=1e-4)
        row = {
            'command': command_name,
            'crossed': bool(crossed),
            'min_lateral_to_anchor': float(min_lat),
            'root0_xy': [float(v) for v in root0[0, :2].detach().cpu().tolist()],
            'root_end_xy': [float(v) for v in root_end[0, :2].detach().cpu().tolist()],
            'anchor_xy': [float(v) for v in obstacle_xy.detach().cpu().tolist()],
            'progress': float(progress[0]),
            'required': float(required[0]),
            'loss': float(loss[0]),
            'small_count': int(small.sum()),
            'low_small_count': int(low_small.sum()),
            'candidate_count': int(candidate.sum()),
        }
        if bool(low_small.any()):
            f = forward[low_small]
            l = lateral[low_small]
            row.update({
                'low_small_forward_min': float(f.min()),
                'low_small_forward_max': float(f.max()),
                'low_small_lateral_abs_min': float(torch.abs(l).min()),
                'low_small_lateral_abs_max': float(torch.abs(l).max()),
            })
        if bool(candidate.any()):
            f = forward[candidate]
            l = lateral[candidate]
            row.update({
                'cand_forward_min': float(f.min()),
                'cand_forward_max': float(f.max()),
                'cand_lateral_abs_min': float(torch.abs(l).min()),
                'cand_lateral_abs_max': float(torch.abs(l).max()),
            })
        out.open('a', encoding='utf-8').write(json.dumps(row, sort_keys=True)+'\n')
finally:
    runtime.close()
