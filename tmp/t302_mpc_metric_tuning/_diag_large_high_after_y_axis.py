from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
from Go2Pvcnn.tests.fixtures import viewer_runtime_diagnostics as viewer_diag
from Go2Pvcnn.tests.test_mpc_body_leg_collision_headless import (
    _crosses_obstacle_along_command,
    _min_root_distance_to_obstacle,
    _planned_collision_metrics,
    _stance_semantic_count,
    _stance_semantic_ratio,
    _obstacle_risk_scale,
)
from extension.batch_mpc_planner.losses.terrain_clearance import _semantic_id_mask, _terrain_grid_world_xy, obstacle_risk_scales
from extension.batch_mpc_planner.terrain import height_at, semantic_at

out = Path('tmp/t302_mpc_metric_tuning/large_high_after_y_axis_diag.jsonl')
out.write_text('', encoding='utf-8')

variants = [
    ('default', {}),
    ('diag_on', {'diagnostics': True}),
    ('semantic_strong', {'stance_weight': 8.0, 'contact_weight': 40.0, 'soft_field': 4.0, 'soft_worst': 16.0}),
    ('semantic_stronger', {'stance_weight': 16.0, 'contact_weight': 80.0, 'large_weight': 120.0, 'soft_field': 8.0, 'soft_worst': 24.0}),
    ('risk_wide', {'yaw_radius': 0.85, 'linear_width': 0.55, 'linear_dist': 1.2}),
    ('risk_wide_semantic_strong', {'yaw_radius': 0.85, 'linear_width': 0.55, 'linear_dist': 1.2, 'stance_weight': 12.0, 'contact_weight': 60.0, 'large_weight': 100.0, 'soft_field': 6.0, 'soft_worst': 20.0}),
]

runtime = viewer_diag.make_real_runtime_fixture(num_envs=2, planner_backend='mpc', device='cuda:0', warmup_steps=6)
try:
    base_cfg = copy.deepcopy(runtime.mpc_planner_cfg)
    for variant, knobs in variants:
        runtime.mpc_planner_cfg = copy.deepcopy(base_cfg)
        cfg = runtime.mpc_planner_cfg
        if knobs.get('diagnostics'):
            cfg.diagnostics.enabled = True
        if 'stance_weight' in knobs:
            cfg.losses.stance_semantic.weight = float(knobs['stance_weight'])
            cfg.losses.touchdown_semantic.weight = float(knobs['stance_weight'])
        if 'contact_weight' in knobs:
            cfg.losses.semantic_contact_avoid.weight = float(knobs['contact_weight'])
        if 'large_weight' in knobs:
            cfg.losses.stance_semantic.large_weight = float(knobs['large_weight'])
            cfg.losses.touchdown_semantic.large_weight = float(knobs['large_weight'])
            cfg.losses.semantic_contact_avoid.large_weight = float(knobs['large_weight'])
        if 'soft_field' in knobs:
            cfg.losses.semantic_contact_avoid.soft_field_weight = float(knobs['soft_field'])
        if 'soft_worst' in knobs:
            cfg.losses.semantic_contact_avoid.soft_worst_field_weight = float(knobs['soft_worst'])
        if 'yaw_radius' in knobs:
            cfg.losses.obstacle_risk.yaw_swept_radius_m = float(knobs['yaw_radius'])
        if 'linear_width' in knobs:
            cfg.losses.obstacle_risk.linear_corridor_width_m = float(knobs['linear_width'])
        if 'linear_dist' in knobs:
            cfg.losses.obstacle_risk.linear_forward_distance_m = float(knobs['linear_dist'])
        for semantic_class, command_name, height in [('large','yaw_left',None), ('large','forward',None), ('small','forward',0.46)]:
            # high small height needs a separate fixture; skip in base runtime and mark later outside.
            if height is not None:
                continue
            plan = runtime.plan_case_near_s4_anchor_command_relative(
                semantic_class,
                command_name=command_name,
                longitudinal_offset_m=-0.15 if command_name.startswith('yaw') else -0.35,
                lateral_offset_m=0.0,
                z_clearance=0.65,
            )
            terrain = runtime._single_env_terrain()
            root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
            anchor = runtime.s4_semantic_course_anchor(semantic_class)
            obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=root.device)
            crossed, min_lat = _crosses_obstacle_along_command(root, obstacle_xy, plan.command)
            breakdown = getattr(plan.result, 'loss_breakdown', None)
            row = {
                'variant': variant,
                'semantic_class': semantic_class,
                'command': command_name,
                'stance_count': _stance_semantic_count(plan.result, terrain),
                'stance_ratio': _stance_semantic_ratio(plan.result, terrain),
                'linear_scale': _obstacle_risk_scale(plan.result, 'obstacle_risk_linear_scale'),
                'yaw_scale': _obstacle_risk_scale(plan.result, 'obstacle_risk_yaw_scale'),
                'crossed': bool(crossed),
                'min_lat': float(min_lat),
                'min_root_dist': _min_root_distance_to_obstacle(root, obstacle_xy),
                'anchor_diameter': float(anchor.target_diameter),
                'has_loss_breakdown': breakdown is not None,
            }
            if breakdown:
                for k in ('obstacle_risk_linear_trigger_count','obstacle_risk_yaw_trigger_count','stance_semantic','semantic_contact_avoid','touchdown_semantic','semantic_obstacle','tracking'):
                    if k in breakdown:
                        row[k] = float(torch.as_tensor(breakdown[k]).reshape(-1)[0].item())
            metrics = _planned_collision_metrics(plan.result, terrain, runtime._viewer)
            row.update({f'metric_{k}': float(v) for k, v in metrics.items()})
            out.open('a', encoding='utf-8').write(json.dumps(row, sort_keys=True)+'\n')
finally:
    runtime.close()

# high small needs its own runtime because object height is event-time configured.
runtime = viewer_diag.make_real_runtime_fixture(num_envs=2, planner_backend='mpc', device='cuda:0', warmup_steps=6, semantic_small_height_m=0.46)
try:
    base_cfg = copy.deepcopy(runtime.mpc_planner_cfg)
    for variant, knobs in variants:
        runtime.mpc_planner_cfg = copy.deepcopy(base_cfg)
        cfg = runtime.mpc_planner_cfg
        if knobs.get('diagnostics'):
            cfg.diagnostics.enabled = True
        if 'stance_weight' in knobs:
            cfg.losses.stance_semantic.weight = float(knobs['stance_weight'])
            cfg.losses.touchdown_semantic.weight = float(knobs['stance_weight'])
        if 'contact_weight' in knobs:
            cfg.losses.semantic_contact_avoid.weight = float(knobs['contact_weight'])
        if 'large_weight' in knobs:
            cfg.losses.stance_semantic.large_weight = float(knobs['large_weight'])
            cfg.losses.touchdown_semantic.large_weight = float(knobs['large_weight'])
            cfg.losses.semantic_contact_avoid.large_weight = float(knobs['large_weight'])
        if 'soft_field' in knobs:
            cfg.losses.semantic_contact_avoid.soft_field_weight = float(knobs['soft_field'])
        if 'soft_worst' in knobs:
            cfg.losses.semantic_contact_avoid.soft_worst_field_weight = float(knobs['soft_worst'])
        if 'yaw_radius' in knobs:
            cfg.losses.obstacle_risk.yaw_swept_radius_m = float(knobs['yaw_radius'])
        if 'linear_width' in knobs:
            cfg.losses.obstacle_risk.linear_corridor_width_m = float(knobs['linear_width'])
        if 'linear_dist' in knobs:
            cfg.losses.obstacle_risk.linear_forward_distance_m = float(knobs['linear_dist'])
        plan = runtime.plan_case_near_s4_anchor_command_relative('small', command_name='forward', longitudinal_offset_m=-0.35, lateral_offset_m=0.0, z_clearance=0.65)
        terrain = runtime._single_env_terrain()
        root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
        anchor = runtime.s4_semantic_course_anchor('small')
        obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=root.device)
        crossed, min_lat = _crosses_obstacle_along_command(root, obstacle_xy, plan.command)
        row = {
            'variant': variant,
            'semantic_class': 'high_small',
            'command': 'forward',
            'stance_count': _stance_semantic_count(plan.result, terrain),
            'stance_ratio': _stance_semantic_ratio(plan.result, terrain),
            'linear_scale': _obstacle_risk_scale(plan.result, 'obstacle_risk_linear_scale'),
            'yaw_scale': _obstacle_risk_scale(plan.result, 'obstacle_risk_yaw_scale'),
            'crossed': bool(crossed),
            'min_lat': float(min_lat),
            'min_root_dist': _min_root_distance_to_obstacle(root, obstacle_xy),
            'anchor_diameter': float(anchor.target_diameter),
            'has_loss_breakdown': getattr(plan.result, 'loss_breakdown', None) is not None,
        }
        breakdown = getattr(plan.result, 'loss_breakdown', None)
        if breakdown:
            for k in ('obstacle_risk_linear_trigger_count','obstacle_risk_yaw_trigger_count','stance_semantic','semantic_contact_avoid','touchdown_semantic','semantic_obstacle','tracking'):
                if k in breakdown:
                    row[k] = float(torch.as_tensor(breakdown[k]).reshape(-1)[0].item())
        metrics = _planned_collision_metrics(plan.result, terrain, runtime._viewer)
        row.update({f'metric_{k}': float(v) for k, v in metrics.items()})
        out.open('a', encoding='utf-8').write(json.dumps(row, sort_keys=True)+'\n')
finally:
    runtime.close()
