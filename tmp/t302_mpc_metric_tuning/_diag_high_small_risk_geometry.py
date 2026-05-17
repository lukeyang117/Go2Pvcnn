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
from extension.batch_mpc_planner.losses.terrain_clearance import _nearby_height_for_sparse_semantic, _terrain_grid_world_xy
from extension.batch_mpc_planner.terrain import height_at


def main():
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        device=os.environ.get("MPC_TEST_DEVICE", "cuda:0"),
        warmup_steps=6,
        semantic_small_height_m=0.46,
    )
    try:
        plan = runtime.plan_case_near_s4_anchor_command_relative(
            "small",
            command_name="forward",
            longitudinal_offset_m=-0.35,
            lateral_offset_m=0.0,
            z_clearance=0.65,
        )
        terrain = runtime._single_env_terrain()
        root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
        quat = torch.as_tensor(plan.result.root_quat_w, dtype=torch.float32, device=root.device)
        rpy = runtime._viewer._quat_wxyz_to_rpy(quat.reshape(-1, 4)).reshape_as(root)
        root0 = root[:, 0]
        height = torch.as_tensor(terrain.height_map, dtype=root.dtype, device=root.device)
        sem = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=root.device)
        grid_xy = _terrain_grid_world_xy(terrain, dtype=root.dtype, device=root.device)
        nearby_z = _nearby_height_for_sparse_semantic(terrain, height, dtype=root.dtype, device=root.device)
        ground_z = height_at(terrain, root0[:, None, :2]).reshape(-1).to(dtype=root.dtype, device=root.device)
        mask = sem.reshape(1, -1) == 1
        rel = grid_xy - root0[:, None, :2]
        forward = rel[..., 0]
        lateral = rel[..., 1]
        yaw = rpy[:, 0, 2]
        cy = torch.cos(yaw).view(1, 1)
        sy = torch.sin(yaw).view(1, 1)
        body_forward = cy * rel[..., 0] + sy * rel[..., 1]
        body_lateral = -sy * rel[..., 0] + cy * rel[..., 1]
        selected = mask[0]
        anchor = runtime.s4_semantic_course_anchor("small")
        row = {
            "anchor_xy": list(anchor.world_xy),
            "root0_xy": root0[0, :2].detach().cpu().tolist(),
            "root0_yaw": float(yaw[0].item()),
            "root_ground_z": float(ground_z[0].item()),
            "small_count": int(selected.sum().item()),
            "small_forward_min": float(forward[0, selected].min().item()) if bool(selected.any().item()) else None,
            "small_forward_max": float(forward[0, selected].max().item()) if bool(selected.any().item()) else None,
            "small_lateral_min": float(lateral[0, selected].min().item()) if bool(selected.any().item()) else None,
            "small_lateral_max": float(lateral[0, selected].max().item()) if bool(selected.any().item()) else None,
            "small_body_forward_min": float(body_forward[0, selected].min().item()) if bool(selected.any().item()) else None,
            "small_body_forward_max": float(body_forward[0, selected].max().item()) if bool(selected.any().item()) else None,
            "small_body_lateral_min": float(body_lateral[0, selected].min().item()) if bool(selected.any().item()) else None,
            "small_body_lateral_max": float(body_lateral[0, selected].max().item()) if bool(selected.any().item()) else None,
            "small_height_delta_max": float((height.reshape(1, -1)[0, selected] - ground_z[0]).max().item()) if bool(selected.any().item()) else None,
            "small_nearby_delta_max": float((nearby_z[0, selected] - ground_z[0]).max().item()) if bool(selected.any().item()) else None,
            "risk_linear_scale": float(torch.as_tensor(plan.result.loss_breakdown.get("obstacle_risk_linear_scale", torch.tensor([1.0]))).reshape(-1)[0].item()),
            "risk_count": float(torch.as_tensor(plan.result.loss_breakdown.get("obstacle_risk_linear_trigger_count", torch.tensor([0.0]))).reshape(-1)[0].item()),
            "high_avoid": float(torch.as_tensor(plan.result.loss_breakdown.get("high_obstacle_avoidance", torch.tensor([0.0]))).reshape(-1)[0].item()),
            "summary": plan.summary,
        }
        out = REPO / "tmp/t302_mpc_metric_tuning/high_small_risk_geometry.json"
        out.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(row, indent=2, sort_keys=True))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
