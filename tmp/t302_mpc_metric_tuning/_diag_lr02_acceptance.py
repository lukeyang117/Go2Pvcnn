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
    _crosses_obstacle_along_command,
    _min_root_distance_to_obstacle,
    _obstacle_risk_scale,
    _planned_collision_metrics,
    _stance_semantic_count,
    _stance_semantic_ratio,
)


def _apply_lr02(runtime):
    runtime.mpc_planner_cfg.runtime.lr = 2.0e-2


def _obstacle_row(runtime, *, semantic_class, command_name, height=None):
    _apply_lr02(runtime)
    plan = runtime.plan_case_near_s4_anchor_command_relative(
        semantic_class,
        command_name=command_name,
        longitudinal_offset_m=-0.35,
        lateral_offset_m=0.0,
        z_clearance=0.65,
    )
    terrain = runtime._single_env_terrain()
    root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
    anchor = runtime.s4_semantic_course_anchor(semantic_class)
    obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=root.device)
    crossed, min_lateral = _crosses_obstacle_along_command(root, obstacle_xy, plan.command)
    metrics = _planned_collision_metrics(plan.result, terrain, runtime._viewer)
    return {
        "case": f"{semantic_class}_{command_name}",
        "height": height,
        "crossed": crossed,
        "min_lateral": min_lateral,
        "min_dist": _min_root_distance_to_obstacle(root, obstacle_xy),
        "required_dist": 0.5 * float(anchor.target_diameter) + 0.08,
        "linear_scale": _obstacle_risk_scale(plan.result, "obstacle_risk_linear_scale"),
        "yaw_scale": _obstacle_risk_scale(plan.result, "obstacle_risk_yaw_scale"),
        "stance_count": _stance_semantic_count(plan.result, terrain),
        "stance_ratio": _stance_semantic_ratio(plan.result, terrain),
        "metrics": metrics,
        "summary": plan.summary,
    }


def _cobble_rows():
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        terrain="cobblestone",
        device=os.environ.get("MPC_TEST_DEVICE", "cuda:0"),
        warmup_steps=6,
    )
    commands = (
        "forward",
        "backward",
        "lateral_left",
        "lateral_right",
        "yaw_left",
        "yaw_right",
        "forward_yaw_left",
        "forward_yaw_right",
        "diagonal_forward_left",
        "diagonal_forward_right",
    )
    rows = []
    try:
        _apply_lr02(runtime)
        for command_name in commands:
            plan = runtime.plan_case(command_name)
            terrain = runtime._single_env_terrain()
            rows.append(
                {
                    "case": f"cobble_{command_name}",
                    "metrics": _planned_collision_metrics(plan.result, terrain, runtime._viewer),
                    "summary": plan.summary,
                }
            )
    finally:
        runtime.close()
    return rows


def main():
    rows = []
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        device=os.environ.get("MPC_TEST_DEVICE", "cuda:0"),
        warmup_steps=6,
        semantic_small_height_m=0.16,
    )
    try:
        for command_name in ("forward", "backward", "lateral_left", "lateral_right"):
            rows.append(_obstacle_row(runtime, semantic_class="small", command_name=command_name, height=0.16))
        rows.append(_obstacle_row(runtime, semantic_class="large", command_name="forward"))
    finally:
        runtime.close()

    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        device=os.environ.get("MPC_TEST_DEVICE", "cuda:0"),
        warmup_steps=6,
        semantic_small_height_m=0.46,
    )
    try:
        rows.append(_obstacle_row(runtime, semantic_class="small", command_name="forward", height=0.46))
    finally:
        runtime.close()

    rows.extend(_cobble_rows())
    out_path = REPO / "tmp/t302_mpc_metric_tuning/lr02_acceptance.jsonl"
    out_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
