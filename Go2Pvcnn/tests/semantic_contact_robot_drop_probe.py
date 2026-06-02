from __future__ import annotations

import argparse
import faulthandler
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


@dataclass(frozen=True)
class DropTarget:
    case: str
    semantic_class: str
    prim_path: str | None
    xy: tuple[float, float]
    top_z: float
    align_body: str | None = None


def _finite_bool(value: Tensor, fn) -> bool:
    return bool(fn(torch.as_tensor(value)).any().item())


def summarize_semantic_contact_step(
    *,
    step: int,
    case_by_env: tuple[str, ...],
    body_names: tuple[str, ...],
    small_force_matrix_w: Tensor,
    large_force_matrix_w: Tensor,
    reward: Tensor,
    force_threshold: float,
) -> list[dict[str, Any]]:
    small = torch.as_tensor(small_force_matrix_w, dtype=torch.float32)
    large = torch.as_tensor(large_force_matrix_w, dtype=torch.float32, device=small.device)
    rew = torch.as_tensor(reward, dtype=torch.float32, device=small.device).reshape(-1)
    small_norm = torch.linalg.vector_norm(small, dim=-1)
    large_norm = torch.linalg.vector_norm(large, dim=-1)
    small_active = small_norm > float(force_threshold)
    large_active = large_norm > float(force_threshold)
    rows: list[dict[str, Any]] = []
    for env_id, case in enumerate(case_by_env):
        active_body_mask = torch.logical_or(small_active[env_id].any(dim=-1), large_active[env_id].any(dim=-1))
        active_body_names = [
            str(body_names[idx])
            for idx, is_active in enumerate(active_body_mask.detach().cpu().tolist())
            if bool(is_active)
        ]
        small_env = small[env_id]
        large_env = large[env_id]
        joined = torch.cat((small_env.reshape(-1, 3), large_env.reshape(-1, 3)), dim=0)
        rows.append(
            {
                "step": int(step),
                "env_id": int(env_id),
                "case": str(case),
                "reward": float(rew[env_id].item()),
                "small_force_sum": float(small_norm[env_id].sum().item()),
                "large_force_sum": float(large_norm[env_id].sum().item()),
                "small_active_count": int(small_active[env_id].sum().item()),
                "large_active_count": int(large_active[env_id].sum().item()),
                "max_force": float(torch.linalg.vector_norm(joined, dim=-1).max().item()),
                "active_body_names": active_body_names,
                "has_nan": _finite_bool(joined, torch.isnan) or bool(torch.isnan(rew[env_id]).item()),
                "has_inf": _finite_bool(joined, torch.isinf) or bool(torch.isinf(rew[env_id]).item()),
            }
        )
    return rows


def _semantic_reward(env) -> Tensor:
    from extension.mdp.semantic_contact_rewards import semantic_global_contact_collision_reward
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        SEMANTIC_CONTACT_BODY_NAMES,
        SEMANTIC_CONTACT_BODY_WEIGHTS,
    )

    return semantic_global_contact_collision_reward(
        env,
        small_sensor_cfg=SimpleNamespace(name="semantic_contact_small"),
        large_sensor_cfg=SimpleNamespace(name="semantic_contact_large"),
        body_names=SEMANTIC_CONTACT_BODY_NAMES,
        body_weights=SEMANTIC_CONTACT_BODY_WEIGHTS,
        force_threshold=1.0,
        force_scale=50.0,
        force_clip=1.0,
        small_weight=1.0,
        large_weight=2.0,
    )


def _world_bbox(path: str):
    from pxr import Gf, Usd, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if bbox.IsEmpty():
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = transform.ExtractTranslation()
        return (float(pos[0]), float(pos[1]), float(pos[2])), (float(pos[0]), float(pos[1]), float(pos[2]))
    min_pt: Gf.Vec3d = bbox.GetMin()
    max_pt: Gf.Vec3d = bbox.GetMax()
    return (
        (float(min_pt[0]), float(min_pt[1]), float(min_pt[2])),
        (float(max_pt[0]), float(max_pt[1]), float(max_pt[2])),
    )


def _semantic_leaf_paths(root: str) -> list[str]:
    import isaaclab.sim as sim_utils
    from go2_pvcnn.sensor.semantic_contacter.semantic_global_contact_sensor import filter_semantic_leaf_obstacle_paths

    return filter_semantic_leaf_obstacle_paths(sim_utils.find_matching_prim_paths(f"{root}/.*/.*/.*"), root)


def _drop_target_from_root(root: str, semantic_class: str, *, index: int = 0) -> DropTarget:
    paths = _semantic_leaf_paths(root)
    if not paths:
        raise RuntimeError(f"No semantic obstacle paths under {root}")
    path = paths[min(max(int(index), 0), len(paths) - 1)]
    bbox_min, bbox_max = _world_bbox(path)
    xy = ((bbox_min[0] + bbox_max[0]) * 0.5, (bbox_min[1] + bbox_max[1]) * 0.5)
    return DropTarget(
        case=f"{semantic_class}_drop",
        semantic_class=semantic_class,
        prim_path=path,
        xy=xy,
        top_z=max(bbox_min[2], bbox_max[2]),
    )


def _empty_target(env, env_id: int) -> DropTarget:
    origin = torch.as_tensor(env.scene.env_origins[env_id], dtype=torch.float32, device=env.device)
    return DropTarget(
        case="empty",
        semantic_class="empty",
        prim_path=None,
        xy=(float(origin[0].item()), float(origin[1].item())),
        top_z=float(origin[2].item()),
        align_body=None,
    )


def _target_with_body(target: DropTarget, align_body: str) -> DropTarget:
    return DropTarget(
        case=target.case,
        semantic_class=target.semantic_class,
        prim_path=target.prim_path,
        xy=target.xy,
        top_z=target.top_z,
        align_body=align_body,
    )


def _set_robot_drop_state(env, targets: list[DropTarget], *, drop_height: float) -> tuple[str, ...]:
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import SEMANTIC_CONTACT_BODY_NAMES

    del SEMANTIC_CONTACT_BODY_NAMES
    robot = env.scene["robot"]
    env_ids = torch.arange(len(targets), dtype=torch.long, device=env.device)
    root_pos_before = torch.as_tensor(robot.data.root_pos_w[: len(targets)], dtype=torch.float32, device=env.device)
    body_pos_before = torch.as_tensor(robot.data.body_pos_w[: len(targets)], dtype=torch.float32, device=env.device)
    body_names = list(getattr(robot.data, "body_names", getattr(robot, "body_names", ())))
    root_pose = torch.zeros((len(targets), 7), dtype=torch.float32, device=env.device)
    for env_id, target in enumerate(targets):
        xy = torch.tensor(target.xy, dtype=torch.float32, device=env.device)
        if target.align_body is not None:
            if target.align_body not in body_names:
                raise RuntimeError(f"Robot body {target.align_body!r} not found in {body_names!r}.")
            body_id = body_names.index(target.align_body)
            offset_xy = body_pos_before[env_id, body_id, :2] - root_pos_before[env_id, :2]
            xy = xy - offset_xy
        root_pose[env_id, 0] = float(target.xy[0])
        root_pose[env_id, 1] = float(target.xy[1])
        root_pose[env_id, 0] = float(xy[0].item())
        root_pose[env_id, 1] = float(xy[1].item())
        root_pose[env_id, 2] = float(target.top_z + drop_height)
        root_pose[env_id, 3] = 1.0
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros((len(targets), 6), dtype=torch.float32, device=env.device), env_ids=env_ids)
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    env.scene.write_data_to_sim()
    return tuple(target.case for target in targets)


def run_probe(*, num_envs: int, steps: int, drop_height: float, output: Path | None) -> dict[str, Any]:
    import gymnasium as gym
    import go2_pvcnn.tasks  # noqa: F401
    from extension.semantic_course import SEMANTIC_COURSE_LARGE_ROOT, SEMANTIC_COURSE_SMALL_ROOT
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        SEMANTIC_CONTACT_BODY_NAMES,
        TeacherElevationTrajectoryMpcSemanticEnvCfg,
    )

    if int(num_envs) < 6:
        raise ValueError("num_envs must be at least 6 for small/large/empty paired checks.")

    cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size = min(
        int(num_envs),
        int(cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size),
    )
    env = None
    rows: list[dict[str, Any]] = []
    try:
        env = gym.make("Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0", cfg=cfg)
        env.reset()
        root = env.unwrapped
        small_target = _drop_target_from_root(SEMANTIC_COURSE_SMALL_ROOT, "small", index=0)
        large_target = _drop_target_from_root(SEMANTIC_COURSE_LARGE_ROOT, "large", index=0)
        targets = [
            _target_with_body(small_target, "FL_foot"),
            _target_with_body(large_target, "base"),
            _empty_target(root, 2),
            _target_with_body(small_target, "FR_foot"),
            _target_with_body(large_target, "base"),
            _empty_target(root, 5),
        ]
        targets.extend(_empty_target(root, env_id) for env_id in range(6, int(num_envs)))
        case_by_env = _set_robot_drop_state(root, targets, drop_height=drop_height)
        for step in range(int(steps)):
            root.scene.write_data_to_sim()
            for _ in range(int(root.cfg.decimation)):
                root.sim.step(render=False)
                root.scene.update(dt=root.physics_dt)
            reward = _semantic_reward(root)
            small = root.scene.sensors["semantic_contact_small"].data.force_matrix_w
            large = root.scene.sensors["semantic_contact_large"].data.force_matrix_w
            rows.extend(
                summarize_semantic_contact_step(
                    step=step,
                    case_by_env=case_by_env,
                    body_names=SEMANTIC_CONTACT_BODY_NAMES,
                    small_force_matrix_w=small,
                    large_force_matrix_w=large,
                    reward=reward,
                    force_threshold=1.0,
                )
            )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return _summary(rows)
    finally:
        if env is not None:
            env.close()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)

    def _case_hit(case: str, semantic: str) -> bool:
        key = f"{semantic}_active_count"
        return any(int(row[key]) > 0 for row in by_case.get(case, ()))

    def _case_other_hit(case: str, semantic: str) -> bool:
        other = "large" if semantic == "small" else "small"
        key = f"{other}_active_count"
        return any(int(row[key]) > 0 for row in by_case.get(case, ()))

    empty_rows = by_case.get("empty", ())
    return {
        "row_count": len(rows),
        "small_drop_hit_small": _case_hit("small_drop", "small"),
        "small_drop_hit_large": _case_other_hit("small_drop", "small"),
        "large_drop_hit_large": _case_hit("large_drop", "large"),
        "large_drop_hit_small": _case_other_hit("large_drop", "large"),
        "empty_hit_any": any(int(row["small_active_count"]) > 0 or int(row["large_active_count"]) > 0 for row in empty_rows),
        "has_nan": any(bool(row["has_nan"]) for row in rows),
        "has_inf": any(bool(row["has_inf"]) for row in rows),
        "min_reward": min((float(row["reward"]) for row in rows), default=0.0),
        "max_reward": max((float(row["reward"]) for row in rows), default=0.0),
    }


def main() -> None:
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--drop-height", type=float, default=0.45)
    parser.add_argument("--output", type=Path, default=Path("Go2Pvcnn/tests/artifacts/semantic_contact_robot_drop_probe.jsonl"))
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        summary = run_probe(num_envs=args.num_envs, steps=args.steps, drop_height=args.drop_height, output=args.output)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
