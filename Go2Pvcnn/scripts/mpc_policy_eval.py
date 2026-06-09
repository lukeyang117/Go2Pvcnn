"""Evaluate a trained policy against MPC reference tracking and semantic collisions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = GO2PVCNN_ROOT / "rsl_rl"
for _path in (GO2PVCNN_ROOT, RSL_RL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Evaluate policy rollout against MPC reference and semantic collisions.")
    parser.add_argument("--mode", choices=["tracking", "small_collision"], required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--terrain-rows", type=str, default="0,3,6,9")
    parser.add_argument("--terrain-cols", type=str, default="0")
    parser.add_argument("--command-mode", choices=["fixed", "random", "sweep"], default="fixed")
    parser.add_argument("--command", type=str, default="0.4 0.0 0.0")
    parser.add_argument("--command-sweep", type=str, default="")
    parser.add_argument("--random-command-interval", type=int, default=100)
    parser.add_argument("--small-count-per-tile", type=int, default=80)
    parser.add_argument("--collision-force-threshold", type=float, default=1.0)
    parser.add_argument(
        "--debug-follow-camera",
        action="store_true",
        help="Log livestream follow-camera viewport/camera diagnostics to stdout and follow_camera_debug.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def validate_eval_args(args: argparse.Namespace) -> None:
    if int(args.num_envs) <= 0:
        raise ValueError("--num-envs must be positive")
    if int(args.num_rounds) <= 0:
        raise ValueError("--num-rounds must be positive")
    if int(args.max_steps) < 0:
        raise ValueError("--max-steps must be non-negative")
    if int(args.max_steps) == 0 and int(getattr(args, "livestream", 0)) not in (1, 2):
        raise ValueError("--max-steps 0 is only valid with --livestream 1 or --livestream 2")
    if float(args.collision_force_threshold) < 0.0:
        raise ValueError("--collision-force-threshold must be non-negative")


def _parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        result.append(int(chunk))
    return result


def parse_command_sweep(value: str) -> list[tuple[float, float, float]]:
    commands: list[tuple[float, float, float]] = []
    for chunk in str(value).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [float(v) for v in chunk.split()]
        if len(parts) != 3:
            raise ValueError("--command-sweep entries must be 'vx vy yaw'")
        commands.append((parts[0], parts[1], parts[2]))
    if not commands:
        raise ValueError("--command-sweep must contain at least one command in sweep mode")
    return commands


def _command_tuple_from_args(args: argparse.Namespace, *, step: int) -> tuple[float, float, float]:
    mode = str(args.command_mode)
    if mode == "fixed":
        parts = [float(v) for v in str(args.command).split()]
        if len(parts) != 3:
            raise ValueError("--command must contain exactly three floats: vx vy yaw")
        return (parts[0], parts[1], parts[2])
    if mode == "sweep":
        commands = parse_command_sweep(args.command_sweep)
        return commands[int(step) % len(commands)]
    if mode == "random":
        interval = max(1, int(args.random_command_interval))
        bucket = int(step) // interval
        candidates = (
            (0.4, 0.0, 0.0),
            (-0.25, 0.0, 0.0),
            (0.0, 0.3, 0.0),
            (0.0, -0.3, 0.0),
            (0.25, 0.0, 0.5),
            (0.25, 0.0, -0.5),
            (0.2, 0.2, 0.0),
            (0.2, -0.2, 0.0),
        )
        return candidates[bucket % len(candidates)]
    raise ValueError(f"Unsupported command mode: {mode}")


def command_for_step(args: argparse.Namespace, *, step: int, env_count: int, device: torch.device) -> torch.Tensor:
    values = _command_tuple_from_args(args, step=step)
    command = torch.tensor(values, dtype=torch.float64, device=device).repeat(int(env_count), 1)
    return torch.round(command * 1_000_000_000_000) / 1_000_000_000_000


def tracking_foot_metrics(actual_foot_pos_w: torch.Tensor, reference_foot_pos_w: torch.Tensor) -> dict[str, object]:
    actual = torch.as_tensor(actual_foot_pos_w, dtype=torch.float32)
    reference = torch.as_tensor(reference_foot_pos_w, dtype=torch.float32, device=actual.device)
    if actual.shape != reference.shape or actual.ndim != 3 or actual.shape[1:] != (4, 3):
        raise ValueError(f"expected foot tensors with shape [N,4,3], got {tuple(actual.shape)} and {tuple(reference.shape)}")
    error = torch.linalg.norm(actual - reference, dim=-1)
    return {
        "foot_tracking_error_mean_m": float(error.mean().item()),
        "foot_tracking_error_p95_m": float(torch.quantile(error.reshape(-1), 0.95).item()),
        "per_leg_foot_error_mean_m": [float(v) for v in error.mean(dim=0).tolist()],
    }


@dataclass
class TrackingRoundAccumulator:
    step_count: int = 0
    valid_step_count: int = 0
    reference_valid_ratio_sum: float = 0.0
    mean_error_sum: float = 0.0
    p95_error_max: float = 0.0
    per_leg_error_sum: torch.Tensor = field(default_factory=lambda: torch.zeros((4,), dtype=torch.float64))

    def update(self, metrics: dict[str, object]) -> None:
        self.step_count += 1
        self.reference_valid_ratio_sum += float(metrics.get("reference_valid_ratio", 0.0) or 0.0)
        if metrics.get("foot_tracking_error_mean_m") is None:
            return
        per_leg = metrics.get("per_leg_foot_error_mean_m")
        if not isinstance(per_leg, list) or len(per_leg) != 4:
            raise ValueError("per_leg_foot_error_mean_m must be a four-element list when tracking metrics are valid")
        self.valid_step_count += 1
        self.mean_error_sum += float(metrics["foot_tracking_error_mean_m"])
        self.p95_error_max = max(self.p95_error_max, float(metrics["foot_tracking_error_p95_m"]))
        self.per_leg_error_sum += torch.tensor([float(v) for v in per_leg], dtype=torch.float64)

    def summary(self) -> dict[str, object]:
        reference_valid_ratio = self.reference_valid_ratio_sum / max(1, self.step_count)
        if self.valid_step_count == 0:
            return {
                "tracking_step_count": int(self.step_count),
                "tracking_valid_step_count": 0,
                "reference_valid_ratio": float(reference_valid_ratio),
                "foot_tracking_error_mean_m": None,
                "foot_tracking_error_p95_m": None,
                "per_leg_foot_error_mean_m": None,
            }
        return {
            "tracking_step_count": int(self.step_count),
            "tracking_valid_step_count": int(self.valid_step_count),
            "reference_valid_ratio": float(reference_valid_ratio),
            "foot_tracking_error_mean_m": float(self.mean_error_sum / self.valid_step_count),
            "foot_tracking_error_p95_m": float(self.p95_error_max),
            "per_leg_foot_error_mean_m": [
                float(v) for v in (self.per_leg_error_sum / self.valid_step_count).tolist()
            ],
        }


@dataclass
class SmallCollisionRoundAccumulator:
    num_envs: int
    threshold: float
    device: torch.device
    collided: torch.Tensor = field(init=False)
    first_step: dict[int, int] = field(default_factory=dict)
    body_names_by_env: dict[int, set[str]] = field(default_factory=dict)
    force_max: float = 0.0

    def __post_init__(self) -> None:
        self.collided = torch.zeros((int(self.num_envs),), dtype=torch.bool, device=self.device)

    def update(self, *, step: int, force_matrix_w: torch.Tensor, body_names: tuple[str, ...] | list[str]) -> None:
        force = torch.as_tensor(force_matrix_w, dtype=torch.float32, device=self.device)
        if force.ndim != 4 or force.shape[0] != int(self.num_envs) or force.shape[-1] != 3:
            raise ValueError(f"force_matrix_w must have shape [N,B,F,3], got {tuple(force.shape)}")
        magnitudes = torch.linalg.norm(force, dim=-1)
        active_by_body = magnitudes > float(self.threshold)
        active_env = active_by_body.any(dim=(1, 2))
        self.force_max = max(self.force_max, float(magnitudes.max().item()) if magnitudes.numel() else 0.0)
        active_ids = torch.nonzero(active_env, as_tuple=False).flatten().tolist()
        for env_id in active_ids:
            env_int = int(env_id)
            if env_int not in self.first_step:
                self.first_step[env_int] = int(step)
            self.collided[env_int] = True
            body_ids = torch.nonzero(active_by_body[env_int].any(dim=1), as_tuple=False).flatten().tolist()
            names = self.body_names_by_env.setdefault(env_int, set())
            for body_id in body_ids:
                if int(body_id) < len(body_names):
                    names.add(str(body_names[int(body_id)]))

    def summary(self) -> dict[str, object]:
        count = int(self.collided.sum().item())
        return {
            "collided_env_count": count,
            "num_envs": int(self.num_envs),
            "small_collision_env_rate_per_round": float(count / max(1, int(self.num_envs))),
            "first_collision_step_by_env": {str(k): int(v) for k, v in sorted(self.first_step.items())},
            "collision_body_names_by_env": {str(k): sorted(v) for k, v in sorted(self.body_names_by_env.items())},
            "round_small_force_max": float(self.force_max),
        }


def make_run_output_dir(base: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    base = Path(base)
    for suffix in range(1000):
        name = stamp if suffix == 0 else f"{stamp}_{suffix:03d}"
        out = base / name
        try:
            out.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return out
    raise FileExistsError(f"could not create unique output directory under {base}")


def _base_env(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _actual_foot_pos_w(env) -> torch.Tensor:
    base = _base_env(env)
    robot = base.scene["robot"]
    foot_ids, _ = robot.find_bodies(".*_foot")
    if len(foot_ids) != 4:
        raise ValueError(f"expected exactly 4 robot foot bodies, got {len(foot_ids)}")
    ids = torch.as_tensor([int(i) for i in foot_ids], dtype=torch.long, device=robot.data.body_pos_w.device)
    foot_pos = robot.data.body_pos_w.index_select(1, ids)
    if foot_pos.ndim != 3 or tuple(foot_pos.shape[1:]) != (4, 3):
        raise ValueError(f"expected actual foot tensor shape [N,4,3], got {tuple(foot_pos.shape)}")
    return foot_pos


def _reference_foot_pos_w_from_cache(env) -> torch.Tensor | None:
    base = _base_env(env)
    cache = getattr(base, "_trajectory_reference_cache", None)
    manager = getattr(base, "_trajectory_manager", None)
    if cache is None or manager is None or not hasattr(manager, "current_frame_ids"):
        return None
    try:
        frame_ids = manager.current_frame_ids()
    except Exception:
        return None
    foot_pos_w = getattr(cache, "foot_pos_w", None)
    if foot_pos_w is None:
        return None
    foot = torch.as_tensor(foot_pos_w)
    idx = torch.as_tensor(frame_ids, dtype=torch.long, device=foot.device)
    env_idx = torch.arange(idx.shape[0], dtype=torch.long, device=foot.device)
    return foot[env_idx, idx]


def _reference_foot_trajectory_w(env) -> torch.Tensor | None:
    base = _base_env(env)
    cache = getattr(base, "_trajectory_reference_cache", None)
    foot_pos_w = None if cache is None else getattr(cache, "foot_pos_w", None)
    if foot_pos_w is None:
        return None
    foot = torch.as_tensor(foot_pos_w)
    if foot.ndim != 4 or tuple(foot.shape[-2:]) != (4, 3):
        return None
    return foot


def _reference_foot_pos_w(env) -> torch.Tensor | None:
    base = _base_env(env)
    manager = getattr(base, "_trajectory_manager", None)
    if manager is not None and hasattr(manager, "current_reference"):
        try:
            reference = manager.current_reference()
        except Exception:
            reference = None
        if isinstance(reference, dict) and reference.get("foot_pos_w") is not None:
            return torch.as_tensor(reference["foot_pos_w"])
    return _reference_foot_pos_w_from_cache(base)


def tracking_metrics_for_env_step(env) -> dict[str, object]:
    reference = _reference_foot_pos_w(env)
    if reference is None:
        return {
            "reference_valid_ratio": 0.0,
            "foot_tracking_error_mean_m": None,
            "foot_tracking_error_p95_m": None,
            "per_leg_foot_error_mean_m": None,
        }
    if reference.ndim != 3 or tuple(reference.shape[1:]) != (4, 3):
        raise ValueError(f"expected reference foot tensor shape [N,4,3], got {tuple(reference.shape)}")
    if not bool(torch.isfinite(reference).all().item()):
        return {
            "reference_valid_ratio": 0.0,
            "foot_tracking_error_mean_m": None,
            "foot_tracking_error_p95_m": None,
            "per_leg_foot_error_mean_m": None,
        }
    actual = _actual_foot_pos_w(env)
    if actual.shape != reference.shape:
        raise ValueError(f"actual/reference foot tensor shape mismatch: {tuple(actual.shape)} vs {tuple(reference.shape)}")
    metrics = tracking_foot_metrics(actual, reference)
    metrics["reference_valid_ratio"] = 1.0
    return metrics


def semantic_small_force_matrix_w(env) -> tuple[torch.Tensor, tuple[str, ...]]:
    base = _base_env(env)
    sensor = base.scene.sensors["semantic_contact_small"]
    matrix = torch.as_tensor(sensor.data.force_matrix_w)
    body_names = tuple(getattr(sensor, "body_names", ()) or getattr(sensor.data, "body_names", ()) or ())
    if not body_names:
        body_names = tuple(f"body_{idx}" for idx in range(int(matrix.shape[1])))
    return matrix, body_names


def aggregate_small_collision_rounds(rounds: list[dict[str, object]]) -> dict[str, object]:
    total_collided = sum(int(row.get("collided_env_count", 0) or 0) for row in rounds)
    total_env_rounds = sum(int(row.get("num_envs", 0) or 0) for row in rounds)
    return {
        "round_count": len(rounds),
        "total_collided_envs": int(total_collided),
        "total_env_rounds": int(total_env_rounds),
        "aggregate_small_collision_env_rate": float(total_collided / max(1, total_env_rounds)),
    }


def write_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def build_eval_env_cfg(args: argparse.Namespace):
    if str(args.mode) == "tracking":
        from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
            TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg,
        )

        env_cfg = TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg()
    elif str(args.mode) == "small_collision":
        from extension.semantic_curriculum import SemanticObstacleCount
        from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
            TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg,
        )

        env_cfg = TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg()
        small_count = int(args.small_count_per_tile)
        env_cfg.small_collision_eval_small_count_per_tile = small_count
        env_cfg.small_collision_eval_large_count_per_tile = 0
        dense_flat = SemanticObstacleCount(small=small_count, large=0)
        zero = SemanticObstacleCount(small=0, large=0)
        env_cfg.semantic_obstacle_curriculum.plane_counts = (dense_flat,)
        env_cfg.semantic_obstacle_curriculum.non_plane_counts = (zero,)
        env_cfg.scene.terrain.semantic_obstacle_curriculum = env_cfg.semantic_obstacle_curriculum
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.sim.device = str(args.device)
    terrain_rows = _parse_int_list(str(args.terrain_rows))
    terrain_cols = _parse_int_list(str(args.terrain_cols))
    tg = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if tg is not None:
        if str(args.mode) == "small_collision":
            sub_terrains = getattr(tg, "sub_terrains", None)
            if not isinstance(sub_terrains, dict) or "flat" not in sub_terrains:
                raise ValueError("small_collision mode requires a terrain generator with a 'flat' sub-terrain")
            tg.sub_terrains = {"flat": sub_terrains["flat"]}
        if terrain_rows:
            tg.num_rows = len(terrain_rows)
            tg.curriculum = False
        if terrain_cols:
            tg.num_cols = len(terrain_cols)
    return env_cfg


def checkpoint_path(args: argparse.Namespace) -> Path:
    path = (
        GO2PVCNN_ROOT
        / "logs"
        / "rsl_rl"
        / "teacher_elevation_trajectory_mpc_semantic"
        / str(args.run_dir)
        / str(args.checkpoint)
    )
    if path.exists():
        return path
    fallback = (
        GO2PVCNN_ROOT.parent
        / "logs"
        / "rsl_rl"
        / "teacher_elevation_trajectory_mpc_semantic"
        / str(args.run_dir)
        / str(args.checkpoint)
    )
    if fallback.exists():
        print(f"[mpc_policy_eval] checkpoint fallback: {path} missing; using {fallback}", flush=True)
        return fallback
    raise FileNotFoundError(f"Checkpoint not found: {path} (fallback also missing: {fallback})")


def _make_eval_env_wrapper(env, *, gym_module, vec_env_cls, clip_actions: float | None = None):
    class SimpleRslRlEnvWrapper(vec_env_cls):
        """RSL-RL wrapper matching the active semantic MPC policy observation contract."""

        def __init__(self, env, clip_actions: float | None = None):
            self.env = env
            self.clip_actions = clip_actions
            self.num_envs = env.num_envs
            self.device = env.device
            self.max_episode_length = env.max_episode_length

            if hasattr(env, "action_manager"):
                self.num_actions = env.action_manager.total_action_dim
            else:
                self.num_actions = gym_module.spaces.flatdim(env.single_action_space)

            if clip_actions is not None:
                self.env.action_space = gym_module.spaces.Box(
                    low=-clip_actions,
                    high=clip_actions,
                    shape=(self.num_actions,),
                    dtype=env.action_space.dtype,
                )

            self._initial_observations = None

        @property
        def unwrapped(self):
            return self.env.unwrapped

        @property
        def cfg(self):
            return self.env.unwrapped.cfg

        @property
        def episode_length_buf(self):
            return self.env.unwrapped.episode_length_buf

        @episode_length_buf.setter
        def episode_length_buf(self, value):
            self.env.unwrapped.episode_length_buf = value

        @property
        def observation_space(self):
            return self.env.observation_space

        @property
        def action_space(self):
            return self.env.action_space

        def _flatten_group(self, obs_dict, group_names: list[str]) -> torch.Tensor:
            values = []
            for name in group_names:
                value = obs_dict[name]
                values.append(value.reshape(value.shape[0], -1))
            return torch.cat(values, dim=-1)

        def _format_observations(self, obs_dict) -> tuple[torch.Tensor, dict]:
            policy_obs = self._flatten_group(obs_dict, ["policy_elevation_semantic_map", "policy_state"])
            critic_obs = self._flatten_group(obs_dict, ["critic_elevation_semantic_map", "critic_state"])
            return policy_obs, {"observations": {"critic": critic_obs}}

        def get_observations(self):
            obs_dict = self.env.unwrapped.observation_manager.compute()
            return self._format_observations(obs_dict)

        def reset(self):
            obs_dict, _ = self.env.reset()
            return self._format_observations(obs_dict)

        def step(self, actions):
            if self.clip_actions is not None:
                actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
            obs_dict, rewards, dones, truncated, extras = self.env.step(actions)
            dones = dones | truncated
            obs, obs_extras = self._format_observations(obs_dict)
            extras.update(obs_extras)
            return obs, rewards, dones, extras

    return SimpleRslRlEnvWrapper(env, clip_actions=clip_actions)


def _attach_reference_manager_if_enabled(env, env_cfg) -> None:
    if not getattr(env_cfg, "planner_owned_reference_cache", False):
        return
    from extension.trajectory_manager_factory import attach_trajectory_manager_if_enabled

    manager = attach_trajectory_manager_if_enabled(
        env,
        env_cfg,
        experiment_name="teacher_elevation_trajectory_mpc_semantic",
        device=getattr(env, "device", env_cfg.sim.device),
    )
    if manager is not None:
        print(
            f"[mpc_policy_eval] attached {getattr(manager, 'planner_backend', 'mpc')} trajectory manager",
            flush=True,
        )


def sync_command_to_policy(env, command: torch.Tensor) -> bool:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    command_manager = getattr(base, "command_manager", None)
    if command_manager is None:
        return False
    for name in ("base_velocity", "base_velocity_command", "velocity_command"):
        term = None
        try:
            term = command_manager.get_term(name)
        except Exception:
            term = None
        if term is None or not hasattr(term, "command"):
            continue
        term.command[:] = command.to(device=term.command.device, dtype=term.command.dtype)
        return True
    return False


def sync_command_to_mpc(env, command: torch.Tensor) -> None:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    manager = getattr(base, "_trajectory_manager", None)
    if manager is None:
        return
    # MPC reads commands from the IsaacLab command manager during refresh_from_env().
    # Marking command changes invalidates the old reference-mask without creating
    # a second source of truth for command state.
    if hasattr(manager, "mark_command_changed"):
        manager.mark_command_changed()


def apply_command_to_env(env, command: torch.Tensor) -> None:
    if sync_command_to_policy(env, command):
        sync_command_to_mpc(env, command)


def _policy_command_body_from_env(env) -> torch.Tensor | None:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    command_manager = getattr(base, "command_manager", None)
    if command_manager is None:
        return None
    for name in ("base_velocity", "base_velocity_command", "velocity_command"):
        try:
            command = command_manager.get_command(name)
        except Exception:
            command = None
        if command is not None:
            return torch.as_tensor(command)
    return None


def _mpc_input_command_body_from_env(env) -> torch.Tensor | None:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    manager = getattr(base, "_trajectory_manager", None)
    if manager is None or not hasattr(manager, "_commands_from_env"):
        return None
    try:
        return torch.as_tensor(manager._commands_from_env(base))
    except Exception:
        return None


def command_body_source_diagnostics(env, requested_command_body: torch.Tensor) -> dict[str, object]:
    requested = torch.as_tensor(requested_command_body).detach().to(dtype=torch.float32).cpu()
    policy_command = _policy_command_body_from_env(env)
    mpc_command = _mpc_input_command_body_from_env(env)
    policy_cpu = None if policy_command is None else torch.as_tensor(policy_command).detach().to(dtype=torch.float32).cpu()
    mpc_cpu = None if mpc_command is None else torch.as_tensor(mpc_command).detach().to(dtype=torch.float32).cpu()
    errors = []
    if policy_cpu is not None and policy_cpu.shape == requested.shape:
        errors.append(torch.max(torch.abs(policy_cpu - requested)))
    if mpc_cpu is not None and mpc_cpu.shape == requested.shape:
        errors.append(torch.max(torch.abs(mpc_cpu - requested)))
    max_error = None if not errors else float(torch.stack(errors).max().item())
    return {
        "requested_command_body": requested.tolist(),
        "policy_command_body": None if policy_cpu is None else policy_cpu.tolist(),
        "mpc_input_command_body": None if mpc_cpu is None else mpc_cpu.tolist(),
        "command_body_match_max_abs_error": max_error,
    }


def _semantic_nonzero_count(env) -> int | None:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    sensor = None
    try:
        sensor = base.scene.sensors.get("height_scanner")
    except Exception:
        sensor = None
    semantic_map = None if sensor is None else getattr(getattr(sensor, "data", None), "semantic_map", None)
    if semantic_map is None:
        return None
    semantic = torch.as_tensor(semantic_map)
    return int(torch.count_nonzero(semantic).item())


def planned_direction_metrics_from_reference_cache(env, requested_command_body: torch.Tensor) -> dict[str, object]:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    cache = getattr(base, "_trajectory_reference_cache", None)
    root_pos = None if cache is None else getattr(cache, "root_pos_w", None)
    root_quat = None if cache is None else getattr(cache, "root_quat_w", None)
    foot_pos = None if cache is None else getattr(cache, "foot_pos_w", None)
    out: dict[str, object] = {
        "planned_root_direction_cosine": None,
        "planned_root_lateral_ratio": None,
        "planned_per_leg_direction_cosine_xy": None,
        "planned_per_leg_lateral_ratio_xy": None,
        "planned_insufficient_motion": None,
        "planned_insufficient_leg_motion": None,
        "semantic_nonzero_count": _semantic_nonzero_count(base),
    }
    if root_pos is None or root_quat is None or foot_pos is None:
        return out
    root = torch.as_tensor(root_pos, dtype=torch.float32)
    quat = torch.as_tensor(root_quat, dtype=torch.float32)
    feet = torch.as_tensor(foot_pos, dtype=torch.float32)
    if root.ndim != 3 or quat.ndim != 3 or feet.ndim != 4 or root.shape[0] < 1 or root.shape[1] < 2:
        return out
    command = torch.as_tensor(requested_command_body, dtype=torch.float32, device=root.device)
    if command.ndim != 2 or command.shape[0] < 1:
        return out
    cmd_xy = command[0, :2]
    speed = torch.linalg.vector_norm(cmd_xy)
    if float(speed.item()) <= 1.0e-6:
        return out
    yaw = _root_yaw_from_quat_wxyz(quat[0, 0])
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    cmd_dir_body = cmd_xy / speed.clamp_min(1.0e-6)
    cmd_dir_w = torch.stack((cy * cmd_dir_body[0] - sy * cmd_dir_body[1], sy * cmd_dir_body[0] + cy * cmd_dir_body[1]))
    cmd_left_w = torch.stack((-cmd_dir_w[1], cmd_dir_w[0]))

    root_delta = root[0, -1, :2] - root[0, 0, :2]
    root_norm = torch.linalg.vector_norm(root_delta)
    insufficient_root = bool(float(root_norm.item()) < 0.05)
    out["planned_insufficient_motion"] = insufficient_root
    if not insufficient_root:
        out["planned_root_direction_cosine"] = float(((root_delta * cmd_dir_w).sum() / root_norm.clamp_min(1.0e-6)).item())
        out["planned_root_lateral_ratio"] = float(torch.abs((root_delta * cmd_left_w).sum()).div(root_norm.clamp_min(1.0e-6)).item())

    leg_delta = feet[0, -1, :, :2] - feet[0, 0, :, :2]
    leg_norm = torch.linalg.vector_norm(leg_delta, dim=-1)
    leg_insufficient = leg_norm < 0.03
    leg_cos = torch.where(
        leg_insufficient,
        torch.full_like(leg_norm, float("nan")),
        (leg_delta * cmd_dir_w.view(1, 2)).sum(dim=-1) / leg_norm.clamp_min(1.0e-6),
    )
    leg_lat = torch.where(
        leg_insufficient,
        torch.full_like(leg_norm, float("nan")),
        torch.abs((leg_delta * cmd_left_w.view(1, 2)).sum(dim=-1)) / leg_norm.clamp_min(1.0e-6),
    )
    out["planned_per_leg_direction_cosine_xy"] = [None if bool(leg_insufficient[i].item()) else float(leg_cos[i].item()) for i in range(int(leg_norm.numel()))]
    out["planned_per_leg_lateral_ratio_xy"] = [None if bool(leg_insufficient[i].item()) else float(leg_lat[i].item()) for i in range(int(leg_norm.numel()))]
    out["planned_insufficient_leg_motion"] = [bool(v) for v in leg_insufficient.tolist()]
    return out


def _make_sphere_marker_cfg(prim_path: str, *, radius: float, color: tuple[float, float, float]):
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkersCfg

    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "marker": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )


def build_mpc_foot_trajectory_markers():
    from isaaclab.markers import VisualizationMarkers

    leg_colors = (
        (0.1, 0.9, 1.0),
        (1.0, 0.4, 0.1),
        (0.3, 1.0, 0.3),
        (1.0, 0.2, 0.8),
    )
    foot_traj = []
    for leg_idx, color in enumerate(leg_colors):
        foot_traj.append(
            VisualizationMarkers(
                _make_sphere_marker_cfg(
                    f"/Visuals/T302oMpcPolicyEval/foot_traj_{leg_idx}",
                    radius=0.025,
                    color=color,
                )
            )
        )
    return foot_traj


def update_mpc_foot_trajectory_markers(foot_traj, env) -> None:
    reference = _reference_foot_trajectory_w(env)
    if reference is None or len(foot_traj) != 4 or reference.shape[0] < 1:
        return
    for leg_idx in range(4):
        foot_traj[leg_idx].visualize(translations=reference[0, :, leg_idx].to(dtype=torch.float32))


def build_mpc_foot_markers():
    return build_mpc_foot_trajectory_markers()


def update_mpc_foot_markers(markers, env) -> None:
    update_mpc_foot_trajectory_markers(markers, env)


def _root_yaw_from_quat_wxyz(root_quat_w: torch.Tensor) -> torch.Tensor:
    quat = torch.as_tensor(root_quat_w)
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _usd_camera_world_position(camera_path: str | None) -> list[float] | None:
    if not camera_path:
        return None
    try:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        prim = stage.GetPrimAtPath(str(camera_path))
        camera = UsdGeom.Camera(prim) if prim else None
        if not camera:
            return None
        pos = camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).Transform(Gf.Vec3d(0, 0, 0))
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    except Exception as exc:
        return [f"error:{type(exc).__name__}:{exc}"]  # type: ignore[list-item]


def _collect_follow_camera_debug(
    *,
    root_pos: torch.Tensor,
    root_yaw: torch.Tensor,
    camera_position,
    target_position,
    step: int | None,
    global_step: int | None,
) -> dict[str, object]:
    debug: dict[str, object] = {
        "step": None if step is None else int(step),
        "global_step": None if global_step is None else int(global_step),
        "root_pos_w": [float(v) for v in root_pos.detach().cpu().tolist()],
        "root_yaw": float(root_yaw.detach().cpu().item()),
        "requested_camera_position": [float(v) for v in camera_position.tolist()],
        "requested_target_position": [float(v) for v in target_position.tolist()],
        "default_camera_path": "/OmniverseKit_Persp",
        "default_camera_world_position": _usd_camera_world_position("/OmniverseKit_Persp"),
    }
    try:
        import omni.kit.viewport.utility as viewport_utility

        viewport_api, window = viewport_utility.get_active_viewport_and_window()
        active_camera_path = None
        if viewport_api is not None and getattr(viewport_api, "camera_path", None) is not None:
            active_camera_path = viewport_api.camera_path.pathString
        debug.update(
            {
                "active_viewport_present": viewport_api is not None,
                "active_window_title": None if window is None else getattr(window, "title", None),
                "active_viewport_camera_path": active_camera_path,
                "active_viewport_camera_string": viewport_utility.get_active_viewport_camera_string(),
                "default_window_camera_string": viewport_utility.get_viewport_window_camera_string(),
                "viewport_window_camera_string": viewport_utility.get_viewport_window_camera_string("Viewport"),
                "active_camera_world_position": _usd_camera_world_position(active_camera_path),
            }
        )
    except Exception as exc:
        debug["viewport_debug_error"] = f"{type(exc).__name__}: {exc}"
    return debug


def update_follow_camera(
    env,
    *,
    distance: float = 3.2,
    height: float = 1.6,
    debug: bool = False,
    debug_path: Path | None = None,
    step: int | None = None,
    global_step: int | None = None,
) -> None:
    base = _base_env(env)
    robot = base.scene["robot"]
    root_pos = torch.as_tensor(robot.data.root_pos_w[0])
    root_yaw = _root_yaw_from_quat_wxyz(torch.as_tensor(robot.data.root_quat_w[0]))
    yaw_val = float(root_yaw.item())
    camera_offset = torch.tensor(
        [-float(distance) * math.cos(yaw_val), -float(distance) * math.sin(yaw_val), float(height)],
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    camera_position = (root_pos + camera_offset).detach().cpu().numpy()
    target_position = (
        root_pos + torch.tensor([0.0, 0.0, 0.35], dtype=root_pos.dtype, device=root_pos.device)
    ).detach().cpu().numpy()
    base.sim.set_camera_view(camera_position, target_position)
    base.sim.render()
    if debug:
        row = _collect_follow_camera_debug(
            root_pos=root_pos,
            root_yaw=root_yaw,
            camera_position=camera_position,
            target_position=target_position,
            step=step,
            global_step=global_step,
        )
        print("[mpc_policy_eval][follow_camera_debug] " + json.dumps(row, sort_keys=True), flush=True)
        if debug_path is not None:
            write_jsonl(debug_path, row)


def _close_env(env) -> None:
    try:
        env.close()
    except Exception:
        pass


def run_eval(args: argparse.Namespace) -> int:
    validate_eval_args(args)
    livestream_enabled = int(getattr(args, "livestream", -1)) in (1, 2)
    out_dir = make_run_output_dir(args.output_dir)
    config_path = out_dir / "config.json"
    metrics_path = out_dir / "metrics.jsonl"
    rounds_path = out_dir / "rounds.jsonl"
    summary_path = out_dir / "summary.json"
    follow_camera_debug_path = out_dir / "follow_camera_debug.jsonl"
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    metrics_path.touch()
    rounds_path.touch()
    if bool(getattr(args, "debug_follow_camera", False)):
        follow_camera_debug_path.touch()

    if livestream_enabled and not getattr(args, "enable_cameras", False):
        args.enable_cameras = True

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    raw_env = None
    wrapped_env = None
    try:
        import gymnasium as gym

        from agent import get_train_cfg
        import go2_pvcnn.tasks.register_envs  # noqa: F401
        from isaaclab.envs import ManagerBasedRLEnv
        from rsl_rl.env import VecEnv
        from rsl_rl.runners import OnPolicyRunner

        env_cfg = build_eval_env_cfg(args)
        render_mode = "rgb_array" if livestream_enabled else None
        if render_mode is not None:
            env_cfg.sim.enable_cameras = True
        checkpoint = checkpoint_path(args)
        print(f"[mpc_policy_eval] checkpoint={checkpoint}", flush=True)
        print(f"[mpc_policy_eval] output_dir={out_dir}", flush=True)

        raw_env = gym.make(
            "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0",
            cfg=env_cfg,
            render_mode=render_mode,
        )
        assert isinstance(raw_env.unwrapped, ManagerBasedRLEnv)
        base_env = raw_env.unwrapped
        _attach_reference_manager_if_enabled(base_env, env_cfg)

        print("[mpc_policy_eval] creating wrapper", flush=True)
        wrapped_env = _make_eval_env_wrapper(base_env, gym_module=gym, vec_env_cls=VecEnv, clip_actions=100.0)
        print("[mpc_policy_eval] wrapper ready", flush=True)
        train_cfg = get_train_cfg("teacher_elevation_trajectory_mpc_semantic")
        runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=env_cfg.sim.device)
        print("[mpc_policy_eval] runner ready", flush=True)
        runner.load(str(checkpoint), load_optimizer=False)
        print("[mpc_policy_eval] policy loaded", flush=True)
        policy = runner.get_inference_policy(device=wrapped_env.device)

        summaries: list[dict[str, object]] = []
        total_steps = 0
        overall_tracking = TrackingRoundAccumulator() if str(args.mode) == "tracking" else None
        mpc_foot_markers = build_mpc_foot_markers() if livestream_enabled else None
        for round_idx in range(int(args.num_rounds)):
            obs, _ = wrapped_env.reset()
            step_limit = int(args.max_steps)
            round_steps = 0
            round_tracking = TrackingRoundAccumulator() if str(args.mode) == "tracking" else None
            collision_acc = (
                SmallCollisionRoundAccumulator(
                    num_envs=int(args.num_envs),
                    threshold=float(args.collision_force_threshold),
                    device=torch.device(str(args.device)),
                )
                if str(args.mode) == "small_collision"
                else None
            )
            while (step_limit == 0 and simulation_app.is_running()) or round_steps < step_limit:
                command = command_for_step(
                    args,
                    step=round_steps,
                    env_count=int(args.num_envs),
                    device=torch.device(str(args.device)),
                )
                apply_command_to_env(base_env, command)
                command_diagnostics = command_body_source_diagnostics(base_env, command)
                with torch.no_grad():
                    actions = policy(obs)
                obs, rewards, dones, extras = wrapped_env.step(actions)
                planned_direction_metrics = planned_direction_metrics_from_reference_cache(base_env, command)
                reward_mean = float(torch.as_tensor(rewards).float().mean().item())
                done_count = int(torch.as_tensor(dones).bool().sum().item())
                metric_row = {
                    "round": round_idx,
                    "step": round_steps,
                    "global_step": total_steps,
                    "mode": str(args.mode),
                    "reward_mean": reward_mean,
                    "done_count": done_count,
                    "command_body_source_diagnostics": command_diagnostics,
                    "planned_direction_metrics": planned_direction_metrics,
                }
                if round_tracking is not None and overall_tracking is not None:
                    tracking = tracking_metrics_for_env_step(base_env)
                    round_tracking.update(tracking)
                    overall_tracking.update(tracking)
                    metric_row["tracking"] = tracking
                if collision_acc is not None:
                    force_matrix, body_names = semantic_small_force_matrix_w(base_env)
                    collision_acc.update(step=round_steps, force_matrix_w=force_matrix, body_names=body_names)
                if mpc_foot_markers is not None:
                    update_mpc_foot_markers(mpc_foot_markers, base_env)
                if livestream_enabled and int(args.num_envs) == 1:
                    update_follow_camera(
                        base_env,
                        debug=bool(getattr(args, "debug_follow_camera", False)),
                        debug_path=follow_camera_debug_path,
                        step=round_steps,
                        global_step=total_steps,
                    )
                write_jsonl(metrics_path, metric_row)
                round_steps += 1
                total_steps += 1
                print(
                    f"[mpc_policy_eval] round={round_idx} step={round_steps}/{step_limit}",
                    flush=True,
                )
                if step_limit == 0 and round_steps > 0 and int(args.num_rounds) > 1:
                    break
            round_summary = {
                "round": round_idx,
                "mode": str(args.mode),
                "num_envs": int(args.num_envs),
                "max_steps": int(args.max_steps),
                "steps": round_steps,
            }
            if round_steps > 0:
                round_summary["command_body_source_diagnostics"] = command_diagnostics
                round_summary["planned_direction_metrics"] = planned_direction_metrics
            if round_tracking is not None:
                round_summary["tracking"] = round_tracking.summary()
            if collision_acc is not None:
                round_summary.update(collision_acc.summary())
            write_jsonl(rounds_path, round_summary)
            summaries.append(round_summary)
            if step_limit == 0:
                break

        summary = {
            "mode": str(args.mode),
            "round_count": len(summaries),
            "num_envs": int(args.num_envs),
            "total_steps": total_steps,
            "output_dir": str(out_dir),
            "rounds": summaries,
        }
        if summaries and "command_body_source_diagnostics" in summaries[-1]:
            summary["command_body_source_diagnostics"] = summaries[-1]["command_body_source_diagnostics"]
        if summaries and "planned_direction_metrics" in summaries[-1]:
            summary["planned_direction_metrics"] = summaries[-1]["planned_direction_metrics"]
        if overall_tracking is not None:
            summary["tracking"] = overall_tracking.summary()
        if str(args.mode) == "small_collision":
            summary.update(aggregate_small_collision_rounds(summaries))
        write_summary(summary_path, summary)
    except BaseException as exc:
        print(f"[mpc_policy_eval] abort: {type(exc).__name__}: {exc!r}", flush=True)
        raise
    finally:
        if wrapped_env is not None:
            _close_env(wrapped_env.env)
        elif raw_env is not None:
            _close_env(raw_env)
        simulation_app.close()
    return 0


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(run_eval(args))


if __name__ == "__main__":
    main()
