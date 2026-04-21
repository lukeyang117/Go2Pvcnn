"""Script to play a trained teacher policy."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Play a trained teacher policy")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during play")
    parser.add_argument("--video_length", type=int, default=2000000, help="Length of recorded video (steps)")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (steps)")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate")
    parser.add_argument("--checkpoint", type=str, default="model_1600.pt", help="Checkpoint file name")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory name")
    parser.add_argument(
        "--experiment",
        type=str,
        default="teacher_semantic",
        choices=[
            "teacher_semantic",
            "teacher_without_semantic",
            "teacher_elevation",
            "teacher_elevation_semantic_map",
            "teacher_elevation_trajectory",
        ],
        help="Experiment/task: teacher_semantic (CNN+state), teacher_without_semantic (state-only), "
        "teacher_elevation (elevation map CNN), teacher_elevation_semantic_map (dual grid CNN), "
        "teacher_elevation_trajectory (high-res elevation + trajectory reward)",
    )
    parser.add_argument("--sample", action="store_true", default=False, help="Sample actions with std instead of using policy")
    parser.add_argument(
        "--use-raw-reference-trajectory",
        action="store_true",
        default=False,
        help="For teacher_elevation_trajectory: fill reference cache from raw go2fp on reset.",
    )
    parser.add_argument(
        "--debug-livestream",
        action="store_true",
        default=False,
        help="Print startup and loop timing diagnostics for WebRTC livestream bottlenecks.",
    )

    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def _prepare_runtime_args(args_cli: argparse.Namespace) -> argparse.Namespace:
    if getattr(args_cli, "livestream", -1) in (1, 2) and not args_cli.enable_cameras:
        args_cli.enable_cameras = True
        print(
            "[INFO][play.py] livestream: enabled AppLauncher --enable_cameras so the simulator "
            "uses a rendering experience (works without X11; WebRTC client on another machine).",
            flush=True,
        )
    return args_cli


def _launch_app(args_cli: argparse.Namespace):
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    return app_launcher, app_launcher.app


def _resolve_render_mode(args_cli: argparse.Namespace) -> str | None:
    if args_cli.video or getattr(args_cli, "livestream", -1) in (1, 2):
        return "rgb_array"
    return None


def _livestream_camera_update_interval(livestream: int) -> int:
    return 4 if livestream in (1, 2) else 1


def _should_update_follow_camera(*, timestep: int, num_envs: int, livestream: int, interval: int) -> bool:
    if num_envs != 1:
        return False
    if livestream in (1, 2):
        return timestep % max(1, interval) == 0
    return True


def _compute_follow_camera_pose(robot_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_direction = np.array([3.0, 0.0, 0.0], dtype=np.float64)
    camera_position = robot_pos - camera_direction + np.array([0.0, 0.0, 1.5], dtype=np.float64)
    return camera_position, robot_pos


def _collect_runtime_debug_snapshot(args_cli: argparse.Namespace, *, argv: list[str] | None = None) -> dict[str, object]:
    return {
        "argv": list(sys.argv if argv is None else argv),
        "env": {
            "LIVESTREAM": os.environ.get("LIVESTREAM"),
            "HEADLESS": os.environ.get("HEADLESS"),
            "ENABLE_CAMERAS": os.environ.get("ENABLE_CAMERAS"),
        },
        "args": {
            "livestream": getattr(args_cli, "livestream", None),
            "headless": getattr(args_cli, "headless", None),
            "enable_cameras": getattr(args_cli, "enable_cameras", None),
            "device": getattr(args_cli, "device", None),
            "debug_livestream": getattr(args_cli, "debug_livestream", None),
        },
    }


def _print_runtime_debug_snapshot(args_cli: argparse.Namespace) -> None:
    snapshot = _collect_runtime_debug_snapshot(args_cli)
    print("[debug-livestream] runtime launch snapshot:", flush=True)
    print(f"[debug-livestream]   argv={snapshot['argv']}", flush=True)
    print(f"[debug-livestream]   env={snapshot['env']}", flush=True)
    print(f"[debug-livestream]   args={snapshot['args']}", flush=True)
    if snapshot["args"]["livestream"] == 0 and snapshot["args"]["headless"]:
        print(
            "[debug-livestream] warning: effective livestream=0 while headless=True; "
            "WebRTC is not actually enabled in this run.",
            flush=True,
        )


@dataclass(slots=True)
class _LivestreamDebug:
    enabled: bool
    startup_marks: list[tuple[str, float]] = field(default_factory=list)
    loop_samples: list[dict[str, float]] = field(default_factory=list)
    _startup_last: float = field(default_factory=perf_counter)

    def mark_startup(self, label: str) -> None:
        if not self.enabled:
            return
        now = perf_counter()
        self.startup_marks.append((label, now - self._startup_last))
        self._startup_last = now

    def add_loop_sample(
        self,
        *,
        policy_s: float,
        env_step_s: float,
        camera_s: float,
        total_s: float,
        timestep: int,
        step_probe: dict[str, float] | None = None,
    ) -> None:
        if not self.enabled:
            return
        sample = {
            "policy_s": policy_s,
            "env_step_s": env_step_s,
            "camera_s": camera_s,
            "total_s": total_s,
            "timestep": float(timestep),
        }
        if step_probe is not None:
            sample.update(step_probe)
        self.loop_samples.append(sample)
        if len(self.loop_samples) in {1, 10, 30}:
            self.print_loop_summary(prefix=f"[debug-livestream][sample={len(self.loop_samples)}]")

    def print_startup_summary(self) -> None:
        if not self.enabled or not self.startup_marks:
            return
        print("[debug-livestream] startup timing summary:", flush=True)
        for label, dt_s in self.startup_marks:
            print(f"[debug-livestream]   {label:<24} {dt_s * 1000.0:8.1f} ms", flush=True)

    def print_loop_summary(self, *, prefix: str = "[debug-livestream]") -> None:
        if not self.enabled or not self.loop_samples:
            return
        count = len(self.loop_samples)
        totals = {"policy_s": 0.0, "env_step_s": 0.0, "camera_s": 0.0, "total_s": 0.0}
        for sample in self.loop_samples:
            for key in totals:
                totals[key] += sample[key]
        mean_total_ms = totals["total_s"] * 1000.0 / count
        fps = 1.0 / (totals["total_s"] / count) if totals["total_s"] > 0.0 else float("inf")
        print(
            f"{prefix} mean step={mean_total_ms:0.2f} ms "
            f"(policy={totals['policy_s'] * 1000.0 / count:0.2f} ms, "
            f"env={totals['env_step_s'] * 1000.0 / count:0.2f} ms, "
            f"camera={totals['camera_s'] * 1000.0 / count:0.2f} ms) "
            f"approx_fps={fps:0.2f}",
            flush=True,
        )
        detail_keys = [
            "action_process_s",
            "action_apply_s",
            "sim_step_s",
            "sim_render_s",
            "scene_update_s",
            "obs_compute_s",
            "reward_compute_s",
            "termination_compute_s",
            "command_compute_s",
        ]
        detail_parts = []
        for key in detail_keys:
            if key in self.loop_samples[0]:
                value_ms = sum(sample.get(key, 0.0) for sample in self.loop_samples) * 1000.0 / count
                detail_parts.append(f"{key.removesuffix('_s')}={value_ms:0.2f} ms")
        if detail_parts:
            print(f"{prefix} env breakdown: " + ", ".join(detail_parts), flush=True)


@dataclass(slots=True)
class _StepProbe:
    enabled: bool
    accumulators: dict[str, float] = field(
        default_factory=lambda: {
            "action_process_s": 0.0,
            "action_apply_s": 0.0,
            "sim_step_s": 0.0,
            "sim_render_s": 0.0,
            "scene_update_s": 0.0,
            "obs_compute_s": 0.0,
            "reward_compute_s": 0.0,
            "termination_compute_s": 0.0,
            "command_compute_s": 0.0,
        }
    )

    def wrap_method(self, obj, attr_name: str, metric_key: str) -> None:
        if not self.enabled or not hasattr(obj, attr_name):
            return
        original = getattr(obj, attr_name)
        if not callable(original):
            return

        def wrapped(*args, **kwargs):
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.accumulators[metric_key] += perf_counter() - start

        setattr(obj, attr_name, wrapped)

    def snapshot_and_reset(self) -> dict[str, float]:
        snapshot = dict(self.accumulators)
        for key in self.accumulators:
            self.accumulators[key] = 0.0
        return snapshot


def _install_env_step_probes(base_env, *, enabled: bool) -> _StepProbe:
    probe = _StepProbe(enabled=enabled)
    if not enabled:
        return probe

    probe.wrap_method(base_env.action_manager, "process_action", "action_process_s")
    probe.wrap_method(base_env.action_manager, "apply_action", "action_apply_s")
    probe.wrap_method(base_env.sim, "step", "sim_step_s")
    probe.wrap_method(base_env.sim, "render", "sim_render_s")
    probe.wrap_method(base_env.scene, "update", "scene_update_s")
    probe.wrap_method(base_env.observation_manager, "compute", "obs_compute_s")
    probe.wrap_method(base_env.reward_manager, "compute", "reward_compute_s")
    probe.wrap_method(base_env.termination_manager, "compute", "termination_compute_s")
    probe.wrap_method(base_env.command_manager, "compute", "command_compute_s")
    return probe


def _make_env_wrapper(env, *, gym_module, vec_env_cls, tensor_dict_cls, clip_actions: float | None = None):
    class SimpleRslRlEnvWrapper(vec_env_cls):
        """Simple wrapper for RSL-RL without PVCNN."""

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

            self.env.reset()

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

        def get_observations(self):
            obs_dict = self.env.unwrapped.observation_manager.compute()
            if isinstance(obs_dict, dict) and not isinstance(obs_dict, tensor_dict_cls):
                return tensor_dict_cls(obs_dict, batch_size=[self.env.unwrapped.num_envs])
            return obs_dict

        def reset(self):
            obs_dict, _ = self.env.reset()
            if isinstance(obs_dict, dict) and not isinstance(obs_dict, tensor_dict_cls):
                obs_dict = tensor_dict_cls(obs_dict, batch_size=[self.env.unwrapped.num_envs])
            return obs_dict, None

        def step(self, actions):
            if self.clip_actions is not None:
                actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

            obs_dict, rewards, dones, truncated, extras = self.env.step(actions)
            dones = dones | truncated

            if isinstance(obs_dict, dict) and not isinstance(obs_dict, tensor_dict_cls):
                obs_dict = tensor_dict_cls(obs_dict, batch_size=[self.env.unwrapped.num_envs])

            return obs_dict, rewards, dones, extras

    return SimpleRslRlEnvWrapper(env, clip_actions=clip_actions)


def _configure_reference_trajectory(env_cfg, *, use_raw_reference_trajectory: bool) -> None:
    if hasattr(env_cfg, "use_batched_reference_trajectory"):
        env_cfg.use_batched_reference_trajectory = True
        if use_raw_reference_trajectory:
            print(
                "[play.py] Warning: --use-raw-reference-trajectory is legacy-only and is ignored "
                "for the batched GPU teacher_elevation_trajectory env.",
                flush=True,
            )
        return

    if hasattr(env_cfg, "use_raw_reference_trajectory"):
        env_cfg.use_raw_reference_trajectory = bool(use_raw_reference_trajectory)


def main() -> int:
    args_cli = _prepare_runtime_args(_parse_args())
    debug = _LivestreamDebug(enabled=bool(args_cli.debug_livestream))
    if args_cli.debug_livestream:
        _print_runtime_debug_snapshot(args_cli)

    _, simulation_app = _launch_app(args_cli)
    debug.mark_startup("app launch")

    import gymnasium as gym

    from agent import get_train_cfg
    from go2_pvcnn.tasks.teacher_elevation_env_cfg import TeacherElevationEnvCfg_PLAY
    from go2_pvcnn.tasks.teacher_elevation_semantic_map_env_cfg import TeacherElevationSemanticMapEnvCfg_PLAY
    from go2_pvcnn.tasks.teacher_elevation_trajectory_env_cfg import TeacherElevationTrajectoryEnvCfg_PLAY
    from go2_pvcnn.tasks.teacher_semantic_env_cfg import TeacherSemanticEnvCfg_PLAY
    from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import TeacherWithoutSemanticEnvCfg_PLAY
    import go2_pvcnn.tasks.register_envs  # noqa: F401
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.dict import print_dict
    from rsl_rl_2_01.env import VecEnv
    from rsl_rl_2_01.runners import OnPolicyRunner
    from tensordict import TensorDict

    debug.mark_startup("python imports")

    experiment_play_map = {
        "teacher_semantic": (TeacherSemanticEnvCfg_PLAY, "Isaac-Teacher-Semantic-Go2-Play-v0"),
        "teacher_without_semantic": (TeacherWithoutSemanticEnvCfg_PLAY, "Isaac-Teacher-Without-Semantic-Go2-Play-v0"),
        "teacher_elevation": (TeacherElevationEnvCfg_PLAY, "Isaac-Teacher-Elevation-Go2-Play-v0"),
        "teacher_elevation_semantic_map": (
            TeacherElevationSemanticMapEnvCfg_PLAY,
            "Isaac-Teacher-Elevation-Semantic-Map-Go2-Play-v0",
        ),
        "teacher_elevation_trajectory": (
            TeacherElevationTrajectoryEnvCfg_PLAY,
            "Isaac-Teacher-Elevation-Trajectory-Go2-Play-v0",
        ),
    }

    experiment_name = args_cli.experiment
    env_cfg_cls, task_id = experiment_play_map[experiment_name]

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    log_dir = os.path.join(log_root_path, args_cli.run_dir)
    checkpoint_path = os.path.join(log_dir, args_cli.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n{'=' * 80}")
    print(f"Playing - {experiment_name}")
    print(f"{'=' * 80}")
    print(f"Task: {task_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Livestream mode: {getattr(args_cli, 'livestream', 0)}")
    print(f"Debug livestream: {args_cli.debug_livestream}")
    print(f"{'=' * 80}\n")

    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if experiment_name == "teacher_elevation_trajectory":
        _configure_reference_trajectory(
            env_cfg,
            use_raw_reference_trajectory=bool(args_cli.use_raw_reference_trajectory),
        )

    render_mode = _resolve_render_mode(args_cli)
    if render_mode is not None:
        env_cfg.sim.enable_cameras = True
    if args_cli.video:
        print(f"[Video] Recording enabled (length={args_cli.video_length})", flush=True)
    debug.mark_startup("env cfg setup")

    print(f"[INFO][play.py] gym.make({task_id!r}) ... (scene build can take several minutes)", flush=True)
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    print("[INFO][play.py] gym.make done.", flush=True)
    debug.mark_startup("gym.make")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{args_cli.checkpoint.split('_')[-1].split('.')[0]}",
        }
        print("[INFO] Recording video during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        debug.mark_startup("video wrapper")

    assert isinstance(env.unwrapped, ManagerBasedRLEnv)
    base_env = env.unwrapped
    step_probe = _install_env_step_probes(base_env, enabled=bool(args_cli.debug_livestream))

    print("\n[Wrapper] Creating RSL-RL environment wrapper...", flush=True)
    wrapped_env = _make_env_wrapper(
        base_env,
        gym_module=gym,
        vec_env_cls=VecEnv,
        tensor_dict_cls=TensorDict,
        clip_actions=100.0,
    )
    debug.mark_startup("wrapper init")

    print("\n[Environment] Created successfully", flush=True)
    print(f"  - Observation space: {wrapped_env.observation_space}", flush=True)
    print(f"  - Action space: {wrapped_env.action_space}", flush=True)
    print(f"  - Device: {wrapped_env.device}", flush=True)
    print(f"  - Render mode: {render_mode}", flush=True)
    print(f"  - Render interval: {base_env.cfg.sim.render_interval}", flush=True)

    train_cfg = get_train_cfg(experiment_name)
    print("\n[Runner] Creating OnPolicyRunner...", flush=True)
    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=env_cfg.sim.device)
    debug.mark_startup("runner init")

    print(f"\n[Checkpoint] Loading model from: {checkpoint_path}", flush=True)
    runner.load(checkpoint_path, load_optimizer=False)
    print("[Policy] Loaded successfully", flush=True)
    debug.mark_startup("checkpoint load")

    if args_cli.sample:
        policy = runner.alg.policy.act
    else:
        policy = runner.get_inference_policy(device=wrapped_env.device)
    print(f"[Policy] Using {'sampling' if args_cli.sample else 'inference'} mode", flush=True)

    obs, _ = wrapped_env.get_observations(), None
    timestep = 0
    camera_interval = _livestream_camera_update_interval(getattr(args_cli, "livestream", 0))
    debug.mark_startup("first observations")
    debug.print_startup_summary()
    if args_cli.debug_livestream:
        print(
            f"[debug-livestream] camera follow interval={camera_interval} "
            f"(livestream={getattr(args_cli, 'livestream', 0)}, num_envs={args_cli.num_envs})",
            flush=True,
        )

    print(f"\n{'=' * 80}")
    print("Starting Play Loop")
    print(f"{'=' * 80}\n")

    try:
        while simulation_app.is_running():
            step_start = perf_counter()
            with torch.inference_mode():
                policy_start = perf_counter()
                actions = policy(obs)
                policy_s = perf_counter() - policy_start

                env_start = perf_counter()
                obs, rewards, dones, extras = wrapped_env.step(actions)
                env_step_s = perf_counter() - env_start

            timestep += 1

            camera_s = 0.0
            if _should_update_follow_camera(
                timestep=timestep,
                num_envs=args_cli.num_envs,
                livestream=getattr(args_cli, "livestream", 0),
                interval=camera_interval,
            ):
                camera_start = perf_counter()
                robot_pos = base_env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                camera_position, target_position = _compute_follow_camera_pose(robot_pos)
                base_env.sim.set_camera_view(camera_position, target_position)
                camera_s = perf_counter() - camera_start

            total_s = perf_counter() - step_start
            debug.add_loop_sample(
                policy_s=policy_s,
                env_step_s=env_step_s,
                camera_s=camera_s,
                total_s=total_s,
                timestep=timestep,
                step_probe=step_probe.snapshot_and_reset() if args_cli.debug_livestream else None,
            )

            if args_cli.video and timestep == args_cli.video_length:
                break

    except KeyboardInterrupt:
        print("\n[Play] Interrupted by user")

    finally:
        wrapped_env.env.close()
        debug.print_loop_summary(prefix="[debug-livestream][final]")
        print(f"\n{'=' * 80}")
        print(f"Play Complete - Timesteps: {timestep}")
        print(f"{'=' * 80}\n")
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
