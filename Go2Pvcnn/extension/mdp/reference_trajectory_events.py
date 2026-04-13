"""Periodic + startup replanning for raw go2fp reference trajectories (parallel per env)."""

from __future__ import annotations

import atexit
import math
import multiprocessing as mp
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from extension.planner.adapters.isaac_heightmap import LocalGridTerrain
from extension.planner.runtime.raw_go2fp_bridge import ensure_kinematic_footsteps_on_syspath
from extension.planner.runtime.reference_cache import (
    ReferenceTrajectoryCache,
    expand_reference_cache_to_num_envs,
)
from extension.planner.runtime.reference_generator import ReferenceGenerator, ReferenceGeneratorConfig

_GO2PVCNN_ROOT = str(Path(__file__).resolve().parents[2])


def _ensure_spawn_worker_pythonpath() -> None:
    """Prepend ``Go2Pvcnn`` to ``PYTHONPATH`` so spawn children can ``import extension``."""
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if _GO2PVCNN_ROOT in parts:
        return
    os.environ["PYTHONPATH"] = os.pathsep.join([_GO2PVCNN_ROOT, *parts]) if parts else _GO2PVCNN_ROOT


def _normalize_env_ids(env, env_ids: torch.Tensor | Sequence[int] | slice | None) -> list[int]:
    if env_ids is None:
        return list(range(env.num_envs))
    if isinstance(env_ids, slice):
        if env_ids == slice(None):
            return list(range(env.num_envs))
        raise TypeError(f"Unsupported env_ids slice: {env_ids!r}")
    if isinstance(env_ids, torch.Tensor):
        return [int(x) for x in env_ids.detach().cpu().flatten().tolist()]
    return [int(x) for x in env_ids]


def _resolve_foot_body_ids(env, asset_cfg: SceneEntityCfg) -> list[int]:
    foot_cfg = SceneEntityCfg(
        asset_cfg.name,
        body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    )
    foot_cfg.resolve(env.scene)
    if foot_cfg.body_ids is None or len(foot_cfg.body_ids) != 4:
        raise RuntimeError(f"Could not resolve four foot bodies, got {foot_cfg.body_ids!r}")
    return [int(i) for i in foot_cfg.body_ids]


@dataclass(frozen=True, slots=True)
class _TrajJob:
    """Picklable / thread-safe payload for one env trajectory (numpy only)."""

    env_index: int
    hits_grid: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_angles: np.ndarray
    foot_pos: np.ndarray
    vx: float
    vy: float
    yaw_rate: float
    horizon: int
    dt: float
    size_xy: tuple[float, float]


def _worker_bootstrap_go2pvcnn_path() -> None:
    """Ensure ``Go2Pvcnn`` root is on ``sys.path`` (needed for spawn workers)."""
    import sys

    go2_root = Path(__file__).resolve().parents[2]
    s = str(go2_root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _worker_traj(job: _TrajJob) -> tuple:
    """Run raw ``generate_trajectory`` in a worker process or thread (CPU / numpy).

    Returns numpy arrays so spawn pickling stays lightweight and CUDA-free.
    """
    _worker_bootstrap_go2pvcnn_path()
    from extension.planner.runtime.raw_go2fp_bridge import (
        ensure_kinematic_footsteps_on_syspath,
        trajectory_result_to_reference_cache,
    )

    ensure_kinematic_footsteps_on_syspath()
    from scripts.go2fp.config import TrajectoryConfig
    from scripts.go2fp.trajectory import generate_trajectory
    from scripts.go2fp.types import Command, RobotState

    terrain = LocalGridTerrain.from_world_ray_hits(
        job.hits_grid,
        root_pos_w=job.root_pos,
        root_quat_w=job.root_quat,
        size_xy=job.size_xy,
    )
    state = RobotState(
        root_pos=job.root_pos,
        root_quat=job.root_quat,
        joint_angles=job.joint_angles,
        foot_pos=job.foot_pos,
    )
    cmd = Command(vx=job.vx, vy=job.vy, yaw_rate=job.yaw_rate)
    tr = generate_trajectory(terrain, state, cmd, job.horizon, job.dt, TrajectoryConfig())
    cache = trajectory_result_to_reference_cache(tr)
    return (
        job.env_index,
        cache.root_pos_w.detach().cpu().numpy(),
        cache.root_quat_w.detach().cpu().numpy(),
        cache.joint_angles.detach().cpu().numpy(),
        cache.foot_pos_root.detach().cpu().numpy(),
        cache.contact_state.detach().cpu().numpy(),
        cache.planned_touchdown_w.detach().cpu().numpy(),
        cache.phase_index.detach().cpu().numpy(),
        cache.valid_mask.detach().cpu().numpy(),
    )


def _build_traj_job(
    env_idx: int,
    sensor: RayCaster,
    asset: Articulation,
    foot_body_ids: list[int],
    size_xy: tuple[float, float],
    env,
    horizon: int,
    dt: float,
) -> _TrajJob:
    e = int(env_idx)
    hits = sensor.data.ray_hits_w[e].detach().cpu().numpy()
    if hits.ndim != 2 or hits.shape[-1] != 3:
        raise RuntimeError(f"Unexpected ray_hits_w shape {hits.shape}")
    root_pos = asset.data.root_pos_w[e].detach().cpu().numpy().reshape(3)
    root_quat = asset.data.root_quat_w[e].detach().cpu().numpy().reshape(4)
    joint_angles = asset.data.joint_pos[e].detach().cpu().numpy().astype(np.float64).reshape(-1)
    if joint_angles.size != 12:
        raise RuntimeError(f"Expected 12 joint positions, got {joint_angles.size}")
    foot_pos = asset.data.body_pos_w[e, foot_body_ids].detach().cpu().numpy().astype(np.float64).reshape(4, 3)
    cmd_t = env.command_manager.get_command("base_velocity")
    if cmd_t.ndim != 2 or cmd_t.shape[-1] < 3:
        raise RuntimeError(f"Unexpected base_velocity command shape {tuple(cmd_t.shape)}")
    vx = float(cmd_t[e, 0].item())
    vy = float(cmd_t[e, 1].item())
    yaw_rate = float(cmd_t[e, 2].item())
    side = int(round(math.sqrt(hits.shape[0])))
    if side * side != hits.shape[0]:
        raise RuntimeError(f"ray_hits_w length {hits.shape[0]} is not a square grid")
    hits_grid = hits.reshape(side, side, 3)
    return _TrajJob(
        env_index=e,
        hits_grid=hits_grid,
        root_pos=root_pos,
        root_quat=root_quat,
        joint_angles=joint_angles,
        foot_pos=foot_pos,
        vx=vx,
        vy=vy,
        yaw_rate=yaw_rate,
        horizon=horizon,
        dt=dt,
        size_xy=size_xy,
    )


def _ensure_batched_cache_structure(
    env,
    horizon: int,
    device: torch.device,
) -> ReferenceTrajectoryCache:
    cache = getattr(env.unwrapped, "_trajectory_reference_cache", None)
    if (
        cache is not None
        and cache.root_pos_w is not None
        and cache.root_pos_w.ndim == 3
        and cache.root_pos_w.shape[0] == env.num_envs
        and cache.horizon_length() == horizon
    ):
        return cache
    gen = ReferenceGenerator(ReferenceGeneratorConfig(horizon_steps=horizon))
    base = gen.generate()
    return expand_reference_cache_to_num_envs(base, env.num_envs).to(device=device)


def _default_num_workers(cfg_workers: int, num_jobs: int) -> int:
    """Upper bound on concurrent workers; capped by ``num_jobs`` and CPU count."""
    cpu = os.cpu_count() or 4
    if cfg_workers > 0:
        n = int(cfg_workers)
    else:
        n = max(1, cpu)
    return max(1, min(num_jobs, n))


def _effective_parallel_backend(cfg_backend: str) -> str:
    env_override = os.environ.get("RAW_REFERENCE_PARALLEL_BACKEND", "").strip().lower()
    if env_override in ("process", "thread"):
        return env_override
    v = (cfg_backend or "process").strip().lower()
    return v if v in ("process", "thread") else "process"


_RAW_REF_PROC_POOL: ProcessPoolExecutor | None = None
_RAW_REF_PROC_N: int = 0
_RAW_REF_PROC_LOCK = threading.Lock()


def shutdown_raw_reference_executor(wait: bool = True) -> None:
    """Shut down the reused process pool (tests or process exit)."""
    global _RAW_REF_PROC_POOL, _RAW_REF_PROC_N
    with _RAW_REF_PROC_LOCK:
        if _RAW_REF_PROC_POOL is not None:
            _RAW_REF_PROC_POOL.shutdown(wait=wait)
            _RAW_REF_PROC_POOL = None
            _RAW_REF_PROC_N = 0


def _acquire_process_pool(workers: int) -> ProcessPoolExecutor:
    """Lazily create a ``spawn`` process pool reused across replans (avoids per-interval fork storms)."""
    global _RAW_REF_PROC_POOL, _RAW_REF_PROC_N
    _ensure_spawn_worker_pythonpath()
    ctx = mp.get_context("spawn")
    with _RAW_REF_PROC_LOCK:
        if _RAW_REF_PROC_POOL is not None and _RAW_REF_PROC_N == workers:
            return _RAW_REF_PROC_POOL
        if _RAW_REF_PROC_POOL is not None:
            _RAW_REF_PROC_POOL.shutdown(wait=True)
            _RAW_REF_PROC_POOL = None
            _RAW_REF_PROC_N = 0
        _RAW_REF_PROC_POOL = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        _RAW_REF_PROC_N = workers
        return _RAW_REF_PROC_POOL


atexit.register(lambda: shutdown_raw_reference_executor(wait=True))


def replan_raw_reference_trajectories(
    env,
    env_ids: torch.Tensor | Sequence[int] | slice | None,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Fill ``_trajectory_reference_cache`` with raw go2fp trajectories for the given env indices.

    Intended for ``mode="startup"`` (``env_ids`` is ``None``), ``mode="interval"`` with
    ``is_global_time=True`` (``env_ids`` is ``None``), or per-env ``mode="interval"`` /
    ``mode="reset"`` subsets (tensor of indices). Not tied to episode reset by default.
    """
    if not getattr(env.cfg, "use_raw_reference_trajectory", False):
        return

    ids = _normalize_env_ids(env, env_ids)
    if not ids:
        return

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids = _resolve_foot_body_ids(env, asset_cfg)
    pat = sensor.cfg.pattern_cfg
    size_xy = (float(pat.size[0]), float(pat.size[1]))

    horizon = int(getattr(env.cfg, "reference_trajectory_horizon", 50))
    dt = float(env.cfg.sim.dt * env.cfg.decimation)

    ensure_kinematic_footsteps_on_syspath()
    cache = _ensure_batched_cache_structure(env, horizon, torch.device("cpu"))

    jobs = [_build_traj_job(i, sensor, asset, foot_ids, size_xy, env, horizon, dt) for i in ids]
    workers = _default_num_workers(int(getattr(env.cfg, "raw_reference_num_threads", 0)), len(jobs))
    backend = _effective_parallel_backend(str(getattr(env.cfg, "raw_reference_parallel_backend", "process")))
    print(
        f"[reference_trajectory_events] replan raw reference for {len(jobs)} env(s), "
        f"backend={backend}, max_workers={workers}",
        flush=True,
    )

    if workers <= 1 or len(jobs) == 1:
        for job in jobs:
            row = _worker_traj(job)
            _write_cache_row(cache, row)
    elif backend == "thread":
        shutdown_raw_reference_executor(wait=False)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker_traj, job) for job in jobs]
            for fut in as_completed(futures):
                _write_cache_row(cache, fut.result())
    else:
        ex = _acquire_process_pool(workers)
        futures = [ex.submit(_worker_traj, job) for job in jobs]
        for fut in as_completed(futures):
            _write_cache_row(cache, fut.result())

    env.unwrapped._trajectory_reference_cache = cache.to(device=env.device)


def _write_cache_row(
    cache: ReferenceTrajectoryCache,
    row: tuple,
) -> None:
    (
        e,
        root_pos_w,
        root_quat_w,
        joint_angles,
        foot_pos_root,
        contact_state,
        planned_touchdown_w,
        phase_index,
        valid_mask,
    ) = row
    if isinstance(root_pos_w, np.ndarray):
        cache.root_pos_w[e] = torch.as_tensor(root_pos_w, dtype=cache.root_pos_w.dtype, device=cache.root_pos_w.device)
        cache.root_quat_w[e] = torch.as_tensor(root_quat_w, dtype=cache.root_quat_w.dtype, device=cache.root_quat_w.device)
        cache.joint_angles[e] = torch.as_tensor(joint_angles, dtype=cache.joint_angles.dtype, device=cache.joint_angles.device)
        cache.foot_pos_root[e] = torch.as_tensor(foot_pos_root, dtype=cache.foot_pos_root.dtype, device=cache.foot_pos_root.device)
        cache.contact_state[e] = torch.as_tensor(contact_state, dtype=torch.bool, device=cache.contact_state.device)
        cache.planned_touchdown_w[e] = torch.as_tensor(
            planned_touchdown_w, dtype=cache.planned_touchdown_w.dtype, device=cache.planned_touchdown_w.device
        )
        cache.phase_index[e] = torch.as_tensor(phase_index, dtype=torch.long, device=cache.phase_index.device)
        cache.valid_mask[e] = torch.as_tensor(valid_mask, dtype=torch.bool, device=cache.valid_mask.device)
    else:
        cache.root_pos_w[e] = root_pos_w
        cache.root_quat_w[e] = root_quat_w
        cache.joint_angles[e] = joint_angles
        cache.foot_pos_root[e] = foot_pos_root
        cache.contact_state[e] = contact_state
        cache.planned_touchdown_w[e] = planned_touchdown_w
        cache.phase_index[e] = phase_index
        cache.valid_mask[e] = valid_mask


# Backward-compatible name (older docs / branches)
reset_raw_reference_trajectories = replan_raw_reference_trajectories
