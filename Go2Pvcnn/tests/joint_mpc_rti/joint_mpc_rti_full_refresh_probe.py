from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronous exact-field plus joint MPC RTI probe.")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    return parser.parse_args()


def _state(batch: int, device: torch.device):
    from extension.joint_mpc_rti.types import JointMpcRtiState

    root_pos = torch.zeros(batch, 3, device=device)
    root_pos[:, 2] = 0.32
    joint = torch.tensor([0.0, 0.8, -1.5] * 4, device=device).expand(batch, -1).clone()
    return JointMpcRtiState(
        root_pos_w=root_pos,
        root_rpy_w=torch.zeros(batch, 3, device=device),
        joint_pos=joint,
        root_lin_vel_b=torch.zeros(batch, 3, device=device),
        root_ang_vel_b=torch.zeros(batch, 3, device=device),
        joint_vel=torch.zeros(batch, 12, device=device),
    )


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.diagnostics.metrics import timing_summary
    from extension.joint_mpc_rti.diagnostics.validation import nonfinite_count
    from extension.joint_mpc_rti.planner import step as planner_step
    from extension.joint_mpc_rti.runtime.cuda_graph import JointMpcCudaGraphRunner
    from extension.joint_mpc_rti.terrain.field_cache import JointMpcTerrainFieldCache

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    batch = int(args.num_envs)
    grid_size = 151
    cfg = JointMpcRtiCfg()
    cfg.runtime.horizon_steps = int(args.horizon)
    cfg.solver.compile_kernels = True
    cfg.solver.emit_loss_breakdown = False
    cfg.solver.line_search_alphas = (1.0, 0.25)
    cfg.solver.diagonal_state_riccati = True

    measured_state = _state(batch, device)
    command = torch.tensor([0.2, 0.0, 0.0], device=device).expand(batch, -1).clone()
    ray_hits = torch.zeros(batch, grid_size * grid_size, 3, device=device)
    height_view = ray_hits[..., 2].reshape(batch, grid_size, grid_size)
    semantic = torch.zeros(batch, grid_size, grid_size, dtype=torch.long, device=device)
    semantic[:, 70:81, 70:81] = 1
    semantic[:, 15:56, 105:146] = 2
    env_ids = torch.arange(batch, device=device)
    origin = torch.zeros(batch, 3, device=device)
    yaw = torch.zeros(batch, device=device)
    timestamp = torch.zeros(batch, device=device)
    cache = JointMpcTerrainFieldCache(
        num_envs=batch,
        grid_size=grid_size,
        device=device,
        resolution=0.01,
    )

    def refresh_field() -> None:
        cache.update_rows(
            env_ids=env_ids,
            height_w=height_view,
            semantic_id=semantic,
            origin_w=origin,
            yaw_w=yaw,
            timestamp=timestamp,
            ordered_full_batch=True,
        )

    refresh_field()
    field = cache.as_field()
    result = planner_step(measured_state, command, field, None, cfg)
    result = planner_step(measured_state, command, field, result.solver_state, cfg)
    torch.cuda.synchronize()
    runner = JointMpcCudaGraphRunner(measured_state, command, field, result.solver_state, cfg)

    for _ in range(int(args.warmup)):
        refresh_field()
        runner.run(measured_state, command, field)
    torch.cuda.synchronize()
    version_before = cache.version.clone()
    torch.cuda.reset_peak_memory_stats()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(int(args.steps))]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(int(args.steps))]
    for start, end in zip(starts, ends):
        start.record()
        refresh_field()
        runner.run(measured_state, command, field)
        end.record()
    torch.cuda.synchronize()

    full_samples = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    version_increment = int((cache.version[0] - version_before[0]).cpu())

    diagnostic_steps = min(100, int(args.steps))
    field_samples: list[float] = []
    mpc_samples: list[float] = []
    for _ in range(diagnostic_steps):
        start = torch.cuda.Event(enable_timing=True)
        field_end = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        refresh_field()
        field_end.record()
        runner.run(measured_state, command, field)
        end.record()
        end.synchronize()
        field_samples.append(float(start.elapsed_time(field_end)))
        mpc_samples.append(float(field_end.elapsed_time(end)))
    field_metrics = timing_summary(field_samples)
    mpc_metrics = timing_summary(mpc_samples)
    full_metrics = timing_summary(full_samples)
    result = runner.captured_result
    return {
        "num_envs": batch,
        "horizon": int(args.horizon),
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "exact_edt": True,
        "synchronous": True,
        "field_version_increment": version_increment,
        "nonfinite_count": nonfinite_count(result),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
        "field_mean_ms": float(field_metrics["mean_ms"]),
        "field_p95_ms": float(field_metrics["p95_ms"]),
        "mpc_mean_ms": float(mpc_metrics["mean_ms"]),
        "mpc_p95_ms": float(mpc_metrics["p95_ms"]),
        "full_total_ms": float(full_metrics["total_ms"]),
        "full_mean_ms": float(full_metrics["mean_ms"]),
        "full_p50_ms": float(full_metrics["p50_ms"]),
        "full_p95_ms": float(full_metrics["p95_ms"]),
        "full_p99_ms": float(full_metrics["p99_ms"]),
        "full_max_ms": float(full_metrics["max_ms"]),
    }


def main() -> None:
    print(json.dumps(run_probe(_parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
