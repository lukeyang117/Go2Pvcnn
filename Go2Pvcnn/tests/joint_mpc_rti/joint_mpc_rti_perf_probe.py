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
    parser = argparse.ArgumentParser(description="Fixed-shape joint MPC RTI CUDA performance probe.")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--compile-kernels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--line-search-alphas", type=float, nargs="+", default=(1.0, 0.5, 0.25))
    parser.add_argument("--diagonal-state-riccati", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action="store_true")
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


def _shared_flat_field(batch: int, device: torch.device):
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
    from extension.joint_mpc_rti.types import JointMpcTerrainField

    base = build_field_batch(
        height_w=torch.zeros(1, 151, 151, device=device),
        semantic_id=torch.zeros(1, 151, 151, dtype=torch.long, device=device),
        origin_w=torch.zeros(1, 3, device=device),
        yaw_w=torch.zeros(1, device=device),
        timestamp=torch.zeros(1, device=device),
        version=torch.ones(1, dtype=torch.long, device=device),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )

    def rows(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.expand(batch, *tensor.shape[1:])

    return JointMpcTerrainField(
        height_w=rows(base.height_w),
        semantic_id=rows(base.semantic_id),
        small_distance_m=rows(base.small_distance_m),
        large_distance_m=rows(base.large_distance_m),
        small_gradient_xy=rows(base.small_gradient_xy),
        large_gradient_xy=rows(base.large_gradient_xy),
        valid_mask=rows(base.valid_mask),
        origin_w=rows(base.origin_w),
        yaw_w=rows(base.yaw_w),
        timestamp=rows(base.timestamp),
        version=rows(base.version),
        resolution=base.resolution,
    )


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.diagnostics.metrics import timing_summary
    from extension.joint_mpc_rti.diagnostics.profiler import benchmark_cuda_replay
    from extension.joint_mpc_rti.diagnostics.validation import nonfinite_count
    from extension.joint_mpc_rti.planner import step
    from extension.joint_mpc_rti.runtime.cuda_graph import JointMpcCudaGraphRunner

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    cfg = JointMpcRtiCfg()
    cfg.runtime.horizon_steps = int(args.horizon)
    cfg.solver.compile_kernels = bool(args.compile_kernels)
    cfg.solver.emit_loss_breakdown = False
    cfg.solver.line_search_alphas = tuple(float(value) for value in args.line_search_alphas)
    cfg.solver.diagonal_state_riccati = bool(args.diagonal_state_riccati)
    measured_state = _state(int(args.num_envs), device)
    command = torch.tensor([0.2, 0.0, 0.0], device=device).expand(int(args.num_envs), -1).clone()
    field = _shared_flat_field(int(args.num_envs), device)

    torch.cuda.reset_peak_memory_stats()
    result = step(measured_state, command, field, None, cfg)
    result = step(measured_state, command, field, result.solver_state, cfg)
    torch.cuda.synchronize()

    if bool(args.cuda_graph):
        runner = JointMpcCudaGraphRunner(measured_state, command, field, result.solver_state, cfg)
        replay = lambda: runner.run(measured_state, command, field)
        metrics = benchmark_cuda_replay(replay, steps=int(args.steps), warmup=int(args.warmup))
        if bool(args.profile):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                for _ in range(5):
                    replay()
                torch.cuda.synchronize()
            print(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=30), file=sys.stderr)
        result = runner.captured_result
        mode = "cuda_graph"
    else:
        samples: list[float] = []
        solver_state = result.solver_state
        for _ in range(int(args.warmup)):
            result = step(measured_state, command, field, solver_state, cfg)
            solver_state = result.solver_state
        torch.cuda.synchronize()
        for _ in range(int(args.steps)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = step(measured_state, command, field, solver_state, cfg)
            solver_state = result.solver_state
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        metrics = timing_summary(samples)
        mode = "eager"

    output: dict[str, object] = {
        "mode": mode,
        "num_envs": int(args.num_envs),
        "horizon": int(args.horizon),
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "compile_kernels": bool(args.compile_kernels),
        "line_search_alphas": list(cfg.solver.line_search_alphas),
        "diagonal_state_riccati": bool(cfg.solver.diagonal_state_riccati),
        "nonfinite_count": nonfinite_count(result),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
        "total_ms": float(metrics["total_ms"]),
        **metrics,
    }
    return output


def main() -> None:
    args = _parse_args()
    print(json.dumps(run_probe(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
