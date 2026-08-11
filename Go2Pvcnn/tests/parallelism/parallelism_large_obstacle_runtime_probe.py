from __future__ import annotations

import argparse
import time

import torch

from extension.parallelism import ParallelismCfg, ParallelismState, ParallelismTerrain
from extension.parallelism.root import _large_obstacle_avoidance_command


def _terrain(num_envs: int, height: int, width: int, device: torch.device) -> ParallelismTerrain:
    semantic = torch.zeros(num_envs, height, width, dtype=torch.long, device=device)
    center_row = height // 2
    center_col = width // 2
    row0 = max(center_row - 8, 0)
    row1 = min(center_row + 9, height)
    col0 = min(center_col + 20, width - 1)
    col1 = min(center_col + 45, width)
    semantic[:, row0:row1, col0:col1] = 2
    return ParallelismTerrain(
        height_w=torch.zeros(num_envs, height, width, dtype=torch.float32, device=device),
        semantic_id=semantic,
        valid_mask=torch.ones(num_envs, height, width, dtype=torch.bool, device=device),
        origin_w=torch.tensor([[-0.75, -0.75, 0.0]], dtype=torch.float32, device=device).repeat(num_envs, 1),
        yaw_w=torch.zeros(num_envs, dtype=torch.float32, device=device),
        resolution=0.01,
    )


def _state(num_envs: int, device: torch.device) -> ParallelismState:
    return ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.34]], dtype=torch.float32, device=device).repeat(num_envs, 1),
        root_rpy_w=torch.zeros(num_envs, 3, dtype=torch.float32, device=device),
        joint_pos=torch.tensor([[0.0, 0.8, -1.5] * 4], dtype=torch.float32, device=device).repeat(num_envs, 1),
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--height", type=int, default=151)
    parser.add_argument("--width", type=int, default=151)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = ParallelismCfg()
    terrain = _terrain(args.num_envs, args.height, args.width, device)
    state = _state(args.num_envs, device)
    command = torch.zeros(args.num_envs, 3, dtype=torch.float32, device=device)
    command[:, 0] = 0.8

    for _ in range(max(args.warmup, 0)):
        _large_obstacle_avoidance_command(state, command, terrain, cfg)
    _sync(device)

    start = time.perf_counter()
    result = None
    for _ in range(max(args.iters, 1)):
        result = _large_obstacle_avoidance_command(state, command, terrain, cfg)
    _sync(device)
    elapsed = time.perf_counter() - start
    per_call_ms = elapsed * 1000.0 / max(args.iters, 1)

    print(f"device={device}")
    print(f"shape={tuple(result.shape) if result is not None else None}")
    print(f"num_envs={args.num_envs} grid={args.height}x{args.width}")
    print(f"iters={args.iters} per_call_ms={per_call_ms:.4f}")
    print(f"mean_command={result.mean(dim=0).detach().cpu().tolist() if result is not None else None}")
    if device.type == "cuda":
        print(f"max_memory_mb={torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
