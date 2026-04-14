"""Numerically compare raw and batched trajectory generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extension.reference.raw_bridge import ensure_kinematic_footsteps_on_syspath

class FlatTerrain:
    def height_at(self, x, y=None):
        if y is None:
            points = torch.as_tensor(x, dtype=torch.float64)
            return torch.zeros(points.shape[:-1], dtype=torch.float64, device=points.device)
        return 0.0

    def roughness_at(self, x, y=None):
        if y is None:
            points = torch.as_tensor(x, dtype=torch.float64)
            return torch.zeros(points.shape[:-1], dtype=torch.float64, device=points.device)
        return 0.0

    def max_height_along_segment(self, p0, p1=None):
        if p1 is None:
            p0_t = torch.as_tensor(p0, dtype=torch.float64)
            return torch.zeros(p0_t.shape[0], dtype=torch.float64, device=p0_t.device)
        if not isinstance(p0, tuple):
            p0_t = torch.as_tensor(p0, dtype=torch.float64)
            return torch.zeros(p0_t.shape[0], dtype=torch.float64, device=p0_t.device)
        return 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw and batched trajectory outputs numerically.")
    parser.add_argument("--no-gui", action="store_true", help="Accepted for plan compatibility; no GUI is launched.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command", type=str, default="0.35 0.0 0.1", help='Body-frame command as "vx vy yaw_rate".')
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.02)
    return parser.parse_args()


def _build_matching_configs():
    from extension.batched_planner.config import BatchedTrajectoryConfig

    ensure_kinematic_footsteps_on_syspath()
    from scripts.go2fp.config import TrajectoryConfig

    batched_cfg = BatchedTrajectoryConfig(step_freq=2.0, duty_factor=0.6)
    raw_cfg = TrajectoryConfig(
        gait_name=batched_cfg.gait_name,
        step_freq=batched_cfg.step_freq,
        duty_factor=batched_cfg.duty_factor,
        step_height=batched_cfg.step_height,
        hip_height=batched_cfg.hip_height,
        body_clearance_margin=batched_cfg.body_clearance_margin,
        foothold_search_radius=batched_cfg.foothold_search_radius,
        foothold_search_step=batched_cfg.foothold_search_step,
        max_foothold_step_down=batched_cfg.max_foothold_step_down,
        max_touchdown_xy_reach=batched_cfg.max_touchdown_xy_reach,
        replan_stop_speed=batched_cfg.replan_stop_speed,
    )
    return batched_cfg, raw_cfg


def _default_batched_state():
    from extension.batched_planner.types import BatchedRobotState

    ensure_kinematic_footsteps_on_syspath()
    from scripts.go2fp.trajectory import default_initial_state

    raw_state = default_initial_state(None, x=0.0, y=0.0)
    batched_state = BatchedRobotState(
        root_pos=torch.as_tensor(raw_state.root_pos, dtype=torch.float64).unsqueeze(0),
        root_quat=torch.as_tensor(raw_state.root_quat, dtype=torch.float64).unsqueeze(0),
        joint_angles=torch.as_tensor(raw_state.joint_angles, dtype=torch.float64).unsqueeze(0),
        foot_pos=torch.as_tensor(raw_state.foot_pos, dtype=torch.float64).unsqueeze(0),
    )
    return batched_state, raw_state


def _max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item()) if a.numel() > 0 else 0.0


def main() -> int:
    args = _parse_args()
    torch.manual_seed(int(args.seed))

    from extension.batched_planner.trajectory import batched_generate_trajectory

    ensure_kinematic_footsteps_on_syspath()
    from scripts.go2fp.trajectory import generate_trajectory
    from scripts.go2fp.types import Command

    terrain = FlatTerrain()
    batched_cfg, raw_cfg = _build_matching_configs()
    batched_state, raw_state = _default_batched_state()
    vx, vy, yaw_rate = [float(part) for part in args.command.split()]

    batched = batched_generate_trajectory(
        terrain,
        batched_state,
        torch.tensor([[vx, vy, yaw_rate]], dtype=torch.float64),
        requested_n_frames=int(args.n_frames),
        dt=float(args.dt),
        cfg=batched_cfg,
    )
    raw = generate_trajectory(
        terrain,
        raw_state,
        Command(vx, vy, yaw_rate),
        int(args.n_frames),
        dt=float(args.dt),
        config=raw_cfg,
    )

    comparisons = [
        ("root_pos", _max_err(batched.root_pos_w[0], torch.as_tensor(raw.root_pos_w, dtype=torch.float64))),
        ("root_quat", _max_err(batched.root_quat_w[0], torch.as_tensor(raw.root_quat_w, dtype=torch.float64))),
        ("joint_angles", _max_err(batched.joint_angles[0], torch.as_tensor(raw.joint_angles, dtype=torch.float64))),
        ("foot_pos", _max_err(batched.foot_pos_w[0], torch.as_tensor(raw.foot_pos_w, dtype=torch.float64))),
        ("touchdown", _max_err(batched.planned_touchdown_w[0], torch.as_tensor(raw.planned_touchdown_w, dtype=torch.float64))),
    ]
    contact_match = bool(torch.equal(batched.contact_state[0], torch.as_tensor(raw.contact_state, dtype=batched.contact_state.dtype)))
    threshold = 1e-5
    all_ok = contact_match and all(err < threshold for _, err in comparisons)

    print("=== Trajectory Alignment Report ===")
    for name, err in comparisons:
        status = "PASS" if err < threshold else "FAIL"
        print(f"{name:<12} max_err: {err:.2e}  {status} (< {threshold:.0e})")
    print(f"{'contact':<12} exact_match: {contact_match}  {'PASS' if contact_match else 'FAIL'}")
    print(f"=== {'ALL FIELDS ALIGNED' if all_ok else 'MISMATCHES FOUND'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
