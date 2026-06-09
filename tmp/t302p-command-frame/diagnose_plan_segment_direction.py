from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
GO2 = REPO / "Go2Pvcnn"
RSL = GO2 / "rsl_rl"
for path in (GO2, RSL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _tolist(value):
    return torch.as_tensor(value).detach().cpu().tolist()


def _direction_metrics(root_pos, root_yaw, command):
    from extension.batch_mpc_planner.parametric import command_frame_axes

    root = torch.as_tensor(root_pos, dtype=torch.float32)
    yaw = torch.as_tensor(root_yaw, dtype=torch.float32, device=root.device).reshape(-1)
    cmd = torch.as_tensor(command, dtype=torch.float32, device=root.device)
    forward, left, active = command_frame_axes(cmd, yaw, linear_eps=1.0e-6)
    delta = root[:, -1, :2] - root[:, 0, :2]
    norm = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-6)
    cosine = (delta * forward).sum(dim=-1) / norm
    lateral = torch.abs((delta * left).sum(dim=-1)) / norm
    return {
        "root_delta_xy": _tolist(delta),
        "expected_forward_w": _tolist(forward),
        "expected_left_w": _tolist(left),
        "linear_active": _tolist(active),
        "cosine": _tolist(cosine),
        "lateral_ratio": _tolist(lateral),
    }


def main() -> int:
    import extension.batch_mpc_planner.manager as manager
    import scripts.mpc_policy_eval as eval_script

    args = eval_script.build_arg_parser().parse_args()
    original = manager.plan_segment
    call_index = 0

    def wrapped_plan_segment(terrain, state, command, *, cfg):
        nonlocal call_index
        result = original(terrain, state, command, cfg=cfg)
        yaw = torch.as_tensor(state.root_rpy, dtype=torch.float32)[:, 2]
        row = {
            "kind": "plan_segment_direction",
            "call_index": call_index,
            "root_yaw_rad": _tolist(yaw),
            "root_pos0_xy": _tolist(torch.as_tensor(state.root_pos, dtype=torch.float32)[:, :2]),
            "command_body": _tolist(command),
            "result": _direction_metrics(result.root_pos, yaw, command),
            "feasible": _tolist(getattr(result, "feasible", torch.empty(0))),
            "safe_fallback": _tolist(getattr(result, "safe_fallback", torch.empty(0))),
            "cost_total": _tolist(getattr(result, "cost_total", torch.empty(0))),
            "cost_breakdown": {
                str(k): _tolist(v)
                for k, v in dict(getattr(result, "cost_breakdown", {}) or {}).items()
            },
            "loss_breakdown": {
                str(k): _tolist(v)
                for k, v in dict(getattr(result, "loss_breakdown", {}) or {}).items()
            }
            if getattr(result, "loss_breakdown", None) is not None
            else None,
        }
        print("[T302pPlanSegmentProbe] " + json.dumps(row, sort_keys=True), flush=True)
        call_index += 1
        return result

    manager.plan_segment = wrapped_plan_segment
    return int(eval_script.run_eval(args))


if __name__ == "__main__":
    raise SystemExit(main())
