"""Load `raw/kinematic_footsteps` go2fp planner and build `ReferenceTrajectoryCache`.

The raw planner expects ``sys.path`` to include ``<repo>/raw/kinematic_footsteps`` so
``scripts.go2fp`` imports resolve. This module registers that path once and converts
``TrajectoryResult`` into the torch cache used by imitation rewards.

Training still uses the placeholder ``ReferenceGenerator`` until env wiring passes
terrain + robot state + command into these helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .reference_cache import ReferenceTrajectoryCache

_RAW_ROOT: Path | None = None


def kinematic_footsteps_repo_root() -> Path:
    """Return ``<repo>/raw/kinematic_footsteps`` (must exist on disk)."""
    global _RAW_ROOT
    if _RAW_ROOT is not None and _RAW_ROOT.is_dir():
        return _RAW_ROOT
    here = Path(__file__).resolve()
    # .../Go2Pvcnn/extension/planner/runtime/raw_go2fp_bridge.py -> parents[4] == repo root
    repo = here.parents[4]
    root = repo / "raw" / "kinematic_footsteps"
    if not root.is_dir():
        raise FileNotFoundError(
            f"Expected raw kinematic_footsteps at {root}; clone or sync raw/ before using the bridge."
        )
    _RAW_ROOT = root
    return root


def ensure_kinematic_footsteps_on_syspath() -> Path:
    """Insert the kinematic_footsteps tree at the front of ``sys.path`` if needed."""
    root = kinematic_footsteps_repo_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def trajectory_result_to_reference_cache(result: Any, *, dtype: torch.dtype = torch.float32) -> ReferenceTrajectoryCache:
    """Map a go2fp ``TrajectoryResult`` (or duck-typed equivalent) to ``ReferenceTrajectoryCache``."""
    root_pos_w = torch.as_tensor(np.asarray(result.root_pos_w), dtype=dtype)
    n = int(root_pos_w.shape[0])

    root_quat_w = torch.as_tensor(np.asarray(result.root_quat_w), dtype=dtype)
    joint_angles = torch.as_tensor(np.asarray(result.joint_angles), dtype=dtype)
    foot_pos_root = torch.as_tensor(np.asarray(result.foot_pos_root), dtype=dtype)

    contact_np = np.asarray(result.contact_state)
    if contact_np.dtype == np.dtype(bool):
        contact_state = torch.as_tensor(contact_np, dtype=torch.bool)
    else:
        contact_state = torch.as_tensor(contact_np > 0.5, dtype=torch.bool)

    planned = np.asarray(result.planned_touchdown_w, dtype=np.float64)
    if planned.ndim == 2:
        planned = np.broadcast_to(planned.reshape(1, 4, 3), (n, 4, 3)).copy()
    planned_touchdown_w = torch.as_tensor(planned, dtype=dtype)

    phase_index = torch.arange(n, dtype=torch.long)
    valid_mask = torch.ones(n, dtype=torch.bool)
    return ReferenceTrajectoryCache(
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        joint_angles=joint_angles,
        foot_pos_root=foot_pos_root,
        contact_state=contact_state,
        planned_touchdown_w=planned_touchdown_w,
        phase_index=phase_index,
        valid_mask=valid_mask,
    )


def generate_reference_cache_with_raw(
    *,
    terrain: Any | None = None,
    initial_state: Any | None = None,
    command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_frames: int = 50,
    dt: float = 0.02,
    trajectory_config: Any | None = None,
) -> ReferenceTrajectoryCache:
    """Run raw ``generate_trajectory`` and return a CPU ``ReferenceTrajectoryCache``."""
    ensure_kinematic_footsteps_on_syspath()
    from scripts.go2fp.config import TrajectoryConfig
    from scripts.go2fp.trajectory import default_initial_state, generate_trajectory
    from scripts.go2fp.types import Command

    if initial_state is None:
        initial_state = default_initial_state(terrain, 0.0, 0.0)
    cmd = Command(float(command[0]), float(command[1]), float(command[2]))
    cfg = trajectory_config if trajectory_config is not None else TrajectoryConfig()
    tr = generate_trajectory(terrain, initial_state, cmd, int(n_frames), float(dt), cfg)
    return trajectory_result_to_reference_cache(tr)


__all__ = [
    "ensure_kinematic_footsteps_on_syspath",
    "generate_reference_cache_with_raw",
    "kinematic_footsteps_repo_root",
    "trajectory_result_to_reference_cache",
]
