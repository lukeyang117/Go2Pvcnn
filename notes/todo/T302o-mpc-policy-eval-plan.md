# T302o MPC Policy Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single Python evaluation entry that loads a policy checkpoint with MPC reference enabled, measures policy-vs-MPC foot tracking on multiple terrains, and measures flat dense-small-obstacle collision rate by collided envs per round.

**Architecture:** Add `Go2Pvcnn/scripts/mpc_policy_eval.py` as the only user-facing entry. Add eval-specific task cfg classes in `teacher_elevation_trajectory_mpc_semantic_env_cfg.py` so the script can enable MPC reference/cache without changing `scripts/play.py` no-MPC behavior. Metrics are collected by small helper functions inside the script first; extract later only if the file becomes hard to maintain.

**Tech Stack:** IsaacLab `AppLauncher`, Gymnasium/IsaacLab envs, RSL-RL runner policy loading, existing `extension.batch_mpc_planner`, existing global semantic contact sensors, PyTorch, JSONL/JSON output.

---

## Source Spec

- [../../docs/superpowers/specs/2026-06-05-mpc-policy-eval-design.html](../../docs/superpowers/specs/2026-06-05-mpc-policy-eval-design.html)

## Current State

- Design doc is committed as `f46eab8 docs: design mpc policy evaluation script`.
- Task 1 static contracts are implemented and verified at `d6a0d45`: eval cfg classes exist, the Python-only CLI skeleton exists, and PLAY no-MPC behavior is covered by static contract tests.
- Task 2 metric helpers are implemented and verified at `e84a78c` after import isolation review: tracking metrics report mean/p95/per-leg foot error, command helpers support fixed/sweep/random, and small-collision accumulation counts each env once per round.
- Existing `scripts/play.py` intentionally uses `TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY` and disables MPC attachment. T302o must not regress this.
- Existing `TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER` enables MPC reference/cache, `reference_foot_pos`, and global semantic contact sensors.
- Existing global semantic contact route provides `semantic_contact_small.data.force_matrix_w` and `semantic_contact_large.data.force_matrix_w`.
- Existing `play.py` has checkpoint loading patterns, AppLauncher livestream handling, and RSL-RL wrapper setup that T302o can mirror.
- Existing `go2_foostep_planner.py` has MPC foot marker visualization patterns that T302o can reuse for livestream overlays.

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T302o.1 | active | P0 | Implement CLI, eval cfg contracts, tracking metrics, small collision env-rate metrics, livestream command sync and MPC foot overlays. | `Go2Pvcnn/scripts/mpc_policy_eval.py`, `teacher_elevation_trajectory_mpc_semantic_env_cfg.py`, tests |

## File Structure

- Create `Go2Pvcnn/scripts/mpc_policy_eval.py`
  - Owns CLI parsing, IsaacLab app launch, env construction, checkpoint loading, rollout loop, command generation, metrics aggregation, JSON output, and livestream overlay hooks.
  - Uses no shell wrapper.
- Modify `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - Adds `TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg`.
  - Adds `TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg`.
  - Preserves `TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY` no-MPC behavior.
- Modify `Go2Pvcnn/tests/test_batch_mpc_backend.py`
  - Adds static cfg contract tests for eval cfgs and play cfg non-regression.
- Modify `Go2Pvcnn/tests/test_viewer_reset.py`
  - Adds static CLI/source tests for `mpc_policy_eval.py` AppLauncher args, no `.sh` dependency, and livestream/round parameters.
- Create `Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py`
  - Tests pure metric helpers without IsaacLab startup.
- Create `Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py`
  - Tests script source contracts that do not need IsaacLab runtime.
- Modify `notes/todo.md`, this page, `notes/log/index.md`, and create per-verification logs as tasks complete.

## Global Constraints

- Do not create a `.sh` launcher.
- Do not modify `Go2Pvcnn/scripts/play.py` except adding tests that guard its current no-MPC behavior.
- Do not modify IsaacLab source.
- Do not restore old `batched_planner`, `batched_together_planner`, or dense residual MPC routes.
- Do not change MPC losses for this task.
- `small_collision` main metric must be env-count based:

```text
small_collision_env_rate_per_round = collided_env_count / num_envs
aggregate_small_collision_env_rate = sum(collided_env_count) / (num_rounds * num_envs)
```

- `max_steps` defines one round. `num_rounds` defines how many rounds to run.
- `max_steps=0` is only valid for livestream/manual viewing. Headless automatic metrics require `max_steps > 0`.
- In livestream mode, the same body-frame command must be written to policy command input and MPC reference planning path.

---

## Task 1: Static Contracts For Eval Cfgs And CLI

**Files:**
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`
- Modify: `Go2Pvcnn/tests/test_viewer_reset.py`
- Create: `Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py`

- [x] **Step 1: Add failing cfg contract tests**

Append these tests to `Go2Pvcnn/tests/test_batch_mpc_backend.py`:

```python
def test_mpc_policy_eval_cfgs_enable_reference_without_changing_play() -> None:
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
        TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg,
        TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg,
    )

    play = TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY()
    assert play.planner_owned_reference_cache is False
    assert play.use_batched_reference_trajectory is False
    assert play.rewards.reference_foot_pos is None
    assert play.scene.semantic_contact_small is None
    assert play.scene.semantic_contact_large is None

    tracking = TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg()
    assert tracking.planner_owned_reference_cache is True
    assert tracking.use_batched_reference_trajectory is True
    assert tracking.planner_backend == "mpc"
    assert tracking.rewards.reference_foot_pos is not None
    assert tracking.scene.semantic_contact_small is not None
    assert tracking.scene.semantic_contact_large is not None
    assert tracking.mpc_planner_cfg.runtime.horizon_steps == 25
    assert tracking.mpc_planner_cfg.runtime.replan_interval_steps == 25

    collision = TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg()
    assert collision.planner_owned_reference_cache is True
    assert collision.use_batched_reference_trajectory is True
    assert collision.planner_backend == "mpc"
    assert collision.scene.semantic_contact_small is not None
    assert collision.scene.semantic_contact_large is not None
    assert hasattr(collision, "small_collision_eval_small_count_per_tile")
    assert collision.small_collision_eval_small_count_per_tile > 0
```

- [x] **Step 2: Add failing script static tests**

Create `Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py`:

```python
from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "Go2Pvcnn/scripts/mpc_policy_eval.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_mpc_policy_eval_script_exists_and_has_required_cli() -> None:
    source = _source()
    for flag in (
        "--mode",
        "--num-rounds",
        "--max-steps",
        "--run-dir",
        "--checkpoint",
        "--command-mode",
        "--small-count-per-tile",
        "--collision-force-threshold",
        "--output-dir",
    ):
        assert flag in source
    assert "AppLauncher.add_app_launcher_args(parser)" in source
    assert "choices=[\"tracking\", \"small_collision\"]" in source


def test_mpc_policy_eval_script_has_no_shell_wrapper_dependency() -> None:
    source = _source()
    assert ".sh" not in source
    assert "subprocess" not in source


def test_mpc_policy_eval_script_defines_round_and_command_helpers() -> None:
    module = ast.parse(_source())
    function_names = {node.name for node in ast.walk(module) if isinstance(node, ast.FunctionDef)}
    assert "build_arg_parser" in function_names
    assert "validate_eval_args" in function_names
    assert "command_for_step" in function_names
    assert "run_eval" in function_names
    assert "main" in function_names
```

- [x] **Step 3: Run tests and confirm RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_policy_eval_cfgs_enable_reference_without_changing_play \
  Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
```

Expected: FAIL because eval cfgs and `mpc_policy_eval.py` do not exist yet.

- [x] **Step 4: Add minimal eval cfg classes**

In `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`, add classes after `TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER`:

```python
@configclass
class TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg(TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER):
    """MPC-enabled policy evaluation config for terrain tracking metrics."""

    def __post_init__(self):
        super().__post_init__()
        self.planner_owned_reference_cache = True
        self.use_batched_reference_trajectory = True
        self.planner_backend = "mpc"
        self.mpc_planner_cfg.runtime.horizon_steps = 25
        self.mpc_planner_cfg.runtime.replan_interval_steps = 25
        self.mpc_planner_cfg.runtime.dt = 0.02
        self.mpc_planner_cfg.runtime.parallel_plan_batch_size = 64
        self.mpc_planner_cfg.diagnostics.emit_runtime_counters = False
        self.mpc_planner_cfg.diagnostics.profile_cuda_sync = False


@configclass
class TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg(TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg):
    """MPC-enabled policy evaluation config for dense small-obstacle flat collision metrics."""

    small_collision_eval_small_count_per_tile: int = 80
    small_collision_eval_large_count_per_tile: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.small_collision_eval_small_count_per_tile = 80
        self.small_collision_eval_large_count_per_tile = 0
```

If `configclass` is not imported in the file, use the same decorator/import style already used by nearby cfg classes.

- [x] **Step 5: Add script skeleton**

Create `Go2Pvcnn/scripts/mpc_policy_eval.py`:

```python
"""Evaluate a trained policy against MPC reference tracking and semantic collisions."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--terrain-rows", type=str, default="0,3,6,9")
    parser.add_argument("--terrain-cols", type=str, default="0")
    parser.add_argument("--command-mode", choices=["fixed", "random", "sweep"], default="fixed")
    parser.add_argument("--command", type=str, default="0.4 0.0 0.0")
    parser.add_argument("--command-sweep", type=str, default="")
    parser.add_argument("--random-command-interval", type=int, default=100)
    parser.add_argument("--small-count-per-tile", type=int, default=80)
    parser.add_argument("--collision-force-threshold", type=float, default=1.0)
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


def command_for_step(args: argparse.Namespace, *, step: int, env_count: int, device: torch.device) -> torch.Tensor:
    values = [float(v) for v in str(args.command).split()]
    if len(values) != 3:
        raise ValueError("--command must contain exactly three floats: vx vy yaw")
    return torch.tensor(values, dtype=torch.float32, device=device).repeat(int(env_count), 1)


def run_eval(args: argparse.Namespace) -> int:
    validate_eval_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    return 0


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(run_eval(args))


if __name__ == "__main__":
    main()
```

- [x] **Step 6: Run tests and confirm GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_policy_eval_cfgs_enable_reference_without_changing_play \
  Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
```

Expected: PASS.

- [x] **Step 7: Commit static contract slice**

Run:

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/scripts/mpc_policy_eval.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py
git commit -m "feat: add mpc policy eval entry contracts"
```

---

## Task 2: Metric Helpers For Tracking And Small Collision

**Files:**
- Create: `Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py`
- Modify: `Go2Pvcnn/scripts/mpc_policy_eval.py`

- [x] **Step 1: Add failing metric tests**

Create `Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from Go2Pvcnn.scripts.mpc_policy_eval import (
    SmallCollisionRoundAccumulator,
    command_for_step,
    parse_command_sweep,
    tracking_foot_metrics,
)


def test_tracking_foot_metrics_report_mean_p95_and_per_leg() -> None:
    actual = torch.zeros((2, 4, 3), dtype=torch.float32)
    reference = torch.zeros((2, 4, 3), dtype=torch.float32)
    actual[0, 0, 0] = 0.10
    actual[1, 1, 1] = 0.20

    metrics = tracking_foot_metrics(actual, reference)

    assert metrics["foot_tracking_error_mean_m"] == pytest.approx(0.0375)
    assert metrics["foot_tracking_error_p95_m"] >= 0.10
    assert metrics["per_leg_foot_error_mean_m"][0] == pytest.approx(0.05)
    assert metrics["per_leg_foot_error_mean_m"][1] == pytest.approx(0.10)
    assert metrics["per_leg_foot_error_mean_m"][2] == pytest.approx(0.0)
    assert metrics["per_leg_foot_error_mean_m"][3] == pytest.approx(0.0)


def test_small_collision_accumulator_counts_each_env_once_per_round() -> None:
    acc = SmallCollisionRoundAccumulator(num_envs=4, threshold=1.0, device=torch.device("cpu"))
    force = torch.zeros((4, 2, 3, 3), dtype=torch.float32)
    force[1, 0, 0, 0] = 2.0
    acc.update(step=0, force_matrix_w=force, body_names=("base", "foot"))
    force.zero_()
    force[1, 1, 2, 1] = 3.0
    force[3, 0, 1, 2] = 4.0
    acc.update(step=5, force_matrix_w=force, body_names=("base", "foot"))

    summary = acc.summary()

    assert summary["collided_env_count"] == 2
    assert summary["num_envs"] == 4
    assert summary["small_collision_env_rate_per_round"] == pytest.approx(0.5)
    assert summary["first_collision_step_by_env"] == {"1": 0, "3": 5}
    assert summary["collision_body_names_by_env"]["1"] == ["base", "foot"]
    assert summary["round_small_force_max"] == pytest.approx(4.0)


def test_command_for_step_supports_fixed_and_sweep_modes() -> None:
    fixed = SimpleNamespace(command_mode="fixed", command="0.4 0.0 0.1", command_sweep="", random_command_interval=100)
    out = command_for_step(fixed, step=0, env_count=2, device=torch.device("cpu"))
    assert out.tolist() == [[0.4, 0.0, 0.1], [0.4, 0.0, 0.1]]

    sweep = SimpleNamespace(
        command_mode="sweep",
        command="0.0 0.0 0.0",
        command_sweep="0.1 0 0;0 0.2 0",
        random_command_interval=100,
    )
    assert parse_command_sweep(sweep.command_sweep) == [(0.1, 0.0, 0.0), (0.0, 0.2, 0.0)]
    assert command_for_step(sweep, step=0, env_count=1, device=torch.device("cpu")).tolist() == [[0.1, 0.0, 0.0]]
    assert command_for_step(sweep, step=1, env_count=1, device=torch.device("cpu")).tolist() == [[0.0, 0.2, 0.0]]
```

- [x] **Step 2: Run tests and confirm RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py -q
```

Expected: FAIL because metric helpers are not implemented.

- [x] **Step 3: Implement metric helpers**

In `Go2Pvcnn/scripts/mpc_policy_eval.py`, add:

```python
from dataclasses import dataclass, field


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
    return torch.tensor(values, dtype=torch.float32, device=device).repeat(int(env_count), 1)


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
            "collision_body_names_by_env": {
                str(k): sorted(v) for k, v in sorted(self.body_names_by_env.items())
            },
            "round_small_force_max": float(self.force_max),
        }
```

Replace the skeleton `command_for_step()` with this implementation.

- [x] **Step 4: Run metric tests and confirm GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py -q
```

Expected: PASS.

- [x] **Step 5: Commit metric helper slice**

Run:

```bash
git add Go2Pvcnn/scripts/mpc_policy_eval.py Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py
git commit -m "feat: add mpc policy eval metric helpers"
```

---

## Task 3: Headless Rollout Skeleton And Output Files

**Files:**
- Modify: `Go2Pvcnn/scripts/mpc_policy_eval.py`
- Modify: `Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py`

- [ ] **Step 1: Add static tests for output contracts**

Append to `Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py`:

```python
def test_mpc_policy_eval_writes_required_output_files() -> None:
    source = _source()
    assert "metrics.jsonl" in source
    assert "rounds.jsonl" in source
    assert "summary.json" in source
    assert "config.json" in source
    assert "write_jsonl" in source
    assert "write_summary" in source


def test_mpc_policy_eval_loads_policy_and_uses_eval_cfgs() -> None:
    source = _source()
    assert "OnPolicyRunner" in source
    assert "runner.load" in source
    assert "TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg" in source
    assert "TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg" in source
```

- [ ] **Step 2: Run static tests and confirm RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
```

Expected: FAIL until rollout/output/policy loading symbols exist.

- [ ] **Step 3: Implement output helpers**

Add to `Go2Pvcnn/scripts/mpc_policy_eval.py`:

```python
from datetime import datetime


def make_run_output_dir(base: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = base / stamp
    out.mkdir(parents=True, exist_ok=False)
    return out


def write_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Implement cfg selection and checkpoint path helpers**

Add:

```python
def build_eval_env_cfg(args: argparse.Namespace):
    if str(args.mode) == "tracking":
        from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
            TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg,
        )

        env_cfg = TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg()
    elif str(args.mode) == "small_collision":
        from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
            TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg,
        )

        env_cfg = TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg()
        env_cfg.small_collision_eval_small_count_per_tile = int(args.small_count_per_tile)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.sim.device = str(args.device)
    return env_cfg


def checkpoint_path(args: argparse.Namespace) -> Path:
    path = GO2PVCNN_ROOT / "logs" / "rsl_rl" / "teacher_elevation_trajectory_mpc_semantic" / str(args.run_dir) / str(args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path
```

- [ ] **Step 5: Implement rollout skeleton with policy loading**

Replace `run_eval()` with a real skeleton:

```python
def run_eval(args: argparse.Namespace) -> int:
    validate_eval_args(args)
    out_dir = make_run_output_dir(args.output_dir)
    config_path = out_dir / "config.json"
    metrics_path = out_dir / "metrics.jsonl"
    rounds_path = out_dir / "rounds.jsonl"
    summary_path = out_dir / "summary.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")

    from isaaclab.app import AppLauncher

    if getattr(args, "livestream", -1) in (1, 2) and not getattr(args, "enable_cameras", False):
        args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        from rsl_rl.runners import OnPolicyRunner
        from go2_pvcnn.tasks.register_envs import register_envs

        register_envs()
        env_cfg = build_eval_env_cfg(args)
        env = gym.make(
            "Isaac-Teacher-Elevation-Trajectory-Go2-Play-v0",
            cfg=env_cfg,
            render_mode="rgb_array" if getattr(args, "livestream", -1) in (1, 2) else None,
        )
        runner = OnPolicyRunner(env, env_cfg.to_dict().get("rl_runner_cfg", {}), log_dir=None, device=str(args.device))
        runner.load(str(checkpoint_path(args)), load_optimizer=False)
        policy = runner.get_inference_policy(device=str(args.device))
        summaries: list[dict[str, object]] = []
        for round_idx in range(int(args.num_rounds)):
            obs, _ = env.reset()
            round_summary = {
                "round": round_idx,
                "mode": str(args.mode),
                "num_envs": int(args.num_envs),
                "max_steps": int(args.max_steps),
            }
            step_limit = int(args.max_steps)
            step = 0
            while (step_limit == 0 and simulation_app.is_running()) or step < step_limit:
                command = command_for_step(args, step=step, env_count=int(args.num_envs), device=torch.device(str(args.device)))
                apply_command_to_env(env, command)
                with torch.inference_mode():
                    action = policy(obs)
                obs, _reward, terminated, truncated, _info = env.step(action)
                write_jsonl(metrics_path, {"round": round_idx, "step": step, "mode": str(args.mode)})
                if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
                    pass
                step += 1
                if step_limit == 0 and step > 0 and int(args.num_rounds) > 1:
                    break
            write_jsonl(rounds_path, round_summary)
            summaries.append(round_summary)
            if step_limit == 0:
                break
        write_summary(summary_path, {"rounds": summaries, "round_count": len(summaries)})
        env.close()
    finally:
        simulation_app.close()
    return 0
```

Also add the temporary command hook:

```python
def apply_command_to_env(env, command: torch.Tensor) -> None:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    command_manager = getattr(base, "command_manager", None)
    if command_manager is None:
        return
    for name in ("base_velocity", "base_velocity_command", "velocity_command"):
        term = None
        try:
            term = command_manager.get_term(name)
        except Exception:
            term = None
        if term is not None and hasattr(term, "command"):
            term.command[:] = command.to(device=term.command.device, dtype=term.command.dtype)
            return
```

If `OnPolicyRunner` construction requires the existing train/play runner cfg rather than `env_cfg.to_dict()`, copy the exact runner cfg pattern from `Go2Pvcnn/scripts/play.py` during implementation and keep this task's tests focused on source contracts.

- [ ] **Step 6: Run static tests and py_compile**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py
```

Expected: PASS and py_compile exit `0`.

- [ ] **Step 7: Commit rollout skeleton**

Run:

```bash
git add Go2Pvcnn/scripts/mpc_policy_eval.py Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py
git commit -m "feat: add mpc policy eval rollout skeleton"
```

---

## Task 4: Tracking Mode Runtime Metrics

**Files:**
- Modify: `Go2Pvcnn/scripts/mpc_policy_eval.py`
- Modify: `Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py`

- [ ] **Step 1: Add tests for tracking aggregation**

Append to `Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py`:

```python
from Go2Pvcnn.scripts.mpc_policy_eval import TrackingRoundAccumulator


def test_tracking_round_accumulator_aggregates_step_metrics() -> None:
    acc = TrackingRoundAccumulator()
    acc.update({"foot_tracking_error_mean_m": 0.1, "foot_tracking_error_p95_m": 0.2, "reference_valid_ratio": 1.0})
    acc.update({"foot_tracking_error_mean_m": 0.3, "foot_tracking_error_p95_m": 0.4, "reference_valid_ratio": 1.0})

    summary = acc.summary()

    assert summary["foot_tracking_error_mean_m"] == pytest.approx(0.2)
    assert summary["foot_tracking_error_p95_m"] == pytest.approx(0.4)
    assert summary["reference_valid_ratio"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py::test_tracking_round_accumulator_aggregates_step_metrics -q
```

Expected: FAIL because `TrackingRoundAccumulator` is missing.

- [ ] **Step 3: Implement tracking accumulator and reference readers**

Add:

```python
@dataclass
class TrackingRoundAccumulator:
    rows: list[dict[str, float]] = field(default_factory=list)

    def update(self, row: dict[str, object]) -> None:
        self.rows.append({k: float(v) for k, v in row.items() if isinstance(v, (int, float))})

    def summary(self) -> dict[str, object]:
        if not self.rows:
            return {
                "foot_tracking_error_mean_m": 0.0,
                "foot_tracking_error_p95_m": 0.0,
                "reference_valid_ratio": 0.0,
            }
        keys = sorted({key for row in self.rows for key in row})
        out: dict[str, object] = {}
        for key in keys:
            values = [row[key] for row in self.rows if key in row]
            if key.endswith("_p95_m"):
                out[key] = max(values)
            else:
                out[key] = float(sum(values) / max(1, len(values)))
        return out


def current_reference_foot_pos_w(env) -> torch.Tensor | None:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    cache = getattr(base, "_trajectory_reference_cache", None)
    if cache is None:
        return None
    foot = getattr(cache, "foot_pos_w", None)
    if foot is None and isinstance(cache, dict):
        foot = cache.get("foot_pos_w")
    if foot is None:
        return None
    tensor = torch.as_tensor(foot)
    if tensor.ndim == 4:
        phase = getattr(getattr(base, "_trajectory_manager", None), "phase_counter", None)
        if phase is None:
            return tensor[:, 0]
        idx = torch.as_tensor(phase, dtype=torch.long, device=tensor.device).clamp(0, tensor.shape[1] - 1)
        return tensor[torch.arange(tensor.shape[0], device=tensor.device), idx]
    if tensor.ndim == 3:
        return tensor
    return None


def current_actual_foot_pos_w(env) -> torch.Tensor:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    robot = base.scene["robot"]
    foot_ids = robot.find_bodies(".*foot")[0]
    foot_ids_t = torch.as_tensor(foot_ids, dtype=torch.long, device=robot.data.body_pos_w.device)
    return robot.data.body_pos_w.index_select(1, foot_ids_t)
```

- [ ] **Step 4: Wire tracking mode into rollout loop**

In `run_eval()`, before each round loop create:

```python
tracking_acc = TrackingRoundAccumulator() if str(args.mode) == "tracking" else None
```

Inside the step loop after `env.step(action)`:

```python
if tracking_acc is not None:
    reference_foot = current_reference_foot_pos_w(env)
    if reference_foot is not None:
        actual_foot = current_actual_foot_pos_w(env)
        row = tracking_foot_metrics(actual_foot, reference_foot.to(device=actual_foot.device))
        row["reference_valid_ratio"] = 1.0
    else:
        row = {"reference_valid_ratio": 0.0}
    tracking_acc.update(row)
    write_jsonl(metrics_path, {"round": round_idx, "step": step, "mode": str(args.mode), **row})
```

At round end:

```python
if tracking_acc is not None:
    round_summary.update(tracking_acc.summary())
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py -q
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py
```

Expected: PASS and py_compile exit `0`.

- [ ] **Step 6: Commit tracking metrics**

Run:

```bash
git add Go2Pvcnn/scripts/mpc_policy_eval.py Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py
git commit -m "feat: collect mpc policy tracking metrics"
```

---

## Task 5: Small Collision Mode Runtime Metrics

**Files:**
- Modify: `Go2Pvcnn/scripts/mpc_policy_eval.py`
- Modify: `Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py`

- [ ] **Step 1: Add aggregate collision test**

Append:

```python
from Go2Pvcnn.scripts.mpc_policy_eval import aggregate_small_collision_rounds


def test_aggregate_small_collision_rounds_uses_env_denominator() -> None:
    summary = aggregate_small_collision_rounds(
        [
            {"collided_env_count": 2, "num_envs": 4},
            {"collided_env_count": 1, "num_envs": 4},
        ]
    )
    assert summary["aggregate_small_collision_env_rate"] == pytest.approx(3 / 8)
    assert summary["round_count"] == 2
    assert summary["total_collided_envs"] == 3
    assert summary["total_env_rounds"] == 8
```

- [ ] **Step 2: Run test and confirm RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py::test_aggregate_small_collision_rounds_uses_env_denominator -q
```

Expected: FAIL because aggregate helper is missing.

- [ ] **Step 3: Implement sensor reader and aggregate helper**

Add:

```python
def semantic_small_force_matrix_w(env) -> tuple[torch.Tensor, tuple[str, ...]]:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    sensor = base.scene.sensors["semantic_contact_small"]
    matrix = torch.as_tensor(sensor.data.force_matrix_w)
    body_names = tuple(getattr(sensor, "body_names", ()) or getattr(sensor.data, "body_names", ()) or ())
    if not body_names:
        body_names = tuple(f"body_{idx}" for idx in range(int(matrix.shape[1])))
    return matrix, body_names


def aggregate_small_collision_rounds(rounds: list[dict[str, object]]) -> dict[str, object]:
    total_collided = sum(int(row.get("collided_env_count", 0)) for row in rounds)
    total_env_rounds = sum(int(row.get("num_envs", 0)) for row in rounds)
    return {
        "round_count": len(rounds),
        "total_collided_envs": total_collided,
        "total_env_rounds": total_env_rounds,
        "aggregate_small_collision_env_rate": float(total_collided / max(1, total_env_rounds)),
    }
```

- [ ] **Step 4: Wire small_collision mode into rollout loop**

Before each round loop:

```python
collision_acc = (
    SmallCollisionRoundAccumulator(
        num_envs=int(args.num_envs),
        threshold=float(args.collision_force_threshold),
        device=torch.device(str(args.device)),
    )
    if str(args.mode) == "small_collision"
    else None
)
```

Inside the step loop after `env.step(action)`:

```python
if collision_acc is not None:
    force_matrix, body_names = semantic_small_force_matrix_w(env)
    collision_acc.update(step=step, force_matrix_w=force_matrix, body_names=body_names)
```

At round end:

```python
if collision_acc is not None:
    round_summary.update(collision_acc.summary())
```

At final summary:

```python
summary = {"rounds": summaries, "round_count": len(summaries)}
if str(args.mode) == "small_collision":
    summary.update(aggregate_small_collision_rounds(summaries))
write_summary(summary_path, summary)
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py -q
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py
```

Expected: PASS and py_compile exit `0`.

- [ ] **Step 6: Commit small collision metrics**

Run:

```bash
git add Go2Pvcnn/scripts/mpc_policy_eval.py Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py
git commit -m "feat: collect small obstacle collision env rates"
```

---

## Task 6: Livestream Command Sync And MPC Foot Markers

**Files:**
- Modify: `Go2Pvcnn/scripts/mpc_policy_eval.py`
- Modify: `Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py`

- [ ] **Step 1: Add static tests for livestream contracts**

Append:

```python
def test_mpc_policy_eval_livestream_syncs_command_and_markers() -> None:
    source = _source()
    assert "sync_command_to_mpc" in source
    assert "sync_command_to_policy" in source
    assert "update_mpc_foot_markers" in source
    assert "VisualizationMarkers" in source
    assert "_trajectory_reference_cache" in source
```

- [ ] **Step 2: Run test and confirm RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py::test_mpc_policy_eval_livestream_syncs_command_and_markers -q
```

Expected: FAIL because these functions are not defined.

- [ ] **Step 3: Split command sync helpers**

Replace `apply_command_to_env()` with:

```python
def sync_command_to_policy(env, command: torch.Tensor) -> None:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    command_manager = getattr(base, "command_manager", None)
    if command_manager is None:
        return
    for name in ("base_velocity", "base_velocity_command", "velocity_command"):
        term = None
        try:
            term = command_manager.get_term(name)
        except Exception:
            term = None
        if term is not None and hasattr(term, "command"):
            term.command[:] = command.to(device=term.command.device, dtype=term.command.dtype)
            return


def sync_command_to_mpc(env, command: torch.Tensor) -> None:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    manager = getattr(base, "_trajectory_manager", None)
    if manager is not None and hasattr(manager, "last_command"):
        manager.last_command = command.detach().clone()
```

Then use:

```python
sync_command_to_policy(env, command)
sync_command_to_mpc(env, command)
```

If the actual manager has no `last_command` field, keep `sync_command_to_mpc()` as a narrow compatibility shim and rely on the command manager as the source consumed by `refresh_from_env()`. Do not fabricate a second command state that changes planner semantics.

- [ ] **Step 4: Add marker overlay helpers**

Add:

```python
def build_mpc_foot_markers():
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sim import SphereCfg

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/T302oMpcPolicyEval/foot_reference",
        markers={
            "foot_ref": SphereCfg(radius=0.025, visual_material=None),
        },
    )
    return VisualizationMarkers(marker_cfg)


def update_mpc_foot_markers(markers, env) -> None:
    reference = current_reference_foot_pos_w(env)
    if reference is None:
        return
    points = reference.reshape(-1, 3).to(dtype=torch.float32)
    markers.visualize(translations=points)
```

Before rollout:

```python
markers = build_mpc_foot_markers() if getattr(args, "livestream", -1) in (1, 2) else None
```

Inside step loop after reference cache refresh:

```python
if markers is not None:
    update_mpc_foot_markers(markers, env)
```

If IsaacLab marker imports differ, adapt to the existing `go2_foostep_planner.py` marker construction pattern and keep function names stable.

- [ ] **Step 5: Run static tests and py_compile**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py
```

Expected: PASS and py_compile exit `0`.

- [ ] **Step 6: Commit livestream overlay slice**

Run:

```bash
git add Go2Pvcnn/scripts/mpc_policy_eval.py Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py
git commit -m "feat: add mpc policy eval livestream markers"
```

---

## Task 7: Real IsaacLab Smoke Tests And Notes

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T302o-mpc-policy-eval-plan.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/YYYY-MM-DD-HHMM-t302o-mpc-policy-eval-smoke.md`

- [ ] **Step 1: Run local/static regression**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py \
  Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_policy_eval_cfgs_enable_reference_without_changing_play -q
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

Expected: PASS and py_compile exit `0`.

- [ ] **Step 2: Run tracking headless smoke**

Use an idle GPU and run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --headless \
  --device cuda:0 \
  --num-envs 4 \
  --num-rounds 1 \
  --max-steps 20 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.4 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/tracking_smoke
```

Expected:

- exit code `0`
- `summary.json` exists
- `rounds.jsonl` contains one row
- `reference_valid_ratio` is present
- `foot_tracking_error_mean_m` is present
- no NaN/Inf in summary

- [ ] **Step 3: Run small_collision headless smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode small_collision \
  --headless \
  --device cuda:0 \
  --num-envs 4 \
  --num-rounds 1 \
  --max-steps 20 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode random \
  --random-command-interval 5 \
  --small-count-per-tile 80 \
  --output-dir logs/mpc_policy_eval/small_collision_smoke
```

Expected:

- exit code `0`
- `summary.json` exists
- `aggregate_small_collision_env_rate` is present
- `collided_env_count` denominator is `num_envs`, not `num_envs * max_steps`
- `semantic_contact_small` force matrix is finite

- [ ] **Step 4: Run livestream smoke if a visual session is needed**

Run with `num_envs=1`:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --livestream 2 \
  --device cuda:0 \
  --num-envs 1 \
  --num-rounds 1 \
  --max-steps 0 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode fixed \
  --command "0.4 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/visual_tracking_smoke
```

Expected:

- app starts WebRTC livestream
- same command path is used for policy and MPC
- MPC foot reference markers appear
- metrics/config files are created

- [ ] **Step 5: Write verification log**

Create `notes/log/YYYY-MM-DD-HHMM-t302o-mpc-policy-eval-smoke.md` with:

```markdown
# T302o MPC Policy Eval Smoke

## Purpose

Verify the new policy evaluation entry, tracking metrics, small collision env-rate metric, and optional livestream overlay.

## Stage

MPC semantic policy evaluation.

## Related Todo

- [T302o](../todo/T302o-mpc-policy-eval-plan.md)

## Git Refs

- Baseline Ref: `f46eab8`
- Candidate Ref: record the output of `git rev-parse --short HEAD` after Task 6.
- Current Work Ref: record the output of `git branch --show-current` and `git status --short` after Task 6.

## Commands

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py \
  Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_policy_eval_cfgs_enable_reference_without_changing_play -q
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --headless \
  --device cuda:0 \
  --num-envs 4 \
  --num-rounds 1 \
  --max-steps 20 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.4 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/tracking_smoke
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode small_collision \
  --headless \
  --device cuda:0 \
  --num-envs 4 \
  --num-rounds 1 \
  --max-steps 20 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode random \
  --random-command-interval 5 \
  --small-count-per-tile 80 \
  --output-dir logs/mpc_policy_eval/small_collision_smoke
```

## Key Metrics

- Static regression:
- Tracking smoke:
- Small collision smoke:
- Livestream smoke:

## Result

Record `Pass` only if the static regression, pycompile, tracking smoke, and small_collision smoke all complete with exit code `0`. If a smoke is blocked by GPU availability, record `Blocked by GPU availability` and include `nvidia-smi` evidence.

## Conclusion

State what is accepted and what remains unverified.
```

- [ ] **Step 6: Update todo/log indexes**

Update:

- `notes/todo.md`
- `notes/todo/T302o-mpc-policy-eval-plan.md`
- `notes/log/index.md`

Record:

- static regression result
- tracking smoke result
- small_collision smoke result
- livestream visual status
- output directory paths

- [ ] **Step 7: Commit notes and final verification**

Run:

```bash
git add notes/todo.md notes/todo/T302o-mpc-policy-eval-plan.md notes/log/index.md notes/log/
git commit -m "docs: record t302o mpc policy eval verification"
```

## Closed Children Archive

- No closed children yet.

## Related Logs

- [../log/2026-06-05-t302o-task2-metric-helpers.md](../log/2026-06-05-t302o-task2-metric-helpers.md)
- [../log/2026-06-05-t302o-task1-static-contracts.md](../log/2026-06-05-t302o-task1-static-contracts.md)
- [../log/2026-06-05-t302o-mpc-policy-eval-plan.md](../log/2026-06-05-t302o-mpc-policy-eval-plan.md)

## Git Refs

- Last Feature Commit: `33cb1f8` for Task 2 metric helpers; import isolation review fix at `e84a78c`
- Last Verified Commit: `e84a78c` for Task 2
- Current Work Ref: `costmap-teacher-ablation` after Task 2 verification follow-up
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/extension/mdp/semantic_contact_rewards.py](../../Go2Pvcnn/extension/mdp/semantic_contact_rewards.py)
  - [../../Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py)
  - [../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py)

## Next Step

- Implement Task 3 headless rollout skeleton and output files.

## Node Details

### T302o.1 Single Python Evaluation Entry

- why-created: user wants a Python-only script under `Go2Pvcnn/scripts/` to evaluate loaded policy checkpoints against MPC reference tracking and dense-small flat collision rate.
- accepted design:
  - one script: `Go2Pvcnn/scripts/mpc_policy_eval.py`;
  - two modes: `tracking` and `small_collision`;
  - `num_rounds` controls repeat count;
  - `max_steps` completes one round;
  - livestream uses the same command for policy and MPC and visualizes MPC foot reference markers;
  - small collision main metric is collided envs divided by env count per round.
- main risks:
  - `play.py` no-MPC behavior must not regress;
  - RSL-RL runner construction may need exact reuse from `scripts/play.py`;
  - command manager term name may differ at runtime;
  - marker construction must match the installed IsaacLab API.
