# Nominal-Only Flat Root Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the nominal root reference integrate the commanded planar velocity across the full horizon and provide a viewer-only nominal path that skips LQ/QP/line search.

**Architecture:** Keep the production RTI path as the default. Change only the nominal root-reference defaults, and add an explicit runtime `nominal_only` switch whose update is the nominal trajectory with alpha zero and zero QP direction. The viewer opts into this mode through a CLI flag.

**Tech Stack:** Python, PyTorch, pytest, Isaac Lab/Isaac Sim, TurboVNC.

## Global Constraints

- Work on `work/joint-mpc-kinematic`; do not create a branch.
- Keep the existing post-IK joint clamp unchanged in this first experiment.
- Do not weaken collision, joint, stance, touchdown, or nominal-safe checks.
- Nominal-only mode must be opt-in and must not change the default production RTI path.
- Real validation must use Isaac Lab on `DISPLAY=:1`, not a tensor-only rollout.

---

### Task 1: Continuous Nominal Root Reference

**Files:**
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- Test: `Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py`

**Interfaces:**
- Consumes: `JointMpcRtiNominalCfg.command_scale`, `step_reference_scale`.
- Produces: default planar root displacement `command_xy * dt` on every horizon edge.

- [ ] Change the root-profile test to require scale `1.0` on near and far edges.
- [ ] Run the focused test and confirm it fails on the old `0.85/0.0` defaults.
- [ ] Set both nominal scales to `1.0` without changing the root integration formula.
- [ ] Run `test_nominal.py` and confirm the root-profile contract passes.

### Task 2: Opt-In Nominal-Only Runtime

**Files:**
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py`
- Test: `Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py`

**Interfaces:**
- Consumes: `cfg.runtime.nominal_only: bool`.
- Produces: `JointMpcRtiTrajectory.state == nominal.state`, `selected_alpha == 0`, zero QP direction, and no LQ/QP/line-search call.

- [ ] Add a planner test that enables nominal-only mode and fails if the RTI update is called.
- [ ] Run the focused test and confirm the current planner still calls RTI.
- [ ] Add `runtime.nominal_only=False` and construct a nominal update directly when enabled.
- [ ] Add `--joint-mpc-nominal-only` and map it to the runtime flag in the viewer config.
- [ ] Verify nominal-only and default RTI tests independently.

### Task 3: Real Isaac Flat Speed Matrix

**Files:**
- Create outside repository: `/tmp/joint_mpc_nominal_root_sweep.py`
- Create evidence: `notes/log/2026-07-26-joint-mpc-nominal-only-root-reference-flat.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo/T302v-joint-mpc-rti-gpu.md`
- Modify: `notes/todo.md`

**Interfaces:**
- Consumes: real Isaac measured root/joint/foot state on flat terrain.
- Produces: before/after metrics for nominal safety, achieved `vx/vy/vyaw`, root-event prediction, touchdown/root body-frame geometry, stance slip, touchdown error, joint margin/step, and selector failure reasons.

- [ ] Run focused CPU tests before Isaac.
- [ ] Run non-headless Isaac on `DISPLAY=:1` for multiple isolated `vx`, `vy`, and `vyaw` commands.
- [ ] Compare against `/tmp/joint_mpc_nominal_isaac_probe.json` and record regressions as well as improvements.
- [ ] Start the nominal-only viewer and report the exact TurboVNC command and process status.
- [ ] Update todo/log notes with commands, metrics, conclusion, git refs, and remaining clamp work.

