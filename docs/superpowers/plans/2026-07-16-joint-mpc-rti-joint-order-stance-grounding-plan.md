# Joint MPC RTI Joint Order And Stance Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the Isaac/planner joint-order boundary and make every scheduled stance foot remain grounded across zero, directional, magnitude-varied, yaw, and mixed velocity commands.

**Architecture:** Centralize joint ordering in one integration helper used by both the Isaac adapter and viewer. Build per-contact-segment world XY anchors and feed the existing stance XY/Z residuals consistently into the RTI LQ approximation and merit evaluation, without changing fixed trot or projecting the published trajectory.

**Tech Stack:** Python, PyTorch, torch.compile/CUDA Graph-compatible fixed-shape tensors, IsaacLab, pytest.

---

### Task 1: Joint-order contract

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/integration/joint_order.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/integration/isaaclab_adapter.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py`

- [ ] Add a failing test with Isaac order `(FL/FR/RL/RR hip, thigh, calf)` and distinct position/velocity values; assert `state_from_env()` returns leg-grouped planner order.
- [ ] Run the focused test and confirm the old direct tensor read fails.
- [ ] Add the shared normalized-name permutation helper and use it for both adapter input fields and viewer input/output conversion.
- [ ] Run focused backend/viewer tests and commit the boundary fix.

### Task 2: Stance-anchor residual contract

**Files:**
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/losses/contact.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`
- Test: `Go2Pvcnn/tests/joint_mpc_rti/test_losses.py`

- [ ] Add a failing test showing `stance_xy_lock` is zero for a stationary world anchor and increases when a contact foot moves away, including a swing->stance anchor reset.
- [ ] Add a fixed-shape helper that propagates one XY anchor per contact segment from measured/nominal feet.
- [ ] Pass anchors into `stance_losses()` so merit evaluation uses anchor tracking while retaining slip velocity diagnostics.
- [ ] Run focused loss tests.

### Task 3: RTI stance linearization

**Files:**
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/config.py` only if measured tuning requires existing stance weights to change.
- Test: `Go2Pvcnn/tests/joint_mpc_rti/test_behavior.py`

- [ ] Add failing rolling tests that execute only x1 for repeated cycles over zero, forward/backward, lateral, slow/fast, yaw, and mixed commands.
- [ ] Assert zero-command root XY/yaw drift remains within tolerance and every scheduled stance foot gap is `<=0.01m` on flat terrain.
- [ ] Add stance XY anchor and ground-Z Jacobian residuals to the LQ joint gradient/Hessian approximation.
- [ ] Tune only existing stance weights if required by the failing metrics; do not add a hard gate or projection.
- [ ] Run the complete pure-tensor behavior and joint suites.

### Task 4: Real Isaac viewer acceptance

**Files:**
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_viewer_reproduction_probe.py`
- Create: `notes/log/2026-07-16-joint-mpc-rti-viewer-grounding-fix.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo.md`
- Modify: `notes/todo/T302v-joint-mpc-rti-gpu.md`

- [ ] Expand the real probe command matrix to zero, directional, magnitude-varied, yaw, and mixed commands.
- [ ] Run it in `env_isaacsim` and record joint-order error, root drift, stance gap, joint step, foot step, and viewer-plan agreement.
- [ ] Run the full joint suite, old MPC compatibility subset, pycompile, and `git diff --check`.
- [ ] Update notes with exact commands/metrics and commit only scoped files on `joint_mpc`.
