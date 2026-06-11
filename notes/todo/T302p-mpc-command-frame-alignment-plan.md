# T302p MPC Command Frame Alignment Implementation Plan

> **For agentic workers:** Execute this plan inline task-by-task unless the user explicitly asks for subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align training, viewer, eval, and MPC runtime so external velocity commands are root-yaw/body-frame `[vx_body, vy_body, yaw_rate]`, while MPC world geometry uses `R(root_yaw) @ command_body[:2]`.

**Architecture:** Keep the public command contract unchanged at the IsaacLab command-manager boundary. Move the body-to-world direction conversion into MPC geometry helpers, remove the viewer's pre-rotation compatibility path, and add flat all-direction direction metrics as hard acceptance guards. This work changes coordinate interpretation only; it must not add losses, remove losses, change loss weights, or add hard trajectory projection.

**Tech Stack:** PyTorch tensor helpers under `Go2Pvcnn/extension/batch_mpc_planner`, existing IsaacLab MPC manager/cache, `Go2Pvcnn/scripts/mpc_policy_eval.py`, pytest static/unit tests, real IsaacLab smoke in `env_isaacsim` on GPU 0.

---

## Source Spec

- Design: [../../docs/superpowers/specs/2026-06-06-mpc-command-frame-alignment-design.html](../../docs/superpowers/specs/2026-06-06-mpc-command-frame-alignment-design.html)
- Reproduction log: [../log/2026-06-06-1633-t302o-flat-forward-mpc-left-bias-reproduction.md](../log/2026-06-06-1633-t302o-flat-forward-mpc-left-bias-reproduction.md)
- Timebase diagnostic: [../log/2026-06-06-1616-t302o-foot-trajectory-timebase-probe.md](../log/2026-06-06-1616-t302o-foot-trajectory-timebase-probe.md)
- Low-small design compatibility: [../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html](../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html)
- MPC RL runtime compatibility: [../../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html](../../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html)

## Current State

- T302p command-frame implementation is in the local working tree:
  - `command_frame_axes()` now rotates body/root-yaw command XY into world XY.
  - MPC world-geometry heading uses in planner, semantic policy, and terrain-clearance helpers now consume the root-yaw world heading.
  - Viewer planning no longer pre-rotates the body-frame command before `plan_segment()`.
  - Eval JSONL/summary now records command-source equality and flat planned-direction diagnostics.
- T302p focused local verification passed:
  - `pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_viewer_reset.py Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q`
  - result: `184 passed, 1 warning`
  - `python -m py_compile ...` for touched planner/viewer/eval files exited `0`
  - `git diff --check` exited `0`
- T302p real IsaacLab short smoke on GPU0/env_isaacsim passed for fixed forward command, but it was only `5` steps:
  - `command_body_match_max_abs_error = 0.0`
  - `planned_root_direction_cosine = 0.9988572597503662`
  - `planned_root_lateral_ratio = 0.04779195412993431`
  - `planned_per_leg_lateral_ratio_xy = [0.03437824174761772, 0.14927777647972107, 0.0919436365365982, 0.023816389963030815]`
  - one leg exceeded the preferred `0.10` lateral-ratio threshold, so full per-leg acceptance remains open.
- Full eight-command flat acceptance and low-small semantic compatibility regression have now been rerun and failed behavioral gates:
  - eight-direction tracking all exited `0` and command-source equality passed with `command_body_match_max_abs_error=0.0`
  - root direction failed for `forward`, `left`, and `right`
  - moving-leg direction failed for all eight commands under the current hard threshold
  - low-small GPU3 regression exited `0` but failed `fk_semantic_collision_count == 0`, `fk_semantic_collision_rate == 0`, and max crossing FK error `<= 0.08m`
- Low-small semantic compatibility was fixed locally on 2026-06-07 without adding a new loss:
  - existing `ik_fk_residual.weight` now scales existing `parametric_trajectory_fk_consistency_loss()`
  - existing `kinematics.weight` and `joint_limit_margin_rad` now feed `parametric_joint_limit`
  - existing FK collision aggregation now uses mean + worst so sparse foot collisions are not diluted by long horizons
  - GPU0/env_isaacsim default `parametric_v1` low-small rerun passed `max_fk_semantic_collision_count=0` and max crossing FK error `0.04201m`
- Flat-left tracking after the low-small fix still reports `planned_root_lateral_ratio=0.2169`; T302p remains open for flat direction/root and moving-leg metrics.
- Direction-loss wiring continuation on 2026-06-07 fixed the flat-left root direction in real smoke and kept low-small compatibility clean:
  - existing `progress.weight/min_progress_m` now feeds `parametric_command_progress`
  - existing `swing_direction_loss()` now feeds `parametric_swing_direction`
  - tracking eval now synchronizes zero-obstacle curriculum to terrain cfg so flat tracking is actually obstacle-free
  - flat-left smoke root lateral ratio improved to `0.0200`
  - direction breakdown probe root lateral ratio reached `0.00315` with active direction losses
  - low-small GPU0 regression still has `max_fk_semantic_collision_count=0` and max crossing FK error `0.04201m`
  - strict per-leg whole-cache endpoint metric remains open: two middle legs still show high lateral ratios under the current metric, likely because the metric uses each FK foot's entire segment endpoint displacement rather than swing-window motion.
- T302o eval and livestream path are implemented and smoke-verified.
- T302o timebase probe shows current eval is not async MPC-vs-policy execution; `refresh_from_env()` runs synchronously during post-step reward computation.
- T302o flat-forward probe reproduced the lateral bias on a flat semantic-free run:
  - robot yaw: `16.05deg`
  - requested command: `[1.0, 0.0, 0.0]`
  - default MPC nominal forward: world `[1.0, 0.0]`
  - default body-left drift: `-0.0937m`
  - manually yaw-rotated command body-left drift: `-0.0044m`
- The likely cause is command-frame mismatch: some MPC world-geometry code treats body command XY as world XY.

## File Structure

- Modify `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`
  - Owns the unified command-to-world axes helper used by parametric nominal/decode geometry.
- Modify `Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py`
  - Reuses the unified helper for semantic corridor geometry without double rotation.
- Modify `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
  - Audits direct `cmd[:, :2]` heading use and converts world-geometry calculations to the unified helper.
- Modify `Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py`
  - Converts terrain/semantic geometry heading uses to the unified helper while keeping body-frame command checks unchanged.
- Modify `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - Removes the viewer pre-rotation path before MPC planning; keeps body-frame arrow display semantics.
- Modify `Go2Pvcnn/scripts/mpc_policy_eval.py`
  - Adds flat all-direction command-source and direction diagnostics without changing the public CLI contract.
- Modify tests:
  - `Go2Pvcnn/tests/test_batch_mpc_parametric.py`
  - `Go2Pvcnn/tests/test_batch_mpc_backend.py`
  - `Go2Pvcnn/tests/test_viewer_reset.py`
  - `Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py`
- Update notes/logs:
  - `notes/todo.md`
  - this branch page
  - `notes/log/index.md`
  - per-task logs under `notes/log/`

## Global Constraints

- No `.sh` launcher or shell wrapper.
- No new loss additions, loss deletions, optimizer changes, hard projection, or postprocess snapping.
- Latest user override allows fixing existing weight/loss coordinate or wiring issues, but not adding a new loss to optimize the metrics.
- External command remains `command_body = [vx_body, vy_body, yaw_rate]` in root-yaw/body horizontal frame.
- MPC manager keeps passing command-manager output unchanged into `plan_segment()`.
- Viewer/eval/training must feed the same body-frame command into policy and MPC.
- World-frame terrain, semantic corridor, root path, foot path, progress, and touchdown geometry use world axes derived from root yaw.
- Body-frame command checks remain body-frame: zero linear command, speed magnitude, `vy_body`, mixed command, and yaw-rate conditions.
- Do not regress low-small crossing and semantic avoidance metrics from T302k/T302l.
- Do not claim per-leg acceptance until the metric contract is resolved: current evidence says planner direction losses are active, but the eval per-leg endpoint metric is not aligned with gait-phase swing motion.

---

## Task 1: Static Contract Guards

**Files:**
- Modify: `Go2Pvcnn/tests/test_batch_mpc_parametric.py`
- Modify: `Go2Pvcnn/tests/test_viewer_reset.py`
- Modify: `Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py`

- [x] **Step 1: Add RED tests for root-yaw command axes**

Add cases proving `command_frame_axes(command_body, root_yaw, ...)` rotates linear command XY by root yaw:

```text
yaw=0, command=[1,0,0]      -> forward_w=[1,0]
yaw=pi/2, command=[1,0,0]   -> forward_w=[0,1]
yaw=pi/2, command=[0,1,0]   -> forward_w=[-1,0]
yaw=pi, command=[-1,0,0]    -> forward_w=[1,0]
yaw=pi/4, command=[0.7,0.7,0] -> forward_w approximately world +Y
zero linear, yaw=pi/2       -> fallback forward_w=[0,1], linear_active=False
```

- [x] **Step 2: Add RED viewer static guard**

Guard that `_plan_viewer_trajectory()` passes the root-frame/body command directly to MPC and no longer calls `_viewer_mpc_world_command_from_root_frame()` in the planning path.

- [x] **Step 3: Add RED eval static guard**

Guard that `mpc_policy_eval.py` records or exposes `requested_command_body`, `policy_command_body`, and `mpc_input_command_body` diagnostics, and does not rotate CLI command before writing the command manager.

- [x] **Step 4: Run RED commands**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py::test_command_frame_axes_rotates_body_command_by_root_yaw -q
pytest Go2Pvcnn/tests/test_viewer_reset.py::test_viewer_mpc_planning_keeps_body_frame_command -q
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py::test_eval_records_body_command_source_diagnostics -q
```

Expected: fail before implementation because current helper ignores root yaw for linear command and viewer still pre-rotates.

## Task 2: Unified MPC Body-To-World Axes Helper

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`
- Modify: `Go2Pvcnn/tests/test_batch_mpc_parametric.py`

- [x] **Step 1: Implement `command_frame_axes()` body semantics**

Keep the existing function name unless the implementation proves a rename is cleaner. Required behavior:

```text
input:
  command_body: Tensor[N, 3]
  root_yaw: Tensor[N]

output:
  forward_w: Tensor[N, 2]
  left_w: Tensor[N, 2]
  linear_active: Tensor[N]

linear active:
  command_dir_body = normalize(command_body[:, :2])
  forward_w = R(root_yaw) @ command_dir_body

linear inactive:
  forward_w = [cos(root_yaw), sin(root_yaw)]

always:
  left_w = [-forward_w.y, forward_w.x]
```

- [x] **Step 2: Keep body-frame command checks outside the helper**

Do not make the helper decide mixed/yaw/sideways semantics. Callers that need `speed`, `vy_body`, or `yaw_rate` continue reading `command_body`.

- [x] **Step 3: Run focused tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Expected: all parametric helper and nominal tests pass.

## Task 3: Convert MPC World-Geometry Heading Uses

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py`
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [x] **Step 1: Audit all direct heading calculations**

Classify each `cmd[:, :2]` use:

```text
world geometry -> replace with command_frame_axes(command_body, root_yaw)
body intent check -> keep command_body
pure speed magnitude -> keep command_body norm
```

Known locations to inspect:

```text
semantic_policy.py around command shaping and semantic corridor logic
planner.py around _command_farthest_touchdown_positions
planner.py around _structured_low_small_touchdown_positions
planner.py around command progress and sampled loss heading blocks
terrain_clearance.py around touchdown keepout, swing clearance, FK collision, and crossing helpers
```

- [x] **Step 2: Convert semantic/world geometry only**

Use `forward_w` and `left_w` anywhere the compared positions are world-frame root/foot/terrain/semantic positions.

- [x] **Step 3: Preserve body checks**

Leave checks such as `speed = norm(command_body[:, :2])`, pure yaw fallback, and `vy_body` branching in body-frame semantics.

- [x] **Step 4: Add regression tests with nonzero root yaw**

Add a flat/no-semantic backend case with `root_yaw != 0` and command `[1,0,0]`. The root trajectory direction should align with `R(root_yaw) @ [1,0]`, not world `[1,0]`.

- [x] **Step 5: Run focused backend tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Expected: pass locally; no loss key or weight changes appear in diff.

## Task 4: Viewer Boundary Migration

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Modify: `Go2Pvcnn/tests/test_viewer_reset.py`

- [x] **Step 1: Remove planning-path pre-rotation**

In `_plan_viewer_trajectory()`, pass the viewer command directly into MPC planning. Do not call `_viewer_mpc_world_command_from_root_frame()` before `plan_segment()`.

- [x] **Step 2: Keep visual command arrow semantics**

Keep arrow/display code as body-frame UI semantics:

```text
display heading = root_yaw + atan2(vy_body, vx_body)
```

- [x] **Step 3: Update tests**

Replace old helper expectations with the new contract: viewer command is body-frame at the MPC boundary.

- [x] **Step 4: Run viewer tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py -q
```

Expected: pass; viewer no longer double-rotates after MPC internal fix.

## Task 5: Eval Flat Direction Diagnostics

**Files:**
- Modify: `Go2Pvcnn/scripts/mpc_policy_eval.py`
- Modify: `Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py`

- [x] **Step 1: Add command-source diagnostics**

For flat direction runs, record these fields in JSON output:

```text
requested_command_body
policy_command_body
mpc_input_command_body
command_body_match_max_abs_error
```

Hard contract:

```text
max_abs(policy_command_body - requested_command_body) <= 1e-6
max_abs(mpc_input_command_body - requested_command_body) <= 1e-6
```

- [x] **Step 2: Add direction metrics**

For each flat no-obstacle command, compute:

```text
root_direction_cosine
root_lateral_ratio
per_leg_direction_cosine_xy
per_leg_lateral_ratio_xy
insufficient_motion
insufficient_leg_motion
semantic_nonzero_count
```

Hard thresholds:

```text
root move norm >= 0.05m:
  root_direction_cosine >= 0.98
  root_lateral_ratio <= 0.10

leg XY step norm >= 0.03m:
  leg_direction_cosine >= 0.98
  leg_lateral_ratio <= 0.10

semantic_nonzero_count == 0
```

Z is diagnostic only.

- [ ] **Step 3: Cover full 2D command directions**

Use this flat no-obstacle command set:

```text
[+1.0,  0.0, 0.0]
[-1.0,  0.0, 0.0]
[ 0.0, +1.0, 0.0]
[ 0.0, -1.0, 0.0]
[+0.7, +0.7, 0.0]
[+0.7, -0.7, 0.0]
[-0.7, +0.7, 0.0]
[-0.7, -0.7, 0.0]
```

- [x] **Step 4: Run static eval tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
```

Expected: pass and no CLI shell wrapper introduced.

## Task 6: Local Regression And Loss-Contract Diff Check

**Files:**
- Modify only test expectations needed by Tasks 1-5.

- [x] **Step 1: Run focused regression**

Run:

```bash
pytest \
  Go2Pvcnn/tests/test_batch_mpc_parametric.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_viewer_reset.py \
  Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py \
  -q
```

Expected: pass.

- [x] **Step 2: Run pycompile**

Run:

```bash
python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/parametric.py \
  Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/scripts/mpc_policy_eval.py
```

Expected: exit `0`.

- [x] **Step 3: Check no loss/weight edits**

Inspect diff for:

```text
no new sampled/final loss keys
no deleted sampled/final loss keys
no changed loss weights
no optimizer step/hyperparameter change
no hard projection or snapping added
```

Expected: only coordinate-frame interpretation and diagnostics changed.

## Task 7: Real IsaacLab Flat All-Direction Smoke

**Files:**
- Runtime verification only; write log under `notes/log/`.

- [x] **Step 1: Run flat tracking smoke on GPU 0**

Use `env_isaacsim` and card 0. Example fixed-forward command:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --headless \
  --device cuda:0 \
  --num-envs 1 \
  --num-rounds 1 \
  --max-steps 200 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode fixed \
  --command "1.0 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/t302p_flat_forward_smoke
```

- [x] **Step 2: Run or script the eight-command flat set**

Each command must produce command-source diagnostics and direction metrics. A run is accepted only when the hard direction metrics pass or the sample is explicitly marked `insufficient_motion` / `insufficient_leg_motion` according to thresholds.

Result on 2026-06-06: executed, but acceptance failed. See [../log/2026-06-06-2317-t302p-real-acceptance-failures.md](../log/2026-06-06-2317-t302p-real-acceptance-failures.md).

- [ ] **Step 3: Optional livestream visual check**

Use livestream only for visual inspection after headless metrics pass:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --livestream 2 \
  --device cuda:0 \
  --num-envs 1 \
  --num-rounds 1 \
  --max-steps 300 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode fixed \
  --command "1.0 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/t302p_livestream_forward_check
```

Expected: camera follows env 0, policy command and MPC command match, and MPC foot markers move in the root-yaw command direction rather than world-X.

## Task 8: Semantic Compatibility Regression

**Files:**
- Runtime verification only; write log under `notes/log/`.

- [x] **Step 1: Rerun low-small hard regression**

Use the existing T302k/T302l low-small verification route before claiming compatibility. Required acceptance:

```text
fk_semantic_collision_count == 0
fk_semantic_collision_rate == 0
covered crossing rows > 0
planned_vs_fk_foot_error_crossing_leg_max_m <= 0.08m
```

Result on 2026-06-06: executed, but acceptance failed. `lateral_v050` produced `fk_semantic_collision_count=21`; `mixed_yaw_v050` produced `fk_semantic_collision_count=1` and crossing FK error `0.13923m`. See [../log/2026-06-06-2317-t302p-real-acceptance-failures.md](../log/2026-06-06-2317-t302p-real-acceptance-failures.md).

Follow-up on 2026-06-07: after wiring existing FK/kinematics losses and changing existing FK collision aggregation from mean-only to mean + worst, default `parametric_v1` passed the low-small hard gates on GPU0/env_isaacsim:

```text
max_fk_semantic_collision_count = 0
max_fk_semantic_collision_rate = 0.0
max planned_vs_fk_foot_error_crossing_leg_max_m = 0.04200904071331024
```

See [../log/2026-06-07-1104-t302p-low-small-fk-loss-wiring.md](../log/2026-06-07-1104-t302p-low-small-fk-loss-wiring.md).

- [ ] **Step 2: Preserve high-small/large avoidance**

Check existing semantic obstacle rows still choose avoidance/crossing behavior according to 2026-05-28 and 2026-05-30 designs. Record any row with changed behavior as a concrete follow-up node, not a vague anomaly.

- [ ] **Step 3: Preserve MPC RL runtime contracts**

Confirm the change does not alter:

```text
reference_trajectory_horizon = 25
reference_replan_interval_steps = 25
MPC participation selector semantics
ReferenceTrajectoryCache consumer contract
semantic_contact_small / semantic_contact_large reward route
```

## Task 9: Notes And Handoff

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T302p-mpc-command-frame-alignment-plan.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/<timestamp>-t302p-*.md`

- [ ] **Step 1: Log each verification pass**

Create one log per distinct local or real verification pass. Include:

```text
purpose
stage
related todo
command/procedure
input conditions
key metrics
result
conclusion
follow-up
git refs
```

- [ ] **Step 2: Update dashboard**

Keep `notes/todo.md` as a dashboard:

```text
Current focus -> T302p
Active branch page -> T302p
Active code surface -> batch_mpc_planner, viewer, eval script, focused tests
Open leaves -> T302p.1 active
Recent logs -> latest T302p verification
```

- [ ] **Step 3: Final report**

Report:

```text
which command contract changed
which files changed
which tests passed
which real IsaacLab commands ran
whether low-small semantic compatibility was verified
what remains unverified
```

## Acceptance Summary

- [x] External command contract is body/root-yaw frame at viewer/eval/MPC boundaries covered by tests.
- [x] MPC world geometry uses root-yaw rotated command direction in audited heading paths.
- [x] Viewer no longer pre-rotates command before MPC.
- [x] Eval records policy/MPC command-source equality.
- [ ] Flat no-obstacle all-direction root XY direction passes.
- [ ] Moving-leg XY direction passes for legs with enough motion.
- [x] Z and distance magnitude remain diagnostic only.
- [x] No new loss, optimizer change, projection, or snapping change.
- [x] Low-small crossing and FK semantic collision hard gates pass after fixing existing FK/kinematics loss wiring.
- [ ] High-small/large semantic obstacle behavior non-regression remains to verify if further planner loss changes are made.
- [x] Real IsaacLab GPU0/env_isaacsim short smoke evidence is logged.

## Related Logs

- [../log/2026-06-09-2026-current-mpc-ppo-html-overview.md](../log/2026-06-09-2026-current-mpc-ppo-html-overview.md)
- [../log/2026-06-06-1858-t302p-command-frame-implementation.md](../log/2026-06-06-1858-t302p-command-frame-implementation.md)
- [../log/2026-06-06-2317-t302p-real-acceptance-failures.md](../log/2026-06-06-2317-t302p-real-acceptance-failures.md)
- [../log/2026-06-07-1104-t302p-low-small-fk-loss-wiring.md](../log/2026-06-07-1104-t302p-low-small-fk-loss-wiring.md)
- [../log/2026-06-06-1633-t302o-flat-forward-mpc-left-bias-reproduction.md](../log/2026-06-06-1633-t302o-flat-forward-mpc-left-bias-reproduction.md)
- [../log/2026-06-06-1616-t302o-foot-trajectory-timebase-probe.md](../log/2026-06-06-1616-t302o-foot-trajectory-timebase-probe.md)

## Git Refs

- Current Work Ref: local dirty worktree after T302o diagnostics and T302p planning
- Last Verified Commit: `996ce1f` for T302o eval smoke baseline
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py`
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - `Go2Pvcnn/scripts/mpc_policy_eval.py`

## Next Step

Continue systematic debugging from the failing flat-direction real rows before changing more code:

- Flat direction: inspect `left` under `logs/mpc_policy_eval/t302p_eight_direction_120step/left/.../metrics.jsonl`.
- Latest low-small regression now passes; keep it as a regression guard.
