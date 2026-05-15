# MPC Continuous Swing Window Redesign

## 1. Purpose

This design replaces the current dense MPC gait/contact logic with a continuous swing-window parameterization and terrain-aware losses. The goal is to remove planner-owned foothold memory and output-side foot grounding, then make the optimizer solve for physically consistent root, foot, contact, touchdown, and IK-feasible trajectories.

The active target remains:

```text
Go2Pvcnn/extension/batch_mpc_planner/
```

This is a redesign of the existing MPC backend, not a new parallel backend.

## 2. Goals

- Build nominal trajectories from the latest IsaacLab root, foot, joint, and scanner state at each replan.
- Treat keyboard/user command velocity as root/body-frame velocity, not world-frame velocity.
- Keep nominal foot positions in world coordinates so terrain and semantic scanner losses can sample directly at `foot_xy`.
- Use a diagonal trot prior, but let MPC decide each leg's swing start and end time.
- Keep nominal diagonal first-pair randomization as an initialization/prior source only; optimized `swing_center` losses decide which diagonal pair actually swings first.
- Guarantee each leg has one continuous swing window per horizon.
- Use only semantic scanner outputs for height and semantic obstacle losses:
  - `height_map`
  - `semantic_map`
- Avoid reading true obstacle object positions, USD prim locations, or other privileged obstacle state.
- Delete `MpcFootholdMemory` and associated manager-side foot memory.
- Delete output-side `_ground_contact_feet_to_terrain`; terrain contact must be enforced by loss.
- Keep all hot-path work GPU vectorized. No NumPy, CPU packages, `.cpu()`, `.item()` control flow, or Python loops over batch/time/legs in the MPC runtime path.

## 3. Non-Goals

- Do not reintroduce legacy/together mode tables or candidate search.
- Do not hard-fix all contact timing to the nominal phase schedule.
- Do not plan obstacle avoidance from true small/large obstacle object positions.
- Do not estimate root yaw from feet. Yaw remains command/root-trajectory driven.
- Do not solve this with post-processing that silently changes optimized foot z after loss evaluation.

## 4. Fixed Runtime Defaults

The new nominal builder uses these MPC gait defaults:

```text
dt = 0.02
horizon_steps = 25
horizon_time = dt * horizon_steps = 0.5
step_freq = 1.0 / horizon_time = 2.0
duty_factor = 0.5
phase_offsets = [0.0, 0.5, 0.5, 0.0]  # FL, FR, RL, RR
swing_height = 0.10
step_gain = 0.5
yaw_gain = 0.5
```

The config layer may expose these as overrides, but the design baseline is fixed to the values above.

## 5. Nominal Construction

### 5.1 Inputs

Each replan reads current IsaacLab state:

```text
root_pos0[B,3]
root_rpy0[B,3]
foot_pos0_w[B,4,3]
joint_pos0[B,12]
cmd_body[B,3] = [Vx, Vy, Vyaw]
terrain.height_map[B,H,W]
terrain.semantic_map[B,H,W] or None
```

`cmd_body` is in the current root/body coordinate frame.

### 5.2 Vectorized Root Nominal

Root nominal must be computed by GPU tensor operations, not by a Python `for i in range(T)` loop.

For interval frames `k = 0..T-2`:

```text
yaw_interval[k] = yaw0 + k * dt * Vyaw
v_world[k] = rotate_by_yaw(cmd_body.xy, yaw_interval[k])
delta_xy[k] = v_world[k] * dt
root_xy[0] = root_xy0
root_xy[1:] = root_xy0 + cumsum(delta_xy, dim=time)
root_z[:] = root_z0
root_yaw[t] = yaw0 + t * dt * Vyaw
```

Roll and pitch nominal start from current `root_rpy0[:2]`. They are later shaped by support-plane loss over every frame.

### 5.3 Diagonal Contact Prior With Random First Pair

Nominal uses diagonal trot groups:

```text
group A = FL / RR
group B = FR / RL
```

Base phase offsets are:

```text
[0.0, 0.5, 0.5, 0.0]
```

To avoid always making the nominal prior start from the same pair, each replan/env samples a GPU-side phase flip:

```text
phase_flip[B] in {0.0, 0.5}
phase[t, leg] = (t / T + phase_offsets[leg] + phase_flip) % 1.0
```

The phase prior initializes swing timing only. It is not a hard schedule and must not decide the final swing order by itself. The actual leading diagonal pair comes from optimized `swing_center` after loss evaluation. In other words, random `phase_flip` supplies diversity for nominal initialization, while `swing_center` optimization can move either diagonal pair earlier when current foot geometry, command direction, terrain, or IK feasibility makes that pair more urgent.

### 5.4 Swing Window Prior

The optimizer uses one continuous swing window per leg:

```text
swing_center[B,4]
swing_width[B,4]
swing_start = swing_center - 0.5 * swing_width
swing_end = swing_center + 0.5 * swing_width
```

Nominal initializes `center/width` from the diagonal trot prior. MPC can freely shift start/end and change width, while soft losses keep diagonal group behavior reasonable.

The center parameterization must leave enough range for the optimizer to swap the leading diagonal pair relative to the random nominal prior. A small local-only center residual is not sufficient here, because command switches may require the pair that nominal initialized second to become first.

### 5.5 World-Frame Foot Nominal

`nominal["foot_pos"]` is always world-frame `[B,T,4,3]`.

Stance nominal:

```text
foot_nom_w[t, leg] = stance_anchor_w[leg]
```

At replan start, anchors come from current IsaacLab foot positions:

```text
stance_anchor_w = foot_pos0_w
```

No `MpcFootholdMemory` is used.

### 5.6 Swing Touchdown Target

Touchdown target must be computed in the root coordinate frame corresponding to that leg's swing timing, not only in the root frame at `t=0`.

For each env and leg:

```text
swing_start_idx[B,4]
touchdown_idx[B,4]
root_start = gather(root_nominal, swing_start_idx)
yaw_start = gather(root_yaw_nominal, swing_start_idx)
foot_start_w = gather(foot_nominal_or_anchor, swing_start_idx)

foot_start_body =
    rotate_world_to_body(foot_start_w - root_start, yaw_start)

step_bias_body =
    step_gain * horizon_time * [Vx, Vy]

yaw_bias_body =
    yaw_gain * Vyaw * horizon_time * [-foot_start_body.y, foot_start_body.x]

target_body_xy =
    foot_start_body.xy + step_bias_body + yaw_bias_body

root_td = gather(root_nominal, touchdown_idx)
yaw_td = gather(root_yaw_nominal, touchdown_idx)

target_world_xy =
    root_td.xy + rotate_body_to_world(target_body_xy, yaw_td)
```

This preserves whether each leg is currently forward or rearward relative to the root at the time it swings. It prevents every leg from being pulled to the same root-frame x distance at cycle boundaries.

This block is a conceptual per-leg description. The implementation must compute all `[B,4]` leg targets with tensor `gather`/broadcast operations and must not loop over envs or legs in the runtime path.

Target z uses terrain height only:

```text
target_z = height_at(target_world_xy)
```

Semantic classes are not used during nominal construction.

### 5.7 Swing Interpolation

Within each leg's swing window:

```text
alpha = normalized progress from swing_start to swing_end
alpha_s = smoothstep(alpha)

foot_xy = lerp(swing_start_xy, target_world_xy, alpha_s)
foot_z =
    lerp(swing_start_z, target_z, alpha_s)
    + swing_height * 4 * alpha * (1 - alpha)
```

This produces continuous swing trajectories and gives the optimizer a feasible initial foot path.

## 6. Optimization Variables

Replace per-frame contact logits with swing window parameters.

Current variables:

```text
root_pos_residual[B,T,3]
root_rpy_residual[B,T,3]
foot_pos_residual[B,T,4,3]
contact_logits[B,T,4]
```

New variables:

```text
root_pos_residual[B,T,3]
root_rpy_residual[B,T,3]
foot_pos_residual[B,T,4,3]
swing_center_raw[B,4]
swing_width_raw[B,4]
```

Raw swing parameters must be mapped to valid cyclic window values on device:

```text
swing_center = wrap01(swing_center_prior + center_scale * tanh(swing_center_raw))
swing_width = min_width + (max_width - min_width) * sigmoid(swing_width_raw)
swing_start = wrap01(swing_center - 0.5 * swing_width)
swing_end = wrap01(swing_center + 0.5 * swing_width)
```

`center_scale` must be large enough to let losses override random first-pair initialization, including an effective half-cycle reordering of the two diagonal groups. The default implementation target is `center_scale >= 0.55`.

All circular distances, start/end comparisons, and time interpolation must use the same wrap convention so windows crossing the horizon boundary stay continuous.

`decode_trajectory()` returns:

```text
root_pos
root_rpy
foot_pos
swing_center
swing_width
swing_start
swing_end
swing_prob[B,T,4]
contact_prob[B,T,4] = 1 - swing_prob
```

`swing_prob` is generated by a smooth circular window, for example:

```text
frame_phase = arange(T) / T
dist = circular_distance(frame_phase, swing_center)
swing_prob = sigmoid(k * (0.5 * swing_width - dist))
contact_prob = 1 - swing_prob
```

Hard `contact_state = contact_prob > threshold` is only for export, diagnostics, and viewer readback.

## 7. Terrain And Semantic Sampling

MPC needs GPU helpers equivalent to together planner terrain APIs:

```text
height_at(points_xy)
semantic_at(points_xy)
slope_at(points_xy)
support_at(points_xy)
```

All helpers operate on `MpcPlannerTerrain.height_map` and optional `semantic_map`.

`height_at`:

- bilinear `grid_sample`
- returns `[B,...]`

`semantic_at`:

- nearest `grid_sample`
- semantic id defaults to terrain id `0` if semantic map is absent

`slope_at`:

- finite difference sampling around `points_xy`
- equivalent to together planner's `slope_at`

`support_at`:

- fixed-radius GPU stencil around `points_xy`
- samples candidate height, slope, and semantic id
- legal support is semantic terrain only
- scores low slope and short distance
- returns preferred support xy, height, and slope
- if no legal terrain support exists in the stencil, returns the query xy/height with an `invalid_support` mask so losses can add a large finite penalty instead of propagating NaN

No helper may query true obstacle positions.

## 8. Loss Design

`compute_total_loss` should take terrain:

```python
compute_total_loss(decoded, nominal, state, command, terrain, cfg)
```

`optimizer.optimize_variables` and `planner.plan_segment` pass terrain into loss evaluation.

### 8.1 Contact Window Losses

`window_width_loss`:

- keeps `swing_width` within `[min_width, max_width]`
- prevents zero-length or nearly full-horizon swing

`diagonal_pair_loss`:

- `FL/RR` centers should be close
- `FR/RL` centers should be close
- the two diagonal groups should remain roughly half a cycle apart
- allows several frames of offset and moderate width differences

`phase_prior_loss`:

- soft pull toward nominal center/width
- lower priority than terrain, IK, and semantic feasibility
- lower priority than swing-order urgency, so the random nominal first pair can be overridden

`swing_center_urgency_order_loss`:

- computes a per-leg urgency score from current foot geometry and command-induced expected displacement
- groups urgency by diagonal pair:
  - `urgency_A = urgency_FL + urgency_RR`
  - `urgency_B = urgency_FR + urgency_RL`
- pushes the more urgent pair's `swing_start` closer to the current replan frame
- uses smooth weighting, not a hard branch, so gradients flow into `swing_center`

One implementation form:

```text
foot_body_now = rotate_world_to_body(foot_pos0_w - root_pos0, yaw0)
expected_disp_leg =
    step_gain * horizon_time * [Vx, Vy]
    + yaw_gain * Vyaw * horizon_time * [-foot_body_now.y, foot_body_now.x]

urgency_leg =
    ||expected_disp_leg||_2
    + reachability_risk(foot_body_now, expected_disp_leg)
    + touchdown_risk_proxy

pair_weight = softmax([urgency_A, urgency_B] / temperature)
early_cost_A = forward_phase_distance(0.0, swing_start_A)
early_cost_B = forward_phase_distance(0.0, swing_start_B)

swing_center_urgency_order_loss =
    pair_weight_A * early_cost_A
    + pair_weight_B * early_cost_B
```

`touchdown_risk_proxy` should use scanner height/semantic and support/slope approximations already available on GPU. The exact risk proxy can be lightweight; full terrain, semantic, touchdown, and IK losses below remain the stronger feasibility signals. This loss exists to make "which diagonal pair swings first" optimizable through `swing_center` instead of decided by random nominal initialization.

### 8.2 Stance And Swing Terrain Losses

Sample terrain under every foot frame:

```text
terrain_z = height_at(decoded.foot_pos[..., :2])
```

`stance_ground_loss`:

```text
contact_prob * smooth_l1(decoded.foot_z - terrain_z)
```

`swing_clearance_terrain_loss`:

```text
swing_prob * relu(terrain_z + clearance - decoded.foot_z)^2
```

This replaces the old placeholder `swing_clearance_loss` and `terrain_clearance_loss`.

### 8.3 Touchdown Surface And Semantic Loss

Touchdown is the event at swing end:

```text
touchdown_phase = (swing_center + 0.5 * swing_width) % 1.0
```

Touchdown world position is sampled by differentiable time interpolation from `decoded.foot_pos`:

```text
touchdown_w[B,4,3] = sample_time(decoded.foot_pos, touchdown_phase)
touchdown_xy = touchdown_w[..., :2]
touchdown_z = touchdown_w[..., 2]
```

Then sample terrain:

```text
td_terrain_z = height_at(touchdown_xy)
td_slope = slope_at(touchdown_xy)
support_xy, support_z, support_slope, invalid_support = support_at(touchdown_xy)
td_semantic = semantic_at(touchdown_xy)
```

`touchdown_ground_loss`:

```text
smooth_l1(touchdown_z - td_terrain_z)
```

`touchdown_slope_loss`:

```text
relu(td_slope - max_touchdown_slope)^2
```

`touchdown_support_loss`:

```text
||touchdown_xy - support_xy|| + relu(abs(touchdown_z - support_z) - tolerance)
+ invalid_support_penalty * invalid_support
+ support_slope_weight * relu(support_slope - max_support_slope)^2
```

`touchdown_semantic_loss`:

```text
small_weight * I(td_semantic == 1)
+ large_weight * I(td_semantic == 2)
```

Touchdowns must not be planned onto small, big, or large obstacles. Large/big obstacle collision receives higher weight. Final diagnostics may mark any touchdown on semantic id `1` or `2` as infeasible.

In the current semantic scanner contract, semantic id `2` covers big/large obstacles. If future scanner ids split big and large, touchdown semantic loss must penalize every non-terrain obstacle id, not only ids `1` and `2`.

### 8.4 Swing Trajectory Losses

`swing_smoothness_loss`:

- penalizes second differences of foot position under swing probability
- prevents jerky swing paths

`swing_direction_loss`:

- compares swing start-to-end displacement with expected body-frame displacement
- uses the root frame at swing start/touchdown timing
- covers pure yaw, so yaw commands still generate meaningful foot swing targets

`swing_direction_loss` replaces the old `swing_stride_loss`. Do not keep the old stride implementation in the active loss registry.

Expected body displacement:

```text
step_gain * horizon_time * [Vx, Vy]
+ yaw_gain * Vyaw * horizon_time * [-foot_start_body.y, foot_start_body.x]
```

### 8.5 Semantic Obstacle Losses

Semantic loss samples only scanner `semantic_map`.

Foot contact obstacle loss:

```text
contact_prob * I(semantic_at(foot_xy) == small_id)
contact_prob * I(semantic_at(foot_xy) == large_id)
```

Swing obstacle loss:

- penalizes swing paths that collide with obstacle tops unless clearance is sufficient
- uses obstacle height inferred from `height_map` and `semantic_map`

Body obstacle loss:

- samples a fixed footprint stencil around root/body
- penalizes small/large semantic occupancy under the body
- large/big obstacle weight is higher

Touchdown semantic loss is separate and stronger than generic foot semantic loss.

### 8.6 IK And Joint Losses

`joint_limit_loss` no longer consumes repeated seed joint angles. It solves IK inside the loss:

```text
ik_joint = IK(decoded.root_pos, decoded.root_rpy, decoded.foot_pos)
joint_limit_loss = per-joint limit penalty(ik_joint)
```

`ik_fk_residual_loss` remains core:

```text
fk_foot = FK(decoded.root_pos, decoded.root_rpy, ik_joint)
||fk_foot - decoded.foot_pos||
```

This directly targets previous yaw failures where planned foot targets were not reproducible after IK/FK.

### 8.7 Root Support Geometry Losses

`root_foot_center_loss`:

```text
foot_center_xy = mean(decoded.foot_pos[..., :2], dim=legs)
||decoded.root_pos[..., :2] - foot_center_xy||
```

This runs every frame. It keeps root xy near the center of the four-foot support footprint.

`support_plane_roll_pitch_loss`:

- fits a weighted plane to four foot positions every frame
- stance feet have higher weight
- swing feet have lower but nonzero weight to avoid degeneracy
- converts the support plane normal to roll/pitch
- penalizes root roll/pitch mismatch

```text
||decoded.root_rpy[..., :2] - estimated_roll_pitch||
```

Yaw is excluded from this loss. Yaw remains command/root-trajectory driven.

### 8.8 Body-Frame Tracking Loss

Command tracking must compare velocity in root/body frame.

For each step:

```text
world_delta = root_pos[t+1] - root_pos[t]
body_delta = rotate_world_to_body(world_delta, root_yaw[t])
body_vel = body_delta.xy / dt
yaw_rate = (yaw[t+1] - yaw[t]) / dt
```

Loss:

```text
||body_vel - [Vx, Vy]|| + yaw_weight * |yaw_rate - Vyaw|
```

### 8.9 Deleted Or Replaced Losses

Delete or remove from the active loss registry:

- `_command_adaptive_weights`
- `contact_schedule_tracking_loss`
- old placeholder `swing_clearance_loss`
- old placeholder `terrain_clearance_loss`
- old placeholder `obstacle_margin_loss`
- `touchdown_support = ||decoded.foot_pos - nominal["foot_pos"]||`

Replace them with the losses in this section.

## 9. Planner Output Contract

`MpcPlannerResult` keeps downstream shapes:

```text
root_pos[B,T,3]
root_rpy[B,T,3]
foot_pos[B,T,4,3]
joint_angles[B,T,12]
contact_state[B,T,4]
touchdown_seq[B,4,E,3]
planned_touchdown_w[B,T,4,3]
```

Important output rules:

- `foot_pos` is optimized output, not post-grounded output.
- `joint_angles` is solved from optimized root/foot via IK.
- `touchdown_seq` comes from swing window touchdown positions.
- `planned_touchdown_w` expands touchdown positions for cache consumers.
- `contact_state` is thresholded from continuous `contact_prob`.

## 10. Manager And Interface Changes

Remove `MpcFootholdMemory` from:

- `types.py`
- `planner.py`
- `manager.py`
- `nominal.py`
- viewer direct MPC path
- tests

Remove manager fields:

```text
_stance_anchor_w
_running_foot_rel_body
_yaw_foot_rel_body
_prev_contact_state
_stable_contact_steps
_prev_yaw_dominance
_yaw_entry_steps
```

Remove methods:

```text
_initialize_foothold_memory
_foothold_memory_for
_update_foothold_memory
```

`plan_segment` becomes:

```python
plan_segment(terrain, state, command, *, cfg, warm_start=None)
```

`build_nominal_trajectory` receives terrain for height sampling:

```python
build_nominal_trajectory(state, command, terrain, runtime_cfg)
```

`optimize_variables` receives state and terrain:

```python
optimize_variables(nominal, variables, state, command, terrain, cfg)
```

## 11. GPU Vectorization Rules

The MPC runtime path must use:

- `torch.arange`
- broadcasting
- `sin/cos`
- `cumsum`
- `where`
- `gather`
- `grid_sample`
- fixed-size stencil sampling

It must not use:

- NumPy
- CPU geometry packages
- Python loops over batch, time, or legs
- dynamic CPU-side lists for trajectory generation
- host sync for optimizer decisions

Small fixed constants such as leg ordering and phase offsets are tensors on the active device.

## 12. Testing And Acceptance

### 12.1 Unit Tests

Add or update tests for:

- body-frame root nominal integration, including yawing root
- GPU-vectorized nominal shapes and no Python time loop dependency
- randomized diagonal first swing pair
- random first pair is only nominal initialization; `swing_center_urgency_order_loss` can override final leading pair
- world-frame foot nominal
- touchdown target using swing-time root frame
- continuous swing window from center/width
- terrain `height_at`, `semantic_at`, `slope_at`, `support_at`
- touchdown on terrain has low loss
- touchdown on small/large has high loss

### 12.2 Backend Tests

Update `Go2Pvcnn/tests/test_batch_mpc_backend.py` to verify:

- no `MpcFootholdMemory`
- no output-side `_ground_contact_feet_to_terrain`
- no old contact schedule/touchdown support losses in breakdown
- no `_command_adaptive_weights`, old `swing_clearance_loss`, old `terrain_clearance_loss`, old `obstacle_margin_loss`, or old `swing_stride_loss` terms remain in the active registry or breakdown
- new breakdown terms exist:
  - `swing_window`
  - `diagonal_pair`
  - `swing_center_urgency`
  - `stance_ground`
  - `swing_clearance_terrain`
  - `touchdown_surface`
  - `touchdown_semantic`
  - `swing_direction`
  - `ik_joint_limit`
  - `ik_fk_residual`
  - `root_foot_center`
  - `support_plane_rp`
- each leg has one continuous swing window
- touchdown does not choose semantic id `1` or `2`
- stance foot z is near terrain height
- swing foot z clears terrain

### 12.3 IsaacLab Runtime Acceptance

Reuse current MPC reproduction gates:

```text
Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py
Go2Pvcnn/tests/mpc_root_cause_probe.py
Go2Pvcnn/tests/test_mpc_runtime_headless.py
```

Acceptance should track:

- pure forward/back/lateral do not regress
- yaw actual stance air ratio decreases
- yaw IK/FK contact error decreases
- joint saturation/near-limit ratio decreases
- touchdown semantic collision count is zero
- touchdown large collision count is zero
- swing windows remain continuous
- viewer visual playback no longer shows yaw flying-foot alternation

### 12.4 Verification Order

```text
1. py_compile targeted MPC files
2. focused test_batch_mpc_backend.py
3. mpc_root_cause_probe small matrix
4. mpc_yaw_gait_failure_probe
5. test_mpc_runtime_headless.py MPC selectors
6. viewer visual inspection
```

## 13. Migration Plan Summary

Implementation should proceed in stages:

1. Add terrain sampling helpers to `batch_mpc_planner/terrain.py`.
2. Rewrite nominal trajectory generation around vectorized body-frame root integration and swing windows.
3. Replace optimization variables and decode output with `center/width`.
4. Pass terrain into optimizer/loss registry.
5. Replace loss registry terms with terrain, semantic, touchdown, IK, and support-geometry losses.
6. Remove foothold memory and output-side grounding.
7. Update planner output touchdown extraction to use swing-window touchdown interpolation.
8. Update tests and runtime probes.

The first implementation plan should keep edits scoped to `Go2Pvcnn/extension/batch_mpc_planner`, viewer MPC direct-path adapter code only where memory removal requires it, and the focused MPC tests.
