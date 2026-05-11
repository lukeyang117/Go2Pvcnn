# K5 Mode-First Small-Obstacle Crossing Redesign

## 1. Problem Statement

The current `together` planner has strong semantic diagnostics and candidate filtering, but the remaining behavior is still too reactive:

- `small` obstacles are often avoided rather than crossed.
- A selected crossing can still look like normal walking plus late penalties instead of a gait designed for crossing.
- The planner can reject velocity too aggressively; for example, a left lateral command may select a right-side route.
- Touchdowns and foot paths can still appear on or through semantic obstacles in runtime playback.
- The previous state machine is mostly a post-rollout classifier. It labels a candidate after root, touchdown, and feet trajectories already exist. It does not first choose a crossing mode and then generate gait, touchdown, and foot trajectories for that mode.

This design replaces that behavior with a mode-first candidate generator:

```text
current root / feet / command / semantic geometry
-> decide one global mode for this replan
-> generate fixed K=5 candidates inside that mode
-> each candidate varies speed, and sometimes command-relative route
-> generate touchdown/apex/feet/base trajectories from current geometry
-> score and select
```

## 2. Goals

- Use a fixed `K=5` candidate axis for all modes.
- Keep planner hot paths pure GPU, fixed-shape, and tensorized.
- Remove per-candidate mode competition; mode is global for the current replan.
- Use command-relative `route`, not base-relative route.
- In obstacle-free semantic conditions, do not route left/right; only evaluate same-direction speed choices from current command down to zero.
- For normal `small` obstacles, prefer gait-based crossing over bypass.
- Merge `CROSS_SMALL_FRONT` and `CROSS_SMALL_REAR` into one `CROSS_SMALL` mode that plans a complete four-leg crossing in one horizon.
- Use a longer common horizon for all modes so `CROSS_SMALL` has enough time to complete front and rear leg crossing.
- Use dynamic obstacle geometry for touchdown and apex generation. Fixed templates may define tensor shape and phase timing, but not fixed world-space foot placements.
- Treat `large` obstacles and too-high `small` obstacles as `BYPASS_OBSTACLE` cases. If no safe bypass exists, the existing infeasible/safe-fallback mechanism may engage, but refusal is not a normal candidate mode.
- Validate with deterministic planner tests and `env_isaacsim` Isaac Lab headless runtime tests, with timeouts so tests do not occupy GPU indefinitely.

## 3. Non-Goals

- No CPU, NumPy, `.cpu()`, `.item()`, `.tolist()`, or dynamic host-side sub-batching in planner hot paths.
- No Python loop over env, candidate, or leg in planner hot paths.
- No global route planner.
- No visual/manual screenshot-based acceptance.
- No hidden crossing progress state inside `CROSS_SMALL`.
- No `HOLD` route or `REFUSE_OR_HOLD` candidate mode.
- No parallel replacement planner in new code files. This redesign must refactor the existing together planner modules in place.
- No long-term compatibility layer that keeps the old `K=3` route-only candidate behavior alive beside the new `K=5` behavior.
- No duplicated old/new crossing state logic. Code covered by the new mode-first contract should be replaced or removed.

## 4. Core Candidate Contract

Every replan has one global mode:

```text
m_t in {CRUISE, APPROACH_SMALL, CROSS_SMALL, BYPASS_OBSTACLE}
```

The candidate axis is fixed:

```text
K = 5
c_k = (beta_k, route_k, J_k)
```

Where:

- `beta_k`: velocity scale applied to the command.
- `route_k`: command-relative `CENTER`, `LEFT`, or `RIGHT`.
- `J_k`: total candidate cost, including barrier terms.

There is no `valid_k` field. Invalid candidates receive a barrier:

```text
J_k -> +inf
```

If all candidates are infeasible, the outer planner status should report infeasibility or use the existing safe fallback mechanism; this design does not add a `HOLD` route candidate.

## 5. Command-Relative Route Frame

All route directions are defined relative to the command direction, not the body frame.

For body-frame command:

```text
u = [vx, vy, wz]
```

World command direction:

```text
d = normalize(R(yaw) [vx, vy])
n = [-d_y, d_x]
```

Routes:

```text
CENTER: along d
LEFT:   lateral offset along +n
RIGHT:  lateral offset along -n
```

This is required to avoid selecting a route opposite to the user's intended lateral command. If the command has a clear lateral direction and the selected bypass route is opposite to that direction, the candidate should receive a strong penalty or barrier unless the same-direction route is impossible and the opposite route is the only safety fallback.

For forward commands:

```text
Delta root_xy dot d >= -epsilon
```

Candidates that move significantly backward under a forward command receive a barrier.

## 6. Mode Classification

Mode is computed once at the start of each replan:

```text
m_t = f(root_t, feet_t, semantic_map, command)
```

Implementation must replace the old post-rollout crossing classifier with an in-place pre-rollout classifier:

```text
classify_mode_and_geometry(...)
  -> mode_code [B]
  -> small_geometry [B, ...]
  -> gate_masks [B, ...]
```

This classifier must run inside the existing together planner path before `expand_segment(...)` creates root, touchdown, foot, and IK rollouts. The old pattern `rollout -> classify state -> score` is not allowed for T116.

The classifier uses a command corridor:

```text
s = (p - root_xy) dot d
l = (p - root_xy) dot n
```

Relevant obstacles are semantic obstacle cells in:

```text
s in [s_min, s_max]
abs(l) <= corridor_width
```

Let:

```text
q(p) = (p_xy - root_xy) dot d
q_f = small_front_s
q_b = small_back_s
```

Mode priority:

```text
1. large in command corridor -> BYPASS_OBSTACLE
2. no relevant semantic obstacle in command corridor -> CRUISE
3. small in command corridor and too high -> BYPASS_OBSTACLE
4. small interaction all_clear -> CRUISE
5. small in command corridor but still far -> APPROACH_SMALL
6. small in crossing window and all crossing gates pass -> CROSS_SMALL
7. small in crossing window but crossing gates fail and too-high/no-cross-route mask is true -> BYPASS_OBSTACLE
8. small in crossing window but crossing gates fail only because root/feet are not yet ready -> APPROACH_SMALL
```

`all_clear` is only defined for an active small-obstacle interaction; it must never override a `large` obstacle or a too-high `small` obstacle:

```text
all_clear =
  body_rear_s > q_b + m_body_clear
  and all(anchor_leg_s > q_b + m_leg_clear for leg in 4)
```

This prevents the planner from switching to `CRUISE` when the base has passed the obstacle but one or more legs have not.

The `CROSS_SMALL` gate requires:

- `small` is in the command corridor.
- `small` relative height is below the crossable threshold.
- root is in the crossing window.
- current foot anchors are not on obstacle surfaces.
- legal terrain support exists beyond the small obstacle for all four touchdown targets.
- each leg can construct an `anchor -> apex -> touchdown` crossing path from current geometry.
- leg reach and IK feasibility are not obviously impossible.

`APPROACH_SMALL` is constrained by inequalities, not only prose:

```text
0 <= delta_root_s <= q_f - m_approach_stop
forall leg: touchdown_leg_s <= q_f - m_touchdown_front
forall leg: semantic(touchdown_leg) == terrain
forall leg: dist(touchdown_leg, small_boundary) >= m_boundary
root/body footprint does not enter small footprint
```

These rules allow the robot to move closer before crossing, while preventing premature crossing or touchdown on/near the small obstacle.

## 7. K=5 Candidate Tables

Mode tables provide only fixed-shape structural parameters. They do not define fixed foot placements.

### 7.1 `CRUISE`

```text
route = [CENTER, CENTER, CENTER, CENTER, CENTER]
beta  = [1.00, 0.75, 0.50, 0.25, 0.00]
```

Behavior:

- no left/right route
- direction does not change
- speed candidates cover command speed down to zero
- terrain, touchdown, foot path, IK, and body quality decide which speed is best

### 7.2 `APPROACH_SMALL`

```text
route = [CENTER, CENTER, CENTER, CENTER, CENTER]
beta  = [0.80, 0.60, 0.40, 0.20, 0.00]
```

Behavior:

- move closer to the small obstacle
- do not cross yet
- do not place touchdowns on or too close to the small obstacle
- do not let the base pass too far into the obstacle region

### 7.3 `CROSS_SMALL`

```text
route = [CENTER, CENTER, CENTER, CENTER, CENTER]
beta  = [0.50, 0.35, 0.20, 0.10, 0.00]
```

Behavior:

- plan a full four-leg crossing in one horizon
- no left/right bypass competition
- velocity tracking is secondary to crossing success
- each candidate's touchdown and apex are generated from current small-obstacle geometry

### 7.4 `BYPASS_OBSTACLE`

```text
route = [LEFT, LEFT, RIGHT, RIGHT, CENTER]
beta  = [0.50, 0.25, 0.50, 0.25, 0.00]
```

Behavior:

- used for `large` obstacles and too-high `small` obstacles
- center candidates that approach the obstacle receive barrier cost
- left/right are relative to command direction
- bypass speeds are intentionally reduced
- a selected `CENTER` candidate is not a successful bypass when the obstacle blocks the command corridor, unless all bypass candidates are infeasible and the outer status reports infeasible/safe fallback

## 8. Dynamic Geometry Contract

The fixed mode tables only define:

- candidate count
- velocity scale
- route class
- contact phase structure

They do **not** define one fixed crossing motion for all `small` obstacles.

For every env and candidate, the planner must dynamically compute:

```text
small_front_s
small_back_s
small_center_xy
small_top_z
touchdown targets
apex targets
foot path
base path
```

Example:

```text
same mode = CROSS_SMALL
same beta table
same phase table

but:
touchdown = function(current feet, root, command direction, small back edge, terrain support)
apex     = function(current feet, small center/top, crossing clearance)
```

This preserves GPU parallelism while still adapting the action to the actual obstacle position, height, terrain, and current foot anchors.

### 8.1 GPU Reduction Rule For Small Geometry

Small-obstacle geometry must be computed with fixed-shape tensor reductions, not host-side indexing:

```text
grid_xy: [B, H*W, 2]
grid_z:  [B, H*W]
small_mask: [B, H*W]
corridor_mask: [B, H*W]
target_mask = small_mask & corridor_mask
```

Allowed pattern:

```text
s_values = dot(grid_xy - root_xy, d)
l_values = dot(grid_xy - root_xy, n)
front = amin(where(target_mask, s_values, +inf), dim=-1)
back  = amax(where(target_mask, s_values, -inf), dim=-1)
top_z = amax(where(target_mask, grid_z, -inf), dim=-1)
center_xy = sum(where(target_mask, grid_xy, 0), dim=-2) / clamp(sum(target_mask), min=1)
```

Hard prohibition for geometry extraction:

- no `nonzero`
- no `argwhere`
- no `masked_select`
- no Python loop over envs
- no CPU round trip

If multiple `small` obstacles are present in the command corridor, T116 must select the nearest connected/contiguous command-corridor small component by a fixed window rule, or otherwise choose the nearest small obstacle in `s`. It must not merge multiple separated small obstacles into one giant obstacle envelope.

## 9. Common Horizon Contract

All modes use the same longer horizon:

```text
T = 1.0 s
dt = 0.02 s
horizon_steps = 50
event_cap = 2
```

Rationale:

- `CROSS_SMALL` needs enough time to complete a four-leg crossing.
- Keeping all modes at the same horizon preserves fixed tensor shapes.
- `dt=0.02` preserves the current step granularity.

This changes the old timing contract:

```text
old: 0.7 s, 35 steps
new: 1.0 s, 50 steps
```

Implementation must update every direct consumer that currently depends on `35` frames or `0.7s`, including:

- `Go2Pvcnn/extension/batched_together_planner/config.py`
- `Go2Pvcnn/extension/batched_together_planner/manager.py`
- reward/cache consumers under `Go2Pvcnn/extension/mdp/rewards_reference.py`
- task/env cfg horizon assumptions under `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py`
- viewer together config/builders under `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- active tests, fixtures, and benchmark expectations that assert 35 frames or 0.7 seconds

## 10. Schedule Contract

### 10.1 Cruise-like schedule

Used by:

- `CRUISE`
- `APPROACH_SMALL`
- `BYPASS_OBSTACLE`

This can remain trot-like, but it must work over 50 steps.

### 10.2 `CROSS_SMALL` schedule

`CROSS_SMALL` uses a dedicated crossing schedule. The schedule is command-relative, not hard-coded to body-front first for every command.

For forward command direction, the initial design uses staggered front pair, then staggered rear pair:

```text
FL swing: 0.10s - 0.38s
FR swing: 0.18s - 0.46s
RL swing: 0.55s - 0.83s
RR swing: 0.63s - 0.91s
```

For backward and lateral commands, the leading legs are those with the largest projection along the command direction:

```text
lead_order = argsort_desc(dot(hip_xy_or_anchor_xy - root_xy, d))
```

The same early/late swing time slots are assigned by `lead_order`, so the two command-leading legs cross before the two command-trailing legs. This preserves one tensorized `CROSS_SMALL` mode while avoiding a body-front-only gait when the command is backward, left, or right.

The schedule is not a hidden progress state. It is the full crossing plan for the current replan horizon.

## 11. `CROSS_SMALL` Touchdown And Apex Generation

For each leg:

```text
anchor -> apex -> touchdown
```

Small obstacle edges in command coordinates:

```text
s_front = min((p - root_xy) dot d for p in small)
s_back  = max((p - root_xy) dot d for p in small)
```

Touchdown requirement:

```text
(touchdown_xy - root_xy) dot d > s_back + touchdown_margin
```

Touchdown must also satisfy:

- legal terrain support
- not `small`
- not `large`
- grounded against selected support height
- boundary margin from `small`
- leg reach and IK feasibility

Apex requirement:

```text
apex_xy = small_center_xy + eta * d
apex_z  = small_top_z + crossing_clearance
```

The generated foot path must pass through or above the apex and remain clear of the small obstacle. This check must be part of candidate generation/scoring, not only a late visualization diagnosis.

## 12. Cost And Barrier System

Every candidate receives:

```text
J_k =
  J_vel
+ J_terrain
+ J_touchdown
+ J_swing
+ J_gait
+ J_body
+ J_ik
+ J_barrier
```

### 12.1 Velocity

`J_vel` compares the generated root displacement to the scaled command `beta_k * command`.

In `CROSS_SMALL`, velocity weight must be lower than crossing success. A slower successful crossing is preferred over a faster trajectory that clips or bypasses the small obstacle.

### 12.2 Terrain

`J_terrain` evaluates:

- support height change
- slope
- roughness
- anchor-to-touchdown ground profile
- base path ground profile

### 12.3 Touchdown

Barrier conditions:

- touchdown on `small`
- touchdown on `large`
- touchdown too close to `small` boundary
- touchdown not grounded
- touchdown behind-edge requirement fails in `CROSS_SMALL`

### 12.4 Swing / Foot Path

Barrier conditions:

- foot path intersects `small`
- crossing path cannot reach required apex
- thigh/calf collision is above hard threshold

### 12.5 Gait

For `CROSS_SMALL`:

- all four legs must have crossing touchdown events
- all four final touchdown targets must be beyond `s_back + margin`
- all four touchdown targets must be grounded
- command-leading legs must cross earlier than command-trailing legs according to the crossing schedule

If these fail:

```text
J_k -> +inf
```

`cross_small_success` is true if and only if:

```text
mode == CROSS_SMALL
and forall leg: touchdown_after_back_edge[leg]
and forall leg: touchdown_terrain_grounded[leg]
and forall leg: foot_path_clear[leg]
and root_path_clear
and body_clear
and thigh_clear
and calf_clear
and root_final_s > q_b + m_root_clear
and body_rear_final_s > q_b + m_body_clear
```

Aggregate counts alone are not sufficient to prove success; the per-leg/per-surface masks above must be available for tests and diagnostics.

### 12.6 Body / Base

Barrier conditions:

- base path penetrates `small`
- body, thigh, or calf penetrates obstacle beyond hard threshold
- root progress violates mode-specific bounds

The base/body may pass above a `small` obstacle during a valid crossing. The forbidden condition is penetration or clearance below threshold, not the existence of an xy overlap with the small obstacle footprint.

### 12.7 Direction Guard

Barrier or strong penalty:

- forward command produces significant backward progress
- lateral command selects the opposite command-relative route while same-direction bypass is feasible

This must be implemented as tensor masks:

```text
progress_ok = delta_root_xy dot d >= -eps
same_direction_feasible = any(candidate_feasible & route_command_compatible, dim=K)
opposite_selected_bad = selected_opposite_route & same_direction_feasible
```

`opposite_selected_bad` receives a barrier or hard failure. The same idea applies to forward/backward commands: a candidate must not make significant progress opposite to `d`.

## 13. GPU Parallelism Contract

The planner must remain fixed-shape:

```text
B -> [B, K] -> [B*K, T, ...]
K = 5
T = 50
```

Allowed pattern:

```text
mode_code: [B]
beta_table: [mode_count, K]
route_table: [mode_count, K]
phase_table: [mode_count, T, 4]

beta: [B, K]
route: [B, K]
phase: [B, T, 4]
```

Then dynamic geometry is computed by tensor operations:

```text
small_geometry: [B, ...]
touchdowns: [B, K, 4, event_cap, 3]
apex: [B, K, 4, 3]
foot_path: [B, K, T, 4, 3]
```

Implementation target in existing code:

```text
mode_code [B] gathered from classify_mode_and_geometry(...)
beta = gather(beta_table, mode_code)       # [B, 5]
route = gather(route_table, mode_code)     # [B, 5]
phase = gather(phase_table, mode_code)     # [B, 50, 4]
candidate_command = command[:, None, :] * beta[..., None]
flattened candidate tensors: [B*K, ...]
```

The old fixed three-route helper must be replaced; `semantic_candidate_costs` becomes `[B, 5]`.

Hard prohibitions in planner hot paths:

- NumPy
- `.cpu()`
- `.item()`
- `.tolist()`
- `nonzero`
- `argwhere`
- `masked_select`
- `to("cpu")`
- `torch.device("cpu")` in planner hot paths
- tensor truthiness such as `if tensor.any():`
- dynamic env/candidate subbatch calls
- Python loops over env, candidate, or leg
- host-side branching that changes which envs/candidates are planned

Small fixed Python loops may be acceptable only outside hot paths, for tests or static table construction, if guardrails explicitly allow them.

## 14. Implementation And Cleanup Contract

This is a major behavior rewrite, but it should not create a second planner implementation.

### 14.1 File policy

Implementation must modify the existing together planner files instead of creating a new code path:

- `Go2Pvcnn/extension/batched_together_planner/config.py`
- `Go2Pvcnn/extension/batched_together_planner/schedule.py`
- `Go2Pvcnn/extension/batched_together_planner/terrain.py`
- `Go2Pvcnn/extension/batched_together_planner/parameterization.py`
- `Go2Pvcnn/extension/batched_together_planner/costs.py`
- `Go2Pvcnn/extension/batched_together_planner/planner.py`
- `Go2Pvcnn/extension/batched_together_planner/types.py`
- `Go2Pvcnn/extension/batched_together_planner/manager.py`
- direct reward/cache/env/viewer consumers affected by the `50` step horizon:
  - `Go2Pvcnn/extension/mdp/rewards_reference.py`
  - `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py`
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- existing runtime harness and test files where needed

New implementation code files are not allowed for this redesign unless a later user-approved design revision explicitly grants that. Tests should also prefer updating the existing test files that already own together planner semantic/runtime behavior; add a new test file only if an existing test file cannot reasonably own the fixture without becoming misleading.

### 14.2 Deletion policy

Code made obsolete by this design should be removed instead of kept as a dormant alternative. Expected cleanup includes:

- remove or replace `K=3` route-only candidate assumptions
- remove old `front_cross` / `rear_follow` / `clear` candidate semantics that conflict with the merged `CROSS_SMALL`
- remove post-rollout state-classification logic that is no longer part of the mode-first flow
- remove tests or fixture expectations that assert old `K=3`, 35-step, or front/rear split behavior, replacing them with `K=5`, 50-step, and `CROSS_SMALL` expectations
- remove stale diagnostics that only explain the old state labels if they no longer map to runtime acceptance
- update or remove active tests that still assert old `K=3`, `35` frames, or `FRONT_CROSS -> REAR_FOLLOW -> CLEAR` as the authoritative crossing success sequence

Cleanup must be dependency-aware. If an old field is still consumed by manager, reward, runtime diagnostics, or tests, update the consumer in the same leaf rather than leaving a dead field or a compatibility shim.

### 14.3 Notes and design memory cleanup

The repository notes should keep historical evidence, but active todo/design memory must not present the old design as the next implementation target. When this design moves to todo:

- mark `T114` and `T115` as completed historical baselines, not active architecture targets
- add a new child under `T100` for this `K=5 mode-first crossing` rewrite
- state clearly that the new child supersedes old `K=3` small-crossing implementation details
- keep old logs as evidence only; do not delete verification logs

## 15. Testing Strategy

### 15.1 Deterministic planner tests

Required fixtures:

- `F1_cruise_no_semantic_k5_speed_ladder`
- `F2_cruise_uneven_terrain_selects_slower_center_speed`
- `F3_lateral_command_direction_guard`
- `F4_forward_command_no_backward_progress`
- `F5_approach_small_does_not_cross_or_touch`
- `F6_cross_small_four_leg_success_all_command_directions`
- `F7_cross_small_dynamic_geometry_changes_with_obstacle_position_and_direction`
- `F8_cross_small_rejects_touchdown_on_small`
- `F9_cross_small_rejects_per_leg_foot_path_collision`
- `F10_cross_small_rejects_base_body_thigh_calf_collision`
- `F11_too_high_small_uses_bypass`
- `F12_large_blocks_center_uses_bypass`
- `F13_k5_shape_consistency_all_modes`
- `F14_horizon_50_contract`

`F6` and `F7` must be direction-parametrized:

```text
direction_id in {forward, backward, lateral_left, lateral_right}
obstacle is placed in +d command corridor for that direction
```

This ensures `small` crossing is tested for front/back/left/right velocity directions rather than only forward motion.

Expected active deterministic test names include:

- `test_t116_f1_cruise_no_semantic_k5_speed_ladder`
- `test_t116_f2_cruise_uneven_terrain_selects_slower_center_speed`
- `test_t116_f6_cross_small_four_leg_success_all_command_directions`
- `test_t116_f7_cross_small_dynamic_geometry_changes_with_obstacle_position_and_direction`
- `test_t116_f8_cross_small_rejects_touchdown_on_small`
- `test_t116_f9_cross_small_rejects_per_leg_foot_path_collision`
- `test_t116_f10_cross_small_rejects_base_body_thigh_calf_collision`
- `test_t116_older_t113_t115_records_are_superseded_non_authoritative`

### 15.2 Guardrail tests

Required checks:

- candidate axis is `K=5`
- horizon is `50`
- no planner hot-path NumPy / CPU sync / dynamic subbatch
- no hot-path Python loop over env/candidate/leg
- no accidental fallback to old `K=3` route-only behavior
- no production hot-path dependency on `semantic_candidate_count == 3`
- no production hot-path `35` frame assumption for together planner outputs
- no active `front_cross/rear_follow/clear` mode contract in together planner hot path
- no `argwhere`, `masked_select`, `to("cpu")`, tensor truthiness, or tensor-mask subbatch passed into `plan_segment(...)` / `expand_segment(...)`

Expected guardrail names include:

- `test_t116_candidate_axis_is_k5_and_no_k3_route_only_fallback`
- `test_t116_horizon_50_no_35_step_contract_in_hot_path`
- `test_t116_no_old_front_rear_clear_mode_contract_in_hot_path`
- `test_t116_no_cpu_sync_or_dynamic_geometry_extraction`

### 15.3 Isaac Lab headless runtime tests

Runtime environment:

```text
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim
```

Runtime tests must be headless and output-based. They may reuse viewer/runtime harness code, but acceptance must not depend on screenshots.

Required runtime cases:

- `R1_cruise_no_semantic_no_bypass`
- `R2_small_cross_runtime_four_leg_success_all_command_directions`
- `R3_small_cross_runtime_no_touchdown_on_small_all_directions`
- `R4_small_cross_runtime_no_foot_path_collision_all_directions`
- `R5_small_cross_runtime_no_base_body_leg_penetration_all_directions`
- `R6_large_runtime_bypass_direction_guard`
- `R7_lateral_runtime_no_opposite_direction_rejection_left_and_right`

Runtime output fields:

- `mode`
- `selected_beta`
- `selected_route`
- `semantic_candidate_costs [B, 5]`
- `direction_id`
- `command_direction_xy`
- `obstacle_s_l`
- `small_front_s`
- `small_back_s`
- `small_top_z`
- `small_relative_height`
- `root_to_front_s`
- `root_to_back_s`
- `approach_window_mask`
- `cross_window_mask`
- `too_high_small_mask`
- `touchdown_on_small_count`
- `foot_small_collision_count`
- `base_small_penetration_count`
- `body_min_clearance`
- `leg_min_clearance`
- `per_leg_touchdown_on_small_count`
- `per_leg_foot_small_collision_count`
- `per_leg_min_clearance_to_small`
- `per_leg_touchdown_beyond_small_back_edge`
- `touchdown_ground_gap_by_leg`
- `touchdown_semantic_by_leg`
- `touchdown_frame_by_leg`
- `command_leading_before_trailing_schedule_ok`
- `cross_small_success`
- `command_direction_violation`

Runtime four-direction cases should be batched/parametrized where possible to reduce Isaac Lab startup cost.

### 15.4 Runtime timeout and GPU cleanup contract

Every Isaac Lab runtime command must be wrapped with timeout and cleanup behavior. Tests must not occupy GPU indefinitely.

Required command style:

```text
timeout -s INT -k 20s <bounded-seconds> bash -lc '<env_isaacsim python pytest ...>; code=$?; echo EXIT_CODE:$code; exit $code'
```

If a runtime case exceeds its timeout:

- mark the runtime case failed or blocked
- ensure the process is terminated
- record the timeout in the log
- do not leave long-running Isaac Lab or viewer processes on the GPU

This timeout contract is part of acceptance, not an optional operator habit.

## 16. Acceptance Indicators

The redesign is acceptable only if:

- all modes use `K=5`
- all modes use the common 50-step horizon
- no semantic obstacle case uses speed ladder with `CENTER` route only
- small obstacle crossing is tested and accepted for forward, backward, lateral-left, and lateral-right command directions
- normal small obstacle case enters `CROSS_SMALL` when the gates pass
- `CROSS_SMALL` produces a full four-leg crossing in one horizon
- `CROSS_SMALL` does not select touchdowns on `small`
- `CROSS_SMALL` foot path, base path, body, thigh, and calf avoid small obstacle penetration
- `CROSS_SMALL` success is proven by per-leg/per-surface masks, not only aggregate counters
- `APPROACH_SMALL` obeys the no-premature-crossing inequalities
- `large` and too-high `small` use `BYPASS_OBSTACLE`
- selected center/zero-speed is not counted as successful bypass when a safe non-center bypass exists
- bypass routes are command-relative
- command-direction reversal is guarded
- old active tests asserting `K=3`, `35` frames, or `FRONT_CROSS -> REAR_FOLLOW -> CLEAR` are deleted or rewritten as non-authoritative history checks
- deterministic tests and headless `env_isaacsim` runtime tests pass on final code state
- runtime tests use bounded timeouts and do not leave GPU-occupying processes behind
- implementation changes happen in the existing together planner modules, not in a new parallel planner
- obsolete old-state and `K=3` code paths are deleted or replaced rather than kept as dormant compatibility code

## 17. Relationship To Previous Work

This design supersedes the previous `T114` and `T115` crossing state structure for future implementation work.

It preserves:

- semantic support validity
- touchdown boundary margin idea
- anchor-to-touchdown path checking idea
- three-surface validity for touchdown/feet/base
- groundedness checks
- headless `env_isaacsim` runtime acceptance

It changes:

- `K=3` -> `K=5`
- route-only candidates -> speed-ladder candidates
- post-rollout state classification -> pre-rollout global mode classification
- `front_cross` / `rear_follow` -> one `CROSS_SMALL`
- 0.7s / 35-step horizon -> 1.0s / 50-step horizon
- cruise-like small crossing -> dedicated crossing schedule and dynamic apex/touchdown generation

## 18. Open Design Risks

- A 1.0s horizon may still be too short for some small obstacle placements. If runtime proves this, the next design revision should compare `1.2s / 60 steps`.
- Simultaneous front/rear pair crossing may challenge stability. The current design uses staggered pair timing to reduce that risk.
- Longer horizon increases planner and IK cost; performance tests must include runtime timing.
- The old reward/cache consumers may assume 35 frames and must be updated carefully.
- Enforcing no hot-path Python loops over legs may require tensorized static leg templates for the crossing schedule and apex generation.
