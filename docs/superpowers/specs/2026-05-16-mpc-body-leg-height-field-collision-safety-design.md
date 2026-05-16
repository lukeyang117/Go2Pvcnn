# MPC Body/Leg Height-Field Collision Safety

## 1. Purpose

This design adds a new T302 safety layer on top of the existing T300e continuous swing-window MPC backend. The goal is to keep the current T300e gait/contact/stance-grounding behavior, while making the planner work on complex height-field terrain and around semantic obstacles:

- walk on `COBBLESTONE_ROAD_CFG` terrain without root/body/knee/shank/swing-foot collisions;
- cross low small obstacles when they are feasible to step over;
- avoid or slow down for high small obstacles and large obstacles;
- never place touchdown or stance feet on semantic obstacle cells.

The active implementation target remains the existing MPC backend:

```text
Go2Pvcnn/extension/batch_mpc_planner/
```

This is not a new planner backend and not a post-processing pass.

## 2. Relationship To T300e

T300e already provides the base MPC contracts:

- continuous `swing_center/swing_width` contact windows;
- scanner height/semantic terrain queries;
- terrain-aware stance and touchdown losses;
- IK-derived joint limit and IK/FK residual losses;
- root-foot center and support-plane losses;
- no `MpcFootholdMemory`;
- no output-side foot z grounding after loss.

T302 adds collision and obstacle-behavior constraints around that foundation. T302 must not regress T300e acceptance metrics:

```text
stance_airborne_ratio
stance_max_gap
support_stability
ik_fk_residual
root_foot_center_error
support_plane_roll_pitch_error
touchdown_anchor behavior
```

## 3. Goals

- Compute root/body/knee/shank/foot collision losses directly from planned horizon trajectories.
- Derive knee and shank world points inside the MPC kinematics pass from planned `root + foot`; do not read knee link poses from IsaacLab runtime.
- Use height map clearance for root/body/knee/shank and swing-foot collision checks.
- Use semantic map ids for touchdown and stance foot obstacle checks:
  - semantic `0`: ground, allowed;
  - semantic `1` / `2`: obstacle classes, penalized by default;
  - actual labels remain config-driven so tests follow the current semantic map contract.
- Treat low small obstacles as crossable when their top is at most `0.3m` above the root-projected ground height.
- Treat high small obstacles and large obstacles as avoid/slow-down obstacles when they affect the command direction or yaw sweep.
- Keep all MPC runtime work GPU-batched and suitable for later reinforcement-learning integration.
- Use TDD: write failing tests first, then implement each slice.
- Run real IsaacLab headless acceptance with `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`.

## 4. Non-Goals

- Do not solve obstacle behavior by first following command velocity and then modifying the path afterward.
- Do not introduce CPU/NumPy obstacle search or per-env/per-time/per-leg Python loops in the MPC runtime hot path.
- Do not require true obstacle USD prim positions for planning. The planner consumes scanner height/semantic tensors.
- Do not add new production source files for T302. Existing production files may be modified; stale conflicting logic should be deleted rather than kept in parallel.
- Do not make touchdown or stance feet legal on small obstacles, even if the obstacle is low.
- Do not add a complex apex-alignment crossing reward in the first version. Low-small crossing should come from velocity tracking, swing-foot clearance, and semantic stance/touchdown penalties.

## 5. Runtime Data Contract

Each MPC replan consumes the same scanner-centered terrain state as T300e:

```text
root_pos[B,3]
root_rpy[B,3]
foot_pos_w[B,4,3]
joint_pos[B,12]
cmd_body[B,3] = [Vx, Vy, Vyaw]
terrain.height_map[B,H,W]
terrain.semantic_map[B,H,W] or None
terrain scanner pose/yaw/ranges
```

Keyboard/user command velocity is root/body-frame velocity. The planner rotates it into world frame when building nominal trajectories or computing command-direction risk.

The optimizer still solves one total loss:

```text
total_loss =
    existing_t300e_losses
  + root_body_heightfield_collision_loss
  + knee_shank_heightfield_collision_loss
  + swing_foot_heightfield_collision_loss
  + touchdown_stance_semantic_obstacle_loss
  + command_risk_scaled_tracking_losses
```

Obstacle safety is part of the MPC optimization itself. It is not an after-the-fact edit to the planned trajectory.

## 6. GPU And Performance Contract

T302 is designed for future RL training throughput:

- represent collision query points as batched tensors, such as `[B,T,P,3]`, `[B,T,4,3]`, or `[B,T,4,K,3]`;
- query height/semantic maps with the existing GPU terrain helpers;
- avoid `.cpu()`, `.numpy()`, `.item()`, host synchronization, or CPU-side obstacle loops in the planner hot path;
- vectorize across batch, horizon, legs, and collision sample points;
- expose diagnostics and tests from already-computed GPU tensors, then serialize only final metrics outside the planner hot path.

Small Python control flow is acceptable for module dispatch and test harness orchestration. It must not become the core MPC loss computation.

## 7. Kinematics Extension

The current IK/FK loss path already computes joint angles and FK foot residuals from planned root/foot targets. T302 extends that same pass to return intermediate leg geometry:

```text
joint_angles[B,T,12]
fk_foot_pos_world[B,T,4,3]
knee_pos_world[B,T,4,3]
shank_sample_world[B,T,4,K,3]
```

`knee_pos_world` and `shank_sample_world` must be generated from the same IK result used by joint-limit and IK/FK residual losses. The design must not run a second independent IK solve just to check collisions.

The first implementation can use `K=1` or `K=2` shank samples per leg. The intent is to catch cases where knee and foot are safe but the lower leg segment passes through terrain or an obstacle.

## 8. Height-Field Collision Losses

### 8.1 Root And Body Bottom

Root collision should not rely on the root center alone. The loss samples root-related world points:

```text
root point or root clearance proxy
body bottom / belly sample points
front/rear/left/right body-bottom offsets
```

Body-bottom offsets are defined in root/body frame and transformed to world using planned root pose. For each point:

```text
terrain_z = height_at(point_xy)
clearance = point_z - terrain_z
loss = relu(required_margin - clearance)^2
```

This detects body or belly collisions with rough terrain, boxes, stairs, small obstacles, or large obstacles present in the height field.

### 8.2 Knee And Shank

For all planned frames:

```text
knee_clearance = knee_z - height_at(knee_xy)
shank_clearance = shank_sample_z - height_at(shank_sample_xy)
```

The loss penalizes any clearance below its configured margin:

```text
loss = relu(knee_margin - knee_clearance)^2
     + relu(shank_margin - shank_clearance)^2
```

This check is height-field based and does not require semantic ids. If the height map includes an object, a terrain bump, a stair, or a random box, the leg geometry must clear it.

### 8.3 Swing Foot

Only swing-phase foot samples use swing-foot collision clearance:

```text
terrain_z = height_at(foot_xy)
clearance = foot_z - terrain_z
loss = swing_mask * relu(foot_margin - clearance)^2
```

This makes a swing foot lift over low obstacles in its path. It does not make touchdown or stance legal on obstacles; that is handled by semantic touchdown/stance loss.

## 9. Touchdown And Stance Semantic Safety

Touchdown and stance foot safety uses semantic ids, not height difference heuristics.

For each leg, evaluate:

```text
touchdown frame
all stance frames after touchdown
all current stance frames inside the horizon
```

At each evaluated foot xy:

```text
semantic_id = semantic_at(foot_xy_world)
```

Rules:

```text
semantic_id == 0 -> ground, allowed
semantic_id in obstacle ids -> penalized
```

The initial configured obstacle ids are semantic `1` for small obstacle and semantic `2` for large obstacle, matching the current semantic-course scanner contract. The implementation should keep this configurable so it follows the actual semantic map labels.

This means:

- low small obstacles are never valid touchdown or stance surfaces;
- high small obstacles are never valid touchdown or stance surfaces;
- large obstacles are never valid touchdown or stance surfaces.

If `semantic_map` is unavailable, this loss is disabled or treated as all-ground according to the existing terrain helper contract. Collision losses still run from the height map.

## 10. Small And Large Obstacle Behavior

Small-obstacle classification must start from configured semantic small-obstacle ids, not from height alone. The relative-height threshold only splits semantic-small cells into low-small versus high-small. Non-obstacle rough terrain, stairs, random boxes, or slopes must not be reclassified as semantic small obstacles by height.

### 10.1 Low Small Obstacles

A small obstacle is crossable when:

```text
root_ground_z = height_at(root_xy)
obstacle_top_z = height_at(obstacle_xy)
relative_height = obstacle_top_z - root_ground_z
relative_height <= 0.3
```

For the first T302 version, low-small crossing does not add a separate apex attraction reward. The expected behavior comes from the combined objective:

- command/nominal tracking still wants the robot to move through the commanded path;
- swing-foot height-field collision requires foot clearance over the obstacle;
- knee/shank/body collision requires the leg and body to clear the obstacle;
- touchdown/stance semantic loss forbids stepping on the obstacle.

The desired result is: the robot crosses low small obstacles by swinging over them, while landing on semantic ground before or after the obstacle.

### 10.2 High Small Obstacles And Large Obstacles

A small obstacle becomes an avoid/slow-down obstacle when:

```text
relative_height > 0.3
```

Large obstacles are always avoid/slow-down obstacles.

These obstacles affect tracking weights when they intersect the command direction or yaw swept region. The planner should be allowed to slow down or rotate less aggressively instead of hard-tracking a command that would drive the robot into an obstacle.

## 11. Command-Direction And Yaw Risk Scaling

Risk scaling modifies tracking-loss weights inside the MPC optimization. It is not a post-processing speed clamp.

Risk and obstacle masks must be computed from all configured obstacle cells in the scanner semantic/height tensors, then filtered by command corridor or yaw swept region with batched tensor masks. The implementation must not assume a single forward direction, a fixed front row, or a one-ray obstacle detector.

### 11.1 Translational Command Risk

When translational command speed is nonzero:

```text
cmd_world = rotate_by_root_yaw([Vx, Vy])
```

The planner evaluates high small / large obstacle candidates in a command corridor:

```text
forward_projection = dot(obstacle_xy - root_xy, cmd_world_unit)
lateral_distance = distance_to_command_line(obstacle_xy, root_xy, cmd_world_unit)
```

If an avoid/slow-down obstacle is ahead of the root and inside the corridor, the linear velocity tracking weight is scaled down, for example:

```text
linear_velocity_tracking_weight *= 0.5
```

This condition should be computed from scanner height/semantic tensors with batched tensor operations. It must not iterate over obstacle prims or CPU object lists.

### 11.2 Yaw-Only And Mixed Yaw Risk

When translational command speed is near zero and yaw speed is active, there is no forward corridor. The planner evaluates the root/body/leg yaw swept region instead.

If a high small / large obstacle intersects the future yaw swept region, yaw tracking weight is scaled down, for example:

```text
yaw_tracking_weight *= 0.5
```

For mixed translation + yaw, both checks may be active:

- translation corridor can scale linear tracking;
- yaw swept region can scale yaw tracking.

This prevents yaw-only commands from sweeping the body, knees, or shanks into nearby obstacles.

## 12. Testing Strategy

T302 implementation must follow TDD:

1. add failing tests under `Go2Pvcnn/tests/`;
2. implement the minimum code slice;
3. verify the slice with `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`;
4. keep a real IsaacLab headless acceptance path, not only pure unit tests.

New test files may be added only under:

```text
Go2Pvcnn/tests/
```

Production source changes must modify existing files only.

### 12.1 COBBLESTONE_ROAD_CFG Complex Terrain

Non-obstacle terrain tests must not be flat-only. They must cover `COBBLESTONE_ROAD_CFG` from:

```text
Go2Pvcnn/go2_pvcnn/tasks/teacher_without_semantic_env_cfg.py
```

The terrain families include:

```text
flat
random_rough
hf_pyramid_slope
hf_pyramid_slope_inv
boxes
pyramid_stairs
pyramid_stairs_inv
```

The command matrix should include forward, backward, lateral, diagonal, yaw-only, translation+yaw, slow/medium/fast, and command-switch cases.

Acceptance checks:

- T300e regression metrics remain clean;
- root/body collision ratio is near zero;
- knee collision ratio is near zero;
- shank collision ratio is near zero;
- swing-foot collision ratio is near zero;
- planner runtime remains suitable for future batched RL use.

### 12.2 Flat Semantic Obstacle Course

Obstacle tests run on a flat course with small and large semantic obstacles. The scene design should reference:

```text
Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py
Go2Pvcnn/extension/semantic_course.py
```

The current scanner contract is:

```text
ground -> semantic 0
small obstacle -> semantic 1
large obstacle -> semantic 2
```

Small obstacle height is test-controlled:

- low-small crossing tests may scale/clamp small height to be `<= 0.3m` relative to root-projected ground;
- high-small avoidance tests set small height to be `> 0.3m` relative to root-projected ground;
- large obstacles use the large semantic obstacle profile.

Command cases must drive the root toward obstacles from multiple directions:

```text
forward toward obstacle
backward toward obstacle
lateral toward obstacle
diagonal toward obstacle
forward + yaw toward obstacle
yaw-only near obstacle
command switch into obstacle
```

Low-small acceptance:

- root trajectory starts on one side of the obstacle corridor and ends on the other side;
- root trajectory passes through the obstacle corridor rather than trivially avoiding far away;
- at least one swing foot trajectory passes over the obstacle neighborhood;
- swing-foot, knee, shank, and body collision ratios stay near zero;
- touchdown and stance semantic obstacle counts are zero;
- T300e regression metrics remain clean.

High-small and large acceptance:

- root trajectory does not pass through the obstacle footprint or forbidden corridor;
- root path shows lateral avoidance, slowdown, or yaw reduction when the obstacle blocks the command direction;
- velocity/yaw risk scaling is triggered when expected;
- swing-foot, knee, shank, and body collision ratios stay near zero;
- touchdown and stance semantic obstacle counts are zero;
- T300e regression metrics remain clean.

### 12.3 Required Metrics

Each acceptance run should report both existing T300e metrics and new T302 metrics:

```text
T300e:
  stance_airborne_ratio
  stance_max_gap
  support_stability
  ik_fk_residual
  root_foot_center_error
  support_plane_roll_pitch_error
  touchdown_anchor behavior

T302:
  root_body_collision_ratio
  knee_collision_ratio
  shank_collision_ratio
  swing_foot_collision_ratio
  touchdown_on_obstacle_count
  stance_on_obstacle_ratio
  low_small_cross_success
  high_small_avoid_success
  large_avoid_success
  linear/yaw weight scale min and mean
  linear/yaw weight scale trigger count
  linear/yaw trigger horizon index
  obstacle semantic class for each trigger
  planner runtime / wall time
```

T300e regression acceptance must reuse the latest accepted T300e command-matrix and root-cause probes as baseline. The T302 test suite should fail if stance airborne, stance gap, IK/FK residual, support stability, root-foot center, support-plane, or touchdown-anchor metrics regress beyond the existing accepted tolerances.

## 13. Expected File Scope

Production code should be limited to existing files, such as:

```text
Go2Pvcnn/extension/batch_mpc_planner/config.py
Go2Pvcnn/extension/batch_mpc_planner/terrain.py
Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py
Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py
Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py
Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py
Go2Pvcnn/extension/batch_mpc_planner/planner.py
Go2Pvcnn/extension/batch_mpc_planner/nominal.py
Go2Pvcnn/extension/semantic_course.py
```

`semantic_course.py` may be adjusted only if existing APIs cannot provide controlled low/high small obstacle heights for tests. Prefer existing configuration hooks before changing it.

New tests may be added under `Go2Pvcnn/tests/`, for example:

```text
Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py
```

## 14. Completion Criteria

T302 is complete when:

- backend tests prove GPU kinematics exposes knee/shank world samples;
- loss tests prove height-field collision, swing-foot clearance, touchdown/stance semantic penalties, and risk-based tracking scales are finite and vectorized;
- IsaacLab headless tests pass on `env_isaacsim`;
- COBBLESTONE complex-terrain command matrix preserves T300e metrics and reports near-zero new collision ratios;
- flat semantic obstacle tests show low small obstacles are crossed without stepping on them;
- high small and large obstacles are avoided or slowed for, including yaw-only cases;
- planner diagnostics include collision, obstacle, risk-scale, and runtime metrics.
