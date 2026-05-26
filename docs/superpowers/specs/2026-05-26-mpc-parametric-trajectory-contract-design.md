# MPC Parametric Trajectory Contract Design

## Context

The current `extension/batch_mpc_planner` optimizes dense per-frame foot position residuals. Recent T302h/T302i/T302j evidence shows that this representation is hard to tune for low-small obstacle crossing:

- the optimized Cartesian foot path can be discontinuous;
- exported touchdown markers can disagree with the swing endpoint;
- planned foot targets can be unreachable after clamped Go2 IK;
- terrain-height changes require many local loss patches;
- scalar loss stacking trades off mixed-yaw direction, posture, reachability, and continuity.

The new design changes the planner contract from optimizing a discrete trajectory to optimizing the parameters that generate a continuous root and foot trajectory. The planner still samples 25 frames for loss evaluation and output, but the 25 frames are decoded from a smaller set of structured variables.

## Goals

- Make low-small obstacle crossing easier to implement and tune.
- Reuse the same trajectory representation across different terrain heights.
- Keep touchdown targets grounded by construction: optimize only touchdown `xy`; derive touchdown `z` from the height map.
- Make planner output match what the clamped-IK robot can realize.
- Preserve diagonal gait semantics unless a later design explicitly changes gait family.
- Reduce or remove losses whose only purpose was to repair dense per-frame foot residual spikes.

## Non-Goals

- Do not redesign high-level training, reward cache ABI, or semantic course generation in this spec.
- Do not change the external command contract.
- Do not add a free leg-order optimizer. The first version keeps diagonal swing order and optimizes timing/width only.
- Do not make viewer markers hide planner errors. Diagnostics should still expose target-vs-realized mismatch.

## Proposed Architecture

The planner has four stages:

1. Build semantic and terrain context from current IsaacLab state, command, height map, and semantic map.
2. Initialize compact optimization variables for root goal, touchdown `xy`, swing curve shape, and diagonal timing.
3. Decode variables into continuous root and foot curves, then sample 25 frames.
4. Run IK, clamp joints, FK the realized feet, compute losses on sampled frames, and export realized trajectory plus diagnostics.

The core change is that `foot_pos_residual [B,T,4,3]` is replaced by per-leg curve parameters. Dense frame samples become derived values, not optimizer variables.

## Optimization Variables

### Foot Variables

- `touchdown_xy [B,4,2]`
  - Optimized as command-frame deltas relative to the current realized IsaacLab foot, then decoded to world `xy`.
  - The command frame uses translational command direction as `along`; its perpendicular axis is `lateral`.
  - If translational command speed is below threshold, use root yaw/body-forward as the fallback frame and disable low-small foot-over requirements.
  - `touchdown_z = height_at(terrain, touchdown_xy)`.
  - No optimized touchdown z offset in the first version.

- `swing_clearance [B,4]`
  - Height margin above terrain and, for low-small crossing, above obstacle top.

- `bezier_a_raw [B,4]`, `bezier_b_raw [B,4]`
  - Converted with sigmoid into bounded `a,b`.
  - Suggested initial bounds: `[0.15, 0.85]`; default around `0.35`.

- `lateral_bias_start [B,4]`, `lateral_bias_end [B,4]`
  - Controls curve shape in the command-frame lateral axis.
  - Low-small crossing losses keep the crossing foot inside the obstacle lane instead of side-bypassing.

### Root Variables

- `root_goal_xy [B,2]`
  - Optimized as a command-frame delta from current root `xy`, then decoded to world `xy`.
- `root_goal_yaw [B]`
- `root_bezier_c1_raw [B]`, `root_bezier_c2_raw [B]`
- `root_lateral_bias_start [B]`, `root_lateral_bias_end [B]`
- `root_height_offset [B]`

Root z is terrain-relative:

```text
root_z(t) = support_height(root_xy(t)) + nominal_base_height + root_height_offset
```

The implementation may start with a simpler support-height approximation and refine it later.

### Gait Variables

- `swing_center [B,4]`
- `swing_width [B,4]`
- `diagonal_phase_offset [B]`
  - Bounded phase adjustment applied to the fixed diagonal pair pattern.

Diagonal pair order remains fixed. Timing variables decide where the diagonal swings fall inside the horizon.

## Curve Definition

### Foot XY Curve

For each leg:

```text
P0 = current IsaacLab foot xy
P3 = world_xy(current_foot_xy, optimized_touchdown_delta_command_frame)
step = P3 - P0
dir = normalize(command_xy) or normalize(step) when command is too small
lat = perpendicular(dir)
L = ||step||

P1 = P0 + dir * (a * L) + lat * lateral_bias_start
P2 = P3 - dir * (b * L) + lat * lateral_bias_end

foot_xy(t) = cubic_bezier(P0, P1, P2, P3, t)
```

The curve starts from the current realized IsaacLab foot, not from the previous planned foot.

### Foot Z Curve

```text
touchdown_z = height_at(terrain, touchdown_xy)
base_z(t) = lerp(current_foot_z, touchdown_z, t)
arc_z(t) = 4 * t * (1 - t) * swing_clearance
terrain_req(t) = height_at(terrain, foot_xy(t)) + terrain_clearance
obstacle_req(t) = obstacle_top + crossing_clearance, gated to low-small obstacle lane

target_foot_z(t) = max(base_z(t) + arc_z(t), terrain_req(t), obstacle_req(t))
```

The first implementation should use a differentiable smooth maximum where practical.

Foot-above-root should be handled as a feasibility or barrier term:

```text
target_foot_z(t) <= root_z(t) - margin
```

If this conflicts with obstacle clearance, the root trajectory should rise or the touchdown/curve should change; the foot should not solve the conflict by becoming unrealistically high.

### Root Curve

```text
R0 = current root xy
R3 = world_xy(current_root_xy, optimized_root_goal_delta_command_frame)
root_xy(t) = cubic_bezier(R0, R1, R2, R3, t)
root_yaw(t) = cubic interpolation from current yaw to root_goal_yaw
root_z(t) = terrain/support-relative height
```

Root Bezier control points use bounded `c1/c2` and lateral biases analogous to the foot curve.

## Sampling And Loss Evaluation

The optimizer evaluates the decoded continuous trajectory by sampling 25 frames:

```text
curve params
-> sample root_target[25], foot_target[25]
-> solve IK
-> clamp joints
-> FK realized feet, knees, shanks
-> compute losses
```

Losses should prefer FK-realized feet for physical and semantic checks. Target-vs-FK mismatch is a reachability loss and diagnostic.

## Output Contract

`MpcPlannerResult` should export the trajectory that the robot can realize:

- `root_pos/root_rpy`: sampled root curve.
- `joint_angles`: clamped IK joint sequence.
- `foot_pos`: FK-realized foot curve from `joint_angles`.
- `planned_touchdown_w`: grounded touchdown derived from optimized `touchdown_xy`.
- `touchdown_seq`: grounded touchdown events.

Diagnostics should retain:

- `target_foot_pos` or equivalent internal debug tensor;
- `target_touchdown_w`;
- `target_vs_fk_error`;
- raw joint-limit violation and clamp saturation.

The viewer and reward-facing default should not display unreachable target feet as if they were realized feet.

## Losses To Keep

- Touchdown support:
  - touchdown `xy` on ground semantic;
  - `touchdown_z = height_at(touchdown_xy)`;
  - slope/support patch acceptable;
  - no touchdown on small/large semantic.

- IK/FK reachability:
  - target foot curve close to FK after clamped IK;
  - raw joint-limit violation;
  - leg workspace bounds.

- Body and leg collision:
  - root bottom vs height field;
  - knee/shank/foot vs height field.

- Semantic collision:
  - high-small/large avoidance for root/body/legs/feet;
  - low-small stance/touchdown exclusion;
  - low-small swing-over is allowed only with clearance.

- Command progress and direction:
  - positive progress along command direction;
  - bounded lateral drift;
  - weaker speed magnitude tracking inside approach/cross windows.

- Gait/contact consistency:
  - diagonal pair timing;
  - minimum support;
  - contact state consistent with swing windows.

## Losses To Rewrite

- Smoothness:
  - replace frame-to-frame dense residual smoothing with curve velocity, acceleration, curvature, and length-ratio penalties.

- Swing clearance:
  - compute against terrain and semantic obstacle envelopes on curve samples.

- Low-small foot-over:
  - require crossing from obstacle front to back in command frame;
  - require lateral lane occupancy;
  - require clearance above obstacle top;
  - require grounded touchdown after the obstacle.

- Root height:
  - use terrain/support-relative root height and body clearance, not free per-frame z residuals.

## Losses To Delete Or Demote

- Dense `foot_pos_residual` priors.
- Per-frame foot stepcap patches designed only to suppress residual spikes.
- Postprocess-style "farthest touchdown export" repairs.
- Viewer/marker losses that encourage target feet while ignoring FK-realized feet.
- Redundant scalar semantic soft-field stacks that duplicate a clearer touchdown/root/curve geometry contract.

These may remain temporarily as diagnostics or compatibility gates during migration, but they should not be part of the final parametric contract.

## Low-Small Crossing Behavior

Low-small crossing should be generated from:

- current IsaacLab foot positions;
- command direction;
- obstacle center, footprint, and top height;
- terrain support around the landing zone;
- leg reachability.

The planner should decide which diagonal swing can cross during the horizon using the existing diagonal gait timing. Non-crossing legs may approach or stay anchored. A successful crossing requires FK-realized feet to pass over the obstacle lane and land on ground beyond it.

## High-Small And Large Behavior

High-small and large obstacles should continue to use avoidance behavior, but with the root curve and touchdown `xy` as the primary knobs:

- shape root goal/lateral bias away from the obstacle;
- choose touchdown `xy` on safe ground;
- keep body and leg samples clear.

The implementation should preserve the current command-shaping behavior until the parametric root/touchdown representation can replace it safely.

## Testing And Acceptance

Local unit tests should verify:

- touchdown z is always derived from `height_at`;
- foot Bezier curves start at current foot and end at grounded touchdown;
- `a,b,c1,c2` stay bounded;
- pure yaw does not require low-small foot-over;
- output `foot_pos` matches FK from exported joints.

IsaacLab probe acceptance should cover:

- low-small forward, lateral, diagonal, mixed yaw, and pure yaw;
- high-small and large non-regression;
- terrain height variation;
- rolling 25-frame replans over a 300-step run.

Primary metrics:

- `touchdown_ik_fk_error_max`;
- `terminal_planned_vs_fk_foot_error_max`;
- FK swing continuity and boundary jump ratios;
- low-small FK foot-over success;
- stance/touchdown semantic contact rates;
- body/knee/shank/foot collision rates;
- command direction cosine and lateral drift;
- planned/FK foot-above-root violations.

## Migration Plan Summary

1. Add parametric variable and decode helpers behind a config/debug switch.
2. Keep root dense residuals only if needed for the first probe; remove dense foot residuals first.
3. Export FK-realized feet while retaining target-vs-realized diagnostics.
4. Run low-small focused probes before high-small/large non-regression.
5. Once accepted, make the parametric decoder the default and delete obsolete dense foot residual losses.
