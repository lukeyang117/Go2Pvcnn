# T302i Low-Small Reachable Crossing Loss Design

## Context

The low-small rolling viewer failure is not a viewer marker bug. Focused trace shows the robot joint writeback and Isaac readback match internal FK, while the exported Cartesian `foot_pos` / touchdown target can be unreachable after IK clamp. The worst reproduced frame requires raw `FL_calf=-0.0`, which clamps to the Go2 upper limit `-0.8378`, producing about `0.2867m` planned-vs-realized foot error.

The viewer should keep visualizing the planner output. The planner must instead find trajectories whose logical crossing targets are also IK/FK feasible.

## Goals

- Keep the fix loss-only inside `extension/batch_mpc_planner`; no phase/state-machine split.
- Support all planar command directions: `vx`, `vy`, and mixed `vx/vy/yaw`.
- Let a rolling segment approach the obstacle safely without forcing crossing in that same segment.
- Cross low-small obstacles only when the robot is in a reasonable crossing window.
- Judge crossing with FK-realized feet after clamped IK, not with unreachable decoded foot targets.
- Prevent false success by lowering the base, over-abducting legs, or walking around with spider-like spread.

## Non-Goals

- Do not change viewer rendering to hide the mismatch.
- Do not export FK-realized feet as a visual patch before improving planner feasibility.
- Do not add a discrete phase machine or selector/postprocess repair.
- Do not require pure yaw commands to cross low-small obstacles.

## Command-Aligned Frame

All low-small approach/crossing tests operate in a local command frame:

- `translation_norm = sqrt(vx^2 + vy^2)`.
- If `translation_norm` is above a small threshold, `forward_axis = normalize([vx, vy])`.
- `lateral_axis` is perpendicular to `forward_axis`.
- Obstacle, root, and FK-realized foot positions are projected as:
  - `along = dot(pos_xy - obstacle_center_xy, forward_axis)`.
  - `lateral = dot(pos_xy - obstacle_center_xy, lateral_axis)`.

This avoids hard-coding world `x` and keeps forward, lateral, diagonal, and mixed yaw commands on the same semantics.

## Yaw Handling

- Translation-dominant commands activate approach/crossing losses in the command-aligned frame.
- Mixed translation + yaw keeps approach/crossing active, but adds stability pressure so yaw does not sweep stance or touchdown feet onto the obstacle.
- Pure or near-pure yaw disables low-small crossing pressure and keeps semantic no-contact, base stability, and bounded foot-spread losses active.

## Approach-Safe Loss

Approach-safe is active when the robot is still before the crossing window. It allows the segment to move closer and leave crossing to a later replan.

Signals:

- positive progress along `forward_axis`, but weak speed-magnitude tracking;
- root, stance feet, and touchdown points do not contact small semantic cells;
- FK-realized feet after clamped IK do not penetrate or step on small obstacles;
- IK/FK residual and joint-limit margin remain bounded, especially calf upper-limit saturation;
- root height stays within a nominal band;
- roll/pitch stay bounded;
- foot lateral spread, hip abduction, and foot-to-root lateral offsets stay bounded.

The intent is: move closer, stay stable, and do not create an unreachable crossing target.

## Cross-When-Ready Loss

Crossing pressure becomes strong only inside a soft crossing window around the obstacle along-axis. The window should be continuous, not a discrete phase.

A valid low-small crossing requires at least one FK-realized swing foot to:

- move from the front side of the obstacle to the far side in the command-aligned `along` coordinate;
- pass above obstacle top plus clearance near the obstacle lane;
- keep lateral distance inside the obstacle lane instead of bypassing around it;
- touchdown on safe ground beyond the obstacle, not on small semantic cells;
- preserve stance-foot semantic safety.

Velocity magnitude tracking is reduced while crossing; direction and bounded lateral drift remain active.

## Anti-Cheat Constraints

The loss must reject visually wrong but numerically tempting solutions:

- Base lowering: penalize root height below a nominal minimum and base-bottom clearance below threshold.
- Body tilt: penalize excessive roll/pitch through the crossing window.
- Spider spread: penalize excessive foot lateral offset from nominal hip lane, excessive support width, and hip abduction near limits.
- Unreachable feet: penalize large `||FK(clamped IK(root, foot_pos)) - foot_pos||` and raw IK joint-limit violation, with emphasis on calf upper-limit saturation.
- Side bypass: penalize crossing credit when FK-realized foot lateral distance exceeds the small-obstacle lane limit.

## Expected Rolling Behavior

With repeated 25-frame replans, the expected behavior is:

1. If far from the small obstacle, plan a safe approach segment.
2. On later replans, once inside the crossing window, produce a reachable FK-realized foot-over trajectory.
3. After crossing, stabilize and leave without semantic contact.

This keeps the behavior loss-only while allowing "walk closer, then cross" across rolling replans.

## Verification Plan

Create a new IsaacLab probe instead of extending the old test entrypoint in-place. It may reuse helpers from `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`, but the new file should make the T302i acceptance contract explicit, for example:

- `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`
- command output under `tmp/t302i-viewer-realized-foot-mismatch/`
- launched with `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- real IsaacLab startup required for final acceptance

The probe must cover low-small forward, lateral, diagonal, mixed yaw, and pure yaw cases.

### Required Metrics

1. Swing foot continuity:
   - `fk_swing_foot_step_max_to_median`: max frame-to-frame FK-realized swing-foot step divided by median step.
   - `fk_swing_foot_accel_max_to_mean`: max FK-realized swing-foot second difference divided by mean.
   - `replan_boundary_fk_foot_step_to_median`: FK-realized foot jump at rolling segment boundaries divided by median in-segment step.
   - Acceptance: no large boundary jump or acceleration spike compared with the current reproduced bad range.

2. Touchdown IK/FK consistency:
   - `touchdown_ik_fk_error_max`: max distance between planned touchdown and `FK(clamped IK(root, touchdown_or_foot_target))`.
   - `terminal_planned_vs_fk_foot_error_max`: max distance between exported planned foot and FK-realized foot at segment terminal frames.
   - `raw_ik_joint_limit_violation_max`: max raw IK violation beyond joint limits, with calf upper-limit saturation reported separately.
   - Acceptance: touchdowns and planned feet used for crossing must be feasible; the planner must not place touchdown markers at unreachable Cartesian targets while the actual FK foot lands elsewhere.

3. No small-obstacle contact:
   - `fk_stance_on_small_rate`: FK-realized stance feet on small semantic cells.
   - `fk_touchdown_on_small_rate`: touchdown targets on small semantic cells.
   - `fk_foot_small_penetration_rate`: FK-realized feet below small-obstacle top surface or inside obstacle footprint.
   - Acceptance: all three are `0` for low-small crossing and approach cases.

4. Direction tracking without speed-magnitude forcing:
   - `command_direction_cosine`: cosine similarity between net root displacement and planar command direction.
   - `along_progress_m`: root progress along command direction.
   - `speed_magnitude_tracking_error`: reported but not a hard crossing failure during approach/crossing windows.
   - Acceptance: translation commands must move in the requested direction with bounded lateral drift; they do not need to match requested speed magnitude exactly while approaching/crossing.

5. Foot-over obstacle arc:
   - `fk_foot_over_low_small_success`: at least one FK-realized swing foot crosses from obstacle-front to obstacle-back in command-aligned `along`.
   - `fk_foot_over_low_small_min_lateral`: minimum absolute lateral distance to obstacle center while the foot is over the obstacle lane.
   - `fk_foot_over_low_small_clearance_max`: max FK-realized foot height over obstacle top during the crossing window.
   - `fk_foot_over_low_small_lift_then_land`: boolean requiring the same foot to rise above obstacle top plus clearance, then later descend to safe ground beyond the obstacle.
   - `fk_foot_over_low_small_touchdown_after`: touchdown lands beyond the obstacle in `along` and not on small semantic cells.
   - Acceptance: crossing credit requires the foot to pass over the obstacle from above and then land; side bypass and unreachable decoded-foot arcs do not count.

6. Anti-cheat stability:
   - `root_height_min`, `base_bottom_clearance_min`, `roll_pitch_abs_max`.
   - `foot_lateral_spread_max`, `foot_to_root_lateral_offset_max`, `hip_abduction_limit_margin_min`.
   - Acceptance: no low-base crawl, excessive body tilt, or spider-like leg spread.

7. Yaw-specific behavior:
   - Translation + yaw cases use command-aligned translation metrics plus yaw stability/contact metrics.
   - Pure yaw reports no-contact and stability metrics, but `fk_foot_over_low_small_success` is not required.

### Acceptance Signals

- `touchdown_ik_fk_error_max` and `terminal_planned_vs_fk_foot_error_max` drop materially from the reproduced `0.286-0.375m` range.
- `fk_stance_on_small_rate=0`, `fk_touchdown_on_small_rate=0`, `fk_foot_small_penetration_rate=0`.
- FK-realized foot-over succeeds for translation commands once the rolling planner reaches the crossing window.
- Approach-only segments before the crossing window are allowed if they make safe positive progress.
- Pure yaw does not force crossing and remains contact-free.
- base height, roll/pitch, lateral spread, and joint-limit metrics stay within thresholds.

## Open Risk

The exact soft gate distances and weights need empirical tuning under `env_isaacsim`. The first implementation should be probe-only or narrowly gated until low-small, high-small, large, and yaw regressions are checked.
