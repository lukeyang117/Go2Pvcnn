# T302u Semantic Map Contact Collision Plan

## Current State

Global semantic filtered contact became too expensive after flat-small objects were correctly generated across all 20 columns. The train/play/viewer default path must stop loading `semantic_contact_small` and `semantic_contact_large`.

## Goal

Use ordinary robot `contact_forces` plus the 0.01m semantic/elevation scanner map to infer semantic collisions, and fold the penalty into the existing body-part clearance reward.

## Open Children

- [x] T302u.1 Add tensor-level map-contact inference tests and helper.
- [x] T302u.2 Combine map-contact penalty into `semantic_body_part_clearance_reward`.
- [x] T302u.3 Replace curriculum small-collision bookkeeping source with map-contact inference.
- [x] T302u.4 Remove global semantic contact sensor loading from train/play/viewer defaults.
- [x] T302u.5 Run focused local tests, pycompile, and an `env_isaacsim` smoke.
- [x] T302u.7 Re-enable `lin_vel_cmd_levels` for flat-small `GoalAnchoredVelocityCommand`.
- [x] T302u.6 Run a 1024-env startup check after older stuck jobs are stopped.
- [x] T302u.8 Replace dense MPC semantic avoidance with proximity-field sampling so 1024 RL envs can use 1024 MPC envs without CUDA OOM.
- [x] T302u.9 Cache low-small touchdown keepout circles once per replan to remove the 1024/1024 MPC collection long tail.
- [x] T302u.10 Remove flat-small `lin_vel_cmd_levels` again per user request while preserving `GoalAnchoredVelocityCommand`.
- [x] T302u.11 Relax flat-small `bad_orientation` termination threshold for small-obstacle crossing.

## Key Files

- [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
- [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)
- [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
- [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
- [../../Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py](../../Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py)
- [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
- [../../Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py](../../Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py)
- [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
- [../../Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py](../../Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py)

## Related Design

- [../../docs/superpowers/specs/2026-06-13-semantic-map-contact-collision-design.md](../../docs/superpowers/specs/2026-06-13-semantic-map-contact-collision-design.md)

## Related Logs

- [../log/2026-06-13-2348-semantic-map-contact-collision-design.md](../log/2026-06-13-2348-semantic-map-contact-collision-design.md)
- [../log/2026-06-14-0008-flat-small-1024-simulation-start-stall.md](../log/2026-06-14-0008-flat-small-1024-simulation-start-stall.md)
- [../log/2026-06-14-0035-semantic-map-contact-collision-implementation.md](../log/2026-06-14-0035-semantic-map-contact-collision-implementation.md)
- [../log/2026-06-15-1649-flat-small-goal-anchored-speed-curriculum.md](../log/2026-06-15-1649-flat-small-goal-anchored-speed-curriculum.md)
- [../log/2026-06-17-flat-small-remove-speed-curriculum-again.md](../log/2026-06-17-flat-small-remove-speed-curriculum-again.md)
- [../log/2026-06-17-flat-small-bad-orientation-threshold.md](../log/2026-06-17-flat-small-bad-orientation-threshold.md)
- [../log/2026-06-16-mpc-proximity-field-semantic-avoidance.md](../log/2026-06-16-mpc-proximity-field-semantic-avoidance.md)
- [../log/2026-06-16-mpc-touchdown-keepout-runtime-cache.md](../log/2026-06-16-mpc-touchdown-keepout-runtime-cache.md)

## Next Step

Use the 1024/1024 MPC path for the next flat-small training check. The memory gate passes and the reproduced `mixed_zero_split_ms` long tail is fixed; remaining runtime cost is the expected `optimize_steps=24` parametric solve cost of about `2.6-2.9s` per 1024-env replan.

## Node Details

### T302u.8 MPC Proximity-Field Semantic Avoidance

The old `parametric_semantic_avoidance` implementation built dense root/foot/touchdown pairwise tensors against the full `150 x 150 = 22500` semantic grid for every horizon frame. The replacement keeps the same loss key and existing loss surface, builds one `[B,1,150,150]` soft proximity field from high-small/large candidate cells, and samples root/foot/touchdown risk with differentiable `grid_sample`.

Verification used `env_isaacsim`: focused MPC tests passed, `TeacherElevationTrajectoryMpcSemanticEnvCfg` ran `1024` RL envs with `1024` MPC envs for `30` steps on GPU1, `max_sampled_plan_count_seen=1024`, `cuda_max_memory_allocated=7431256576`, and `cuda_max_memory_reserved=9265217536`. No loss name was added and low-small thresholds were not relaxed.

### T302u.9 MPC Touchdown Keepout Runtime Cache

The 1024/1024 memory fix exposed a separate runtime long tail: later replans sometimes showed outer `plan.mixed_zero_split_ms` around `35s`. Term-level profiling showed `mixed_zero_split` was only the wrapper; the inner nonzero subset planner spent `33820ms` in `term.touchdown_clearance`.

The root cause was `parametric_touchdown_keepout_loss()` rebuilding `low_small_component_circles()` inside every sampled-frame loss call. Low-small component circles depend on the semantic map and scanner range, not on optimizer variables, so the helper is now built once per replan and passed through the existing sampled loss path. No new loss key or loss weight was added.

Verification used `env_isaacsim`: focused MPC regression `169 passed`, pycompile exit `0`, and a real `TeacherElevationTrajectoryMpcSemanticEnvCfg` `1024` RL / `1024` MPC / `60`-step probe completed with `epoch_seconds=15.2575`, `replan_event_count=3`, `max_sampled_plan_count_seen=1024`, CUDA allocated `7.55GB`, reserved `9.29GB`. The same reproduced slow profile before the cache had `term.touchdown_clearance_ms=33820.227` and `epoch_seconds=45.894`.

### T302u.10 Flat-small Speed Curriculum Removed Again

The flat-small cfg briefly re-enabled `lin_vel_cmd_levels` to make `GoalAnchoredVelocityCommand` compatible with the old velocity curriculum. The user then explicitly requested removing the speed curriculum from `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg`.

The current contract is now: base semantic MPC cfg keeps `lin_vel_cmd_levels`; flat-small train cfg sets `self.curriculum.lin_vel_cmd_levels = None`; flat-small still uses `GoalAnchoredVelocityCommand`; terrain curriculum remains active. Real 4-env `env_isaacsim` smoke confirms Curriculum Manager has only `terrain_levels`.
