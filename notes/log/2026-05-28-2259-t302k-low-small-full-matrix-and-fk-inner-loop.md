# 2026-05-28 22:59 T302k Low-Small Full Matrix And FK Inner Loop

## Purpose

Record the final verification for the T302k low-small redesign follow-up: move FK collision/consistency into the Adam sampled loss path, correct the 第 0 条 diagnostic to evaluate optimized output after planning with matching rolling segment terrain, and validate the plane low-small command matrix.

## Stage

`extension/batch_mpc_planner` parametric MPC losses and IsaacLab diagnostic probe.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)
- [../todo/T302k-low-small-loss-redesign-plan.md](../todo/T302k-low-small-loss-redesign-plan.md)

## Command / Procedure

Local regression:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'parametric or low_small or touchdown_keepout or fk_body_leg or plane_root_z or teacher_mpc_semantic_env_raises_fk_body_leg_collision_weight or segmented_plane_low_small or ignore_non_crossing'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py Go2Pvcnn/extension/batch_mpc_planner/parametric.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/semantic_geometry.py Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/kinematics.py Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

IsaacLab full matrix:

```bash
PYTHONPATH=Go2Pvcnn CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 \
  --requested-n-frames 300 \
  --cycles 2 \
  --commands 'forward:0.50 0.00 0.00,backward:-0.50 0.00 0.00,left:0.00 0.50 0.00,right:0.00 -0.50 0.00,turn_left:0.00 0.00 1.00,turn_right:0.00 0.00 -1.00,diag_fl:0.35 0.35 0.00,diag_fr:0.35 -0.35 0.00,mixed_turn_l:0.35 0.25 1.00,mixed_turn_r:0.35 -0.25 -1.00' \
  > tmp/t302k-low-small-redesign/plane_fk_collision_full_gpu0_after_segmented_crossing_only.jsonl 2>&1
```

## Input Conditions

- Environment: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`.
- GPU selection: `CUDA_VISIBLE_DEVICES=0`.
- Device: `cuda:0`.
- Commands: forward, backward, left, right, turn_left, turn_right, diag_fl, diag_fr, mixed_turn_l, mixed_turn_r.
- Cycles: `2`.
- Requested frames: `300`.
- Probe semantics: 第 0 条 is post-optimization diagnostic only; FK semantic collision is counted only for legs whose target foot XY probes trigger `semantic == 1`.
- Rolling replan diagnostic uses per-segment terrain snapshots, not one stale or mutable scanner terrain.

## Key Metrics

- `test_batch_mpc_parametric.py`: `15 passed`.
- Backend focused subset: `33 passed, 90 deselected`.
- Additional focused subset after cleanup: `5 passed, 118 deselected`.
- Pycompile: pass.
- `git diff --check`: pass.
- IsaacLab cycle rows: `20`.
- Crossing-covered rows: `12`.
- Covered commands: `backward`, `diag_fl`, `diag_fr`, `forward`, `mixed_turn_l`, `mixed_turn_r`, `turn_left`, `turn_right`.
- Not-covered rows: `forward cycle1`, `backward cycle1`, `left cycle0`, `left cycle1`, `right cycle0`, `right cycle1`, `diag_fl cycle1`, `diag_fr cycle1`.
- Covered-row failures: `0`.
- Covered-row `fk_semantic_collision_count` max: `0`.
- Covered-row `fk_semantic_collision_rate` max: `0.0`.
- Covered-row `fk_semantic_min_clearance_over_semantic_m` min: `0.0`.
- Covered-row `planned_vs_fk_foot_error_crossing_leg_max_m` max: `0.0634172260761261`.
- Soft rows over preferred `0.05m` but within acceptable `0.08m`:
  - `turn_left cycle1`: `0.0634172260761261`
  - `turn_right cycle0`: `0.05196942761540413`
  - `diag_fr cycle0`: `0.05960841476917267`
  - `mixed_turn_r cycle0`: `0.06311921030282974`

## Result

Pass for the hard acceptance checks on rows that actually covered crossing legs:

```text
fk_semantic_collision_count == 0
fk_semantic_collision_rate == 0
fk_semantic_min_clearance_over_semantic_m >= 0
planned_vs_fk_foot_error_crossing_leg_max_m <= 0.08m
```

The stricter preferred `0.05m` FK deviation threshold is not met by four rows, so this remains a documented tuning risk rather than a reason to add an unapproved loss.

## Conclusion

The earlier mixed-turn and forward diagnostic collisions were mainly measurement-scope issues:

- rolling replan metrics were evaluating concatenated trajectory slices against stale/mutable terrain instead of the terrain used for each segment;
- 第 0 条 was counting FK collisions on non-crossing legs, while the approved design says only triggered legs are part of this diagnostic.

The implemented code now keeps FK body/leg collision and optimized-vs-FK consistency inside the sampled loss dictionary used during optimization, raises the semantic task FK collision weight to `120.0`, and keeps the post-optimization 第 0 条 diagnostic separate from optimizer losses.

## Follow-Up

Do not add new losses or hard repairs for the remaining soft FK deviation rows without user approval. Further debugging should inspect loss breakdown rows and tune confirmed weights/parameters first.

## Git Refs

- Baseline Ref: `a9f9b1c`
- Candidate Ref: `305fefe`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
