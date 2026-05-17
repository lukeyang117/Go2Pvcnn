# T302 MPC Metric-Driven Tuning

## Purpose

Tune the active `batch_mpc_planner` MPC from numeric headless metrics rather than framework-only changes, focusing on the remaining COBBLESTONE swing-foot height-field collision ratio while preserving T302 body/knee/shank and semantic obstacle behavior.

## Stage

Production `Go2Pvcnn/extension/batch_mpc_planner` loss/config tuning plus real IsaacLab headless metric verification.

## Related Todo

- [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)

## Baseline Ref

- `769f7d4`

## Candidate Ref

- Working tree on top of `769f7d4`

## Key Files

- [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
- [../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py)
- [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
- [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
- [../../Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py](../../Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py)

## Commands

```bash
TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q

TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp MPC_T302_HEADLESS=1 MPC_TEST_DEVICE=cuda:0 \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py -q

TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp MPC_T302_HEADLESS=1 MPC_TEST_DEVICE=cuda:0 \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
  # real IsaacLab COBBLESTONE matrix scripts wrote JSONL under tmp/t302_mpc_metric_tuning/
PY

TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py

git diff --check
```

## Input Conditions

- Python env: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- Device: `cuda:0`
- `TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp`
- COBBLESTONE commands: forward, backward, lateral left/right, yaw left/right, forward+yaw left/right, diagonal forward left/right.
- Metric files kept under [../../tmp/t302_mpc_metric_tuning/](../../tmp/t302_mpc_metric_tuning/), not system `/tmp`.

## Key Metrics

- Baseline after earlier clearance tuning still had sparse swing-foot failures:
  - `final_cobblestone_after_min_clearance_010.jsonl`: worst `yaw_left`, `max_swing_foot_collision_ratio=0.0297029702970297`, root/knee/shank collision ratios `0.0`.
- Root cause pattern:
  - Collisions concentrated at swing/contact boundary frames such as `22-24` and `47-48`.
  - `stance_ground_loss` and `stance_semantic_obstacle_loss` were weighting all nonzero `contact_prob`, including frames that final `contact_state` exports as swing.
- TDD red/green:
  - `red_stance_threshold_losses.txt`: new stance threshold tests failed because losses had no `min_contact_prob`.
  - `green_stance_threshold_losses.txt`: losses pass after threshold gating.
  - `red_default_contact_threshold_040.txt` and `red_default_swing_weight_8.txt`: default parameter tests failed before config tuning.
  - `green_default_weight8_and_stance_losses.txt`: tuned defaults pass.
  - `red_default_swing_weight_12.txt`: exact default guard failed while production still used `weight=8.0`.
  - `red_swing_clearance_export_threshold.txt`: swing-clearance threshold test failed because the loss had no exported-swing threshold contract.
  - `green_swing_clearance_export_threshold.txt`: swing-clearance loss now filters/weights active swing frames with `min_swing_prob=1-contact_threshold`.
- Candidate sweeps:
  - `after_stance_threshold_candidate_matrix.jsonl`: `min12_ct40_w12` reached `max_swing_foot_collision_ratio=0.010309278350515464`.
  - `production_default_cobblestone_matrix.jsonl`: same `ct=0.40/min_clearance=0.12/weight=5` was not stable under a new terrain draw, `max=0.03`.
  - `after_stance_threshold_targeted_sweep.jsonl`: `weight=8,opt24` reduced targeted max to `0.010101010101010102`; `opt32` reached `0.0` but costs more runtime.
  - `full_matrix_weight8_vs_opt32.jsonl`: `weight=8,opt24` passed full matrix with `max=0.010101010101010102`; `opt32` was `0.0` but not selected to preserve RL rollout speed.
  - `sweep_opt24_weight_minheight.jsonl`: with `optimize_steps=24`, `weight=12,min_clearance=0.12,worst=12` was the only broad candidate in that sweep with `max_swing_foot_collision_ratio=0.0`; increasing nominal swing height alone did not help.
  - `production_default_weight12_cobblestone_matrix.jsonl`: `weight=12` alone improved but still had one diagonal boundary-frame failure, `max=0.01`, `min_swing_foot_clearance=-0.0031905174255371094`.
  - `production_default_swing_threshold_aligned_cobblestone.jsonl`: adding exported-swing threshold alignment reduced the remaining boundary deficit to a near-zero numerical edge, `max=0.010101010101010102`, `mean=0.0006734006734006734`, `min_swing_foot_clearance=-4.8160552978515625e-05`, with root/knee/shank ratios still `0.0`.
  - `sweep_threshold_aligned_weight16_targeted.jsonl`: targeted hard cases showed `w12_min12`, `w16_min12`, `w16_min13`, and `w16_min12_worst16` at `0.0`; `w20_min12` regressed to `0.01`, so larger weight is not monotonic.
- Selected production default:
  - `production_default_weight8_cobblestone_matrix.jsonl`
  - `contact_threshold=0.4`
  - `min_clearance=0.12`
  - `swing_clearance_terrain.weight=12.0`
  - `worst_deficit_weight=12.0`
  - `optimize_steps=24`
  - `swing_clearance_terrain_loss` uses `min_swing_prob=1-contact_threshold=0.6` and hard active weights so the loss matches exported `contact_state`.
  - final broad COBBLESTONE matrix after threshold alignment: `max_swing_foot_collision_ratio=0.010101010101010102`
  - final broad COBBLESTONE matrix after threshold alignment: `mean_swing_foot_collision_ratio=0.0006734006734006734`
  - final broad COBBLESTONE matrix after threshold alignment: `min_swing_foot_clearance=-4.8160552978515625e-05`
  - `max_root_bottom_collision_ratio=0.0`
  - `max_knee_collision_ratio=0.0`
  - `max_shank_collision_ratio=0.0`
- Final verification:
  - backend suite after final change: `56 passed`
  - real T302 headless pytest after final change: exit code `0` with the known empty stdout caveat
  - `py_compile`: exit code `0`
  - `git diff --check`: exit code `0`

## Result

Pass with metric-driven tuning.

## Conclusion

The remaining COBBLESTONE collision issue was not solved by framework additions alone. The useful fix was to align stance-only losses and swing-clearance losses with the final exported `contact_state` threshold, then tune default swing clearance to `contact_threshold=0.40`, `min_clearance_m=0.12`, and `weight=12.0` while leaving `optimize_steps=24` for speed. `optimize_steps=32` produced cleaner earlier metrics but was intentionally not selected because the user emphasized future RL throughput. Higher weight was not monotonic in the targeted sweep (`w20` regressed), so the selected production setting is `w12 + threshold alignment`.

## Follow-Up

- `test_mpc_body_leg_collision_headless.py -q` exits `0`, but stdout remains empty due Isaac/pytest buffering observed in prior logs; JSONL metric scripts are the stronger evidence for the final numeric claims.
- The final broad COBBLESTONE matrix still has one near-zero negative swing-foot clearance sample (`-4.8e-05m`), so future strict metrics should use a small numerical tolerance or add a longer targeted rerun before claiming mathematically exact zero.
- Longer command-switch/yaw and 4096-scale runtime counters remain the next confidence layer if T302 is promoted into full RL rollout.

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: working tree on top of `769f7d4`
