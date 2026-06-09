# T302p Direction Loss Wiring And Metric Boundary

## Purpose

Record the continuation after the low-small FK wiring pass. This pass localized the flat-left direction failure, wired existing direction-related losses/configuration, and identified the remaining per-leg metric as a separate metric/gait-phase boundary issue rather than a simple command-frame issue.

## Stage

MPC policy evaluation / parametric batch MPC sampled loss path / tracking eval configuration.

## Related Todo

- [../todo/T302p-mpc-command-frame-alignment-plan.md](../todo/T302p-mpc-command-frame-alignment-plan.md)
- Previous log: [2026-06-07-1104-t302p-low-small-fk-loss-wiring.md](2026-06-07-1104-t302p-low-small-fk-loss-wiring.md)

## Changes

- Wired existing `cfg.losses.progress.weight` and `cfg.losses.progress.min_progress_m` into existing `parametric_command_progress`.
- Added flat/empty semantic lateral endpoint direction error inside the existing `parametric_command_progress` key, gated off when semantic obstacles are present.
- Wired existing `swing_direction_loss()` into the parametric sampled path as `parametric_swing_direction`.
- Changed `parametric_swing_direction` to consume FK-realized `foot_pos`, matching the feet returned by the planner/eval metrics.
- Added tracking-eval-only direction weights and zero-obstacle curriculum, and synchronized that curriculum to `scene.terrain.semantic_obstacle_curriculum` so the terrain importer sees the same no-obstacle tracking setup.
- No new loss key was introduced to optimize a new metric; this pass wires and scopes existing loss/config families.

## Commands

Local focused verification:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git diff --check
```

Flat-left tracking smoke:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 300s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking --headless --device cuda:0 --num-envs 1 \
  --num-rounds 1 --max-steps 120 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode fixed \
  --command "0.0 1.0 0.0" \
  --terrain-rows 0 --terrain-cols 0 \
  --output-dir tmp/t302p-command-frame/left_120_after_terrain_curriculum_sync
```

Direction breakdown probe:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 300s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  tmp/t302p-command-frame/diagnose_plan_segment_direction.py \
  --mode tracking --headless --device cuda:0 --num-envs 1 \
  --num-rounds 1 --max-steps 120 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode fixed \
  --command "0.0 1.0 0.0" \
  --terrain-rows 0 --terrain-cols 0 \
  --output-dir tmp/t302p-command-frame/left_120_curriculum_sync_probe
```

Low-small compatibility regression:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 420s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 \
  --variants parametric_v1 \
  --requested-n-frames 300 \
  --warmup-steps 6 \
  > tmp/t302p-command-frame/low_small_after_direction_gated_gpu0.jsonl 2>&1
```

## Key Metrics

Local:

- Focused regression: `158 passed, 1 warning`.
- Pycompile: exit `0`.
- `git diff --check`: exit `0`.

Flat-left tracking smoke:

- Output: `tmp/t302p-command-frame/left_120_after_terrain_curriculum_sync/2026-06-07_12-06-02-104417`
- Exit: `0`
- Final root cosine: `0.9997996091842651`
- Final root lateral ratio: `0.02001659944653511`
- Final per-leg lateral ratios: `[0.04113054275512695, 0.5664749145507812, 0.46883082389831543, 0.04083206504583359]`
- Interpretation: root direction is fixed for this smoke; two middle legs still fail the strict whole-cache endpoint metric.

Direction breakdown probe:

- Output: `tmp/t302p-command-frame/left_120_curriculum_sync_probe/2026-06-07_12-07-05-456452`
- Final root cosine: `0.9999949932098389`
- Final root lateral ratio: `0.0031525271479040384`
- Final per-leg lateral ratios: `[0.05650303140282631, 0.6935920119285583, 0.7383890748023987, 0.07511633634567261]`
- `parametric_swing_direction` is nonzero and large in all plan segments: about `3.86-6.99`.
- `parametric_command_progress` is nonzero in all plan segments: about `0.059-0.123`.
- Interpretation: direction-related losses are active; remaining per-leg failure is not caused by command source mismatch or disabled direction losses.

Low-small compatibility:

- Output: `tmp/t302p-command-frame/low_small_after_direction_gated_gpu0.jsonl`
- Cycles: `5`
- `max_fk_semantic_collision_count = 0`
- `max_fk_semantic_collision_rate = 0.0`
- `max planned_vs_fk_foot_error_crossing_leg_max_m = 0.04200904071331024`
- Per command collisions:
  - `forward_v050 = 0`
  - `lateral_v050 = 0`
  - `diagonal_v050 = 0`
  - `mixed_yaw_v050 = 0`
  - `yaw100 = 0`

## Result

The root direction issue is fixed for flat-left tracking smoke and no longer appears to be a command-frame or async/cache interpretation problem. Low-small semantic compatibility still passes after the direction wiring.

T302p is not complete: the current per-leg endpoint metric remains open. The evidence points to a metric/gait-phase boundary issue: the eval metric compares each FK foot's whole cached segment endpoint displacement, while the existing gait-direction loss constrains swing-window motion. With alternating gait, two legs can have endpoint displacement dominated by stance/body-relative carry rather than a clean swing displacement.

## Follow-Up

- Keep T302p.2 active.
- Decide the per-leg acceptance metric before more planner tuning:
  - either measure moving/swing windows only, matching `swing_direction_loss()`;
  - or define a new explicit whole-segment per-leg planner contract, then evaluate whether that is compatible with gait and low-small crossing.
- Do not claim full flat all-direction acceptance yet.
- Keep low-small regression as a required guard for any subsequent direction/metric changes.

## Git Refs

- Baseline Ref: local dirty T302p after low-small FK loss wiring.
- Candidate Ref: local dirty worktree with existing progress/swing-direction wiring and tracking eval zero-obstacle synchronization.
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py`
  - `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - `Go2Pvcnn/tests/test_batch_mpc_backend.py`
