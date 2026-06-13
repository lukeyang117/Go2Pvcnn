# T302s model23600 controlled crossing eval

## Purpose

正式把 `model_23600.pt` 放到 flat-small avoidance 训练场景上做受控跨越评估，验证“速度命令 + 路径小障碍机会 + 跨越检测”是否能证明策略已经学会小障碍不踩并跨过去。

## Stage

- Checkpoint evaluation / flat-small low-small crossing behavior
- Related todo: [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --headless --device cuda:0 \
  --mode controlled_crossing \
  --run-dir ../teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-12_10-53-23 \
  --checkpoint model_23600.pt \
  --num-envs 16 \
  --num-rounds 1 \
  --max-steps 1000 \
  --terrain-rows 0,1,2,3,4,5,6,7,8,9 \
  --terrain-cols 0 \
  --crossing-speeds 0.6,0.8,1.0 \
  --crossing-lateral-offsets=-0.08,0.0,0.08 \
  --crossing-obstacles-per-env 24 \
  --output-dir logs/mpc_policy_eval/flat_small_model23600_controlled_crossing
```

Output directory:

```text
logs/mpc_policy_eval/flat_small_model23600_controlled_crossing/2026-06-12_18-07-11-931892
```

## Code Change Before Run

- `Go2Pvcnn/scripts/mpc_policy_eval.py` adds `controlled_crossing` mode.
- The mode uses `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg`, constrains terrain to flat, sets semantic curriculum to small-only, disables center safety hole for eval, and sets `crossing_obstacles_per_env=24`.
- Each env receives a fixed command from speed/lateral groups.
- Per-step metrics record path small-cell opportunities, root crossing, foot-over, touchdown-on-small, and real `semantic_contact_small` contacts.

## Verification

- `pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q`
  - Result: `23 passed`
- `python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py`
  - Result: exit `0`
- Real IsaacLab eval:
  - Result: exit `0`
  - Steps: `1000`
  - Envs: `16`

## Key Metrics

- Path-obstacle opportunity: `15/16`
- Root crossed locked obstacle: `14/15` opportunity envs
- Foot-over count: `0`
- Touchdown-on-small envs: `3`
- Real small-contact envs over opportunities: `10`
- Real small-contact envs overall: `11/16`
- Max small contact force: `200.662 N`
- Successful overpass: `0/15`
- Success rate over opportunities: `0.0`

By speed:

- `0.6 m/s`: opportunity `6`, success `0`
- `0.8 m/s`: opportunity `5`, success `0`
- `1.0 m/s`: opportunity `4`, success `0`

By lateral offset:

- `-0.08 m/s`: opportunity `6`, success `0`
- `0.0 m/s`: opportunity `4`, success `0`
- `0.08 m/s`: opportunity `5`, success `0`

## Conclusion

This controlled eval removes the previous “too few opportunities” ambiguity. The model sees enough path-small opportunities and mostly moves its root past them, but does not produce a measured foot-over event and frequently collides with small obstacles. `model_23600.pt` should not be considered to have learned clean low-small overpass.

## Follow-up

- Next tuning should change the learning signal, not just keep training this checkpoint blindly.
- Candidate directions: denser foot-over/clearance shaping tied to swing phase, stronger penalty for true small contact, and a staged first-layer curriculum where obstacles are placed on the commanded path before returning to random dense layouts.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py)
  - [../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py)
