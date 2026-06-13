# T302s model28900 controlled crossing eval

## Purpose

Use the latest checkpoint from `2026-06-12_19-05-27` to test whether the added foot-over reward and longer training produced clean low-small overpass behavior.

## Stage

- Checkpoint evaluation / flat-small low-small crossing behavior
- Related todo: [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --headless --device cuda:0 \
  --mode controlled_crossing \
  --run-dir /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-12_19-05-27 \
  --checkpoint model_28900.pt \
  --num-envs 16 \
  --num-rounds 1 \
  --max-steps 1000 \
  --terrain-rows 0,1,2,3,4,5,6,7,8,9 \
  --terrain-cols 0 \
  --crossing-speeds 0.6,0.8,1.0 \
  --crossing-lateral-offsets=-0.08,0.0,0.08 \
  --crossing-obstacles-per-env 24 \
  --output-dir logs/mpc_policy_eval/flat_small_190527_model28900_controlled_crossing
```

Output directory:

```text
logs/mpc_policy_eval/flat_small_190527_model28900_controlled_crossing/2026-06-13_10-40-40-696479
```

## Input Conditions

- Gym id: `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Flat-Small-Avoidance-Go2-v0`
- Envs: `16`
- Steps: `1000`
- Commands: controlled speed/lateral grid, speeds `0.6,0.8,1.0`, lateral offsets `-0.08,0.0,0.08`
- Controlled crossing eval forces dense path-small opportunities with `24` obstacles/env.

## Key Metrics

- Path-obstacle opportunity: `15/16`
- Root crossed locked obstacle: `14/15` opportunity envs
- Foot-over count: `0`
- Foot-over event count: all envs `0`
- Touchdown-on-small envs: `4/15` opportunities
- Real small-contact envs over opportunities: `7/15`
- Real small-contact envs overall: `7/16`
- Max small contact force: `609.303 N`
- Successful overpass: `0/15`
- Success rate over opportunities: `0.0`
- Command source max error: `0.0`
- Planned root direction cosine: `0.999653`

By speed:

- `0.6 m/s`: opportunity `5`, success `0`
- `0.8 m/s`: opportunity `6`, success `0`
- `1.0 m/s`: opportunity `4`, success `0`

By lateral offset:

- `-0.08 m/s`: opportunity `5`, success `0`
- `0.0 m/s`: opportunity `5`, success `0`
- `0.08 m/s`: opportunity `5`, success `0`

## Result

Diagnostic pass. The checkpoint is stable enough to move through the controlled path-obstacle scene, and opportunity coverage is sufficient. It does not satisfy the overpass definition: no environment produced a measured foot-over event, and clean success remains `0/15`.

Compared with `model_23600.pt`, real small-contact count improved from `11/16` overall to `7/16`, but the core behavior did not change: the robot still gets its root past low-small obstacles without learning the intended foot-over motion.

## Conclusion

Longer training plus the sparse `semantic_foot_over_clearance` reward did not solve low-small overpass. The next training design should make the crossing behavior a staged, dense learning signal rather than relying on rare foot-over reward hits inside random locomotion.

## Follow-up

- Treat TensorBoard locomotion success and terrain curriculum success as insufficient for this task.
- Redesign the training layer so low rows create repeated path-aligned obstacle opportunities.
- Make reward/metrics distinguish approach, lift before obstacle, above-obstacle clearance, landing after obstacle, and no real small contact.

## Git Refs

- Baseline Ref: `23182ce`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
