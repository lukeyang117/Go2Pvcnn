# 2026-06-30 Flat-Small 15:34 Foot-Over Weight 1 Eval

## Purpose

Inspect run `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-29_15-34-52`, where `semantic_foot_over_clearance.weight` was increased from `0.12` to `1.0`, and test whether the stronger signal creates the desired low-small obstacle overpass behavior.

## Stage

Training metrics / controlled crossing checkpoint evaluation / flat-small semantic avoidance.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

Read TensorBoard scalars from:

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-29_15-34-52/events.out.tfevents.1782718642.enine.1102716.0
```

Then ran controlled crossing eval for:

- `model_18600.pt`: best TensorBoard stability window around step `18643`
- `model_19699.pt`: latest / highest foot-over scalar window

Representative command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 timeout 900s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode controlled_crossing \
  --headless \
  --device cuda:0 \
  --num-envs 16 \
  --num-rounds 1 \
  --max-steps 300 \
  --run-dir unused \
  --checkpoint /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-29_15-34-52/model_19699.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.6 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/flat_small_20260629_153452_model19699_controlled_crossing_absckpt
```

## Input Conditions

Saved config/tag confirms:

- `semantic_foot_over_clearance.weight: 0.12 -> 1`
- `entropy_coef=0.002`
- `bad_orientation.limit_angle=1.1`
- dense approach settings unchanged:
  - `approach_sample_count=9`
  - `approach_min_distance_m=0.05`
  - `approach_along_sigma_m=0.18`
  - `approach_lateral_sigma_m=0.16`
- stability weights unchanged:
  - `flat_orientation_l2=-3.5`
  - `base_angular_velocity=-0.12`
  - `feet_slide=-0.18`

## TensorBoard Metrics

Compared with `2026-06-27_20-47-47` (`weight=0.12`), the foot-over scalar is much stronger.

Last-100:

- `Train/mean_episode_length`: `954.00`
- `Train/mean_reward`: `-3.84`
- `Curriculum/terrain_levels/mean_terrain_level`: `6.118`
- `Policy/mean_noise_std`: `0.433`
- `Episode_Termination/bad_orientation`: `0.177`
- `Episode_Termination/base_contact`: `0.0285`
- `Episode_Reward/semantic_foot_over_clearance`: `0.05425`
- `Episode_Reward/semantic_body_part_clearance`: `-0.3896`
- `Episode_Reward/reference_foot_pos`: `0.0446`
- `Episode_Reward/reference_contact`: `0.00380`

Foot-over signal:

- `semantic_foot_over_clearance` nonzero `953/5000`
- max `0.48075`
- previous `weight=0.12` run had nonzero `315/5000` and last100 `0.000390`

Side effects:

- reward is negative in the latest window (`-3.84`)
- `base_contact` is higher than the `weight=0.12` latest window (`0.0285` vs `0.0125`)
- `base_angular_velocity`, `feet_slide`, and `flat_orientation_l2` penalties are also larger than the `weight=0.12` run

Best 100-step stability window:

- around `model_18600.pt`
- episode length about `977.6`
- terrain about `6.09`
- bad_orientation about `0.089`
- base_contact about `0.009`
- foot-over reward about `0.0028`

Highest foot-over window:

- latest `model_19699.pt`
- foot-over reward about `0.054`
- episode length about `954`
- bad_orientation about `0.177`
- base_contact about `0.0285`

## Controlled Crossing Metrics

| Checkpoint | Opportunity | Root crossed | Foot-over | Small contact | Touchdown on small | Success | Resets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `18600` | `11/16` | `5/16` | `0/16` | `1/16` | `0/16` | `0/11` | `0/16` |
| `19699` | `11/16` | `3/16` | `0/16` | `3/16` | `0/16` | `0/11` | `base_contact=1/16` |

Per-env clearance:

- `model_18600`: root-crossed envs still have negative max clearance; best observed values include about `-0.014m`, `-0.019m`, `-0.051m`, `-0.062m`.
- `model_19699`: root-crossed envs still have negative max clearance; examples include about `-0.030m`, `-0.027m`, `-0.053m`.
- `foot_over_by_env` is false for all envs in both evals.

## Result

Diagnostic pass. Increasing `semantic_foot_over_clearance.weight` to `1.0` makes the TensorBoard scalar much stronger, and it appears to lift feet somewhat closer to the obstacle top, but it still does not produce measured foot-over or clean overpass in controlled crossing.

The latest checkpoint has worse behavioral trade-offs than the stable window: fewer root crossings, more small contacts, and one base-contact reset.

## Conclusion

Weight magnitude alone is not sufficient. It improves the scalar and nudges clearance upward, but the reward is still not aligned tightly enough with the eval's true overpass criterion. The policy can collect foot-over-shaped reward without producing positive foot clearance over the path-small obstacle.

Most likely next issue: the dense approach reward is rewarding near-obstacle swing/height in a way that is not spatially/timing-aligned with the strict `foot_over_count` condition. The next change should not simply increase the weight further; it should reshape the existing reward to require actual positive clearance over the path-small cell/region, and/or widen the approach window while keeping the final positive-clearance gate.

## Follow-Up

- Do not continue this exact `weight=1.0` run as solved.
- Prefer a shape fix over another pure weight increase:
  - explicitly reward positive foot clearance over the path-small cell/region;
  - make "root crossed but max clearance stayed negative" visible as an eval/training diagnostic;
  - consider an intermediate weight after shape correction, instead of `1.0` with the current shape.
- Keep low entropy and finite orientation reset because the run remains mostly stable.

## Git Refs

- Baseline Ref: working tree
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
