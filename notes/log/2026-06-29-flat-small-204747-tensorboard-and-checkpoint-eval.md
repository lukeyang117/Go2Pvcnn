# 2026-06-29 Flat-Small 20:47 TensorBoard And Checkpoint Eval

## Purpose

Inspect run `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-27_20-47-47` and evaluate whether its checkpoints can perform the desired low-small obstacle overpass behavior.

## Stage

Training metrics / controlled crossing checkpoint evaluation / flat-small semantic avoidance.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

Read TensorBoard scalars from:

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-27_20-47-47/events.out.tfevents.1782564621.enine.1966374.0
```

Then ran controlled crossing eval for:

- `model_18100.pt`: best TensorBoard stability window around step `18122`
- `model_19699.pt`: latest checkpoint

Representative command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 900s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode controlled_crossing \
  --headless \
  --device cuda:0 \
  --num-envs 16 \
  --num-rounds 1 \
  --max-steps 300 \
  --run-dir unused \
  --checkpoint /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-27_20-47-47/model_19699.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.6 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/flat_small_20260627_204747_model19699_controlled_crossing_absckpt
```

## Input Conditions

Saved config/tag confirms the intended combination:

- dense approach `semantic_foot_over_clearance` enabled
- `entropy_coef=0.002`
- `bad_orientation.limit_angle=1.1`
- `semantic_foot_over_clearance` reward weight `0.12`
- stability weights: `flat_orientation_l2=-3.5`, `base_angular_velocity=-0.12`, `feet_slide=-0.18`
- tag says `mpc num_env 2048`

## TensorBoard Metrics

The run is stable and does not show the high-entropy collapse seen in `2026-06-26_19-34-12`.

Last-100:

- `Train/mean_episode_length`: `961.16`
- `Train/mean_reward`: `2.83`
- `Curriculum/terrain_levels/mean_terrain_level`: `6.126`
- `Policy/mean_noise_std`: `0.360`
- `Episode_Termination/bad_orientation`: `0.1438`
- `Episode_Termination/base_contact`: `0.0125`
- `Episode_Reward/semantic_foot_over_clearance`: `0.000390`
- `Episode_Reward/semantic_body_part_clearance`: `-0.363`
- `Episode_Reward/reference_foot_pos`: `0.0472`
- `Episode_Reward/reference_contact`: `0.00410`

Foot-over signal:

- `semantic_foot_over_clearance` nonzero `315/5000`
- previous high-entropy dense run `2026-06-26_19-34-12` had nonzero `3279/5000`
- stable low-entropy pre-dense run `2026-06-24_21-47-13` had nonzero `19/5000`

Best 100-step stability window:

- around `model_18100.pt`
- episode length about `976.6`
- terrain about `6.12`
- bad_orientation about `0.089`
- base_contact about `0.003`
- foot-over reward about `7.3e-05`

## Controlled Crossing Metrics

| Checkpoint | Opportunity | Root crossed | Foot-over | Small contact | Touchdown on small | Success | Resets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `18100` | `12/16` | `6/16` | `0/16` | `3/16` | `1/16` | `0/12` | `0/16` |
| `19699` | `15/16` | `5/16` | `0/16` | `1/16` | `0/16` | `0/15` | `bad_orientation=1/16` |

Per-env clearance details show this is not just a metric-miss:

- `model_18100` root-crossed envs still have negative max clearance, roughly `-0.03m` to `-0.10m`.
- `model_19699` root-crossed envs with clearance records also remain negative, roughly `-0.03m` to `-0.10m`.
- `foot_over_by_env` is false for all envs in both evals.

## Environment Warning

IsaacSim printed many `errno=28/No space left on device` change-watch warnings during startup. The evals still completed and wrote summaries.

Local check:

- `/mnt/mydisk` has about `3.1T` free
- root `/` is at `100%` usage with about `16G` available reported
- inotify settings: `max_user_watches=65536`, `max_user_instances=128`

Treat this as an environment hygiene issue, not the cause of the eval result.

## Result

Diagnostic pass. This new run is much better than the previous high-entropy dense run in stability and reset behavior, but it still does not implement the desired crossing behavior.

The latest checkpoint reduces small contact compared with the unstable dense run (`1/16` contact, `1/16` bad_orientation), but `foot_over_count` remains `0/16` and clean overpass success remains `0`.

## Conclusion

The current bottleneck is no longer global policy collapse. The policy can stay upright and reach terrain level about `6`, but it is not learning the actual foot-lift-over-small-obstacle behavior.

Most likely problem: the dense foot-over reward is still too weak/rare relative to locomotion and clearance penalties, and the learned behavior prefers stable forward movement that avoids or brushes obstacles instead of raising feet above them. Since root crossing happens with negative foot clearance, the agent is not being sufficiently pushed toward positive swing-foot clearance at the obstacle.

## Follow-Up

- Do not claim this model solves low-small overpass.
- Next change should target the behavior signal, not stability:
  - make foot-over shaping denser/stronger during the approach window, or stage/curriculum-gate early training to force enough path-small opportunities;
  - add diagnostics separating "root crossed but feet never got positive clearance" from true foot-over;
  - preserve low entropy and finite orientation reset, because those fixed collapse.
- Also clean the root filesystem or raise inotify limits before long parallel IsaacSim work.

## Git Refs

- Baseline Ref: working tree
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
