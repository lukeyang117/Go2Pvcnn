# Flat-Small 09:49 TensorBoard And Checkpoint Eval

## Purpose

Read TensorBoard and evaluate the new flat-small run with `bad_orientation` termination disabled:

`logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-24_09-49-41`

## Stage

Training metrics / flat-small avoidance / checkpoint evaluation.

## Related Todo

[T302s env-level collision curriculum](../todo/T302s-env-level-collision-curriculum-plan.md)

## Input Conditions

- Run cfg confirms `terminations.bad_orientation: null`.
- `env_cfg.yaml` uses `num_envs: 2048`.
- Tag notes: `mpc num_env 2048`, speed curriculum removed, stability weights strengthened, `semantic_foot_over_clearance` reduced to `0.12`, and `bad_orientation` termination removed.

## TensorBoard Readout

Event range: training steps `14700..17145`, `2446` scalar points.

Final last-100 metrics:

- `Train/mean_episode_length`: `17.4339`
- `Train/mean_reward`: `-5.33772`
- `Curriculum/terrain_levels/mean_terrain_level`: `0`
- `Episode_Termination/base_contact`: `117.812`
- `Episode_Termination/time_out`: `0`
- `Episode_Reward/reference_foot_pos`: `0`
- `Episode_Reward/reference_contact`: `0`
- `Episode_Reward/semantic_foot_over_clearance`: `2.29e-06`
- `Policy/mean_noise_std`: `1.14847`

Early useful window:

- `15400..15600` still has episode length around `713..729`, terrain level rises from `3.37` to `5.37`, and base contact remains around `1.1`.
- `15700` begins the visible slide: episode length drops to `521`, base contact rises to `3.54`, and policy std reaches `0.954`.
- By `16200..17100`, episode length is only `16..17`, base contact is `117..126`, and terrain level is zero.

## Evaluation Commands

Controlled crossing and tracking were run for `model_15600.pt` and `model_17100.pt` using absolute checkpoint paths because `mpc_policy_eval.py` currently hard-codes the base semantic run directory for relative checkpoint lookup.

## Key Metrics

Controlled crossing:

| Checkpoint | Opportunity | Root crossed | Foot-over | Small contact | Success | Reset reasons |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| old `2026-06-17_12-01-10/model_14700.pt` | `16/16` | `7/16` | `2/16` | `3/16` | `0/16` | not recorded in old summary |
| new `2026-06-24_09-49-41/model_15600.pt` | `12/16` | `5/16` | `1/16` | `3/16` | `0/16` | `base_contact=3`, no bad orientation |
| new `2026-06-24_09-49-41/model_17100.pt` | `16/16` | `8/16` | `9/16` | `5/16` | `0/16` | `base_contact=16`, no bad orientation |

MPC tracking, 20-step:

| Checkpoint | Mean foot error | P95 foot error |
| --- | ---: | ---: |
| old `model_14700.pt` | `0.08757m` | `0.20221m` |
| new `model_15600.pt` | `0.07753m` | `0.19968m` |
| new `model_17100.pt` | `0.52028m` | `1.41519m` |

## Result

Disabling `bad_orientation` did not produce clean overpass learning. The latest checkpoint has more foot-over detections, but they coincide with full base-contact collapse: all `16/16` envs reset from `base_contact`, often at steps `15..18`.

`model_15600.pt` is the best checkpoint in this run by stability and tracking, but it still does not solve the task: controlled crossing success is `0/16` and foot-over is only `1/16`.

## Conclusion

Do not continue this run from `model_17100.pt`. The run collapses after the `15600..15700` window, likely because removing `bad_orientation` allows unstable recovery attempts to enter base-contact failures instead of resetting earlier.

The useful signal from this experiment is diagnostic: the policy can trigger more foot-over-like events when allowed to fall further, but the behavior is not dynamically stable and does not become a clean crossing skill.

## Follow-Up

- Treat `model_15600.pt` as the only candidate checkpoint from this run worth visual inspection.
- Do not use `bad_orientation=None` as a final training setting.
- A better next training attempt should restore a finite orientation reset or use staged relaxation, while adding a direct stability constraint around crossing rather than letting fallen states dominate.

## Git Refs

- Baseline Ref: `feea80f`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
