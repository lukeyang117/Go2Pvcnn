# T302s flat-small 11:20 TensorBoard readout

## Purpose

Inspect run `2026-06-13_11-20-40` after increasing `semantic_foot_over_clearance` scale, and compare whether the foot-over signal became learnable.

## Stage

- Training metrics / TensorBoard scalar interpretation
- Related todo: [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

Parsed TensorBoard scalars from:

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-13_11-20-40
```

Also checked saved `env_cfg.yaml` for reward settings.

## Input Conditions

- Checkpoints present: `model_20000.pt` through `model_22100.pt`
- Event scalar step range: `19999` to `22165`
- Saved foot-over reward config:
  - `semantic_foot_over_clearance.weight=1.0`
  - `bonus_scale=4.0`
  - `bonus_clip=1.0`
  - `corridor_width_m=0.42`
  - `lookahead_m=1.6`
  - `clearance_margin_m=0.05`

## Key Metrics

- `semantic_foot_over_clearance`
  - nonzero `9/2167`
  - max `0.101822`
  - mean `0.000130966`
  - last-100 mean `0.0`
  - last-20 mean `0.0`
- `semantic_contact_collision`
  - nonzero `1731/2167`
  - mean `-0.001601`
  - last-100 mean `-0.002103`
  - last-20 mean `-0.001798`
- `semantic_body_part_clearance`
  - nonzero `2092/2167`
  - mean `-0.003091`
  - last-100 mean `-0.003577`
- `mean_terrain_level`
  - last `5.846`
  - max `6.781`
  - last-100 mean `5.924`
- `Train/mean_episode_length`
  - last `997.48`
  - last-100 mean `988.99`
- `Train/mean_reward`
  - last `27.61`
  - last-100 mean `26.79`

Same step-window comparison against `2026-06-12_19-05-27`:

- Foot-over mean increased from `6.24e-08` to `1.31e-04`.
- Foot-over max increased from `4.21e-05` to `0.1018`.
- Foot-over nonzero count increased only from `6` to `9`.
- Contact mean worsened from `-0.001062` to `-0.001601`.
- Terrain level and episode length are both healthy in the new run.

## Result

Diagnostic pass. The scale increase is visible when the foot-over event fires, but event frequency is still too sparse. The last `100` logged points have zero foot-over reward.

## Conclusion

Increasing `semantic_foot_over_clearance` weight/scale made rare positive events larger, but did not turn the reward into a dense learning signal. The run is stable and curriculum is open, but the semantic behavior is still not convincingly improving from TensorBoard alone.

## Follow-up

- Do not treat larger foot-over max as success; the trigger count and recent zero window matter more.
- If continuing, inspect a later checkpoint with controlled crossing eval, but expect sparse foot-over behavior unless the opportunity/reward design changes.
- Next training design should make path-aligned obstacle opportunities and staged crossing rewards denser.

## Git Refs

- Baseline Ref: `23182ce`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
