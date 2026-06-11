# T302q Flat-Small 1024 Timing Localization

## Purpose

Localize why `teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance` remains slow after the user removed the `semantic_body_part_clearance` reward mount.

## Stage

RL collection / semantic scanner observation.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Command / Procedure

Baseline:

```bash
T302G_STEP_TIMING=1 T302G_STEP_TIMING_CUDA_SYNC=0 T302G_STEP_TIMING_STEPS=5 \
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless --device cuda:0 --num_envs 1024 --max_iterations 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic --planner-backend mpc
```

Flat-small resume:

```bash
T302G_STEP_TIMING=1 T302G_STEP_TIMING_CUDA_SYNC=0 T302G_STEP_TIMING_STEPS=5 \
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless --device cuda:0 --num_envs 1024 --max_iterations 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance --planner-backend mpc \
  --resume \
  --load_run /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07 \
  --load_checkpoint model_19999.pt
```

Logs:

```text
/tmp/t302q_timing/baseline_semantic_1024_timing.log
/tmp/t302q_timing/flat_small_1024_resume_timing.log
```

## Input Conditions

- Current working tree has the flat-small cfg but the user removed the `self.rewards.semantic_body_part_clearance = _semantic_body_part_clearance_reward_term()` mount.
- Both runs use 1024 envs and `T302G_STEP_TIMING_CUDA_SYNC=0`.

## Key Metrics

Baseline `teacher_elevation_trajectory_mpc_semantic`:

- Reward manager: 18 active terms.
- Step 3: `env.step.total=77.18ms`, `sim.step=33.52ms`, `observation.compute=27.33ms`, `reference_foot_pos=1.84ms`, `semantic_contact_collision=1.34ms`.
- Step 1 has a startup reference refresh: `reference_foot_pos=2516.04ms`.

Flat-small:

- Reward manager: 18 active terms, no `semantic_body_part_clearance`.
- Step 3: `env.step.total=3709.97ms`, `sim.step=13.62ms`, `observation.compute=3675.54ms`, `reference_foot_pos=1.99ms`, `semantic_contact_collision=0.85ms`.
- Step 5: `env.step.total=3727.90ms`, `observation.compute=3688.21ms`, `reference_foot_pos=1.72ms`.
- Step 1 has both startup reference refresh and slow observation: `reference_foot_pos=4091.97ms`, `observation.compute=3974.38ms`.

## Result

Diagnostic pass. The persistent flat-small slowdown is in `observation.compute`, not in steady-state MPC planning/reference reward and not in the removed body-part reward.

## Conclusion

The hot path is:

```text
ObservationManager.compute()
-> downsampled_elevation_semantic_scan()
-> sensor.data
-> SemanticGridRayCaster._update_outdated_buffers()
-> SemanticGridRayCaster._update_buffers_impl()
-> raycast_mesh(...)
```

Flat-small changes the semantic scene distribution: it uses flat-only terrain and small-obstacle plane counts for the flat rows, so the semantic scanner observation path is much heavier than the baseline mixed-terrain run where low-level non-plane rows can have zero semantic objects.

## Follow-Up

Design fixes should target observation/scanner, not MPC:

- cache policy/critic `downsampled_elevation_semantic_scan` within a step;
- align `semantic_height_scanner.update_period` to policy dt;
- consider reducing flat-small scanner/object load if exact per-step dense map is not required.

## Git Refs

- Baseline Ref: `working tree @ 2026-06-10 22:36 CST`
- Candidate Ref: same working tree
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/observations.py](../../Go2Pvcnn/extension/mdp/observations.py)
  - [../../Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py](../../Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
