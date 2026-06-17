# MPC Participation 1024 Count Probe

## Purpose

Check whether the user's 1024-env training command actually gives MPC reference reward to all 1024 environments, or whether terrain participation filtering reduces the effective count.

## Stage

Training env setup / MPC trajectory manager participation / reference reward mask.

## Related Todo

- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)
- [../todo/T302u-semantic-map-contact-collision-plan.md](../todo/T302u-semantic-map-contact-collision-plan.md)

## Procedure

A temporary read-only probe was created under `Go2Pvcnn/scripts/_tmp_mpc_participation_probe.py`, run once, then deleted. No production code was kept.

The probe used the same experiment and scale as the user's training command, but did not start PPO. It created the real IsaacLab training env, attached the MPC trajectory manager, computed `eligible_mpc_reference_envs(...)`, triggered `manager.refresh_from_env(...)`, and recorded `manager.reference_reward_mask().sum()`.

Command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/_tmp_mpc_participation_probe.py \
  --headless \
  --device cuda:0 \
  --num_envs 1024 \
  --mpc_num_envs 1024 \
  --experiment teacher_elevation_trajectory_mpc_semantic \
  --planner-backend mpc \
  --steps 3 \
  --output logs/mpc_policy_eval/mpc_participation_probe_1024.json
```

## Output

- [../../logs/mpc_policy_eval/mpc_participation_probe_1024.json](../../logs/mpc_policy_eval/mpc_participation_probe_1024.json)
- [../../logs/mpc_policy_eval/mpc_participation_probe_1024.log](../../logs/mpc_policy_eval/mpc_participation_probe_1024.log)

## Key Metrics

- Exit code `0`.
- `num_envs=1024`.
- `mpc_num_envs=1024`.
- `parallel_plan_batch_size=1024`.
- `eligible_count=1024`.
- `ineligible_count=0`.
- `reference_reward_mask_count=1024` for all three refreshes.
- Initial terrain level distribution:
  - row `0`: `24`
  - row `1`: `517`
  - row `2`: `483`
- Exclusion pairs were configured for terrain names `random_rough`, `boxes`, `pyramid_stairs`, `pyramid_stairs_inv` on rows `5-9`, but no envs started in those rows in this probe.

## Result

Diagnostic pass. For the tested command at reset/startup, the effective MPC reference reward count is exactly `1024/1024`.

## Conclusion

The user's specific startup command is not reduced below 1024 by participation filtering in the initial terrain distribution. The earlier caveat still applies later in curriculum: if environments move into rows `5-9` for blacklisted terrain names, the eligible count can become less than 1024. This probe only proves the initial/default reset distribution for this command.

## Follow-Up

- If training later reaches terrain rows `5-9`, run the same count probe after forcing or observing high rows to measure the reduced count.
- Consider adding a lightweight runtime counter for `eligible_count` and `reference_reward_mask_count` in training logs if this becomes a recurring confusion point.

## Git Refs

- Baseline Ref: current working tree on `costmap-teacher-ablation`
- Candidate Ref: no production code change; temporary probe deleted after run
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/participation.py](../../Go2Pvcnn/extension/batch_mpc_planner/participation.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
