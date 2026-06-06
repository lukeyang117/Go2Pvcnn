# T302o Foot Trajectory Timebase Probe

## Purpose

Check whether the MPC reference cache, reward consumption, eval metrics, and livestream marker path are using different time bases for the policy-vs-MPC foot trajectory comparison.

## Stage

MPC semantic policy evaluation / policy-vs-MPC foot tracking diagnostics.

## Related Todo

- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Baseline Ref

- Working tree after [2026-06-06-1512-t302o-foot-trajectory-lag-reproduction.md](2026-06-06-1512-t302o-foot-trajectory-lag-reproduction.md).

## Candidate Ref

- No runtime code change. The probe monkeypatched one `MpcTrajectoryManager.refresh_from_env` instance inside a one-off Python process only.

## Current Work Ref

- Branch: `costmap-teacher-ablation`

## Key Files

- [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
- [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
- [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)
- [../../../IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py](../../../IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)

## Command / Procedure

One-off real IsaacLab probe:

- `CUDA_VISIBLE_DEVICES=0`
- `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- env id: `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0`
- checkpoint: `2026-05-31_20-03-27/model_14000.pt`
- fixed command: `"1.0 0.0 0.0"`
- `num_envs=1`, `max_steps=30`

The probe logged per rollout step:

- before policy action
- before `wrapped_env.step(actions)`
- `refresh_from_env` entry
- `refresh_from_env` exit
- after `wrapped_env.step(actions)`

Output:

- `logs/mpc_policy_eval/timebase_probe/2026-06-06_16-16-20-212827/timebase_rows.jsonl`
- `logs/mpc_policy_eval/timebase_probe/2026-06-06_16-16-20-212827/summary.json`

## Key Metrics

- Rows: `150`
- Analyzed steps: `30`
- Warm steps: `24`
- `refresh_entry.episode_length` was already one step ahead of `before_env_step.episode_length` on every analyzed step.
- `after_env_step.episode_length` equaled `refresh_entry.episode_length`.
- Average `entry_minus_before_root_x`: `-0.007326571146647135m`
- Average `after_minus_entry_root_x`: `0.0m`
- Warm average `entry_current_along`: `+0.17652602689359856m`
- Warm average `exit_current_along`: `+0.18622366499352386m`
- Warm average `after_current_along`: `+0.18622366499352386m`
- Warm average `after_frame0_along`: `+0.0819799992316m`
- Warm average `after_l2_current`: `0.22343191877007484m`
- Warm average `after_l2_frame0`: `0.1068117218092084m`
- `phase_after_counts` covered frames `0..24`; frames `0..4` appeared twice because the 25-step horizon replanned/reset phase during the 30-step probe.
- `reference_reward_mask`: `[True]` on `2` replan/fresh-cache steps and `[False]` on `28` non-replan steps.

## Result

Pass as a diagnostic reproduction. No runtime code was changed.

## Conclusion

The current eval path is not asynchronous in the sense of a separate MPC thread planning while the actor keeps stepping. The observed time base is synchronous but post-step:

1. Policy action is computed from the current observation.
2. `wrapped_env.step(actions)` advances IsaacLab physics.
3. IsaacLab increments `episode_length_buf`.
4. IsaacLab reward computation calls `reference_foot_pos_reward()`.
5. `reference_foot_pos_reward()` calls `ensure_reference_cache()`.
6. `ensure_reference_cache()` calls `manager.refresh_from_env(env)`.
7. `refresh_from_env()` reads the already-advanced robot state, optionally replans, then advances `_phase_counter`.
8. `mpc_policy_eval.py` reads metrics and marker data after `wrapped_env.step()` returns.

Therefore eval metrics and marker visualization are reading the same post-step cache/phase that reward just refreshed. There is no evidence from this probe that the eval marker is stale relative to reward.

There is, however, a real convention offset inside `refresh_from_env`: on non-replan steps, entry uses the previous current frame id, then exit advances `_phase_counter` by one. That means after-step metrics and marker "current reference" use the next phase relative to the phase that existed at refresh entry. In this run, advancing phase made the along-command mismatch slightly larger (`+0.1765m` entry current vs `+0.1862m` exit/after current), but the dominant mismatch remains larger than a one-frame issue and frame-0 still matched better in L2.

Most likely interpretation:

- Not async MPC vs policy execution.
- Not a pure livestream marker delay.
- There is a post-step reward/visualization time base plus phase advancement convention.
- The larger visual mismatch is still mainly policy/reference gait phase or shape mismatch: actual feet are ahead of the current MPC reference along command direction, while L2 often matches early cache frames better.

## Follow-Up

- Do not fix runtime code from this diagnostic alone.
- If changing semantics later, decide explicitly whether `current_reference()` after refresh should mean "frame consumed for this reward step" or "next frame for the next step".
- A targeted follow-up can compare reward error using `phase_entry`, `phase_exit`, and `frame0` in one run without changing production behavior.
