# Flat-Small Disable Bad-Orientation Reset Experiment

## Purpose

Temporarily disable the flat-small training `bad_orientation` termination so the user can test whether orientation reset is masking recoverable obstacle-crossing behavior.

## Stage

RL config / termination wiring / flat-small avoidance train cfg.

## Related Todo

[T302s env-level collision curriculum](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command / Procedure

- Changed `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg.__post_init__` from tuning `bad_orientation.limit_angle` to disabling the term:

```python
self.terminations.bad_orientation = None
```

- Static verification:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
git diff --check -- Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

## Input Conditions

- Baseline ref: `feea80f`
- Candidate ref: working tree
- Key file: [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)

## Key Metrics

- `py_compile`: exit `0`
- `git diff --check`: exit `0`
- Real IsaacLab smoke: not run, per user request to test manually.

## Result

Static checks pass. The flat-small train cfg now removes `bad_orientation` from the Termination Manager instead of setting its `limit_angle` parameter to `None`.

## Conclusion

This is an experimental reset ablation, not a final stability fix. It should show whether the policy can recover after large pitch/roll during small-obstacle crossing when the orientation reset is absent.

## Follow-Up

The next evidence should come from the user's manual training/eval test: compare episode length, base contact, action rate, flat orientation reward, small contact, and controlled-crossing success against the previous collapsed run.

## Git Refs

- Baseline Ref: `feea80f`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
