# T302o Flat Forward MPC Lateral Bias Reproduction

## Purpose

Reproduce the user-observed lateral bias of MPC foot/root trajectories on flat terrain with a forward command.

## Stage

MPC semantic policy evaluation / planner command-frame diagnostics.

## Related Todo

- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Candidate Ref

- No runtime code change. One-off real IsaacLab probe only.

## Procedure

Real IsaacLab probe:

- `CUDA_VISIBLE_DEVICES=0`
- `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- env id: `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0`
- `num_envs=1`
- terrain row/col args: `--terrain-rows 0 --terrain-cols 0`
- fixed command: `"1.0 0.0 0.0"`

The probe compared three planner calls from the same state:

1. current default command `[1, 0, 0]`
2. command rotated by robot yaw into world XY
3. default command with semantic map removed

Output:

- `logs/mpc_policy_eval/left_bias_probe/2026-06-06_16-33-15-182355/summary.json`

## Key Metrics

- Robot yaw: `0.280182868 rad` = `16.053295 deg`
- Command manager value: `[1.0, 0.0, 0.0]`
- Body command rotated into world: `[0.9610049, 0.2765314]`
- Current nominal command: `[1.0, 0.0, 0.0]`
- Current nominal forward: `[1.0, 0.0]`
- Semantic map nonzero count: `0`
- Semantic policy mode: `0`
- Semantic command shaped: `false`
- Default MPC root delta in robot body frame:
  - forward: `0.3109417m`
  - left: `-0.0936782m`
- Rotated-command MPC root delta in robot body frame:
  - forward: `0.3251034m`
  - left: `-0.0044367m`
- No-semantic default-command root delta in robot body frame:
  - forward: `0.3109417m`
  - left: `-0.0936782m`

## Result

Reproduced. The lateral bias appears on flat terrain with no semantic obstacle trigger.

## Conclusion

The reproduced bias is a command-frame mismatch, not semantic avoidance.

`semantic_policy.classify_semantic_obstacle_mode()` rotates the body-frame linear command by root yaw before classifying obstacles. But the parametric nominal path uses `command_frame_axes()` from [../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py), and when linear speed is active that helper uses `cmd[:, :2]` directly as the world-frame forward vector.

In this run the robot yaw was about `16 deg`, so a body-forward command `[1, 0]` should correspond to world direction `[0.961, 0.277]`. The current planner instead used world `[1, 0]`, producing about `9.4cm` side drift in the robot body frame over the horizon. When the probe manually rotated the command to world XY, the side drift dropped to about `4.4mm`.

## Follow-Up

- Do not change runtime code from this log alone.
- If fixing, align the parametric nominal command-frame convention with the rest of the runtime: body-frame command from IsaacLab should be rotated by current root yaw before being used as a world-frame planning direction, or `command_frame_axes()` should explicitly own that rotation.
