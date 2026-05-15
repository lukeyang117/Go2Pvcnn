# T300d MPC IK/FK Residual Headless Comparison

## Purpose

Test whether temporarily enabling `ik_fk_residual` in the MPC loss reduces the user-requested foot realization errors without changing production defaults.

## Stage

Viewer-style IsaacLab headless runtime diagnostics for `Go2Pvcnn/extension/batch_mpc_planner`.

## Related Todo

- [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)

## Command

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2 \
MPC_ROOT_CAUSE_CYCLES=10 \
MPC_ROOT_CAUSE_VARIANTS='baseline:none;ikfk4:4.0;ikfk8:8.0' \
MPC_ROOT_CAUSE_SEQUENCES='linear:forward,forward_fast,backward,backward_fast;linear_switch:forward,backward,forward;yaw:yaw_left,yaw_left_fast,yaw_right,yaw_right_fast;yaw_switch:yaw_left,yaw_right,yaw_left;mixed:forward_yaw_left,forward_yaw_right,lateral_left_yaw_right,lateral_right_yaw_left' \
MPC_ROOT_CAUSE_OUTPUT=/tmp/mpc_ikfk_loss_fresh_compare_10.jsonl \
timeout 1200s python Go2Pvcnn/tests/mpc_root_cause_probe.py
```

## Input Conditions

- Device: `cuda:2`
- Envs: `2`
- Cycles per segment: `10`
- Variants:
  - `baseline`: `ik_fk_residual.enabled=False`
  - `ikfk4`: temporary runtime config override with `ik_fk_residual.weight=4.0`
  - `ikfk8`: temporary runtime config override with `ik_fk_residual.weight=8.0`
- Direction coverage: forward, backward, forward/backward switch, yaw left/right, yaw reversal, and mixed linear/lateral+yaw commands.

## Key Metrics

`/tmp/mpc_ikfk_loss_fresh_compare_10.jsonl` contained `540` cycle rows, `54` runtime segment summaries, and `0` exceptions.

| Variant | Group | Segments | Actual Air | Gap Abs | IK/FK Contact Err | Actual-Plan Contact Err | Actual-FK Err | Joint Sat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | linear | 7 | 0.0000 | 0.0047 | 0.0288 | 0.0288 | 0.0000 | 0.0161 |
| ikfk4 | linear | 7 | 0.0000 | 0.0008 | 0.0021 | 0.0021 | 0.0000 | 0.0008 |
| ikfk8 | linear | 7 | 0.0000 | 0.0007 | 0.0019 | 0.0019 | 0.0000 | 0.0006 |
| baseline | yaw | 7 | 0.8714 | 0.2145 | 0.2463 | 0.2463 | 0.0000 | 0.1709 |
| ikfk4 | yaw | 7 | 0.4571 | 0.0524 | 0.0915 | 0.0915 | 0.0000 | 0.1115 |
| ikfk8 | yaw | 7 | 0.3714 | 0.0351 | 0.0685 | 0.0685 | 0.0000 | 0.0911 |
| baseline | mixed | 4 | 0.5750 | 0.0621 | 0.1181 | 0.1181 | 0.0000 | 0.2538 |
| ikfk4 | mixed | 4 | 0.0125 | 0.0006 | 0.0075 | 0.0075 | 0.0000 | 0.1086 |
| ikfk8 | mixed | 4 | 0.0375 | 0.0021 | 0.0070 | 0.0070 | 0.0000 | 0.0953 |

Notable residuals:

- `ikfk8` still leaves `yaw_right` with actual air `0.9500` and contact error `0.1601m`.
- `ikfk8` still leaves the final `yaw_switch/yaw_left` segment with actual air `0.8000` and contact error `0.1515m`.
- `actual-FK err` stays `0.0000` across all groups, confirming Isaac actual foot follows FK from the written root/joint playback.

## Result

Pass as a diagnostic comparison. Temporarily enabling `ik_fk_residual` strongly improves the two meaningful target errors:

- `IK/FK contact err`
- `actual-plan foot err`

`actual-FK foot err` remains zero because the playback articulation and FK are aligned; the mismatch is primarily planned foot target reachability, not Isaac playback drift.

## Conclusion

`ikfk8` is the stronger diagnostic variant for yaw on this run, while `ikfk4` is already enough to make linear and mixed commands much cleaner. The remaining failure is yaw-specific, especially yaw-right and yaw reversal tail segments, so an IK/FK residual loss is useful but not a complete yaw gait fix by itself.

## Follow-Up

- Keep production defaults unchanged until visual/runtime acceptance confirms the residual trade-off.
- If this direction is promoted later, prefer a gated or tuned default around `weight=4-8`, with special attention to yaw-right and yaw-switch tails.
- Continue preserving forward/backward/lateral metrics while testing any yaw-specific follow-up.

## Git Refs

- Baseline Ref: `57b5c64`
- Candidate Ref: working tree with temporary runtime variant override only
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_root_cause_probe.py](../../Go2Pvcnn/tests/mpc_root_cause_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py)
