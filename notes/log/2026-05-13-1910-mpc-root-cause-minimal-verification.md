# 2026-05-13 19:10 MPC Root-Cause Minimal Verification

## Purpose

Run the minimal validations proposed for the current MPC gait failures:

- nominal-only pair alternation bias
- yaw swing-stride activation
- planned foot -> IK/FK -> actual playback residual
- joint-limit saturation
- yaw memory anchor/seed distances

## Stage

T300d production `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics.

## Related Todo

- [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)

## Procedure

Added and ran [../../Go2Pvcnn/tests/mpc_root_cause_probe.py](../../Go2Pvcnn/tests/mpc_root_cause_probe.py).

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2 \
MPC_ROOT_CAUSE_CYCLES=10 \
MPC_ROOT_CAUSE_SEQUENCES='linear:forward,forward_fast,backward,backward_fast;pair_switch:forward,backward,forward;yaw:yaw_left,yaw_left_fast,yaw_right,yaw_right_fast;yaw_switch:yaw_left,yaw_right,yaw_left;mixed:forward_yaw_left,forward_yaw_right,lateral_left_yaw_right,lateral_right_yaw_left' \
MPC_ROOT_CAUSE_OUTPUT=/tmp/mpc_root_cause_probe_10.jsonl \
timeout 1200s python Go2Pvcnn/tests/mpc_root_cause_probe.py
```

## Input Conditions

- Device: `cuda:2`
- Runtime: IsaacLab headless via `env_isaacsim`
- Planner backend: `mpc`
- Cycles per segment: `10`
- Segment summaries: `18` nominal + `18` runtime
- Exceptions: `0`

## Key Metrics

Runtime group summary:

| Group | actual air | actual gap | IK/FK contact err | actual-plan foot err | joint saturation | joint near limit | yaw seed-plan XY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | `0.000` | `0.0050` | `0.0299` | `0.0299` | `0.015` | `0.017` | `0.0378` |
| yaw | `0.871` | `0.2089` | `0.2423` | `0.2423` | `0.170` | `0.174` | `0.1758` |
| mixed | `0.550` | `0.0597` | `0.1166` | `0.1166` | `0.254` | `0.262` | `0.4234` |

Important invariants:

- `actual_fk_foot_err_mean = 0.0000` across all rows.
- `actual_plan_root_err = 0.0000`.
- `actual_plan_joint_err_mean = 0.0000`.
- `plan_last_stance_airborne_ratio = 0.0000`.

Nominal-only highlights:

- `yaw` segments have `nominal_swing_stride_active=0.000`.
- `linear` and `mixed` segments have `nominal_swing_stride_active=1.000`.
- Nominal pair bias exists before optimizer:
  - `forward_fast`: front pair `FL=0.759`, `FR=0.200`
  - `backward`: front pair `FL=0.991`, `FR=0.000`
  - yaw often has one side fixed, e.g. `yaw_right`: front pair `FR=1.000`, switches `0`

## Result

Pass as root-cause evidence.

The largest direct explanatory signal is `IK/FK contact err == actual-plan foot err`, while root/joint playback errors and actual-FK foot error are zero. This means playback writes the planner root/joints correctly; the foot mismatch is already present when the planner foot target is converted through IK/FK.

## Conclusion

Most likely root cause is planner foot targets becoming kinematically infeasible or near joint limits, especially in yaw and mixed-yaw. Nominal/phase bias is also real and explains pair alternation asymmetry, but it is secondary for the airborne yaw failure.

## Follow-Up

- Fix yaw foot target generation and/or IK feasibility before blaming viewer playback.
- Add yaw-specific stride/alternation loss because pure yaw currently disables `swing_stride`.
- Add pair-wise lead balance metrics to forward/backward acceptance.

## Git Refs

- Baseline Ref: working tree on top of `e90e3a4`
- Candidate Ref: working tree with [../../Go2Pvcnn/tests/mpc_root_cause_probe.py](../../Go2Pvcnn/tests/mpc_root_cause_probe.py)
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_root_cause_probe.py](../../Go2Pvcnn/tests/mpc_root_cause_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
