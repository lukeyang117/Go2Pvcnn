# Joint MPC RTI Kinematic Task 04 Nominal

## Purpose

Build the complete cold or rolling warm nominal trajectory in one tensor call without semantic XY search.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 4 of 16

## TDD Evidence

RED: test collection failed with `ModuleNotFoundError` for `model.nominal`.

Focused GREEN: `7 passed in 3.20s` for B=1/40/512/1024, warm formulas, semantic invariance, and source-loop checks.

Combined Tasks 1-4 command:

```bash
PYTHONPATH=Go2Pvcnn \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_new_contract.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_fixed_gait_schedule.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_analytic_ik.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py -q
```

Result: `23 passed in 3.58s`.

## Contract

- One call returns state `[B,31,18]`, foot/touchdown references `[B,31,4,3]`, and contact `[B,31,4]`.
- Cold root pose is integrated from body command without Python environment/time/leg loops.
- Lift, touchdown, and stance event indices use broadcast/gather; all reference heights use one packed terrain query.
- Swing height uses one `h_swing * 4*tau*(1-tau)` parameter.
- Cold IK uses one fixed batch-mask reduced step scale only when the full step is geometrically unreachable; there is no semantic XY search.
- Warm start shifts accepted `Z*`, sets terminal `q30=q6`, rebases root relative to prior x1 in SE(2), applies fixed joint measurement decay, and strictly injects measured z0.
- Changing raw semantic IDs without changing elevation does not alter nominal root, foot XY, or touchdown XY.

## Git Refs

- Baseline Ref: `52b19ba`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py`, `Go2Pvcnn/extension/joint_mpc_rti/runtime/warm_start.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py`

## Follow-Up

Build the unified convolutional elevation-semantic cost map and differentiable packed query.
