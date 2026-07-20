# Joint MPC RTI Kinematic Task 11 Pipeline

## Purpose

Route production through one vectorized nominal, one direct-Z SQP linearization, the approved active trajectory scan, and one five-candidate line search, then publish x1.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 11 of 16

## TDD And Debugging Evidence

- RED: old planner import failed through the removed control objective API.
- First integration: `19 passed, 7 failed`; all seven failures traced to B×5 candidates using an unrepeated B-sized non-terrain loss context.
- After context mapping, a targeted rolling test exposed `valid=False`; tracing found unused stance-leg lift queries near `-480m` from a clamped `tau0=1` inverse. Non-swing inferred lift now uses measured foot coordinates while the approved swing equation and touchdown remain unchanged.
- Focused planner/backend/rolling integration: `26 passed`.
- Tasks 1-11 selected CPU regression: `108 passed, 1 CUDA compile deselected in 90.57s`.
- Package/test `compileall`, forbidden production keyword audit, and `git diff --check`: exit `0`.

## Contract

- Production call order is nominal -> direct-Z linearization -> active H30/32 scan -> five-candidate line search.
- Output state is `[B,31,18]`; state node zero exactly equals measured state.
- Only x1 is published through `JointMpcPendingReference(target_step=1)`.
- Solver state contains only accepted trajectory, gait phase, and valid mask.
- CUDA graph fixed-address copies use only those three fields.
- Diagnostics consume derived state velocity, not an optimized control.
- B=1/40 pipeline shape and x0/x1 contracts are covered; rolling manager reset and phase advancement are covered.
- Deleted old production modules: control dynamics/rollout and semantic/clearance/rollout objective.
- Deleted stale tests that required the old control solver and dynamics.
- Production source contains no recovery/startup/restoration/adaptive-contact/constraint-ranking/control-rollout path.

## Git Refs

- Baseline Ref: `0ef672a`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`, `Go2Pvcnn/extension/joint_mpc_rti/solver/sqp_rti.py`, `Go2Pvcnn/extension/joint_mpc_rti/runtime/manager.py`, `Go2Pvcnn/extension/joint_mpc_rti/runtime/cuda_graph.py`

## Follow-Up

Rebuild the shared applicability-aware metrics and monitored child-process runner before running flat and small behavior matrices.
