# Final Legacy Path Deletion

## Purpose

Close final-plan Task 15 by making the perceptive H30 direct-state RTI path the only Joint MPC production architecture and removing tests/probes that required superseded control-rollout, adaptive-contact, Gaussian-map, or seven-loss contracts.

## Stage And Refs

- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Baseline Ref: `41a67cc`
- Candidate Ref: current Task 15 working tree
- Branch: `work/joint-mpc-kinematic`

## Changes

- Moved the final `LossContext` owner to `solver/context.py` and migrated planner, LQ, SQP, and line-search imports.
- Deleted the seven-loss objective modules, Gaussian cost map, pending-reference module, control rollout, old linearization/ILQR/Gauss-Newton/TVLQR solvers, and unreachable legacy line search.
- Kept exact semantic/distance scanner compatibility outside the final max-pool perceptive safety path; exact semantic occupancy no longer exposes Gaussian sigma/gain/kernel tuning.
- Replaced the legacy seven-loss config/report boundary with the eight final LQ residual-family contract.
- Removed historical behavior probes/tests that required `trajectory.control`, adaptive contact/recovery, H30 as `15+15`, output projection, or horizon exploration. Final flat/small/large acceptance remains owned by Tasks 16-18.

## Verification

RED evidence:

- Initial Task 15 focused run failed because `terrain/__init__.py` imported deleted `cost_map` and old tests imported deleted objective fixtures.
- Full package after the first migration reported `59 failed, 256 passed`; every failure mapped to a superseded control/adaptive-contact/Gaussian/horizon contract.
- Expanded forbidden-path/config test reported `2 failed`, then the final loss-family test reported `3 failed` before production cleanup.

GREEN evidence:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_final_import_graph.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py -q
# 28 passed

PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti -q
# 247 passed in 60.55s
```

`git diff --check` passes and static scans find no production/test import of deleted modules.

## Conclusion

Task 15 is GREEN. This is architecture and regression closure only; no flat, small-obstacle, large-obstacle, actual viewer, B=1024 parity, or performance acceptance is claimed.

## Key Files

- `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- `Go2Pvcnn/extension/joint_mpc_rti/solver/context.py`
- `Go2Pvcnn/extension/joint_mpc_rti/solver/line_search.py`
- `Go2Pvcnn/extension/joint_mpc_rti/terrain/__init__.py`
- `Go2Pvcnn/tests/joint_mpc_rti/test_final_import_graph.py`
