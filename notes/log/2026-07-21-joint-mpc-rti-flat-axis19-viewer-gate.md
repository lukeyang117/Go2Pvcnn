# Joint MPC RTI Flat Axis-19 And Viewer Gate

## Purpose

Close Task 13 flat behavior after the command matrix was changed from Cartesian products to axis-isolated commands.

## Stage

`extension/joint_mpc_rti` pure-kinematic flat acceptance and real viewer playback.

## Related Todo

- [T302v](../todo/T302v-joint-mpc-rti-gpu.md)

## Inputs And Procedure

- Formal planner matrix: 19 axis-isolated commands, 144 steps per cell, CUDA device 1.
- Viewer: `standstill`, forward slow/fast, backward, lateral left/right, yaw left/right; 8 cycles each.
- Viewer preserves H30, grounds initial FR/RL stance feet from scanner data, executes one full SQP/RTI per cycle, and disables CUDA Graph replay for this behavior-only gate.

## Results

- Formal planner matrix: `19/19 passed` in `123.743s`.
- Viewer report: `/tmp/joint_mpc_viewer_flat_axis8_nograph.json`, `passed=true` in `33.443s`.
- Viewer adapter joint-order error: `0`.
- Maximum viewer joint step: below `0.35 rad` in every case.
- Maximum stance ground gap: below `0.012 m` in every case.
- Planner-to-actual foot error: below `1e-4 m` in every case.
- Standstill root XY and yaw drift passed.

## CUDA Graph Follow-Up

Warm graph capture is not yet closed. Diagnostics found capture-time tensor construction in gait constants, nominal posture constants, and then loss scalar creation. Gait and nominal constants now use `constant_like`; remaining loss scalar capture work belongs to Stage B and is not claimed complete here.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current worktree
- Key Files: `scenario_matrix.py`, `run_joint_acceptance.py`, `joint_mpc_rti_viewer_reproduction_probe.py`, `model/gait_schedule.py`, `model/nominal.py`

## Conclusion

Task 13 planner and real-viewer flat behavior gates are green under the approved 19-command axis-isolated definition. Small-obstacle acceptance may proceed. CUDA Graph performance remains a later explicit gate.
