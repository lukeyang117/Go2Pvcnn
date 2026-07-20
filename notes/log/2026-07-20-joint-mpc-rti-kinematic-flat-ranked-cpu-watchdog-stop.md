# Joint MPC RTI Kinematic Flat Ranked CPU Watchdog Stop

## Purpose

Check the first real flat trace runner after Task 13 wiring and ensure an unproductive CPU fallback does not remain running silently.

## Procedure

The ranked three-cell flat trace was launched with the pure-kinematic planner for 24 control steps on CPU. It produced no progress line after startup and was terminated by the supervising agent after roughly 45 seconds; the child shell and Python process were both confirmed absent afterward.

## Result

The trace code is functionally valid for short tests, but CPU execution is not a usable formal-matrix backend at this horizon. This is an execution/resource issue, not acceptance evidence. Formal ranked/full behavior must use a monitored CUDA run when a GPU slot is available, with heartbeat output and the existing child-group watchdog.

## Follow-up

Keep the runner batch-oriented and add an explicit device selector. Do not launch a CPU formal matrix or leave a no-output process running.

## Git refs

- Baseline ref: `4ed0ce9`
- Candidate ref: uncommitted Task 13 flat runner
- Key files: `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`, `Go2Pvcnn/extension/joint_mpc_rti/config.py`
