# Joint MPC RTI Kinematic Flat Ranked CUDA Gate

## Purpose

Run the first monitored ranked flat behavior cells through the actual pure-kinematic rolling planner.

## Procedure

GPU `cuda:1`, three ranked commands, 24 control steps, resolution-preserving flat field, wrapped by `run_monitored_joint_mpc.py` with 120 s timeout and 5 s heartbeat.

## Results

- Process completed in about `13.2 s`; heartbeat reported tree RSS about `1.49 GiB`, task GPU memory about `2.09 GiB`, and active GPU utilization. No watchdog trigger occurred.
- All three cells produced finite reports and per-cell progress events.
- Flat gate failed. Zero command root drift was about `8.4e-5 m`; stance root-carry ratio was about `0.30`.
- Forward/backward stance XY slip reached about `0.134 m` per frame, stance stationary ratio was below `1`, root direction/velocity errors exceeded thresholds, swing relative progress was below the gate, and line search selected alpha zero for a long run in the forward cell.
- This is behavior failure evidence, not a runner/resource failure. Small-obstacle execution is correctly not authorized yet.

## Conclusion

Task 13 remains open. The current direct-Z RTI candidate needs root/stance/swing behavior diagnosis within the frozen seven-loss/KKT/line-search architecture before any formal 275-cell run.

## Git refs

- Baseline ref: `4ed0ce9`
- Candidate ref: uncommitted Task 13 flat runner/config tuning
- Key files: `Go2Pvcnn/extension/joint_mpc_rti/config.py`, `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`, `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`
