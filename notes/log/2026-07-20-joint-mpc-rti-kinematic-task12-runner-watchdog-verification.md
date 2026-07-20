# Joint MPC RTI Kinematic Task 12 Runner And Watchdog Verification

## Purpose

Verify the Task 12 shared metrics, formal cell schema, progress protocol, and resource-safe process supervisor without launching CUDA or Isaac Lab.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 12 of 16
- Baseline ref: `c435746`
- Candidate ref: uncommitted Task 12 checkpoint

## Commands

```text
PYTHONPATH=Go2Pvcnn ... pytest test_joint_metrics.py test_run_joint_acceptance.py test_process_watchdog.py test_contracts.py test_performance.py -q
PYTHONPATH=Go2Pvcnn ... pytest test_run_joint_acceptance.py test_process_watchdog.py -q
pytest --collect-only -q Go2Pvcnn/tests/joint_mpc_rti
git diff --check
```

## Results

- Focused contract/metrics/runner/watchdog/performance: `20 passed`.
- Formal selector and watchdog regression: `11 passed`.
- Full joint-MPC test collection: `130 tests collected`.
- No CUDA or IsaacLab process was launched by these commands.
- Watchdog covers child process groups, heartbeat, tree RSS, ptxas RSS, available memory, swap, selected GPU memory/utilization, and CPU-only compiler growth without GPU progress.
- Old behavior/stop/crossing pytest routes were removed because they directly launched long CUDA work and referenced retired `trajectory.control`/H16 APIs.

## Conclusion

Task 12's shared metric/report/watchdog foundation is green. Formal flat execution and the remaining real-state scenario adapters are intentionally deferred to Tasks 13 and 14.

## Follow-up

Implement the flat trace executor and acceptance aggregation before any full command matrix run. Keep all heavy commands behind `run_monitored_joint_mpc.py`.
