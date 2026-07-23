# Final H30/H32 Associative Scan

## Purpose

Complete final-plan Task 10 by replacing the temporary sequential block-pentadiagonal solve with the required H30 padded-to-H32 five-level associative recovery while preserving the new full-horizon constraint contract.

## Stage And Refs

- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Baseline Ref: `650e9b2`
- Candidate Ref: current uncommitted Task 10 checkpoint
- Branch: `work/joint-mpc-kinematic`

## Implementation

- Fold fixed equality rows and two active-mask inequality refinements into block-pentadiagonal augmented systems.
- Map each system to 36D separators `y_k=[Delta z_(k-1),Delta z_k]`.
- Append two identity/no-cost factors to H30, execute five explicit pair-combine levels, and recover all 31 state-node directions.
- Use graph-safe cached identities and fixed-size SPD/general solves.
- Delete the sequential block solver; dense remains test-only.

## Verification

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_diagnostics.py -q
```

Result: `14 passed in 11.25s`.

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py -q
```

Result: `11 passed in 21.64s`, including real CUDA Graph capture/replay.

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=Go2Pvcnn \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_perf_probe.py \
  --num-envs 40 --horizon 30 --steps 10 --warmup 2
```

Result: nonfinite `0`, mean `122.66998ms`, p95 `137.53948ms`, peak `1410.72MiB`.

## Conclusion

Task 10's structural and numerical contracts pass. The performance result is explicitly red; true completion of the user requirement still depends on Task 19 factor/kernel optimization and the formal B=1024 workload.

## Key Files

- `Go2Pvcnn/extension/joint_mpc_rti/solver/associative_scan.py`
- `Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_scan.py`
- `Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py`
