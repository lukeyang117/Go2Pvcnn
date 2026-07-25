# Joint MPC Cross Flat Metrics Controlled Result

## Purpose

Verify that small-obstacle crossing uses the complete flat/common metric gate, including stance grounding and forbidden-semantic support, before further crossing tuning.

## Stage

Final perceptive-kinematic plan Task 17, controlled center-cuboid crossing.

## Related Todo

- [T302v](../todo/T302v-joint-mpc-rti-gpu.md)

## Procedure

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/joint_mpc_rti/test_lq_problem.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py
```

## Result

- `117 passed`, `2 failed`.
- Controlled cuboid strict crossing geometry succeeds and all part-collision, penetration, stance/touchdown forbidden-semantic checks pass.
- Complete common gate remains red: `joint_step_max_rad=0.3563213`, `publish_ratio=0.5655172`, `trajectory_valid_ratio=0.5655172`, `stop_ratio=0.4344828`.
- First sustained invalid window is caused by nominal joint-rate safety near touchdown; this is not a stance-ground semantic failure.
- Independent CUDA/CPU associative-scan parity also failed in the combined run with maximum direction difference `0.0168854`; keep this as a separate solver parity issue.

## Stance Metric Contract

- `stance_ground_gap <= 0.012m`.
- `stance_ground_penetration <= 0.001m`.
- continuing and touchdown-onset stance must query current world-map semantic `normal`.
- small, large, invalid, and out-of-map cells are forbidden for stance and touchdown.
- Small crossing runs these metrics together with every other flat/common metric; crossing success cannot mask a stance failure.

## Conclusion

Metric/schema integration is complete, but controlled small behavior is not accepted. The next owner-level change must make the swing-to-touchdown joint trajectory feasible before stance onset without changing the touchdown target, weakening collision checks, or relaxing common thresholds.

## Git Refs

- Baseline Ref: `5250bf1`
- Candidate Ref: uncommitted `work/joint-mpc-kinematic`
- Key Files: `joint_metrics.py`, `acceptance_thresholds.py`, `run_joint_acceptance.py`, `test_joint_metrics.py`, `test_small_acceptance.py`
