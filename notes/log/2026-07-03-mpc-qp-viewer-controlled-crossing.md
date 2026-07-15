# 2026-07-03 MPC QP Viewer Controlled Crossing

## Purpose

Verify the remaining T302v viewer/controlled acceptance gap: the opt-in `mpc_qp` backend must cross a low-small semantic object without semantic body collision, touchdown-on-small, or small-object penetration in real IsaacLab playback.

## Stage

MPC-QP backend / viewer playback / low-small semantic crossing acceptance.

## Related Todo

[T302v MPC QP safety-constrained backend plan](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Candidate Ref

Current worktree after adding `mpc_qp_viewer_crossing_probe.py`, semantic-frame shank clearance lift, and final post-repair shank lift.

## Key Files

- `Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py`
- `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- `Go2Pvcnn/tests/test_mpc_qp_backend.py`

## Commands

Focused/static:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q

python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py

git diff --check
```

Real GPU1 viewer lanes:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 1 --longitudinal-offset-m -0.35 --lateral-offset-m -0.12

CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 1 --longitudinal-offset-m -0.35 --lateral-offset-m 0.12
```

## Results

Static/focused:

- QP + participation + current MPC backend regression: `170 passed, 1 warning`.
- Pycompile: exit `0`.
- `git diff --check`: exit `0`.

GPU1 lane `lateral=-0.12`:

- Exit `0`.
- `viewer_crossing_acceptance_passed=true`.
- `crossing_leg_mask=[1,0,0,0]`.
- `fk_foot_over_low_small_success_count=1`.
- `max_fk_semantic_collision_count=0`.
- `max_fk_foot_small_penetration_rate=0.0`.
- `max_fk_touchdown_on_small_rate=0.0`.
- `fk_semantic_min_clearance_over_semantic_m≈0.0556`.
- `max_playback_readback_error_m≈0.00653` diagnostic only.

GPU1 lane `lateral=0.12`:

- Exit `0`.
- `viewer_crossing_acceptance_passed=true`.
- `crossing_leg_mask=[0,1,0,0]`.
- `fk_foot_over_low_small_success_count=1`.
- `max_fk_semantic_collision_count=0`.
- `max_fk_foot_small_penetration_rate=0.0`.
- `max_fk_touchdown_on_small_rate=0.0`.
- `fk_semantic_min_clearance_over_semantic_m≈0.00247`.
- `max_playback_readback_error_m≈0.000002` diagnostic only.

## Conclusion

The initial T302v viewer controlled-crossing acceptance is satisfied for the two tested forward foot lanes with `qp_iterations=1` on GPU card 1. The backend remains isolated behind `planner_backend="mpc_qp"` and does not replace the current `mpc` backend.

## Follow-Up

Broader sweeps can add multi-cycle, diagonal/lateral commands, rougher height-change terrain, and more obstacle offsets. Those are quality/stress coverage items beyond the initial isolated backend acceptance.
