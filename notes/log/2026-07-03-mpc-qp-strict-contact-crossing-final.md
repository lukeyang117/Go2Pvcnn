# MPC QP Strict Contact Crossing Final

## Purpose

Close the remaining T302v strict low-small crossing gap for `mpc_qp`: low-small semantic obstacles must be crossed when reachable, without FK semantic collision, small penetration, touchdown-on-small, or stance contact on small semantic cells.

## Stage

MPC-QP backend / viewer playback / low-small strict semantic crossing.

## Related Todo

[T302v MPC QP safety-constrained backend plan](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Commands

Focused/local:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py Go2Pvcnn/tests/test_mpc_qp_backend.py
git diff --check -- Go2Pvcnn/extension/batch_mpc_qp_planner/config.py Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py Go2Pvcnn/tests/test_mpc_qp_backend.py
```

GPU1 viewer matrix:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0;backward:-0.45,0.0,0.0;left:0.0,0.35,0.0;right:0.0,-0.35,0.0;diag_fl:0.32,0.32,0.0;diag_fr:0.32,-0.32,0.0;mixed_turn_l:0.45,0.15,0.6;mixed_turn_r:0.45,-0.15,-0.6' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 1 --longitudinal-offset-m -0.35 --lateral-offset-m -0.12

CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0;backward:-0.45,0.0,0.0;left:0.0,0.35,0.0;right:0.0,-0.35,0.0;diag_fl:0.32,0.32,0.0;diag_fr:0.32,-0.32,0.0;mixed_turn_l:0.45,0.15,0.6;mixed_turn_r:0.45,-0.15,-0.6' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 1 --longitudinal-offset-m -0.35 --lateral-offset-m 0.12
```

GPU1 throughput:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 1024 --mpc-num-envs 1024 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_final_1024.json --planner-backend mpc_qp --qp-iterations 1
```

## Result

Pass.

- Focused QP suite: `20 passed`.
- Regression subset: `177 passed, 1 warning`.
- Pycompile: exit `0`.
- Diff check: exit `0`.
- GPU1 left-lane 8-command matrix: exit `0`, `crossing_opportunity_count=8`, `fk_foot_over_low_small_required_success_count=8`, `max_fk_semantic_collision_count=0`, `max_fk_stance_on_small_rate=0`, `max_fk_touchdown_on_small_rate=0`, `max_fk_foot_small_penetration_rate=0`.
- GPU1 right-lane 8-command matrix: exit `0`, `crossing_opportunity_count=5`, `fk_foot_over_low_small_required_success_count=5`, `max_fk_semantic_collision_count=0`, `max_fk_stance_on_small_rate=0`, `max_fk_touchdown_on_small_rate=0`, `max_fk_foot_small_penetration_rate=0`.
- GPU1 1024/1024 smoke: exit `0`, `completed_steps=30`, `max_sampled_plan_count_seen=1024`, `qp_replan_event_count=2`, CUDA allocated/reserved `7477361152/9261023232` bytes, `epoch_seconds=7.3453`, `max_qp_solve_ms_seen=32.93`, `max_qp_repair_ms_seen=13.40`, `max_qp_total_ms_seen=1686.11`.

## Code Notes

- `mpc_qp` remains opt-in; current `mpc` default path is unchanged.
- Contact-over low-small repair now uses a footprint keepout query, not only the foot center.
- `low_small_contact_reland_forward_m` is `0.16` to clear the real small obstacle footprint plus sampling margin.
- Viewer QP results apply an FK/yaw-aligned contact cleanup so visible/reference playback does not mark stance on small semantic cells.
- `mpc_qp_viewer_crossing_probe.py` now treats `fk_stance_on_small_rate` as a strict acceptance gate.

## Follow-Up

Keep watching playback readback and step smoothness in future tuning. The final strict semantic safety gate is passed for the tested two-lane matrix and 1024/1024 smoke.
