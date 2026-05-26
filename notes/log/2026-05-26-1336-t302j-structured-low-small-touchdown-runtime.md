# T302j Structured Low-Small Touchdown Runtime

## Purpose

Record the first default-runtime attempt to replace low-small exported touchdowns with current-foot/command/obstacle-relative structured targets, then align low-small swing segments toward the same targets for viewer inspection.

## Stage

`extension/batch_mpc_planner` default MPC runtime path.

## Related Todo

- [../todo/T302j-touchdown-endpoint-consistency.md](../todo/T302j-touchdown-endpoint-consistency.md)

## Command / Procedure

- Edited [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py) only for runtime behavior.
- Verification:
  - `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/planner.py`
  - `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_plan_segment_exports_command_farthest_touchdown_by_default`
  - `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:1 --commands 'forward_v050:0.50 0.00 0.00,mixed_yaw_v050:0.50 0.25 1.00' --variants baseline --cycles 1 --requested-n-frames 300 --warmup-steps 6`
  - `timeout 80s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:1 --num_envs 1 --terrain task --planner-backend mpc --terrain-row 6`

## Input Conditions

- Default MPC, no `--mpc-debug-variant`, no scripted command, no forced `--n-frames 25`.
- Low-small task terrain probe, commands:
  - `forward_v050:0.50 0.00 0.00`
  - `mixed_yaw_v050:0.50 0.25 1.00`

## Key Metrics

First structured-touchdown-only run:

- forward: `touchdown_behind_swing_foot_along_max_m=0.425903`, `fk_foot_over_low_small_success=1`, planned/FK swing above root `+0.025513/+0.073637`.
- mixed-yaw: `touchdown_behind_swing_foot_along_max_m=0.689770`, `fk_foot_over_low_small_success=1`, planned/FK swing above root `-0.005118/-0.008270`.

After aligning low-small swing to the structured target:

- forward: `touchdown_behind_swing_foot_along_max_m=0.052865`, `fk_foot_over_low_small_success=0`, planned/FK swing above root `-0.031535/-0.021394`, small contact/penetration `0`.
- mixed-yaw: `touchdown_behind_swing_foot_along_max_m=0.102925`, `fk_foot_over_low_small_success=1`, planned/FK swing above root `-0.177394/-0.118376`, small contact/penetration `0`.

After widening the cross window, tightening cross lateral target, and increasing the planned arc:

- forward: `touchdown_behind_swing_foot_along_max_m=0.063104`, `fk_foot_over_low_small_success=0`, planned/FK swing above root `-0.065887/+0.048357`, small contact/penetration `0`, `base_bottom_clearance_min=0.084870`.
- mixed-yaw: `touchdown_behind_swing_foot_along_max_m=0.102925`, `fk_foot_over_low_small_success=1`, planned/FK swing above root `-0.138340/-0.088079`, small contact/penetration `0`.

Local verification:

- `py_compile`: pass.
- Focused default export test: `1 passed`.

Viewer smoke:

- WebRTC streaming server started.
- The 80s smoke reached Isaac app/environment startup but timed out before `[Viewer] Attached mpc trajectory manager`; no traceback was found in the captured tail.

## Result

Partial improvement, not an accepted fix.

The runtime code now reduces the exported touchdown vs planned swing endpoint conflict in the tested low-small rows, especially compared with the structured-touchdown-only attempt. However, forward low-small no longer satisfies the FK foot-over metric, and FK can still put swing foot above root even when the planned swing is below root.

## Conclusion

The user's diagnosis is supported: touchdown marker generation and swing trajectory must be coupled. But coupling only in Cartesian planned space is insufficient because IK clamp/FK realization can still deform the actual foot path. The next fix should generate or filter low-small structured targets in an FK-reachable space, then export the same reachable endpoint without hiding the viewer mismatch.

## Follow-Up

- T302j.2 remains active: align exported touchdown with FK-reachable trajectory.
- T302j.3 remains active: prevent FK swing foot above root while preserving foot-over clearance.
- T302j.7 should inspect per-leg low-small phase/state in viewer and decide whether the forward failure is target timing, lateral lane, or FK clamp.

## Git Refs

- Baseline Ref: working tree before this runtime slice.
- Candidate Ref: working tree after editing [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py).
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../tmp/t302i-viewer-realized-foot-mismatch/2026-05-26-structured-touchdown-default-probe.jsonl](../../tmp/t302i-viewer-realized-foot-mismatch/2026-05-26-structured-touchdown-default-probe.jsonl)
  - [../../tmp/t302i-viewer-realized-foot-mismatch/2026-05-26-structured-swing-touchdown-default-probe.jsonl](../../tmp/t302i-viewer-realized-foot-mismatch/2026-05-26-structured-swing-touchdown-default-probe.jsonl)
  - [../../tmp/t302i-viewer-realized-foot-mismatch/2026-05-26-structured-swing-touchdown-v2-default-probe.jsonl](../../tmp/t302i-viewer-realized-foot-mismatch/2026-05-26-structured-swing-touchdown-v2-default-probe.jsonl)
  - [../../tmp/t302i-viewer-realized-foot-mismatch/2026-05-26-structured-touchdown-viewer-smoke.log](../../tmp/t302i-viewer-realized-foot-mismatch/2026-05-26-structured-touchdown-viewer-smoke.log)
