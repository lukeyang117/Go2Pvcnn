# T300d MPC gait-coupling loss minimal fix

- Time: 2026-05-11 16:35
- Stage: planner runtime/loss coupling (`extension/batch_mpc_planner`)
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline ref: `working tree on top of 130c635`
- Candidate ref: `working tree (uncommitted)`

## Purpose

Address the runtime symptom "root moves but feet only jitter" by adding minimal gait-coupling losses before larger planner architecture changes.

## Code scope

- `Go2Pvcnn/extension/batch_mpc_planner/config.py`
  - Added tunable loss configs:
    - `MpcStanceSlipLossCfg.slip_tolerance_m_per_step`
    - `MpcSwingStrideLossCfg.min_swing_span_m`
    - `MpcSwingStrideLossCfg.command_speed_deadzone_mps`
  - Added task-cfg override fields:
    - `mpc_loss_stance_slip_tolerance_m_per_step`
    - `mpc_loss_swing_stride_min_swing_span_m`
    - `mpc_loss_swing_stride_command_speed_deadzone_mps`
  - Lowered default `touchdown_support.weight` from `0.8` -> `0.25`.
- `Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py` (new)
  - Added differentiable:
    - `stance_slip_loss(...)`
    - `swing_stride_loss(...)`
- `Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py`
  - Wired `stance_slip` and `swing_stride` into total loss accumulation.
- `Go2Pvcnn/tests/test_batch_mpc_backend.py`
  - Added regression checks:
    - diagnostics `loss_breakdown` contains `stance_slip` and `swing_stride`
    - new config overrides map correctly into planner cfg

## Verification

1. `python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q`
   - Result: `12 passed in 4.26s`
2. `MPC_TEST_DEVICE=cuda:2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py -k "forward_plan_has_time_varying_joint_angles" -q`
   - Result: `1 passed`
3. `MPC_TEST_DEVICE=cuda:2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py -k "plan_case_headless_smoke or diagnostics_layer_emits_hard_mask" -q`
   - Result: `2 passed`

## Conclusion

Minimal coupling losses are now active, configurable from RL task cfg, and validated on both focused unit tests and IsaacLab headless runtime path (`cuda:2`).

## Follow-up

- Re-run viewer qualitative check for walking quality and compare foot span/contact cadence metrics before deciding if structural loss redesign is still required.
