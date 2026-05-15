# T300d MPC yawfix direction sweep

- Time: 2026-05-13 14:29
- Stage: `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline Ref: `57b5c64` plus working tree test-layer diagnostics
- Candidate Ref: test-layer monkeypatch variants only
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py](../../Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)

## Purpose

Compare five yaw foot-alternation fix directions in IsaacLab, then run a longer top-candidate sweep across pure velocity directions and command combinations. The acceptance focus is yaw improvement without forward/lateral regression.

## Procedure

Environment:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2
```

Short screen:

```bash
MPC_LONG_DRIFT_VARIANTS=baseline,yawfix1_horizon_anchor_blend,yawfix2_foot_spike_loss,yawfix3_touchdown_continuity_loss,yawfix4_body_relative_yaw_anchor,yawfix5_early_stance_guard \
MPC_LONG_DRIFT_SEQUENCES='forward_only:forward;backward_only:backward;lateral_left_only:lateral_left;lateral_right_only:lateral_right;yaw_left_only:yaw_left;yaw_right_only:yaw_right;mix_forward_yaw_left_only:mix_forward_yaw_left;mix_forward_yaw_right_only:mix_forward_yaw_right;mix_diag_yaw_left_only:mix_diag_yaw_left;mix_diag_yaw_right_only:mix_diag_yaw_right;forward_yaw_left_forward:forward,yaw_left,forward;lateral_left_yaw_right_lateral_left:lateral_left,yaw_right,lateral_left;yaw_left_forward_yaw_right:yaw_left,forward,yaw_right;diag_mix_to_yaw:mix_diag_yaw_left,yaw_left,mix_diag_yaw_right' \
MPC_PROBE_CYCLES=12 \
MPC_PROBE_TRANSITION_WINDOW=4 \
MPC_TEST_DEVICE=cuda:2 \
timeout 900s python Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py
```

Output copied to `/tmp/mpc_yawfix_screen_12.jsonl`.

Long top-candidate sweep:

```bash
MPC_LONG_DRIFT_VARIANTS=baseline,yawfix2_foot_spike_loss,yawfix4_body_relative_yaw_anchor \
MPC_LONG_DRIFT_SEQUENCES='forward_only:forward;backward_only:backward;lateral_left_only:lateral_left;lateral_right_only:lateral_right;yaw_left_only:yaw_left;yaw_right_only:yaw_right;mix_forward_yaw_left_only:mix_forward_yaw_left;mix_forward_yaw_right_only:mix_forward_yaw_right;mix_diag_yaw_left_only:mix_diag_yaw_left;mix_diag_yaw_right_only:mix_diag_yaw_right;forward_yaw_left_forward:forward,yaw_left,forward;lateral_left_yaw_right_lateral_left:lateral_left,yaw_right,lateral_left;yaw_left_forward_yaw_right:yaw_left,forward,yaw_right;diag_mix_to_yaw:mix_diag_yaw_left,yaw_left,mix_diag_yaw_right' \
MPC_PROBE_CYCLES=30 \
MPC_PROBE_TRANSITION_WINDOW=6 \
MPC_TEST_DEVICE=cuda:2 \
timeout 1200s python Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py
```

Output copied to `/tmp/mpc_yawfix_top_30.jsonl`.

## Key Metrics

Short screen, pure yaw:

| Variant | foot_err | transition_err | foot_step_mean | foot_step_max | touchdown_jump | ground_gap | airborne |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.0641 | 0.0172 | 0.0193 | 0.4402 | 0.0853 | 0.0000 | 0.0000 |
| yawfix1 horizon anchor blend | 0.0642 | 0.0240 | 0.0200 | 0.4463 | 0.0869 | 0.0000 | 0.0000 |
| yawfix2 foot spike loss | 0.0667 | 0.0175 | 0.0226 | 0.3348 | 0.1096 | 0.0000 | 0.0000 |
| yawfix3 touchdown continuity | 0.0639 | 0.0167 | 0.0192 | 0.4701 | 0.0680 | 0.0000 | 0.0000 |
| yawfix4 body-relative yaw anchor | 0.0349 | 0.0043 | 0.0147 | 0.2244 | 0.1281 | 0.0000 | 0.0000 |
| yawfix5 early stance guard | 0.0623 | 0.0170 | 0.0193 | 0.4831 | 0.0829 | 0.0000 | 0.0000 |

Long top-candidate sweep, group averages:

| Group | Variant | n | foot_err | transition_err | step_mean | step_max | td_jump | td_jump_max | abs_drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pure yaw | baseline | 2 | 0.1119 | 0.0313 | 0.0260 | 0.6080 | 0.1337 | 0.3280 | 0.0585 |
| pure yaw | yawfix2 | 2 | 0.1157 | 0.0344 | 0.0297 | 0.4518 | 0.1552 | 0.3100 | 0.0896 |
| pure yaw | yawfix4 | 2 | 0.0989 | 0.0124 | 0.0236 | 0.4976 | 0.1862 | 0.3922 | 0.0773 |
| pure linear/lateral | baseline | 4 | 0.0303 | 0.0032 | 0.0120 | 0.2227 | 1.4113 | 8.6327 | 0.0339 |
| pure linear/lateral | yawfix2 | 4 | 0.0303 | 0.0032 | 0.0120 | 0.2227 | 1.4113 | 8.6327 | 0.0339 |
| pure linear/lateral | yawfix4 | 4 | 0.0304 | 0.0031 | 0.0120 | 0.2228 | 1.4134 | 8.6217 | 0.0344 |
| mixed single | baseline | 4 | 0.0086 | 0.0028 | 0.0134 | 0.3464 | 0.8897 | 6.4236 | 0.0133 |
| mixed single | yawfix2 | 4 | 0.0130 | 0.0039 | 0.0136 | 0.1855 | 0.8928 | 6.4153 | 0.0191 |
| mixed single | yawfix4 | 4 | 0.0135 | 0.0061 | 0.0145 | 0.5393 | 0.8731 | 6.4002 | 0.0155 |
| command sequences | baseline | 12 | 0.0710 | 0.0543 | 0.0806 | 8.9070 | 1.3353 | 17.7593 | 0.0334 |
| command sequences | yawfix2 | 12 | 0.1130 | 0.0849 | 0.0995 | 8.8600 | 1.3789 | 17.7252 | 0.0498 |
| command sequences | yawfix4 | 12 | 0.0715 | 0.0544 | 0.0525 | 4.7506 | 1.6516 | 17.6866 | 0.0314 |

All variants kept `touchdown_ground_gap_mean=0.0000` and `touchdown_airborne_ratio=0.0000` in this production-grounded probe.

## Result

`yawfix4_body_relative_yaw_anchor` is the best direction overall:

- short pure yaw: `foot_err 0.0641 -> 0.0349`, `foot_step_max 0.4402 -> 0.2244`
- long pure yaw: `foot_err 0.1119 -> 0.0989`, `transition_err 0.0313 -> 0.0124`, `foot_step_max 0.6080 -> 0.4976`
- pure forward/backward/lateral stayed effectively unchanged
- command-sequence `foot_step_max` improved strongly: `8.9070 -> 4.7506`

Risks:

- `yawfix4` increases pure-yaw touchdown jump (`0.1337 -> 0.1862`) and touchdown max (`0.3280 -> 0.3922`)
- `yawfix4` worsens mixed diagonal single-command peak step in the 30-cycle test (`mix_diag_yaw_left/right` local step-max regressions)
- `lateral_left -> yaw_right -> lateral_left` foot error regresses (`0.0640 -> 0.0869`) despite lower peak step
- `yawfix2` is not a good production direction alone: it reduces some step peaks but raises foot error and transition error across yaw/sequence groups

## Conclusion

The root issue is no longer touchdown height or contact pairing. The best current production direction is to add a yaw-gated body-relative nominal anchor, but combine it with a touchdown-continuity or jump limiter before rollout. This should preserve the `yawfix4` yaw/sequence peak-step benefit while controlling the touchdown jump side effect.

## Follow-Up

- Implement production `yawfix4` cautiously as a yaw-gated, command-transition-aware body-relative anchor blend.
- Add a touchdown-jump limiter or continuity term around contact transitions before replacing production behavior.
- Re-test pure forward/backward/lateral, pure yaw, mixed diagonal yaw, and lateral-yaw-lateral after production implementation.
