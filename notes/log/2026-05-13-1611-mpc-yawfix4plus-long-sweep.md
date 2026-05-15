# T300d MPC yawfix4-plus long sweep

- Time: 2026-05-13 16:11
- Stage: `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline Ref: `57b5c64` plus working tree test-layer diagnostics
- Candidate Ref: test-layer monkeypatch variants only
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py](../../Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)

## Purpose

Test five yawfix4-derived directions over longer horizons and richer velocity combinations:

- `yawfix4a_yaw_gate_body_anchor`
- `yawfix4b_touchdown_jump_limiter`
- `yawfix4c_early_stance_hold`
- `yawfix4d_command_ramp`
- `yawfix4e_near_touchdown_mask`
- plus `yawfix4f_full_guarded_combo` as a combined guarded variant

The goal is yaw improvement without degrading forward/lateral behavior, touchdown grounding, or command-switch stability.

## Procedure

Environment:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2
```

24-cycle broad screen:

- variants: baseline, original `yawfix4`, `yawfix4a` through `yawfix4f`
- sequences: pure forward/back/lateral/yaw, mixed forward-yaw, mixed diagonal-yaw, boundary equal commands, and yaw-enter/yaw-exit command switches
- artifacts:
  - `/tmp/mpc_yawfix4plus_screen_24.jsonl`
  - `/tmp/mpc_yawfix4plus_remaining_24.jsonl`

48-cycle top sweep:

- variants: baseline, original `yawfix4`, `yawfix4a`, `yawfix4d`, `yawfix4f`
- sequences: pure directions, mixed diagonal/boundary commands, and the riskiest yaw switch sequences
- artifact: `/tmp/mpc_yawfix4plus_top_48.jsonl`

## 24-Cycle Broad Screen

| Group | Variant | foot_err | transition_err | step_max | touchdown_jump | result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| pure yaw | baseline | 0.0986 | 0.0312 | 0.4388 | 0.1172 | baseline |
| pure yaw | yawfix4 | 0.0791 | 0.0118 | 0.4058 | 0.1608 | best short/mid yaw, jump worse |
| pure yaw | yawfix4a | 0.0824 | 0.0148 | 0.4031 | 0.1572 | close to yawfix4 |
| pure yaw | yawfix4b | 0.0791 | 0.0149 | 0.3919 | 0.1318 | best jump compromise for pure yaw |
| pure yaw | yawfix4d | 0.0797 | 0.0155 | 0.4121 | 0.1602 | close to yawfix4 |
| pure yaw | yawfix4f | 0.0897 | 0.0222 | 0.4335 | 0.1315 | conservative, weaker yaw gain |
| sequence all | baseline | 0.0674 | 0.0482 | 7.1553 | 0.7022 | baseline |
| sequence all | yawfix4 | 0.0575 | 0.0414 | 3.8159 | 0.9726 | best sequence peak reduction, jump worse |
| sequence all | yawfix4a | 0.0615 | 0.0418 | 4.3169 | 0.9355 | decent |
| sequence all | yawfix4d | 0.0600 | 0.0439 | 4.3203 | 0.9761 | decent |
| sequence all | yawfix4f | 0.0623 | 0.0444 | 7.1349 | 0.7052 | no regression, little peak benefit |

24-cycle notes:

- Pure forward/back/lateral stayed effectively unchanged for all variants.
- Original `yawfix4` remained the best 24-cycle direction.
- `yawfix4b`, `yawfix4c`, and `yawfix4e` had bad command-switch spikes and are not good standalone directions.
- `yawfix4f` had zero sequence regression against baseline but also lost the main `yawfix4` sequence-peak improvement.

## 48-Cycle Top Sweep

| Group | Variant | foot_err | transition_err | step_max | touchdown_jump | abs_drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pure yaw | baseline | 0.1489 | 0.0467 | 0.6609 | 0.1829 | 0.1188 |
| pure yaw | yawfix4 | 0.1519 | 0.0197 | 0.7363 | 0.2521 | 0.1869 |
| pure yaw | yawfix4a | 0.1509 | 0.0205 | 0.7346 | 0.2427 | 0.1774 |
| pure yaw | yawfix4d | 0.1469 | 0.0190 | 0.7391 | 0.2448 | 0.1794 |
| pure yaw | yawfix4f | 0.1569 | 0.0325 | 0.7530 | 0.2205 | 0.1782 |
| sequence all | baseline | 0.0944 | 0.0726 | 14.1656 | 1.8380 | 0.0640 |
| sequence all | yawfix4 | 0.1085 | 0.0830 | 7.6353 | 2.3414 | 0.0730 |
| sequence all | yawfix4a | 0.1167 | 0.0899 | 8.6567 | 2.2878 | 0.0672 |
| sequence all | yawfix4d | 0.1106 | 0.0868 | 8.8291 | 2.3432 | 0.0730 |
| sequence all | yawfix4f | 0.1016 | 0.0823 | 14.1538 | 1.8512 | 0.0908 |

48-cycle notes:

- All tested variants kept touchdown grounding fixed: `touchdown_ground_gap_mean=0.0000`, `touchdown_airborne_ratio=0.0000`.
- Pure forward/back/lateral did not regress.
- Original `yawfix4`, `yawfix4a`, and `yawfix4d` reduced command-sequence peak foot-step strongly, but pure yaw long-horizon `step_max`, `touchdown_jump`, and drift became worse than baseline.
- `yawfix4f` is conservative and keeps touchdown jump closer to baseline, but it does not preserve the sequence peak-step improvement.

## Conclusion

The five yawfix4-derived test directions do not provide a production-ready long-horizon fix. The useful signal is narrower:

- body-relative yaw anchoring helps short/mid-horizon command-switch peak steps
- but with persistent replanning it over-couples the yaw anchor to drifting running body-foot memory
- simple yaw gates, entry ramps, early stance holds, and local caps do not stop the long-horizon pure-yaw degradation

The next production direction should not be another local mask around `yawfix4`. It should change the yaw anchor memory contract:

- use a fixed yaw-mode body footprint reference rather than continuously updated `running_foot_rel_body`
- update yaw anchors only on stable stance, not every contact/touchdown frame
- reset or rebase yaw anchor memory on command-regime transitions
- add a hard per-replan XY target displacement cap at the anchor-memory update layer, not inside the horizon nominal tensor

## Follow-Up

- Treat original `yawfix4` as a diagnostic direction, not a production patch.
- Implement the next test direction around yaw-mode anchor-memory lifecycle instead of per-frame nominal masks.
- Re-test with the 48-cycle top matrix before production integration.
