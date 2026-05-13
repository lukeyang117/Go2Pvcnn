# MPC Long Replan Iterative Direction Search

- Time: 2026-05-12 23:46 CST
- Purpose: continue the drift investigation with multiple hypothesis/test iterations instead of choosing from the first five-direction sweep.
- Stage: `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics.
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline Ref: `e90e3a4`
- Candidate Ref: working tree with extended opt-in drift sweep variants
- Key Files:
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)

## Iteration 1: Yaw-Only Anchor Nominal

Hypothesis:

The first sweep showed hard anchor nominal replacement helps yaw but harms linear/lateral commands. If anchor replacement is gated to yaw-dominant commands only, it should keep the yaw benefit without linear/lateral side effects.

Test:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2 MPC_RUNTIME_LONG_DRIFT_SWEEP=1 MPC_LONG_DRIFT_CYCLES=40 \
MPC_LONG_DRIFT_VARIANTS=baseline,dir6_yaw_anchor_nominal_proxy,dir7_yaw_anchor_blend_proxy,dir8_moderate_stance_loss \
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_variant_sweep -s -q
```

Result:

| Variant | cycles | mean_abs_drift | max_name | max_abs_drift | Conclusion |
| --- | ---: | ---: | --- | ---: | --- |
| baseline | 40 | 0.0530 | yaw_left | 0.0919 | reference |
| dir6 yaw hard anchor | 40 | 0.0384 | backward | 0.0714 | best yaw-only signal |
| dir7 yaw blend anchor | 40 | 0.0396 | backward | 0.0749 | useful but weaker |
| dir8 moderate loss | 40 | 0.0440 | forward | 0.0764 | helps yaw_right but hurts forward/lateral_left |

Key observations:

- `dir6` preserved forward/backward/lateral metrics and reduced yaw drift:
  - `yaw_left 0.0919 -> 0.0480`
  - `yaw_right 0.0611 -> 0.0171`
- Global loss tuning remained direction-dependent and was not the clean path.

## Iteration 2: Full 120-Cycle Confirmation For dir6

Hypothesis:

If `dir6` is a stable yaw fix, the yaw benefit should persist over the full 120-cycle six-command sweep without side effects on linear/lateral commands.

Test:

```bash
MPC_TEST_DEVICE=cuda:2 MPC_RUNTIME_LONG_DRIFT_SWEEP=1 MPC_LONG_DRIFT_CYCLES=120 \
MPC_LONG_DRIFT_VARIANTS=baseline,dir6_yaw_anchor_nominal_proxy \
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_variant_sweep -s -q
```

Result:

| Variant | cycles | mean_abs_drift | max_name | max_abs_drift |
| --- | ---: | ---: | --- | ---: |
| baseline | 120 | 0.0618 | yaw_left | 0.1442 |
| dir6 yaw hard anchor | 120 | 0.0438 | forward | 0.0842 |

Direction-level result:

| Command | baseline abs | dir6 abs | delta |
| --- | ---: | ---: | ---: |
| forward | 0.0842 | 0.0842 | +0.0000 |
| backward | 0.0719 | 0.0719 | +0.0000 |
| lateral_left | 0.0052 | 0.0052 | +0.0000 |
| lateral_right | 0.0323 | 0.0323 | +0.0000 |
| yaw_left | 0.1442 | 0.0660 | -0.0782 |
| yaw_right | 0.0329 | 0.0030 | -0.0299 |

Conclusion:

`dir6` is a clean yaw-specific fix candidate. After it, the maximum remaining drift shifts from yaw to forward/backward linear commands.

## Iteration 3: Linear Body-Frame Footprint Seed

Hypothesis:

Forward/backward drift is likely caused by rebuilding nominal from the current world foot positions each replan. For linear-dominant commands, seed nominal feet from a persistent initial body-frame footprint transformed by the current root pose. This should stabilize root-relative radius during long straight motion.

Test:

```bash
MPC_TEST_DEVICE=cuda:2 MPC_RUNTIME_LONG_DRIFT_SWEEP=1 MPC_LONG_DRIFT_CYCLES=40 \
MPC_LONG_DRIFT_COMMANDS=forward,backward,yaw_left,yaw_right \
MPC_LONG_DRIFT_VARIANTS=baseline,dir9_linear_body_seed_proxy,dir10_yaw_anchor_linear_seed_proxy \
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_variant_sweep -s -q
```

Result:

| Variant | cycles | mean_abs_drift | max_name | max_abs_drift |
| --- | ---: | ---: | --- | ---: |
| baseline | 40 | 0.0735 | yaw_left | 0.0985 |
| dir9 linear body seed | 40 | 0.0387 | yaw_left | 0.0884 |
| dir10 yaw anchor + linear seed | 40 | 0.0190 | yaw_left | 0.0531 |

Key direction-level result:

| Command | baseline abs | dir9 abs | dir10 abs |
| --- | ---: | ---: | ---: |
| forward | 0.0630 | 0.0025 | 0.0012 |
| backward | 0.0748 | 0.0079 | 0.0031 |
| yaw_left | 0.0985 | 0.0884 | 0.0531 |
| yaw_right | 0.0577 | 0.0562 | 0.0184 |

Conclusion:

`dir9` strongly fixes linear drift; `dir10` combines the linear fix with the yaw fix.

## Iteration 4: Full 120-Cycle Confirmation For dir10

Test:

```bash
MPC_TEST_DEVICE=cuda:2 MPC_RUNTIME_LONG_DRIFT_SWEEP=1 MPC_LONG_DRIFT_CYCLES=120 \
MPC_LONG_DRIFT_VARIANTS=baseline,dir10_yaw_anchor_linear_seed_proxy \
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_variant_sweep -s -q
```

Result:

| Variant | cycles | mean_abs_drift | max_name | max_abs_drift |
| --- | ---: | ---: | --- | ---: |
| baseline | 120 | 0.0646 | yaw_left | 0.1689 |
| dir10 yaw anchor + linear seed | 120 | 0.0140 | yaw_left | 0.0624 |

Direction-level result:

| Command | baseline abs | dir10 abs | delta |
| --- | ---: | ---: | ---: |
| forward | 0.0821 | 0.0008 | -0.0813 |
| backward | 0.0765 | 0.0048 | -0.0717 |
| lateral_left | 0.0052 | 0.0034 | -0.0018 |
| lateral_right | 0.0323 | 0.0029 | -0.0293 |
| yaw_left | 0.1689 | 0.0624 | -0.1065 |
| yaw_right | 0.0228 | 0.0098 | -0.0130 |

## Conclusion

Current best direction is `dir10`: split the persistent memory fix by command regime.

- Linear-dominant commands: seed nominal from a persistent body-frame footprint transformed by current root pose.
- Yaw-dominant commands: use contact-gated stance anchors in nominal stance frames.
- Avoid global stance/root loss tuning as the main fix; tests show it is direction-dependent.

The production implementation should convert this test-layer proxy into manager-owned state rather than monkeypatching:

- `initial_foot_rel_body[B,4,3]` or reset-time nominal footprint memory
- `stance_anchor_w[B,4,3]`
- command-regime gates from `[vx, vy, wz]`
- nominal builder input extension for optional persistent footprint/anchor state
- contact-aware anchor update and diagnostics
