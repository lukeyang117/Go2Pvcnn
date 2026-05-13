# MPC Long Replan Second Direction Expansion

- Time: 2026-05-13 00:30 CST
- Purpose: continue long-horizon direction search after `dir10`, expanding the test-layer hypotheses instead of moving into production code.
- Stage: `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics.
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline Ref: `e90e3a4`
- Candidate Ref: working tree with extended sweep variants through `dir14`
- Key Files:
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)

## New Directions

Based on the previous `dir10` result, this round explored whether we could improve stability by changing how the memory is updated or how command-regime gates are applied.

| Variant | Hypothesis |
| --- | --- |
| `dir11_running_linear_body_seed_proxy` | replace reset-fixed linear body-frame footprint with slowly updated running footprint memory |
| `dir12_stance_only_yaw_anchor_linear_seed_proxy` | apply yaw anchor only on prior-contact stance legs, not all contact frames |
| `dir13_strict_gate_yaw_anchor_linear_seed_proxy` | make command-regime gates stricter than `dir10` |
| `dir14_soft_gate_yaw_anchor_linear_seed_proxy` | replace binary regime gates with continuous soft weights |

## Iteration A: Broad Six-Variant Long Sweep

Test:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2 MPC_RUNTIME_LONG_DRIFT_SWEEP=1 MPC_LONG_DRIFT_CYCLES=120 \
MPC_LONG_DRIFT_VARIANTS=baseline,dir10_yaw_anchor_linear_seed_proxy,dir11_running_linear_body_seed_proxy,dir12_stance_only_yaw_anchor_linear_seed_proxy,dir13_strict_gate_yaw_anchor_linear_seed_proxy,dir14_soft_gate_yaw_anchor_linear_seed_proxy \
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_variant_sweep -s -q
```

Observed before interruption by `dir12` test-layer failure:

- baseline summary: `mean_abs_drift=0.0602`, `max yaw_left=0.1059`
- `dir10` summary: `mean_abs_drift=0.0307`, `max yaw_left=0.0649`
- `dir11` summary: `mean_abs_drift=0.0404`, `max yaw_left=0.1059`

Interpretation:

- `dir11` was clearly worse than `dir10`; slowly updating the linear footprint memory removed the strong yaw benefit and did not improve lateral behavior.
- The useful signal from `dir10` appears to rely on keeping linear footprint memory fixed while independently using yaw anchors.

## Iteration B: `dir12` Focused Probe

Test:

```bash
MPC_TEST_DEVICE=cuda:2 MPC_RUNTIME_LONG_DRIFT_SWEEP=1 MPC_LONG_DRIFT_CYCLES=20 \
MPC_LONG_DRIFT_VARIANTS=dir12_stance_only_yaw_anchor_linear_seed_proxy \
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_variant_sweep -s -q
```

Result:

- test-layer execution still failed before producing metrics
- failure appears isolated to the experimental proxy path, not the underlying runtime fixture or baseline

Conclusion:

`dir12` remains unresolved as an experiment harness direction and is not yet evidence for or against a production approach.

## Iteration C: Gate-Shape Comparison

Test:

```bash
MPC_TEST_DEVICE=cuda:2 MPC_RUNTIME_LONG_DRIFT_SWEEP=1 MPC_LONG_DRIFT_CYCLES=120 \
MPC_LONG_DRIFT_VARIANTS=baseline,dir13_strict_gate_yaw_anchor_linear_seed_proxy,dir14_soft_gate_yaw_anchor_linear_seed_proxy \
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_variant_sweep -s -q
```

Result:

| Variant | mean_abs_drift | max_name | max_abs_drift |
| --- | ---: | --- | ---: |
| baseline | 0.0552 | yaw_left | 0.1309 |
| dir13 strict gate | 0.0313 | yaw_left | 0.0655 |
| dir14 soft gate | 0.0313 | yaw_left | 0.0655 |

Direction-level result:

| Command | baseline abs | dir13 abs | dir14 abs |
| --- | ---: | ---: | ---: |
| forward | 0.0821 | 0.0328 | 0.0328 |
| backward | 0.0765 | 0.0180 | 0.0180 |
| lateral_left | 0.0052 | 0.0234 | 0.0234 |
| lateral_right | 0.0323 | 0.0378 | 0.0378 |
| yaw_left | 0.1309 | 0.0655 | 0.0655 |
| yaw_right | 0.0042 | 0.0104 | 0.0104 |

Interpretation:

- `dir13` and `dir14` were numerically identical on this discrete command set, so soft-vs-strict gate shape did not materially matter under the current test matrix.
- Both improved forward/backward/yaw-left over baseline, but both worsened lateral behavior and slightly worsened `yaw_right` relative to this baseline run.
- They do not beat the best historical `dir10` run (`mean_abs=0.0140`).

## Conclusion

This expansion did not dethrone `dir10`.

- `dir11` is worse than `dir10`
- `dir12` is still a broken experiment path
- `dir13` / `dir14` behave the same on the current command matrix and are weaker than the best `dir10` evidence

The strongest current choice remains:

- keep the `dir10` regime split
- do not change the linear footprint memory into a running-updated memory
- do not spend more time on gate softness until the command matrix includes mixed commands near the yaw/linear boundary

## Next Direction Suggestions

If we continue searching before implementing production code, the next highest-value test directions are:

1. mixed commands near the regime boundary:
   - e.g. `(vx=0.2, wz=0.1)`, `(vx=0.1, vy=0.1, wz=0.1)`
2. command-ramp / command-switch sequences:
   - forward -> yaw_left -> forward
   - lateral -> yaw -> lateral
3. anchor update cadence variants:
   - touchdown-only vs touchdown+stable-contact, but implemented carefully enough that the test harness itself stays stable
