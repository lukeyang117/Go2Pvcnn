# MPC Mixed Command And Sequence Long-Horizon Sweep

- Time: 2026-05-13 09:32 CST
- Purpose: move beyond the old fixed command matrix and test the long-horizon drift directions under mixed commands and command-switch sequences, then inspect any new bad indicators before proposing the next search direction.
- Stage: `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics.
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline Ref: `e90e3a4`
- Candidate Ref: working tree with new opt-in mixed/sequence sweep harness in `Go2Pvcnn/tests/test_mpc_runtime_headless.py`
- Key Files:
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)

## Added Test Surface

This pass extended the existing opt-in long-drift runtime test with:

- mixed-command cases near the linear/yaw regime boundary
- command-switch sequence cases
- segment-level metrics
- transition-window metrics for the first few replans after a command switch

New environment gate:

```bash
MPC_RUNTIME_LONG_DRIFT_SEQUENCE_SWEEP=1
```

Useful knobs:

```bash
MPC_LONG_DRIFT_SEQUENCE_CYCLES=40
MPC_LONG_DRIFT_TRANSITION_WINDOW=5
MPC_LONG_DRIFT_SEQUENCES=...
MPC_LONG_DRIFT_MIXED_COMMANDS=...
```

## Test 1: Baseline vs `dir10` On Mixed And Sequence Cases

Command:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2 \
MPC_RUNTIME_LONG_DRIFT_SEQUENCE_SWEEP=1 \
MPC_LONG_DRIFT_SEQUENCE_CYCLES=40 \
MPC_LONG_DRIFT_VARIANTS=baseline,dir10_yaw_anchor_linear_seed_proxy \
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_variant_sequence_sweep -s -q
```

Key sequence summaries:

| Sequence | baseline mean_abs | `dir10` mean_abs | Key observation |
| --- | ---: | ---: | --- |
| `forward_yaw_left_forward` | `0.0685` | `0.0313` | `dir10` clearly improves both linear drift and switch transient error |
| `forward_stop_backward` | `0.0382` | `0.0158` | `dir10` is very strong for pure linear/stop switching |
| `lateral_left_yaw_right_lateral_left` | `0.0342` | `0.0211` | drift improves, but yaw segment shows suspiciously large foot motion |
| `yaw_left_forward_yaw_right` | `0.1091` | `0.0370` | huge recovery on `yaw_left -> forward`, but yaw segments become aggressive |
| `diag_mix_to_yaw` | `0.0708` | `0.0511` | partial drift improvement, but mixed-to-yaw transition shows a new instability mode |

### Strong positive signal

The most important new confirmation is that the previous `dir10` result is not only a steady-state straight-line fix. It also suppresses the severe switch transient seen in:

- `yaw_left -> forward`

At 40 cycles per segment:

| Segment | baseline abs_drift | `dir10` abs_drift | baseline transition_foot_err | `dir10` transition_foot_err |
| --- | ---: | ---: | ---: | ---: |
| `yaw_left_forward_yaw_right / forward` | `0.1855` | `0.0279` | `0.0478` | `0.0010` |

This is a strong sign that persistent linear footprint memory is helping replans re-enter forward motion without rebuilding the foot-body relation from a bad yaw-drifted state.

### New bad indicators

The same run also exposed a real side effect that was not obvious from the old fixed command matrix:

1. yaw segments under `dir10` can become too aggressive:
   - `forward_yaw_left_forward / yaw_left`: `foot_step_mean=0.3479`
   - `lateral_left_yaw_right_lateral_left / yaw_right`: `foot_step_mean=0.2815`
   - `yaw_left_forward_yaw_right / yaw_right`: `foot_step_mean=0.3507`
2. mixed-boundary into yaw can worsen the immediate switch transient:
   - `diag_mix_to_yaw / yaw_left`
   - baseline `transition_foot_err_mean=0.0278`
   - `dir10` `transition_foot_err_mean=0.0785`
   - baseline `foot_step_mean=0.0052`
   - `dir10` `foot_step_mean=0.1497`

Interpretation:

- `dir10` is likely reducing drift partly by pulling the nominal/anchor state too hard during yaw-dominant replans.
- The old discrete command matrix hid this because it did not stress the linear/yaw boundary enough.

## Test 2: Re-check `dir14` Soft Gate On The New Hard Cases

Purpose:

- the old fixed command matrix made `dir13` and `dir14` look identical
- mixed/boundary commands are the correct place to test whether soft gates matter

Command:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
export MPC_TEST_DEVICE=cuda:2
export MPC_RUNTIME_LONG_DRIFT_SEQUENCE_SWEEP=1
export MPC_LONG_DRIFT_SEQUENCE_CYCLES=40
export MPC_LONG_DRIFT_VARIANTS=baseline,dir10_yaw_anchor_linear_seed_proxy,dir14_soft_gate_yaw_anchor_linear_seed_proxy
export MPC_LONG_DRIFT_SEQUENCES='forward_yaw_left_forward:forward,yaw_left,forward;yaw_left_forward_yaw_right:yaw_left,forward,yaw_right;diag_mix_to_yaw:mix_diag_yaw_left,yaw_left,mix_diag_yaw_right'
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_variant_sequence_sweep -s -q
```

### Result

`dir14` remained very close to `dir10` on the fully discrete switch cases:

- `forward_yaw_left_forward`
- `yaw_left_forward_yaw_right`

But it behaved meaningfully differently on the mixed boundary case:

| Segment | baseline abs | `dir10` abs | `dir14` abs | baseline transition_foot_err | `dir10` | `dir14` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `diag_mix_to_yaw / mix_diag_yaw_left` | `0.0991` | `0.0991` | `0.0127` | `0.0000` | `0.0000` | `0.0000` |
| `diag_mix_to_yaw / yaw_left` | `0.0654` | `0.0519` | `0.0533` | `0.0271` | `0.0770` | `0.0515` |
| `diag_mix_to_yaw / mix_diag_yaw_right` | `0.0421` | `0.0094` | `0.0158` | `0.0149` | `0.0168` | `0.0000` |

Important detail:

- `dir14` did not solve the yaw-side foot-step spike:
  - `diag_mix_to_yaw / yaw_left` still had `foot_step_mean=0.1441`
  - `yaw_left_forward_yaw_right / yaw_right` still had `foot_step_mean=0.3500`
- but it *did* strongly improve the mixed boundary segment itself:
  - `mix_diag_yaw_left` drift `0.0991 -> 0.0127`

## Conclusion

The new tests change the search picture.

Before this pass, `dir10` looked like the clear winner.

After this pass:

- `dir10` is still the strongest pure drift reducer on discrete switch cases, especially for:
  - forward/backward
  - `yaw -> forward`
- but it introduces a new side effect:
  - yaw segments can show abnormally large foot-step motion
  - mixed-to-yaw transitions can still overshoot
- `dir14` proves that soft regime weighting *does matter* once the command set includes mixed boundary commands
- however, gate softness alone does not eliminate the yaw-anchor aggressiveness

## Next Direction Suggestions

Highest-value next directions after this evidence:

1. keep the mixed/sequence harness and extend it to longer horizons (`120` cycles)
2. focus next search on yaw-anchor aggressiveness, not just drift:
   - anchor blend ratio instead of hard replacement
   - touchdown-only anchor update vs touchdown+stable-contact
   - cap yaw-anchor replacement by foot displacement magnitude
3. keep soft boundary weighting in the search because it clearly helps mixed commands
4. compare `dir10`-style linear memory with a *softer* yaw path rather than with another hard regime split
