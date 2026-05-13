# MPC Touchdown Grounding Probe

- Time: 2026-05-13 10:25 CST
- Purpose: add and validate touchdown-grounding metrics under real IsaacLab MPC runtime, then check whether baseline / drift-fix variants keep planned touchdowns on the ground instead of in the air.
- Stage: `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics.
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline Ref: `e90e3a4`
- Candidate Ref: working tree with touchdown-grounding metric bridge and JSONL probe updates
- Key Files:
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
  - [../../Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py](../../Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py)

## What Changed In The Test Layer

This pass added touchdown-grounding metrics and debugged them against the current MPC terrain contract:

- `touchdown_ground_gap_mean`
- `touchdown_airborne_ratio`
- `touchdown_airborne_max_gap`

The first attempt failed because current MPC runtime terrain is `batch_mpc_planner.types.MpcPlannerTerrain`, which does not expose `height_at()`. The metric bridge was updated to sample `height_map + world_x/y_range` directly with bilinear grid sampling inside the test layer.

## Minimal Probe Result

Command:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
timeout 240s python Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py
```

Key result from the single-segment minimal probe:

- `forward`
  - `touchdown_ground_gap_mean=0.0667`
  - `touchdown_airborne_ratio=1.0`
  - `touchdown_airborne_max_gap=0.0701`

Interpretation:

- planned touchdown points are not landing on the terrain surface
- even the simplest forward probe shows all planned touchdown points above ground by about `6-7 cm`

## Small Sequence Sweep Result

Command:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_LONG_DRIFT_VARIANTS=baseline,dir10_yaw_anchor_linear_seed_proxy,dir14_soft_gate_yaw_anchor_linear_seed_proxy \
MPC_PROBE_MAX_SEQUENCES=1 \
MPC_PROBE_CYCLES=5 \
MPC_PROBE_TRANSITION_WINDOW=3 \
timeout 300s python Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py
```

Sequence:

- `forward_yaw_left_forward`

### Baseline

- first `forward`
  - `abs_drift=0.0049`
  - `touchdown_ground_gap_mean=0.0971`
  - `touchdown_airborne_ratio=1.0`
- `yaw_left`
  - `abs_drift=0.0027`
  - `touchdown_ground_gap_mean=0.1924`
  - `touchdown_airborne_ratio=1.0`
- second `forward`
  - `abs_drift=0.0001`
  - `touchdown_ground_gap_mean=0.2090`
  - `touchdown_airborne_ratio=1.0`

Important trend:

- touchdown gap grows with repeated replans
- yaw segment is much worse than forward
- after switching back to forward, touchdown gap remains high instead of recovering

### `dir14_soft_gate_yaw_anchor_linear_seed_proxy`

- first `forward`
  - `abs_drift=0.0069`
  - `touchdown_ground_gap_mean=0.0653`
  - `touchdown_airborne_ratio=1.0`
- `yaw_left`
  - `abs_drift=0.0595`
  - `touchdown_ground_gap_mean=0.1017`
  - `touchdown_airborne_ratio=1.0`
- second `forward`
  - `abs_drift=0.0035`
  - `touchdown_ground_gap_mean=0.0954`
  - `touchdown_airborne_ratio=1.0`

Important trend:

- soft-gated `dir14` improves touchdown gap substantially relative to baseline on this short sequence
- especially on yaw and forward-after-yaw segments
- but it still does not put planned touchdowns onto the ground: airborne ratio remains `1.0`

### `dir10_yaw_anchor_linear_seed_proxy`

The JSONL run advanced cleanly into the `dir10` segment loop after the metric bridge fix, confirming the metric path is now working under the same runtime path. This pass was used mainly to validate metric stability and expose the new touchdown signal before launching a longer multi-sequence sweep.

## Conclusion

This pass changes the search picture in an important way:

1. touchdown-grounding is a real and strong failure signal in current MPC runtime, not just a derived metric artifact
2. the failure is present even in short forward probes
3. repeated replans increase touchdown-ground gap over time
4. yaw segments are worse than forward segments
5. `dir14` appears better than baseline on touchdown gap in the short `forward_yaw_left_forward` sequence, but does not actually ground touchdowns

## Next Follow-Up

1. run the same touchdown metrics on longer horizons and all approved directions/sequences
2. compare `baseline / dir10 / dir14` jointly on:
   - `abs_drift`
   - `transition_foot_err_mean`
   - `foot_step_mean`
   - `touchdown_ground_gap_mean`
   - `touchdown_airborne_ratio`
   - `touchdown_airborne_max_gap`
3. prioritize directions that reduce touchdown gap, not only drift
