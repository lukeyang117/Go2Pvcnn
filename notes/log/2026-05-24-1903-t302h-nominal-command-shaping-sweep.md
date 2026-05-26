# T302h Nominal Command Shaping Sweep

## Purpose

Test the user's accepted direction A in probe/test code only: before nominal generation, shape command for high-small and large obstacles, while preserving low-small crossing behavior and trajectory continuity.

## Stage

- `extension/batch_mpc_planner` test-only semantic MPC diagnostics
- Probe files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`

## Related Todo

- [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Procedure

Added probe-only helpers and variants:

- `_semantic_command_shape_for_variant`: for high-small/large linear commands, reduces `vx`, adds lateral `vy` toward the freer semantic side, and records requested vs nominal command diagnostics.
- `_effective_planning_variant_for_semantic`: routes combined test-only variants by obstacle class/height.
- `nominal_cmd_shape_a_v1`: direct nominal-before command shaping.
- `nominal_cmd_shape_a_conservative_v4`: smaller lateral escape and less `vx` reduction.
- `nominal_cmd_shape_a_low_exact_v4`, `nominal_cmd_shape_a_low_accel_v5`, `nominal_cmd_shape_a_low_accel_anchor_v5`: low-small crossing/continuity controls.
- `nominal_cmd_shape_a_combined_v6/v7/v8`: attempted class-conditioned combined policies.

All runtime outputs were written under `tmp/t302h/`.

## Commands

```bash
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

Representative real IsaacLab sweeps:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --commands 'forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00' --variants baseline,nominal_cmd_shape_a_combined_v8
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases large --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,nominal_cmd_shape_a_combined_v8
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --semantic-small-height-m 0.46 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,yaw100:0.00 0.00 1.00' --variants baseline,nominal_cmd_shape_a_combined_v8
```

## Metrics

- `small_overpass_success`: low-small success; requires local root passage over the obstacle lane, no stance/touchdown semantic contact, no foot penetration, and bounded continuity.
- `large_avoid_success`: high-small/large success; requires clearance margin, no semantic contact, no root semantic occupancy, and bounded continuity.
- `semantic_task_violation`: main fail flag.
- `semantic_task_contact_violation`: stance/touchdown/foot penetration failure.
- `semantic_task_continuity_violation`: foot/root acceleration, jump, or boundary failure.
- `semantic_policy_margin_deficit`: missing avoidance distance; lower is better, `0` passes.
- `worst_max_to_median_step`, `worst_boundary_to_median_step`, `foot_accel_max_to_mean`, `root_accel_max_to_mean`: trajectory continuity/jitter.
- `requested_vx/vy/yaw`, `nominal_vx/vy/yaw`, `command_shaped`, `command_shape_reason`: command-shaping diagnostics.

## Key Results

Large forward:

- Baseline: `semantic_task=1`, `large_avoid=0`, margin deficit `0.029888`, jump `102.418`, footacc `32.687`, rootacc `29.909`, score `1196.588`.
- `nominal_cmd_shape_a_v1`: passed in one run with `semantic_task=0`, `large_avoid=1`, jump `6.091`, footacc `5.600`, rootacc `4.042`, score `246.027`.
- `nominal_cmd_shape_a_conservative_v4`: passed in targeted run with `semantic_task=0`, `large_avoid=1`, jump `2.778`, footacc `3.021`, rootacc `5.819`, score `236.173`.
- Improvement for best large run: score `-80.3%`, jump `-97.3%`, footacc `-90.8%`, rootacc `-80.5%`, margin deficit cleared to `0`.

High-small `0.46m`:

- Baseline was usually already task-clean, but had larger jitter in some runs: score `423.703`, jump `28.329`, rootacc `20.178`.
- `nominal_cmd_shape_a_combined_v7`: passed `3/3`, score `277.413`, jump `8.664`, footacc `22.165`, rootacc `14.820`.
- `nominal_cmd_shape_a_combined_v8`: improved score but failed `1/3` due boundary continuity (`boundary=12.196` > gate `12`).
- Best high-small run improvement: score `-34.5%`, jump `-69.4%`, rootacc `-26.6%`, contact stayed `0`.

Low-small mixed `vx=0.50, vy=0.25, yaw=1.00`:

- Baseline in repeat run: `semantic_task=1`, `small_overpass=0`, stance `0.017713`, score `863.444`.
- `loss_low_small_cont_v2`: repeat pass, `semantic_task=0`, `small_overpass=1`, stance `0`, footacc `14.540`, rootacc `6.761`, score `221.240`.
- `nominal_cmd_shape_a_low_accel_anchor_v5`: passed once (`score=261.271`) but failed in repeat (`footacc=66.173`), so it is not stable.
- `nominal_cmd_shape_a_combined_v8`: failed low-small mixed in final run due `foot_accel_max_to_mean=40.402`.
- Stable low-small improvement belongs to `loss_low_small_cont_v2`, not the combined command-shaping variants: score `-76.6%`, stance contact `-100%`, rootacc `-78.5%`.

## Result

Partial pass.

Command shaping before nominal is a strong test-only direction for large forward avoidance and useful for high-small jitter reduction. It is not sufficient as a single combined solution yet:

- large-forward can be solved by conservative command shaping in a targeted run;
- low-small crossing still needs the `loss_low_small_cont_v2` style loss path;
- combining low-small continuity loss and high/large command shaping is sensitive to run/process initialization and can regress foot acceleration or boundary continuity.

Production planner/runtime code remains unchanged.

## Follow-Up

- Do not productionize `nominal_cmd_shape_a_combined_v6/v7/v8` as-is.
- The next useful test is not more scalar stacking; it should isolate why the same low-small loss path changes under combined/rerun conditions, likely by logging optimizer seed/initial residuals and per-loss breakdown for the failing frames.
- A production design, if pursued, should separate:
  - low-small crossing loss/tracking;
  - high/large nominal command shaping;
  - first-frame handoff continuity.

## Verification

- `pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`: `42 passed`
- `py_compile`: pass
- `git diff --check`: pass
- `git diff -- Go2Pvcnn/extension/batch_mpc_planner/config.py | wc -l`: `0`

## Git Refs

- Baseline Ref: working tree before nominal-command-shaping probe pass
- Candidate Ref: working tree
- Key Files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`
  - `tmp/t302h/nominal_cmd_shape_a_v4_large_forward_sweep.jsonl`
  - `tmp/t302h/nominal_cmd_shape_a_v5_low_small_mixed_sweep.jsonl`
  - `tmp/t302h/nominal_cmd_shape_a_v7_high_small_sweep.jsonl`
  - `tmp/t302h/nominal_cmd_shape_a_v8_large_forward_sweep.jsonl`
  - `tmp/t302h/nominal_cmd_shape_a_v8_low_small_mixed_sweep.jsonl`
