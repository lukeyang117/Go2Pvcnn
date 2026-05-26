# T302h Loss-Only Direction Sweep

## Purpose

Test the user's revised constraint: solve semantic obstacle traversal by loss terms only, without selector, nominal-before command rewrite, handoff blend, or postprocess trajectory repair.

## Stage

- `extension/batch_mpc_planner` test-only semantic MPC diagnostics
- Probe files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`

## Related Todo

- [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Procedure

Added probe-only loss helpers and variants:

- `loss_low_small_cross_v1`: low-small local crossing path, foot semantic exclusion, touchdown semantic exclusion, swing clearance.
- `loss_high_large_avoid_v1`: high-small/large body field, corridor avoidance, lateral escape.
- `loss_continuity_anchor_v1`: foot boundary/accel/jerk, root accel, early foot anchor.
- `loss_semantic_all_v1`: combined v1.
- v2-v6 focused variants:
  - `loss_low_small_cont_v2`: low-small crossing plus stronger continuity/worst-step/anchor.
  - `loss_high_large_escape_v2`: stronger high/large avoidance, clears margin but worsens continuity.
  - `loss_high_large_smooth_v3`: soft high/large avoidance plus strong continuity, smooth but too near.
  - `loss_high_large_margin_v4`: explicit distance margin, smooth but still too near.
  - `loss_high_large_balanced_v5`: balanced v2/v3, smooth but still too near.
  - `loss_high_large_scurve_v6`: differentiable side-bypass shape, but real run enters root semantic contact.

All outputs were written under `tmp/t302h/`.

## Commands

```bash
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --commands 'forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00' --variants baseline,loss_low_small_cont_v2,loss_semantic_all_v2 > tmp/t302h/loss_only_v2_low_small_mixed_sweep.jsonl 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases large --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,loss_high_large_escape_v2,loss_high_large_smooth_v3,loss_high_large_balanced_v5,loss_high_large_scurve_v6
```

Equivalent large variants were run in separate JSONL files:

- `tmp/t302h/loss_only_v2_large_forward_sweep.jsonl`
- `tmp/t302h/loss_only_v3_large_forward_sweep.jsonl`
- `tmp/t302h/loss_only_v4_large_forward_sweep.jsonl`
- `tmp/t302h/loss_only_v5_large_forward_sweep.jsonl`
- `tmp/t302h/loss_only_v6_large_forward_sweep.jsonl`

## Metrics

- `small_overpass_success`: low-small success; requires local root passage over the obstacle lane, no stance/touchdown semantic contact, no foot penetration, and bounded continuity.
- `large_avoid_success`: high-small/large success; requires clearance margin, no semantic contact, no root semantic occupancy, and bounded continuity.
- `semantic_task_violation`: main fail flag.
- `semantic_task_contact_violation`: stance/touchdown/foot penetration failure.
- `semantic_task_continuity_violation`: foot/root acceleration, jump, or boundary failure.
- `semantic_policy_margin_deficit`: missing avoidance distance; lower is better, `0` passes.
- `worst_max_to_median_step`, `worst_boundary_to_median_step`, `foot_accel_max_to_mean`, `root_accel_max_to_mean`: trajectory continuity/jitter.

## Key Results

Low-small mixed `vx=0.50, vy=0.25, yaw=1.00`:

- Baseline: `semantic_task=1`, `small_overpass=0`, `stance=0.035313`, `rootacc=31.425`, `score=945.5`.
- `loss_low_small_cont_v2`: `semantic_task=0`, `small_overpass=1`, `stance=0`, `footacc=14.540`, `rootacc=6.761`, `score=221.2`.
- Improvement: task failure removed, stance contact `-100%`, score `-76.6%`, root acceleration `-78.5%`, foot acceleration `-21.7%`.

Large forward:

- Baseline in v1 run: `semantic_task=1`, `large_avoid=0`, `margin=0.029888`, `jump=102.434`, `footacc=32.683`, `score=1196.5`.
- `loss_high_large_avoid_v1`: score `-28.0%`, jump `-54.5%`, footacc `-47.1%`, rootacc `-47.0%`, but margin worsened `0.029888->0.139955`.
- `loss_high_large_escape_v2`: margin cleared `0.030535->0`, but continuity failed (`jump=62.702`, `footacc=46.618`).
- `loss_high_large_smooth_v3`: continuity clean (`jump=1.933`, `footacc=24.295`) but margin failed (`0.066292`).
- `loss_high_large_balanced_v5`: continuity clean (`jump=1.732`, `footacc=18.649`) but margin failed (`0.076708`).
- `loss_high_large_scurve_v6`: continuity clean, but root semantic contact appeared (`root_on=0.066667`) and margin failed (`0.112994`).

High-small `0.46m`:

- Baseline was already clean in the tested rows.
- Several loss-only variants regressed high-small through new margin or continuity failures, so no high-small improvement is claimed.

## Result

Partial pass.

Loss-only testing found a strong low-small mixed solution: `loss_low_small_cont_v2`.

Loss-only testing did not find a large-forward solution satisfying both avoidance clearance and trajectory continuity. The tested high/large loss shapes split into two failure modes:

- strong avoidance clears margin but causes foot jump/acceleration spikes;
- strong continuity smooths the trajectory but lets root pass too close, or in v6 creates root semantic contact.

## Verification

- `pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`: `33 passed`
- `py_compile`: pass
- `git diff --check`: pass
- `git diff -- Go2Pvcnn/extension/batch_mpc_planner/config.py | wc -l`: `0`

## Git Refs

- Baseline Ref: working tree before this test-only loss-only pass
- Candidate Ref: working tree
- Key Files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`
  - `tmp/t302h/loss_only_low_small_mixed_sweep.jsonl`
  - `tmp/t302h/loss_only_v2_low_small_mixed_sweep.jsonl`
  - `tmp/t302h/loss_only_v2_large_forward_sweep.jsonl`
  - `tmp/t302h/loss_only_v3_large_forward_sweep.jsonl`
  - `tmp/t302h/loss_only_v4_large_forward_sweep.jsonl`
  - `tmp/t302h/loss_only_v5_large_forward_sweep.jsonl`
  - `tmp/t302h/loss_only_v6_large_forward_sweep.jsonl`
