# T302h Structural Selector Loss Sweep

## Purpose

Continue the T302h semantic obstacle jitter investigation with test-only changes. The goal was to try new directions without modifying production planner defaults, then compare semantic policy behavior, collision/contact rates, and trajectory continuity metrics.

## Stage

- `extension/batch_mpc_planner` test-only semantic MPC diagnostics
- Probe files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`

## Related Todo

- [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Procedure

- Added test-only structural loss injection by patching `extension.batch_mpc_planner.optimizer.compute_total_loss` inside the probe context only.
- Added class-split structural directions:
  - `struct_lowfoot_highbody`
  - `struct_lowfoot_highbody_strong`
  - `opt40_struct_lowfoot_highbody`
  - `struct_lowfoot_only`
  - `struct_lowfoot_largebody`
  - `struct_lowfoot_largebody_gentle`
  - `struct_lowfoot_largebody_gentle_smooth`
- Added semantic policy metrics:
  - low small + linear command should cross
  - high small / large should not cross
  - yaw-only should not force crossing
- Added selector directions:
  - `select_baseline_gentle_smooth`
  - `select_policy_pool`
- Ran real IsaacLab sweeps through `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`, `CUDA_VISIBLE_DEVICES=2`, `--device cuda:0`, 300 frames, single cycle.

## Commands

```bash
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small,large --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --variants baseline,opt40_body_hard_contact_progress,struct_lowfoot_highbody,struct_lowfoot_highbody_strong,opt40_struct_lowfoot_highbody
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --semantic-small-height-m 0.46 --variants baseline,opt40_body_hard_contact_progress,struct_lowfoot_highbody,struct_lowfoot_highbody_strong,opt40_struct_lowfoot_highbody
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small,large --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --variants baseline,struct_lowfoot_largebody_gentle_smooth,select_policy_pool
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --semantic-small-height-m 0.46 --variants baseline,struct_lowfoot_largebody_gentle_smooth,select_policy_pool
```

## Key Metrics

Low-small/large structural split:

- `struct_lowfoot_largebody_gentle` had one clean run with `policy_violation=0/6`, `stance/touchdown/root_on=0`, low-small `cross=2/3`, large `cross=0/3`.
- Its continuity was still weak in that run: `max_foot_accel_max_to_mean=38.835`, `max_root_accel_max_to_mean=46.742`.
- `struct_lowfoot_largebody_gentle_smooth` reduced continuity spikes in a later focused run:
  - `footacc 35.372 -> 22.710`
  - `boundary 11.437 -> 5.314`
  - `root_on 0.010 -> 0.000`
  - but policy remained unstable at `2/6` in that rerun.

Selector pool:

- Low-small/large:
  - baseline: `policy_violation=2/6`, `max_stance=0.028814`, `max_root_on=0.003333`, `max_footacc=31.733`, `max_rootacc=30.477`.
  - `struct_lowfoot_largebody_gentle_smooth`: `policy_violation=2/6`, `max_stance=0`, `max_root_on=0.006667`, `max_footacc=31.828`, `max_rootacc=23.718`.
  - `select_policy_pool`: `policy_violation=1/6`, `max_stance=0`, `max_root_on=0`, `max_footacc=23.544`, `max_rootacc=17.625`, but still missed one low-small crossing.
- High-small:
  - baseline: `policy_violation=1/3`, `max_root_on=0.016667`, `min_dist=0.034`, `max_jump=77.193`.
  - `struct_lowfoot_largebody_gentle_smooth`: `policy_violation=1/3`, `min_dist=0.305`, `max_footacc=27.185`.
  - `select_policy_pool`: `policy_violation=0/3`, `max_stance=0`, `max_root_on=0`, `min_dist=0.106`, `max_footacc=22.950`, `max_rootacc=17.841`.

## Result

Partial pass.

The best new direction is not a single scalar loss. The most useful evidence is:

- class-split structural loss removes stance/touchdown semantic contact more reliably than scalar-only tuning;
- adding smoothness improves continuity but does not fully stabilize the semantic policy;
- multi-candidate selection improves robustness and high-small behavior, but the current pool still has `1/6` low-small/large policy violation.

## Conclusion

Do not productionize this yet. The next direction should be a stronger test-only mode/candidate design or hard policy gate:

- explicit low-small crossing candidate for linear commands;
- explicit high/large avoidance candidate;
- selector score should reject wrong crossing policy before comparing smoothness;
- then rerun multi-cycle low-small/high-small/large before production changes.

## Verification

- `pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`: `11 passed`
- `python -m py_compile Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`: pass
- `git diff --check`: pass
- `git diff -- Go2Pvcnn/extension/batch_mpc_planner/config.py | wc -l`: `0`

## Git Refs

- Baseline Ref: working tree before this test-only structural sweep
- Candidate Ref: working tree
- Key Files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`
  - `tmp/t302h/struct_*_sweep.jsonl`
  - `tmp/t302h/selector_*_sweep.jsonl`
