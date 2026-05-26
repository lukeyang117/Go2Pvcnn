# T302h Clearance Jitter Selector Sweep

## Purpose

Continue T302h with test-only selector/loss variants, using real IsaacLab metrics to verify directions before any production planner change.

## Stage

- `extension/batch_mpc_planner` test-only semantic MPC diagnostics
- Probe files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`

## Related Todo

- [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Procedure

- Added and tested probe-only directions:
  - `select_policy_class_hardcross_margin`
  - `select_policy_class_jitter_margin`
  - `select_policy_class_risk_jitter_margin`
  - `select_policy_class_priority_jitter_margin`
  - `select_policy_class_clearance_jitter_margin`
- Added `semantic_clearance_policy_violation` as a diagnostic policy metric:
  - low small + linear command still must cross;
  - high-small / large may pass around the obstacle without being treated as a failure;
  - avoid cases fail on margin deficit, stance/root semantic contact, or repeated foot penetration;
  - tiny single-frame swing-foot penetration is tolerated in the diagnostic selector.
- All changes stayed inside test/probe code; production `Go2Pvcnn/extension/batch_mpc_planner/config.py` remained unchanged.

## Commands

```bash
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small,large --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --variants baseline,select_policy_pool,select_policy_class_wide_margin,select_policy_class_hardcross_margin > tmp/t302h/policy_class_hardcross_low_large_sweep.jsonl 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --semantic-small-height-m 0.46 --variants baseline,select_policy_pool,select_policy_class_wide_margin,select_policy_class_hardcross_margin > tmp/t302h/policy_class_hardcross_high_small_sweep.jsonl 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small,large --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --variants baseline,select_policy_class_jitter_margin,select_policy_class_risk_jitter_margin > tmp/t302h/policy_class_risk_jitter_low_large_sweep.jsonl 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases large --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --variants select_policy_class_priority_jitter_margin,select_policy_class_clearance_jitter_margin > tmp/t302h/clearance_tolerance_large_sweep.jsonl 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --semantic-small-height-m 0.46 --variants select_policy_class_priority_jitter_margin,select_policy_class_clearance_jitter_margin > tmp/t302h/clearance_tolerance_high_small_sweep.jsonl 2>&1
```

## Key Metrics

Hard-cross selector:

- Low/large: `select_policy_class_hardcross_margin` regressed to `policy_violation=2/6`.
- Conclusion: stronger low-small crossing loss is not the right next axis; it can still miss low-small forward and can worsen large forward selection.

Wide-margin selector:

- Low/large first run: `select_policy_class_wide_margin` reached `policy_violation=0/6`, `max_stance=0`, `max_root_on=0`, `max_pen=0.000833`.
- High-small: `policy_violation=0/3`, `max_stance=0`, `max_root_on=0`, `max_pen=0.000833`.
- Weakness: continuity spikes remained high in some runs (`max_footacc=51.849` low/large).

Jitter-aware selector:

- Lowered continuity spikes in one low/large comparison:
  - `select_policy_class_wide_margin`: `max_footacc=26.383`, `max_jump=61.397`.
  - `select_policy_class_jitter_margin`: `max_footacc=17.598`, `max_jump=8.624`.
- Weakness: still retained a large-forward legacy policy violation in that run.

Risk/priority jitter selector:

- `select_policy_class_risk_jitter_margin` could recover low/large `policy_violation=0/6`, but large forward-yaw sometimes selected near-obstacle candidates with high `jump=68.069`.
- `select_policy_class_priority_jitter_margin` kept stance/root semantic contact at `0`, with lower `max_footacc=15.895` and `max_jump=14.328` in one low/large run, but legacy policy still counted large forward bypass as `1/6` because crossing the obstacle projection line is not necessarily collision.

Clearance-aware diagnostic selector:

- Large targeted sweep: `select_policy_class_clearance_jitter_margin` had `legacy=0/3`, `clearance=0/3`, `max_footacc=13.333`, `max_jump=20.804`, `max_root_on=0`, `max_pen=0.000833`, `max_margin=0`.
- High-small targeted sweep: `legacy=0/3`, `clearance=0/3`, `max_footacc=10.159`, `max_jump=5.632`, `max_root_on=0`, `max_pen=0`, `max_margin=0`.

## Result

Partial pass with a sharper direction.

- Best next direction is not `hardcross`.
- Best current test-only idea is selector semantics: class-conditioned candidate pools plus a clearance-aware policy metric and jitter-aware ranking.
- The probe now shows that old `crossed_obstacle_along_command` over-penalizes a valid large-obstacle bypass when the root clears the semantic object laterally.

## Conclusion

Before production changes, use a multi-cycle acceptance gate centered on:

- low-small linear commands must cross;
- high-small and large must avoid by clearance, not by never crossing an infinite projection line;
- selector must reject margin/root/stance violations before optimizing jitter;
- repeated foot penetration should remain a failure, while single-frame tiny swing-foot penetration should be diagnostic, not a hard policy blocker.

## Verification

- `pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`: `17 passed`
- `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`: pass
- `git diff --check`: pass
- `git diff -- Go2Pvcnn/extension/batch_mpc_planner/config.py | wc -l`: `0`

## Git Refs

- Baseline Ref: working tree before this test-only selector sweep
- Candidate Ref: working tree
- Key Files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`
  - `tmp/t302h/policy_class_hardcross_low_large_sweep.jsonl`
  - `tmp/t302h/policy_class_hardcross_high_small_sweep.jsonl`
  - `tmp/t302h/policy_class_jitter_low_large_sweep.jsonl`
  - `tmp/t302h/policy_class_risk_jitter_low_large_sweep.jsonl`
  - `tmp/t302h/policy_class_priority_jitter_low_large_sweep.jsonl`
  - `tmp/t302h/policy_class_clearance_jitter_low_large_sweep.jsonl`
  - `tmp/t302h/clearance_tolerance_large_sweep.jsonl`
  - `tmp/t302h/clearance_tolerance_high_small_sweep.jsonl`
