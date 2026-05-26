# T302k IsaacLab Current-Foot Touchdown Check

## Purpose

Verify in IsaacLab that each parametric replan starts from the current IsaacLab foot positions, and measure whether planned touchdowns agree with those current stance/current-foot positions.

## Stage

`extension/batch_mpc_planner` parametric MPC plus IsaacLab GPU3 runtime probe.

## Related Todo

[../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)

## Command

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 --variants parametric_v1 --requested-n-frames 300 --warmup-steps 6 \
  --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,pure_yaw:0.00 0.00 1.00' \
  > tmp/t302k-parametric-mpc/low_small_parametric_v1_t302k_currentfoot_gpu3.jsonl 2>&1
```

Before IsaacLab, local probe tests passed:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py -q
```

Result: `35 passed`.

## Input Conditions

- Variant: `parametric_v1`.
- Commands:
  - `forward_v050`
  - `forward_yaw_v050_vy025_yaw100`
  - `pure_yaw`
- Added rolling-segment metrics:
  - `replan_initial_foot_error_max`
  - `replan_initial_touchdown_to_current_foot_error_max`

## Key Metrics

Summary:

```json
{
  "cycle_count": 3,
  "fk_foot_over_low_small_success_count": 3,
  "max_fk_foot_small_penetration_rate": 0.0,
  "max_fk_stance_on_small_rate": 0.0,
  "max_fk_touchdown_on_small_rate": 0.0,
  "max_replan_initial_foot_error": 0.0,
  "max_replan_initial_touchdown_to_current_foot_error": 0.44666171073913574,
  "max_terminal_planned_vs_fk_foot_error": 3.814813680946827e-06,
  "max_touchdown_ik_fk_error": 0.6610360145568848
}
```

Per command:

| Command | Initial Foot Error | Initial Touchdown To Current Foot | Touchdown IK/FK | Terminal Planned/FK | Foot-Over |
| --- | ---: | ---: | ---: | ---: | ---: |
| `forward_v050` | `0.0` | `0.4243` | `0.4966` | `1.92e-6` | `1` |
| `forward_yaw_v050_vy025_yaw100` | `0.0` | `0.4467` | `0.6610` | `3.81e-6` | `1` |
| `pure_yaw` | `0.0` | `0.3590` | `0.5345` | `1.92e-6` | `1` |

## Result

Partial.

Frame0 exported feet are now exactly current IsaacLab feet at every rolling replan boundary measured by the probe. However, planned touchdown markers still differ from those current stance/current-foot positions by up to `0.4467m`; this matches the user's observation that actual foot and touchdown do not land together.

## Conclusion

T302k.11 fixed the replan initial foot discontinuity, but not touchdown/current-stance consistency. The next change should separate current stance touchdown anchors from future swing touchdown targets, or export frame-aware/current-contact touchdowns instead of broadcasting only the future optimized touchdown target across all frames.

## Follow-Up

Open T302k.12: make current stance/current-foot touchdowns consistent at replan boundaries while preserving future touchdown targets for swing legs.

## Git Refs

- Baseline Ref: `working tree @ 1b799cd` plus T302k local changes before IsaacLab current-foot check.
- Candidate Ref: `working tree 2026-05-26 17:17 +0800`.
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
