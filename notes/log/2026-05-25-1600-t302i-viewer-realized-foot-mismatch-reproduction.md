# T302i Viewer Realized-Foot Mismatch Reproduction

## Purpose

Reproduce the user-reported current viewer failure: low-small obstacle crossing shows planned touchdown/swing markers that do not match actual foot ends, and swing appears discontinuous.

## Stage

- `extension/batch_mpc_planner`
- `extension/viz/go2_foostep_planner.py`
- semantic MPC viewer/runtime diagnostics
- related todo: [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Command / Procedure

```bash
mkdir -p tmp/t302i-viewer-realized-foot-mismatch
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 24 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00' --variants baseline > tmp/t302i-viewer-realized-foot-mismatch/rolling25_low_small_repro.jsonl 2>&1
```

## Input Conditions

- Python: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- IsaacLab device: `--device cuda:0`
- Visible GPU: `CUDA_VISIBLE_DEVICES=0`
- Case: `small`
- Commands:
  - `forward_v050: 0.50 0.00 0.00`
  - `forward_yaw_v050_vy025_yaw100: 0.50 0.25 1.00`
- Total rollout: `--requested-n-frames 300`
- Segment playback frame: `24`
- Warmup: `6`
- Output: `tmp/t302i-viewer-realized-foot-mismatch/rolling25_low_small_repro.jsonl`

Viewer entrypoint smoke:

```bash
timeout -s INT -k 20s 90s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc --n-frames 25 --plan-dt 0.02 --warmup-steps 6 --scripted-command "0.50 0.00 0.00" --scripted-command-cycles 1 > tmp/t302i-viewer-realized-foot-mismatch/viewer_mpc_low_small_smoke.log 2>&1
timeout -s INT -k 20s 120s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc --n-frames 50 --plan-dt 0.02 --warmup-steps 6 --scripted-command "0.50 0.00 0.00" --scripted-command-cycles 1 > tmp/t302i-viewer-realized-foot-mismatch/viewer_mpc_low_small_smoke_n50.log 2>&1
```

## Key Metrics

Cycle rows:

| Command | Contact Violation | Continuity Violation | Foot-Over | Small Overpass | Playback Foot Error Max | Rolling Terminal Foot Error Max | Replan Boundary Ratio | Foot Accel Ratio | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `forward_v050` | `0` | `1` | `1` | `0` | `0.286667` | `0.286667` | `14.279280` | `22.193964` | `0.396874` |
| `forward_yaw_v050_vy025_yaw100` | `0` | `1` | `1` | `0` | `0.288352` | `0.375311` | `19.159033` | `29.451223` | `0.324858` |

Other key values:

- `touchdown_on_semantic_rate=0` for both rows.
- `foot_semantic_penetration_rate=0` for both rows.
- `stance_on_semantic_rate=0` for both rows.
- `playback_root_error_max` is near zero: `2.98e-08` and `1.19e-07`.
- Variant summary: `semantic_task_violation_count=2`, `continuity_violation_count=2`, `contact_violation_count=0`, `max_rolling_segment_terminal_foot_error=0.375311`.

## Result

Pass as reproduction, with caveats.

The real IsaacLab probe reproduced a planned-vs-realized foot mismatch and swing discontinuity indicators without semantic contact/penetration. The mismatch is foot-specific rather than root playback: root error is near zero while foot error is about `0.29m` at playback frame `24`, with rolling terminal foot error up to `0.375m`.

The viewer entrypoint smoke also exposed a separate entrypoint mismatch:

- `--planner-backend mpc --n-frames 25` still hits a together fixed-horizon guard and exits with `ValueError`.
- `--planner-backend mpc --n-frames 50` attaches the MPC manager and reaches playback setup, but the smoke was interrupted by timeout before natural completion.

## Conclusion

The user's screenshot complaint is reproduced numerically as a viewer/runtime mismatch front:

- The low-small route has no semantic contact or penetration.
- The feet do pass over the low-small object by the foot-over metric.
- The accepted planned crossing is still visually unsafe because the realized/readback foot can diverge from planned foot by roughly `0.29-0.38m`, and the swing path has high boundary/acceleration discontinuity metrics.

This should not be treated as solved by the prior T302h task gate. The next diagnostic should record per-frame/per-leg planned foot, realized foot, and touchdown marker positions around the worst frames (`22`, `24`, and `49`) before changing planner losses.

## Follow-Up

- If `playback_foot_error_max` remains high, create a focused diagnostic for per-frame/per-leg planned foot, realized foot, and touchdown marker positions.
- If task metrics regress, route the regression back into T302h rather than treating it only as viewer mismatch.
- Fix or isolate the viewer entrypoint horizon guard so MPC viewer can exercise rolling25 semantics directly with `--n-frames 25`.

## Git Refs

- Baseline Ref: `working tree @ c54dc5c`
- Candidate Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - `../../tmp/t302i-viewer-realized-foot-mismatch/rolling25_low_small_repro.jsonl`
  - `../../tmp/t302i-viewer-realized-foot-mismatch/viewer_mpc_low_small_smoke.log`
  - `../../tmp/t302i-viewer-realized-foot-mismatch/viewer_mpc_low_small_smoke_n50.log`
