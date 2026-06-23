# Model14700 Flat-Small Eval

## Purpose

Evaluate `2026-06-17_12-01-10/model_14700.pt` for semantic small-object collision/crossing behavior and policy-vs-MPC foot trajectory overlap, then compare with previous recorded checkpoints.

## Stage

- Checkpoint evaluation / flat-small low-small crossing behavior
- `mpc_policy_eval.py` tracking / MPC reference overlap
- Related todos:
  - [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)
  - [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Code Compatibility Note

Current flat-small configs no longer mount the old `semantic_contact_small` sensor. `Go2Pvcnn/scripts/mpc_policy_eval.py` was made compatible with both routes:

- if `semantic_contact_small` exists, keep using `semantic_small_force_matrix_w`
- otherwise infer small semantic contact from the current semantic/elevation map plus ordinary `contact_forces`

This means `model_14700.pt` contact counts use the current map-contact metric. Old `model_23600.pt` / `model_28900.pt` summaries used the old semantic contact sensor, so the contact-rate comparison is useful but not perfectly apples-to-apples.

## Commands

Controlled crossing:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 1800s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --headless --device cuda:0 \
  --mode controlled_crossing \
  --run-dir /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-17_12-01-10 \
  --checkpoint model_14700.pt \
  --num-envs 16 \
  --num-rounds 1 \
  --max-steps 1000 \
  --terrain-rows 0,1,2,3,4,5,6,7,8,9 \
  --terrain-cols 0 \
  --crossing-speeds 0.6,0.8,1.0 \
  --crossing-lateral-offsets=-0.08,0.0,0.08 \
  --crossing-obstacles-per-env 24 \
  --output-dir logs/mpc_policy_eval/flat_small_20260617_120110_model14700_controlled_crossing
```

Tracking, 100 steps:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 900s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --headless --device cuda:0 \
  --mode tracking \
  --run-dir /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-17_12-01-10 \
  --checkpoint model_14700.pt \
  --num-envs 4 \
  --num-rounds 1 \
  --max-steps 100 \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.4 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/flat_small_20260617_120110_model14700_tracking
```

Tracking, 20 steps, comparable with the old `model_14000.pt` tracking smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 600s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --headless --device cuda:0 \
  --mode tracking \
  --run-dir /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-17_12-01-10 \
  --checkpoint model_14700.pt \
  --num-envs 4 \
  --num-rounds 1 \
  --max-steps 20 \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.4 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/flat_small_20260617_120110_model14700_tracking_20step
```

Verification:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py
git diff --check
```

## Outputs

- Controlled crossing: [../../logs/mpc_policy_eval/flat_small_20260617_120110_model14700_controlled_crossing/2026-06-23_19-58-02-463800/summary.json](../../logs/mpc_policy_eval/flat_small_20260617_120110_model14700_controlled_crossing/2026-06-23_19-58-02-463800/summary.json)
- Tracking 100-step: [../../logs/mpc_policy_eval/flat_small_20260617_120110_model14700_tracking/2026-06-23_20-34-22-041127/summary.json](../../logs/mpc_policy_eval/flat_small_20260617_120110_model14700_tracking/2026-06-23_20-34-22-041127/summary.json)
- Tracking 20-step: [../../logs/mpc_policy_eval/flat_small_20260617_120110_model14700_tracking_20step/2026-06-23_20-42-09-388547/summary.json](../../logs/mpc_policy_eval/flat_small_20260617_120110_model14700_tracking_20step/2026-06-23_20-42-09-388547/summary.json)

## Key Metrics

Controlled crossing for `model_14700.pt`:

- opportunity envs: `16/16`
- root crossed: `7/16`
- map-contact small collision envs: `3/16`
- touchdown-on-small envs: `3/16`
- foot-over envs: `2/16`
- overpass success: `0/16`
- command source max error: `0.0`
- planned root direction cosine: `0.999041`
- planned root lateral ratio: `0.043782`

Previous controlled crossing summaries:

- `model_23600.pt`: opportunity `15/16`, root crossed `14`, small contact `11/16`, foot-over `0`, success `0/15`
- `model_28900.pt`: opportunity `15/16`, root crossed `14`, small contact `7/16`, foot-over `0`, success `0/15`

Tracking overlap:

| Checkpoint / run | Steps | Mean foot error | P95 foot error | Root direction cosine | Root lateral ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_14000.pt` old smoke | 20 | `0.09341m` | `0.20858m` | `0.999999` | `0.00104` |
| `model_14700.pt` new comparable run | 20 | `0.08757m` | `0.20221m` | `0.999689` | `0.02495` |
| `model_14700.pt` longer run | 100 | `0.13909m` | `1.01024m` | `0.918492` | `0.39544` |

## Result

Diagnostic pass. `model_14700.pt` shows lower current small-contact rate in the controlled crossing setup and has the first recorded foot-over events in this comparison set, but it still does not satisfy the clean overpass definition because overpass success remains `0/16`.

For policy-vs-MPC overlap, the strict comparable 20-step tracking run is slightly better than the old `model_14000.pt` tracking smoke (`0.0876m` mean vs `0.0934m`; `0.202m` p95 vs `0.209m`). The 100-step run gets worse over time, so longer-horizon drift or resets remain a separate concern.

## Conclusion

There is partial improvement:

- semantic small-contact metric improved under the current map-contact route
- foot-over events appeared (`2/16`)
- 20-step MPC foot-reference overlap is slightly better than the old smoke

There is not yet a solved crossing behavior:

- controlled crossing clean success is still `0`
- root crossing count dropped to `7/16`, so the policy is not reliably completing the path
- old contact counts are not perfectly comparable because the sensor route changed

## Verification Result

- pycompile: exit `0`
- focused eval tests: `23 passed`
- `git diff --check`: exit `0`

## Follow-up

- Re-run older checkpoints with the new map-contact fallback if an apples-to-apples contact-rate comparison is needed.
- For behavior, continue treating clean overpass success and root-cross completion as the hard gates, not just lower contact count.

## Git Refs

- Baseline Ref: working tree
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../logs/mpc_policy_eval/flat_small_20260617_120110_model14700_controlled_crossing/2026-06-23_19-58-02-463800/summary.json](../../logs/mpc_policy_eval/flat_small_20260617_120110_model14700_controlled_crossing/2026-06-23_19-58-02-463800/summary.json)
  - [../../logs/mpc_policy_eval/flat_small_20260617_120110_model14700_tracking_20step/2026-06-23_20-42-09-388547/summary.json](../../logs/mpc_policy_eval/flat_small_20260617_120110_model14700_tracking_20step/2026-06-23_20-42-09-388547/summary.json)
