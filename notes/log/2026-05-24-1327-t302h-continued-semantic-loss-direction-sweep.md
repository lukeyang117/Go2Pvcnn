# T302h Continued Semantic Loss Direction Sweep

## Purpose

Continue testing new semantic-obstacle loss directions in the probe only, without changing production MPC defaults, until the metrics show a better candidate for the user's desired behavior.

## Stage

- `extension/batch_mpc_planner`
- test-only T302h semantic obstacle diagnostics
- real IsaacLab 300-step MPC probes

## Related Todo

- [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Command / Procedure

Helper verification:

```bash
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
python -m py_compile \
  Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py \
  Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

Representative sweeps:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py \
  --device cuda:0 --cases small,large --cycles 1 \
  --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 \
  --variants baseline,body_crossing_progress_only,opt40_body_crossing_progress,body_hard_contact_crossing_progress,opt40_body_hard_contact_progress

CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py \
  --device cuda:0 --cases small --cycles 1 \
  --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 \
  --semantic-small-height-m 0.46 \
  --variants baseline,body_crossing_progress_only,opt40_body_crossing_progress,body_hard_contact_crossing_progress,opt40_body_hard_contact_progress
```

## Input Conditions

- Same T302h setup as prior log:
  - one env
  - S4 semantic anchors
  - commands `forward_v050`, `forward_yaw_v050_vy025_yaw100`, `yaw100`
  - `300` frames
  - high-small override `0.46m`

## Tested Directions

- `body_light_*`: lighter high/large body margin than previous `body_stance_crossing`.
- `hard_contact_*`: direct foot contact avoidance without soft field/root push.
- `crossing_progress_only`: low-small crossing progress without increasing swing clearance.
- `long_swing_*`: longer swing window and higher nominal swing height.
- `opt40_*`: more optimizer steps (`40`) and lower lr (`0.015`) to test whether failures are budget/convergence-limited.
- `foot_soft_*`: foot-only semantic soft field with low-small progress.
- `support_touchdown_*`: stronger touchdown/support search away from semantic surfaces.

## Key Metrics

Most useful single-cycle result:

| Variant | Case | contact_sum | pen_max | root_max | stance_max | min_dist | cross_count | jump_max | boundary_max | R2_min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `opt40_body_hard_contact_progress` | low-small | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.149 | 2/3 | 20.11 | 8.27 | 0.032 |
| `opt40_body_hard_contact_progress` | large | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.350 | 1/3 | 13.60 | 5.34 | 0.422 |
| `opt40_body_hard_contact_progress` | high-small | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.075 | 1/3 | 20.61 | 9.02 | 0.678 |

Follow-up with explicit risk:

| Variant | Case | contact_sum | min_dist | cross_count | jump_max | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `opt40_body_hard_contact_risk_progress` | large | 0.0000 | 0.350 | 0/3 | 27.60 | improves high/large avoidance |
| `opt40_body_hard_contact_risk_progress` | low-small | 0.0524 | 0.144 | 1/3 | 11.86 | regresses low-small stance |
| `opt40_body_hard_contact_risk_progress` | high-small | 0.0008 | 0.350 | 0/3 | 7.28 | high-small avoidance improves but penetration appears |

Rejected directions:

| Direction | Reason |
| --- | --- |
| `long_swing_*` | reduces some stance contact but worsens large jump/boundary and does not reach zero low-small stance robustly |
| `foot_soft_*` | keeps high-small clean but reintroduces low-small stance contact |
| `support_touchdown_*` | does not reliably reduce stance contact and worsens continuity in opt40 form |
| `risk_*` combined with low crossing | improves high/large but disturbs low-small crossing/stance |

## Result

Partial pass as direction search.

The best current scalar/config-only hypothesis is `opt40_body_hard_contact_progress`: it was the first tested direction to hit zero semantic contact on low-small and large in the same single-cycle sweep while keeping large trajectory jump much lower than several previous variants. It still fails high-small avoidance robustness in a later seed/run (`min_dist=0.075`, `cross=1/3`) and is not production-ready.

## Conclusion

Existing config-only weight tuning is hitting a three-way conflict:

- low-small progress wants root/path crossing,
- high-small/large wants root/body lateral avoidance,
- foot no-stance wants differentiable foothold displacement rather than only hard semantic contact penalties.

The next direction should be test-only structural loss, not more scalar-only sweeps:

- class-conditioned root/body margin: apply to large/high-small, not low-small
- low-small foothold exclusion: differentiable foot XY soft field for stance/touchdown only, with no root push
- optional optimizer budget gate: keep `opt40` only as a diagnostic, not as the likely training default

## Verification

- helper pytest: `5 passed`
- py_compile: pass
- real IsaacLab sweeps: completed with exit `0`

## Git Refs

- Baseline Ref: working tree with T302h probe
- Candidate Ref: test-only probe variants, no production MPC default change
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py)
