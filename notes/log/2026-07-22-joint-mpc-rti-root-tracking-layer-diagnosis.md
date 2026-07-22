# Joint MPC RTI Root Tracking Layer Diagnosis

## Scope

- Behavior changes: none
- Added evidence: viewer-only root tracking layer report
- Scenario: actual-state S4 sphere `small_forward`, `cuda:1`
- Report: `/tmp/joint_mpc_viewer_mixed_swing_floor_root_layers.json`
- Executed RTI cycles: `49`
- Ranked/formal: not run

## Diagnostic Contract

For every published edge, the report reconstructs body-frame XY velocity and command error at four layers:

1. actual published x1
2. warm/cold nominal x1
3. nominal plus full QP direction
4. nominal plus selected line-search alpha times the direction

It also records line alpha, lifecycle mode, and x1 root XY trust utilization. A pure tensor test was RED because the helper was missing, then GREEN at `1 passed`.

## Evidence

| Layer | All 49 cycles | Warm-only 48 cycles |
| --- | ---: | ---: |
| actual | `0.2239295m/s` | `0.1963434m/s` |
| nominal | `0.0904533m/s` | `0.0715044m/s` |
| full QP | `0.2334324m/s` | `0.2060442m/s` |
| selected | `0.2239295m/s` | `0.1963434m/s` |

- Full QP error is worse than nominal in `41/49` cycles.
- Line search improves the full-QP root error in only `3/49` cycles.
- Root XY trust utilization averages `0.2782`; it saturates only at cycles `0, 1, 29, 48`.
- The cold first cycle has nominal error `1.0m/s`, full/selected error `1.54806m/s`, and saturated trust.
- Large selected errors repeat at cycles `12`, `24`, `36`, and `48`, aligned with the 12-frame half-cycle boundary; obstacle/gait transition cycles `16-18` also spike.

The exact formal metric is reproduced by the `actual` and `selected` columns. Excluding the one cold cycle yields `0.19634m/s`, but the formal metric intentionally evaluates every valid edge and must not be changed to manufacture a pass.

## Root Cause

Warm nominal command tracking is already below threshold, so `command_scale=0.45` is not the steady-state blocker. The full constrained QP commonly moves x1 away from the command-optimal nominal because the hard published-foot rows couple root translation and joints through complete FK, while command tracking remains one normalized soft loss among seven. Five-alpha line search minimizes the same aggregate objective and usually accepts alpha `1`; it is not a command-specific repair stage.

Trust is a cap, not the missing authority. Increasing it would permit a direction that is already moving farther from the command and is therefore contraindicated. The cold first-edge hold and periodic gait-boundary KKT corrections explain why the aggregate crosses `0.2m/s` even though warm-only selected tracking narrowly passes.

## Decision

Keep the acceptance metric, mixed six-row KKT, and all frozen architecture counts unchanged. Do not sweep root trust or command weights. The next behavior proposal must explicitly define root/foot priority at published x1 while preserving stance XY, swing floor, cold foot lead, joint bounds, one QP/RTI, five alphas, four filters, and seven loss families.

## Fresh Regression

- QP/scan/loss/line-search/RTI/backend focused union: `91 passed in 23.90s`
- Fixed-contract and terrain-field supplement: `30 passed in 6.18s`
- modified production/viewer `py_compile`: exit `0`
- `git diff --check`: exit `0`
