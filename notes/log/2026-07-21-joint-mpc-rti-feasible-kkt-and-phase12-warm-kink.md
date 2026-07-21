# Joint MPC RTI Feasible KKT And Phase-12 Warm Kink

## Purpose

Close the stale active-KKT blocker, preserve the approved differentiable semantic-field contract, and locate the remaining ranked flat tracking/jump failure before more tuning.

## Stage

Task 13 flat behavior diagnosis on `work/joint-mpc-kinematic`.

## Solver Result

- The fixed two-refinement solver now starts from feasible zero, moves toward each KKT target only to the first blocking joint-position or joint-velocity bound, activates reached bounds, and repeats the two approved refinements.
- Permanent equalities such as `delta z0=0` are active from the start and are excluded from subsequent blocking-fraction calculations.
- The real cold QP is feasible and descending after the change; dense/scan final difference is `9.64e-8` with identical active masks.
- Focused QP verification passed, and the current QP/nominal/seven-loss/RTI union is `32 passed`.

## Perception Contract

The approved perception path is already present in both design and implementation: raw discrete semantic masks feed fixed grouped convolution, propagated class height, Scharr XY gradients, and differentiable bilinear trajectory-coordinate queries. The optimizer consumes the soft fields; acceptance detectors continue to consume raw semantic/elevation/geometry. No eighth loss or discrete semantic optimizer branch was introduced.

## Ranked Diagnostic

Read-only monitored CUDA configuration:

```text
regularization=0.1
command_scale=0.45
root_position_trust=0.01
contact_anchor_xy=200
contact_ground=32
command_early_swing=0
swing_speed_early=1
command_weight=1
```

The run completed in `13.11s` with about `2.10GiB` task GPU memory. All three cells had `25/25` valid nodes, no alpha-zero selection, passing joint limits/margins, passing stance metrics, `20ms` foot lead, and zero root leak before foot.

The remaining root failures are not uniform half-speed. Away from the gait boundary, signed command root velocity starts near the `0.45` nominal scale and rises toward the target. At edge `12`, however, the forward trace jumps approximately `(-64.8,-32.7)mm` and the backward trace jumps approximately `(+71.3,+26.6)mm` in one `20ms` edge. The corresponding body velocities are about `(-3.44,-1.15)m/s` and the yaw-rate spikes are about `-3.71/+4.00rad/s`.

The same discontinuity already exists before LQ in the shifted warm nominal: forward approximately `(-63.8,-32.2)mm`, backward approximately `(+70.3,+26.2)mm`. Alpha `1` reduces total loss but changes the discontinuity only slightly. Therefore line search is not creating the jump; a rolling horizon phase-boundary kink is being shifted into published `x1`.

Full-horizon tracing proves the lineage. The first accepted update creates its largest internal jump at edge `11`; subsequent warm starts move the same dominant edge through `10,9,...,1,0`, while its magnitude grows from roughly `19mm` to `80mm`. After publication, the next gait boundary appears around edge `22` and starts the same shift. This is a repeated swing-transition objective kink, not terminal extrapolation.

Worst swing clearance occurs at `tau=1`: forward frame/phase `11`, leg `3`; backward frame/phase `23`, leg `2`. Increasing the parabolic `h_swing` cannot change these endpoints. Raising existing `smooth_second` from `1` to `5` reduces root jump only to about `66/64mm`, while backward stance slip exceeds `0.5mm` and stationary ratio falls below `1`, so it is rejected.

Two existing-parameter checks expose the trade-off:

- `command_early_swing=0.1` reduces signed root jump to about `32-33mm`, but foot lead collapses from `20ms` to `0ms`, root leak rises to about `3.3mm`, and zero drift worsens to about `5.4mm`; reject.
- `smooth_first=0.2` preserves `20ms` lead and zero leak but leaves signed jump at about `81/77mm`; reject.

The current `command_early_swing=0` tune therefore makes the rolling objective time-inconsistent at every future swing transition: command pressure vanishes at that internal edge, contact support changes, and the resulting kink is later shifted into the published edge. Existing scalar tuning tested here cannot simultaneously preserve the strict lead/leak gate and suppress that kink.

Zero-command drift remains separate. With the ranked base it is about `0.452mm`; `command_early_swing=0` removes first-edge command pressure even for zero command.

## Result

Partial. The active-KKT blocker is closed and the differentiable convolution perception design is already implemented. Task 13 remains open on the phase-12 shifted-warm kink, zero drift, and swing touchdown endpoint clearance. Production regularization is restored to the stronger ranked base `0.1`; no nominal formula, loss category, KKT count, or line-search rule changed in this checkpoint.

## Follow-Up

Do not continue blind scalar sweeps. A minimal design amendment is required to define how startup-only foot lead should coexist with a time-consistent rolling objective. It must stay inside the existing command/swing loss families, add no candidate/filter/recovery path, and avoid creating future horizon edges whose objective differs merely because they will later become `x1`.

## Git Refs

- Baseline ref: `724a1c3` plus current working tree
- Candidate ref: `724a1c3` plus current working tree
- Key files: `solver/trajectory_qp.py`, `runtime/warm_start.py`, `config.py`, `terrain/field_builder.py`, `terrain/query.py`
