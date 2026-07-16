# Joint MPC RTI Stop-On-Small Floating Reproduction

- Purpose: reproduce the reported transition from flat walking to stopping over a crossable small object, where semantic avoidance prevents feet from returning to support.
- Stage: `extension/joint_mpc_rti` rolling x1 gait, semantic loss, and stance grounding.
- Related todo: [T302v.5](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `625768f`.
- Candidate Ref: `5fa0b8d`; no planner or test code changed.
- Key Files: `planner.py`, `config.py`, `losses/semantic.py`, `losses/contact.py`, `small_obstacle_crossing_probe.py`.

## Procedure

- Use the existing native sphere/cuboid/cylinder/capsule/cone construction and world-fixed object coordinates.
- Walk forward at `0.2m/s` from flat ground.
- Sweep `13` root stop offsets from `-0.24m` through `+0.24m` around each object center, for `65` batched cases.
- Once each root reaches its stop position, switch its command to `[0,0,0]` and publish only rolling `x1` for `64` hold cycles.
- A grounded support is a scheduled stance foot whose center satisfies `abs(foot_z - queried_height - 0.022) <= 0.012m`.
- A floating stance frame has scheduled contact but foot surface gap above `0.03m`.

## Reproduction

All `65/65` cases reproduce floating scheduled stance and all `65/65` contain at least `64` consecutive hold cycles with zero grounded support.

Representative cuboid, root stopped at the object center:

```text
hold cycles: 64
scheduled stance-leg frames: 128
floating stance-leg frames: 128
grounded support legs per cycle: 0
maximum stance surface gap: 0.295m
maximum stop XY drift: 0.0117m
```

The per-leg contact schedule remains fixed trot, so each leg has `32` scheduled stance frames. The failure is geometric support, not a missing contact schedule.

## Semantic/Height Controls

For the same cuboid stop phase over `64` hold cycles:

| Input | Grounded / stance-leg frames | Floating frames | Longest zero-support run | Max stance gap |
| --- | ---: | ---: | ---: | ---: |
| flat | `128/128` | `0` | `0` | `0.00124m` |
| height only | `114/128` | `5` | `4` | `0.10958m` |
| semantic only | `0/128` | `114` | `64` | `0.17858m` |
| semantic + height | `0/128` | `128` | `64` | `0.26261m` |

This isolates the sustained failure to the small-semantic branch rather than the fixed trot schedule or elevation step alone.

## Loss Ablation

For `32` stopped hold cycles (`64` scheduled stance-leg frames):

| Ablation | Grounded frames | Floating frames | Longest zero-support run |
| --- | ---: | ---: | ---: |
| current | `0/64` | `64` | `32` |
| no touchdown avoidance | `0/64` | `64` | `32` |
| no foot clearance | `0/64` | `64` | `32` |
| no calf/thigh clearance | `0/64` | `64` | `32` |
| no foot-over | `48/64` | `16` | `0` |
| only foot-over retained | `0/64` | `64` | `32` |
| all semantic losses disabled | `50/64` | `5` | `4` |

`small_object_foot_over` is the primary trigger. It raises the swing trajectory, but rolling one-step execution and one RTI update do not recover a grounded stance at the phase transition. Link/clearance terms increase the height error but are not sufficient to trigger the full zero-support failure alone.

## Root/Foot Time Series

During a `32`-cycle full-input hold:

- root z stays exactly `0.32m`; root z rise is `0`.
- zero command root x settles at about `0.347m`.
- scheduled stance feet remain about `0.07-0.29m` above their queried support surfaces.
- grounded support count is zero on every cycle.

The visible floating is therefore a root held at its kinematic height while all feet are retracted, not an explicit upward root trajectory.

## Conclusion And Follow-Up

The previous crossing acceptance does not cover stop-on-object support viability: it rewards collision-free stance locations and successful stance-swing-stance crossings, but never requires recovery when command progress ends while the robot is still inside the small-object influence region.

The fix must remain continuous and must not introduce a hard zero-command or semantic gate. It must preserve strict cross `254/254`, foot/calf/thigh/base collision frames `0`, stance-on-small `0`, flat stance grounding, command tracking, and dynamic batch contracts.
