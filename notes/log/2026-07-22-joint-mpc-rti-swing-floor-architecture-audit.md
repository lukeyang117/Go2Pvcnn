# Joint MPC RTI Swing-Floor Architecture Audit

## Purpose

Determine why the four-row XY-only support candidate still publishes negative swing clearance, and compare the smallest architectures that can make published-x1 clearance feasible without another scalar sweep.

## Stage

- Planner: `Go2Pvcnn/extension/joint_mpc_rti`
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Input report: `/tmp/joint_mpc_viewer_support_xy_only.json`

## Read-Only Evidence

The current Terrain family is a soft objective, not a clearance constraint. It concatenates `31 x (4 foot + 4 knee + 12 calf + 12 thigh + 9 body + 4 touchdown) = 1395` rows and divides all rows by `sqrt(1395)`. A foot row is

```text
r = temperature * softplus((h_effective + 0.022 + terrain_foot_margin - foot_z) / temperature)
```

with `temperature=0.015m`, `terrain_foot_margin=0.027m`, and top-level Terrain weight `8000`.

At the worst viewer node, the full-QP foot center is `19.407mm` over flat height zero. The metric floor is `22mm`, while the Terrain soft target is `49mm`; its deficit is therefore `29.593mm`. Despite that large deficit, normalization leaves the local weighted foot-z Gauss-Newton curvature at only approximately

```text
(sqrt(8000) / sqrt(1395) * sigmoid(29.593/15))^2 ~= 4.4
```

The full QP remains `2.593mm` below the metric floor and all five line-search candidates pass every existing filter. Alpha one has the minimum seven-loss merit, so candidate selection is correct. A new filter alone would reject the best available direction rather than create a feasible one.

## Architecture Options

### A. Fixed-Rank Mixed Published KKT (Recommended)

Use six fixed x1 rows with new semantics:

```text
4 rows = continuing/current stance world-XY
2 rows = current swing world-Z floor
```

For each of the two x1 swing feet:

```text
b_z = max(h_effective + foot_contact_offset + clearance_buffer - nominal_foot_z, 0)
J_foot_z delta_z1 = b_z
```

If nominal is already above the floor, the row preserves its nominal z. If it is below, the full direction raises it to the floor. The fourth existing line-search filter becomes one published-kinematics filter: continuing stance XY must remain within `0.5mm`, and swing z must remain above the queried floor. This keeps four filter families, one QP, one RTI, five alphas, seven losses, and fixed-rank dense/scan Schur logic.

The only new tuning value is a visible `published_swing_clearance_buffer`, initially zero for exact parity with the acceptance metric. Viewer diagnostics should draw/report nominal, full, selected, and safe z deficits.

### B. Dynamic Swing-Z Inequality Active Set

Add true `J_z delta >= floor_error` inequalities to the active-set machinery. This is mathematically more general and does not freeze already-safe nominal z, but it expands the per-interval active-set contract, compile budget, seed feasibility, dense/scan parity surface, and failure modes. It conflicts with the requested simple, directly tunable architecture.

### C. Merit Or Filter Only

Add an exact penalty or clearance filter to line search without changing the QP direction. Reject this option: the measured full candidate is already unsafe, so selection logic cannot manufacture a safe state from the current five candidates.

## Conclusion

Option A is the smallest architecture that addresses the proved failure at its source. It is not the old six-row stance-XYZ design: old rows were `stance xyz + stance xyz`; proposed rows are `stance xy + stance xy + swing z + swing z`. Production implementation remains pending explicit user approval.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current Task 14 working tree
- Key Files: `losses/terrain.py`, `solver/linearization.py`, `solver/trajectory_qp.py`, `solver/trajectory_scan.py`, `solver/line_search.py`, `solver/sqp_rti.py`
