# Nonzero Speed Candidate And Hard-Reason Diagnostics Design

## 1. Problem Statement

The current `together` planner can select a `beta=0` candidate even when the user command is nonzero. In the viewer this appears as:

```text
cmd=(+0.40, +0.00, +0.00)
delta=(+0.00, +0.00, +0.00)
standstill=True
```

This is not a replan-loop failure. It is a candidate-set problem: the K=5 speed ladder includes a zero-speed candidate, and the selector may choose it when other candidates are expensive or infeasible.

There is a second related issue. If every candidate enters hard constraints, the current large barrier can make candidates nearly indistinguishable. The planner may still choose one minimum-cost candidate, but the output does not explain which hard constraint caused infeasibility.

## 2. Goals

- Remove `beta=0` from the nonzero-command candidate tables.
- Preserve true standstill only when the input command itself is zero.
- Keep K fixed at `5` for all modes.
- Keep planner hot paths GPU tensorized and fixed-shape.
- Add hard-constraint reason diagnostics for every candidate.
- When all candidates are infeasible, rank nonzero candidates by hard-constraint severity instead of relying only on a flat barrier.
- Surface the selected hard reason in viewer/test terminal output so failures are debuggable without screenshots.
- Validate with IsaacSim/IsaacLab headless tests using `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`.

## 3. Non-Goals

- Do not add a new planner implementation file.
- Do not add CPU, NumPy, `.cpu()`, `.item()`, `.tolist()`, `nonzero`, `argwhere`, `masked_select`, or dynamic host-side sub-batching in planner hot paths.
- Do not make visual screenshot inspection part of acceptance.
- Do not reintroduce `HOLD` or `REFUSE_OR_HOLD` as candidate modes.
- Do not solve global route planning in this change.

## 4. Speed Candidate Contract

For a command:

```text
u = [vx, vy, wz]
```

define:

```text
moving = max(abs(u)) > idle_command_eps
```

If `moving` is false, the existing hold/standstill path remains valid.

If `moving` is true, every generated candidate must satisfy:

```text
beta_k >= beta_min > 0
```

Candidate command remains:

```text
u_k = beta_k * u + route_offset_k
```

The proposed fixed tables are:

```text
CRUISE:
  beta  = [1.00, 0.80, 0.60, 0.40, 0.20]
  route = [CENTER, CENTER, CENTER, CENTER, CENTER]

APPROACH_SMALL:
  beta  = [0.80, 0.65, 0.50, 0.35, 0.20]
  route = [CENTER, CENTER, CENTER, CENTER, CENTER]

CROSS_SMALL:
  beta  = [0.60, 0.50, 0.40, 0.30, 0.20]
  route = [CENTER, CENTER, CENTER, CENTER, CENTER]

BYPASS_OBSTACLE:
  beta  = [0.60, 0.40, 0.60, 0.40, 0.20]
  route = [LEFT, LEFT, RIGHT, RIGHT, CENTER]
```

This means "slowing down" is still allowed, but "no velocity" is not a candidate response to a nonzero user command.

## 5. Hard-Reason Diagnostics

The planner should expose fixed-shape tensor diagnostics:

```text
candidate_hard_reason_mask: [B, K, R] bool
selected_hard_reason_mask:  [B, R] bool
candidate_hard_rank_cost:   [B, K] float
selected_hard_rank_cost:    [B] float
```

`R` is a fixed reason axis. Initial reason names:

```text
boundary_invalid
path_collision
body_hard_collision
leg_hard_collision
crossing_not_grounded
touchdown_on_small
front_foot_small_collision
rear_foot_small_collision
per_leg_foot_small_collision
base_small_penetration
touchdown_on_large
foot_large_collision
direction_violation
```

The hot path stores masks and numeric rank costs only. Human-readable strings are produced only in viewer/test/reporting code.

## 6. All-Hard Selection

Normal selection order remains:

```text
1. If any feasible candidate exists:
     select argmin(total) among feasible candidates.

2. Else if any safe_fallback candidate exists:
     select argmin(total) among safe_fallback candidates.

3. Else all candidates are hard/infeasible:
     select argmin(candidate_hard_rank_cost) among nonzero-speed candidates.
     Keep status = ALL_INFEASIBLE.
```

The hard-rank cost should preserve severity ordering:

```text
J_hard_rank =
  w_direction      * direction_violation
+ w_large          * large obstacle collision counts
+ w_base           * base/body penetration severity
+ w_leg            * leg collision counts / clearance violation
+ w_touchdown      * touchdown semantic violation counts
+ w_ground         * crossing grounded failure
+ w_boundary_path  * boundary/path invalidation
+ small numeric tie-breaker from original total
```

This avoids the current "everything got a huge barrier, now values are almost tied" behavior. It also makes failure diagnosis actionable:

```text
touchdown_on_small       -> touchdown/support selection problem
foot_small_collision     -> swing or anchor-to-touchdown path problem
base_small_penetration   -> root/body path problem
leg_hard_collision       -> thigh/calf clearance or collision body problem
crossing_not_grounded    -> touchdown height/grounding problem
direction_violation      -> command/route direction problem
```

## 7. Viewer And Terminal Output

When planner status is infeasible or any selected hard reason is true, viewer/runtime diagnostics must include:

```text
status=ALL_INFEASIBLE
selected_beta=...
selected_route=...
selected_hard_rank_cost=...
selected_hard_reasons=reason_a|reason_b
candidate_hard_rank=[... five values ...]
candidate_hard_reasons=[... five compact reason strings ...]
```

For successful normal plans, output can remain compact but must still allow tests to inspect the tensor fields.

## 8. Headless Acceptance Tests

All runtime acceptance commands must use:

```text
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python
```

and must be timeout-wrapped so IsaacSim does not occupy the GPU indefinitely.

### T1: Flat Terrain, No Semantic Objects

Use headless runtime output for forward, backward, lateral-left, lateral-right, yaw-left, and yaw-right commands.

Required metrics:

```text
standstill=False for nonzero commands
selected_beta > 0
mode=CRUISE
selected_route=CENTER for non-bypass cases
command_direction_violation=False
selected_hard_reason_mask all false
```

### T2: Flat Small-Obstacle Crossing

Use headless runtime output for forward, backward, lateral-left, and lateral-right commands with a small obstacle in the command corridor.

Required metrics:

```text
selected_beta > 0
CROSS_SMALL appears in the sequence, or APPROACH_SMALL transitions to CROSS_SMALL
cross_small_success_count > 0
touchdown_on_small_count = 0
per_leg_touchdown_on_small_count = (0, 0, 0, 0)
foot_small_collision_count = 0
per_leg_foot_small_collision_count = (0, 0, 0, 0)
base_small_penetration_count = 0
body_min_clearance >= 0
leg_min_clearance >= 0
selected_hard_reason_mask all false on accepted crossing plans
```

### T3: All-Hard Deterministic Fixture

Construct a deterministic planner fixture where every K candidate violates at least one hard constraint.

Required metrics:

```text
status=ALL_INFEASIBLE
selected_beta > 0
candidate_hard_reason_mask shape = [B, 5, R]
candidate_hard_rank_cost shape = [B, 5]
selected index equals argmin(candidate_hard_rank_cost)
selected_hard_reason_mask matches candidate_hard_reason_mask at selected index
diagnostics identify at least one concrete reason
```

### T4: Headless Infeasible Output Includes Reasons

If a headless runtime case reaches `ALL_INFEASIBLE`, terminal output must include:

```text
selected_hard_reasons=
candidate_hard_rank=
candidate_hard_reasons=
```

This test prevents future regressions where the planner says only "infeasible" without explaining why.

## 9. Implementation Scope

Modify existing files only:

```text
Go2Pvcnn/extension/batched_together_planner/config.py
Go2Pvcnn/extension/batched_together_planner/costs.py
Go2Pvcnn/extension/batched_together_planner/planner.py
Go2Pvcnn/extension/batched_together_planner/types.py
Go2Pvcnn/extension/viz/go2_foostep_planner.py
Go2Pvcnn/tests/test_batched_together_core.py
Go2Pvcnn/tests/test_batched_together_guardrails.py
Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py
Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py
```

Do not create new planner source files.

## 10. Design Self-Review

- No `beta=0` remains in nonzero-command candidate tables.
- Zero command can still hold via the existing command hold path.
- All-hard fallback is explicit and diagnostic, not hidden behind a flat barrier.
- Tests cover no-semantic motion, small-obstacle crossing, deterministic all-hard ranking, and headless reason output.
- GPU hot path remains tensor-only; string formatting is limited to viewer/test reporting.
