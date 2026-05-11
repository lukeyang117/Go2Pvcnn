# Semantic Touchdown, Bypass, And Collision Redesign For Together Planner

## Metadata

- **Date**: 2026-05-07
- **Topic**: redesign together planner touchdown semantics, fixed `K=3` candidate policy, and pure-GPU collision coverage for semantic terrain
- **Status**: Draft for review
- **Primary scope**: `Go2Pvcnn/extension/batched_together_planner/`

## 1. Problem Statement

The current `together` planner already supports semantic maps and fixed `3` route candidates when semantic mode is active, but it still misses several behaviors the user now requires:

- small obstacles may still function too much like generic cost bumps instead of explicit foothold constraints
- touchdown selection can still mismatch support `xy` and support `z`, causing footholds that visually penetrate geometry or appear to land on the wrong surface
- swing motion still behaves too much like a fixed-shape arc with only limited terrain-aware adjustment instead of using terrain height along the full airborne segment
- the candidate space is too weak even when `semantic_candidate_count = 3`, because the current candidates differ mostly by route offset rather than by true touchdown / foothold intent
- large-obstacle avoidance is still too trajectory-centric; the user wants avoidance to start from foothold policy, including the ability to reject forward progression and route around large obstacles instead of merely adding a few later path variants
- collision checking does not yet cover the body and leg segments strongly enough to prevent cases where feet clear but thighs, calves, or the body contact terrain or obstacles
- the redesign must preserve the `together` backend's pure-GPU fixed-shape execution model and may not reintroduce CPU sync, host-branching on tensor masks, or dynamic sub-batches

The user also clarified two important behavioral rules:

- a small obstacle may encourage a step beyond it, but the planner must still be allowed to choose legal terrain before the obstacle if that terrain is safer or scores better overall; the only hard rule is that touchdown/support may not land on the obstacle surface itself
- even without semantic obstacles, the planner must always evaluate fixed `K=3` candidates and may not assume the straight/center candidate is inherently best; terrain quality still decides

## 2. Goals

- make touchdown/support selection semantic-aware so that only `terrain` cells are legal touchdown/support surfaces
- prevent touchdown/support from ever returning `small` or `large` surfaces as legal footholds
- preserve `small` as a crossable-but-not-step-on semantic class
- preserve `large` as a non-crossable class that should cause foothold-level bypass or forward refusal instead of straight-line stepping
- make fixed `K=3` candidate evaluation a hard planner invariant in all cases, regardless of whether semantic obstacles are present
- redesign the `K=3` candidates so they differ at the foothold/touchdown-policy level rather than only through later root-route offsets
- keep touchdown semantics separate from trajectory semantics:
  - touchdown reads `height + semantic`
  - trajectory may stay `height-only` in this redesign
- strengthen swing and trajectory collision reasoning so the planner catches body, thigh, and calf contact against both terrain and raised obstacles through the merged height surface
- keep the full redesign pure GPU, fixed shape, and compatible with current together guardrails
- define tests with explicit metrics, not only pass/fail assertions

## 3. Non-Goals

- changing PPO observation tensors
- training-policy redesign in this phase
- introducing a global path planner or persistent world-scale obstacle map
- exact mesh-level collision or CPU-side analytical collision routines
- modifying the legacy `extension/batched_planner` backend
- adding dynamic obstacle motion or scene-semantic updates beyond the current scanner contract

## 4. User Requirements Captured From Discussion

The following points were explicitly confirmed during the discussion and are hard requirements for this design:

1. touchdown planning must read semantic map information in addition to height information
2. trajectory planning may remain height-focused in this redesign; it does not need to become fully semantic-aware end-to-end
3. `support_at()` should answer where the robot is allowed to support itself; it should not own the higher-level preference logic for stepping beyond or around obstacles
4. touchdown/support may only land on legal `terrain`; `small` and `large` obstacle surfaces are never legal touchdown/support surfaces
5. `small` should encourage crossing behavior when that improves the plan, but legal terrain before the obstacle must remain admissible if it is safer or better overall
6. `large` avoidance must be designed from the foothold/touchdown level, not only by adding later trajectory costs or route offsets
7. fixed `K=3` candidates must always exist, even when no large obstacles are present
8. the planner may not assume the center candidate is always best in obstacle-free scenes; terrain quality and safety still decide
9. four-leg coordination matters:
   - the robot may cross in phases, with front and rear legs moving at different times
   - but all touchdown events must remain legal throughout the crossing process
   - body and leg segments may not collide with terrain or obstacles
10. collision coverage must include:
   - one body hull
   - four thighs
   - four calves
11. body and leg collision checks must consider both terrain and obstacles; in practice the merged height surface may represent both
12. continuous collision checks should use a mixed rule:
   - soft penalty for close but acceptable approaches
   - hard infeasible rejection for sufficiently bad penetration
13. implementation must remain pure GPU and fixed shape:
   - no `.cpu()`, `.item()`, `.numpy()`, `.tolist()`
   - no host-side branching on tensor masks for planner logic
   - no dynamic sub-batches
   - no Python loops over envs, candidates, or legs in planner hot paths
14. tests must include explicit metrics and acceptance indicators, not only broad success claims

## 5. Primary Use Cases

### 5.1 No semantic obstacle, uneven terrain

The planner still evaluates `K=3` fixed candidates and may choose a non-center candidate if local terrain support, slope, or collision clearance makes it safer.

### 5.2 Low small obstacle in the command direction

The planner may prefer a foothold beyond the obstacle if that yields safer forward progress on legal terrain, but it may also choose legal terrain before the obstacle if that scores better. It must never place touchdown/support on the obstacle surface.

### 5.3 Large obstacle directly ahead

The planner should not continue forward stepping through the obstacle. The center-progress candidate should be strongly discouraged or refused, while left/right bypass candidates offer alternative foothold policies.

### 5.4 Narrow or blocked bypass terrain

If the obstacle blocks the forward path and both bypass directions are unsafe, the planner should refuse progression or fall back safely rather than crossing the obstacle illegally.

### 5.5 Feet-clear but body/leg collision scene

Even if touchdown legality looks acceptable, the final candidate must be rejected if the body hull, thighs, or calves collide with terrain or raised obstacles during the trajectory.

## 6. Workflow Overview

The redesign splits behavior into four layers:

1. semantic terrain and legal-support queries
2. touchdown / foothold candidate generation with obstacle-class semantics
3. height-aware rollout generation with fixed `K=3` candidate evaluation
4. continuous collision checking and candidate scoring / infeasible masking

High-level data flow:

```text
ray hits + semantic map
-> TogetherPlannerTerrain queries
-> semantic-valid support query
-> touchdown candidate generation
-> foothold-level small-cross / large-bypass policy
-> height-aware rollout generation
-> body/thigh/calf continuous clearance checks
-> candidate scoring + infeasible masking
-> final chosen result + diagnostics
```

The main responsibility shift from the current implementation is this:

- `small` and `large` semantics should influence touchdown/foothold policy first
- trajectory scoring validates whether that motion stays safe through the whole horizon
- `K=3` is always active and must encode three different foothold intentions rather than one foothold policy plus two later path offsets

## 7. Trigger And Session Contract

This redesign stays inside the existing `together` planner architecture and does not change user-facing command structure for train/play/viewer, except where later implementation may expose additional diagnostics. The semantic scanner contract remains the entry for obstacle meaning.

The training/runtime contract that remains fixed:

- planner execution remains full-batch and fixed shape
- the candidate axis remains statically sized
- no CPU fallback or dynamic candidate count is introduced
- viewer and training should continue to share the same terrain-query contract, even if viewer remains the first adoption target for some checks

## 8. Input Classification And Analysis Modes

This design is not an `inspire` skill redesign; it is a planner redesign inside the `together` semantic-terrain path. The primary runtime inputs are:

- terrain height surface
- optional semantic surface aligned to the height surface
- robot root pose and body orientation
- current foot positions
- command vector
- fixed schedule/contact timing

The redesign interprets those inputs in two semantically different ways:

- touchdown / support legality:
  - uses height + semantic
- trajectory collision / swing clearance:
  - uses merged height surface only in this phase

That boundary is explicit and intentional.

## 9. Design-Generation Contract

### 9.1 Approaches Considered

#### Approach A: minimal patching

- keep current candidate structure
- add semantic touchdown filtering
- lightly patch swing height
- extend body underside samples only

Pros:

- smallest code diff
- lowest short-term implementation risk

Cons:

- still too weak for the user's four-leg coordination and segment-collision requirements
- keeps large-obstacle avoidance too trajectory-centric
- does not make `K=3` meaningfully different at the foothold layer

#### Approach B: layered touchdown/foothold redesign with continuous collision validation

- separate semantic-valid support from touchdown preference logic
- make `K=3` always active and foothold-distinct
- move small/large semantics into touchdown/foothold generation
- add height-aware swing clearance and body/leg continuous collision validation
- keep the trajectory layer height-driven rather than fully semantic-aware

Pros:

- matches the user's clarified boundaries exactly
- solves touchdown legality, foothold bypass, and continuous collision coverage together
- remains compatible with pure GPU fixed-shape constraints

Cons:

- larger refactor surface than approach A
- requires a more systematic test matrix

#### Approach C: full semantic rollout redesign

- make touchdown, swing, body, and full rollout all consume semantic geometry directly
- potentially add richer world-aware obstacle semantics everywhere

Pros:

- highest theoretical expressiveness

Cons:

- exceeds the user's chosen boundary for this phase
- higher implementation and verification cost
- greater risk of mixing responsibilities and regressing training-path simplicity

### 9.2 Recommended Design

Use **Approach B**.

This is the smallest design that still satisfies all confirmed requirements. It preserves the user's chosen separation between touchdown semantics and trajectory semantics while making large-obstacle bypass and body/leg collision coverage strong enough to be meaningful.

## 10. Primary-Agent And Subagent Responsibilities

This design itself does not assign implementation workers yet, but it fixes responsibility boundaries the implementation must preserve:

- `terrain.py` owns query semantics and legal-support filtering
- `parameterization.py` owns touchdown/foothold policy, candidate differentiation, and rollout generation
- `planner.py` owns candidate-axis orchestration and selection
- `costs.py` owns continuous clearance penalties and infeasible masks
- tests own deterministic fixtures and metric assertions that prove the redesigned behavior

### 10.1 Terrain-query responsibilities

`TogetherPlannerTerrain` should provide:

- height queries
- semantic id queries
- legal-support queries that only admit terrain cells as support surfaces
- fixed-shape local obstacle summaries needed by touchdown policy, such as:
  - small-presence along a forward corridor
  - legal terrain before/after an obstacle in a bounded support window
  - large-obstacle occupancy in the support corridor

`support_at()` should not own higher-level strategic preferences such as "step beyond the small obstacle" or "bypass left". It only provides legal support candidates and quality signals.

### 10.2 Touchdown / foothold responsibilities

`parameterization.py` should generate candidate-specific touchdown/foothold policies.

Hard rules:

- touchdown/support may only land on `terrain`
- `small` and `large` surfaces are never valid touchdown/support surfaces
- `K=3` is always active

Behavior rules:

- for `small`:
  - legal terrain before the obstacle remains admissible
  - legal terrain beyond the obstacle may be preferred when it produces better overall progress, stability, and collision safety
  - the planner must never encode "must land behind the small obstacle" as a hard rule
- for `large`:
  - center-forward stepping through the obstacle must not remain a normal valid strategy
  - the planner should refuse or strongly suppress continued forward foothold placement through the large obstacle corridor
  - left and right bypass foothold policies should be generated explicitly

### 10.3 Candidate-axis responsibilities

The `K=3` candidate axis should represent real behavior intent, not merely post hoc route offsets.

Recommended semantics:

- candidate 0: center-progress foothold policy
- candidate 1: left-bypass foothold policy
- candidate 2: right-bypass foothold policy

These three candidates always exist.

Even in obstacle-free terrain, all three candidates should be scored under terrain support quality, slope, collision clearance, and progress objectives. The center candidate is not hard-coded as the best candidate.

### 10.4 Collision responsibilities

Collision validation should be split by layer:

- touchdown layer:
  - local static precheck around touchdown/support legality
- trajectory layer:
  - continuous clearance validation along the whole horizon

The collision objects are:

- one body hull
- four thighs
- four calves

Collision must cover contact against both terrain and obstacles. Because the planner already reasons over a merged height surface, the trajectory-layer check may remain height-driven while still covering raised obstacles whose relevant collision risk is represented as height-bearing occupied surface in the local raster window. This redesign does not claim side-overhang or non-height-representable obstacle geometry coverage beyond that merged-surface contract.

## 11. Todo-First Planning Contract

When later converted to todo work, the implementation should decompose into at least these slices:

1. terrain-query contract upgrade
2. touchdown/support legality and support-xy/z consistency fix
3. fixed-`K=3` candidate-axis redesign at foothold level
4. height-aware swing clearance redesign
5. body/thigh/calf continuous collision model
6. deterministic tests and metrics
7. manager/viewer/training-path compatibility and diagnostics updates

This design intentionally avoids writing the todo tree now. It only fixes the slices the later todo breakdown must preserve.

Test authority rule:

- a focused test only proves the code state that existed when that test ran
- once later overlapping code is edited, earlier focused passes lose final-acceptance authority
- in this design, `overlapping behavior` means any later edit that touches one or more of:
  - terrain / semantic query logic
  - support legality or touchdown generation
  - candidate generation, candidate selection, or candidate-state logic
  - swing clearance logic
  - body / thigh / calf collision or infeasible-mask logic
  - shared diagnostics / result fields consumed by tests
  - shared fixture or helper code consumed by metric-bearing tests
- `affected test union` means the closure of:
  - direct unit tests for the edited surface
  - deterministic fixture tests that consume the same output, metric, or state transition
  - downstream integration tests that consume the same shared diagnostics, result fields, or fixture helpers
- if the closure boundary cannot be identified confidently, the default escalation is to rerun all metric-bearing tests for the redesign scope rather than a narrower guess
- `final code state` means one traceable candidate ref or one traceable working-tree snapshot, and final acceptance may only cite rerun evidence produced from that single state
- any todo leaf that changes overlapping behavior must rerun the affected test union on the final code state before the leaf can be marked `pass`
- intermediate focused checks remain useful for debugging, but they do not replace final affected-test-union reruns
- earlier focused passes may remain in debug logs, but they must be treated as `superseded / non-authoritative` in the final acceptance summary once a later overlapping edit exists

## 12. Testing And Acceptance Indicators

### 12.1 Test Layers

The redesign requires four layers of tests.

#### A. Terrain / semantic query unit tests

Verify:

- `height_at(...)` and `semantic_at(...)` remain aligned
- `support_at(...)` never returns `small` or `large` as legal support
- support windows with no legal terrain surface report failure / infeasibility instead of silently using obstacle surfaces

#### B. Touchdown / foothold policy unit tests

Verify:

- `K=3` candidates exist in all scenes
- `small` scenes never produce touchdown on the obstacle surface
- legal terrain before a small obstacle remains admissible
- legal terrain beyond a small obstacle can win when it scores better overall
- large-obstacle center-forward candidates are refused or heavily suppressed
- left/right bypass candidates are generated as true foothold-policy alternatives

#### C. Trajectory / collision unit tests

Verify:

- swing height is terrain-aware across the airborne segment rather than only through a fixed global arc
- body-hull minimum clearance is measured over the whole horizon
- thigh/calf minimum clearance is measured over the whole horizon
- mild close approaches contribute penalty
- sufficiently bad penetration produces infeasible masks

#### D. Planner integration tests

Verify:

- `plan_segment(...)` returns the intended candidate diagnostics
- the chosen candidate is not only low-cost but also touchdown-legal and collision-safe
- support `xy` and support `z` are internally consistent

### 12.2 Required Test Metrics

Tests should assert explicit metrics, not only selected candidate labels.

Test authority and rerun rule:

- explicit metric assertions are only authoritative for the final code state under test
- if later edits touch overlapping behavior, the impacted metric-bearing tests must be rerun together on the final code state
- the required rerun scope is the affected test union defined in section 11, including downstream consumers of the same shared outputs, diagnostics, and fixture helpers
- if that union cannot be bounded confidently, the default escalation is to rerun all redesign metric-bearing fixtures/tests rather than preserve a narrower uncertain subset
- final acceptance for a leaf or the whole redesign requires the affected test union to pass together, not a collection of older partial passes from earlier intermediate code states
- final acceptance evidence must be recorded as one final verification record that lists the exact tests, fixtures, and metrics rerun against the final code state

Required metrics include:

- `candidate_count`
  - expected: always `3`
- `touchdown_semantic_valid_ratio`
  - expected: `1.0` in deterministic legal fixtures
- `small_surface_touchdown_count`
  - expected: `0`
- `large_surface_touchdown_count`
  - expected: `0`
- `small_cross_preference_outcome`
  - expected: in a fixture where beyond-small terrain is clearly better, the selected touchdown should land beyond the obstacle
  - expected: in a fixture where before-small terrain is safer/better, the selected touchdown may remain before the obstacle
- `large_forward_refusal_ratio`
  - expected: center-forward candidate should be suppressed or infeasible in direct-block fixtures
- `body_min_clearance`
  - expected: above threshold in safe fixtures; below hard threshold in body-collision fixtures
- `leg_min_clearance`
  - expected: above threshold in safe fixtures; below hard threshold in leg-collision fixtures
- `collision_penalty_breakdown`
  - expected: separate body and leg collision contributions remain visible
- `support_xy_z_consistency`
  - expected: support `xy` and support `z` come from the same support solution; regression fixtures should fail if they mismatch
- `forward_progress_metric`
  - expected: obstacle avoidance should not collapse all motion into standstill when safe progress exists

### 12.3 Deterministic Fixture Set

Minimum required fixtures:

- `F1_flat_no_obstacle`
  - no semantic obstacle
  - proves `K=3` always active and terrain scoring still decides
- `F2_small_forward_beyond_better`
  - small obstacle ahead, beyond-small legal terrain is clearly better
  - selected touchdown should prefer beyond-small legal terrain
- `F3_small_forward_front_better`
  - small obstacle ahead, front-side legal terrain scores better
  - planner may stay before the obstacle legally
- `F4_large_forward_blocking`
  - center-forward stepping blocked by large obstacle
  - center candidate suppressed or infeasible; bypass candidates remain available
- `F5_large_both_sides_blocked`
  - no safe bypass exists
  - planner should refuse progression or use safe fallback rather than cross illegally
- `F6_body_collision_only`
  - footholds legal but body clearance unsafe
  - candidate must become infeasible
- `F7_leg_collision_only`
  - footholds and body look acceptable but thigh/calf sweep collides
  - candidate must become infeasible
- `F8_support_xy_z_mismatch_regression`
  - catches old touchdown/support mismatch behavior
- `F9_mild_clearance_penalty_but_feasible`
  - body or leg comes close enough to trigger collision penalty increase without crossing the hard infeasible threshold

### 12.4 Acceptance Indicators

The redesign is acceptable only if all of the following are true:

1. fixed `K=3` candidate count holds in all together-planner scenes
2. touchdown/support legality never returns `small` or `large` as legal support
3. `small` is treated as non-step-on but potentially step-beyond, without forcing beyond-small touchdown in every case
4. `large` avoidance begins at foothold/touchdown policy rather than only through later route penalties
5. body, thigh, and calf collisions against terrain/obstacles are all checked over the whole horizon
6. clear continuous collisions trigger infeasible masks rather than merely larger costs
7. at least one deterministic fixture proves that mild body/leg close approach increases soft collision penalty while the candidate remains feasible
8. all planner-hot-path logic remains pure GPU and fixed shape
9. any overlapping later code edits have been followed by rerunning the affected test union on the final code state
10. earlier focused passes that predate later overlapping edits are treated as `superseded / non-authoritative` in the final acceptance summary
11. the final verification record lists the exact rerun tests, fixtures, and metrics used as final evidence
12. test metrics explicitly demonstrate the above behavior rather than inferring it indirectly

## 13. Requirement Coverage Checklist

- [x] touchdown reads semantic map and height map
- [x] trajectory layer may remain height-focused in this redesign
- [x] `support_at()` separated from higher-level preference logic
- [x] `small` and `large` forbidden as touchdown/support surfaces
- [x] `small` crossing is a preference, not a mandatory behind-obstacle rule
- [x] legal terrain before a small obstacle remains admissible
- [x] large-obstacle avoidance begins from touchdown/foothold policy
- [x] fixed `K=3` is always active
- [x] no assumption that center is always best, even without obstacles
- [x] body and leg segment collisions must be checked
- [x] collision covers terrain and obstacles
- [x] pure GPU / fixed-shape constraints preserved
- [x] tests and metrics explicitly included

## 14. Open Questions

The behavioral direction is fixed. Remaining open items are implementation-detail choices that should be resolved during todo breakdown or coding, not by changing the design intent:

- exact fixed sampling template sizes for body box and leg capsules
- exact soft/hard clearance thresholds for body vs leg collision
- exact names for new diagnostics fields in planner results and tests
- whether touchdown precheck diagnostics belong in `parameterization.py` only or should also surface through `TogetherPlannerResult.cost_breakdown`
