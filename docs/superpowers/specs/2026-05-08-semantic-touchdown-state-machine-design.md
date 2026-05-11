# Semantic Touchdown State-Machine Redesign For Together Planner

## Metadata

- **Date**: 2026-05-08
- **Topic**: redesign together planner touchdown selection as a pure-GPU state machine over approach / cross / bypass behavior, with four-leg consistency and anchor-to-touchdown path validity
- **Status**: Draft for review
- **Primary scope**: `Go2Pvcnn/extension/batched_together_planner/`

## 1. Problem Statement

The current `together` planner has already been improved with legal-support filtering, fixed `K=3` candidate generation, explicit small/large touchdown semantics, and continuous body/thigh/calf collision checks. However, the user's latest viewer inspection shows a deeper issue: the remaining failures are now primarily front-end candidate-quality failures rather than only back-end trajectory filtering failures.

Observed failure direction from discussion and image review:

- the planner can detect small obstacles but still often fails to produce a stable crossing action
- some touchdown markers still appear too close to or on top of small obstacles, suggesting that touchdown generation itself is still the first weak link
- even when a touchdown is nominally legal, the robot can produce an awkward whole-body posture because enlarged touchdown freedom makes four-leg consistency more fragile
- thigh penetration is still possible, meaning later collision checks are not sufficient to rescue bad front-end candidates if the candidate set is already poor
- the planner still behaves too much like "pick one touchdown point and let later rollout/collision terms rescue it" instead of "generate a coherent small action segment that already respects approach, crossing, and four-leg consistency"

The user also clarified a more precise small-obstacle behavior target:

- a small obstacle is **not** something the robot must always cross
- if local terrain is too high or the resulting posture/path is poor, the robot may bypass it
- if the obstacle is still far away, the robot should be allowed to move closer first and only then cross
- therefore small-obstacle behavior is not a single touchdown preference but a staged decision over `approach`, `cross`, or `bypass`

The key problem shift is this:

- the current planner still treats candidate quality too much as a scalar score over a few endpoints
- the redesign must instead treat each candidate as a **front-end action segment** with an explicit state, touchdown margin, four-leg consistency, and anchor-to-touchdown path validity

## 2. Goals

- redesign the touchdown front-end around an explicit pure-GPU state machine rather than only static touchdown scoring
- keep the state machine active in all scenes, including scenes with no small obstacles; no parallel legacy "no-small" front-end path should exist
- allow small-obstacle handling to choose among:
  - `approach`
  - `cross`
  - `bypass`
- preserve the existing rule that touchdown/support may never land on small or large obstacle surfaces
- add explicit front-end quality constraints for:
  - touchdown distance margin from small-obstacle boundaries
  - four-leg pair consistency and whole-body posture quality
  - anchor-to-touchdown path clearance for foot and leg segments
- ensure that bad candidates can be invalidated at the touchdown/candidate stage rather than only being weakly discouraged by later rollout costs
- preserve pure GPU, fixed-shape, no-CPU, no-`for` planner-hot-path implementation constraints
- keep later trajectory/collision validation as the second layer of safety, but not the first place where obviously bad touchdown strategies are discovered
- add tests that explicitly prove state selection, candidate action quality, margin control, four-leg consistency, and final rerun authority rules

## 3. Non-Goals

- replacing the full planner with a Python control-flow state machine
- introducing a global path planner or world-scale memory system
- removing the existing `K=3` candidate-axis contract
- requiring small obstacles to always be crossed
- requiring large obstacles to use the same state progression as small obstacles
- implementing non-height-representable obstacle collision coverage beyond the existing merged-height design contract

## 4. User Requirements Captured From Discussion

The following points were explicitly confirmed during the discussion and are hard requirements for this design:

1. the current main issue is more likely in touchdown/candidate generation than in later trajectory filtering, because if only a few bad candidates exist, later filtering still has to choose one of them
2. touchdown markers appearing on top of small obstacles should be treated as evidence that touchdown generation itself remains faulty or too weak, not only as later swing/contact artifacts
3. small-obstacle behavior should not be a single static preference between front-side and beyond-obstacle terrain; it should be staged as "move closer first, then cross when appropriate, or bypass if crossing is poor"
4. a small obstacle must not always force crossing; if local terrain is too high or the posture/path is poor, bypass remains valid
5. the same front-end framework must still apply when no small obstacle is present; in that case it should naturally stay in a normal `cruise` / non-obstacle mode, not branch into a separate planner path
6. each candidate should be treated as an action segment, not only a touchdown endpoint
7. candidate quality must include:
   - touchdown legality
   - touchdown distance margin from small-obstacle boundaries
   - four-leg consistency
   - whole-body posture quality
   - anchor-to-touchdown foot clearance
   - anchor-to-touchdown leg clearance
8. touching or nearly touching a small obstacle is itself a front-end problem; tests must explicitly detect touchdown points that are too close to obstacle boundaries, not just points that are directly on obstacle surfaces
9. four-leg consistency is critical because larger touchdown freedom can create individually legal but globally awkward solutions
10. all of the above must remain pure GPU and fixed shape:
   - no `for`
   - no `numpy`
   - no CPU sync or host branching
11. test authority rule:
   - earlier focused passes only prove the earlier code state
   - later overlapping edits invalidate those earlier passes as final evidence
   - final acceptance must rerun the affected test union on the final code state

## 5. Primary Use Cases

### 5.1 Cruise on terrain without nearby small obstacles

No small obstacle is active in the relevant corridor. The state machine remains in a normal cruise mode, but candidate generation and four-leg/path-quality checks still use the same unified front-end framework.

### 5.2 Small obstacle detected but still too far to cross

The planner enters an approach mode. It may continue placing touchdown on legal terrain before the obstacle, but it must not allow touchdown to sit too close to the obstacle boundary.

### 5.3 Small obstacle detected and crossing window reached

The planner may switch into a crossing mode. It should generate candidates that truly represent a crossing action, not merely an endpoint score, and should reject candidates whose anchor-to-touchdown path or pair consistency is poor.

### 5.4 Small obstacle present but local terrain/posture quality makes crossing poor

The planner should be allowed to bypass rather than force a crossing, even though the obstacle is small.

### 5.5 Large obstacle directly ahead

Large-obstacle logic should continue to favor bypass/refusal, not a small-style crossing path.

### 5.6 Candidate endpoints legal but whole-body posture strange

The front-end should detect that the four-leg and root posture quality are poor before later rollout filtering has to rescue the candidate.

## 6. Workflow Overview

The redesigned front-end becomes a stateful candidate generator:

```text
ray hits + semantic map + current foothold anchors + command
-> terrain / support / obstacle-corridor summaries
-> front-end state classification
-> candidate action-segment generation
-> touchdown legality + obstacle-boundary margin checks
-> four-leg consistency / posture checks
-> anchor-to-touchdown path clearance checks
-> candidate invalidation or scoring
-> later rollout + continuous collision validation
-> final candidate selection
```

The crucial shift is from:

`candidate = touchdown point`

to:

`candidate = state-tagged mini action segment`

This means later rollout/collision filtering is no longer the first line of defense against obviously bad candidates.

## 7. Trigger And Session Contract

This redesign stays inside the `together` planner front-end and preserves the repository's fixed-shape GPU runtime contract.

The active runtime contract remains:

- planner calls stay full-batch
- candidate-axis size remains static
- no CPU fallback is introduced
- the front-end state machine must be represented by tensorized masks or equivalent fixed-shape state encoding, not by per-env Python loops or host-side branches

## 8. Input Classification And Analysis Modes

The redesign still consumes the same primary runtime inputs:

- terrain height surface
- optional semantic surface aligned to the height surface
- robot root pose and body orientation
- current foot positions / anchors
- command vector
- fixed gait/contact schedule

But the interpretation changes:

- `touchdown` front-end now reasons over **state + candidate action segment**
- later rollout/collision remains a second-stage validator

Input usage split:

- semantic + height:
  - state classification
  - touchdown legality
  - obstacle-boundary margin
  - small-vs-large state transitions
- height + kinematic summary:
  - anchor-to-touchdown path clearance
  - four-leg posture validation
  - later continuous rollout collision validation

## 9. Design-Generation Contract

### 9.1 Approaches Considered

#### Approach A: keep static touchdown scoring and add more penalties

- continue treating candidates mostly as touchdown endpoints
- add penalties for being near small obstacles, inconsistent posture, or bad path clearance

Pros:

- smallest implementation diff

Cons:

- still allows the front-end to produce poor candidates and hope later scores rescue them
- does not make "move closer first, then cross, or bypass" explicit

#### Approach B: action-segment candidates without explicit state machine

- treat each candidate as an action segment
- add margin, path, and posture checks
- but keep behavior transitions implicit in scores

Pros:

- stronger than endpoint scoring
- less invasive than a full state-machine interpretation

Cons:

- still leaves approach/cross/bypass semantics partially implicit
- harder to explain and test the exact reason a candidate was chosen

#### Approach C: explicit touchdown front-end state machine with action-segment candidates

- always run a unified front-end state framework
- represent `cruise / approach / ready_to_cross / front_cross / rear_follow / bypass / clear`
- generate candidates as action segments whose interpretation depends on state
- reject bad candidates early using state-aware endpoint/path/posture tests

Pros:

- best match for the user's clarified behavior goals
- naturally supports "small may cross or bypass" and "move closer first"
- makes tests and diagnostics more explainable

Cons:

- largest redesign effort
- requires more careful state/transition testing

### 9.2 Recommended Design

Use **Approach C**.

This is now justified because the user has explicitly clarified that the real problem is not only endpoint legality but the absence of a structured crossing behavior. The front-end must therefore become state-aware rather than only more heavily penalized.

## 10. Primary-Agent And Subagent Responsibilities

This design fixes the implementation boundaries that later execution must preserve.

### 10.1 Terrain-query responsibilities

`terrain.py` should provide the fixed-shape query surfaces that the state machine consumes:

- legal support candidates
- obstacle presence in forward corridors
- obstacle boundary / margin summaries
- legal terrain before, near, or beyond obstacle regions in a bounded support window

It still must not own the higher-level policy decision itself.

### 10.2 State-machine responsibilities

`parameterization.py` should own state classification and candidate action-segment generation.

Required state set:

- `cruise`
- `approach`
- `ready_to_cross`
- `front_cross`
- `rear_follow`
- `bypass`
- `clear`

These states must remain active in all scenes. In no-small scenes, the system remains in `cruise` instead of switching to a separate legacy path.

Required state semantics:

- `cruise`
  - no small obstacle is active in the relevant corridor
  - normal terrain-aware motion remains active
- `approach`
  - small obstacle is active but still outside the crossing window
  - front-side legal terrain remains admissible
- `ready_to_cross`
  - small obstacle is close enough that crossing candidates become preferred candidates, but crossing is not yet committed
- `front_cross`
  - front-leg pair is actively executing a crossing segment
  - the candidate must prove front-pair consistency and anchor-to-touchdown path clearance
- `rear_follow`
  - rear-leg pair is actively following the already-started crossing
  - the candidate must prove rear-pair follow consistency and whole-body posture stability
- `bypass`
  - crossing is disfavored or invalid due to terrain, boundary margin, posture, or path-clearance quality
  - the planner may route around the obstacle instead of forcing a cross
- `clear`
  - the crossing or bypass transition is complete and the front-end can safely return to normal cruise semantics

### 10.3 Candidate responsibilities

Each candidate should contain at least:

- state tag
- touchdown targets
- current anchor references
- anchor-to-touchdown path summary
- pair-consistency summary
- whole-body posture summary

The candidate axis remains fixed-size, but candidate meaning becomes richer than a single touchdown endpoint.

### 10.4 Small-obstacle responsibilities

For `small`:

- touchdown/support may never land on the obstacle surface
- touchdown that is too close to the small-obstacle boundary should be penalized or invalidated
- the planner may stay on front-side legal terrain during `approach`
- the planner may cross when in a crossing-ready/crossing state and the candidate is good enough
- the planner may bypass when local terrain, path clearance, or posture quality make crossing poor

So `small` behavior is tri-modal:

- `approach`
- `cross`
- `bypass`

### 10.5 Large-obstacle responsibilities

`large` should continue to prefer bypass/refusal rather than small-style crossing.

### 10.6 Collision and path responsibilities

The candidate generator must directly evaluate:

- touchdown legality
- touchdown boundary margin to `small`
- anchor-to-touchdown foot clearance
- anchor-to-touchdown leg clearance
- four-leg pair consistency
- whole-body posture quality

Later trajectory/collision validation still exists, but it is the second defense layer, not the first detector of obviously bad front-end behavior.

## 11. Todo-First Planning Contract

When converted to todo work, the implementation should decompose into slices that preserve this front-end hierarchy:

1. state classification masks and corridor summaries
2. candidate action-segment representation
3. touchdown-to-small-boundary margin checks
4. four-leg consistency and posture scoring
5. anchor-to-touchdown path clearance checks
6. state-aware invalidation and selection rules
7. deterministic state/metric/traceability tests

This design intentionally does not create a standalone implementation plan document.

Test authority rule:

- a focused test only proves the code state that existed when that test ran
- once later overlapping code is edited, earlier focused passes lose final-acceptance authority
- in this design, `overlapping behavior` includes any later edit touching:
  - terrain / semantic queries
  - support legality or touchdown generation
  - state classification
  - candidate generation / selection
  - anchor-to-touchdown path generation or clearance
  - four-leg consistency / posture scoring
  - collision or infeasible-mask logic
  - shared diagnostics / result fields / fixtures consumed by metric-bearing tests
- `affected test union` means the closure of:
  - direct unit tests for the edited surface
  - deterministic fixtures consuming the same outputs, metrics, or state transitions
  - downstream integration tests consuming the same diagnostics or shared fixture helpers
- if that closure cannot be bounded confidently, default escalation is to rerun all redesign metric-bearing tests for this design scope
- `final code state` means one traceable candidate ref or one traceable working-tree snapshot, and final acceptance may cite only rerun evidence from that one state
- earlier focused passes may remain in debug logs but must be marked `superseded / non-authoritative` in final acceptance once later overlapping edits exist

## 12. Testing And Acceptance Indicators

### 12.1 Test Layers

The redesigned front-end requires at least five test layers.

#### A. State classification tests

Verify:

- no-small scenes remain in `cruise`
- small-far scenes enter `approach`
- crossing-window scenes enter `ready_to_cross`
- committed crossing scenes enter `front_cross`
- rear-leg follow scenes enter `rear_follow`
- poor small-cross scenes can enter `bypass`
- post-cross scenes return to `clear`
- large-block scenes prefer `bypass` / refusal

#### B. Touchdown endpoint tests

Verify:

- touchdown never lands on small/large surfaces
- touchdown margin to small boundaries is above threshold in valid candidates
- near-boundary but off-surface touchdowns can still be classified as bad candidates via penalty or invalidation depending on threshold regime
- front-side legal terrain remains admissible where appropriate
- beyond-small legal terrain can win where appropriate

#### C. Anchor-to-touchdown path tests

Verify:

- foot path from anchor to touchdown clears relevant obstacle geometry
- thigh/calf path from anchor to touchdown clears relevant obstacle geometry
- candidate path collisions can invalidate otherwise endpoint-legal candidates

#### D. Four-leg consistency / posture tests

Verify:

- front-leg pair consistency
- rear-leg follow consistency
- support-polygon / body-posture quality remains inside allowed range
- enlarged touchdown freedom does not produce individually legal but globally awkward candidates

#### E. Later rollout/collision validation tests

Verify:

- later rollout still rejects remaining whole-horizon body/leg collisions
- front-end and later validation do not contradict each other on key deterministic fixtures

### 12.2 Required Test Metrics

Tests should assert explicit metrics, not only selected route labels or final feasibility.

Test authority and rerun rule:

- explicit metric assertions are only authoritative for the final code state under test
- if later edits touch overlapping behavior, the impacted metric-bearing tests must be rerun together on the final code state
- the required rerun scope is the affected test union defined in section 11
- if that union cannot be bounded confidently, the default escalation is to rerun all redesign metric-bearing fixtures/tests rather than preserve a narrower uncertain subset
- final acceptance for a leaf or the whole redesign requires the affected test union to pass together, not a collection of older partial passes from earlier intermediate code states
- final acceptance evidence must be recorded as one final verification record listing the exact tests, fixtures, and metrics rerun against the final code state

Required metrics include:

- `state_mode`
- `candidate_count`
- `touchdown_semantic_valid_ratio`
- `small_surface_touchdown_count`
- `large_surface_touchdown_count`
- `touchdown_small_margin`
- `small_cross_preference_outcome`
- `large_forward_refusal_ratio`
- `front_pair_consistency`
- `rear_pair_follow_consistency`
- `body_posture_score`
- `anchor_to_touchdown_foot_clearance`
- `anchor_to_touchdown_leg_clearance`
- `candidate_path_collision_flag`
- `body_min_clearance`
- `leg_min_clearance`
- `collision_penalty_breakdown`
- `support_xy_z_consistency`
- `forward_progress_metric`
- `small_strategy_outcome`
- `candidate_action_segment_diagnostics_present`

### 12.3 Deterministic Fixture Set

Minimum required fixtures:

- `F1_cruise_no_small`
  - no small obstacle in corridor
  - verifies `cruise`
- `F1b_cruise_no_small_uneven_terrain`
  - no small obstacle but uneven terrain
  - verifies non-center candidate can still win in `cruise`
- `F2_approach_small_far`
  - small ahead but still too far to cross
  - verifies `approach`, non-contact touchdown, and positive margin
- `F3_ready_to_cross_small_near`
  - small ahead and near enough to cross
  - verifies crossing-preferred state activation
- `F4_cross_small_beyond_better`
  - beyond-small legal terrain is best
  - verifies crossing outcome
- `F5_approach_small_front_better`
  - front-side legal terrain is still better
  - verifies approach outcome without forced crossing
- `F6_bypass_small_high_terrain`
  - small exists but terrain/posture/path quality makes crossing poor
  - verifies bypass is allowed for small
- `F7_bypass_large_blocking`
  - large directly blocks forward progression
  - verifies large bypass/refusal behavior
- `F8_front_cross_state`
  - explicit front-leg crossing case
  - verifies `front_cross` state entry and candidate-stage segment validity
- `F9_rear_follow_state`
  - explicit rear-leg follow case after a valid front crossing
  - verifies `rear_follow` state entry and consistency checks
- `F10_clear_state_after_cross`
  - verifies return to `clear` after a completed crossing sequence
- `F11_body_collision_only`
  - endpoint can look legal but whole-body/path behavior is unsafe
- `F12_leg_collision_only`
  - endpoint can look legal but leg path is unsafe
- `F13_mild_path_clearance_penalty_but_feasible`
  - path gets close enough to penalize but not invalidate
- `F14_support_xy_z_mismatch_regression`
  - preserves prior touchdown/support consistency regression coverage
- `F15_near_boundary_penalize_or_invalidate`
  - touchdown remains off the obstacle surface but is too close to the small boundary
  - verifies thresholded penalty vs invalidation behavior

### 12.4 Acceptance Indicators

The redesign is acceptable only if all of the following are true:

1. the unified state framework is active in both small-obstacle and no-small scenes
2. small obstacles can lead to `approach`, `cross`, or `bypass` depending on geometry and posture quality
3. large obstacles still favor bypass/refusal rather than crossing
4. touchdown/support never lands on small/large surfaces
5. touchdown that is too close to small boundaries is penalized or invalidated
6. four-leg pair consistency and whole-body posture quality are explicitly checked at candidate stage
7. anchor-to-touchdown foot and leg path clearance are explicitly checked at candidate stage
8. later rollout/collision validation still protects the full horizon
9. `front_cross`, `rear_follow`, and `clear` each have explicit deterministic fixture coverage and are not only listed as state names
10. a near-boundary off-surface fixture proves that boundary proximity can still mark a candidate bad even when touchdown is not directly on the obstacle surface
11. candidate-stage diagnostics explicitly prove that a candidate is an action segment with state/path/consistency-or-posture evidence rather than only an endpoint score
12. all planner-hot-path logic remains pure GPU and fixed shape, and the final affected test union includes the static/guardrail checks that enforce no `for`, no `numpy`, and no CPU sync in hot-path production code
13. any overlapping later code edits have been followed by rerunning the affected test union on the final code state
14. earlier focused passes that predate later overlapping edits are treated as `superseded / non-authoritative` in final acceptance
15. the final verification record lists the exact rerun tests, fixtures, and metrics used as final evidence
16. test metrics explicitly demonstrate the above behavior rather than inferring it indirectly

## 13. Requirement Coverage Checklist

- [x] touchdown reads semantic map and height map
- [x] no-small scenes still use the same front-end framework
- [x] small does not always force crossing
- [x] small may bypass if terrain/posture/path quality is poor
- [x] candidate is treated as an action segment, not only a touchdown point
- [x] touchdown-too-close-to-small is explicitly addressed
- [x] four-leg consistency is elevated to a first-class candidate check
- [x] anchor-to-touchdown path clearance is explicitly addressed
- [x] large avoidance remains foothold-level and distinct from small-crossing logic
- [x] pure GPU / fixed-shape constraints preserved
- [x] test-authority / rerun rule made explicit

## 14. Open Questions

The main behavior direction is fixed. Remaining open items should be resolved during later todo breakdown or coding rather than by changing the design intent:

- exact tensor encoding for state tags / masks
- exact touchdown-small-boundary margin thresholds
- exact pair-consistency and posture-quality formulas
- exact path-clearance sample templates for candidate-stage anchor-to-touchdown checks
- how much of the state/candidate diagnostics should be surfaced on `TogetherPlannerResult` versus only asserted inside focused tests
