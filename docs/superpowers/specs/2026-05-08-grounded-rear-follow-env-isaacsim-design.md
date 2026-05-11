# Grounded Rear-Follow Crossing In Isaac Lab

## 1. Problem Statement

The current `together` planner already has:

- legal semantic support filtering
- fixed `K=3` candidates
- explicit `small` / `large` touchdown semantics
- a front-end state machine over `cruise / approach / ready_to_cross / front_cross / rear_follow / bypass / clear`
- candidate-stage boundary margin, pair/posture scoring, and anchor-to-touchdown path checks

However, the user's latest inspection changes the real problem framing.

The remaining failure is no longer best described as "the planner cannot see the small obstacle" or only as "generic small-obstacle state-machine quality is still weak". The sharper failure is:

- the motion often does not look like a true crossing sequence
- front legs may appear to cross first, but rear-leg touchdown targets can still remain airborne
- the current behavior still contains bypass behavior, but when crossing is chosen it is not yet a grounded `front_cross -> rear_follow` progression

The user explicitly does **not** want this next round to be driven by visualization acceptance. Instead:

- redesign should target the real Isaac Lab environment path
- runtime/test environment should be `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`
- acceptance should be environment behavior and numeric evidence, not viewer screenshots

The key diagnosis from the current code is:

- the current `rear_follow` state in `parameterization.py` is still driven mainly by corridor-relative geometry such as whether the obstacle is now "behind" the root
- it does not yet require that:
  - front-leg crossing has already produced grounded touchdown beyond the obstacle
  - rear-leg follow has its own grounded touchdown availability
  - rear touchdown is truly on legal terrain rather than still floating above the support surface

So the next design target is not another generic semantic refinement. It is a narrower, more physical contract:

- if crossing is selected, the planner must produce a **grounded phased crossing**
- front legs cross first and land on valid terrain
- rear legs follow and also land on valid terrain
- rear touchdown must not remain airborne

## 2. Goals

- redesign small-obstacle crossing around a **grounded phase contract** rather than only a semantic/corridor state label
- preserve the idea that `small` may still lead to:
  - `approach`
  - `cross`
  - `bypass`
- make `front_cross` require grounded front touchdown beyond the obstacle
- make `rear_follow` require grounded rear touchdown support rather than only obstacle-relative root position
- explicitly detect and penalize or invalidate rear touchdown that remains airborne above terrain/support
- require `clear` to become reachable only after both grounded front-cross and grounded rear-follow have completed
- keep touchdown/support off obstacle surfaces
- keep pure-GPU fixed-shape planner-hot-path constraints
- move acceptance away from viewer imagery and toward:
  - deterministic planner fixtures
  - `env_isaacsim` Isaac Lab runtime/integration tests
- ensure final acceptance is based on final-code-state rerun authority, not intermediate passes

## 3. Non-Goals

- replacing the entire `T114` state machine with a brand-new planner family
- removing `bypass` for small obstacles
- requiring every small obstacle to be crossed
- using visualization/manual screenshots as the primary acceptance source
- moving planner hot paths to CPU, NumPy, host-side branching, or Python loops
- introducing a full global route planner

## 4. User Requirements Captured From Discussion

1. the next design should not continue down viewer/manual verification as the main line
2. the primary runtime environment for this round is `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`
3. acceptance should be based on real Isaac Lab environment behavior and tests, not on visual screenshots
4. the current behavior does not look like a true crossing sequence
5. the intended crossing order is:
   - front legs cross first
   - rear legs cross later
6. rear-leg touchdown currently appears airborne in some cases, and this must become an explicit design target
7. bypass behavior may still exist for small obstacles when crossing is poor; small is not mandatory-cross
8. new testing must be redone under the new framing rather than inheriting the old viewer-centric acceptance story
9. pure GPU / no CPU / no NumPy / no `for` in planner hot paths remain hard constraints
10. final acceptance still requires whole affected test unions to pass on the final code state
11. a selected crossing outcome with airborne rear touchdown must fail rather than merely receive a soft penalty

## 5. Primary Use Cases

### 5.1 No small obstacle nearby

The front-end remains in normal `cruise`, and nothing in this redesign should force crossing behavior when the environment does not require it.

### 5.2 Small obstacle still far away

The planner stays in `approach`, may continue to move closer, and should not force a premature cross.

### 5.3 Small obstacle close enough and crossing is viable

The planner may enter crossing mode, but only if it can establish grounded front-leg touchdown beyond the obstacle and later grounded rear-leg follow touchdown.

### 5.4 Small obstacle close enough but crossing quality is poor

If terrain, posture, path clearance, or touchdown groundedness is poor, the planner may still choose `bypass`.

### 5.5 Crossing selected but rear touchdown would remain airborne

This must now be treated as an explicit front-end failure or invalid candidate, not as an acceptable `rear_follow` outcome.

## 6. Workflow Overview

The redesigned crossing logic becomes:

```text
height + semantic + anchors + command
-> corridor summaries
-> state classification
-> candidate action-segment generation
-> grounded front-cross contract check
-> grounded rear-follow contract check
-> touchdown legality + obstacle-boundary margin
-> pair/posture/path-quality checks
-> candidate invalidation or scoring
-> later rollout/collision validation
-> Isaac Lab runtime integration verification
```

The critical new insertion is:

- `grounded front-cross contract check`
- `grounded rear-follow contract check`

These are stronger than the current state labels alone.

## 7. Trigger And Session Contract

This design is a continuation of the `together` planner front-end line, but it opens a new narrower design front under `T100`.

It preserves:

- full-batch planner calls
- fixed candidate axis
- pure GPU tensor paths
- no host-side branching in the hot path

It changes:

- the meaning of successful `front_cross`
- the meaning of successful `rear_follow`
- the acceptance surface, which must now include real Isaac Lab runtime evidence in `env_isaacsim`

## 8. Input Classification And Analysis Modes

This redesign keeps the same base inputs:

- terrain height surface
- aligned semantic surface
- robot root pose
- foot anchors / current stance
- command vector
- gait/contact schedule

But it adds a stricter classification of crossing success.

### 8.1 Semantic and terrain remain responsible for:

- obstacle presence
- obstacle class (`small` / `large`)
- legal support regions
- obstacle-boundary margin

### 8.2 Candidate-stage kinematics remain responsible for:

- anchor-to-touchdown path quality
- pair consistency
- posture quality

### 8.3 New groundedness classification is responsible for:

- whether touchdown is physically close enough to its support surface to count as landed rather than floating
- whether front legs have established a valid crossing foothold before rear legs are allowed to "follow"
- whether rear touchdown is genuinely grounded on legal terrain

## 9. Design-Generation Contract

### 9.1 Approaches Considered

#### Approach A: keep current T114 states and only tune penalties

- keep `front_cross` / `rear_follow` state meanings mostly unchanged
- add more penalty on airborne or poor rear touchdown

Pros:

- smallest code diff

Cons:

- still treats grounded rear-follow as a score issue rather than a state contract
- likely to preserve ambiguous cases where `rear_follow` is nominally selected but not truly landed

#### Approach B: add grounded touchdown checks but keep acceptance mostly deterministic/offline

- strengthen `front_cross` / `rear_follow` definitions
- keep acceptance mostly inside planner deterministic tests

Pros:

- more targeted than A
- easier than adding new Isaac Lab runtime evidence

Cons:

- does not fully answer the user's request to move acceptance into the real Isaac Lab environment path

#### Approach C: grounded phase contract plus Isaac Lab runtime acceptance

- redefine `front_cross` and `rear_follow` using touchdown groundedness
- keep deterministic planner tests
- add `env_isaacsim` Isaac Lab runtime/integration tests that numerically prove the crossing sequence is grounded

Pros:

- best matches the user's latest clarified requirement
- directly targets the observed failure mode
- removes overdependence on viewer interpretation

Cons:

- broadest testing surface
- requires runtime-facing diagnostics and integration assertions

### 9.2 Recommended Design

Use **Approach C**.

The user has explicitly shifted acceptance away from viewer imagery and toward Isaac Lab environment behavior. The remaining bug is specifically about crossing phase groundedness, especially rear touchdown. So the design must be both:

- **phase-aware**
- **runtime-verifiable in `env_isaacsim`**

## 10. Primary-Agent And Subagent Responsibilities

### 10.1 Terrain-query responsibilities

`terrain.py` continues to own:

- legal support search
- small/large surface classification
- obstacle-boundary margin summaries

It must not itself decide the crossing phase, but it must provide the support and reference heights needed to determine touchdown groundedness.

### 10.2 State-machine responsibilities

`parameterization.py` continues to own state classification, but state semantics must tighten.

Required state set remains:

- `cruise`
- `approach`
- `ready_to_cross`
- `front_cross`
- `rear_follow`
- `bypass`
- `clear`

But `front_cross` and `rear_follow` now mean:

- `front_cross`
  - front-leg crossing touchdown exists beyond the obstacle on legal terrain
  - front touchdown is grounded within configured touchdown-to-support tolerance
  - front path clearance / pair consistency / posture checks remain acceptable
- `rear_follow`
  - front-cross grounding has already been established
  - rear-leg follow touchdown exists on legal terrain
  - rear touchdown is grounded within configured touchdown-to-support tolerance
  - rear follow consistency and posture remain acceptable
- `clear`
  - reachable only after grounded front-cross success and grounded rear-follow success are both established
  - direct `front_cross -> clear` is forbidden
  - direct `approach -> clear` is forbidden

Required transition contract:

- `approach -> ready_to_cross -> front_cross` is the normal crossing lead-in
- `front_cross -> rear_follow` is allowed only if grounded front-cross success is already true and a grounded rear touchdown candidate exists
- `front_cross -> bypass` is allowed when grounded rear follow is not achievable cleanly
- `rear_follow -> clear` is allowed only after grounded rear-follow success is true
- `rear_follow -> bypass` is allowed when rear grounded completion degrades and no valid grounded completion remains
- any selected crossing outcome with airborne rear touchdown is invalid

### 10.3 Candidate responsibilities

Candidates remain action segments, but now must carry explicit groundedness evidence:

- state tag
- touchdown targets
- support heights / support validity
- touchdown grounded gap or equivalent touchdown-to-support error
- pair summary
- posture summary
- anchor-to-touchdown path summary

Groundedness contract for this redesign:

- touchdown groundedness is measured against the selected support solution height, not only a terrain-reference height
- per-leg signed gap:
  - `touchdown_ground_gap_m = touchdown_z - selected_support_height`
- per-leg absolute grounded error:
  - `touchdown_ground_gap_abs_m = abs(touchdown_ground_gap_m)`
- grounded touchdown:
  - `touchdown_ground_gap_abs_m <= 0.02`
- airborne touchdown:
  - `touchdown_ground_gap_m > 0.02`
- penetrating touchdown:
  - `touchdown_ground_gap_m < -0.02`
- this redesign uses the same `0.02 m` threshold for front and rear legs
- any selected crossing outcome that requires a touchdown and violates this groundedness contract must fail acceptance

### 10.4 Small-obstacle responsibilities

For `small`:

- touchdown/support may never land on the obstacle surface
- `approach` remains valid when the obstacle is still far or crossing is not yet good
- `cross` is valid only when grounded front-cross and grounded rear-follow contracts can be satisfied
- `bypass` remains valid when crossing quality is poor

### 10.5 Large-obstacle responsibilities

`large` continues to favor bypass/refusal and does not need this grounded crossing contract.

### 10.6 Runtime-integration responsibilities

Runtime/integration tests must verify:

- the selected planner/reference behavior in real Isaac Lab runtime under `env_isaacsim`
- that crossing phases produce grounded touchdown behavior numerically
- that the failure mode "rear touchdown remains airborne" can be detected without relying on human visual interpretation

Runtime metric source contract:

- `TogetherPlannerResult` or a planner-owned test-facing diagnostics wrapper must expose:
  - selected support heights for touchdown legs
  - `front_touchdown_ground_gap`
  - `rear_touchdown_ground_gap`
  - `state_mode`
  - `small_strategy_outcome`
- reward-facing canonical cache does not need to change if these diagnostics are stably available through planner/runtime debug surfaces
- any host-side aggregation used only inside tests must happen after the planner call returns and must not change planner hot-path constraints

## 11. Todo-First Planning Contract

If approved for execution, the implementation should decompose into slices such as:

1. grounded touchdown metrics and state-contract tightening
2. front-cross grounded gating
3. rear-follow grounded gating
4. runtime/result diagnostics surfacing for groundedness
5. deterministic planner tests for grounded phase behavior
6. `env_isaacsim` Isaac Lab runtime/integration tests for grounded crossing
7. final rerun-authority closure on the final code state

No standalone implementation plan document should be created.

Final-code-state authority remains unchanged:

- overlapping later edits supersede earlier focused passes
- final acceptance must rerun the affected test union on one final code state

## 12. Testing And Acceptance Indicators

### 12.1 Test Layers

The redesign should use two required layers.

#### A. Deterministic planner/front-end tests

Still required for:

- state classification
- semantic legality
- boundary margin
- pair/posture/path checks
- grounded touchdown metrics
- `front_cross` / `rear_follow` / `clear` logic

#### B. `env_isaacsim` Isaac Lab runtime/integration tests

New required acceptance surface.

These tests should exercise the real environment/runtime path and numerically inspect planner/reference behavior rather than viewer images.

### 12.1.1 Three collision/placement surfaces that must be checked separately

Headless Isaac Lab acceptance must not collapse everything into a single "crossed or not" score. It must separately check:

- `touchdown surface`
  - whether touchdown lands on `small`
  - whether touchdown remains too close to `small`
  - whether touchdown is grounded
- `feet path surface`
  - whether the foot trajectory from anchor to touchdown collides with `small`
- `base path surface`
  - whether the base/body path penetrates or crosses through `small`

Crossing success requires all three surfaces to remain valid.

### 12.2 Required Test Metrics

Existing metrics remain relevant:

- `state_mode`
- `touchdown_semantic_valid_ratio`
- `touchdown_small_margin`
- `front_pair_consistency`
- `rear_pair_follow_consistency`
- `body_posture_score`
- `anchor_to_touchdown_foot_clearance`
- `anchor_to_touchdown_leg_clearance`
- `candidate_path_collision_flag`

Carry-forward final-union metrics from `T113` / `T114` remain mandatory:

- legal touchdown/support semantic validity
- obstacle-surface touchdown exclusion
- fixed `K=3`
- small-boundary margin
- pair/posture quality
- anchor-to-touchdown path clearance
- support `xy/z` consistency
- body/leg collision coverage
- guardrail/static no-CPU/no-NumPy/no-`for` checks

New required grounded-crossing metrics:

- `front_touchdown_ground_gap`
  - touchdown z minus selected support solution height for front legs
- `rear_touchdown_ground_gap`
  - touchdown z minus selected support solution height for rear legs
- `front_cross_grounded_ratio`
  - fraction of front-cross candidates or selected segments whose front touchdown is grounded within tolerance
- `rear_follow_grounded_ratio`
  - fraction of rear-follow candidates or selected segments whose rear touchdown is grounded within tolerance
- `rear_touchdown_airborne_count`
  - count of rear touchdowns whose ground gap exceeds the allowed tolerance
- `cross_phase_progression_valid`
  - verifies that `front_cross` grounded success precedes `rear_follow`
- `cross_outcome_grounded`
  - selected cross outcome is only accepted if both front and rear grounding contracts are satisfied
- `runtime_selected_state_mode`
  - selected runtime state observed at sampled replan boundaries
- `runtime_bypass_selected_when_rear_not_groundable`
  - proves runtime chooses bypass when grounded rear completion is unavailable
- `touchdown_on_small_count`
  - count of selected touchdown legs whose semantic support class is `small`
- `front_foot_small_collision_count`
  - count of front-foot path collisions against `small`
- `rear_foot_small_collision_count`
  - count of rear-foot path collisions against `small`
- `front_foot_min_clearance_to_small`
  - minimum front-foot path clearance to `small`
- `rear_foot_min_clearance_to_small`
  - minimum rear-foot path clearance to `small`
- `base_small_penetration_count`
  - count of base/body path penetrations through `small`
- `base_min_clearance_to_small`
  - minimum base/body path clearance to `small`
- `base_path_crosses_small_flag`
  - boolean indicating that the selected base path crosses or penetrates `small`

### 12.3 Deterministic Fixture Set

Existing `T114` fixtures should remain, but this redesign requires new grounded-phase fixtures:

- `G1_front_cross_grounded`
  - front legs cross and land beyond obstacle on legal terrain
- `G2_rear_follow_grounded`
  - rear follow lands on legal terrain after front-cross grounding is established
- `G3_rear_follow_airborne_invalid`
  - rear touchdown looks like a follow candidate geometrically but remains airborne; crossing acceptance must fail and the candidate must be invalidated
- `G4_cross_degrades_to_bypass_when_rear_follow_not_groundable`
  - small is still bypassed if grounded rear-follow cannot be achieved cleanly
- `G5_front_cross_then_rear_follow_then_clear`
  - explicit phase progression sequence

These complement, rather than replace, the earlier `F*` fixtures.

### 12.4 Isaac Lab Runtime Acceptance Cases

At least these `env_isaacsim` runtime/integration cases are required:

- `R1_small_cross_runtime_grounded`
  - setup:
    - deterministic targeted `small` obstacle placement using a fixed semantic-course anchor helper
    - post-obstacle terrain chosen to be flat enough for grounded crossing
  - command:
    - constant forward command `(0.30, 0.0, 0.0)`
  - sample window:
    - inspect consecutive selected replans for up to `3` planning cycles or until `clear`
  - expected:
    - sampled phase sequence contains `front_cross`, then `rear_follow`, then `clear`
    - `front_touchdown_ground_gap_abs_m <= 0.02`
    - `rear_touchdown_ground_gap_abs_m <= 0.02`
    - `rear_touchdown_airborne_count == 0`
    - `touchdown_on_small_count == 0`
    - `front_foot_small_collision_count == 0`
    - `rear_foot_small_collision_count == 0`
    - `base_small_penetration_count == 0`
    - `base_path_crosses_small_flag == 0`
    - `cross_phase_progression_valid == 1`
- `R2_small_bypass_runtime`
  - setup:
    - deterministic targeted `small` placement where grounded rear completion is intentionally poor or blocked
  - command:
    - constant forward command `(0.30, 0.0, 0.0)`
  - sample window:
    - inspect consecutive selected replans for up to `3` planning cycles
  - expected:
    - `runtime_bypass_selected_when_rear_not_groundable == 1`
    - selected runtime state never accepts a grounded crossing completion
    - no accepted crossing outcome may coexist with airborne rear touchdown
    - no accepted crossing outcome may coexist with touchdown on `small`
    - no accepted crossing outcome may coexist with foot/base path penetration through `small`
- `R3_rear_touchdown_airborne_regression`
  - setup:
    - deterministic runtime placement matching the reported airborne rear-touchdown failure class
  - command:
    - constant forward command `(0.30, 0.0, 0.0)`
  - sample window:
    - inspect the first selected `rear_follow` candidate/result boundary and continue until a safe alternative outcome is selected
  - expected:
    - if `rear_touchdown_ground_gap_m > 0.02`, the selected crossing outcome is invalid
    - runtime may remain pre-cross or choose `bypass`, but may not accept a grounded crossing outcome with airborne rear touchdown
    - even if endpoint semantics look acceptable, any concurrent `touchdown_on_small_count > 0`, `rear_foot_small_collision_count > 0`, or `base_small_penetration_count > 0` also invalidates the crossing outcome
- `R4_runtime_clear_requires_grounded_completion`
  - setup:
    - deterministic crossing-favorable placement similar to `R1`
  - expected:
    - `clear` is not allowed before both grounded front-cross and grounded rear-follow have been observed
    - direct `front_cross -> clear` is absent from the sampled runtime sequence
    - the final accepted sequence also preserves:
      - `touchdown_on_small_count == 0`
      - no foot-path collision with `small`
      - no base-path penetration through `small`

These tests may use real runtime fixtures similar to existing Isaac Lab diagnostics infrastructure, but they must not rely on viewer screenshots as acceptance proof.

### 12.5 Acceptance Indicators

The redesign is acceptable only if:

1. crossing no longer means only "front legs seem to pass the obstacle"
2. `front_cross` requires grounded front touchdown beyond the obstacle
3. `rear_follow` requires grounded rear touchdown on legal terrain
4. a rear touchdown that remains airborne is explicitly detectable and causes crossing acceptance to fail
5. phase progression `front_cross -> rear_follow -> clear` is numerically testable
6. small obstacles may still choose `bypass` when grounded crossing is poor
7. deterministic planner tests and `env_isaacsim` runtime tests both pass
8. no primary acceptance criterion depends on visual inspection
9. planner hot paths remain pure GPU / fixed shape / no CPU / no `for` / no NumPy
10. final acceptance is based on a final-code-state rerun of the affected test union
11. `clear` is only reachable after grounded front-cross and grounded rear-follow completion
12. final affected-union acceptance explicitly includes static/guardrail checks for no CPU sync, no NumPy, and no Python loops in planner hot paths
13. successful crossing requires all three surfaces to remain valid:
   - touchdown not on `small`
   - foot path not colliding with `small`
   - base path not penetrating `small`

### 12.6 Carry-Forward Final-Union Matrix

This redesign extends `T113` / `T114`; it does not replace them.

| Source | Carry-forward obligation | Final-union status |
| --- | --- | --- |
| `T113` | touchdown/support never land on obstacle surfaces | mandatory |
| `T113` | fixed `K=3` candidate axis | mandatory |
| `T113` | support `xy/z` consistency | mandatory |
| `T113` | body/leg collision and infeasible coverage | mandatory |
| `T114` | unified state framework with no-small compatibility | mandatory |
| `T114` | small-boundary margin | mandatory |
| `T114` | four-leg consistency and posture scoring | mandatory |
| `T114` | anchor-to-touchdown path clearance | mandatory |
| `T114` | final-code-state rerun authority | mandatory |
| `T114` | guardrail/static no-CPU/no-NumPy/no-`for` checks | mandatory |
| this design | grounded front-cross / rear-follow / clear contract | mandatory |
| this design | `env_isaacsim` runtime/integration grounded acceptance | mandatory |

## 13. Requirement Coverage Checklist

- [x] acceptance moved away from visualization and toward environment behavior
- [x] runtime environment explicitly set to `env_isaacsim`
- [x] remaining bug reframed as grounded rear-follow failure
- [x] intended crossing order captured as front legs first, rear legs second
- [x] small may still bypass
- [x] rear touchdown airborne failure made explicit
- [x] `clear` grounded completion contract made explicit
- [x] selected airborne crossing outcomes must fail rather than merely be penalized
- [x] groundedness reference and threshold are specified
- [x] runtime metric source is specified
- [x] deterministic tests preserved
- [x] Isaac Lab runtime/integration tests added as required acceptance
- [x] touchdown / feet path / base path are all included in runtime acceptance
- [x] T113/T114 carry-forward obligations are explicitly preserved
- [x] pure GPU / fixed-shape constraints preserved
- [x] final rerun-authority rule preserved

## 14. Open Questions

- which existing Isaac Lab runtime fixture should be extended first for `R1-R4`
- whether runtime diagnostics should live directly on `TogetherPlannerResult`, or be surfaced from it into a manager-owned test-facing snapshot helper
