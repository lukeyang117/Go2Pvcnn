# Flat-Small Path Swing-Over Reward Design

## Background

The latest flat-small run `2026-06-30_22-43-39` is stable but does not learn the requested behavior.

Evidence:

- TensorBoard last100: `mean_episode_length=958.80`, `terrain=6.067`, `bad_orientation=0.1505`, `base_contact=0.00125`.
- `semantic_foot_over_clearance` is nonzero `373/5000`, so the near-footprint ramp is active.
- Controlled crossing on GPU 1:
  - `model_18200.pt`: opportunity `12/16`, root crossed `4/16`, foot_over `0/16`, contact `3/16`, success `0/12`.
  - `model_19699.pt`: opportunity `12/16`, root crossed `6/16`, foot_over `0/16`, contact `0/16`, success `0/12`.

Conclusion: the current reward can improve contact avoidance, but it does not force MPC-aligned foot swing-over.

## Goal

Change the existing `semantic_foot_over_clearance` reward so that path-small obstacle opportunities reward true swing-over behavior, not merely near-footprint clearance or contact avoidance.

The behavior target is:

```text
If the commanded root path crosses a low-small semantic obstacle,
at least one MPC-reference swing foot should pass over the obstacle footprint
with positive clearance.
```

## Non-Goals

- Do not add a new reward term.
- Do not add or change MPC planner losses.
- Do not change the policy observation or action shape.
- Do not remove `bad_orientation`.
- Do not solve this by only increasing `semantic_foot_over_clearance.weight`.

## Design

### 1. Path-Small Opportunity

Inside `semantic_foot_over_clearance_bonus_from_tensors`, sample the command-heading corridor in front of the root as the code already does. A path-small opportunity exists for an env when sampled cells satisfy all conditions:

- semantic id is in `small_semantic_ids`;
- obstacle height is above flat noise, e.g. `> 0.015m`;
- obstacle height is no larger than `low_small_max_height_m`;
- sample lies within `lookahead_m` and command corridor.

This opportunity mask becomes the gate for both positive foot-over reward and missed-over penalty.

### 2. Swing Foot Gate

When `reference_contact_state` is available, only reference swing feet can receive foot-over reward:

```text
swing = not reference_contact_state
```

If `reference_contact_state` is unavailable, preserve the existing fallback behavior and allow all feet. Training paths should normally have reference contact.

### 3. Strict Footprint Overpass

The main positive reward should require a swing foot to be over or very near a sampled path-small footprint:

- foot along/lateral error to path-small sample is within strict limits;
- foot vertical clearance over that sample is positive beyond `clearance_margin_m`;
- the foot belongs to the swing gate.

The strict score is based on:

```text
clearance = foot_z - obstacle_top_z
strict_score = relu(clearance - clearance_margin_m)
```

This strict reward is the primary signal. It should be able to fill the existing `bonus_clip` when the foot clearly passes over the obstacle.

### 4. Dense/Near Shaping Becomes Auxiliary

Keep dense approach and near-footprint ramp only as auxiliary shaping:

- dense approach remains capped by `dense_approach_bonus_fraction`;
- near-footprint ramp remains capped by `strict_near_bonus_fraction`;
- both are gated by path-small opportunity and swing feet;
- neither should by itself be enough to represent task success.

Recommended next config:

```text
semantic_foot_over_clearance.weight = 1.0
strict_over_cell_bonus_scale = 2.0
dense_approach_bonus_fraction = 0.05
strict_near_bonus_fraction = 0.25
```

This lowers the reward available to "almost over" behavior while keeping a usable learning gradient.

### 5. Missed Overpass Penalty

Add a negative component inside the same reward function. It is not a new reward term.

For each env:

```text
if path-small opportunity exists
and root has crossed the nearest path-small sample
and no swing foot achieved strict overpass:
    apply missed_over_penalty
```

Root crossing can be approximated in the current per-step function by:

```text
root_crossed_sample = candidate_along < root_crossed_margin_m
```

Since samples are expressed relative to the current root, a small or negative candidate-along means the root has reached or passed that sampled obstacle location. The first implementation should use a conservative margin, for example `0.02m`, to avoid penalizing too early.

Recommended params:

```text
missed_over_penalty = 0.15
root_crossed_margin_m = 0.02
```

The final return remains clamped to a safe range:

```text
bonus = positive_bonus - missed_penalty
bonus = clamp(bonus, min=-missed_over_penalty, max=bonus_clip)
```

The wrapper already multiplies by `bonus_scale`, so this is large enough to matter without overwhelming stability terms.

## Data Flow

Existing inputs:

- `terrain`
- `foot_pos_w`
- `root_pos_w`
- `root_quat_w`
- `command`
- `reference_contact_state`
- `reference_reward_mask`

Internal tensors:

1. Build heading and lateral axes from command/root yaw.
2. Sample path cells ahead of the root.
3. Detect low-small path samples.
4. Compute foot-to-sample along/lateral errors.
5. Compute swing-gated strict overpass score.
6. Compute auxiliary dense/near shaping.
7. Compute root-crossed-but-no-strict-overpass penalty.
8. Apply `reference_reward_mask`.

## Evaluation Diagnostics

Update controlled crossing diagnostics only if needed to expose the failure mode clearly:

- `root_crossed_but_no_foot_over_count`
- optional per-env booleans for root-crossed-without-foot-over

The acceptance metric remains controlled crossing success:

```text
opportunity > 0
root_crossed > 0
foot_over_count > 0
small_contact does not increase
bad_orientation does not increase
```

## Tests

Add focused RED tests first:

1. Strict swing-over reward:
   - swing foot over a low-small path sample gets positive reward.
   - same foot too low gets zero or penalty.

2. Dense/near shaping cap:
   - near but not over cannot exceed the configured auxiliary cap.

3. Stance gate:
   - stance foot over the obstacle does not receive foot-over reward when reference contact is present.

4. Missed overpass:
   - path-small sample behind/crossed root with no swing overpass returns a negative value.
   - if a swing foot overpasses, missed penalty is removed.

5. Mask gate:
   - `reference_reward_mask=False` returns zero even when opportunity exists.

6. Static config:
   - flat-small cfg keeps the same reward term name and exposes the new params.

Run focused tests:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Then run `py_compile` and a small real smoke before long training.

## Risks

- If missed-over penalty is too large, it may reintroduce bad orientation or freezing. Start with `0.15`.
- If strict footprint limits are too narrow, the reward becomes sparse again. Keep near shaping, but cap it lower.
- Current per-step reward does not store episode history. This design uses current root-relative path samples as a local approximation. If that is insufficient, the next step is to add lightweight diagnostics/cache, not to increase weight blindly.

## Acceptance

The change is accepted only if:

- focused tests pass;
- real smoke runs without CUDA OOM or reward-manager wiring errors;
- a short warm-start run remains stable;
- controlled crossing shows nonzero `foot_over_count` without increasing small contact or bad-orientation resets.
