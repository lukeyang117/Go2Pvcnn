# Go2 Walking Fix Design

**Date:** 2026-03-12  
**Scope:** feet_air_time, reset, velocity curriculum, time_outs, bad_orientation

---

## 1. feet_air_time

### Current State
- **air_time_variance_penalty** (weight -1.0): Penalizes variance in `last_air_time` and `last_contact_time` across feet. Encourages symmetric gait.
- **feet_air_time_positive_reward** exists in `rewards.py` but is **not** used in teacher_semantic rewards.
- Contact sensor for feet uses `track_air_time=True`; other contact sensors use `track_air_time=False`.
- `feet_air_time_positive_reward` logic: rewards feet with `current_air_time > threshold` when not in contact, only when command norm > 0.1.

### Issues
1. No direct reward for desirable swing phase (lifting feet when moving).
2. `air_time_variance_penalty` alone may discourage exploration of different gaits.
3. Potential logic bug in `feet_air_time_positive_reward`: `first_contact` and `~in_contact` semantics may be reversed (rewarding air time at first contact vs. in-air duration).

### Design Changes
- **Add** `feet_air_time_positive_reward` to teacher rewards with a small weight (e.g., 0.5–1.0).
- Use threshold tuned for Go2 stride (e.g., 0.1–0.15 s for trot).
- Verify logic: reward feet that were in the air for at least `threshold` seconds at the moment of ground contact (encourages proper swing phase).
- Optionally reduce `air_time_variance_penalty` weight if positive air-time reward dominates and causes instability.
- Ensure `contact_forces` sensor for `.*_foot` has `track_air_time=True` (already the case).

---

## 2. reset

### Current State
- **reset_base**: `reset_root_state_uniform` with zero pose/velocity range (spawn at origin).
- **reset_robot_joints**: `reset_joints_by_scale` with `position_range=(0.9, 1.1)`, `velocity_range=(0.0, 0.0)`.
- **reset_* (YCB objects)**: Each dynamic object reset via `reset_root_state_uniform` with pose/velocity ranges.
- **push_robot**: Interval 10–15 s, `push_by_setting_velocity` with x,y ∈ (-0.5, 0.5).

### Issues
1. Zero spawn pose may put robot on difficult terrain (stairs, slopes).
2. Joint position range (0.9–1.1) can produce slightly crouched or extended poses; may affect first-step stability.
3. No explicit base height reset; relies on terrain height at origin.

### Design Changes
- **reset_base**: Add small random pose perturbation (e.g., roll/pitch ∈ (-0.05, 0.05), yaw ∈ (-0.1, 0.1)) to avoid always starting perfectly flat.
- **reset_robot_joints**: Consider tightening to `position_range=(0.98, 1.02)` for more consistent standing; or keep (0.9, 1.1) if diversity helps.
- **Terrain-aware reset**: Optionally restrict spawn to `flat` or `random_rough` sub-terrains in early curriculum.
- **Base height**: Ensure robot is reset at appropriate height above terrain (Isaac Lab typically handles this via `env_origins`).
- Keep dynamic object and push event configuration; tune push magnitude if robot falls too often.

---

## 3. velocity curriculum

### Current State
- **CommandsCfg**: `UniformVelocityCommandCfg` with:
  - `lin_vel_x=(0.5, 1.0)`
  - `lin_vel_y=(0.0, 0.0)`
  - `ang_vel_z=(0.0, 0.0)`
- **CurriculumCfg**: Empty (`pass`); no curriculum implementation.

### Issues
1. Fixed velocity range from start; no easy-to-hard progression.
2. No angular velocity during training; turning may be under-practiced.
3. High initial linear velocity (0.5–1.0 m/s) may cause early failures.

### Design Changes
- **Add velocity curriculum**: Implement `CurriculumCfg` term that scales velocity range based on episode success or progress.
  - Phase 1: `lin_vel_x=(0.2, 0.5)`, `ang_vel_z=(0.0, 0.0)`.
  - Phase 2: `lin_vel_x=(0.3, 0.7)`, `ang_vel_z=(-0.3, 0.3)`.
  - Phase 3: `lin_vel_x=(0.5, 1.0)`, `ang_vel_z=(-0.5, 0.5)`.
- Use Isaac Lab `CurriculumManager` or custom event that adjusts command distribution.
- Trigger phase upgrade when mean episode length or reward exceeds threshold (e.g., > 80% of max steps without termination).
- Optional: Add `lin_vel_y` in later phases for lateral walking.

---

## 4. time_outs

### Current State
- `episode_length_s = 20.0`.
- `time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)` — marks episode end as truncation, not failure.
- RSL-RL wrapper: `extras["time_outs"] = truncated` for value bootstrap.

### Issues
1. 20 s may be too long for early training; many episodes end in failure before timeout.
2. Truncation vs. termination handling must be correct in RSL-RL (GAE, value target).

### Design Changes
- **Shorter initial episodes**: Consider `episode_length_s = 10.0` or curriculum that starts at 10 s and increases to 20 s as policy improves.
- **Curriculum**: Add event/curriculum that increases `episode_length_s` when success rate is high.
- **Verify RSL-RL**: Ensure `time_out=True` is propagated to `extras["time_outs"]` and used correctly in `on_policy_runner` for last-value bootstrap.
- Keep `time_out=True` so timeouts are not treated as failures for logging/evaluation.

---

## 5. bad_orientation

### Current State
- `bad_orientation` termination is **commented out** in `TerminationsCfg`.
- Function in `terminations.py`: terminates when `|projected_gravity.z| < cos(limit_angle)` (robot tilted beyond `limit_angle` radians from upright).

### Issues
1. Without `bad_orientation`, robot can continue training when severely tilted, wasting samples and potentially learning bad behaviors.
2. `limit_angle=0.5` rad (~29°) may be too strict or too loose; needs tuning.

### Design Changes
- **Re-enable** `bad_orientation` termination.
- Start with `limit_angle=0.6`–0.7 rad (~34–40°) to allow some recovery; tighten to 0.5 if policy is stable.
- Consider adding a **soft penalty** (e.g., `flat_orientation_l2` with stronger weight) in addition to hard termination for smoother learning signal.
- Ensure termination is triggered for both roll and pitch; current implementation uses `projected_gravity.z` which covers both.

---

## 6. Implementation Order

1. **bad_orientation** — Re-enable with conservative limit; quick win to avoid training on invalid poses.
2. **velocity curriculum** — Implement curriculum manager; biggest impact on early learning.
3. **feet_air_time** — Add positive reward; tune threshold and weight.
4. **time_outs** — Shorten initial episode length; add curriculum if needed.
5. **reset** — Refine spawn and joint reset once baseline is stable.

---

## 7. Verification

- Log termination reasons (time_out, base_contact, bad_orientation) to TensorBoard.
- Monitor `mean_episode_length` and `success_rate` (episodes ending in time_out vs. early termination).
- Compare reward components before/after each change.
- Run short training seeds (e.g., 1000 iterations) to validate no regressions.
