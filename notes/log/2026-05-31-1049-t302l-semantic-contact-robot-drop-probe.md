# T302l Semantic Contact Robot Drop Probe

## Purpose

Diagnose whether slow convergence of `semantic_contact_collision` comes from semantic contact logic errors, multi-env/object misalignment, NaN/Inf values, or sparse physical contact signals.

## Stage

MPC semantic RL reward and real IsaacLab contact sensor path.

Key files:

- `Go2Pvcnn/tests/semantic_contact_robot_drop_probe.py`
- `Go2Pvcnn/tests/test_semantic_contact_robot_drop_probe.py`
- `Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/semantic_global_contact_sensor.py`
- `Go2Pvcnn/extension/mdp/semantic_contact_rewards.py`

## Related Todo

- [T302l](../todo/T302l-mpc-rl-participation-and-reward-plan.md)

## Baseline Ref

- `8ff7e72`

## Candidate Ref

- Worktree diagnostic additions on top of `8ff7e72`

## Procedure

The diagnostic uses the real `TeacherElevationTrajectoryMpcSemanticEnvCfg`, but avoids the full RL `env.step()` reset/reward-manager path. It directly writes robot root poses, runs `sim.step + scene.update`, then reads:

- `semantic_contact_small.data.force_matrix_w`
- `semantic_contact_large.data.force_matrix_w`
- `semantic_global_contact_collision_reward(...)`

Robot placement:

- envs 0 and 3: foot-aligned robot drop onto a small obstacle.
- envs 1 and 4: base/root robot drop onto a large obstacle.
- envs 2, 5, 6, 7: empty-terrain controls.

## Commands

Pure summary unit test:

```bash
pytest Go2Pvcnn/tests/test_semantic_contact_robot_drop_probe.py::test_summarize_semantic_contact_step_reports_active_envs -q
```

Result: PASS, `1 passed`.

Real IsaacLab smoke:

```bash
CUDA_VISIBLE_DEVICES=1 timeout 180s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_semantic_contact_robot_drop_probe.py::test_semantic_contact_robot_drop_probe_real_isaaclab_small -q
```

Result: PASS.

Manual JSONL probe:

```bash
CUDA_VISIBLE_DEVICES=1 timeout 180s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/semantic_contact_robot_drop_probe.py --num-envs 8 --steps 80 --drop-height 0.35 --output /tmp/semantic_contact_robot_drop_probe_card1.jsonl
```

Summary:

```json
{
  "empty_hit_any": false,
  "has_inf": false,
  "has_nan": false,
  "large_drop_hit_large": true,
  "large_drop_hit_small": false,
  "max_reward": -0.0,
  "min_reward": -1.0,
  "row_count": 640,
  "small_drop_hit_large": false,
  "small_drop_hit_small": true
}
```

Per-case metrics from the JSONL:

- `empty`: `active_steps=0`, `max_force=0.0`, `min_reward=-0.0`
- `large_drop`: `active_steps=150`, `max_force=2842.8406`, `min_reward=-1.0`
- `small_drop`: `active_steps=5`, `max_force=78.9663`, `min_reward=-1.0`

## Conclusion

The current semantic contact chain can detect both small and large semantic obstacle contacts with the real robot bodies. The probe found no NaN/Inf values and no cross-talk from obstacle envs into empty envs.

The important behavioral signal is sparsity: large obstacles generate contact on many frames, while small obstacles generate contact on only a few frames under the drop setup. This supports the hypothesis that training convergence can be slow because the semantic collision reward is very sparse, especially for low small obstacles.

## Follow-Up

If training still converges slowly, debug next with rollout statistics rather than changing the sensor first:

- contact hit rate per semantic type during real policy rollout
- active body distribution
- reward clipping saturation rate
- small-obstacle visibility/approach frequency by terrain row/col

Do not change the reward formula or add auxiliary losses until those rollout metrics are reviewed.
