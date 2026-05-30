# T302l Semantic Global Contact Smoke

- Purpose: Validate `SemanticGlobalContactSensor` with the real IsaacLab MPC semantic env at a small env count before running 1024-env acceptance.
- Stage: MPC RL semantic contact reward.
- Related todo: [T302l](../todo/T302l-mpc-rl-participation-and-reward-plan.md)
- Baseline Ref: `1f368bf`
- Candidate Ref: working tree after semantic global contact sensor implementation.
- Environment: `env_isaacsim` at `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`, `CUDA_VISIBLE_DEVICES=0`.
- Command:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py::test_mpc_semantic_global_contact_sensors_real_isaaclab_small -q
```

- Input conditions: real `TeacherElevationTrajectoryMpcSemanticEnvCfg`, `num_envs=4`, `semantic_contact_small` and `semantic_contact_large`.
- Result: PASS, exit code 0.
- Key checks:
  - `semantic_contact_small.data.force_matrix_w` shape matches `[4, 13, N_small_slot, 3]`.
  - `semantic_contact_large.data.force_matrix_w` shape matches `[4, 13, N_large_slot, 3]`.
  - `body_names` match `SEMANTIC_CONTACT_BODY_NAMES`.
  - PhysX view sensor/filter counts match expected env/body/object counts.
- Follow-up: Run 1024-env quantity alignment acceptance and record slot counts.
