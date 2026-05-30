# T302l Semantic Global Contact 1024 Quantity Alignment

- Purpose: Validate real 1024-env quantity alignment for the two global semantic contact sensors.
- Stage: MPC RL semantic contact reward.
- Related todo: [T302l](../todo/T302l-mpc-rl-participation-and-reward-plan.md)
- Baseline Ref: `9398545`
- Candidate Ref: working tree after adding 1024 acceptance test.
- Environment: `env_isaacsim` at `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`, `CUDA_VISIBLE_DEVICES=0`.
- Command:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py::test_mpc_semantic_global_contact_sensors_quantity_alignment_1024 -q
```

- Input conditions:
  - Real `TeacherElevationTrajectoryMpcSemanticEnvCfg`.
  - `num_envs=1024`.
  - Body list length: `13`.
  - Semantic course leaf objects are counted from `/World/semantic_course/{small,large}/row_*/col_*/slot_*`.
- Expected semantic slot counts from the 10x20 semantic course layout:
  - `N_small_slot = 640`.
  - `N_large_slot = 100`.
- Expected shapes:
  - `semantic_contact_small.data.force_matrix_w`: `[1024, 13, 640, 3]`.
  - `semantic_contact_large.data.force_matrix_w`: `[1024, 13, 100, 3]`.
- Result: PASS, exit code 0.
- Key checks:
  - `contact_physx_view.sensor_count == 1024 * 13` for both small and large.
  - `contact_physx_view.filter_count == N_small_slot` / `N_large_slot`.
  - `body_names == SEMANTIC_CONTACT_BODY_NAMES`.
  - No `expected 1024, found 7/5` failure occurred.
- Follow-up: Run focused unit tests, backend regression subset, performance probe, and one-iteration train smoke.
