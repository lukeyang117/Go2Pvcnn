# T302l Semantic Contact Sensor Smoke

- Purpose: Validate per-body filtered contact sensors in real IsaacLab.
- Env: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`
- Command: `CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py::test_mpc_semantic_contact_sensors_real_isaaclab -q`
- num_envs: 4
- Result: PASS
- Key Metrics:
  - sensor_count: 26
  - force_matrix_shape: `[4, 1, filter_count, 3]`
- Follow-up: none.
