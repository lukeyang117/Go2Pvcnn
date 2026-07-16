# Joint MPC RTI Exact EDT Regression

- Purpose: verify the synchronous exact-field implementation does not regress planner behavior or the old `mpc` backend.
- Stage: joint planner, terrain query, factory/reward/viewer compatibility.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `643a172`; Candidate Ref: `4bad2e0`.
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/`, `Go2Pvcnn/tests/joint_mpc_rti/`.

## Commands And Results

- `CUDA_VISIBLE_DEVICES=2 .../env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/joint_mpc_rti -q`
  - Result: `93 passed`, including configured batch sizes `1/40/512/1024`, the public `num_envs` factory, ordered/full versus permuted-row cache behavior, and explicit rebuild-on-resize behavior.
- `.../env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q`
  - Result: `193 passed`.
- `.../env_isaacsim/bin/python -m py_compile $(find Go2Pvcnn/extension/joint_mpc_rti -name '*.py' -type f | sort)`
  - Result: exit `0`.
- `git diff --check`
  - Result: exit `0`.

## Conclusion

- Exact EDT, query-time gradients, repeated candidate-row field gathers, rolling x1 reference timing and synchronous version behavior pass.
- Default old `mpc` factory/reward/viewer behavior remains unchanged.
- Different environment counts are supported by constructing a new fixed-shape manager/cache/graph. In-place batch resize on an existing manager raises a clear rebuild error instead of failing later in the reference buffer.
- Upper-level API: `create_trajectory_manager(cfg, device=device, num_envs=N)`; `attach_trajectory_manager()` forwards `env.unwrapped.num_envs` automatically.
- `raw/mpx/` remains untracked reference material and is excluded from commits.
