# T300d MPC 4096 Train Command End-to-End Success (`max_iterations=1`)

- timestamp: 2026-05-11 14:28 CST
- todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- result: pass

## Purpose

Run the exact user command in `env_isaacsim` and keep fixing until it completes:

```bash
python Go2Pvcnn/scripts/train.py \
  --headless \
  --device cuda:2 \
  --num_envs 4096 \
  --max_iterations 1 \
  --experiment teacher_elevation_trajectory \
  --planner-backend mpc
```

## Stage

MPC runtime integration + 4096-scale training command stability (`train.py` -> env.step planner/reward path -> PPO update path).

## Triggered Failures And Root Causes

1. `RuntimeError: element 0 of tensors does not require grad`  
   Root cause: `rsl_rl` rollout executes `env.step()` under `torch.inference_mode()`, but MPC optimizer calls `backward()` inside that path.

2. `RuntimeError: Inplace update to inference tensor outside InferenceMode`  
   Root cause: optimization variables were created as inference tensors when planner is invoked from inference-mode rollout.

3. `torch.OutOfMemoryError` during PPO critic CNN update (`max_pool2d`)  
   Root cause: 4096-env trajectory rollout default trainer settings (`num_steps_per_env=40`, `num_mini_batches=4`) exceeded 24GB update-time memory budget.

## Changes

- [../../Go2Pvcnn/extension/batch_mpc_planner/optimizer.py](../../Go2Pvcnn/extension/batch_mpc_planner/optimizer.py)
  - wrapped MPC optimization in `torch.inference_mode(False)` + `torch.enable_grad()`.
- [../../Go2Pvcnn/extension/batch_mpc_planner/variables.py](../../Go2Pvcnn/extension/batch_mpc_planner/variables.py)
  - create optimization variables under `torch.inference_mode(False)`.
  - force residual/logit tensors to normal grad-enabled leaf tensors via `detach().clone().requires_grad_(True)`.
- [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - added regression test: planner works under external `torch.inference_mode()` with `optimize_steps > 0`.
- [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)
  - added `teacher_elevation_trajectory` 4096-env runtime mem-guard:
    - clamp `num_steps_per_env` to `24` when default is larger.
    - increase `num_mini_batches` to keep per-minibatch samples bounded.

## Verification

- `python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q` -> `12 passed`
- `python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/optimizer.py Go2Pvcnn/extension/batch_mpc_planner/variables.py Go2Pvcnn/scripts/train.py Go2Pvcnn/tests/test_batch_mpc_backend.py` -> pass
- exact user command in `env_isaacsim` -> `EXIT_CODE:0`

Runtime artifacts from successful run:

- log dir: `logs/rsl_rl/teacher_elevation_trajectory/2026-05-11_14-26-48`
- checkpoint produced: `model_0.pt`
- saved trainer cfg confirms mem-guarded values:
  - `num_steps_per_env: 24`
  - `algorithm.num_mini_batches: 8`

## Conclusion

The exact 4096-env MPC training command now completes end-to-end for one iteration on `cuda:2` in `env_isaacsim`.

## Follow-Up

- Re-profile longer runs (`max_iterations` > 1) for throughput/VRAM headroom.
- If needed, expose mem-guard thresholds as explicit CLI/task config knobs for fine-tuning.

## Git Refs

- Baseline Ref: `979b2b5`
- Candidate Ref: working tree with inference-mode autograd fix + 4096 mem-guard + runtime success verification
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/optimizer.py](../../Go2Pvcnn/extension/batch_mpc_planner/optimizer.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/variables.py](../../Go2Pvcnn/extension/batch_mpc_planner/variables.py)
  - [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
