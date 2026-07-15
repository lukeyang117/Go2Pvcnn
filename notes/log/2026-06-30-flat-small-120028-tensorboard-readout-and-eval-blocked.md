# 2026-06-30 Flat-Small 12:00 TensorBoard Readout And Eval Blocked

## Purpose

Inspect run `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-30_12-00-28`, which uses the strict-over-cell reward reshape, and attempt controlled-crossing evaluation.

## Stage

Training metrics / flat-small semantic avoidance / controlled crossing evaluation.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

Read TensorBoard scalars from:

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-30_12-00-28/events.out.tfevents.1782792172.enine.407209.0
```

Confirmed saved config:

- `semantic_foot_over_clearance.weight=0.6`
- `strict_over_cell_bonus_scale=2.0`
- `dense_approach_bonus_fraction=0.2`
- `entropy_coef=0.002`

Attempted controlled crossing for the TensorBoard candidate `model_17400.pt`:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 timeout 900s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode controlled_crossing \
  --headless \
  --device cuda:0 \
  --num-envs 16 \
  --num-rounds 1 \
  --max-steps 300 \
  --run-dir unused \
  --checkpoint /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-30_12-00-28/model_17400.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.6 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/flat_small_20260630_120028_model17400_controlled_crossing
```

The eval did not run because the current shell/IsaacSim session could not see CUDA:

- `nvidia-smi` failed to communicate with the NVIDIA driver.
- IsaacSim reported `No CUDA GPUs are available`.
- Output `metrics.jsonl` and `rounds.jsonl` are empty.

User requested retry on card 2:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 timeout 900s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode controlled_crossing \
  --headless \
  --device cuda:0 \
  --num-envs 16 \
  --num-rounds 1 \
  --max-steps 300 \
  --run-dir unused \
  --checkpoint /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-30_12-00-28/model_17400.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.6 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/flat_small_20260630_120028_model17400_controlled_crossing_card2
```

This also failed before simulation with `No CUDA GPUs are available`; `CUDA_VISIBLE_DEVICES=2 nvidia-smi` also failed to communicate with the NVIDIA driver.

Additional terminal/environment checks:

- `bash -lc` and `conda run -n env_isaacsim` both report `torch.cuda.is_available() == False` and `device_count == 0`.
- `/proc/driver/nvidia/version` exists and reports NVIDIA kernel module `560.35.03`.
- `/proc/driver/nvidia/gpus` lists four RTX 4090 devices with minors `0-3`.
- `/dev/nvidia*` device nodes are absent.
- `nvidia-modprobe` is not installed.
- Attempting `mknod /dev/nvidia0-3` and `/dev/nvidiactl` fails with `Operation not permitted`.
- Current user groups are `lhy nogroup`; not `video` or `render`.

Conclusion for the retry: the kernel driver information is present, but this execution session does not expose usable NVIDIA device nodes, so changing conda env or shell is insufficient.

The user provided evidence from a normal host terminal showing `/dev/nvidia0-3`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, and `/dev/nvidia-uvm-tools` exist and `CUDA_VISIBLE_DEVICES=2` works in `env_isaacsim`.

The remaining difference is the Codex tool sandbox itself:

- `ps` inside the tool session shows PID 1 is `bwrap`.
- The launch command includes `--unshare-user --unshare-pid --unshare-net --dev /dev`.
- The tool mount namespace differs from the normal terminal namespace.
- The sandbox `/dev` is a fresh device filesystem/tmpfs without NVIDIA nodes.
- `systemd-run --user`, `ssh localhost`, `machinectl`, and manual `mknod` are blocked or unavailable inside this sandbox.

Therefore this cannot be repaired from inside the current Codex tool session. The outer Codex launcher must expose the host `/dev/nvidia*` nodes or run without this device-isolating sandbox.

## TensorBoard Metrics

Run length:

- checkpoints: `model_14700.pt` through `model_17600.pt`
- scalar steps: `14700` through `17629`

Last-100:

- `Train/mean_episode_length`: `932.30`
- `Train/mean_reward`: `1.20684`
- `Curriculum/terrain_levels/mean_terrain_level`: `6.0283`
- `Policy/mean_noise_std`: `0.36208`
- `Episode_Termination/bad_orientation`: `0.23525`
- `Episode_Termination/base_contact`: `0.02025`
- `Episode_Reward/semantic_foot_over_clearance`: `0.00140183`
- `Episode_Reward/semantic_body_part_clearance`: `-0.338497`
- `Episode_Reward/reference_foot_pos`: `0.0476251`
- `Episode_Reward/reference_contact`: `0.00413965`

Last-20:

- `Train/mean_episode_length`: `953.617`
- `Train/mean_reward`: `1.98435`
- `mean_terrain_level`: `6.02129`
- `bad_orientation`: `0.1975`
- `base_contact`: `0.01375`
- `semantic_foot_over_clearance`: `0.00066814`

Foot-over scalar:

- nonzero `203/2930`
- max `0.105152`
- top stable spike near step `17448`: foot-over scalar `0.10099`, episode length `939.4`, bad_orientation `0.15`, base_contact `0`

Best stable 100-step foot-over window:

- `17437-17536`
- foot-over mean `0.002412`
- episode length `934.05`
- terrain `6.094`
- bad_orientation `0.243`
- base_contact `0.0145`
- reward `2.636`
- std `0.350`

Best episode-length / low-bad window:

- around `17050-17151`
- episode length about `938`
- terrain about `6.062`
- bad_orientation about `0.208`
- base_contact about `0.011`
- foot-over mean only `0.0006`

## Result

TensorBoard readout completed. Controlled-crossing behavior evaluation is blocked inside the Codex tool sandbox by CUDA/driver visibility, not by a policy result. The user reran the same card-2 command from a normal host terminal and produced valid evaluation output for `model_17400.pt`.

Valid host-terminal controlled-crossing result for `model_17400.pt`:

- opportunity `12/16`
- root crossed `8/16`
- foot-over `0/16`
- small contact `1/16`
- touchdown on small `0/16`
- overpass success `0/12`
- resets `1/16`, reason `bad_orientation`, stage `before_foot_over`
- best observed max clearance is still negative; examples include `-0.0144m`, `-0.0186m`, `-0.0360m`, `-0.0394m`, `-0.0873m`, `-0.1062m`

## Conclusion

The new strict-over-cell reward shape did not collapse training. Stability is acceptable compared with earlier bad-orientation collapse runs: terrain stays around level `6`, episode length recovers above `930`, and base-contact remains low.

However, the strict foot-over reward is still sparse and did not translate into true controlled-crossing foot-over for `model_17400.pt`. Compared with the previous `weight=1.0` run, the scalar is much lower (`last100 0.0014` vs `0.05425`), which is expected after capping dense shaping but also means the policy still is not learning true over-cell clearance.

Do not claim overpass improvement. If continuing training, do it only briefly and re-evaluate around `model_18000-18200`; do not continue blindly for many thousands of iterations unless `semantic_foot_over_clearance` becomes more frequent without `bad_orientation` or `base_contact` rising.

## Follow-Up

- Rerun controlled crossing for:
  - `model_17600.pt`: latest available checkpoint.
  - optionally `model_17000.pt` or `model_17100.pt`: best stability window.
- Since `model_17400.pt` still has `foot_over=0`, next likely change is to make the strict reward less sparse spatially/timing-wise, not to merely increase dense shaping.

## Git Refs

- Baseline Ref: `2c8f1fb`
- Candidate Ref: run artifact `2026-06-30_12-00-28`
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
