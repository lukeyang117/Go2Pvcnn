#!/usr/bin/env bash
set -euo pipefail

cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn

export DISPLAY="${DISPLAY:-:1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ISAAC_ENV="/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim"
export LD_LIBRARY_PATH="${ISAAC_ENV}/lib/python3.10/site-packages/torch/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${ISAAC_ENV}/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

"${ISAAC_ENV}/bin/python" \
  Go2Pvcnn/scripts/train.py \
  --experiment parallelism_tracking_cross_large_complex_distillation \
  --num_envs 1024 \
  --headless \
  --max_iterations 2000 \
  --resume \
  --load_run 2026-08-17_18-15-07/82a2edc \
  --load_checkpoint model_800.pt \
  --keep_std \
  --teacher-ratio-start 0.0 \
  --teacher-ratio-end 0.0 \
  --ppo-coef 1.0 \
  --teacher-coef 0.01
