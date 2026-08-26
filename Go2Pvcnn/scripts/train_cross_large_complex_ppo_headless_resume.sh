#!/usr/bin/env bash
set -euo pipefail

cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn

export DISPLAY="${DISPLAY:-:1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
NUM_ENVS="${NUM_ENVS:-1024}"
LOAD_RUN="2026-08-24_20-00-26/4514f86"
LOAD_CHECKPOINT="model_9999.pt"

ISAAC_ENV="/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim"
export LD_LIBRARY_PATH="${ISAAC_ENV}/lib/python3.10/site-packages/torch/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${ISAAC_ENV}/lib/python3.10/site-packages:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cuda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

"${ISAAC_ENV}/bin/python" \
  Go2Pvcnn/scripts/train.py \
  --experiment cross_large_complex_ppo \
  --num_envs "${NUM_ENVS}" \
  --headless \
  --max_iterations "${MAX_ITERATIONS}" \
  --resume \
  --load_run "${LOAD_RUN}" \
  --load_checkpoint "${LOAD_CHECKPOINT}" \
  --keep_std
