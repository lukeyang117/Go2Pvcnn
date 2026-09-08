#!/usr/bin/env bash
set -euo pipefail

cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn

ISAAC_ENV="/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim"
export PATH="${ISAAC_ENV}/bin:${PATH}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="${ISAAC_ENV}/lib/python3.10/site-packages/torch/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${ISAAC_ENV}/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cuda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

args=(
  Go2Pvcnn/scripts/train_cross_large_complex_ame.py
  --num_envs "${NUM_ENVS:-1024}"
  --max_iterations "${MAX_ITERATIONS:-10000}"
  --device "${DEVICE:-cuda:0}"
  --headless
)

if [[ -n "${CHECKPOINT:-}" ]]; then
  args+=(--resume --checkpoint "${CHECKPOINT}")
  if [[ "${KEEP_STD:-0}" == "1" ]]; then
    args+=(--keep_std)
  fi
fi

exec "${ISAAC_ENV}/bin/python" "${args[@]}"
