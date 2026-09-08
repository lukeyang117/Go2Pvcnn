#!/usr/bin/env bash
set -euo pipefail

cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn
ISAAC_ENV="/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim"
export PATH="${ISAAC_ENV}/bin:${PATH}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "${ISAAC_ENV}/bin/python" Go2Pvcnn/scripts/train_cross_large_complex_semloco.py \
  --num_envs "${NUM_ENVS:-1024}" \
  --max_iterations "${MAX_ITERATIONS:-10000}" \
  --device "${DEVICE:-cuda:0}" --headless "$@"
