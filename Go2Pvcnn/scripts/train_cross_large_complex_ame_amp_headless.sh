#!/usr/bin/env bash
set -euo pipefail

cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn

ISAAC_ENV="/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim"
DEFAULT_CHECKPOINT="/share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/logs/rsl_rl/cross_large_complex_ame/2026-09-08_15-40-23/bfb8b7f/model_19999.pt"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

export PATH="${ISAAC_ENV}/bin:${PATH}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export AME_AMP_EXPERIMENT_NAME="parallelism_tracking_cross_large_complex_ame_amp"
export LD_LIBRARY_PATH="${ISAAC_ENV}/lib/python3.10/site-packages/torch/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${ISAAC_ENV}/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cuda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

exec "${ISAAC_ENV}/bin/python" Go2Pvcnn/scripts/train_cross_large_complex_ame_amp.py \
  --num_envs "${NUM_ENVS:-1024}" \
  --max_iterations "${MAX_ITERATIONS:-3500}" \
  --checkpoint "$(realpath "${CHECKPOINT}")" \
  --resume \
  --keep_std \
  --device "${DEVICE:-cuda:0}" \
  --headless
