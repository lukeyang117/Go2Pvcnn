#!/usr/bin/env bash
set -euo pipefail

cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn

CHECKPOINT="${1:?Usage: $0 /path/to/policy_or_amp_checkpoint.pt}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi
CHECKPOINT="$(realpath "${CHECKPOINT}")"

export DISPLAY="${DISPLAY:-:1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ISAAC_ENV="/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim"
export LD_LIBRARY_PATH="${ISAAC_ENV}/lib/python3.10/site-packages/torch/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${ISAAC_ENV}/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cuda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

exec "${ISAAC_ENV}/bin/python" \
  Go2Pvcnn/scripts/train.py \
  --experiment parallelism_tracking_cross_large_complex_amp \
  --num_envs "${NUM_ENVS:-1024}" \
  --headless \
  --max_iterations "${MAX_ITERATIONS:-10000}" \
  --resume \
  --load_checkpoint "${CHECKPOINT}" \
  --keep_std
