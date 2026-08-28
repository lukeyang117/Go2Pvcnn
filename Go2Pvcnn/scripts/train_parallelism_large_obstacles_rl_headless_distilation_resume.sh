#!/usr/bin/env bash
set -euo pipefail

cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn

export DISPLAY="${DISPLAY:-:1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ISAAC_ENV="/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim"
export LD_LIBRARY_PATH="${ISAAC_ENV}/lib/python3.10/site-packages/torch/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${ISAAC_ENV}/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

STUDENT_CHECKPOINT="/share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/logs/rsl_rl/cross_large_complex_ppo/2026-08-26_17-47-24/11d453a/model_19998.pt"

"${ISAAC_ENV}/bin/python" \
  Go2Pvcnn/scripts/train.py \
  --experiment parallelism_tracking_cross_large_complex_distillation \
  --num_envs 1024 \
  --headless \
  --max_iterations 6000 \
  --resume \
  --load_checkpoint "${STUDENT_CHECKPOINT}" \
  --keep_std \
  --teacher_checkpoint /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/logs/rsl_rl/parallelism_tracking_cross_large_complex/2026-08-18_20-30-59/6def073/model_4999.pt \
  --teacher-ratio-start 0.0 \
  --teacher-ratio-end 0.0 \
  --ppo-coef 1.0 \
  --teacher-coef 0.05
