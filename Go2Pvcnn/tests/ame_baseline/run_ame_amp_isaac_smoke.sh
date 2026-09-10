#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn"
ISAAC_PYTHON="/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python"
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ame-amp-smoke.XXXXXX")"
LOG_FILE="${SMOKE_ROOT}/smoke.log"
TRAIN_PID=""

cleanup() {
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
AME_AMP_OUTPUT_ROOT="${SMOKE_ROOT}/logs" NUM_ENVS=1024 MAX_ITERATIONS=4 \
  bash Go2Pvcnn/scripts/train_cross_large_complex_ame_amp_headless.sh 2>&1 | tee "${LOG_FILE}"

RUN_DIR="$(find "${SMOKE_ROOT}/logs" -type f -name 'model_3.pt' -print -quit | xargs -r dirname)"
if [[ -z "${RUN_DIR}" || ! -f "${RUN_DIR}/model_3.pt" ]]; then
  echo "Smoke test did not produce model_3.pt under ${SMOKE_ROOT}" >&2
  exit 1
fi

"${ISAAC_PYTHON}" - "${RUN_DIR}/model_3.pt" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu")
for key, value in checkpoint.items():
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            if torch.is_tensor(nested_value):
                assert torch.isfinite(nested_value).all(), f"non-finite tensor: {key}.{nested_key}"
    elif torch.is_tensor(value):
        assert torch.isfinite(value).all(), f"non-finite tensor: {key}"
print(f"[AME-AMP smoke] finite checkpoint: {path}")
PY

if pgrep -af 'train_cross_large_complex_ame_amp.py' | grep -v grep >/dev/null; then
  echo "AME-AMP smoke left a training process behind" >&2
  exit 1
fi
echo "[AME-AMP smoke] PASS run_dir=${RUN_DIR}"
