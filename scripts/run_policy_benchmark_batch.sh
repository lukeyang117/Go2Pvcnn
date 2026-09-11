#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_policy_benchmark_batch.sh \
  --experiment-type {amp|distillation|ppo|teacher|ame|ame_amp|semloco} \
  --checkpoint FILE \
  [--suites complex_mixed,large_runway,small_runway] \
  [--suite-timeout SECONDS] [--gpu-release-timeout SECONDS] [benchmark args...]
EOF
}

experiment_type=""
checkpoint=""
suites_csv="complex_mixed,large_runway,small_runway"
suite_timeout=1800
gpu_release_timeout=60
forward_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-type)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      experiment_type="$2"
      shift 2
      ;;
    --checkpoint)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      checkpoint="$2"
      shift 2
      ;;
    --suites)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      suites_csv="$2"
      shift 2
      ;;
    --suite-timeout)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      suite_timeout="$2"
      shift 2
      ;;
    --gpu-release-timeout)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      gpu_release_timeout="$2"
      shift 2
      ;;
    --suite|--output-dir)
      echo "$1 is managed by the batch launcher; use --suites or omit --output-dir" >&2
      exit 2
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

[[ "$experiment_type" =~ ^(amp|distillation|ppo|teacher|ame|ame_amp|semloco)$ ]] || {
  echo "--experiment-type must be amp, distillation, ppo, teacher, ame, ame_amp, or semloco" >&2
  exit 2
}
[[ -n "$checkpoint" && -f "$checkpoint" ]] || {
  echo "--checkpoint must be an explicit checkpoint file" >&2
  exit 2
}
[[ "$suite_timeout" =~ ^[1-9][0-9]*$ ]] || {
  echo "--suite-timeout must be a positive integer" >&2
  exit 2
}
[[ "$gpu_release_timeout" =~ ^[1-9][0-9]*$ ]] || {
  echo "--gpu-release-timeout must be a positive integer" >&2
  exit 2
}

IFS=',' read -r -a suites <<< "$suites_csv"
[[ "${#suites[@]}" -gt 0 ]] || { echo "--suites cannot be empty" >&2; exit 2; }
for suite in "${suites[@]}"; do
  case "$suite" in
    complex_mixed|large_runway|small_runway) ;;
    *) echo "unknown suite: $suite" >&2; exit 2 ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
single_script="$script_dir/run_policy_benchmark.sh"
eval_root="${EVAL_OUTPUT_ROOT:-$project_root/eval_output}"
mkdir -p "$eval_root"

timestamp="$(date +%Y%m%d_%H%M%S)"
checkpoint_hash="$(sha256sum "$checkpoint" | awk '{print substr($1,1,12)}')"
weight_name="$(basename "$checkpoint" .pt | tr -c 'A-Za-z0-9._-' '_')"
batch_dir="$eval_root/${timestamp}_${experiment_type}_${weight_name}_${checkpoint_hash}"
mkdir -p "$batch_dir/logs"

active_pid=""
cleanup_active_group() {
  local pid="$active_pid"
  [[ -n "$pid" ]] || return 0

  if kill -0 -- "-$pid" 2>/dev/null; then
    echo "[batch] terminating process group -$pid" >&2
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
      kill -0 -- "-$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  active_pid=""
}

cleanup_on_exit() {
  cleanup_active_group
}

cleanup_on_signal() {
  local status="$1"
  cleanup_active_group
  exit "$status"
}

trap cleanup_on_exit EXIT
trap 'cleanup_on_signal 130' INT
trap 'cleanup_on_signal 143' TERM

wait_for_gpu_release() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[batch] nvidia-smi unavailable; skipping GPU release polling" >&2
    return 0
  fi

  for _ in $(seq 1 "$gpu_release_timeout"); do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 512 )); then
      echo "[batch] GPU released: ${used} MiB used"
      return 0
    fi
    sleep 1
  done

  echo "[batch] GPU did not return below 512 MiB within ${gpu_release_timeout}s" >&2
  return 75
}

run_suite() {
  local suite="$1"
  local log_path="$batch_dir/logs/${suite}.log"
  local summary_path="$batch_dir/$experiment_type/$suite/summary.json"
  local rc=0

  echo "[batch] START suite=$suite batch_dir=$batch_dir"
  setsid timeout --foreground --signal=TERM --kill-after=30s "$suite_timeout" "$single_script" \
    --experiment-type "$experiment_type" \
    --checkpoint "$checkpoint" \
    --suite "$suite" \
    --output-dir "$batch_dir" \
    "${forward_args[@]}" > >(tee "$log_path") 2>&1 &
  active_pid=$!

  if wait "$active_pid"; then
    rc=0
  else
    rc=$?
  fi
  cleanup_active_group

  if (( rc != 0 )); then
    echo "[batch] FAIL suite=$suite exit_code=$rc; stopping remaining suites" >&2
    return "$rc"
  fi
  if [[ ! -f "$summary_path" ]]; then
    echo "[batch] FAIL suite=$suite exited 0 but did not write $summary_path" >&2
    return 74
  fi

  echo "[batch] DONE suite=$suite"
  wait_for_gpu_release
}

echo "[batch] output_dir=$batch_dir"
for suite in "${suites[@]}"; do
  run_suite "$suite"
done
echo "[batch] all suites completed: ${suites_csv}"
