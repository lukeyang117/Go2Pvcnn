#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 --experiment-type {amp|distillation|ppo|teacher|ame|ame_amp|semloco} --checkpoint FILE [benchmark args...]" >&2
  exit 2
fi

experiment_type=""
checkpoint=""
has_output_dir=0
forward_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-type)
      [[ $# -ge 2 ]] || { echo "--experiment-type requires a value" >&2; exit 2; }
      experiment_type="$2"; forward_args+=("$1" "$2"); shift 2 ;;
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "--checkpoint requires a value" >&2; exit 2; }
      checkpoint="$2"; forward_args+=("$1" "$2"); shift 2 ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      has_output_dir=1; forward_args+=("$1" "$2"); shift 2 ;;
    *) forward_args+=("$1"); shift ;;
  esac
done

[[ -n "$experiment_type" ]] || { echo "--experiment-type is required" >&2; exit 2; }
[[ -n "$checkpoint" && -f "$checkpoint" ]] || { echo "--checkpoint must be an explicit checkpoint file" >&2; exit 2; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-$project_root:$project_root/rsl_rl}"

case "$experiment_type" in
  amp) task_id="Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AMP-v0" ;;
  distillation) task_id="Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-Distillation-v0" ;;
  ppo) task_id="Isaac-Go2-Cross-Large-Complex-PPO-v0" ;;
  teacher) task_id="Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-v0" ;;
  ame) task_id="Isaac-Go2-Cross-Large-Complex-PPO-v0" ;;
  ame_amp) task_id="Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AME-Go2-v0" ;;
  semloco) task_id="Isaac-Go2-Cross-Large-Complex-SemLoco-v0" ;;
  *) echo "unknown experiment type: $experiment_type" >&2; exit 2 ;;
esac

eval_root="${EVAL_OUTPUT_ROOT:-$project_root/eval_output}"
mkdir -p "$eval_root"
started_at="$(date --iso-8601=seconds)"
timestamp="$(date +%Y%m%d_%H%M%S)"
checkpoint_hash="$(sha256sum "$checkpoint" | awk '{print substr($1,1,12)}')"
weight_name="$(basename "$checkpoint" .pt | tr -c 'A-Za-z0-9._-' '_')"
if [[ "$has_output_dir" -eq 0 ]]; then
  auto_output_dir="$eval_root/${timestamp}_${experiment_type}_${weight_name}_${checkpoint_hash}"
  forward_args+=("--output-dir" "$auto_output_dir")
else
  auto_output_dir="(provided-by-user)"
  for ((index=0; index<${#forward_args[@]}; index++)); do
    if [[ "${forward_args[$index]}" == "--output-dir" && $((index + 1)) -lt ${#forward_args[@]} ]]; then
      auto_output_dir="${forward_args[$((index + 1))]}"
      break
    fi
  done
fi

index_path="$eval_root/run_index.jsonl"
finish_record() {
  local rc="$?"
  local finished_at
  finished_at="$(date --iso-8601=seconds)"
  printf '{"experiment_type":"%s","rl_task":"%s","checkpoint":"%s","checkpoint_sha256":"%s","started_at":"%s","finished_at":"%s","exit_code":%s,"output_dir":"%s"}\n' \
    "$experiment_type" "$task_id" "$checkpoint" "$checkpoint_hash" "$started_at" "$finished_at" "$rc" "$auto_output_dir" >> "$index_path"
  exit "$rc"
}
trap finish_record EXIT

python "$script_dir/policy_benchmark.py" "${forward_args[@]}"
