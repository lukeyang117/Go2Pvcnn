#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 --experiment-type {amp|distillation|ppo|teacher} --checkpoint FILE [benchmark args...]" >&2
  exit 2
fi

experiment_type=""
checkpoint=""
forward_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-type)
      [[ $# -ge 2 ]] || { echo "--experiment-type requires a value" >&2; exit 2; }
      experiment_type="$2"; forward_args+=("$1" "$2"); shift 2 ;;
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "--checkpoint requires a value" >&2; exit 2; }
      checkpoint="$2"; forward_args+=("$1" "$2"); shift 2 ;;
    *) forward_args+=("$1"); shift ;;
  esac
done

[[ -n "$experiment_type" ]] || { echo "--experiment-type is required" >&2; exit 2; }
[[ -n "$checkpoint" && -f "$checkpoint" ]] || { echo "--checkpoint must be an explicit checkpoint file" >&2; exit 2; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-$project_root:$project_root/rsl_rl}"
exec python "$script_dir/policy_benchmark.py" "${forward_args[@]}"
