# T305 implementation and smoke evidence

Implemented the reproducible policy benchmark for AMP, Distillation, Pure PPO, and planner-conditioned Teacher. It includes deterministic manifests, mixed/runway suites, batched episode metrics, valid-only 24-frame alignment, resumable JSONL shards, model adapters, formal summary output, and an explicit-checkpoint launcher.

Verification:

- Evaluation unit suite: `20 passed`.
- Python compilation passed for the evaluation package, benchmark scripts, and smoke wrapper.
- Real Isaac run: AMP `model_300.pt`, 1024 environments, 48 transitions, headless, seed `20260903`.
- Result: exit `0`; `env_steps_per_second=5683.8239`; `episodes=6`; `valid_windows=27`.
- All checked observations/actions were finite; no NaN, Inf, OOM, or traceback occurred.
- Output: `results/policy_benchmark_smoke_final/amp/complex_mixed/summary.json` (ignored by git).

The smoke output is marked `is_smoke_test=true` and is rejected by the formal summarizer.
