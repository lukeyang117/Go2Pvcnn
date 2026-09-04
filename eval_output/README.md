# Policy Evaluation Outputs

`scripts/run_policy_benchmark.sh` creates one directory per execution:

```text
eval_output/YYYYMMDD_HHMMSS_<experiment>_<weight>_<sha256-prefix>/
```

Each run is recorded in `eval_output/run_index.jsonl` with:

- start and finish time;
- experiment type and RL task ID;
- absolute checkpoint path and SHA256 prefix;
- generated output directory;
- process exit code.

The four benchmark weights currently used are:

- AMP: `experiment_type=amp`
- Distillation student: `experiment_type=distillation`
- Pure PPO baseline: `experiment_type=ppo`
- Parallelism Teacher: `experiment_type=teacher`

Do not pass `--output-dir` when automatic timestamp/checkpoint naming is desired.
