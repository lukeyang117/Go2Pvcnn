# Train MPC Num Envs CLI

## Purpose

Add a `train.py` command-line override for the number of environments sampled by the MPC planner per replan.

## Stage

Training entrypoint / MPC runtime configuration.

## Related Todo

- [../todo/T302u-semantic-map-contact-collision-plan.md](../todo/T302u-semantic-map-contact-collision-plan.md)

## Baseline Ref

- Working tree after T302u semantic map-contact implementation.

## Candidate Ref

- Working tree after [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py) CLI update.

## Key Files

- [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)
- [../../Go2Pvcnn/tests/test_train_script_static.py](../../Go2Pvcnn/tests/test_train_script_static.py)
- [../human/human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md)

## Result

Pass. `train.py` now accepts:

```text
--mpc_num_envs <N>
```

When provided, it sets:

```text
env_cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size = N
```

`--num_envs` still controls Isaac environment count; `--mpc_num_envs` controls MPC sampled/participating planner count per replan.

## Verification

RED:

```text
pytest Go2Pvcnn/tests/test_train_script_static.py::test_train_exposes_mpc_num_envs_cli_override -q
```

Observed `1 failed` before implementation.

GREEN:

```text
pytest Go2Pvcnn/tests/test_train_script_static.py -q
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_train_play_parsers_and_gym_registration_are_isolated -q
python -m py_compile Go2Pvcnn/scripts/train.py
git diff --check
```

Observed:

```text
3 passed
1 passed
py_compile exit 0
diff check exit 0
```

## Follow-Up

Use `--mpc_num_envs 64` with 1024-env runs when you want the old 1024/64 MPC participation profile.
