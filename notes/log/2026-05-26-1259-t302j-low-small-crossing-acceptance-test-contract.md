# T302j Low-Small Crossing Acceptance Test Contract

## Purpose

Define the metric/test contract requested by the user:

1. touchdown must be behind no swing point in the velocity/translation direction;
2. the FK-realized foot must actually cross over the small obstacle;
3. planned and FK swing feet must not rise above root height.

## Stage

- `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`
- `Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py`

## Related Todo

- [../todo/T302j-touchdown-endpoint-consistency.md](../todo/T302j-touchdown-endpoint-consistency.md)

## Test Design

Added two focused metric tests:

- `test_low_small_crossing_acceptance_combines_endpoint_foot_over_and_root_height_guard`
  - builds a valid synthetic low-small crossing arc;
  - asserts `touchdown_behind_swing_foot_along_max_m == 0`;
  - asserts FK foot-over success, lift-then-land, and touchdown-after all pass;
  - asserts planned/FK swing foot above root are `<= 0`.
- `test_low_small_crossing_acceptance_rejects_foot_over_arc_above_root`
  - uses the same style of crossing arc but lowers root height;
  - asserts the above-root metric catches the violation.

## Verification

```bash
pytest -q \
  Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py::test_low_small_crossing_acceptance_combines_endpoint_foot_over_and_root_height_guard \
  Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py::test_low_small_crossing_acceptance_rejects_foot_over_arc_above_root
```

Result: `2 passed`.

```bash
pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
```

Result: `34 passed`.

```bash
python -m py_compile \
  Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py
```

Result: pass.

## Current Default MPC Status Against This Contract

Latest real IsaacLab reproduction:

- log: [2026-05-26-1249-t302j-default-mpc-foot-above-root-reproduction.md](2026-05-26-1249-t302j-default-mpc-foot-above-root-reproduction.md)
- output: `tmp/t302i-viewer-realized-foot-mismatch/t302j_default_mpc_above_root_repro.jsonl`

Forward low-small crossing:

- touchdown endpoint condition: pass, `touchdown_behind_swing_foot_along_max_m=0.0`;
- foot-over condition: pass, `fk_foot_over_low_small_success=1`;
- no high-foot condition: fail, planned/FK swing foot above root `+0.025513/+0.073637m`.

## Conclusion

The test contract is now explicit. Current default MPC satisfies endpoint and foot-over for forward low-small crossing, but fails the root-height guard. The next implementation should reduce swing height relative to root without losing FK foot-over clearance or reintroducing touchdown-backtracking.

## Git Refs

- Baseline Ref: `working tree @ c54dc5c`
- Candidate Ref: `working tree @ c54dc5c plus acceptance metric tests`
- Key Files:
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
