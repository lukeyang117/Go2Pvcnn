# Viewer Zero-Command Handoff Idempotence

## Purpose

Fix the follow-up viewer issue reported after T110: together zero-command recovery visually kept repeating every `0.7s`, causing an up-and-down "still recovering" motion instead of settling after the first recovery segment.

## Stage

`Go2Pvcnn/extension/viz/go2_foostep_planner.py` together viewer handoff logic, under T100/T110.

## Related Todo

[T100 batched together planner GPU migration](../todo/T100-batched-together-planner-gpu-migration.md#t110-zero-command-rehome-upright-recovery)

## Command / Procedure

Red test:

```bash
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q --tb=short Go2Pvcnn/tests/test_viz_playback.py::TestKinematicPlaybackLogic::test_together_viewer_zero_command_rehome_does_not_replay_vertical_recovery_after_handoff
```

Expected failure before the fix:

```text
second_delta_z = -0.0999999940, expected |second_delta_z| < 0.01
```

Green verification:

```bash
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q --tb=short Go2Pvcnn/tests/test_viz_playback.py::TestKinematicPlaybackLogic::test_together_viewer_zero_command_rehome_does_not_replay_vertical_recovery_after_handoff Go2Pvcnn/tests/test_viz_playback.py::TestKinematicPlaybackLogic::test_together_viewer_handoff_does_not_accumulate_root_height_on_flat_walk
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q --tb=short Go2Pvcnn/tests/test_viz_playback.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_parity.py Go2Pvcnn/tests/test_batched_together_guardrails.py Go2Pvcnn/tests/test_batched_together_runtime_path.py
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_viz_playback.py
```

Viewer-free diagnostic script:

```bash
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python - <<'PY'
# Replan the same zero-command recovery segment multiple times and print first/last z.
PY
```

## Input Conditions

- Python: `/home/lhy/anaconda3/envs/env_isaaclab/bin/python`
- Flat terrain, together backend, viewer-free CPU test fixture.
- Initial state for the red case: root z `0.40`, zero command, full-contact hold semantics.

## Key Metrics

- Red test failed before the fix with repeated second-segment recovery `delta_z=-0.10`.
- Targeted viewer tests: `2 passed`.
- Wider subset (`viz_playback` + together core/parity/guardrail/runtime): `56 passed`.
- `py_compile`: pass.
- Viewer-free diagnostic after the fix:
  - segment 0: `first_z=0.4`, `last_z=0.3`, `delta_z=-0.1`
  - handoff after segment 0: `0.3`
  - segment 1: `first_z=0.3`, `last_z=0.3`, `delta_z=0.0`
  - segment 2: `first_z=0.3`, `last_z=0.3`, `delta_z=0.0`
- A fresh headless viewer attempt was not usable as acceptance evidence because Isaac/PhysX hit a GPU memory / backend initialization failure before stable playback. This was treated as environment noise, not as a regression signal for the viewer handoff logic.

## Result

Pass with scoped caveat.

Together viewer handoff now distinguishes hold-like segments from walking segments:

- walking / moving segments still reconstruct root z from contact support height plus initial clearance, preserving the T109 anti-ratchet fix;
- hold-like zero-command segments now hand off their terminal root z directly, so the next segment starts from the settled base height and does not replay recovery.

## Conclusion

The repeated "still recovering" motion was not a second planner bug. It came from the viewer reusing the anti-ratchet clearance reconstruction for zero-command hold segments, which made every new segment restart from the pre-recovery clearance. The fix stays entirely in viewer handoff logic and leaves the training path unchanged.

## Follow-Up

- Manual interactive viewer confirmation is still useful because the latest real Isaac headless attempt failed during GPU/PhysX startup for environment reasons.
- Training-path semantics and guardrails remain covered by T110 and were not changed in this fix.

## Git Refs

- Baseline Ref: `7cf6c11`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_viz_playback.py](../../Go2Pvcnn/tests/test_viz_playback.py)
