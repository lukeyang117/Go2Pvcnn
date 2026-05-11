# T205 Full-Grid Runtime Verification

## Meta

- Time: `2026-05-08 16:47 +0800`
- Stage: `semantic static course viewer runtime verification`
- Result: `done with concerns`
- Todo: [T200/T205](../todo/T200-semantic-static-course-viewer.md#t205-full-grid-interactive-viewer-startup--manual-confirmation)

## Purpose

- Determine how much stronger automated evidence we can get for the semantic viewer/runtime path beyond the existing compact `4x1` smoke.
- Measure whether the training-aligned full-grid semantic viewer env can really start on this machine, instead of only inferring from compact fixtures.
- Recheck that the `T208` lower `small obstacle` geometry does not obviously break the existing semantic runtime/viewer path.
- Separate:
  - `full-grid startup is impossible`
  - from
  - `full-grid startup is possible, but current pytest/runtime harness is too costly or awkward to use as the standing automated acceptance path`

## Pre-Read

- [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
- [2026-04-29-2359-semantic-static-course-env-isaaclab-compact-runtime-smoke.md](2026-04-29-2359-semantic-static-course-env-isaaclab-compact-runtime-smoke.md)
- [2026-04-30-1432-semantic-native-shape-pool-compact-runtime-acceptance.md](2026-04-30-1432-semantic-native-shape-pool-compact-runtime-acceptance.md)
- [2026-05-07-2248-t208-small-obstacle-height-reduction.md](2026-05-07-2248-t208-small-obstacle-height-reduction.md)
- [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
- [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
- [../human/human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md)

## What Was Verified

### 1. Runtime fixture behavior was re-read

- The existing real-runtime fixture still forcibly compacts the semantic viewer terrain grid to `4 x 1` via:
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - `_configure_compact_semantic_runtime_grid()`
- The semantic viewer play env itself still restores the training-aligned terrain generator size from `SEMANTIC_TERRAIN_CFG`, which is:
  - `num_rows = 10`
  - `num_cols = 20`
  - via [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_semantic_env_cfg.py)
  - consumed by [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py)

### 2. Compact vs full-grid real startup cost was measured

No repository code was changed. Full-grid was exercised by monkeypatching the compact-grid helper to a no-op inside one-off verification scripts.

Observed real runtime startup timings on this machine under `env_isaacsim`:

- compact `4x1` runtime fixture:
  - scene creation: `2.494933s`
  - simulation start: `0.788194s`
- full-grid default semantic viewer env:
  - scene creation: `5.531064s` and `5.766482s` on two independent runs
  - simulation start: `6.134866s` and `6.672670s` on two independent runs

Interpretation:

- full-grid startup is **not** impossible on this machine
- but it is materially heavier than compact smoke:
  - scene creation roughly `2.2x`
  - simulation start roughly `7.8x`

### 3. Current runtime acceptance path still works after `T208`

On this machine, the currently available Isaac env is `env_isaacsim`, not the historical `env_isaaclab`.

Reconfirmation points:

- the shared semantic-together runtime subset can still complete with exit `0` under `env_isaacsim`
- the prior `T208` targeted small-anchor runtime evidence remains consistent with this run’s environment assumptions
- no new evidence appeared that the reduced `small obstacle` height breaks semantic startup, scanner attachment, or targeted semantic-anchor paths

### 4. Current pytest/runtime harness remains awkward for broader full-grid automation

Two practical findings emerged:

- a narrow shared-fixture subset:
  - `viewer_together_semantic_smoke_reports_required_obstacle_hits`
  - `viewer_together_targeted_s4_small_scan_reports_semantic_hits`
  - `viewer_together_targeted_s4_large_scan_reports_semantic_hits`
  - `compact_semantic_runtime_shape_pool_includes_capsule_and_cone`
  - can finish under `env_isaacsim` with exit `0`
- a broader mixed runtime subset that also includes separate runtime-fixture setup remains prone to long-running/no-summary behavior and did not finish within `180s`

So the remaining limitation is more specific than the old open question:

- it is **not** “full-grid cannot start”
- it is closer to:
  - “the existing pytest/runtime fixture strategy is still expensive/awkward enough that compact smoke remains the practical standing automated acceptance path”

## Commands

### Readback / fixture inspection

```bash
rg -n "compact|4x1|num_rows|num_cols|semantic_scan_near_s4_anchor" \
  Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py \
  Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_semantic_env_cfg.py -S
```

### Real compact startup timing evidence

```bash
PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim \
VIEWER_RUNTIME_DIAGNOSTICS_DEVICE=cuda:2 \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -u - <<'PY'
from tests.fixtures import viewer_runtime_diagnostics as vd
runtime = vd.make_real_runtime_fixture(num_envs=1, planner_backend='together')
runtime.plan_case('forward')
runtime.semantic_scan_near_s4_anchor('small')
runtime.semantic_scan_near_s4_anchor('large')
runtime.compact_semantic_shape_kinds()
runtime.close()
PY
```

Timing evidence was read from:

- `/tmp/t205_compact_force_stdout.log`

### Real full-grid startup timing evidence

```bash
PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim \
VIEWER_RUNTIME_DIAGNOSTICS_DEVICE=cuda:3 \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -u - <<'PY'
from tests.fixtures import viewer_runtime_diagnostics as vd
orig = vd.RealViewerRuntimeFixture._configure_compact_semantic_runtime_grid
vd.RealViewerRuntimeFixture._configure_compact_semantic_runtime_grid = lambda self: None
try:
    runtime = vd.make_real_runtime_fixture(num_envs=1, planner_backend='together')
    runtime.plan_case('forward')
    runtime.semantic_scan_near_s4_anchor('small')
    runtime.semantic_scan_near_s4_anchor('large')
    runtime.close()
finally:
    vd.RealViewerRuntimeFixture._configure_compact_semantic_runtime_grid = orig
PY
```

And a second minimal startup-only proof:

```bash
PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim \
VIEWER_RUNTIME_DIAGNOSTICS_DEVICE=cuda:2 \
timeout 90s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -u - <<'PY'
import os
from tests.fixtures import viewer_runtime_diagnostics as vd
orig = vd.RealViewerRuntimeFixture._configure_compact_semantic_runtime_grid
vd.RealViewerRuntimeFixture._configure_compact_semantic_runtime_grid = lambda self: None
try:
    runtime = vd.make_real_runtime_fixture(num_envs=1, planner_backend='together')
    tg = runtime.env_cfg.scene.terrain.terrain_generator
    print(int(tg.num_rows), int(tg.num_cols), flush=True)
    os._exit(0)
finally:
    vd.RealViewerRuntimeFixture._configure_compact_semantic_runtime_grid = orig
PY
```

Timing evidence was read from:

- `/tmp/t205_fullgrid_force_stdout.log`
- `/tmp/t205_fullgrid_ae4L.log`

### Focused runtime subset under current machine env

```bash
timeout 150s bash -lc '
PYTHONPATH=Go2Pvcnn conda run -n env_isaacsim python -m pytest --noconftest \
  Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k
  "viewer_together_semantic_smoke_reports_required_obstacle_hits or
   viewer_together_targeted_s4_small_scan_reports_semantic_hits or
   viewer_together_targeted_s4_large_scan_reports_semantic_hits or
   compact_semantic_runtime_shape_pool_includes_capsule_and_cone";
printf "\\nEXIT_CODE:%s\\n" $?'
```

Result:

- `EXIT_CODE:0`

### Broader mixed runtime subset

```bash
timeout 180s bash -lc '
PYTHONPATH=Go2Pvcnn conda run -n env_isaacsim python -m pytest --noconftest \
  Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k
  "viewer_runtime_uses_semantic_height_scanner_contract or
   viewer_together_semantic_smoke_reports_required_obstacle_hits or
   viewer_together_targeted_s4_small_scan_reports_semantic_hits or
   viewer_together_targeted_s4_large_scan_reports_semantic_hits or
   compact_semantic_runtime_shape_pool_includes_capsule_and_cone";
printf "\\nEXIT_CODE:%s\\n" $?'
```

Result:

- did not complete within the `180s` cap
- timeout exit observed during this verification pass

## Key Evidence

- Compact startup remains inexpensive enough for standing smoke:
  - scene creation `2.494933s`
  - simulation start `0.788194s`
- Full-grid startup is now measured instead of assumed:
  - scene creation `5.53s` to `5.77s`
  - simulation start `6.13s` to `6.67s`
- The semantic viewer full-grid path can really boot on this machine under `env_isaacsim`.
- The existing compact runtime evidence path still survives the `T208` lower-small-obstacle geometry.
- The practical automation boundary is now clearer:
  - compact/shared-fixture runtime subset is usable
  - broader mixed runtime pytest remains expensive/awkward and can exceed `180s`

## Conclusion

- `T205` is stronger than before at the automated-evidence level:
  - we now have direct full-grid startup cost evidence
  - we now know full-grid startup is feasible on this machine
  - we now know compact runtime smoke remains the pragmatic standing automated acceptance path
- The irreducible remaining gap is still **manual visual confirmation**, not startup feasibility:
  - visual distribution/appearance of the richer native shape pool
  - subjective viewer-side quality of `sphere/cuboid/cylinder/capsule/cone`
- So the remaining open portion of `T205` should be treated as:
  - `manual semantic viewer confirmation still open`
  - rather than
  - `full-grid startup cost unmeasured`

## Git Refs

- Baseline Ref: `8e8acc0`
- Candidate Ref: `working tree on top of 8e8acc0 with unrelated dirty entries preserved; verification-only T205 pass`
- Key Files:
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [index.md](index.md)
