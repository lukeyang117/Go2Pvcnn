# Parallelism Candidate Radius Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Parallelism debug panel slider that controls the candidate foothold circle radius from the current default `0.24 m` up to `0.50 m`.

**Architecture:** Keep the candidate generation logic unchanged and pass the viewer slider value through `ViewerTestTerminalState` into `_parallelism_cfg_from_viewer_args()`. The existing planner result already exposes `parallelism_candidate_radius_m`, so the visualized circles update naturally after replan.

**Tech Stack:** Python 3.10, IsaacSim `omni.ui` viewer panel, PyTorch tests with pytest.

## Global Constraints

- Only touch `Go2Pvcnn/extension/viz/go2_foostep_planner.py`, `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`, and this plan.
- Do not change candidate count; keep each foot at `50` candidates.
- Do not change score, semantic margin, collision filtering, standstill fallback, or root planning.
- Slider min is `0.24`; slider max is `0.50`; default is `ParallelismCfg.candidate_radius_m`.

---

### Task 1: Candidate Radius Viewer Config

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Modify: `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`

**Interfaces:**
- Consumes: `ViewerTestTerminalState.candidate_radius_m: float`
- Produces: `_parallelism_cfg_from_viewer_args(...).candidate_radius_m`

- [ ] **Step 1: Write the failing test**

Add or extend the existing config helper test:

```python
def test_parallelism_cfg_from_viewer_uses_semantic_margin():
    from argparse import Namespace
    from extension.viz.go2_foostep_planner import ViewerTestTerminalState, _parallelism_cfg_from_viewer_args

    cfg = _parallelism_cfg_from_viewer_args(
        Namespace(plan_dt=0.02),
        ViewerTestTerminalState(
            swing_height=0.11,
            semantic_touchdown_margin_m=0.04,
            candidate_radius_m=0.42,
            standstill_fallback_enabled=False,
        ),
    )

    assert cfg.swing_height_m == 0.11
    assert cfg.semantic_touchdown_margin_m == 0.04
    assert cfg.candidate_radius_m == 0.42
    assert cfg.standstill_fallback_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/parallelism/test_viewer_adapter.py -q
```

Expected: FAIL because `ViewerTestTerminalState` does not yet accept `candidate_radius_m`.

- [ ] **Step 3: Implement minimal code**

In `ViewerTestTerminalState`, add:

```python
candidate_radius_m: float = 0.24
```

In `_create_viewer_test_terminal`, add the slider after `swing_height`:

```python
_slider("candidate_radius", "candidate_radius_m", 0.24, 0.50)
```

In `_parallelism_cfg_from_viewer_args`, pass:

```python
candidate_radius_m=float(test_terminal_state.candidate_radius_m)
if test_terminal_state is not None
else ParallelismCfg.candidate_radius_m
```

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/parallelism/test_viewer_adapter.py docs/superpowers/plans/2026-07-30-parallelism-candidate-radius-panel.md
git commit -m "feat: add candidate radius viewer control"
```

### Task 2: Verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes all Task 1 outputs.
- Produces final confidence.

- [ ] **Step 1: Run syntax check**

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py
```

Expected: exit code 0.

- [ ] **Step 2: Run parallelism tests**

```bash
PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/parallelism -q
```

Expected: all tests pass.
