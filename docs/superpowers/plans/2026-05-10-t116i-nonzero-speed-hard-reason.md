# T116i Nonzero Speed And Hard-Reason Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove nonzero-command `beta=0` candidates, add all-hard hard-reason/rank diagnostics, and verify the behavior with deterministic and IsaacSim headless terminal-output tests.

**Architecture:** Keep the existing `Go2Pvcnn/extension/batched_together_planner` implementation in place. Add fixed-shape tensor diagnostics to the planner/cost/result path, then let viewer/test reporting convert those tensors to reason strings outside the hot path.

**Tech Stack:** Python, PyTorch tensors, pytest, IsaacSim/IsaacLab runtime through `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`.

---

## Controller Contract

- Main agent does not edit production code or test code during implementation.
- Worker subagents own code/test modifications.
- Main agent reviews worker diffs, test files, and command output, then decides whether the flow passes.
- Tests may be added in a new file: `Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py`.

## Scope Files

- Create: `Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py`
- Modify: `Go2Pvcnn/extension/batched_together_planner/types.py`
- Modify: `Go2Pvcnn/extension/batched_together_planner/costs.py`
- Modify: `Go2Pvcnn/extension/batched_together_planner/planner.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Modify: `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py`
- Optional modify: `Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py` only if shared runtime fixture assertions need widening.

Do not create new planner source files.

## Task 1: RED Tests For Nonzero Candidate Tables

**Files:**
- Create: `Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py`
- Read: `Go2Pvcnn/extension/batched_together_planner/planner.py`
- Read: `Go2Pvcnn/extension/batched_together_planner/parameterization.py`

- [ ] **Step 1: Add candidate-table tests**

Create tests that import `_t116_candidate_tables`, `TogetherPlannerConfig`, and T116 mode constants. Assert exact K=5 beta tables:

```python
def test_t116i_nonzero_command_beta_tables_do_not_include_zero():
    modes = torch.tensor(
        [T116_MODE_CRUISE, T116_MODE_APPROACH_SMALL, T116_MODE_CROSS_SMALL, T116_MODE_BYPASS_OBSTACLE],
        dtype=torch.long,
    )
    betas, routes, _signs = _t116_candidate_tables(
        modes,
        device=torch.device("cpu"),
        dtype=torch.float32,
        cfg=TogetherPlannerConfig(),
    )
    expected = torch.tensor(
        [
            [1.00, 0.80, 0.60, 0.40, 0.20],
            [0.80, 0.65, 0.50, 0.35, 0.20],
            [0.60, 0.50, 0.40, 0.30, 0.20],
            [0.60, 0.40, 0.60, 0.40, 0.20],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(betas, expected)
    assert torch.all(betas > 0.0)
    assert routes.shape == (4, 5)
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "beta_tables"
```

Expected before implementation: FAIL because existing tables still contain `0.00`.

## Task 2: RED Tests For Hard-Reason Schema And All-Hard Ranking

**Files:**
- Modify: `Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py`
- Read: `Go2Pvcnn/extension/batched_together_planner/types.py`
- Read: `Go2Pvcnn/extension/batched_together_planner/costs.py`
- Read: `Go2Pvcnn/extension/batched_together_planner/planner.py`

- [ ] **Step 1: Add result-schema test**

Write a lightweight test that calls an existing small deterministic planner fixture or helper from current together tests. If no helper is reusable, build the smallest flat terrain/state fixture already used in `test_batched_together_core.py`. Assert the result exposes:

```python
assert result.candidate_hard_reason_mask.shape == (batch_size, 5, hard_reason_count)
assert result.selected_hard_reason_mask.shape == (batch_size, hard_reason_count)
assert result.candidate_hard_rank_cost.shape == (batch_size, 5)
assert result.selected_hard_rank_cost.shape == (batch_size,)
assert result.candidate_hard_reason_mask.dtype == torch.bool
assert result.selected_hard_reason_mask.dtype == torch.bool
```

- [ ] **Step 2: Add all-hard ranking test**

Construct a deterministic case where all K candidates are hard/infeasible. Prefer using a semantic obstacle layout or monkeypatch-free terrain fixture that makes every candidate violate an existing hard rule. Assert:

```python
assert int(result.status.item()) == int(TogetherPlannerStatus.ALL_INFEASIBLE)
assert float(result.selected_beta.item()) > 0.0
expected_idx = torch.argmin(result.candidate_hard_rank_cost, dim=1)
assert int(result.selected_candidate_index.item()) == int(expected_idx.item())
torch.testing.assert_close(
    result.selected_hard_reason_mask,
    result.candidate_hard_reason_mask.gather(
        1,
        expected_idx.view(1, 1, 1).expand(1, 1, result.candidate_hard_reason_mask.shape[-1]),
    ).squeeze(1),
)
assert bool(result.selected_hard_reason_mask.any().item())
```

If `selected_candidate_index` does not exist yet, this test should fail until the worker either exposes it or uses an already exposed equivalent index. Prefer exposing `selected_candidate_index [B]` because it makes all-hard auditing explicit.

- [ ] **Step 3: Run RED**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "hard_reason or all_hard"
```

Expected before implementation: FAIL because hard-reason fields and all-hard rank selection do not exist.

## Task 3: Implement Nonzero Tables, Hard Reasons, And All-Hard Selection

**Files:**
- Modify: `Go2Pvcnn/extension/batched_together_planner/types.py`
- Modify: `Go2Pvcnn/extension/batched_together_planner/costs.py`
- Modify: `Go2Pvcnn/extension/batched_together_planner/planner.py`
- Test: `Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py`

- [ ] **Step 1: Update candidate tables**

In `_t116_candidate_tables`, replace only beta values:

```text
CRUISE          [1.00, 0.80, 0.60, 0.40, 0.20]
APPROACH_SMALL  [0.80, 0.65, 0.50, 0.35, 0.20]
CROSS_SMALL     [0.60, 0.50, 0.40, 0.30, 0.20]
BYPASS_OBSTACLE [0.60, 0.40, 0.60, 0.40, 0.20]
```

- [ ] **Step 2: Add fixed reason axis**

Add fixed reason constants in an existing planner/cost module, not a new file:

```text
boundary_invalid
path_collision
body_hard_collision
leg_hard_collision
crossing_not_grounded
touchdown_on_small
front_foot_small_collision
rear_foot_small_collision
per_leg_foot_small_collision
base_small_penetration
touchdown_on_large
foot_large_collision
direction_violation
```

Expose the reason names for viewer/test formatting through an importable tuple such as `HARD_REASON_NAMES`.

- [ ] **Step 3: Extend cost result**

Extend the internal cost result so `compute_costs(...)` returns:

```text
hard_reason_mask [B*K, R] bool
hard_rank_cost [B*K] float
```

Keep all calculations tensorized. Do not add CPU conversion in the planner hot path.

- [ ] **Step 4: Include direction violation as a hard reason**

`candidate_direction_violation` is computed in `planner.py` after `compute_costs(...)`. Append or OR it into the reshaped reason mask before selection so candidate reasons include direction failures.

- [ ] **Step 5: Implement all-hard selector**

Change selection order to:

```text
feasible argmin(total)
else safe_fallback argmin(total)
else argmin(candidate_hard_rank_cost)
```

Keep `status=ALL_INFEASIBLE` when selected candidate is not feasible.

- [ ] **Step 6: Expose result fields**

Add these fields to `TogetherPlannerResult`:

```text
candidate_hard_reason_mask [B,5,R]
selected_hard_reason_mask [B,R]
candidate_hard_rank_cost [B,5]
selected_hard_rank_cost [B]
selected_candidate_index [B]
```

- [ ] **Step 7: Run GREEN deterministic tests**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_guardrails.py -q
```

Expected after implementation: PASS, or document remaining failures with exact assertion output.

## Task 4: Viewer/Runtime Reason Formatting And Headless Tests

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Modify: `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py`
- Modify: `Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py`
- Optional modify: `Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py`

- [ ] **Step 1: Propagate fields through viewer adapter**

Update viewer result adaptation so `ViewerTrajectoryResult` carries:

```text
candidate_hard_reason_mask
selected_hard_reason_mask
candidate_hard_rank_cost
selected_hard_rank_cost
selected_candidate_index
```

- [ ] **Step 2: Add reason string formatting outside hot path**

Add formatting helpers in viewer/test reporting code only. Output strings should include:

```text
selected_hard_reasons=
candidate_hard_rank=
candidate_hard_reasons=
```

Only print the detailed fields when `status=ALL_INFEASIBLE` or selected hard reason is non-empty.

- [ ] **Step 3: Add headless-output tests**

In the new T116i test file, add runtime-oriented tests using existing `viewer_runtime_diagnostics` fixture patterns:

```text
flat/no-semantic commands:
  forward, backward, lateral_left, lateral_right, yaw_left, yaw_right
  assert nonzero commands are not standstill and selected_beta > 0

small obstacle commands:
  forward, backward, lateral_left, lateral_right
  assert CROSS_SMALL evidence, selected_beta > 0, no touchdown/foot/base/body/leg collision metrics

all-infeasible terminal formatting:
  construct or fake diagnostics with ALL_INFEASIBLE
  assert formatted output contains selected_hard_reasons, candidate_hard_rank, candidate_hard_reasons
```

- [ ] **Step 4: Run GREEN runtime/unit tests**

Run lightweight unit path:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "format or runtime_summary or headless_output"
```

Run IsaacSim headless path with timeout:

```bash
timeout -s INT -k 20s 240s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "headless or flat or small"
```

Expected: PASS. If runtime exceeds timeout, stop and report timeout output; do not leave IsaacSim running.

## Task 5: Worker Self-Review Output

**Files:**
- No new files unless notes/log updates are explicitly needed by worker.

- [ ] **Step 1: Report changed files**

Worker final response must list every changed file.

- [ ] **Step 2: Report RED/GREEN evidence**

Worker final response must include exact commands and high-signal output:

```text
RED command(s): ...
RED result: failed because ...
GREEN command(s): ...
GREEN result: ...
IsaacSim headless command: ...
IsaacSim headless result: ...
```

- [ ] **Step 3: Report residual risks**

Worker must explicitly say whether any runtime test was skipped, timed out, or only unit-tested.

## Main-Agent Review Checklist

Main agent reviews, but does not edit code:

- [ ] New test file exists and directly covers T116i.
- [ ] Tests were observed RED before implementation.
- [ ] No nonzero-command beta table contains `0.0`.
- [ ] Zero command hold path still exists.
- [ ] Hard-reason fields are fixed shape and tensor-only in planner hot path.
- [ ] All-hard selection uses `candidate_hard_rank_cost`.
- [ ] Infeasible output exposes selected and per-candidate reasons.
- [ ] Deterministic tests pass.
- [ ] IsaacSim headless tests pass or failures are concrete and actionable.
- [ ] No new planner source file was created.
- [ ] No hot-path CPU/NumPy/dynamic-subbatch pattern was introduced.
