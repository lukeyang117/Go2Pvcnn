# Joint MPC RTI Infeasible Fallback Status Fix

## Purpose

Prevent an alpha-zero nominal fallback that fails the existing finite/joint-position/joint-velocity filters from being reported and published as solver status zero.

## Stage

Task 13 principled implementation correction discovered during approved nominal-scale diagnosis.

## Root Cause

The five-candidate line search correctly computed its three filter masks. When no candidate was feasible it selected index 4 as the nominal fallback, but `sqp_rti_update` checked only finite state/loss and discarded the selected candidate's feasibility. A finite nominal outside joint limits could therefore receive status zero and `trajectory.valid=True`.

## TDD Evidence

- RED: `2 failed, 12 passed`.
- The line-search regression failed because `selected_feasible` was absent.
- The planner regression demonstrated a finite, all-filtered nominal fallback receiving status zero.
- GREEN after the minimal fix: `14 passed in 4.40s`.

## Change

- `LineSearchResult` now exposes `selected_feasible = valid[row, selected_index]` from the existing three masks.
- SQP status zero now requires both finite output and `selected_feasible`.
- Candidate count, alpha values, three filters, seven-loss scoring, tie preference, returned nominal fallback state, KKT, and planner recovery structure are unchanged.

## Result

Scoped pass. A broader focused regression and fresh ranked default run remain required; this log does not claim the flat gate.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: working tree on `724a1c3`
- Key files: `solver/line_search.py`, `solver/sqp_rti.py`, `test_line_search_v2.py`, `test_rti_pipeline.py`
