# Joint MPC RTI Foot-Lead and Nominal-Validity Diagnostic

## Purpose

Separate the reported `foot_root_lead_time=0` from the approved convolutional semantic-field design and determine whether the rolling trace is valid over future control steps.

## Stage

Task 13 flat behavior diagnosis, before any loss or line-search tuning.

## Procedure

- Ran the focused metric regression:

  `PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py -q`

- Ran an 8-step CUDA forward diagnostic on `cuda:1` with command `(1.0, 0.5, 1.0)` and printed root progress, contact schedule, relative foot progress, and onset masks.
- Compared nominal validity for lower and mixed commands and varied only the already-approved nominal reference scales as a read-only diagnostic.

## Findings

- Focused metric regression: `5 passed`.
- The mixed high command `(1.0, 0.5, 1.0)` produces `trajectory.valid=False` from the first planner step; lower commands such as `(0.2, 0, 0)` and `(0.8, 0, 0)` remain nominally valid.
- For the mixed command, the first future node moves a swing foot about `0.134m` relative to the root while the root advances about `0.022m`; subsequent nodes largely carry that foot with the root. This is a real first-node jump, not a semantic-field gradient failure.
- The shared rolling trace records the scalar per-step trajectory validity as the node-valid mask. Once the planner reports invalid, future samples are masked from metrics, which makes `foot_root_lead_time=0` and other ratios inconclusive for that cell.
- The approved soft semantic path is present and covered: grouped Gaussian convolution, propagated class height, Scharr XY gradient, and nonzero XY autograd query gradient all pass in `test_terrain_fields.py`.

## Result

Partial diagnostic. No acceptance claim and no small-obstacle run authorized. No loss, KKT, line-search, nominal contract, or detector was changed.

## Follow-up

1. Trace why the mixed-command cold nominal reports unreachable despite reduced step construction; identify whether the issue is the analytic IK reachability result or the command-conditioned target geometry.
2. Add a minimal regression for the trace validity shape/semantics before changing acceptance metrics.
3. After valid traces exist, evaluate foot speed versus root speed on the first swing edge and only then decide whether any metric implementation change is justified.

## Git refs

- Baseline ref: `724a1c3`
- Candidate ref: `724a1c3`
- Key files: `model/nominal.py`, `model/analytic_ik.py`, `tests/joint_mpc_rti/run_joint_acceptance.py`, `tests/joint_mpc_rti/joint_metrics.py`, `terrain/cost_map.py`, `terrain/query.py`
