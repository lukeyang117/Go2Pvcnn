# Joint MPC RTI Early-Phase Subweights And Active-KKT Blocker

## Purpose

Implement the approved early-swing tuning knobs, then explain why ranked cold starts keep selecting alpha-zero nominal instead of responding to those losses.

## Stage

Task 13 flat gate, seven-loss completion and constrained H30 QP diagnosis.

## Approved Loss Completion

- Added `command_early_swing` and `swing_speed_early` under the existing `command` and `swing_speed` loss families; defaults are `1.0` and preserve old scoring.
- Both use the continuous continuing-swing midpoint phase `e=1-tau_mid`; command pressure is reduced by its configured early weight and swing-speed pressure is increased by its configured early weight.
- RED: command lacked schedule input and swing early/late residuals were identical (`2 failed`).
- GREEN: phase tests `2 passed`; complete loss/derivative/QP regression `16 passed`.
- Read-only `command_early_swing=0.1`, `swing_speed_early=10` did not change the first-frame `10.062mm` root leak because both signed commands still selected alpha zero. Do not adopt these values.

## Joint Tuning Stop

After the warm-terminal fix, clean single-variable reruns reject:

- `joint_trust=0.20`: forward `23/25`, margin near zero.
- command scale `0.44`: backward `22/25`; scale fine-tuning remains stopped.
- `posture_joint=0.15`: forward margin only `0.0177rad`, backward `23/25`.
- top-level posture `5`: forward/backward `23/25`, `22/25`.

The scale-0.45 base remains all `25/25` valid, but forward frame 23 leg-4 calf reaches `-0.83705rad`, only `5e-5rad` from the physical upper bound.

## Root Cause

Frame-1 line-search scoring with default early weights:

- Forward alpha losses `(1,.5,.25,.125,0)` are approximately `(5.835,1.519,0.444,0.180,0.0954)`; all but alpha 1 are filter-feasible, so alpha zero is a correct loss-only selection.
- Extra read-only alpha scores down to `0.0078125` approach nominal from above; candidate spacing is not the cause.
- The H30 QP direction has negative `g^T delta` but positive full convex model change: forward `+5.566`, backward `+13.369`. Since zero direction is feasible, this cannot be a constrained QP minimum.

Production-round dense/scan diagnosis on the forward cold QP:

- Free scan: model `-0.090`, box violation `0.593rad`, 248 active candidates.
- First add-only active solve: model `+1.973`, new box violation `0.570rad`, two velocity violations.
- Second add-only solve: model `+5.567`, box violation `0.0986rad`.
- Dense/scan under the same final mask agree within about `4.25e-5`; mask selection, not line search, is the main defect.
- A standard feasible blocking path stays descent (`-0.0067`, `-0.0073`, `-0.0345`) but its second KKT solution encounters another `0.217rad` blocking bound. Larger allowed trust values still leave `0.105-0.220rad` new violations.

## TDD State

`test_refined_cold_command_direction_does_not_increase_convex_qp_model` is intentionally RED and also checks final box/velocity feasibility. Current failure is box violation `0.098587rad` with model change `+5.567`.

The experimental production mask change was reverted because it did not fix the RED case. No third refinement, projection, candidate, filter, or recovery was added.

## Blocker

The frozen combination of exactly two add-only active refinements, final active-KKT parity, and full H30 bound feasibility does not close on the real cold QP. Continuing requires an explicit design amendment: either permit more active KKT rounds, or permit a two-round feasible blocking step whose returned direction is descent/feasible but is not the exact final active-KKT solution.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: working tree on `724a1c3`
- Key files: `config.py`, `losses/command.py`, `losses/swing_speed.py`, `losses/objective.py`, `test_trajectory_losses.py`, `test_trajectory_qp.py`
