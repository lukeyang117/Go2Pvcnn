# Current MPC PPO HTML Overview

## Purpose

- Create a Chinese HTML overview diagram for the current MPC and PPO relationship.

## Stage

- Documentation / MPC semantic RL workflow explanation.

## Related Todo

- [T302p](../todo/T302p-mpc-command-frame-alignment-plan.md)
- [T302l](../todo/T302l-mpc-rl-participation-and-reward-plan.md)

## Procedure

- Read the mandatory repository memory and relevant MPC/PPO notes.
- Created [../../docs/current-mpc-ppo-overview.html](../../docs/current-mpc-ppo-overview.html).
- Parsed the HTML with Python `html.parser`.

## Input Conditions

- Current dashboard focus: T302p MPC command-frame alignment / flat all-direction direction metrics.
- This was a documentation-only task; no runtime code or training configuration was changed.

## Key Metrics

- HTML parse: pass.
- File size: `11830` bytes.

## Result

- Pass. The HTML is now a framework-style diagram similar to a model architecture figure, with dashed module groups, stacked blocks, arrows, and iteration labels. It summarizes:
  - IsaacLab state -> MPC planning -> reference cache -> reward -> PPO update.
  - Current command-frame contract.
  - PLAY / VIEWER split.
  - Current verified metrics and open T302p per-leg endpoint metric issue.

## Conclusion

- The framework diagram is ready for local browser viewing.

## Follow-Up

- Update the HTML if T302p per-leg direction metric contract changes or if MPC reference enters observations in any future design.

## Git Refs

- Baseline Ref: working tree at `2026-06-09 20:26 +0800`
- Candidate Ref: working tree with [../../docs/current-mpc-ppo-overview.html](../../docs/current-mpc-ppo-overview.html)
- Key Files:
  - [../../docs/current-mpc-ppo-overview.html](../../docs/current-mpc-ppo-overview.html)
  - [../todo.md](../todo.md)
  - [index.md](index.md)
