# Analysis Lenses

Use multiple lenses visibly in normal `/inspire` replies. Do not collapse immediately to one interpretation when uncertainty remains.

## Core Lenses

- `visible facts`: what is directly observable from the user's image, logs, files, or wording
- `likely cause`: the most plausible explanations or causal layers
- `missing evidence`: what is still unknown and blocks confidence
- `constraints`: repo rules, user constraints, environment limits, or workflow gates
- `risks`: what could go wrong if the team acts on the current understanding too early
- `acceptance targets`: what evidence would make the discussion feel resolved enough to design or break down into todo work

## Mode-Specific Emphasis

- `image-analysis`: emphasize `visible facts`, `likely cause`, `missing evidence`
- `bug-log-analysis`: emphasize `visible facts`, `likely cause`, `missing evidence`, `constraints`
- `reference-mapping`: emphasize `visible facts`, `constraints`, `risks`, `acceptance targets`
- `generic-demand`: emphasize `constraints`, `risks`, `acceptance targets`

## Reply Skeleton

Each discussion reply should make these elements easy to spot:

1. current understanding
2. lenses being applied
3. interpretations or solution directions
4. evidence gaps or unresolved constraints
5. one focused follow-up question or the allowed next-step choices

## Guardrails

- Do not present a multi-step implementation proposal as if it is already approved.
- Do not ask a bloated questionnaire.
- Do not hide uncertainty.
- Do not offer `写 todo` during discussion mode.
