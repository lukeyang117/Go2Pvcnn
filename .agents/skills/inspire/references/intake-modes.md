# Intake Modes

Pick one primary intake mode at the start of each `/inspire` session. If the request spans modes, name the secondary mode but keep one primary structure.

## `image-analysis`

Use when the user provides a screenshot, plot, visualization, or rendered scene.

First-round pattern:

1. State visible facts separately from interpretation.
2. List the most plausible cause directions, not a single forced diagnosis.
3. Call out missing context or evidence.
4. End with one best next question or the allowed next-step choices.

Default lenses:

- visible facts
- possible causes
- missing context
- best next question

## `bug-log-analysis`

Use when the user provides an error, stack trace, runtime anomaly, or terminal output.

First-round pattern:

1. Describe the observed failure plainly.
2. Identify the most likely failing layer or boundary.
3. Separate confirmed evidence from inference.
4. Ask for the next diagnostic fact that most reduces ambiguity.

Default lenses:

- observed failure
- likely failing layer
- missing reproduction evidence
- next diagnostic check

Keep the session in diagnosis-before-fix mode.

## `reference-mapping`

Use when the user provides reference code, reference docs, or a reference directory and wants to discuss how it relates to the current project.

First-round pattern:

1. Identify which reference components appear relevant.
2. Map them to current-project targets or boundaries.
3. Separate transferable ideas from direct-copy risk.
4. Recommend the next reading or tracing step before design or implementation.

Default lenses:

- relevant reference components
- current-project mapping targets
- safe-to-transfer ideas
- direct-copy risks
- next reading or tracing step

## `generic-demand`

Use when the user is discussing a desired outcome or project direction without the other three modes dominating.

First-round pattern:

1. Restate the target outcome.
2. Surface hidden assumptions and constraints.
3. Name likely success criteria and non-goals.
4. Ask one focused question that sharpens the requirement.

Default lenses:

- target outcome
- hidden assumptions
- constraints
- success criteria
- non-goals
