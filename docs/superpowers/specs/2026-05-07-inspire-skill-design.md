# Inspire Skill Design

## Metadata

- **Date**: 2026-05-07
- **Topic**: project-local `/inspire` skill for requirement analysis, design gating, todo-first execution handoff, and subagent-only implementation
- **Status**: Draft for spec review
- **Primary Trigger**: `/inspire <content>`
- **Target Location**: `.agents/skills/inspire/`

## 1. Problem Statement

The current collaboration pattern can jump too quickly from an initial request to code edits, especially when the user speaks directly and already has a preferred direction in mind. That creates two recurring problems:

1. the agent may treat an exploratory request as an implementation request
2. the shared understanding of goals, constraints, success metrics, and reference material may still be incomplete when implementation starts

The user wants a project-local skill named `inspire` that creates an explicit discussion mode before any code changes. In that mode, the agent should analyze the user's input from multiple angles, ask targeted follow-up questions, help deepen the shared understanding, and only move to design, todo writing, or implementation when the user explicitly approves the transition.

`/inspire` must also integrate with this repository's existing `notes/todo.md`, branch pages, and `notes/log/` workflow. Once work moves beyond discussion, execution should remain subagent-driven: the primary agent owns orchestration, todo/log maintenance, and review; subagents own code changes, testing, and structured result reporting.

## 2. Goals

### In Scope

- Add a project-local skill triggered explicitly by `/inspire ...`.
- Create a discussion-first mode that does not edit code by default.
- Support four primary intake modes:
  - image analysis
  - bug or terminal-output analysis
  - reference-code or reference-doc mapping
  - general requirement discussion
- Require explicit user approval before:
  - generating a design document
  - writing todo nodes
  - starting implementation
- Generate a design document by explicitly entering the `brainstorming` workflow, while applying the repository-specific overrides from this spec.
- Run a subagent review after each design draft, with the review focused on requirement coverage and test/acceptance-metric coverage.
- Express implementation planning as todo nodes and child nodes, not as a separate plan document.
- Keep the primary agent out of direct code editing during execution phases; code changes and testing belong to subagents.

### Out Of Scope

- Automatic triggering from natural-language requests without `/inspire`.
- Silent transitions from discussion into implementation.
- Writing a standalone implementation plan under `docs/...plans/...`.
- Allowing the primary agent to directly modify code during the implementation phase owned by this workflow.
- Replacing repository-wide default agent behavior outside explicit `/inspire` sessions.

## 3. User Requirements Captured From Discussion

The design must preserve these confirmed requirements:

1. `inspire` is manually triggered with `/inspire`, not automatically.
2. The user wants to finish discussing the requirement before any code edits happen.
3. The user may provide:
   - an image and ask for cause analysis
   - a bug and terminal output and ask for cause analysis
   - reference code or many documents and ask how they should combine with the current project
3a. `inspire` should analyze the request from multiple angles, ask focused follow-up questions, and help deepen the shared understanding rather than stopping at one shallow interpretation.
3b. `inspire` must integrate with the repository's `notes/todo.md`, relevant branch pages under `notes/todo/`, and official logs under `notes/log/` when the user approves downstream actions.
4. Direct user phrasing must not be treated as implicit approval to implement.
5. Before any implementation work, the agent must wait for explicit approval to:
   - generate a design
   - write todo
   - implement
6. Design generation should follow `brainstorming` expectations and formatting.
6a. When the user chooses `生成设计`, `inspire` should explicitly transition into the `brainstorming` design workflow rather than only imitating its style.
7. After writing a design, the agent must use a subagent to review whether the previously discussed requirement points and test indicators were actually captured.
8. There should be no standalone implementation plan document in this workflow.
9. If the user later wants execution planning, the agent should write todo nodes rather than a plan doc.
10. After todo writing, implementation and testing should still be performed by subagents, not by the primary agent.
11. The primary agent should own orchestration, todo-state maintenance, log writing, and review.
12. If results or metrics look strange, the primary agent should create a follow-up node and dispatch another subagent instead of directly editing code.

## 4. Primary Use Cases

### 4.1 Image Analysis

The user provides a screenshot, plot, visualization, or rendered scene and asks for analysis. `inspire` should separate:

- what is directly visible
- what those observations could imply
- what evidence is still missing
- what question would most improve the diagnosis

### 4.2 Bug Or Terminal Output Analysis

The user provides an error, stack trace, runtime anomaly, or terminal output and asks why it happened. `inspire` should keep the conversation in a root-cause-analysis mode rather than a fix-now mode.

### 4.3 Reference Mapping

The user provides a reference implementation, a large documentation set, or a reference directory such as `raw/kinematic_footsteps/`, and wants to discuss how it should connect to the current repository. `inspire` should help map:

- which reference components matter
- which current-project components they correspond to
- which parts are transferable
- which parts should not be copied directly

### 4.4 Direct Requirement Discussion

The user describes a desired direction in plain language and wants the agent to ask better questions, widen the analysis lens, and help converge on a shared design-ready understanding before any implementation starts.

## 5. Workflow Overview

`/inspire` should behave as a lightweight conversation state machine with explicit approval gates.

### 5.1 Main Discussion Path

```text
/inspire <content>
-> classify input mode
-> analyze from multiple lenses
-> identify evidence gaps or ambiguity
-> ask one focused follow-up question or present a small set of options
-> refine shared understanding
-> present next-step choices
-> wait for explicit user approval
```

The available next-step choices in the discussion path are:

- `继续分析`
- `生成设计`
- `结束`

No code edits, todo writes, or design-file writes may happen during this path without explicit user approval.
Discussion mode does **not** allow skipping directly to todo writing. The required progression is discussion first, then design, then optional todo writing after design approval.

### 5.2 Design Path

```text
user approves "生成设计"
-> enter `brainstorming`-governed design mode
-> write design spec
-> run subagent review
-> primary agent syncs design-state memory in `notes/todo.md` and the relevant branch page
-> primary agent writes the official design-stage log
-> return design + review summary
-> stop and wait for user
```

The available next-step choices after the design path are:

- `修改 design`
- `写 todo`
- `结束`

`写 todo` is only valid after a design exists and the user accepts that design path as the basis for execution breakdown.
The design path updates design-state repository memory, but it does not create implementation todo leaves until the user explicitly chooses `写 todo`.

### 5.3 Todo Path

```text
user approves "写 todo"
-> primary agent maps design into notes/todo system
-> primary agent writes the official todo-stage log
-> no standalone plan doc is written
-> stop and wait for user
```

### 5.4 Execution Path

```text
user explicitly approves implementation of a specific todo leaf
-> primary agent dispatches a subagent
-> subagent edits code + runs tests/verification
-> subagent returns structured results
-> primary agent reviews, writes official log, updates todo status
-> if metrics look strange, create follow-up leaf and dispatch another subagent
```

## 6. Trigger And Session Contract

### 6.1 Trigger Rule

`inspire` only activates when the user explicitly invokes it with `/inspire ...`.

Supported trigger examples:

- `/inspire 分析一下这张图为什么会这样`
- `/inspire 这个 bug 的终端输出说明了什么`
- `/inspire 看一下 raw/kinematic_footsteps 和当前项目应该怎么结合`

Unsupported trigger behavior:

- natural-language auto-triggering without `/inspire`
- silently entering inspire mode just because the user sounds exploratory

### 6.2 Session Freeze Rule

While the session remains in discussion mode, the agent must not:

- edit code
- write todo nodes
- write a design file
- produce a standalone implementation plan
- start implementation

The only exception is after explicit user approval for a specific transition such as `生成设计` or `写 todo`.
During discussion mode, `写 todo` is not a valid direct transition; it only becomes valid after the design phase has completed.
User phrasing that sounds directional but does not explicitly approve a workflow transition or a todo leaf, such as "应该这样做" or "就改成这样", still does not count as approval.

### 6.3 Approval Rule

Direct phrasing from the user such as "应该这样做" or "就改成这样" must not be treated as execution approval while `/inspire` is active. The agent must still ask for an explicit next-step approval.

## 7. Input Classification

The session must classify the initial request into one of four intake modes:

- `image-analysis`
- `bug-log-analysis`
- `reference-mapping`
- `generic-demand`

If an input spans multiple modes, the primary mode should be chosen and the secondary mode can be called out explicitly.

### 7.1 image-analysis

Default analysis lenses:

- visible facts
- possible causes
- missing context
- best next question

### 7.2 bug-log-analysis

Default analysis lenses:

- observed failure
- likely failing layer
- missing reproduction evidence
- next diagnostic check

This mode should align with the spirit of `systematic-debugging`: diagnosis before fixes.

### 7.3 reference-mapping

Default analysis lenses:

- relevant reference components
- current-project mapping targets
- safe-to-transfer ideas
- direct-copy risks
- next reading or tracing step

This mode should be compatible with `code-flow-guide` and `wiki-ingest` when deeper source understanding is needed.

### 7.4 generic-demand

Default analysis lenses:

- target outcome
- hidden assumptions
- constraints
- success criteria
- non-goals

## 8. Discussion Output Contract

Each normal `/inspire` response should be structured around five elements:

1. current understanding of the user's request
2. analysis lenses being applied
3. possible causes, interpretations, or solution directions
4. missing evidence, ambiguity, or unresolved constraints
5. one focused follow-up question or a small set of next-step options

The conversation should avoid:

- multi-step implementation proposals presented as if they are already approved
- jumping directly to file edits or patch suggestions
- bloated questionnaires with many unrelated questions at once
- pretending uncertainty is resolved when evidence is still missing

`inspire` should also make the "multiple-angle analysis" visible in the response rather than silently collapsing to one interpretation. The user should be able to see that the agent considered different lenses and narrowed them deliberately.

## 9. Design Generation Contract

When the user explicitly chooses `生成设计`, `inspire` should transition into a design-writing phase governed by the `brainstorming` skill while honoring this spec's overrides.

### 9.1 Brainstorming Alignment

The generated design should preserve the useful parts of the `brainstorming` structure:

- clear problem framing
- goals and non-goals
- constraints
- alternative approaches with trade-offs
- recommended design
- testing or acceptance expectations
- explicit pause for review

### 9.2 Inspire Overrides

The design phase must additionally enforce:

- the workflow explicitly uses `brainstorming` as the design-generation parent process
- no standalone implementation plan document
- no automatic todo writing immediately after design
- required subagent review before the design is considered ready for user review
- explicit stop after design review until the user chooses the next action

### 9.3 Required Design Sections

The design spec generated by `inspire` must include:

1. problem statement
2. goals
3. non-goals
4. user requirements captured from discussion
5. primary use cases
6. workflow overview
7. trigger and session contract
8. input classification and analysis modes
9. design-generation contract
10. primary-agent and subagent responsibilities
11. todo-first planning contract
12. testing and acceptance indicators
13. requirement coverage checklist
14. open questions, if any

The exact headings may vary slightly by topic, but the content must still be present.

## 10. Design Review Contract

After each design draft, the primary agent must dispatch a subagent reviewer.

### 10.1 Review Focus

The review must focus on:

- whether previously discussed requirement points were actually captured
- whether test and acceptance indicators were actually captured
- whether the design introduces ambiguity that could cause the wrong implementation

The review should not expand into a broad stylistic critique.

### 10.2 Required Review Output

The reviewing subagent must return:

- `已落实的需求点`
- `缺失或不清楚的需求点`
- `已落实的测试/验收指标`
- `缺失或不清楚的测试/验收指标`
- `建议修改项`
- `是否建议通过本轮 design`

### 10.3 Post-Review Behavior

After receiving the review:

- the primary agent summarizes the findings
- updates the design if needed
- returns the design path result to the user
- stops and waits for the user's next explicit choice

If the primary agent makes a substantive design change that affects requirements, transitions, responsibilities, or acceptance indicators after a subagent review, the updated design must be reviewed again before it is presented as the ready-for-user-review version. Pure editorial cleanup that does not change meaning does not require another review pass.
The official design-stage log should be written after the review result is integrated and before the primary agent presents the design as ready for user review.
The primary agent should also synchronize design-state memory in `notes/todo.md` and the relevant branch page so the repository remembers that a design exists even before implementation todo leaves are created.

## 11. Todo-First Planning Contract

`inspire` must not create a separate implementation plan document.

If the user wants to move from design into executable work breakdown, the primary agent should write todo nodes directly into the existing notes system.

### 11.1 No Standalone Plan Rule

Disallowed output:

- `docs/.../plans/...`
- implementation-plan markdown created as a separate artifact

Required output:

- updates to `notes/todo.md`
- updates to the relevant branch page under `notes/todo/`
- an official todo-stage log under `notes/log/`
- task decomposition expressed as root/child/leaf nodes

### 11.2 Todo Content Requirements

Each important todo leaf should capture:

- what the leaf is solving
- which design section it maps to
- which acceptance or test indicators apply
- whether the leaf depends on another leaf
- current status

The primary agent should treat `notes/todo.md` as the dashboard and the branch page as the durable problem memory. Todo writing is not complete if only one of those surfaces is updated.

## 12. Primary-Agent And Subagent Responsibilities

### 12.1 Primary Agent

The primary agent owns:

- conversation control
- approval gates
- requirement convergence
- design synthesis
- design-review integration
- todo writing and state maintenance
- official `notes/log` writing
- review of subagent implementation results
- creation of follow-up nodes when results are strange or incomplete

This includes official logs for:

- design-stage completion after review integration
- todo-stage completion after todo writing
- implementation and verification stages after subagent execution

The primary agent does **not** directly edit code during the execution phase governed by this workflow.

### 12.2 Subagents

Subagents own:

- bounded implementation tasks
- code edits
- test execution
- verification runs
- structured result reporting

Subagents do **not** own the official project memory as the default writer. They report back; the primary agent writes the official todo/log updates.

### 12.3 Strange-Metric Escalation

If a subagent returns metrics or outcomes that look strange, inconsistent, or contradictory:

1. the primary agent records the anomaly in todo/log memory
2. creates a concrete follow-up leaf
3. dispatches another subagent
4. continues orchestration without directly editing code

## 13. Proposed Skill Package Layout

```text
.agents/skills/inspire/
├── SKILL.md
└── references/
    ├── intake-modes.md
    ├── analysis-lenses.md
    ├── design-template.md
    ├── design-review-checklist.md
    ├── todo-write-contract.md
    └── delegation-contract.md
```

### 13.1 SKILL.md

Should contain:

- the trigger rule
- the session-freeze rule
- the main workflow state machine
- approval gates
- role boundaries between primary agent and subagents

### 13.2 references/intake-modes.md

Should define the four intake modes and the first-round analysis pattern for each.

### 13.3 references/analysis-lenses.md

Should define reusable thinking lenses such as:

- visible facts
- likely cause
- missing evidence
- constraints
- risks
- acceptance targets

### 13.4 references/design-template.md

Should define the design skeleton that stays close to `brainstorming` but uses the `inspire` overrides from this spec.

### 13.5 references/design-review-checklist.md

Should contain the mandatory review questions for:

- requirement coverage
- acceptance-metric coverage
- ambiguity

### 13.6 references/todo-write-contract.md

Should define how approved designs turn into todo trees rather than plan docs.

### 13.7 references/delegation-contract.md

Should define the structured report format expected back from subagents after implementation and testing.

## 14. Testing And Acceptance Indicators

The later implementation should be considered acceptable only if it can demonstrate the following behaviors.

### 14.1 Triggering

- `/inspire ...` enters inspire mode
- non-`/inspire` messages do not automatically activate the skill

### 14.2 Discussion Freeze

- before explicit approval, inspire mode does not edit code
- before explicit approval, inspire mode does not write todo
- before explicit approval, inspire mode does not generate a design file
- directional but unapproved phrases such as "应该这样做" or "就改成这样" do not count as workflow approval

### 14.3 Input Handling

- image-analysis requests are handled with observation/cause/evidence-gap framing
- bug-log requests are handled with diagnosis-before-fix framing
- reference-mapping requests are handled with reference-to-project mapping framing
- generic-demand requests are handled with goal/constraint/success framing
- each discussion response shows multiple analysis lenses rather than a single unexplained conclusion
- each discussion response ends with one focused follow-up question or a small next-step choice set

### 14.4 Design Path

- when the user approves `生成设计`, the agent explicitly enters the `brainstorming` workflow and writes a design doc under that workflow
- while still in discussion mode, the available next-step choices remain `继续分析 / 生成设计 / 结束`; `写 todo` does not appear until after the design path has completed
- the design doc includes explicit requirement capture and test/acceptance indicators
- a subagent review is run after the design draft
- the design review output uses the fixed six-section Chinese format defined in [10.2](#102-required-review-output)
- the design review stays narrowly focused on requirement coverage, acceptance-indicator coverage, and implementation-critical ambiguity, without broad style critique
- after design review, the workflow stops until the user explicitly chooses the next action
- if the design is substantively changed after review, another review is run before the design is treated as ready for user review
- after review integration, the primary agent syncs design-state memory in `notes/todo.md` and the relevant branch page
- the primary agent writes an official design-stage log after review integration

### 14.5 Todo-First Planning

- when the user later approves `写 todo`, the agent updates todo memory instead of creating a standalone plan doc
- todo writing updates both `notes/todo.md` and the relevant branch page under `notes/todo/`
- each important todo leaf records the mapped design section, applicable acceptance indicators, dependencies, and status
- the primary agent writes an official todo-stage log after todo writing

### 14.6 Execution Orchestration

- implementation begins only after the user explicitly approves work on a specific todo leaf
- implementation leaves are delegated to subagents
- the primary agent does not directly edit code during the execution phase
- subagents return structured results
- the primary agent writes the official log
- strange metrics create follow-up leaves and another delegation cycle

## 15. Requirement Coverage Checklist

| Requirement From Discussion | Section |
| --- | --- |
| manual `/inspire` trigger only | [6.1](#61-trigger-rule), [14.1](#141-triggering) |
| discuss before code edits | [5.1](#51-main-discussion-path), [6.2](#62-session-freeze-rule), [14.2](#142-discussion-freeze) |
| support image / bug-log / reference / general requirement inputs | [4](#4-primary-use-cases), [7](#7-input-classification), [14.3](#143-input-handling) |
| direct user tone must not imply approval | [6.3](#63-approval-rule) |
| explicit approval needed for design, todo, implementation | [5](#5-workflow-overview), [6.2](#62-session-freeze-rule) |
| multiple-angle analysis + focused follow-up + deepen shared understanding | [5.1](#51-main-discussion-path), [7](#7-input-classification), [8](#8-discussion-output-contract), [14.3](#143-input-handling) |
| design follows brainstorming style | [9.1](#91-brainstorming-alignment) |
| design reviewed by subagent against requirement points and test indicators | [10](#10-design-review-contract), [14.4](#144-design-path) |
| no standalone plan doc | [11](#11-todo-first-planning-contract), [14.5](#145-todo-first-planning) |
| write todo after design when approved | [5.3](#53-todo-path), [11](#11-todo-first-planning-contract) |
| integrate approved downstream actions with todo dashboard, branch page, and official logs | [11](#11-todo-first-planning-contract), [12.1](#121-primary-agent), [14.5](#145-todo-first-planning), [14.6](#146-execution-orchestration) |
| implementation and testing done by subagents | [5.4](#54-execution-path), [12.2](#122-subagents), [14.6](#146-execution-orchestration) |
| primary agent maintains workflow, todo, log, and review | [12.1](#121-primary-agent), [14.6](#146-execution-orchestration) |
| strange metrics trigger follow-up delegation, not direct primary-agent edits | [12.3](#123-strange-metric-escalation), [14.6](#146-execution-orchestration) |

## 16. Open Questions

There are no blocking open questions for the first implementation pass of this skill. Future revisions can decide whether the skill should gain helper scripts, packaged review prompts, or reusable todo-writing templates, but those are not required to begin implementation.
