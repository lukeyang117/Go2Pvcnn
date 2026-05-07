# Inspire Design Template

Use this only after the user explicitly chooses `生成设计`. The primary agent must explicitly say the workflow is entering `brainstorming`, actually use the `brainstorming` workflow, then apply this template with the inspire-specific overrides below.

## Required Overrides

- discussion-first behavior remains the gate into this design
- no standalone implementation plan document
- no automatic todo writing after design
- required subagent review before the design is ready for user review
- explicit stop after design review until the user chooses the next action

## Suggested Structure

```markdown
# <Topic>

## 1. Problem Statement

## 2. Goals

## 3. Non-Goals

## 4. User Requirements Captured From Discussion

## 5. Primary Use Cases

## 6. Workflow Overview

## 7. Trigger And Session Contract

## 8. Input Classification And Analysis Modes

## 9. Design-Generation Contract

## 10. Primary-Agent And Subagent Responsibilities

## 11. Todo-First Planning Contract

## 12. Testing And Acceptance Indicators

## 13. Requirement Coverage Checklist

## 14. Open Questions
```

## Brainstorming Alignment

The design must include these required contents from the `brainstorming` path:

- clear problem framing
- goals and non-goals
- constraints
- alternative approaches with trade-offs
- recommended design
- testing or acceptance expectations
- explicit pause for review

The following are hard-required and must not be omitted or treated as optional guidance:

- constraints
- alternative approaches with trade-offs
- recommended design

## Review And Stop Rules

Before presenting the design as ready:

1. run the required subagent review
2. integrate review findings
3. re-run review if a substantive change alters requirements, transitions, responsibilities, or acceptance indicators
4. sync design-state todo memory and the official design-stage log
5. stop and wait for the user

After this path, offer only:

- `修改 design`
- `写 todo`
- `结束`
