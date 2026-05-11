# Delegation Contract

Use this after the user explicitly approves entering implementation after the todo stage. Once execution starts, do not require any further user approval.

## Role Boundary

- primary agent: orchestration, execution autonomy after stage entry, official notes/log updates, review, anomaly follow-up
- subagent: code edits, test execution, verification runs, structured result reporting

The primary agent must not directly edit code during this execution phase.

## Subagent Input Package

Give the subagent:

- the exact ready leaf or ready leaf batch and parent context
- the relevant design sections
- acceptance or test indicators for the leaf
- file and module boundaries
- any repository constraints that matter for the leaf
- the expected report format below

## Required Subagent Report

The subagent should return:

- `status`: `DONE`, `DONE_WITH_CONCERNS`, `NEEDS_CONTEXT`, or `BLOCKED`
- `summary`: what changed
- `files_touched`: code and test files changed
- `tests_or_checks`: commands run and high-level results
- `acceptance_coverage`: which leaf acceptance indicators were covered
- `open_concerns`: remaining risks, caveats, or missing verification
- `recommended_follow_up`: next action for the primary agent

## Continuation Rule

After a subagent returns, the primary agent should either:

1. continue with the next ready leaf or ready leaf batch, or
2. surface an objective blocker that prevents continued execution

Do not ask the user for any further approval during execution.

## Strange-Metric Escalation

If outcomes or metrics look strange, inconsistent, or contradictory:

1. the primary agent records the anomaly in official memory
2. creates a concrete follow-up leaf
3. dispatches another subagent
4. continues orchestration without editing code directly
