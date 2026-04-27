# TestPvcnnWithIsaacsim Agent Rules

## Project Focus

Isaac Lab + Go2 + PVCNN 训练/播放主线，以及 extension planner 与轨迹奖励相关的对齐、验证和知识沉淀。

Primary code entry points:

- `Go2Pvcnn/scripts/train.py`
- `Go2Pvcnn/scripts/play.py`
- `Go2Pvcnn/go2_pvcnn/`
- `Go2Pvcnn/extension/`

Primary notes entry points:

- `notes/index.md`
- `notes/todo.md`
- `notes/todo/`
- `notes/log/index.md`

---

## Hard Constraints

- Every new conversation in this repository must begin by checking `notes/index.md` first.
- Treat the documented active architecture as the default working architecture unless the user explicitly asks to work on legacy or experimental paths.
- `raw/` and `onlyReference/` are reference areas by default, not the active implementation target, unless the user explicitly says otherwise.
- Do not silently route new work into legacy paths.
- Any code change must first be understood in the context of the stage, module, or workflow it belongs to.
- Do not make cross-stage or cross-module edits casually. Modify the smallest coherent part of the system and its direct upstream/downstream contracts.
- Do not claim a fix for training, simulation, observation wiring, curriculum logic, LiDAR, PVCNN integration, planner alignment, or reward behavior without stating what was verified.
- Do not claim completion without checking whether code behavior, verification, and notes are aligned.
- Keep `notes/todo.md` as a dashboard. Use branch pages for detailed memory and logs for evidence.
- When `notes/todo.md`, branch pages, or `notes/log/index.md` grow too large, use `$compact-todo` if available. The system must remain usable without it.
- All note links must remain repository-relative so they work both in the server workspace and the local `Z:` Obsidian mount.

---

## Mandatory Pre-Read Before Work

Before every analysis session, debugging session, code reading pass, code change, test run, or final completion claim, read:

1. `notes/todo.md`
2. relevant branch pages under `notes/todo/` for active/open nodes
3. relevant logs from `notes/log/index.md`
4. `notes/index.md`
5. relevant files under `notes/ai/` or `notes/human/` when the task touches documented stages, contracts, or user-facing behavior

If the task involves `Go2Pvcnn/extension/planner`, planner-based trajectory rewards, or `raw/kinematic_footsteps` sync, also read:

1. `notes/human/human-08-extension-planner-reading-guide.md`
2. `notes/human/human-09-extension-planner-mapping.md`
3. the relevant `raw/kinematic_footsteps/notes` documents before editing planner core logic

The purpose is to continue the current investigation instead of restarting from scratch.

---

## Required Change Workflow

For every meaningful investigation update, test, code change, or behavior change:

1. Read the mandatory notes.
2. Identify the affected stage, module, or workflow.
3. Confirm current input, method, output, and downstream consumers before editing.
4. Modify only the smallest coherent scope.
5. Run or describe appropriate verification.
6. Update `notes/todo.md` when focus, active fronts, open leaves, or recent logs changed.
7. Update the relevant branch page under `notes/todo/`.
8. Update `notes/log/index.md`.
9. Create one per-test or per-verification log under `notes/log/`.
10. Align `notes/ai/` and `notes/human/` when stage behavior, contracts, inputs, or outputs changed.
11. Report what changed in code, verification, and notes.

---

## Todo And Test Log Rule

Required files:

- `notes/todo.md`
- `notes/todo/`
- `notes/todo/archive/`
- `notes/log/index.md`
- `notes/log/archive/`
- one per-test Markdown file under `notes/log/`

### Root Dashboard Requirement

`notes/todo.md` is a dashboard, not a full database.

It must keep:

- `Start Here`
- `Active Fronts`
- `Root Map`
- `Open Leaves`
- `Branch Pages`
- `Recent Logs`

It must not expand every closed child node. Closed detail belongs in branch pages or archives.

### Tree-Structured Todo Requirement

Agents must maintain root problem nodes, child problem nodes, explicit relationships, and links from important nodes into branch pages.

For each new issue, classify it as:

- new root node
- child of an existing node
- sibling of an existing node
- related by cause, blocking, or dependency

Do not append isolated flat todos.

### Dynamic-Issue Discovery Requirement

If a new issue is discovered during testing, debugging, or code reading:

1. create a node in `notes/todo.md` if it is open/active, or in the relevant branch page if closed/immediate context only
2. attach it to the correct parent/sibling/related position
3. explain why it was created in the branch page
4. link the triggering test log

New issues must not live only inside a test log.

### Metric-Anomaly Expansion Requirement

If metrics look strange, conclusions look strange, or results conflict with expectations, create a concrete follow-up node. Avoid vague placeholders.

### Branch Memory Requirement

Important or evolving issues must have branch pages under `notes/todo/`.

Branch pages should include:

- `Current State`
- `Open Children`
- `Closed Children Archive`
- `Related Logs`
- `Git Refs`
- `Next Step`
- `Node Details`

### Log Requirement

Every distinct test or verification pass must create a Markdown file under `notes/log/`.

Each log records:

- purpose
- stage
- related todo
- command/procedure
- input conditions
- key metrics
- result
- conclusion
- follow-up
- git refs

### Git Traceability Requirement

Track:

- `Last Feature Commit`
- `Last Verified Commit`
- `Current Work Ref`
- `Key Files`

For logs, track:

- `Baseline Ref`
- `Candidate Ref`
- `Key Files`

Do not overwrite module refs merely because newer unrelated commits exist.

### Relative-Link Requirement

All links must be relative.

---

## Final Response Rule

When closing a task, include:

- which stage/module/workflow changed
- what contracts changed, if any
- what metrics or constraints were checked
- whether todo/log/notes were aligned
- what remains unverified
