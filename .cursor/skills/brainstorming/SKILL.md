---
name: brainstorming
description: Explores user intent, requirements, and design before any creative work. Mandatory gate before creating features, building components, adding functionality, or modifying behavior. Use before implementation of any kind.
---

# Brainstorming Ideas Into Designs

Turn ideas into fully formed designs and specs through collaborative dialogue. Do NOT invoke implementation skills, write code, scaffold projects, or take implementation action until design is presented and approved.

## Anti-Pattern: "This Is Too Simple To Need A Design"

Every project goes through this process — todo lists, single-function utilities, config changes included. "Simple" projects are where unexamined assumptions cause the most wasted work. The design can be short (a few sentences for trivial cases), but you MUST present it and get approval.

## Mandatory Checklist

Complete in order. Create a task for each and track completion:

1. **Explore project context** — Check files, docs, recent commits
2. **Ask clarifying questions** — One at a time; understand purpose, constraints, success criteria
3. **Propose 2–3 approaches** — With trade-offs and your recommendation
4. **Present design** — In sections scaled to complexity; get user approval after each section
5. **Write design doc** — Save to `docs/plans/YYYY-MM-DD-<topic>-design.md` and commit
6. **Transition to implementation** — Invoke writing-plans skill to create implementation plan

**Terminal state:** Invoke writing-plans only. Do NOT invoke frontend-design, mcp-builder, or any other implementation skill.

## Process Flow

```
Explore context → Ask questions (one at a time) → Propose approaches → Present design
                                                                           ↓
Invoke writing-plans ← Write design doc ← User approves? ─── no ─→ Revise design
```

## The Process

### Understanding the idea

- Check project state first (files, docs, recent commits)
- Ask **one question at a time** to refine the idea
- Prefer multiple choice when possible
- Focus on: purpose, constraints, success criteria
- If a topic needs more exploration, break it into multiple questions

### Exploring approaches

- Propose 2–3 different approaches with trade-offs
- Lead with your recommended option and explain why
- Present options conversationally

### Presenting the design

- Scale each section to complexity: a few sentences if straightforward, up to 200–300 words if nuanced
- Ask after each section whether it looks right so far
- Cover: architecture, components, data flow, error handling, testing
- Be ready to revise if something doesn't make sense

### After the design

**Documentation:**

- Write the validated design to `docs/plans/YYYY-MM-DD-<topic>-design.md`
- Use elements-of-style:writing-clearly-and-concisely skill if available
- Commit the design document to git

**Implementation:**

- Invoke the writing-plans skill to create a detailed implementation plan
- Do NOT invoke any other skill; writing-plans is the next step

## Key Principles

| Principle | Application |
|-----------|-------------|
| One question at a time | Don't overwhelm; let the user answer fully |
| Multiple choice preferred | Easier to answer than open-ended when possible |
| YAGNI ruthlessly | Remove unnecessary features from all designs |
| Explore alternatives | Always propose 2–3 approaches before settling |
| Incremental validation | Present design, get approval before moving on |
| Be flexible | Go back and clarify when something doesn't make sense |
