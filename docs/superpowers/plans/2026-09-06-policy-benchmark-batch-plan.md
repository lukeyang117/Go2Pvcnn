# Policy Benchmark Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run multiple policy benchmark suites sequentially under one timestamped output directory while cleaning each suite's process group and preventing hangs.

**Architecture:** Add a batch shell launcher that creates one batch directory, invokes the existing single-suite launcher once per suite with the shared `--output-dir`, and owns a per-suite process group. Each suite has a timeout and TERM/KILL cleanup; a trap cleans the active group on interruption. Existing `run_policy_benchmark.sh` behavior remains unchanged.

**Tech Stack:** Bash, GNU `setsid`/`timeout`, existing Python IsaacLab benchmark runner, pytest static shell-contract tests.

## Global Constraints

- Suites run sequentially, never concurrently.
- A suite failure stops later suites and preserves its partial output.
- Cleanup targets only the process group created for the current suite.
- The batch directory is generated once from timestamp, experiment type, checkpoint name, and hash.
- Existing single-suite invocation remains backward compatible.

### Task 1: Add Batch Launcher

**Files:**
- Create: `scripts/run_policy_benchmark_batch.sh`

**Interfaces:**
- Inputs: `--experiment-type`, `--checkpoint`, optional `--suites` (comma-separated), `--suite-timeout`, and all remaining benchmark arguments.
- Outputs: one shared `eval_output/<timestamp>_<experiment>_<weight>_<hash>/` directory containing per-suite results; exit status of the first failed suite.

- [x] **Step 1: Implement argument validation and batch directory generation.**
- [x] **Step 2: Implement isolated process-group launch using `setsid`, timeout, and cleanup trap.**
- [x] **Step 3: Pass the shared directory to the existing `run_policy_benchmark.sh` for each suite.**
- [x] **Step 4: Add GPU-release polling with a bounded wait after cleanup.**
- [x] **Step 5: Run `bash -n scripts/run_policy_benchmark_batch.sh`.**

### Task 2: Add Static Contract Tests

**Files:**
- Modify: `tests/evaluation/test_policy_benchmark_static.py`

- [x] **Step 1: Test that the batch launcher exists and validates required checkpoint/experiment arguments.**
- [x] **Step 2: Test that the launcher creates one batch directory and forwards the same output directory to every suite.**
- [x] **Step 3: Test that timeout, process-group cleanup, trap cleanup, and stop-on-failure contracts are present.**
- [x] **Step 4: Run the focused pytest module.**

### Task 3: Real Smoke Verification

**Files:**
- No source changes.

- [x] **Step 1: Run one batch with `complex_mixed,large_runway,small_runway`, 16 environments, and a short timeout.**
- [x] **Step 2: Verify all three suites share one output directory and each has a `summary.json`.**
- [x] **Step 3: Verify `nvidia-smi` shows no residual IsaacLab process after completion.**
