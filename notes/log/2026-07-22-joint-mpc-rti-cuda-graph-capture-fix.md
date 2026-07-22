# Joint MPC RTI CUDA Graph Capture Fix

## Scope

- Worktree: `testPvcnnWithIsaacsim-joint-mpc-kinematic`
- Branch: `work/joint-mpc-kinematic`
- Runtime: pure-kinematic H30, one SQP/RTI and five-alpha line search
- Behavior formulas, loss weights, and constraint definitions were not changed.

## Failure

The local viewer failed while constructing `JointMpcCudaGraphRunner` because execution-path `new_tensor(...)` allocations and `torch.linalg.solve` calls are not permitted during CUDA Graph capture in the installed PyTorch runtime.

## Fix

- Reused graph-safe cached constants through `constant_like(...)`.
- Routed fixed general CUDA systems through a fixed-size Triton partial-pivot solve.
- Materialized transposed solve inputs as contiguous tensors at the Triton boundary.
- Kept CPU solving on `torch.linalg.solve_ex(..., check_errors=False)`.
- Made the fifth line-search candidate an exact nominal copy. This preserves the mathematical `alpha=0` candidate when a degenerate solve produces a non-finite direction, avoiding IEEE `0 * NaN` contamination.
- Added a real GPU capture/replay regression and an explicit non-finite-direction nominal-fallback regression.

## Verification

- Focused CPU runtime and line search: `26 passed in 26.05s`.
- Broader focused suite before the nominal-copy fix: `117 passed, 1 failed`; the sole failing warm reinjection case is covered by the final focused pass.
- CUDA scan compile plus complete RTI graph capture/replay: `2 passed in 141.41s` on physical GPU 1.
- Earlier isolated graph capture/replay: `1 passed in 5.85s`.
- `py_compile`: pass.
- `git diff --check`: pass.

## Boundary

This closes the reported CUDA Graph startup exception. It does not close the separate Task 14E terrain behavior gate, and no full interactive Isaac viewer session was used as acceptance in this fix.
