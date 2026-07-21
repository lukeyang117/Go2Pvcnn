# Joint MPC RTI Ranked Flat After Validity Fix

## Purpose

Re-evaluate the three ranked flat cells after correcting fixture geometry and accepted-candidate validity propagation.

## Stage

Task 13 ranked CUDA diagnosis. This is a failed behavior gate, not acceptance evidence.

## Command

The acceptance runner executed on `cuda:1` under `run_monitored_joint_mpc.py` with a 120-second timeout, 5-second heartbeat, `--stage flat --ranked-cells 3 --steps 24`, and report `/tmp/flat_report_after_validity.json`.

## Resource Result

- Completed in `10.533s`; monitor reason `completed`.
- Peak process-tree RSS about `1.53GiB`.
- Selected-GPU task memory about `2.09GiB`.
- No watchdog trigger and no residual task process.

## Key Metrics

- Zero `(0,0,0)`: stance slip `1.72e-6m`, alpha-zero ratio `0`, but root drift `1.24e-4m` and swing clearance `-2.27e-4m` fail.
- Forward `(1,0.5,1)`: stance slip `7.32e-5m` and joint margin `0.322rad` pass; alpha-zero ratio `0.214`, root leak before foot `0.266m`, lead time `0ms`, and swing clearance `-6.70mm` fail.
- Backward `(-1,-0.5,-1)`: joint safe margin reaches `0`, alpha-zero ratio `0.348` with run `8`, joint step `0.394rad`, stance slip `50.9mm`, stance gap `69.3mm`, root leak `19.7mm`, yaw-rate error `0.294rad/s`, swing clearance `-64.6mm`, and active-motion ratio `0.245` fail.
- All three cells are finite and complete enough to expose the failures; the prior nominal-validity masking is no longer the primary diagnostic blocker.

## Conclusion

The next root-cause target is the first backward frame where joint margin collapses and line search begins its eight-frame nominal fallback run. Zero drift and foot/root temporal ordering remain separate issues. No parameter was changed by this run.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: working tree on `724a1c3`
- Key files: `planner.py`, `run_joint_acceptance.py`, `joint_metrics.py`, `config.py`
