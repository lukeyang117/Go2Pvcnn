# Joint MPC RTI Kinematic Task 10 RED

## Purpose

Prove the approved five-candidate state-trajectory line search is absent before replacement.

## Result

Focused collection failed because the old control/merit line-search module did not export the frozen three-filter contract and did not provide the new state-candidate API.

## Git Refs

- Baseline Ref: `0e04839`
- Candidate Ref: uncommitted Task 10 RED test
- Key Files: `Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py`

## Follow-Up

Replace the old improving-only control search with exactly five state candidates, three filters, and seven-loss-only selection.
