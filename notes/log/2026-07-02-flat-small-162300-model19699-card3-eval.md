# 2026-07-02 Flat-Small 16:23 Model19699 Card3 Eval

## Purpose

Read the user-run controlled-crossing result for the `2026-07-01_16-23-00/model_19699.pt` checkpoint.

## Stage

Training metrics / controlled crossing / flat-small semantic avoidance.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

Read:

```text
logs/mpc_policy_eval/flat_small_20260630_120028_model17400_controlled_crossing_card3/2026-07-02_09-46-13-151851/summary.json
logs/mpc_policy_eval/flat_small_20260630_120028_model17400_controlled_crossing_card3/2026-07-02_09-46-13-151851/rounds.jsonl
```

Important naming caveat: the output directory name says `20260630_120028_model17400`, but `config.json` records the actual checkpoint as:

```text
/mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-07-01_16-23-00/model_19699.pt
```

## Key Metrics

- `num_envs`: `16`
- `steps`: `300`
- `opportunity_env_count`: `16/16`
- `root_crossed_count`: `7/16`
- `foot_over_count`: `0/16`
- `small_contact_env_count`: `1/16`
- `small_overpass_success_count`: `0/16`
- `small_overpass_success_rate_over_opportunities`: `0.0`
- `touchdown_on_small_env_count`: `1/16`
- `reset_env_count`: `3/16`
- `reset_reason_counts`: `bad_orientation=3`, `base_contact=0`, `time_out=0`, `unknown=0`
- `reset_stage_counts`: all three resets were `before_foot_over`
- `command_body_match_max_abs_error`: `0.0`

Clearance stayed negative in observed foot-over checks; best max clearance was about `-0.0134m`.

## Result

Diagnostic. The checkpoint reduces direct small-object contact to `1/16`, but it still has no true measured foot-over or overpass success.

## Conclusion

This is not solved crossing behavior. It is still mostly a contact-avoidance / pass-without-swing-over behavior: the root can cross in some envs, but the measured swing-foot clearance over the small obstacle remains negative and `foot_over_count` stays `0`.

## Follow-Up

- Do not treat this as a successful overpass checkpoint.
- Next code/training direction should target making actual swing-foot lift over the path-small cell happen, not just reducing contact.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
