# MPC Policy Gait Period Probe

## Purpose

Test the user's hypothesis that the trained policy's foot swing/contact cycle is faster than the MPC reference cycle. The suspected policy cycle was about `10-15` frames while MPC visually appeared to use a `25` frame swing/reference horizon.

## Stage

`mpc_policy_eval.py` tracking mode / MPC reference cache / policy-vs-reference gait timing.

## Related Todo

- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Procedure

A temporary read-only probe was created under `Go2Pvcnn/scripts/_tmp_mpc_gait_period_probe.py`, run once, then deleted. No production code was kept.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/_tmp_mpc_gait_period_probe.py \
  --mode tracking \
  --headless \
  --device cuda:0 \
  --num-envs 4 \
  --num-rounds 1 \
  --max-steps 120 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.4 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/gait_period_probe
```

The probe recorded per-step actual foot positions/contact, current MPC reference foot positions/contact, every-horizon best matching frame, and period estimates from contact transitions and foot-height autocorrelation.

## Output

- [../../logs/mpc_policy_eval/gait_period_probe/2026-06-16_19-10-51-566363/summary.json](../../logs/mpc_policy_eval/gait_period_probe/2026-06-16_19-10-51-566363/summary.json)
- [../../logs/mpc_policy_eval/gait_period_probe/2026-06-16_19-10-51-566363/period_rows.jsonl](../../logs/mpc_policy_eval/gait_period_probe/2026-06-16_19-10-51-566363/period_rows.jsonl)

## Key Metrics

- Exit code `0`.
- Steps: `120`, envs: `4`, fixed command `[0.4, 0.0, 0.0]`.
- Current-frame foot error mean: `0.09982m`.
- Best-horizon-frame foot error mean: `0.03282m`.
- Frame-0 foot error mean: `0.03980m`.
- Contact mismatch against current reference: `0.5000`.
- Actual foot z minus current reference z: `-0.03883m`.
- Current reference period estimates:
  - foot-z autocorrelation peak: `25` frames for all four legs.
  - contact same-transition period: about `25` frames for all four legs.
- MPC cache contact switch interval:
  - legs `0/3`: about `11.5` frames.
  - legs `1/2`: about `12.0` frames.
- Policy actual contact/height estimates:
  - rear-left contact change interval: `12.0` frames, same-transition period `23.0` frames.
  - rear-right contact change interval: `8.98` frames, same-transition period `17.17` frames.
  - front legs had no reliable contact transitions in this short rollout; foot-z autocorrelation peaks were noisy (`5-19` frames).
- Best-frame histogram over `480` env-step samples:
  - frame `0`: `306`
  - frame `1`: `79`
  - frame `2`: `38`
  - later frames are sparse, with small clusters around `9-17` and `24`.

## Result

Diagnostic pass. The MPC current reference clearly has a `25` frame cycle when read through `current_frame_ids()`. The MPC cache itself switches contact about every `11.5-12` frames, which is the expected half-cycle inside a `25` frame gait/reference cycle.

The policy does show faster landing/contact behavior on the rear legs, with contact-change intervals in the `9-12` frame range, but this is better interpreted as a half-cycle / early-contact mismatch rather than proof that the full policy gait cycle is only `10-15` frames.

The stronger signal remains phase/reference consumption mismatch: actual feet match early horizon frames far better than the current frame, and actual feet are lower than the current reference by about `3.9cm`.

## Conclusion

The user's intuition is partly right: the policy's realized contact timing is earlier/faster than the current MPC frame being compared against, especially in landing. But the data does not support simply saying "policy full swing cycle is 10-15 frames while MPC is 25 frames." More precise:

- MPC current reference full cycle is `25` frames.
- MPC contact half-cycle is about `12` frames.
- Policy realized rear-leg contact changes are around `9-12` frames and lower than reference.
- The dominant mismatch is still that eval/reward current-frame consumption advances through the cache while the policy behavior remains closest to frame `0-2`.

## Follow-Up

- Test a phase-offset reward/eval diagnostic without changing training behavior: compare current frame, previous frame, frame `0`, and best-frame metrics.
- Inspect whether old `model_14000.pt` was trained under a different phase-consumption convention.
- Do not tune MPC losses from this result alone.

## Git Refs

- Baseline Ref: current working tree on `costmap-teacher-ablation`
- Candidate Ref: no production code change; temporary probe deleted after run
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)
