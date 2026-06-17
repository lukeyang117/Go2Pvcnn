# MPC Policy Phase Offset Probe

## Purpose

Test whether the old policy is best aligned with the MPC current frame, a fixed lag/lead relative to current frame, frame `0`, or a per-step best horizon frame.

## Stage

`mpc_policy_eval.py` tracking mode / MPC reference cache / policy-vs-reference phase convention.

## Related Todo

- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Procedure

A temporary read-only probe was created under `Go2Pvcnn/scripts/_tmp_mpc_phase_offset_probe.py`, run once, then deleted. No production code was kept.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/_tmp_mpc_phase_offset_probe.py \
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
  --output-dir logs/mpc_policy_eval/phase_offset_probe
```

The probe compared actual feet/contact against:

- relative frame offsets from current frame: `-12,-8,-4,-2,-1,0,+1,+2,+4,+8,+12`
- absolute horizon frames: `0..24`
- per-step best absolute horizon frame

## Output

- [../../logs/mpc_policy_eval/phase_offset_probe/2026-06-16_19-19-49-752254/summary.json](../../logs/mpc_policy_eval/phase_offset_probe/2026-06-16_19-19-49-752254/summary.json)
- [../../logs/mpc_policy_eval/phase_offset_probe/2026-06-16_19-19-49-752254/phase_offset_rows.jsonl](../../logs/mpc_policy_eval/phase_offset_probe/2026-06-16_19-19-49-752254/phase_offset_rows.jsonl)

## Key Metrics

- Exit code `0`.
- Steps: `120`, envs: `4`, fixed command `[0.4, 0.0, 0.0]`.
- Relative-offset foot-position error:
  - best relative offset: `current-12`
  - `current-12`: `0.06465m`
  - `current-8`: `0.06773m`
  - `current-4`: `0.07602m`
  - `current`: `0.09011m`
  - `current+12`: `0.13974m`
- Relative-offset contact mismatch:
  - best relative contact offset: `current-4` / `current-2`, both `0.4875`
  - `current`: `0.4974`
  - `current-12`: `0.5130`
- Absolute-frame foot-position error:
  - best absolute frame: frame `0`, `0.06477m`
  - frame `1`: `0.06584m`
  - frame `2`: `0.07143m`
  - frame `12`: `0.10149m`
  - frame `24`: `0.16198m`
- Absolute-frame contact mismatch:
  - best absolute frame: frame `1`, `0.2526`
  - frame `2`: `0.2969`
  - frames `3-12`: about `0.314-0.340`
  - frame `0`: `0.6729`
- Best absolute frame histogram over `480` env-step samples:
  - frame `0`: `224`
  - frame `1`: `57`
  - frame `2`: `71`
  - frames `15/16`: `32` each
  - frame `24`: `16`

## Result

Diagnostic pass. The old policy does not align best with the current MPC frame. For foot position, the best fixed relative correction is approximately `current-12`, which is close to the MPC contact half-cycle. For absolute horizon position, frame `0` remains the best single frame, with frames `1-2` close behind.

Contact tells a slightly different story: absolute frame `1` has the best contact match, while `current-2/current-4` are the best relative contact offsets. This means the position mismatch and contact mismatch are not solved by one exact scalar offset.

## Conclusion

The evidence supports a mixed phase mismatch:

- There is a strong half-cycle component: `current-12` greatly improves foot-position error versus current frame.
- There is also a strong replan-start component: absolute frames `0-2` dominate best-frame selection.
- Contact phase is closer to early absolute frame `1` or a small negative current offset, not exactly `current-12`.

So the next implementation candidate should not be "blindly shift everything by 12 frames" as a final fix. A safer next diagnostic is to expose phase-offset metrics in eval/reward analysis first, or test a narrowly scoped reward-frame convention change behind a config flag.

## Follow-Up

- Compare reward-consumed frame inside `rewards_reference.py` against eval-read frame to confirm whether training already used `current`, `previous`, or post-advance phase.
- If changing behavior, prefer a config-gated phase offset / phase convention switch and test against both old `model_14000.pt` and a short fresh warm-start.
- Do not tune MPC losses from this result alone.

## Git Refs

- Baseline Ref: current working tree on `costmap-teacher-ablation`
- Candidate Ref: no production code change; temporary probe deleted after run
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)
