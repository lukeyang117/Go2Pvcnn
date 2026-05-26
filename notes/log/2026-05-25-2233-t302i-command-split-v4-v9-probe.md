# T302i Command-Split V4-V9 Probe

## Purpose

Continue the user's requested loss-only optimization for the three command cases:

- pure translation: approach to a suitable distance, then FK-realized foot-over;
- mixed translation + yaw: follow translation direction while allowing yaw;
- pure yaw: no crossing pressure, only no-contact/stability/reachability/continuity.

## Stage

- `extension/batch_mpc_planner`
- probe-only variants in `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`

## Related Todo

- [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Commands

```bash
pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
python -m py_compile Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py

CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'diagonal_v050:0.35 0.35 0.00,mixed_yaw_v050:0.50 0.25 1.00,yaw100:0.00 0.00 1.00' --variants baseline,reachable_fk_cross_v4 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_focused_fk_cross_v4.jsonl 2>&1

CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'diagonal_v050:0.35 0.35 0.00,mixed_yaw_v050:0.50 0.25 1.00,yaw100:0.00 0.00 1.00' --variants baseline,reachable_fk_cross_v5 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_focused_fk_cross_v5.jsonl 2>&1

CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'diagonal_v050:0.35 0.35 0.00,mixed_yaw_v050:0.50 0.25 1.00,yaw100:0.00 0.00 1.00' --variants baseline,reachable_fk_cross_v6 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_focused_fk_cross_v6.jsonl 2>&1

CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'mixed_yaw_v050:0.50 0.25 1.00' --variants baseline,reachable_fk_cross_v7 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_mixed_yaw_fk_cross_v7.jsonl 2>&1

CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'mixed_yaw_v050:0.50 0.25 1.00' --variants baseline,reachable_fk_cross_v8 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_mixed_yaw_fk_cross_v8.jsonl 2>&1

CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'mixed_yaw_v050:0.50 0.25 1.00' --variants baseline,reachable_fk_cross_v9 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_mixed_yaw_fk_cross_v9.jsonl 2>&1
```

## Key Metrics

Local checks:

- `pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py` -> `25 passed`
- `py_compile` -> exit `0`

Focused three-case findings:

- v4 split extra loss only, but original low-small losses still affected pure yaw/mixed yaw.
  - diagonal v4 still had penetration/contact `0.020000/0.010000`;
  - mixed-yaw v4 direction remained poor (`cos=0.055557`, progress `0.035727`);
  - pure yaw v4 step/accel worsened (`14.337611/14.404465`).
- v5 added command-specific cfg, but still inherited too much `reachable_loss_small_v1`.
  - diagonal and mixed-yaw contacts worsened;
  - pure yaw step/accel became `19.957234/19.582006`.
- v6 made pure yaw baseline-like with crossing/foot-over off.
  - pure yaw penetration cleared (`0.0075 -> 0`) and accel stayed similar (`6.297041 -> 6.291710`), but step still worsened (`7.813674 -> 10.040354`);
  - mixed-yaw remained weak (`cos=0.085962`, drift `0.563904`).
- v7 added explicit mixed-yaw command-direction cosine/progress loss.
  - mixed-yaw direction recovered strongly: `cos -0.595472 -> 0.999811`, progress `-0.296376 -> 1.477873`, drift `0.399854 -> 0.028722`, contact `0`;
  - but root height collapsed: `0.289896 -> 0.099448`.
- v8 added mixed-yaw root-height/posture guard.
  - direction stayed clean (`cos=0.999926`, drift `0.014845`) and root height improved to `0.122849`;
  - but planned-vs-FK/touchdown regressed (`0.430775/0.785582`).
- v9 added mixed-yaw reachability barrier.
  - root height improved to `0.147039`, planned-vs-FK `0.388287 -> 0.349839`, touchdown `0.661772 -> 0.563632`, swing step `12.329788 -> 4.355378`, contact `0`;
  - but lateral drift regressed to `0.473070` and direction cosine dropped to `0.897310`.

## Result

Partial pass as diagnosis; no production-ready variant yet.

The investigation now isolates the mixed-yaw tradeoff:

- v7 solves translation direction but uses low base.
- v8 solves direction plus part of posture but loses reachability.
- v9 improves reachability and posture but loses lateral path tightness.

Pure yaw also confirms it should not inherit the low-small crossing cfg: baseline-like cfg with crossing off is closer, but still needs continuity guarded separately.

## Conclusion

The next loss-only direction should not keep stacking global weights. It should combine the three working signals with command-specific gates:

1. mixed-yaw command-direction cosine/progress from v7;
2. mixed-yaw root-height/posture guard from v8, but with softer interaction so it does not force unreachable feet;
3. mixed-yaw reachability barrier from v9;
4. pure-yaw baseline-like cfg with crossing/foot-over disabled and no extra approach/cross loss.

If the next variant still cannot keep direction, posture, reachability, and no-contact together, this is evidence that loss-only is hitting a structural limit and the planner output/target generation contract should be revisited.

## Git Refs

- Baseline Ref: working tree before probe-only v4-v9
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
