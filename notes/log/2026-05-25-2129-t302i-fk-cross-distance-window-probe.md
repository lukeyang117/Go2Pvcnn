# T302i FK Cross Distance Window Probe

## Purpose

Test the user's proposed loss behavior: approach to a suitable command-frame distance before crossing, then require the FK-realized foot to pass over the low-small obstacle without lowering the base or spreading around it.

## Stage

- `extension/batch_mpc_planner`
- probe-only loss injection in `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`

## Related Todo

- [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Commands

```bash
pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
python -m py_compile Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py

CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,reachable_fk_cross_v1 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_fk_cross_v1.jsonl 2>&1

CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,reachable_fk_cross_v2 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_fk_cross_v2.jsonl 2>&1

CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,reachable_fk_cross_v3 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_fk_cross_v3.jsonl 2>&1

CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --variants baseline,reachable_fk_cross_v3 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_all_direction_fk_cross_v3.jsonl 2>&1
```

## Key Metrics

Local checks:

- `pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py` -> `17 passed`
- `py_compile` -> exit `0`

Forward-only:

| Variant | planned-vs-FK | touchdown IK/FK | raw IK violation | swing step | swing accel | lateral drift | root height | contact/penetration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | `0.323130` | `0.682283` | `2.470606` | `20.058457` | `13.486868` | `0.078791` | `0.142901` | `0` |
| reachable_fk_cross_v1 | `0.335894` | `0.631323` | `0.871720` | `8.783737` | `9.599930` | `0.442013` | `0.076920` | `0` |
| reachable_fk_cross_v2 | `0.330634` | `0.620743` | `0.837800` | `8.551348` | `9.657334` | `0.436312` | `0.139652` | `0` |
| reachable_fk_cross_v3 | `0.319953` | `0.599546` | `0.837800` | `8.930955` | `11.705757` | `0.328989` | `0.208851` | `0` |

All-direction `reachable_fk_cross_v3`:

- Summary max planned-vs-FK remained `0.452422` because baseline rows are included; candidate rows improved many commands but did not pass all-direction gates.
- `lateral_v050`: planned-vs-FK `0.428282 -> 0.268152`, touchdown `0.656021 -> 0.564321`, drift `0.203820 -> 0.114864`, no contact.
- `diagonal_v050`: planned-vs-FK `0.452422 -> 0.300097`, touchdown `1.063463 -> 0.809546`, but small penetration/contact appeared (`penetration=0.020833`, stance `0.011667`, touchdown `0.020833`) and drift worsened `0.109287 -> 0.421675`.
- `mixed_yaw_v050`: planned-vs-FK `0.388287 -> 0.326888`, swing step `12.329788 -> 7.503036`, but direction remained bad (`cos -0.595472 -> 0.107891`, progress `-0.296376 -> 0.054758`) and root height dropped to `0.106747`.
- `yaw100`: crossing not required; v3 disabled foot-over (`0`) but swing continuity regressed (`step 9.862889 -> 18.807239`, accel `9.516147 -> 19.045691`).

## Result

Partial pass as diagnosis, rejected as a fix.

The distance-window + FK-realized crossing idea is useful: it improved forward reachability, touchdown consistency, and swing continuity while preventing the v1 low-base shortcut by v3. It also improved lateral planned-vs-FK substantially.

It is not production-ready because all-direction behavior still fails:

- diagonal can touch/penetrate small obstacles;
- mixed yaw still does not track the commanded translation direction;
- pure yaw should not be affected by crossing pressure, but continuity regressed;
- forward/mixed lateral drift remains too high for the user's "track direction, not speed magnitude" requirement.

## Conclusion

The next loss-only slice should split gates more carefully:

1. Pure yaw must bypass low-small foot-over/crossing pressure entirely and keep only no-contact/stability/continuity.
2. Mixed translation+yaw needs command-frame heading and lateral-path handling that preserves translation direction while allowing yaw.
3. Crossing credit should remain FK-realized and posture-gated, but diagonal contact requires a stronger no-small-contact gate tied to the same FK foot path.

## Git Refs

- Baseline Ref: working tree before probe-only `reachable_fk_cross_v*`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
