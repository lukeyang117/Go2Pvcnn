# Joint MPC Small-Obstacle Collision Quantification

- Purpose: reproduce and quantify small-obstacle crossings without changing planner code.
- Stage: joint MPC RTI rolling x1 geometry against semantic-course native small shapes.
- Related todo: [T302v.4](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `50a7cf6`.
- Candidate Ref: `50a7cf6` (diagnostic only; no production-code change).
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`, `Go2Pvcnn/extension/joint_mpc_rti/model/go2_kinematics.py`, `Go2Pvcnn/extension/semantic_course.py`.

## Controlled Matrix

- World height/semantic field: `151x151`, `0.01m`, matching the joint MPC field contract.
- Shapes: sphere, cuboid, cylinder, capsule, cone.
- Course profile: diameter `0.12m`, target height `0.16m`, embed depth `0.015m`.
- Speeds: forward `0.1/0.2/0.4m/s`.
- Placement: obstacle centered on the left-leg travel lane; 96 rolling x1 updates.
- Geometry: foot sphere radius `0.022m`; calf and thigh capsules radius `0.04m`; base bottom samples from current Go2 FK.
- Each shape/speed cross-rate case uses six longitudinal placements to vary gait/obstacle phase alignment.

## Collision Metrics

Mean per-frame collision rate across the five shapes:

| speed | foot | calf | thigh | base |
| --- | ---: | ---: | ---: | ---: |
| `0.1m/s` | `16.0%` | `48.1%` | `0.4%` | `0%` |
| `0.2m/s` | `9.6%` | `43.5%` | `4.6%` | `0%` |
| `0.4m/s` | `7.7%` | `47.3%` | `4.4%` | `0%` |

Worst geometry penetration by shape over the three speeds:

| shape | foot | calf | thigh | base |
| --- | ---: | ---: | ---: | ---: |
| sphere | `30.3mm` | `48.3mm` | none | none |
| cuboid | `25.9mm` | `47.4mm` | `17.2mm` | none |
| cylinder | `32.7mm` | `50.7mm` | `32.7mm` | none |
| capsule | `31.6mm` | `49.6mm` | `1.9mm` | none |
| cone | `13.8mm` | `40.0mm` | `8.9mm` | none |

The calf is the dominant collision part at every speed. The base remained collision-free in this controlled low-small setup.

## Cross Success Definition

One per-leg event is a complete `stance -> contiguous swing -> stance` sequence. It is a crossing opportunity only when the swing foot center passes over the small semantic footprint. It succeeds only when:

1. the stance immediately before swing is not on the semantic object;
2. the first stance after swing is not on the semantic object;
3. foot, calf, and thigh geometry for that leg have no contact throughout the event.

## Cross Success Results

| shape | `0.1m/s` | `0.2m/s` | `0.4m/s` | aggregate |
| --- | ---: | ---: | ---: | ---: |
| sphere | `0/7` | `0/11` | `0/16` | `0/34` |
| cuboid | `1/17` | `0/8` | `0/16` | `1/41 (2.44%)` |
| cylinder | `0/7` | `0/12` | `0/8` | `0/27` |
| capsule | `0/8` | `0/9` | `0/12` | `0/29` |
| cone | `0/39` | `0/33` | `0/20` | `0/92` |

- Overall: `1/223 = 0.45%`.
- By speed: `1/78 = 1.28%` at `0.1m/s`; `0/73` at `0.2m/s`; `0/72` at `0.4m/s`.
- Calf contact invalidated `218/223 = 97.8%` of opportunities.
- Foot contact invalidated `174/223 = 78.0%` of opportunities.
- Thigh contact invalidated `3/223 = 1.35%` of opportunities.

## Measurement Boundary

The viewer uses direct kinematic pose playback and disables semantic contact sensors in the viewer config, so force-based PhysX contact rates are not available in this mode. These are native-shape geometric overlap metrics using the same Go2 FK dimensions and semantic-course shapes; they directly quantify the visible mesh intersection reported by the user.

## Conclusion

The reported behavior is reproduced. The current trajectory mostly moves through low-small obstacles instead of completing collision-free leg crossings. The primary failure is calf clearance, followed by foot clearance/landing; base collision is not the limiting factor in this controlled case.
