# Go2 Body Geometry Clearance HTML Design

## Purpose

Create a Chinese HTML design for replacing sparse point-based `semantic_body_part_clearance` with Go2 body-geometry semantic clearance using USD-derived dimensions and GPU-batched map queries.

## Stage

RL reward design / flat-small semantic clearance.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Command / Procedure

Inspected Go2 USD assets through IsaacLab / USD APIs:

- `assets/teacher_object/Robots/Unitree/Go2/go2.usd`
- `assets/teacher_object/Robots/Unitree/Go2/Props/instanceable_meshes.usd`

Created:

- [../../docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html](../../docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html)

Verified HTML parse with:

```bash
python - <<'PY'
from html.parser import HTMLParser
from pathlib import Path
p=Path('docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html')
parser=HTMLParser()
parser.feed(p.read_text(encoding='utf-8'))
print('html_parse_ok', p.stat().st_size)
PY
```

## Input Conditions

- User asked for a Chinese HTML design, not code implementation.
- The design includes base geometry in addition to foot/calf/thigh.
- The design explicitly explains why capsule/box geometry is still implemented as fixed section / footprint map queries on discrete scanner maps.

## Key Metrics

- HTML parse: `html_parse_ok 15220`.
- USD-derived geometry captured in the design:
  - foot collision sphere radius `0.022m`
  - thigh collision cube `0.213 x 0.0245 x 0.034m`
  - calf visual `0.058 x 0.040 x 0.267m`
  - base collision box `0.376 x 0.0935 x 0.114m`
- Recommended first-version reward geometry:
  - foot sphere query radius `0.035m`
  - calf/thigh capsule radius `0.040m`, query radius `0.045m`, sections `7`
  - base half extents `[0.20, 0.06, 0.07]`, footprint grid `[5,3]`, query radius `0.03m`

## Result

Pass. A Chinese HTML design now documents the intended geometry, parallel query model, fixed-shape performance constraints, reward aggregation, validation plan, and risk boundaries.

## Conclusion

The design keeps the reward GPU-friendly by using fixed-shape section/footprint neighborhood queries over the scanner semantic/elevation maps, not USD mesh collision or per-env loops.

## Follow-Up

- If approved, create a focused implementation plan before changing runtime reward code.
- Preserve the already-fixed flat mask/curriculum bookkeeping and keep `semantic_contact_collision` as the authoritative real-contact penalty.

## Git Refs

- Current Work Ref: working tree on 2026-06-11
- Key Files:
  - [../../docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html](../../docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html)
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
