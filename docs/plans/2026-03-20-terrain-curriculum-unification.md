# 地形与 Curriculum 统一方案

**日期**: 2026-03-20  
**目标**: 将 teacher_elevation 的地形改为 teacher_semantic 的地形，并在 teacher_semantic 中启用地形 curriculum（terrain_levels）

---

## 1. 流程概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│  修改 1: teacher_elevation_env_cfg.py                                    │
│  - 导入 teacher_semantic 的地形配置                                      │
│  - TeacherElevationSceneCfg 覆盖 terrain 为 teacher_semantic 的地形      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  修改 2: teacher_semantic_env_cfg.py                                     │
│  - CurriculumCfg 添加 terrain_levels                                     │
│  - __post_init__ 添加 terrain curriculum 开关逻辑                         │
│  - TeacherSemanticEnvCfg_PLAY 在禁用 curriculum 时关闭 terrain curriculum │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 修改详情

### 2.1 teacher_elevation_env_cfg.py

**目标**: 使用 teacher_semantic 的地形配置（COBBLESTONE_ROAD_CFG + TerrainImporterCfg）

**变更**:
1. 新增导入：`isaaclab.sim`, `isaaclab.terrains.TerrainImporterCfg`，以及 teacher_semantic 的 `COBBLESTONE_ROAD_CFG`、`_TEACHER_OBJECTS_DIR`
2. 在 `TeacherElevationSceneCfg` 中覆盖 `terrain` 属性，使用与 teacher_semantic 相同的 TerrainImporterCfg（含本地材质路径）

**依赖**: teacher_semantic 的 `_TEACHER_OBJECTS_DIR` 需存在，即需运行 `assets/download_teacher_objects.py`

---

### 2.2 teacher_semantic_env_cfg.py

**目标**: 启用地形 curriculum，复用 `terrain_levels_vel_unitree_rl_lab`

**变更**:
1. **CurriculumCfg**: 添加 `terrain_levels = CurrTerm(func=custom_mdp.terrain_levels_vel_unitree_rl_lab)`
2. **TeacherSemanticEnvCfg.__post_init__**: 添加与 teacher_without_semantic 相同的地形 curriculum 开关逻辑：
   - 若 `curriculum.terrain_levels` 存在 → `terrain_generator.curriculum = True`
   - 否则 → `terrain_generator.curriculum = False`
3. **TeacherSemanticEnvCfg_PLAY.__post_init__**: 在设置 `curriculum = CurriculumCfg_Empty()` 后，显式设置 `terrain_generator.curriculum = False`（因为 PLAY 模式禁用 curriculum）

---

## 3. 兼容性说明

| 组件 | 兼容性 |
|------|--------|
| terrain_levels_vel_unitree_rl_lab | ✅ 与 teacher_semantic 地形兼容（size、sub_terrains、update_env_origins 一致） |
| teacher_elevation 继承链 | ✅ 仍继承 TeacherWithoutSemanticEnvCfg，curriculum 含 terrain_levels |
| teacher_semantic PLAY 模式 | ✅ CurriculumCfg_Empty 无 terrain_levels，需显式关闭 terrain curriculum |

---

## 4. 验证步骤

1. **teacher_elevation 训练**:
   ```bash
   python scripts/train.py --task Isaac-Teacher-Elevation-Env-v0
   ```
   确认地形为 teacher_semantic 风格（本地材质），且地形难度随训练变化。

2. **teacher_semantic 训练**:
   ```bash
   python scripts/train.py --task Isaac-Teacher-Semantic-Env-v0
   ```
   确认地形难度随训练变化（terrain_levels 生效）。

3. **teacher_semantic PLAY**:
   ```bash
   python scripts/play.py --task Isaac-Teacher-Semantic-Env-v0
   ```
   确认无 terrain curriculum 相关报错，地形保持初始难度。

---

## 5. 文件变更清单

| 文件 | 变更类型 |
|------|----------|
| `go2_pvcnn/tasks/teacher_elevation_env_cfg.py` | 修改：导入 + terrain 覆盖 |
| `go2_pvcnn/tasks/teacher_semantic_env_cfg.py` | 修改：CurriculumCfg + __post_init__ |
