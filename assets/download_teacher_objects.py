#!/usr/bin/env python
"""
下载 teacher_semantic_env_cfg.py 用到的所有资产到本地。

包含：
- ISAAC_NUCLEUS_DIR:
  - 家具 (Office/Props): SM_Sofa.usd, SM_Armchair.usd, SM_TableA.usd
  - YCB 物体 (Axis_Aligned_Physics): 003_cracker_box.usd, 004_sugar_box.usd, 005_tomato_soup_can.usd
- ISAACLAB_NUCLEUS_DIR:
  - Materials/TilesMarbleSpiderWhiteBrickBondHoned/ (地形材质，含 .mdl 及依赖)
  - Robots/Unitree/Go2/ (机器狗 Go2 及依赖)
"""

import argparse
from pathlib import Path

# 必须先初始化 AppLauncher 才能导入 isaaclab 模块
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="下载 teacher_semantic 用到的 USD/Material 资产")
parser.add_argument("--headless", action="store_true", default=True, help="以无头模式运行")
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="下载到的本地目录（默认: assets/teacher_object）",
)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import omni.client

# 脚本在 assets/ 目录下，输出默认到 assets/teacher_object
SCRIPT_DIR = Path(__file__).resolve().parent
out_dir = Path(args.out_dir) if args.out_dir else (SCRIPT_DIR / "teacher_object")
out_dir = out_dir.resolve()
out_dir.mkdir(parents=True, exist_ok=True)


def copy_dir_recursive(src_dir: str, dst_base: Path, rel_prefix: str = "") -> int:
    """递归复制 Nucleus 目录到本地，返回成功复制的文件数。"""
    (dst_base / rel_prefix).mkdir(parents=True, exist_ok=True)
    result, entries = omni.client.list(src_dir)
    if result != omni.client.Result.OK:
        print(f"  ⚠️ 无法列出: {src_dir}")
        return 0

    count = 0
    for entry in entries:
        name = entry.relative_path
        entry_src = f"{src_dir.rstrip('/')}/{name}"
        entry_rel = f"{rel_prefix}/{name}".lstrip("/")
        entry_dst = dst_base / entry_rel

        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            count += copy_dir_recursive(entry_src, dst_base, entry_rel)
        else:
            entry_dst.parent.mkdir(parents=True, exist_ok=True)
            res = omni.client.copy(entry_src, str(entry_dst))
            if res == omni.client.Result.OK:
                print(f"    ✓ {entry_rel}")
                count += 1
            else:
                print(f"    ✗ {entry_rel}: {res}")
    return count


# teacher_semantic_env_cfg.py 用到的资产
# ISAAC_NUCLEUS_DIR 下的单文件
TEACHER_ISAAC_USD = [
    "Environments/Office/Props/SM_Sofa.usd",
    "Environments/Office/Props/SM_Armchair.usd",
    "Environments/Office/Props/SM_TableA.usd",
    "Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
    "Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
    "Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
]
# 天空光 HDR 纹理（DomeLight 用）- 需保留目录结构
TEACHER_ISAAC_SKY = "Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"

# ISAACLAB_NUCLEUS_DIR 下的目录（保留结构）
TEACHER_ISAACLAB_DIRS = [
    "Materials/TilesMarbleSpiderWhiteBrickBondHoned",
    "Robots/Unitree/Go2",  # 机器狗 Go2
]

print("=" * 80)
print("📥 下载 teacher_semantic 所需资产 (ISAAC + ISAACLAB)")
print("=" * 80)
print(f"ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")
print(f"ISAACLAB_NUCLEUS_DIR: {ISAACLAB_NUCLEUS_DIR}")
print(f"输出目录: {out_dir}")
print()

total_ok = 0

# 1. 下载 ISAAC_NUCLEUS 单文件
print("─" * 40)
print("📦 ISAAC_NUCLEUS_DIR (单文件)")
print("─" * 40)
for rel_path in TEACHER_ISAAC_USD:
    src = f"{ISAAC_NUCLEUS_DIR}/{rel_path}"
    # 家具/YCB 保存到根目录，与 teacher_semantic_env_cfg 中 _TEACHER_OBJECTS_DIR/"SM_Sofa.usd" 等路径一致
    dst_path = out_dir / Path(rel_path).name
    print(f"→ {rel_path}")
    result = omni.client.copy(src, str(dst_path))
    if result == omni.client.Result.OK:
        print("  ✅ 成功")
        total_ok += 1
    else:
        print(f"  ❌ 失败: {result}")

# 天空光纹理（保留目录结构，供 DomeLight 使用）
print(f"→ {TEACHER_ISAAC_SKY}")
dst_sky = out_dir / TEACHER_ISAAC_SKY
dst_sky.parent.mkdir(parents=True, exist_ok=True)
result = omni.client.copy(f"{ISAAC_NUCLEUS_DIR}/{TEACHER_ISAAC_SKY}", str(dst_sky))
if result == omni.client.Result.OK:
    print("  ✅ 成功")
    total_ok += 1
else:
    # 尝试从 ISAACLAB 下载（部分版本 sky 在 IsaacLab）
    result2 = omni.client.copy(f"{ISAACLAB_NUCLEUS_DIR}/{TEACHER_ISAAC_SKY}", str(dst_sky))
    if result2 == omni.client.Result.OK:
        print("  ✅ 成功 (via ISAACLAB)")
        total_ok += 1
    else:
        print(f"  ❌ 失败: ISAAC={result}, ISAACLAB={result2}")
print()

# 2. 下载 ISAACLAB_NUCLEUS 目录
print("─" * 40)
print("📦 ISAACLAB_NUCLEUS_DIR (目录)")
print("─" * 40)
for rel_dir in TEACHER_ISAACLAB_DIRS:
    src_dir = f"{ISAACLAB_NUCLEUS_DIR}/{rel_dir}"
    print(f"→ 递归复制: {rel_dir}/")
    n = copy_dir_recursive(src_dir, out_dir, rel_dir)
    total_ok += n
    print(f"  ✅ 共 {n} 个文件")
print()

print(f"完成: 共下载/复制 {total_ok} 个资源")
print()
print("teacher_semantic_env_cfg.py 已配置为使用本地路径:")
print(f"  _TEACHER_OBJECTS_DIR = {out_dir}")

simulation_app.close()
