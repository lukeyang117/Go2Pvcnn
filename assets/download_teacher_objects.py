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

注意：
- teacher 配置当前使用“扁平化”家具路径（直接引用 `SM_Sofa.usd`），
    这些 USD 内部仍使用相对路径引用材质（例如 `../Materials/*.mdl`）。
- YCB 的 USD 在不同 Isaac 版本里也可能保留旧的相对贴图路径
    （例如 `Materials/Textures/*.png` 或 `../Axis_Aligned/Materials/Textures/*.png`）。
- 因此脚本除了下载主 USD，还会下载并镜像依赖目录，避免运行时报材质缺失。
"""

import argparse
import shutil
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


def copy_dir_from_candidates(src_candidates: list[str], dst_base: Path, rel_prefix: str) -> int:
    """从候选源目录中选择可访问目录并递归复制。"""
    for src_dir in src_candidates:
        result, _ = omni.client.list(src_dir)
        if result == omni.client.Result.OK:
            print(f"  使用源目录: {src_dir}")
            return copy_dir_recursive(src_dir, dst_base, rel_prefix)

    print("  ⚠️ 所有候选源目录均不可访问:")
    for src_dir in src_candidates:
        print(f"    - {src_dir}")
    return 0


def copy_file_from_candidates(src_candidates: list[str], dst_path: Path) -> bool:
    """从候选源路径中复制单文件，命中任意一个即成功。"""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for src in src_candidates:
        result = omni.client.copy(src, str(dst_path))
        if result == omni.client.Result.OK:
            print(f"  ✅ 成功 (source: {src})")
            return True
        failures.append(f"{src}: {result}")

    print("  ❌ 失败，候选源均不可用:")
    for msg in failures:
        print(f"    - {msg}")
    return False


def mirror_local_tree(src_dir: Path, dst_dir: Path) -> int:
    """将本地目录镜像到另一个目录，返回复制的文件数。"""
    if not src_dir.exists():
        print(f"  ⚠️ 本地源目录不存在: {src_dir}")
        return 0

    count = 0
    for src_file in src_dir.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src_dir)
        dst_file = dst_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        print(f"    ✓ {dst_file.relative_to(dst_dir.parent)}")
        count += 1
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

# 依赖目录（用于修复 USD 内部相对路径引用）
OFFICE_MATERIALS_DIR = "Environments/Office/Materials"
OFFICE_PROPS_TEXTURES_DIR = "Environments/Office/Props/Textures"
YCB_TEXTURES_CANDIDATES = [
    "Props/YCB/Axis_Aligned_Physics/Materials/Textures",
    "Props/YCB/Axis_Aligned/Materials/Textures",
    "Props/YCB/Materials/Textures",
]
YCB_TEXTURE_FILES = [
    "003_cracker_box_COLOR.png",
    "004_sugar_box_COLOR.png",
    "005_tomato_soup_can_COLOR.png",
]

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
    src_candidates = [f"{ISAAC_NUCLEUS_DIR}/{rel_path}"]
    # YCB 物体在不同版本中可能存在 Axis_Aligned 回退路径
    if "Axis_Aligned_Physics" in rel_path:
        src_candidates.append(f"{ISAAC_NUCLEUS_DIR}/{rel_path.replace('Axis_Aligned_Physics', 'Axis_Aligned')}")

    # 家具/YCB 保存到根目录，与 teacher_semantic_env_cfg 中 _TEACHER_OBJECTS_DIR/"SM_Sofa.usd" 等路径一致
    dst_path = out_dir / Path(rel_path).name
    print(f"→ {rel_path}")
    ok = copy_file_from_candidates(src_candidates, dst_path)
    if ok:
        total_ok += 1

# 天空光纹理（保留目录结构，供 DomeLight 使用）
print(f"→ {TEACHER_ISAAC_SKY}")
dst_sky = out_dir / TEACHER_ISAAC_SKY
if copy_file_from_candidates(
    [
        f"{ISAAC_NUCLEUS_DIR}/{TEACHER_ISAAC_SKY}",
        f"{ISAACLAB_NUCLEUS_DIR}/{TEACHER_ISAAC_SKY}",
    ],
    dst_sky,
):
    total_ok += 1
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

# 3. 下载 ISAAC_NUCLEUS 依赖目录（修复材质引用）
print("─" * 40)
print("📦 ISAAC_NUCLEUS_DIR (依赖目录)")
print("─" * 40)

# 3.1 YCB 贴图，供 003/004/005 的 USD 引用 Materials/Textures/*.png
print("→ 复制 YCB 贴图目录到 teacher_object/Materials/Textures")
n = copy_dir_from_candidates(
    [f"{ISAAC_NUCLEUS_DIR}/{p}" for p in YCB_TEXTURES_CANDIDATES],
    out_dir,
    "Materials/Textures",
)
if n == 0:
    # 某些版本目录结构不同，退化为逐文件候选下载
    print("  ↪️ 目录复制未命中，尝试逐文件下载 YCB 贴图")
    for tex in YCB_TEXTURE_FILES:
        src_candidates = [f"{ISAAC_NUCLEUS_DIR}/{p}/{tex}" for p in YCB_TEXTURES_CANDIDATES]
        if copy_file_from_candidates(src_candidates, out_dir / "Materials" / "Textures" / tex):
            n += 1
total_ok += n
print(f"  ✅ 共 {n} 个文件")

# 3.1.1 为本地扁平 YCB USD 提供额外回退路径
# 某些 USD 会引用:
# - Materials/Textures/*.png
# - ../Axis_Aligned/Materials/Textures/*.png
# 因此在 teacher_object 根目录之外，再额外镜像到 assets/Axis_Aligned/Materials/Textures
print(f"→ 镜像本地 YCB 贴图到: {out_dir.parent / 'Axis_Aligned' / 'Materials' / 'Textures'}")
n = mirror_local_tree(out_dir / "Materials" / "Textures", out_dir.parent / "Axis_Aligned" / "Materials" / "Textures")
total_ok += n
print(f"  ✅ 共 {n} 个文件")

print(f"→ 镜像本地 YCB 贴图到: {out_dir.parent / 'Axis_Aligned_Physics' / 'Materials' / 'Textures'}")
n = mirror_local_tree(
    out_dir / "Materials" / "Textures",
    out_dir.parent / "Axis_Aligned_Physics" / "Materials" / "Textures",
)
total_ok += n
print(f"  ✅ 共 {n} 个文件")

# 3.2 Office 材质 + 纹理，供家具 USD 内部材质网络引用
print("→ 复制 Office Materials 到 teacher_object/Materials")
n = copy_dir_from_candidates([f"{ISAAC_NUCLEUS_DIR}/{OFFICE_MATERIALS_DIR}"], out_dir, "Materials")
total_ok += n
print(f"  ✅ 共 {n} 个文件")

print("→ 复制 Office Props 纹理到 teacher_object/Textures")
n = copy_dir_from_candidates([f"{ISAAC_NUCLEUS_DIR}/{OFFICE_PROPS_TEXTURES_DIR}"], out_dir, "Textures")
total_ok += n
print(f"  ✅ 共 {n} 个文件")
print()

# 4. 兼容“扁平化家具 USD”中的 ../Materials 相对路径
# SM_*.usd 存在于 teacher_object 根目录时，../Materials 解析到 out_dir.parent/Materials
print("─" * 40)
print("📦 扁平路径兼容镜像")
print("─" * 40)
mirror_base = out_dir.parent
print(f"→ 镜像 Office Materials 到: {mirror_base / 'Materials'}")
n = copy_dir_from_candidates([f"{ISAAC_NUCLEUS_DIR}/{OFFICE_MATERIALS_DIR}"], mirror_base, "Materials")
total_ok += n
print(f"  ✅ 共 {n} 个文件")

print(f"→ 镜像 Office Props 纹理到: {mirror_base / 'Materials' / 'Textures'}")
n = copy_dir_from_candidates([f"{ISAAC_NUCLEUS_DIR}/{OFFICE_PROPS_TEXTURES_DIR}"], mirror_base, "Materials/Textures")
total_ok += n
print(f"  ✅ 共 {n} 个文件")
print()

print(f"完成: 共下载/复制 {total_ok} 个资源")
print()
print("teacher_semantic_env_cfg.py 已配置为使用本地路径:")
print(f"  _TEACHER_OBJECTS_DIR = {out_dir}")
print("若你继续使用扁平家具 USD (SM_*.usd)，请保留父目录镜像 Materials。")
print("若你继续使用扁平 YCB USD (003/004/005_*.usd)，请保留父目录镜像 Axis_Aligned*/Materials/Textures。")

simulation_app.close()
