#!/usr/bin/env python
"""
给本地家具 USD 批量添加物理属性（RigidBody + Collision）。
"""

import argparse
from pathlib import Path

# 必须先初始化 AppLauncher 才能导入 isaaclab/omni
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="给家具 USD 添加物理属性")
parser.add_argument("--headless", action="store_true", default=True, help="以无头模式运行")
parser.add_argument(
    "--input_dir",
    type=str,
    default="assets/downloaded_furniture",
    help="输入 USD 目录（相对项目根目录）",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="assets/downloaded_furniture_physics",
    help="输出 USD 目录（相对项目根目录）",
)
parser.add_argument("--kinematic", action="store_true", default=True, help="设置为静态刚体")
parser.add_argument("--disable_gravity", action="store_true", default=True, help="禁用重力")
parser.add_argument(
    "--collision_approx",
    type=str,
    default="none",
    choices=["none"],
    help="保留接口但不设置 PhysX 近似（避免 API 缺失报错）",
)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
input_dir = (PROJECT_ROOT / args.input_dir).resolve()
output_dir = (PROJECT_ROOT / args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

usd_files = sorted(input_dir.glob("*.usd"))

print("=" * 80)
print("🧩 添加物理属性到 USD")
print("=" * 80)
print(f"输入目录: {input_dir}")
print(f"输出目录: {output_dir}")
print(f"USD 数量: {len(usd_files)}")
print()

if not usd_files:
    print("⚠️ 未找到 USD 文件")
    simulation_app.close()
    raise SystemExit(0)


def add_physics(stage: Usd.Stage) -> None:
    # 取默认 prim 作为根；如果没有，找一个顶层 Xform
    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        for prim in stage.Traverse():
            if prim.GetParent() == stage.GetPseudoRoot() and prim.IsA(UsdGeom.Xform):
                root_prim = prim
                break
    if not root_prim or not root_prim.IsValid():
        root_prim = stage.GetPseudoRoot()

    # 刚体（在根上）
    rb_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rb_api.CreateRigidBodyEnabledAttr(True)
    if args.kinematic:
        rb_api.CreateKinematicEnabledAttr(True)

    physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    if args.disable_gravity:
        if hasattr(physx_rb_api, "CreateDisableGravityAttr"):
            physx_rb_api.CreateDisableGravityAttr(True)

    # 碰撞（对所有 Mesh）
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            PhysxSchema.PhysxCollisionAPI.Apply(prim)


for usd_path in usd_files:
    print(f"→ 处理: {usd_path.name}")
    stage = Usd.Stage.Open(str(usd_path))

    if stage is None:
        print("  ❌ 无法打开")
        continue

    add_physics(stage)

    out_path = output_dir / usd_path.name
    stage.GetRootLayer().Export(str(out_path))
    print(f"  ✅ 已保存: {out_path}")

print("\n完成")

simulation_app.close()
