#!/usr/bin/env python
"""
检查 USD 资产是否包含物理和碰撞属性
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="检查 USD 物理属性")
parser.add_argument("--headless", action="store_true", default=True, help="以无头模式运行")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.client
from pxr import Usd, UsdPhysics, UsdGeom

print("="*80)
print("🔍 检查 Warehouse Props 物理属性")
print("="*80)

def check_usd_physics(usd_path):
    """检查 USD 文件是否包含物理属性"""
    # 尝试打开 USD 文件
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return None
    
    has_physics = False
    has_collision = False
    has_rigidbody = False
    
    # 遍历所有 prim
    for prim in stage.Traverse():
        # 检查是否有 PhysicsCollisionAPI
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            has_collision = True
        
        # 检查是否有 PhysicsRigidBodyAPI
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            has_rigidbody = True
        
        # 检查是否有 PhysicsMeshCollisionAPI
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            has_collision = True
    
    has_physics = has_collision or has_rigidbody
    
    return {
        "has_physics": has_physics,
        "has_collision": has_collision,
        "has_rigidbody": has_rigidbody
    }

# 测试几个 warehouse props
test_props = [
    "SM_CardBoxA_01.usd",
    "SM_BarelPlastic_A_01.usd",
    "SM_BottlePlasticA_01.usd",
    "SM_BucketPlastic_B.usd"
]

print("\n测试 Simple_Warehouse Props 样本:\n")

for prop_name in test_props:
    prop_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/{prop_name}"
    print(f"📦 {prop_name}")
    print(f"   路径: {prop_path}")
    
    # 下载并检查
    result, entry = omni.client.stat(prop_path)
    if result == omni.client.Result.OK:
        print(f"   ✓ 文件存在")
        
        # 尝试检查物理属性
        try:
            props = check_usd_physics(prop_path)
            if props:
                if props["has_physics"]:
                    print(f"   ✅ 有物理属性:")
                    if props["has_collision"]:
                        print(f"      - 有碰撞体 (Collision)")
                    if props["has_rigidbody"]:
                        print(f"      - 有刚体 (RigidBody)")
                else:
                    print(f"   ❌ 无物理属性 (仅几何模型)")
            else:
                print(f"   ⚠️  无法解析 USD 文件")
        except Exception as e:
            print(f"   ⚠️  检查失败: {e}")
    else:
        print(f"   ❌ 文件不存在")
    
    print()

# 对比：检查 YCB 物体
print("="*80)
print("🔧 对比：YCB Physics 物体")
print("="*80)

ycb_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
print(f"\n📦 003_cracker_box.usd (Axis_Aligned_Physics)")
print(f"   路径: {ycb_path}")

try:
    props = check_usd_physics(ycb_path)
    if props:
        if props["has_physics"]:
            print(f"   ✅ 有物理属性:")
            if props["has_collision"]:
                print(f"      - 有碰撞体 (Collision)")
            if props["has_rigidbody"]:
                print(f"      - 有刚体 (RigidBody)")
        else:
            print(f"   ❌ 无物理属性")
except Exception as e:
    print(f"   ⚠️  检查失败: {e}")

print("\n" + "="*80)
print("📝 结论")
print("="*80)
print("""
如果 Warehouse Props 没有物理属性，你有两个选择：

1. ✅ 使用 YCB Axis_Aligned_Physics 物体（确保有碰撞检测）
   - 003_cracker_box.usd
   - 004_sugar_box.usd
   - 005_tomato_soup_can.usd
   - 006_mustard_bottle.usd

2. 🔧 手动给 Warehouse Props 添加物理属性（在 Isaac Sim 中）
   - 打开 USD 文件
   - 添加 Physics -> Collision API
   - 添加 Physics -> Rigid Body API
   - 保存到本地

推荐使用方案 1，因为 YCB 物体已经配置好了物理属性。
""")

simulation_app.close()
