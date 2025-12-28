#!/usr/bin/env python
"""
详细列出 YCB 物体和环境 Props
"""

import argparse
from pathlib import Path

# 必须先初始化 AppLauncher 才能导入 isaaclab 模块
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="详细列出资产")
parser.add_argument("--headless", action="store_true", default=True, help="以无头模式运行")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.client
import json

print("="*80)
print("📋 详细资产列表")
print("="*80)

def list_files_in_dir(path, recursive=False, max_depth=2):
    """列出目录中的所有 USD 文件"""
    result, entries = omni.client.list(path)
    
    if result != omni.client.Result.OK:
        return []
    
    files = []
    for entry in entries:
        full_path = f"{path}/{entry.relative_path}"
        
        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            if recursive and max_depth > 0:
                sub_files = list_files_in_dir(full_path, recursive, max_depth - 1)
                for sf in sub_files:
                    files.append(f"{entry.relative_path}/{sf}")
        elif entry.relative_path.endswith(('.usd', '.usda', '.usdc')):
            files.append(entry.relative_path)
    
    return files

# 列出 YCB 物体
print("\n" + "="*80)
print("🔧 YCB Objects (常见家居物体)")
print("="*80)

ycb_categories = ["Axis_Aligned", "Axis_Aligned_Physics"]
ycb_assets = {}

for cat in ycb_categories:
    path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/{cat}"
    print(f"\n📁 {cat}/")
    files = list_files_in_dir(path)
    
    ycb_assets[cat] = files
    
    for f in sorted(files):
        print(f"   • {f}")

# 列出仓库 Props
print("\n" + "="*80)
print("📦 Simple Warehouse Props")
print("="*80)

warehouse_props_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props"
print(f"\n扫描: {warehouse_props_path}")

warehouse_props = list_files_in_dir(warehouse_props_path, recursive=True, max_depth=1)
print(f"\n找到 {len(warehouse_props)} 个 USD 文件\n")

# 按类型分组
prop_types = {}
for prop in warehouse_props:
    prop_type = prop.split('/')[0] if '/' in prop else '根目录'
    if prop_type not in prop_types:
        prop_types[prop_type] = []
    prop_types[prop_type].append(prop)

for prop_type, props in sorted(prop_types.items()):
    print(f"\n  📂 {prop_type}/ ({len(props)} 个)")
    for p in sorted(props)[:10]:
        print(f"     • {p}")
    if len(props) > 10:
        print(f"     ... 还有 {len(props) - 10} 个")

# 列出 Office Props
print("\n" + "="*80)
print("🏢 Office Props")
print("="*80)

office_props_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props"
print(f"\n扫描: {office_props_path}")

office_props = list_files_in_dir(office_props_path, recursive=True, max_depth=1)
print(f"\n找到 {len(office_props)} 个 USD 文件\n")

# 按类型分组
prop_types = {}
for prop in office_props:
    prop_type = prop.split('/')[0] if '/' in prop else '根目录'
    if prop_type not in prop_types:
        prop_types[prop_type] = []
    prop_types[prop_type].append(prop)

for prop_type, props in sorted(prop_types.items()):
    print(f"\n  📂 {prop_type}/ ({len(props)} 个)")
    for p in sorted(props)[:10]:
        print(f"     • {p}")
    if len(props) > 10:
        print(f"     ... 还有 {len(props) - 10} 个")

# 保存完整列表
output = {
    "isaac_nucleus_dir": ISAAC_NUCLEUS_DIR,
    "ycb_objects": ycb_assets,
    "warehouse_props": warehouse_props,
    "office_props": office_props
}

json_file = Path("assets/detailed_furniture_list.json")
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\n" + "="*80)
print(f"✓ 详细列表已保存到: {json_file}")
print("="*80)

# 显示使用示例
print("\n💡 使用示例:")
print("""
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# YCB 物体（带物理属性）
cracker_box = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
mug = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/025_mug.usd"

# 仓库环境
warehouse = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"

# Office 环境
office = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd"
""")

simulation_app.close()
