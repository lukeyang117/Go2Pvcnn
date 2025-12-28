#!/usr/bin/env python
"""
快速列出家具相关的 USD 资产（只扫描 Props 和 Environments）
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

# 必须先初始化 AppLauncher 才能导入 isaaclab 模块
from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="列出家具相关资产")
parser.add_argument("--headless", action="store_true", default=True, help="以无头模式运行")
args = parser.parse_args()

# 启动 Isaac Sim 应用
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 现在可以安全导入 isaaclab 模块
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR

import omni.client
import json

print("="*80)
print("🏠 IsaacLab 家具资产快速扫描")
print("="*80)
print(f"\nISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")
print()

def list_directory_shallow(path, max_items=100):
    """浅层列出目录，不递归，用于快速预览"""
    result, entries = omni.client.list(path)
    
    if result != omni.client.Result.OK:
        return [], []
    
    files = []
    dirs = []
    
    for entry in entries:
        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            dirs.append(entry.relative_path)
        elif entry.relative_path.endswith(('.usd', '.usda', '.usdc')):
            files.append(entry.relative_path)
    
    return files, dirs

# 定义要扫描的类别
categories_to_scan = {
    "Props": ["Blocks", "YCB", "Warehouse", "S1", "SM"],
    "Environments": ["Hospital", "Office", "Simple_Room", "Simple_Warehouse", "Modular_Warehouse"]
}

all_assets = {}

for category, subdirs in categories_to_scan.items():
    print(f"\n{'='*80}")
    print(f"📦 {category}")
    print('='*80)
    
    category_path = f"{ISAAC_NUCLEUS_DIR}/{category}"
    
    if subdirs:
        # 只扫描指定的子目录
        for subdir in subdirs:
            subdir_path = f"{category_path}/{subdir}"
            print(f"\n  📁 {category}/{subdir}/")
            
            files, sub_subdirs = list_directory_shallow(subdir_path)
            
            if files:
                print(f"     ✓ 根目录下 {len(files)} 个 USD 文件")
                for f in sorted(files)[:5]:
                    print(f"       • {f}")
                if len(files) > 5:
                    print(f"       ... 还有 {len(files) - 5} 个")
            
            # 检查子子目录
            if sub_subdirs:
                print(f"     📂 子目录: {', '.join(sorted(sub_subdirs)[:5])}")
                if len(sub_subdirs) > 5:
                    print(f"        ... 还有 {len(sub_subdirs) - 5} 个")
            
            # 保存到结果
            if category not in all_assets:
                all_assets[category] = {}
            all_assets[category][subdir] = {
                "files": files,
                "subdirs": sub_subdirs,
                "path": f"{category}/{subdir}"
            }
    else:
        # 列出类别下所有内容
        files, subdirs = list_directory_shallow(category_path)
        print(f"\n  找到 {len(files)} 个 USD 文件, {len(subdirs)} 个子目录")

# 保存结果
print("\n" + "="*80)
print("💾 保存资产列表")
print("="*80)

output_data = {
    "isaac_nucleus_dir": ISAAC_NUCLEUS_DIR,
    "scan_time": "2025-11-19",
    "categories": all_assets
}

json_file = Path("assets/furniture_assets_quick.json")
json_file.parent.mkdir(exist_ok=True)

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n✓ 资产列表已保存到: {json_file}")

# 生成一个简单的下载脚本模板
print("\n" + "="*80)
print("📝 推荐的资产路径")
print("="*80)

print("\n# Props (道具)")
print(f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/")
print(f"{ISAAC_NUCLEUS_DIR}/Props/YCB/")
print(f"{ISAAC_NUCLEUS_DIR}/Props/Warehouse/")

print("\n# Environments (环境)")
print(f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/")
print(f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/")
print(f"{ISAAC_NUCLEUS_DIR}/Environments/Office/")
print(f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/")

print("\n" + "="*80)
print("🎯 使用方法")
print("="*80)
print("\n在你的代码中，可以这样引用资产:")
print(f"""
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# 示例：Simple Warehouse
warehouse_usd = f"{{ISAAC_NUCLEUS_DIR}}/Environments/Simple_Warehouse/warehouse.usd"

# 示例：YCB 物体
mug_usd = f"{{ISAAC_NUCLEUS_DIR}}/Props/YCB/Axis_Aligned/003_cracker_box.usd"
""")

# 关闭应用
simulation_app.close()
