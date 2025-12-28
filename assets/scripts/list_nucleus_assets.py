#!/usr/bin/env python
"""
列出 IsaacLab Nucleus 目录下的所有 USD 资产
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

# 必须先初始化 AppLauncher 才能导入 isaaclab 模块
from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="列出 IsaacLab Nucleus 资产")
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
print("📂 IsaacLab Nucleus 资产目录扫描")
print("="*80)
print(f"\nISAACLAB_NUCLEUS_DIR: {ISAACLAB_NUCLEUS_DIR}")
print(f"ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")
print()

def list_files_recursive(path, extension=".usd", max_depth=3, current_depth=0):
    """递归列出目录中的所有文件，限制深度避免太慢"""
    files = []
    
    if current_depth >= max_depth:
        return files
    
    # 列出当前目录
    print(f"{'  ' * current_depth}🔍 扫描: {path.split('/')[-1]}/ (深度 {current_depth})")
    result, entries = omni.client.list(path)
    
    if result != omni.client.Result.OK:
        print(f"{'  ' * current_depth}⚠️  无法访问: {path}")
        return files
    
    print(f"{'  ' * current_depth}   找到 {len(entries)} 个条目")
    
    for entry in entries:
        entry_path = f"{path}/{entry.relative_path}"
        
        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            # 这是一个目录，递归搜索
            files.extend(list_files_recursive(entry_path, extension, max_depth, current_depth + 1))
        else:
            # 这是一个文件
            if entry.relative_path.endswith(extension) or entry.relative_path.endswith(extension + "a") or entry.relative_path.endswith(extension + "c"):
                files.append(entry_path)
    
    if current_depth == 0:
        print(f"   ✓ 本类别共找到 {len(files)} 个 USD 文件\n")
    
    return files

print("🔍 开始扫描 ISAAC_NUCLEUS_DIR (Isaac Sim 官方资产)...")
print(f"路径: {ISAAC_NUCLEUS_DIR}")
print()

# 首先列出主要类别
result, entries = omni.client.list(ISAAC_NUCLEUS_DIR)

if result == omni.client.Result.OK:
    categories = {}
    
    print(f"找到 {len(entries)} 个顶级类别\n")
    
    for entry in entries:
        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            category_name = entry.relative_path
            print(f"{'='*80}")
            print(f"📁 扫描类别: {category_name}")
            print(f"{'='*80}")
            
            category_path = f"{ISAAC_NUCLEUS_DIR}/{category_name}"
            usd_files = list_files_recursive(category_path, max_depth=4)  # 限制深度为4
            
            if usd_files:
                categories[category_name] = usd_files
    
    # 打印详细信息
    print("\n" + "="*80)
    print("📊 扫描结果汇总")
    print("="*80)
    
    total_files = sum(len(files) for files in categories.values())
    print(f"\n总计: {total_files} 个 USD 文件，分布在 {len(categories)} 个类别中\n")
    
    for category, files in sorted(categories.items()):
        print(f"\n{'='*80}")
        print(f"📦 {category.upper()} ({len(files)} 个文件)")
        print('='*80)
        
        # 按子目录分组
        subdirs = defaultdict(list)
        for f in files:
            # 获取相对路径
            rel_path = f.replace(f"{ISAAC_NUCLEUS_DIR}/{category}/", "")
            subdir = str(Path(rel_path).parent) if "/" in rel_path else "."
            subdirs[subdir].append(Path(rel_path).name)
        
        for subdir in sorted(subdirs.keys()):
            if subdir == ".":
                print(f"\n  📄 根目录:")
            else:
                print(f"\n  📁 {subdir}/")
            
            for filename in sorted(subdirs[subdir])[:10]:
                print(f"     • {filename}")
            
            if len(subdirs[subdir]) > 10:
                print(f"     ... 还有 {len(subdirs[subdir]) - 10} 个文件")
    
    # 保存到文件
    output_file = Path("assets/nucleus_asset_list.txt")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"IsaacLab Nucleus 资产列表\n")
        f.write(f"{'='*80}\n")
        f.write(f"扫描路径: {ISAAC_NUCLEUS_DIR}\n")
        f.write(f"总文件数: {total_files}\n")
        f.write(f"{'='*80}\n\n")
        
        for category, files in sorted(categories.items()):
            f.write(f"\n{'='*80}\n")
            f.write(f"{category.upper()} ({len(files)} 个文件)\n")
            f.write(f"{'='*80}\n\n")
            
            for file_path in sorted(files):
                # 只写相对路径
                rel_path = file_path.replace(f"{ISAAC_NUCLEUS_DIR}/", "")
                f.write(f"  {rel_path}\n")
    
    print(f"\n\n✓ 完整列表已保存到: {output_file}")
    
    # 生成 JSON 格式
    json_output = {
        "isaaclab_nucleus_dir": ISAACLAB_NUCLEUS_DIR,
        "isaac_nucleus_dir": ISAAC_NUCLEUS_DIR,
        "total_files": total_files,
        "categories": {}
    }
    
    for category, files in categories.items():
        # 保存相对路径
        rel_files = [f.replace(f"{ISAAC_NUCLEUS_DIR}/", "") for f in files]
        json_output["categories"][category] = sorted(rel_files)
    
    json_file = Path("assets/nucleus_asset_catalog.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON 目录已保存到: {json_file}")
    
    # 显示一些推荐的家具/环境资产
    print("\n" + "="*80)
    print("🏠 推荐的家具和环境资产")
    print("="*80)
    
    furniture_keywords = ['furniture', 'table', 'chair', 'shelf', 'cabinet', 'desk', 
                          'warehouse', 'room', 'office', 'apartment', 'house']
    
    for category, files in categories.items():
        matching_files = []
        for f in files:
            file_lower = f.lower()
            if any(kw in file_lower for kw in furniture_keywords):
                matching_files.append(f.replace(f"{ISAAC_NUCLEUS_DIR}/", ""))
        
        if matching_files:
            print(f"\n📦 {category}:")
            for f in sorted(matching_files)[:5]:
                print(f"   • {f}")
            if len(matching_files) > 5:
                print(f"   ... 还有 {len(matching_files) - 5} 个相关文件")

else:
    print(f"❌ 无法访问 Nucleus 目录: {ISAAC_NUCLEUS_DIR}")
    print(f"错误代码: {result}")

# 关闭应用
simulation_app.close()
