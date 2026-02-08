#!/usr/bin/env python
"""
下载 Office/Props 下的 3 个家具 USD 到本地目录。
"""

import argparse
from pathlib import Path

# 必须先初始化 AppLauncher 才能导入 isaaclab 模块
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="下载 3 个家具 USD")
parser.add_argument("--headless", action="store_true", default=True, help="以无头模式运行")
parser.add_argument(
    "--out_dir",
    type=str,
    default="assets/downloaded_furniture",
    help="下载到的本地目录（相对项目根目录）",
)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
out_dir = (PROJECT_ROOT / args.out_dir).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

FURNITURE_USD = [
    "Environments/Office/Props/SM_Sofa.usd",
    "Environments/Office/Props/SM_Armchair.usd",
    "Environments/Office/Props/SM_TableA.usd",
]

print("=" * 80)
print("📥 开始下载家具 USD")
print("=" * 80)
print(f"Nucleus: {ISAAC_NUCLEUS_DIR}")
print(f"输出目录: {out_dir}")
print()

success = 0
for rel_path in FURNITURE_USD:
    src = f"{ISAAC_NUCLEUS_DIR}/{rel_path}"
    dst_path = out_dir / Path(rel_path).name
    dst = dst_path.as_uri()

    print(f"→ 下载: {src}")
    print(f"  到: {dst_path}")

    result = omni.client.copy(src, dst)
    if result == omni.client.Result.OK:
        print("  ✅ 成功")
        success += 1
    else:
        print(f"  ❌ 失败: {result}")
    print()

print(f"完成: {success}/{len(FURNITURE_USD)}")

simulation_app.close()
