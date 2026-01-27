#!/usr/bin/env python3
"""Download a small set of USD files referenced by the Go2Pvcnn configs.

This script downloads three USD files used by `go2_lidar_env_cfg.py` into the
local `assets/` directory, preserving the remote subdirectory structure under
`IsaacLab/...` when possible.

Usage: run from the repository root:
    python3 Go2Pvcnn/scripts/download_usd.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
from urllib.error import URLError, HTTPError


URLS = [
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Environments/Office/Props/SM_Sofa.usd",
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Environments/Office/Props/SM_Armchair.usd",
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Environments/Office/Props/SM_TableA.usd",
]


def local_path_for_url(base_assets_dir: str, url: str) -> str:
    """Compute a local file path under base_assets_dir for the given URL.

    We try to preserve the path portion starting from '/IsaacLab/'. If that
    substring is not present, fall back to using the URL basename.
    """
    marker = "/IsaacLab/"
    idx = url.find(marker)
    if idx != -1:
        suffix = url[idx + 1 :]  # drop leading slash
        return os.path.join(base_assets_dir, suffix)
    # fallback
    return os.path.join(base_assets_dir, os.path.basename(url))


def download(url: str, target: str, timeout: int = 30) -> bool:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
        print(f"Downloading {url} -> {target}")
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                print(f"Failed to download {url}: HTTP {resp.status}")
                return False
            data = resp.read()
        with open(target, "wb") as f:
            f.write(data)
        print(f"Saved {target} ({len(data)} bytes)")
        return True
    except HTTPError as e:
        print(f"HTTP error for {url}: {e}")
    except URLError as e:
        print(f"URL error for {url}: {e}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(repo_root, "assets")

    success = True
    for url in URLS:
        target = local_path_for_url(assets_dir, url)
        ok = download(url, target)
        if not ok:
            success = False

    if not success:
        print("Some downloads failed. Check network access or try manually.")
        return 2

    print("All downloads completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())