#!/usr/bin/env python3
"""Download UrbanVerse training scenes from HuggingFace, resumably.

    python scripts/download_training_scenes.py --local-dir /path/to/scenes            # all 107
    python scripts/download_training_scenes.py --local-dir ... --scene Africa_Egypt_Cairo_walk_02_Cousin_01
    python scripts/download_training_scenes.py --local-dir ... --list

Re-running skips files already present, so a killed download just resumes.
"""
import argparse
import re
import time
from collections import defaultdict

from huggingface_hub import HfApi, snapshot_download

REPO = "UCLA-VAIL/UrbanVerse-Training-Scenes"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=REPO)
    p.add_argument("--local-dir", required=True)
    p.add_argument("--scene", action="append", default=[], help="Scene dir name; repeatable. Default: all.")
    p.add_argument("--walk", action="append", default=[], help="Source walk prefix (all its cousins); repeatable.")
    p.add_argument("--list", action="store_true", help="List scenes with sizes and exit.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=20)
    a = p.parse_args()

    api = HfApi()
    info = api.dataset_info(a.repo, files_metadata=True)
    by_scene = defaultdict(int)
    for f in info.siblings:
        if "/" in f.rfilename:
            by_scene[f.rfilename.split("/")[0]] += f.size or 0

    if a.list:
        for s, b in sorted(by_scene.items()):
            print(f"{b / 1e9:6.2f} GB  {s}")
        print(f"{sum(by_scene.values()) / 1e9:6.1f} GB  total, {len(by_scene)} scenes")
        return

    scenes = set(a.scene)
    for w in a.walk:
        scenes.update(s for s in by_scene if re.sub(r"_Cousin_\d+$", "", s) == w)
    unknown = scenes - set(by_scene)
    if unknown:
        raise SystemExit(f"unknown scene(s): {sorted(unknown)}")
    patterns = [f"{s}/*" for s in sorted(scenes)] if scenes else None
    gb = sum(by_scene[s] for s in scenes) / 1e9 if scenes else sum(by_scene.values()) / 1e9
    print(f"[download] {len(scenes) or len(by_scene)} scene(s), {gb:.1f} GB -> {a.local_dir}")

    # Multi-hour pulls occasionally die on a transient xet CAS error (seen: a
    # 401 mid-stream); snapshot_download skips finished files, so just retry.
    for attempt in range(1, a.retries + 1):
        try:
            snapshot_download(
                repo_id=a.repo, repo_type="dataset", local_dir=a.local_dir,
                allow_patterns=patterns, max_workers=a.workers,
            )
            break
        except Exception as e:  # noqa: BLE001
            if attempt == a.retries:
                raise
            wait = min(60 * attempt, 300)
            print(f"[download] attempt {attempt} failed: {type(e).__name__}: {str(e)[:160]}\n"
                  f"[download] retrying in {wait}s")
            time.sleep(wait)
    print("[download] done")


if __name__ == "__main__":
    main()
