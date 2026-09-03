#!/usr/bin/env python3
"""Upload sim-ready scenes to HF as one tar per scene, streaming.

Loose files would put ~300k objects in the repo (HF recommends < 100k and the
urbanverse-scene toolkit already works tar-per-scene, as CraftBench does), so
each verified scene becomes ``<scene>/scene.tar`` (uncompressed; extract in
place to get ``World0.usd`` + ``SubUSDs/``). Tars are built one at a time in
--staging, uploaded, and deleted, so disk use stays at a couple of scenes.
Scenes whose tar is already on the hub with the same size are skipped, so a
killed run resumes.

    python scripts/upload_scene_tars.py \
        --local-dir "/media/hollis/Extreme SSD1/mnt_new/UrbanVerseAll/Collected_Urban_Cousins_Training_Scenes" \
        --manifest scenes/training_simready_manifest.json \
        --repo UCLA-VAIL/UrbanVerse-Training-Scenes-Sim-Ready \
        --staging "/media/hollis/Extreme SSD1/urbanverse_scene_tars"
"""
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi


def log(msg: str) -> None:
    print(f"[tars {time.strftime('%F %T')}] {msg}", flush=True)


def build_tar(scene_dir: Path, tar_path: Path) -> None:
    tmp = tar_path.with_suffix(".tar.part")
    # everything in the scene dir except our own README / previews / cache dirs
    subprocess.run(["tar", "-C", str(scene_dir), "--exclude=.cache", "--exclude=README.md",
                    "-cf", str(tmp), "."], check=True)
    tmp.replace(tar_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--local-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--staging", required=True)
    p.add_argument("--extra", action="append", default=[],
                   help="Extra local files to upload to the repo root (README.md, previews dir); repeatable.")
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--retries", type=int, default=5)
    a = p.parse_args()

    api = HfApi()
    local, staging = Path(a.local_dir), Path(a.staging)
    staging.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(a.manifest).read_text())
    scenes = sorted(s for s, r in manifest["scenes"].items() if r.get("ok") and (local / s).is_dir())

    on_hub = {}
    for f in api.dataset_info(a.repo, files_metadata=True).siblings:
        on_hub[f.rfilename] = f.size or 0
    todo = [s for s in scenes if f"{s}/scene.tar" not in on_hub]
    log(f"{len(scenes)} verified scenes, {len(scenes) - len(todo)} already on hub, {len(todo)} to upload")

    def one(scene: str) -> tuple[str, bool, str]:
        tar_path = staging / f"{scene}.tar"
        for attempt in range(1, a.retries + 1):
            try:
                if not tar_path.exists():
                    build_tar(local / scene, tar_path)
                size = tar_path.stat().st_size
                api.upload_file(path_or_fileobj=str(tar_path), path_in_repo=f"{scene}/scene.tar",
                                repo_id=a.repo, repo_type="dataset",
                                commit_message=f"add {scene} ({size / 1e9:.2f} GB)")
                tar_path.unlink(missing_ok=True)
                return scene, True, f"{size / 1e9:.2f} GB"
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:120]}"
                log(f"{scene}: attempt {attempt} failed: {err}")
                time.sleep(min(60 * attempt, 300))
        tar_path.unlink(missing_ok=True)
        return scene, False, err

    n_ok = n_bad = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for i, (scene, ok, info) in enumerate(ex.map(one, todo), 1):
            n_ok += ok
            n_bad += not ok
            el = (time.perf_counter() - t0) / 60
            log(f"{i}/{len(todo)} {'ok ' if ok else 'BAD'} {scene} {info}  ({el:.0f} min elapsed)")

    for extra in a.extra:
        ep = Path(extra)
        if ep.is_dir():
            api.upload_folder(folder_path=str(ep), path_in_repo=ep.name, repo_id=a.repo, repo_type="dataset",
                              commit_message=f"add {ep.name}")
        elif ep.is_file():
            api.upload_file(path_or_fileobj=str(ep), path_in_repo=ep.name, repo_id=a.repo, repo_type="dataset",
                            commit_message=f"add {ep.name}")
    log(f"done: {n_ok} uploaded, {n_bad} failed, {len(scenes) - len(todo)} were already there")
    if n_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
