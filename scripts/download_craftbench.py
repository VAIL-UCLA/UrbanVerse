#!/usr/bin/env python3
"""Download the sim-ready CraftBench scenes from HuggingFace, extracted and verified.

    python scripts/download_craftbench.py --root /path/to/urbanverse_scenes             # all 12
    python scripts/download_craftbench.py --root ... --scene scene_03 --scene scene_10  # by id
    python scripts/download_craftbench.py --list

Scenes land in the ``urbanverse-scene`` toolkit layout, so ``uvs.set(root)``
finds them instead of fetching the original (pre-sim-ready) release::

    <root>/CraftBench/<scene>/
    ├── Collected_export_version/export_version.usd   # open in Isaac Sim >= 5
    ├── cam0_to_world.txt
    └── preview_{front,topdown,closeup}.png, preview_video.mp4

Each ``Collected_export_version.tar`` is deleted once extracted (``--keep-tar``
keeps it), and every layer the sim-ready conversion changed is checked against
its sha256 in ``scenes/craftbench_simready_manifest.json``. Re-running skips
finished scenes, so a killed download just resumes; a scene extracted from an
earlier release (a changed layer no longer matches the manifest) is fetched and
extracted again. On a complete, current tree it only re-verifies.
"""
import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
from collections import defaultdict
from pathlib import Path

# A one-shot pull gains nothing from hf_xet's chunk cache, which lives under
# ~/.cache and can fill a small home disk.
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

REPO = "UCLA-VAIL/UrbanVerse-CraftBench-Sim-Ready"
MANIFEST = Path(__file__).resolve().parents[1] / "scenes" / "craftbench_simready_manifest.json"
TAR = "Collected_export_version.tar"
USD = "Collected_export_version/export_version.usd"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(scene_dir: Path) -> None:
    # Extract beside the target and rename into place, so a killed run never
    # leaves a half-filled Collected_export_version/ that looks finished.
    tmp = scene_dir / ".extracting"
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(scene_dir / TAR) as tf:
        tf.extractall(tmp, filter="data")
    shutil.rmtree(scene_dir / "Collected_export_version", ignore_errors=True)  # an earlier release's copy
    (tmp / "Collected_export_version").rename(scene_dir / "Collected_export_version")
    tmp.rmdir()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="Toolkit cache root; scenes go to <root>/CraftBench/. Required unless --list.")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--scene", action="append", default=[],
                   help="Scene name or id prefix (e.g. scene_03); repeatable. Default: all.")
    p.add_argument("--list", action="store_true", help="List scenes with sizes and exit.")
    p.add_argument("--keep-tar", action="store_true", help="Keep each scene tar after extracting it.")
    p.add_argument("--manifest", default=str(MANIFEST),
                   help="Sim-ready manifest to verify against; pass '' to skip the sha256 check.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=20)
    a = p.parse_args()

    info = HfApi().dataset_info(a.repo, files_metadata=True)
    files = defaultdict(dict)  # scene -> {file name: bytes}
    for f in info.siblings:
        if "/" in f.rfilename:
            scene, name = f.rfilename.split("/", 1)
            files[scene][name] = f.size or 0

    if a.list:
        for s, fs in sorted(files.items()):
            print(f"{sum(fs.values()) / 1e9:6.2f} GB  {s}")
        print(f"{sum(sum(fs.values()) for fs in files.values()) / 1e9:6.2f} GB  total, {len(files)} scenes")
        return
    if not a.root:
        p.error("--root is required (unless --list)")

    scenes = set()
    for want in a.scene:
        hits = [s for s in files if s == want or s.startswith(want + "_")]
        if len(hits) != 1:
            raise SystemExit(f"--scene {want!r} matches {len(hits)} scenes; see --list")
        scenes.update(hits)
    scenes = sorted(scenes) or sorted(files)

    bench = Path(a.root).expanduser().resolve() / "CraftBench"
    manifest = json.loads(Path(a.manifest).read_text())["scenes"] if a.manifest else None
    stale = [s for s in scenes if manifest and (bench / s / USD).is_file() and any(
        not (bench / e["layer"]).is_file() or sha256(bench / e["layer"]) != e["sha256"]
        for e in manifest.get(s, {}).get("layers_changed", []))]
    for s in stale:
        print(f"[download] {s}: extracted from an earlier release, fetching it again")
    todo = [s for s in scenes if not (bench / s / USD).is_file() or s in stale]
    # Tars only for scenes not yet extracted, else a re-run would pull them all again.
    wanted = [(s, n) for s in scenes for n in files[s] if n != TAR or s in todo]
    fetch = sum(files[s][n] for s, n in wanted
                if not (bench / s / n).is_file() or (bench / s / n).stat().st_size != files[s][n])
    tars = [files[s][TAR] for s in todo]
    need = fetch + (sum(tars) if a.keep_tar else max(tars, default=0))  # each tar extracts to ~its own size
    bench.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(bench).free
    if need > free:
        raise SystemExit(f"need ~{need / 1e9:.1f} GB under {bench}, only {free / 1e9:.1f} GB free")
    print(f"[download] {a.repo}@{info.sha[:8]}: {len(scenes)} scene(s), {fetch / 1e9:.1f} GB to fetch, "
          f"{len(todo)} to extract -> {bench}")

    if fetch:
        # Same transient xet CAS errors as download_training_scenes.py; finished files are skipped on retry.
        for attempt in range(1, a.retries + 1):
            try:
                snapshot_download(
                    repo_id=a.repo, repo_type="dataset", revision=info.sha, local_dir=str(bench),
                    allow_patterns=[f"{s}/{n}" for s, n in wanted] + ["README.md"], max_workers=a.workers,
                )
                break
            except Exception as e:  # noqa: BLE001
                if attempt == a.retries:
                    raise
                wait = min(60 * attempt, 300)
                print(f"[download] attempt {attempt} failed: {type(e).__name__}: {str(e)[:160]}\n"
                      f"[download] retrying in {wait}s")
                time.sleep(wait)

    for s in scenes:
        if s in todo:
            print(f"[extract] {s}")
            extract(bench / s)
        if not a.keep_tar:  # right away, so peak disk is one tar over the final size (as budgeted above)
            (bench / s / TAR).unlink(missing_ok=True)

    n_bad = 0
    for s in scenes:
        d = bench / s
        issues = [n for n in files[s] if n != TAR and not (d / n).is_file()]
        if not (d / USD).is_file():
            issues.append(USD)
        if manifest is not None:
            if s not in manifest:
                issues.append("not in manifest")
            for layer in manifest.get(s, {}).get("layers_changed", []):
                q = bench / layer["layer"]
                if not q.is_file() or sha256(q) != layer["sha256"]:
                    issues.append(f"sha256 mismatch: {layer['layer'][len(s) + 1:]}")
        n_bad += bool(issues)
        print(f"[verify] {'BAD' if issues else 'ok '} {s}" + "".join(f"\n           {i}" for i in issues))

    print(f"[download] {len(scenes) - n_bad}/{len(scenes)} scene(s) ready under {bench}\n"
          f"[download] urbanverse-scene users: uvs.set({str(bench.parent)!r})")
    if n_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
