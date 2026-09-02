#!/usr/bin/env python3
"""Upload a sim-ready scene tree to a HuggingFace dataset repo, resumably.

    python scripts/upload_simready_scenes.py \
        --local-dir "/media/hollis/Extreme SSD1/UrbanVerse-Training-Scenes" \
        --repo UCLA-VAIL/UrbanVerse-Training-Scenes-Sim-Ready \
        --manifest scenes/training_simready_manifest.json

Only scenes the manifest records as converted and verified (``ok``) are
uploaded - everything else under --local-dir is ignored. Uses
``upload_large_folder``, which chunks, hashes and retries per file and keeps
its own progress cache in ``<local-dir>/.cache/huggingface``, so a killed
run resumes where it stopped.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--local-dir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--manifest", required=True, help="convert_scenes_simready.py manifest.")
    p.add_argument("--extra", action="append", default=[],
                   help="Extra root-level file/dir names to include (e.g. README.md); repeatable.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    manifest = json.loads(Path(a.manifest).read_text())
    ok = sorted(s for s, r in manifest["scenes"].items() if r.get("ok"))
    skipped = sorted(s for s, r in manifest["scenes"].items() if not r.get("ok"))
    local = Path(a.local_dir)
    missing = [s for s in ok if not (local / s).is_dir()]
    if missing:
        raise SystemExit(f"{len(missing)} ok scene(s) not found under {local}: {missing[:3]}")

    patterns = [f"{s}/**" for s in ok] + [f"{e}" for e in a.extra] + [f"{e}/**" for e in a.extra]
    size = sum(f.stat().st_size for s in ok for f in (local / s).rglob("*") if f.is_file())
    print(f"[upload] {len(ok)} verified scene(s), {size / 1e9:.1f} GB -> {a.repo}")
    if skipped:
        print(f"[upload] not uploading {len(skipped)} unverified scene(s): {skipped}")
    if a.dry_run:
        return

    api = HfApi()
    api.upload_large_folder(
        repo_id=a.repo, repo_type="dataset", folder_path=str(local),
        allow_patterns=patterns, num_workers=a.workers, print_report=True,
    )
    print("[upload] done")


if __name__ == "__main__":
    main()
