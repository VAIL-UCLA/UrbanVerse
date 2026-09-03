#!/usr/bin/env python3
"""Stream the UrbanVerse-100K GLB corpus through GLB -> USD conversion and HF upload.

The corpus is ~1 TB of GLB and the USD output is ~1.8x that, so it never exists
locally in full. This loop takes the next --batch-size unconverted uids, converts
them with scripts/profile_glb_to_usd.py (Isaac Lab MeshConverter, headless),
validates every stage, uploads the batch to the HF dataset, records the uids as
done, deletes the local batch, bumps the README progress bar and pushes.
Conversion of batch k+1 overlaps the upload of batch k.

    python scripts/convert_assets_batched.py \
        --glb-root "/media/hollis/Extreme SSD1/mnt_new/UrbanVerseAll/assets_std_glb_flat" \
        --work-dir "/media/hollis/Extreme SSD1/urbanverse_assets_simready_work" \
        --repo UCLA-VAIL/UrbanVerse-Assets-Sim-ready \
        --state scenes/assets_batches_state.json --batch-size 4000

Resumable: killed mid-batch, it re-converts that batch (outputs are skipped if
present) and continues. --no-push keeps git out of it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PY = sys.executable


def log(msg: str) -> None:
    print(f"[assets {time.strftime('%F %T')}] {msg}", flush=True)


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"uploaded": [], "failed": {}, "batches": []}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(path)


def already_on_hub(repo: str) -> set[str]:
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(repo)
    uids = set()
    for f in info.siblings:
        parts = f.rfilename.split("/")
        if len(parts) == 3 and parts[0] == "usd" and parts[2].endswith(".usd") and parts[1].startswith("std_"):
            uids.add(parts[1][4:])
    return uids


def validate_batch(usd_root: Path, uids: list[str]) -> tuple[list[str], dict[str, str], dict[str, float]]:
    """Light per-stage check: opens, has mesh + rigid body + collision; reports off-ground bases."""
    from pxr import Usd, UsdGeom, UsdPhysics
    good, bad, warn = [], {}, {}
    for uid in uids:
        p = usd_root / f"std_{uid}" / f"std_{uid}.usd"
        if not p.is_file() or p.stat().st_size == 0:
            bad[uid] = "no output"
            continue
        try:
            st = Usd.Stage.Open(str(p))
            prims = list(st.Traverse())
            if not any(x.IsA(UsdGeom.Mesh) for x in prims):
                bad[uid] = "no mesh"; continue
            if not any(x.HasAPI(UsdPhysics.RigidBodyAPI) for x in prims):
                bad[uid] = "no RigidBodyAPI"; continue
            if not any(x.HasAPI(UsdPhysics.CollisionAPI) for x in prims):
                bad[uid] = "no CollisionAPI"; continue
            # Base height is reported, not enforced: a few source GLBs are not
            # bottomed (floating signs, hanging objects) and the conversion is
            # still faithful to them.
            b = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]) \
                .ComputeWorldBound(st.GetPseudoRoot()).ComputeAlignedRange()
            z0 = b.GetMin()[2]
            if abs(z0) > 0.05:
                warn[uid] = round(float(z0), 3)
            good.append(uid)
        except Exception as e:  # noqa: BLE001
            bad[uid] = f"{type(e).__name__}: {e}"
    return good, bad, warn


_CONVERTING_RE = re.compile(r"^\s+\[\d+/\d+\] \S+ / ([0-9a-f]+)\s*$", re.M)


def _last_uid_in_log(log: Path) -> str | None:
    """uid of the asset a converter shard was working on when its log ends."""
    hits = _CONVERTING_RE.findall(log.read_text(errors="replace"))
    return hits[-1] if hits else None


def _run_shards(a, batch_dir: Path, uid_file: Path, prof_dir: Path, tag: str) -> list[tuple[int, Path]]:
    """Launch --convert-shards headless workers; return (returncode, log) per shard."""
    procs = []
    for i in range(1, a.convert_shards + 1):
        cmd = [PY, str(HERE / "profile_glb_to_usd.py"), "--glb-root", a.glb_root, "--flat",
               "--uid-list", str(uid_file), "--out-dir", str(batch_dir / "usd"),
               "--profile-dir", str(prof_dir), "--headless"]
        if a.convert_shards > 1:
            cmd += ["--shard", f"{i}/{a.convert_shards}"]
        log = batch_dir / f"convert{tag}_{i}.log"
        f = open(log, "w")
        procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT), f, log))
    out = []
    for pr, f, log in procs:
        pr.wait()
        f.close()
        out.append((pr.returncode, log))
    return out


def convert_batch(a, batch_dir: Path, uids: list[str]) -> tuple[Path, dict[str, str]]:
    """Run the batch through concurrent headless Isaac Sim workers.

    A GLB that crashes Kit's converter (segfault, no Python exception) kills
    its whole shard, so every crashed shard is retried without the asset it
    died on, until all shards exit cleanly or the retry budget is spent.
    Returns the profile dir and {uid: reason} for assets given up on.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    uid_file = batch_dir / "uids.txt"
    uid_file.write_text("\n".join(uids) + "\n")
    prof_dir = batch_dir / "profile"
    crashed: dict[str, str] = {}
    results = _run_shards(a, batch_dir, uid_file, prof_dir, "")
    for attempt in range(1, a.crash_retries + 1):
        bad = [(rc, log) for rc, log in results if rc != 0]
        if not bad:
            break
        for rc, log in bad:
            uid = _last_uid_in_log(log)
            if uid is not None:
                crashed[uid] = f"converter crashed (exit {rc}) on this asset"
            print(f"[assets] shard {log.name} exited {rc} on {uid}; retry {attempt}/{a.crash_retries}",
                  flush=True)
        # --skip-existing makes the rerun pick up only what is still missing.
        retry_file = batch_dir / f"uids_retry{attempt}.txt"
        retry_file.write_text("\n".join(u for u in uids if u not in crashed) + "\n")
        results = _run_shards(a, batch_dir, retry_file, prof_dir, f"_retry{attempt}")
    else:
        still = [log.name for rc, log in results if rc != 0]
        if still:
            print(f"[assets] shards still crashing after {a.crash_retries} retries: {still}", flush=True)
    return prof_dir, crashed


def upload_batch(a, batch_dir: Path) -> None:
    from huggingface_hub import HfApi
    HfApi().upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(batch_dir),
                                allow_patterns=["usd/**"], num_workers=a.upload_workers,
                                print_report=False)


def bump_progress(total_done: int, push: bool) -> None:
    subprocess.run([PY, str(HERE / "update_progress.py"), "--assets-done", str(total_done)],
                   cwd=REPO_ROOT, check=False, capture_output=True)
    if not push:
        return
    subprocess.run(["git", "add", "README.md", "scenes/assets_batches_state.json"], cwd=REPO_ROOT, check=False)
    r = subprocess.run(["git", "commit", "-q", "-m", f"progress: assets {total_done:,} / 102,445 sim-ready"],
                       cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if r.returncode == 0:
        subprocess.run(["git", "push", "-q", "origin", "main"], cwd=REPO_ROOT, check=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glb-root", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--batch-size", type=int, default=4000)
    p.add_argument("--max-batches", type=int, default=0, help="0 = run until the corpus is done")
    p.add_argument("--upload-workers", type=int, default=6)
    p.add_argument("--convert-shards", type=int, default=3,
                   help="Concurrent Isaac Sim converter processes per batch (CPU-bound; one GPU is enough).")
    p.add_argument("--crash-retries", type=int, default=5,
                   help="How many times to relaunch converter shards that crashed on a bad GLB")
    p.add_argument("--no-push", action="store_true")
    a = p.parse_args()

    state_path = Path(a.state)
    state = load_state(state_path)
    all_uids = sorted(f.stem[4:] for f in Path(a.glb_root).iterdir()
                      if f.suffix == ".glb" and f.stem.startswith("std_"))
    hub = already_on_hub(a.repo)
    done = set(state["uploaded"]) | hub
    todo = [u for u in all_uids if u not in done and u not in state["failed"]]
    log(f"corpus {len(all_uids):,} uids; on hub {len(hub):,}; recorded uploaded {len(state['uploaded']):,}; "
        f"failed {len(state['failed']):,}; to do {len(todo):,}")
    # record hub-only uids so the count is right even if state was lost
    state["uploaded"] = sorted(done)
    save_state(state_path, state)

    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    upload_thread: threading.Thread | None = None
    upload_err: list[str] = []
    batch_no = len(state["batches"])

    while todo and (not a.max_batches or batch_no - len(state["batches"]) < a.max_batches):
        uids = todo[: a.batch_size]
        batch_dir = work / f"batch_{batch_no:04d}"
        t0 = time.perf_counter()
        log(f"batch {batch_no}: converting {len(uids):,} assets -> {batch_dir}")
        prof_dir, crashed = convert_batch(a, batch_dir, uids)
        t_conv = time.perf_counter() - t0
        good, bad, warn = validate_batch(batch_dir / "usd", uids)
        bad.update(crashed)  # "no output" -> the real reason
        good = [u for u in good if u not in crashed]
        if warn:
            state.setdefault("off_ground", {}).update(warn)
        for u, why in bad.items():
            shutil.rmtree(batch_dir / "usd" / f"std_{u}", ignore_errors=True)
        log(f"batch {batch_no}: {len(good):,} ok, {len(bad):,} bad, {len(warn):,} off-ground, "
            f"convert {t_conv / 60:.1f} min")
        summary, out_mb, secs, n_ok = {}, 0.0, 0.0, 0
        for pj in sorted(prof_dir.glob("profile_*.json")):
            sm = json.loads(pj.read_text()).get("summary", {})
            out_mb += sm.get("output_mb_total", 0) or 0
            secs += sm.get("convert_s_sum", 0) or 0
            n_ok += sm.get("assets_ok", 0) or 0
        if n_ok:
            summary = {"mean_s_per_asset": round(secs / n_ok, 3), "output_mb_total": round(out_mb, 1)}

        # wait for the previous upload before starting this one (one upload at a time)
        if upload_thread is not None:
            upload_thread.join()
            if upload_err:
                log(f"previous upload failed: {upload_err[-1]} - stopping")
                sys.exit(1)

        def _upload(bdir=batch_dir, guids=good, buids=bad, bno=batch_no, summ=summary, tconv=t_conv):
            try:
                t1 = time.perf_counter()
                upload_batch(a, bdir)
                t_up = time.perf_counter() - t1
                state["uploaded"] = sorted(set(state["uploaded"]) | set(guids))
                state["failed"].update(buids)
                state["batches"].append({
                    "batch": bno, "ok": len(guids), "bad": len(buids),
                    "convert_min": round(tconv / 60, 1), "upload_min": round(t_up / 60, 1),
                    "mean_s_per_asset": summ.get("mean_s_per_asset"),
                    "output_mb": summ.get("output_mb_total"),
                    "finished": time.strftime("%F %T"),
                })
                save_state(state_path, state)
                shutil.rmtree(bdir, ignore_errors=True)
                total = len(state["uploaded"])
                bump_progress(total, push=not a.no_push)
                log(f"batch {bno}: uploaded in {t_up / 60:.1f} min; total sim-ready {total:,}")
            except Exception as e:  # noqa: BLE001
                upload_err.append(f"{type(e).__name__}: {e}")

        upload_thread = threading.Thread(target=_upload, daemon=False)
        upload_thread.start()
        todo = todo[len(uids):]
        batch_no += 1

    if upload_thread is not None:
        upload_thread.join()
        if upload_err:
            log(f"last upload failed: {upload_err[-1]}")
            sys.exit(1)
    log("ALL DONE" if not todo else f"stopped with {len(todo):,} to do")


if __name__ == "__main__":
    main()
