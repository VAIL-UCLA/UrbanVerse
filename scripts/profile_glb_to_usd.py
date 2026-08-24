#!/usr/bin/env python3
"""Profile the UrbanVerse GLB -> USD (sim-ready) conversion.

Same conversion path as ``urbanverse_asset/_glb_to_usd.py`` (Isaac Lab's
``MeshConverter``), but records per-asset wall time, input/output sizes and
geometry counts so the batch throughput can be measured instead of guessed.

Must run inside an Isaac Sim + Isaac Lab environment::

    python scripts/profile_glb_to_usd.py \
        --glb-root /path/to/urban_verse_assets_seq1 \
        --out-dir  /path/to/usd_out \
        --profile-dir profiles --limit 10 --headless

Or against an explicit task list (same schema as ``_glb_to_usd.py``)::

    python scripts/profile_glb_to_usd.py --tasks tasks.json --headless

Writes ``<profile-dir>/profile_<stamp>.{json,csv,md}``.
"""

import importlib.util
import sys

_missing = [m for m in ("isaacsim", "isaaclab") if importlib.util.find_spec(m) is None]
if _missing:
    print(
        f"ERROR: requires Isaac Sim + Isaac Lab, missing: {', '.join(_missing)}\n"
        f"  Run with Isaac Lab's python, e.g.\n"
        f"    isaaclab -p {sys.argv[0]} --glb-root <dir> --headless",
        file=sys.stderr,
    )
    sys.exit(1)

import argparse
import time

# Isaac Sim's shutdown can bypass interpreter cleanup, so don't let progress
# output sit in a block buffer and get lost when stdout is redirected to a log.
sys.stdout.reconfigure(line_buffering=True)

_T_PROC_START = time.perf_counter()

parser = argparse.ArgumentParser(description="Profile UrbanVerse GLB -> USD conversion.")
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--glb-root", type=str, help="Root of <category>/<uid>/<uid>.glb assets.")
src.add_argument("--tasks", type=str, help="Task JSON ({'tasks':[{glb,usd_dir,usd_name}]}).")
parser.add_argument("--out-dir", type=str, default=None, help="USD output root (--glb-root mode).")
parser.add_argument("--profile-dir", type=str, default="profiles", help="Where to write profiles.")
parser.add_argument("--limit", type=int, default=0, help="Max assets to convert (0 = all).")
parser.add_argument("--glb-name", type=str, default="{uid}.glb",
                    help="GLB filename template inside each uid dir.")
parser.add_argument("--collision-approximation", type=str, default="convexDecomposition",
                    choices=["convexDecomposition", "convexHull", "boundingCube",
                             "boundingSphere", "meshSimplification", "none"])
parser.add_argument("--make-instanceable", action="store_true", default=False)
parser.add_argument("--mass", type=float, default=None)
parser.add_argument("--corpus-assets", type=int, default=102_445,
                    help="Full corpus size, used for the ETA extrapolation.")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_T_APP_START = time.perf_counter()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_STARTUP_S = time.perf_counter() - _T_APP_START

import csv  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
from isaaclab.sim.schemas import schemas_cfg  # noqa: E402

# Isaac Sim 4.5 took ``collision_approximation="convexDecomposition"``; Isaac Sim
# >= 5 replaced it with a ``mesh_collision_props`` cfg object. Support both.
_MESH_CFG_FIELDS = {f.name for f in dataclasses.fields(MeshConverterCfg)}
_LEGACY_COLLISION_API = "collision_approximation" in _MESH_CFG_FIELDS

_COLLISION_CFG = {
    "convexDecomposition": "ConvexDecompositionPropertiesCfg",
    "convexHull": "ConvexHullPropertiesCfg",
    "boundingCube": "BoundingCubePropertiesCfg",
    "boundingSphere": "BoundingSpherePropertiesCfg",
    "meshSimplification": "TriangleMeshSimplificationPropertiesCfg",
}


def collision_kwargs(approximation: str) -> dict:
    """Collision arguments for whichever MeshConverterCfg API is installed."""
    if _LEGACY_COLLISION_API:
        return {"collision_approximation": approximation}
    if approximation == "none":
        return {}
    cls_name = _COLLISION_CFG.get(approximation)
    cls = getattr(schemas_cfg, cls_name, None) if cls_name else None
    if cls is None:
        raise SystemExit(
            f"collision approximation '{approximation}' is unavailable in this "
            f"Isaac Lab build (no schemas_cfg.{cls_name})"
        )
    return {"mesh_collision_props": cls()}


def dir_bytes(path: Path) -> int:
    """Total bytes of a converted asset directory (USD + textures/materials)."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def usd_geometry(usd_path: str):
    """(mesh prim count, triangle count) of a converted USD, or (None, None)."""
    try:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(usd_path)
        meshes = tris = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            meshes += 1
            counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or []
            # a face with n verts triangulates into n-2 triangles
            tris += sum(max(int(c) - 2, 0) for c in counts)
        return meshes, tris
    except Exception:
        return None, None


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_tasks():
    """Task dicts from either --tasks or a --glb-root scan."""
    if args_cli.tasks:
        data = json.loads(Path(args_cli.tasks).read_text())
        tasks = data["tasks"]
        for t in tasks:
            t.setdefault("uid", Path(t["glb"]).stem)
            t.setdefault("category", Path(t["glb"]).parent.parent.name)
        return tasks

    root = Path(args_cli.glb_root).resolve()
    if args_cli.out_dir is None:
        raise SystemExit("--out-dir is required with --glb-root")
    out_root = Path(args_cli.out_dir).resolve()

    tasks = []
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for uid_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            uid = uid_dir.name
            glb = uid_dir / args_cli.glb_name.format(uid=uid)
            if not glb.exists():
                continue
            tasks.append({
                "uid": uid,
                "category": cat_dir.name,
                "glb": str(glb),
                "usd_dir": str(out_root / f"std_{uid}"),
                "usd_name": f"std_{uid}.usd",
            })

    # Round-robin across categories so a truncated run stays representative.
    by_cat = {}
    for t in tasks:
        by_cat.setdefault(t["category"], []).append(t)
    interleaved, cats = [], list(by_cat)
    while any(by_cat[c] for c in cats):
        for c in cats:
            if by_cat[c]:
                interleaved.append(by_cat[c].pop(0))
    return interleaved


def main() -> None:
    tasks = build_tasks()
    if args_cli.limit:
        tasks = tasks[: args_cli.limit]
    if not tasks:
        raise SystemExit("no GLB assets found")

    profile_dir = Path(args_cli.profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mass_props = (schemas_cfg.MassPropertiesCfg(mass=args_cli.mass)
                  if args_cli.mass is not None else None)
    rigid_props = schemas_cfg.RigidBodyPropertiesCfg(
        rigid_body_enabled=True, kinematic_enabled=True,
    )
    collision_props = schemas_cfg.CollisionPropertiesCfg(
        collision_enabled=args_cli.collision_approximation != "none",
    )
    coll_kwargs = collision_kwargs(args_cli.collision_approximation)
    print(f"[profile] MeshConverterCfg collision API: "
          f"{'legacy (<=4.5)' if _LEGACY_COLLISION_API else 'mesh_collision_props (>=5)'}")

    records = []
    t_batch = time.perf_counter()

    for i, task in enumerate(tasks, 1):
        glb_path, usd_dir, usd_name = task["glb"], task["usd_dir"], task["usd_name"]
        os.makedirs(usd_dir, exist_ok=True)
        glb_bytes = Path(glb_path).stat().st_size

        print(f"\n{'|' + '-' * 80 + '|'}")
        print(f"  [{i}/{len(tasks)}] {task.get('category', '?')} / {task.get('uid', '?')}")
        print(f"  GLB: {glb_bytes / 1e6:.1f} MB -> {os.path.join(usd_dir, usd_name)}")
        print(f"{'|' + '-' * 80 + '|'}")

        rec = {
            "index": i,
            "uid": task.get("uid"),
            "category": task.get("category"),
            "glb": glb_path,
            "glb_mb": round(glb_bytes / 1e6, 3),
        }

        t0 = time.perf_counter()
        try:
            cfg = MeshConverterCfg(
                mass_props=mass_props,
                rigid_props=rigid_props,
                collision_props=collision_props,
                asset_path=glb_path,
                force_usd_conversion=True,
                usd_dir=usd_dir,
                usd_file_name=usd_name,
                make_instanceable=args_cli.make_instanceable,
                **coll_kwargs,
            )
            usd_path = MeshConverter(cfg).usd_path
            elapsed = time.perf_counter() - t0
            meshes, tris = usd_geometry(usd_path)
            rec.update({
                "ok": True, "usd": usd_path, "seconds": round(elapsed, 3),
                "usd_mb": round(dir_bytes(Path(usd_dir)) / 1e6, 3),
                "meshes": meshes, "triangles": tris, "error": None,
            })
            print(f"  OK {elapsed:.1f}s -> {rec['usd_mb']:.1f} MB, {tris} tris")
        except Exception as e:  # noqa: BLE001 - profiling must survive any failure
            elapsed = time.perf_counter() - t0
            rec.update({
                "ok": False, "usd": None, "seconds": round(elapsed, 3),
                "usd_mb": 0.0, "meshes": None, "triangles": None, "error": str(e),
            })
            print(f"  FAILED after {elapsed:.1f}s: {e}")

        records.append(rec)

    batch_s = time.perf_counter() - t_batch
    ok = [r for r in records if r["ok"]]
    conv_s = sum(r["seconds"] for r in ok)
    in_mb = sum(r["glb_mb"] for r in ok)

    summary = {
        "assets_attempted": len(records),
        "assets_ok": len(ok),
        "assets_failed": len(records) - len(ok),
        "startup_s": round(_STARTUP_S, 2),
        "batch_s": round(batch_s, 2),
        "total_s": round(time.perf_counter() - _T_PROC_START, 2),
        "convert_s_sum": round(conv_s, 2),
        "mean_s_per_asset": round(conv_s / len(ok), 2) if ok else None,
        "median_s_per_asset": round(sorted(r["seconds"] for r in ok)[len(ok) // 2], 2) if ok else None,
        "min_s_per_asset": round(min(r["seconds"] for r in ok), 2) if ok else None,
        "max_s_per_asset": round(max(r["seconds"] for r in ok), 2) if ok else None,
        "input_mb_total": round(in_mb, 2),
        "output_mb_total": round(sum(r["usd_mb"] for r in ok), 2),
        "s_per_input_mb": round(conv_s / in_mb, 3) if in_mb else None,
        "assets_per_hour": round(3600 * len(ok) / conv_s, 1) if conv_s else None,
    }
    if summary["assets_per_hour"]:
        n = args_cli.corpus_assets
        summary["corpus_assets"] = n
        summary["corpus_hours_1_worker"] = round(n / summary["assets_per_hour"], 1)
        summary["corpus_days_1_worker"] = round(n / summary["assets_per_hour"] / 24, 1)

    env = {
        "timestamp": stamp,
        "gpu": gpu_name(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "collision_approximation": args_cli.collision_approximation,
        "make_instanceable": args_cli.make_instanceable,
        "headless": bool(getattr(args_cli, "headless", False)),
    }

    payload = {"env": env, "summary": summary, "assets": records}
    json_path = profile_dir / f"profile_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = profile_dir / f"profile_{stamp}.csv"
    cols = ["index", "category", "uid", "glb_mb", "usd_mb", "seconds",
            "meshes", "triangles", "ok", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    md = [
        f"# Sim-ready conversion profile - {stamp}",
        "",
        f"- GPU: {env['gpu']}  |  CPU cores: {env['cpu_count']}  |  headless: {env['headless']}",
        f"- Collision: `{env['collision_approximation']}`  |  instanceable: {env['make_instanceable']}",
        f"- Isaac Sim startup: **{summary['startup_s']}s** (once per batch)",
        "",
        "## Throughput",
        "",
        f"- Converted: **{summary['assets_ok']}/{summary['assets_attempted']}**"
        f" in {summary['convert_s_sum']}s",
    ]
    if ok:
        md += [
            f"- Mean **{summary['mean_s_per_asset']}s/asset**, median"
            f" {summary['median_s_per_asset']}s, range"
            f" {summary['min_s_per_asset']}-{summary['max_s_per_asset']}s",
            f"- {summary['s_per_input_mb']}s per input MB"
            f" ({summary['input_mb_total']} MB in -> {summary['output_mb_total']} MB out)",
            f"- **{summary['assets_per_hour']} assets/hour** on one worker",
            "",
        ]
    else:
        errs = {r["error"] for r in records if r["error"]}
        md += ["- **All conversions failed** - no throughput to report.", ""]
        md += [f"  - `{e}`" for e in sorted(errs)] + [""]
    if summary.get("corpus_hours_1_worker"):
        md += [
            f"## Extrapolation to {summary['corpus_assets']:,} assets",
            "",
            f"- 1 worker: **{summary['corpus_days_1_worker']} days**"
            f" ({summary['corpus_hours_1_worker']} h)",
            f"- 4 workers: {round(summary['corpus_days_1_worker'] / 4, 1)} days",
            f"- 8 workers: {round(summary['corpus_days_1_worker'] / 8, 1)} days",
            "",
            "<sub>Linear scaling assumed; sample may not match the full size"
            " distribution.</sub>",
            "",
            "> **Scope of the timing.** This measures glTF->USD conversion plus"
            " physics-schema authoring only. It excludes (a) downloading the GLBs"
            " and (b) PhysX collision *cooking*, which happens on first stage load,"
            " not at conversion time. For the full corpus the download is expected"
            f" to dominate: this sample averages"
            f" {round(summary['input_mb_total'] / max(summary['assets_ok'], 1), 1)} MB"
            " per asset.",
            "",
        ]
    md += ["## Per-asset", "",
           "| # | category | uid | GLB MB | USD MB | tris | sec | ok |",
           "|---|---|---|---|---|---|---|---|"]
    for r in records:
        md.append(
            f"| {r['index']} | {r['category']} | `{str(r['uid'])[:12]}` |"
            f" {r['glb_mb']} | {r['usd_mb']} | {r['triangles']} |"
            f" {r['seconds']} | {'yes' if r['ok'] else 'NO'} |"
        )
    md_path = profile_dir / f"profile_{stamp}.md"
    md_path.write_text("\n".join(md) + "\n")

    print(f"\n{'=' * 82}")
    print(f"  Done: {summary['assets_ok']}/{summary['assets_attempted']} converted")
    print(f"  Mean {summary['mean_s_per_asset']}s/asset"
          f"  ->  {summary['assets_per_hour']} assets/hour")
    print(f"  Profiles: {json_path}")
    print(f"            {csv_path}")
    print(f"            {md_path}")
    print(f"{'=' * 82}")


if __name__ == "__main__":
    main()
    simulation_app.close()
