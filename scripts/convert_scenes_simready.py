#!/usr/bin/env python3
"""Batch-convert UrbanVerse scenes to Isaac Sim >= 5 sim-ready form, in place.

For every scene directory under ``--root`` that contains ``--pattern``
(``World0.usd`` for the training scenes, ``Collected_export_version/
export_version.usd`` for CraftBench) this:

1. walks the scene's USD composition (sublayers / references / payloads),
2. backs up every layer it is about to change to ``--backup-dir`` (same
   relative path), once - a re-run never overwrites a backup,
3. applies ``upgrade_scene_for_isaacsim5.py``'s texture_scale fix in place,
4. re-walks the composition and refuses to mark the scene done if any scalar
   ``inputs:texture_scale`` survives,
5. appends a record (layers changed, attributes fixed, sha256 of each changed
   layer) to ``--manifest``.

Scenes already recorded as ok in the manifest are skipped, so a killed run
resumes. Example::

    python scripts/convert_scenes_simready.py \
        --root "/media/hollis/Extreme SSD1/UrbanVerse-Training-Scenes" \
        --pattern World0.usd --value mdl-default --keep-float2 \
        --backup-dir "/media/hollis/Extreme SSD1/UrbanVerse-Training-Scenes.orig-layers" \
        --manifest scenes/training_simready_manifest.json --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("upgrade", _HERE / "upgrade_scene_for_isaacsim5.py")
up = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(up)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar_texture_scale_count(layer_path: Path) -> int:
    """Scalar inputs:texture_scale specs left in a layer (0 = clean)."""
    from pxr import Sdf
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None:
        return 0
    n = 0
    for ps in up._all_prim_specs(layer):
        a = ps.attributes.get("inputs:texture_scale")
        if a is not None and a.typeName in up._SCALAR_TYPES and a.default is not None:
            n += 1
    return n


def convert_scene(scene_dir: str, root_usd: str, root: str, backup_dir: str,
                  scale: float, value: float | str | None, keep_float2: bool,
                  mdl_default_fallback: float = 1.0) -> dict:
    t0 = time.perf_counter()
    scene = Path(scene_dir)
    rec = {"scene": scene.name, "root_usd": str(Path(root_usd).relative_to(root)),
           "ok": False, "layers_changed": [], "attrs_fixed": 0, "error": None}
    try:
        layers = up._gather_sublayers(Path(root_usd))
        rec["layers_in_composition"] = len(layers)
        for layer in layers:
            if not up._layer_has_patchable_attr(layer):
                continue
            # Only layers with something to fix get backed up + rewritten.
            if scalar_texture_scale_count(layer) == 0 and keep_float2:
                continue
            rel = layer.relative_to(root)
            bak = Path(backup_dir) / rel
            if not bak.exists():
                bak.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(layer, bak)
            log = io.StringIO()
            with redirect_stdout(log):
                fixed = up._patch_layer(layer, dry_run=False, scale=scale,
                                        override_value=value, keep_float2=keep_float2,
                                        mdl_default_fallback=mdl_default_fallback)
            if fixed:
                rec["attrs_fixed"] += fixed
                rec["layers_changed"].append({"layer": str(rel), "attrs_fixed": fixed,
                                              "sha256": sha256(layer), "bytes": layer.stat().st_size})
        # Verify nothing scalar survived anywhere in the composition.
        left = sum(scalar_texture_scale_count(l) for l in layers)
        rec["scalar_left"] = left
        rec["ok"] = left == 0
        if left:
            rec["error"] = f"{left} scalar texture_scale attribute(s) remain"
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Directory containing one sub-directory per scene.")
    p.add_argument("--pattern", default="World0.usd",
                   help="Root layer path relative to each scene dir (default World0.usd).")
    p.add_argument("--backup-dir", required=True, help="Where originals of changed layers go.")
    p.add_argument("--manifest", required=True, help="JSON manifest (created / appended).")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--value", type=up._value_arg, default=None,
                   help="float, or 'mdl-default' (each shader's MDL parameter default - what "
                        "Isaac Sim 4.5 actually rendered after rejecting the scalar).")
    p.add_argument("--mdl-default-fallback", type=float, default=1.0)
    p.add_argument("--keep-float2", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--redo", action="store_true", help="Re-process scenes already ok in the manifest.")
    a = p.parse_args()

    root = Path(a.root).resolve()
    manifest_path = Path(a.manifest)
    manifest = {"root": str(root), "pattern": a.pattern,
                "policy": {"scale": a.scale, "value": a.value, "keep_float2": a.keep_float2,
                           "mdl_default_fallback": a.mdl_default_fallback},
                "scenes": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("scenes", {})

    scenes = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    todo, missing_root, done = [], [], 0
    for d in scenes:
        root_usd = d / a.pattern
        if not root_usd.is_file():
            missing_root.append(d.name)
            continue
        if not a.redo and manifest["scenes"].get(d.name, {}).get("ok"):
            done += 1
            continue
        todo.append((d, root_usd))
    if a.limit:
        todo = todo[: a.limit]
    print(f"[convert] {len(scenes)} scene dirs: {done} already done, {len(missing_root)} missing "
          f"{a.pattern}, {len(todo)} to convert")
    for m in missing_root:
        print(f"[convert]   no root layer: {m}")

    def save() -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=1))
        tmp.replace(manifest_path)

    n_ok = n_bad = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(convert_scene, str(d), str(u), str(root), a.backup_dir,
                          a.scale, a.value, a.keep_float2, a.mdl_default_fallback): d.name
                for d, u in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            manifest["scenes"][rec["scene"]] = rec
            n_ok += rec["ok"]
            n_bad += not rec["ok"]
            status = "ok " if rec["ok"] else "BAD"
            print(f"[convert] {i:4d}/{len(todo)} {status} {rec['scene']:50} "
                  f"{rec['attrs_fixed']:4d} attrs, {len(rec['layers_changed'])} layers, "
                  f"{rec['seconds']:5.1f}s" + (f"  {rec['error']}" if rec["error"] else ""))
            if i % 5 == 0 or i == len(todo):
                save()
    save()
    total_ok = sum(1 for r in manifest["scenes"].values() if r.get("ok"))
    print(f"[convert] this run: {n_ok} ok, {n_bad} failed. Manifest: {total_ok} ok of "
          f"{len(manifest['scenes'])} recorded -> {manifest_path}")
    if n_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
