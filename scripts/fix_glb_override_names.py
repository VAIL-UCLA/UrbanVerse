#!/usr/bin/env python3
"""Rename scene overrides into .glb models from Isaac Sim 4.5 to Isaac Sim 5 prim names, in place.

Kit's glTF importer names a node's prim after the node, with each invalid character
replaced by '_': per UTF-8 byte in Isaac Sim 4.5, per character in Isaac Sim 5. The two
only differ for non-ASCII names, e.g. the CraftBench bench node 'P\\ufffd\\ufffds_Material_0'
(two U+FFFD, 3 bytes each):

    Isaac Sim 4.5   P______s_Material_0
    Isaac Sim 5     P__s_Material_0

A scene authored in 4.5 overrides that node as ``over "P______s_Material_0"``, which in 5
matches nothing, so the node loses the placement, collision or active = false the scene
gave it (CraftBench scenes 03, 09, 11: benches misplaced without collision, a switched-off
building part back on). This renames each such override to the Isaac Sim 5 name. Like
convert_scenes_simready.py it backs up every layer it changes to --backup-dir (once) and
records the layer's new sha256 in --manifest::

    python scripts/fix_glb_override_names.py \\
        --root /path/to/UrbanVerse-Scenes/CraftBench \\
        --backup-dir /path/to/UrbanVerse-Scenes/CraftBench.orig-layers \\
        --manifest scenes/craftbench_simready_manifest.json

--dry-run only lists the renames. A re-run finds nothing left to rename;
sanity_check_sim.py's 'overrides' check confirms the result in Isaac Sim.
"""
import argparse
import json
import os
import shutil
import struct
import time
from collections import defaultdict
from pathlib import Path

from pxr import Sdf, Usd

import convert_scenes_simready as convert

PATTERN = "Collected_export_version/export_version.usd"


def glb_names(path: str) -> set:
    """Node and mesh names in a .glb's JSON chunk (empty if it cannot be read)."""
    try:
        with open(path, "rb") as f:
            f.seek(12)
            length, kind = struct.unpack("<I4s", f.read(8))
            js = json.loads(f.read(length)) if kind == b"JSON" else {}
    except (OSError, ValueError, struct.error):
        return set()
    return {n.get("name", "") for key in ("nodes", "meshes") for n in js.get(key, [])}


def prim_name(name: str, per_byte: bool) -> str:
    """The prim name Kit's glTF importer gives a node: invalid characters become '_', one per
    UTF-8 byte in Isaac Sim 4.5 (per_byte) and one per character in Isaac Sim 5."""
    if per_byte:
        return "".join(c if c.isascii() and (c.isalnum() or c == "_") else "_" * len(c.encode()) for c in name)
    return "".join(c if ("_" + c).isidentifier() else "_" for c in name)


def stale_overrides(stage: Usd.Stage) -> list:
    """(override prim, Isaac Sim 5 name) for every 'over' that targets a .glb node by its 4.5 name.

    Works on a stage opened without Isaac Sim: the .glb payloads do not load, so the
    overrides are the only children the payload prims have."""
    found = []
    for prim in stage.TraverseAll():
        payload = prim.GetMetadata("payload")
        if not payload or not prim.IsActive():
            continue
        layer = next(s.layer for s in prim.GetPrimStack() if s.HasInfo("payload"))
        for item in payload.GetAddedOrExplicitItems():
            path = layer.ComputeAbsolutePath(item.assetPath)
            if not path.lower().endswith(".glb") or not os.path.isfile(path):
                continue
            for name in sorted(glb_names(path)):
                old, new = prim_name(name, per_byte=True), prim_name(name, per_byte=False)
                if old == new or not Sdf.Path.IsValidIdentifier(old):
                    continue
                child = stage.GetPrimAtPath(prim.GetPath().AppendChild(old))
                # only pure overrides: a 'def' under that name is the scene's own prim, not a miss
                if child and all(s.specifier == Sdf.SpecifierOver for s in child.GetPrimStack()):
                    found.append((child, new))
    return found


def fix_scene(scene: Path, root: Path, pattern: str, backup_dir: Path, dry_run: bool) -> list:
    """Rename the scene's stale overrides; returns [{layer, from, to}] (what would change on a dry run)."""
    stage = Usd.Stage.Open(str(scene / pattern), Usd.Stage.LoadNone)
    edits = defaultdict(list)  # layer -> [(spec path, new name)]
    for prim, new in stale_overrides(stage):
        for spec in prim.GetPrimStack():
            sibling = spec.path.GetParentPath().AppendChild(new)
            if spec.layer.GetPrimAtPath(sibling):
                raise SystemExit(f"{scene.name}: cannot rename {spec.path} in {spec.layer.realPath}: "
                                 f"{new} already exists")
            edits[spec.layer].append((spec.path, new))
    done = []
    for layer, renames in edits.items():
        rel = Path(layer.realPath).relative_to(root)
        done += [{"layer": str(rel), "from": str(path), "to": new} for path, new in renames]
        if dry_run:
            continue
        bak = backup_dir / rel
        if not bak.exists():  # the original, never overwritten by a re-run
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(layer.realPath, bak)
        for path, new in renames:
            layer.GetPrimAtPath(path).name = new
        if not layer.Save():
            raise SystemExit(f"{scene.name}: could not save {layer.realPath}")
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Directory with one sub-directory per scene.")
    p.add_argument("--pattern", default=PATTERN, help=f"Root layer relative to each scene dir (default {PATTERN}).")
    p.add_argument("--scene", action="append", default=[], help="Scene name or id prefix (scene_03); repeatable.")
    p.add_argument("--backup-dir", required=True, help="Where originals of changed layers go.")
    p.add_argument("--manifest", required=True, help="Sim-ready manifest to record changed layers in.")
    p.add_argument("--dry-run", action="store_true", help="List the renames, change nothing.")
    a = p.parse_args()

    root = Path(a.root).expanduser().resolve()
    scenes = sorted(d for d in root.iterdir() if d.is_dir() and (d / a.pattern).is_file())
    if a.scene:
        scenes = [d for d in scenes if any(d.name == w or d.name.startswith(w + "_") for w in a.scene)]
    manifest_path = Path(a.manifest)
    manifest = json.loads(manifest_path.read_text())
    if Path(manifest["root"]).resolve() != root:
        raise SystemExit(f"--manifest records scenes under {manifest['root']}, not {root}")

    n = 0
    for d in scenes:
        t0 = time.perf_counter()
        done = fix_scene(d, root, a.pattern, Path(a.backup_dir), a.dry_run)
        n += len(done)
        for r in done:
            print(f"[fix] {d.name}: {r['from']} -> {r['to']}  ({r['layer'][len(d.name) + 1:]})")
        if done and not a.dry_run:
            rec = manifest["scenes"].setdefault(d.name, {"scene": d.name, "layers_changed": []})
            rec.setdefault("overrides_renamed", []).extend(done)
            for rel in sorted({r["layer"] for r in done}):
                entry = next((e for e in rec["layers_changed"] if e["layer"] == rel), None)
                if entry is None:
                    entry = {"layer": rel}
                    rec["layers_changed"].append(entry)
                entry["prims_renamed"] = entry.get("prims_renamed", 0) + sum(r["layer"] == rel for r in done)
                entry.update(sha256=convert.sha256(root / rel), bytes=(root / rel).stat().st_size)
            tmp = manifest_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(manifest, indent=1))
            tmp.replace(manifest_path)
            print(f"[fix] {d.name}: {len(done)} renamed in {time.perf_counter() - t0:.0f}s, manifest updated")
    print(f"[fix] {n} override(s) {'to rename' if a.dry_run else 'renamed'} in {len(scenes)} scene(s)")


if __name__ == "__main__":
    main()
