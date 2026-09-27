#!/usr/bin/env python3
"""Static sanity check of downloaded scenes: pure USD, no Isaac Sim, seconds per scene.

    python scripts/sanity_check_static.py --root /path/to/urbanverse_scenes/CraftBench
    python scripts/sanity_check_static.py --root ... --scene scene_03 --json /tmp/static.json -v
    python scripts/sanity_check_static.py --root .../Training-Scenes --pattern World0.usd

Per scene (``<scene>/<pattern>``, CraftBench's root layer by default):

  deps      every layer and asset the composition reaches resolves to a file inside the
            scene dir. Bare ``*.mdl`` names (OmniPBR.mdl, ...) are Kit's built-in MDL
            library and resolve at runtime. A missing texture only WARNs, and is INFO when
            every prim using it is invisible or inactive (e.g. the unused Grey_Studio sky).
  glb       every .glb has a glTF 2.0 header whose length matches the file size, and no
            scene override targets a .glb node by its Isaac Sim 4.5 name where Isaac Sim 5
            names it differently (non-ASCII glTF names): in 5 that override misses and the
            node loses its placement, collision or active = false. fix_glb_override_names.py
            renames them; sanity_check_sim.py's 'overrides' check confirms in Isaac Sim.
  simready  no composed ``inputs:texture_scale`` holds a scalar (the Isaac Sim 5 fix), whatever
            type it is declared as. Scalars left in layers the scene never composes are INFO.
  stage     Z-up, metersPerUnit 1, defaultPrim set.
  physics   at least one collider; rigid bodies (kinematic or not) are reported.
  camera    cam0_to_world.txt, when present, holds finite rigid 4x4 transforms.

Exit status 1 if any scene FAILs (``--strict``: or WARNs). Whether it all loads, compiles,
collides and renders in Isaac Sim is sanity_check_sim.py's job.
"""
import argparse
import importlib.util
import json
import os
import struct
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("convert", _HERE / "convert_scenes_simready.py")
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)
_spec = importlib.util.spec_from_file_location("fix", _HERE / "fix_glb_override_names.py")
fix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fix)

PATTERN = "Collected_export_version/export_version.usd"
LAYER_EXTS = {".usd", ".usda", ".usdc", ".usdz", ".glb", ".gltf", ".obj", ".fbx"}
ASSET_TYPES = {Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray}


@contextmanager
def quiet_stderr():
    """Mute USD's C++ diagnostics, e.g. every .glb payload failing to open outside Kit."""
    sys.stderr.flush()
    saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def is_builtin_mdl(path: str) -> bool:
    return path.endswith(".mdl") and "/" not in path and "\\" not in path


def glb_ok(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(12)
    if len(head) < 12:
        return False
    magic, version, length = struct.unpack("<4sII", head)
    return magic == b"glTF" and version == 2 and length == os.path.getsize(path)


def check_scene(scene_dir: Path, pattern: str) -> dict:
    issues, stats = [], {}

    def add(level: str, check: str, msg: str) -> None:
        issues.append({"level": level, "check": check, "msg": msg})

    def rel(p: str) -> str:
        return os.path.relpath(p, scene_dir) if p.startswith(str(scene_dir)) else p

    root = scene_dir / pattern
    with quiet_stderr():
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(root))
        stage = Usd.Stage.Open(str(root), Usd.Stage.LoadAll)
    if stage is None:
        add("FAIL", "stage", f"cannot open {pattern}")
        return {"issues": issues, "stats": stats}
    stats.update(layers=len(layers), assets=len(assets),
                 glb=sum(a.lower().endswith(".glb") for a in assets))

    # deps: self-contained, nothing remote, nothing missing that matters
    top = str(scene_dir.resolve()) + os.sep
    for p in [layer.realPath for layer in layers] + list(assets):
        if not os.path.realpath(p.split("[")[0]).startswith(top):
            add("FAIL", "deps", f"resolves outside the scene dir: {p}")
    builtin = sorted({u for u in unresolved if is_builtin_mdl(u)})
    if builtin:
        add("INFO", "deps", f"built-in MDL (resolved by Kit): {', '.join(builtin)}")
    missing = [u for u in unresolved if u not in builtin]
    for u in missing:
        if "://" in u:
            add("FAIL", "deps", f"remote dependency: {u}")
        elif os.path.splitext(u.split("[")[-1].rstrip("]"))[1].lower() in LAYER_EXTS:
            add("FAIL", "deps", f"missing layer: {rel(u)}")
        elif u.endswith(".mdl"):
            add("FAIL", "deps", f"missing MDL module: {rel(u)}")
    # Missing textures: severity depends on whether anything visible uses them.
    users = defaultdict(list)  # authored asset path -> [(prim path, used)]
    for prim in Usd.PrimRange.Stage(stage, Usd.PrimAllPrimsPredicate):
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in ASSET_TYPES:
                continue
            value = attr.Get()
            for v in (value if isinstance(value, (list, Sdf.AssetPathArray)) else [value]):
                if v is None or not v.path or v.resolvedPath or is_builtin_mdl(v.path) or v.path.endswith(".mdl"):
                    continue
                used = prim.IsActive() and not (prim.IsA(UsdGeom.Imageable) and
                                                UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible)
                users[v.path].append((str(prim.GetPath()), used))
    seen = set()
    for path, refs in users.items():
        seen.add(os.path.basename(path))
        live = [p for p, used in refs if used]
        if live:
            add("WARN", "deps", f"missing asset {path} used by {len(live)} visible prim(s), e.g. {live[0]}")
        else:
            add("INFO", "deps", f"missing asset {os.path.basename(path)} only on invisible/inactive prims: "
                                f"{', '.join(p for p, _ in refs[:3])}")
    for u in missing:
        name = os.path.basename(u.split("[")[-1].rstrip("]"))
        if name not in seen and not u.endswith(".mdl") and "://" not in u \
                and os.path.splitext(name)[1].lower() not in LAYER_EXTS:
            add("INFO", "deps", f"missing asset {rel(u)} is not reached by any composed prim")

    # glb: truncated or corrupt payloads
    for a in assets:
        if a.lower().endswith(".glb") and not glb_ok(a):
            add("FAIL", "glb", f"bad glTF header or length: {rel(a)}")
    stale = fix.stale_overrides(stage)
    if stale:
        prim, new = stale[0]
        add("FAIL", "glb", f"{len(stale)} override(s) target a .glb node by its Isaac Sim 4.5 name, which "
                           f"Isaac Sim 5 does not match (fix_glb_override_names.py renames them), "
                           f"e.g. {prim.GetPath()} -> {new}")

    # simready: judged on the composed stage, since that is what Isaac Sim renders. Layers
    # the scene never composes (e.g. unused layers inside a .usdz) can still hold scalars.
    # Judged by the value: some are declared float2 and still hold a scalar ('float2 ... = 1000').
    scalar = [(str(prim.GetPath()), attr) for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
              if (attr := prim.GetAttribute("inputs:texture_scale")) and attr.HasAuthoredValue()
              and not isinstance(attr.Get(), (Gf.Vec2f, Gf.Vec2d, Gf.Vec2h))]
    if scalar:
        path, attr = scalar[0]
        mistyped = sum(attr.GetTypeName() not in conv.up._SCALAR_TYPES for _, attr in scalar)
        add("FAIL", "simready", f"{len(scalar)} composed inputs:texture_scale hold a scalar (renders black/white in "
                                f"Isaac Sim 5), {mistyped} of them declared float2, e.g. {path} = "
                                f"{attr.GetTypeName()} {attr.Get()!r}")
    used = {layer.identifier for layer in stage.GetUsedLayers()}
    unused = {layer.identifier: n for layer in layers if layer.identifier not in used
              and (n := conv.scalar_texture_scale_count(layer.identifier))}
    if unused:
        add("INFO", "simready", f"{sum(unused.values())} scalar texture_scale only in layers the scene never "
                                f"composes: {', '.join(rel(u) for u in unused)}")

    # stage metadata
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        add("FAIL", "stage", f"upAxis is {UsdGeom.GetStageUpAxis(stage)}, expected Z")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-6:
        add("FAIL", "stage", f"metersPerUnit is {UsdGeom.GetStageMetersPerUnit(stage)}, expected 1")
    default = stage.GetDefaultPrim()
    if not default:
        add("WARN", "stage", "no defaultPrim")
    elif default.IsA(UsdGeom.Xformable):
        xf = UsdGeom.Xformable(default).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if xf != Gf.Matrix4d(1.0):  # anything spawned under it by local coords lands elsewhere
            t = xf.ExtractTranslation()
            add("INFO", "stage", f"defaultPrim {default.GetPath()} is transformed (translate "
                                 f"{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}): place robots/probes in world coords")

    # physics authoring
    colliders = rigid = kinematic = phys_scenes = 0
    for prim in stage.Traverse():
        colliders += prim.HasAPI(UsdPhysics.CollisionAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid += 1
            kinematic += bool(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())
        phys_scenes += prim.IsA(UsdPhysics.Scene)
    stats.update(colliders=colliders, rigid_bodies=rigid, kinematic=kinematic, physics_scenes=phys_scenes)
    if not colliders:
        add("FAIL", "physics", "no prim has CollisionAPI - robots fall through")
    if not phys_scenes:
        add("INFO", "physics", "no PhysicsScene authored (Isaac Sim adds a default one)")

    # camera trajectory (CraftBench ships one per scene)
    cam = scene_dir / "cam0_to_world.txt"
    if cam.is_file():
        rows = [line.split() for line in cam.read_text().splitlines() if line.strip()]
        bad = 0
        for r in rows:
            try:
                m = np.array(r[1:], dtype=float).reshape(4, 4)
            except ValueError:
                bad += 1
                continue
            rot = m[:3, :3]
            if not (np.isfinite(m).all() and np.allclose(rot @ rot.T, np.eye(3), atol=1e-3)
                    and abs(np.linalg.det(rot) - 1) < 1e-3 and np.allclose(m[3], [0, 0, 0, 1])):
                bad += 1
        stats["camera_poses"] = len(rows)
        if not rows or bad:
            add("FAIL", "camera", f"{bad} of {len(rows)} poses in cam0_to_world.txt are not rigid transforms")
    return {"issues": issues, "stats": stats}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Directory with one sub-directory per scene.")
    p.add_argument("--pattern", default=PATTERN, help=f"Root layer relative to each scene dir (default {PATTERN}).")
    p.add_argument("--scene", action="append", default=[], help="Scene name or id prefix (scene_03); repeatable.")
    p.add_argument("--json", help="Write the full report here.")
    p.add_argument("--strict", action="store_true", help="Exit 1 on WARN too.")
    p.add_argument("-v", "--verbose", action="store_true", help="Also print INFO lines.")
    a = p.parse_args()

    root = Path(a.root).expanduser().resolve()
    scenes = sorted(d for d in root.iterdir() if d.is_dir() and (d / a.pattern).is_file())
    if not scenes:
        raise SystemExit(f"no scene under {root} has {a.pattern}")
    if a.scene:
        picked = []
        for want in a.scene:
            hits = [d for d in scenes if d.name == want or d.name.startswith(want + "_")]
            if len(hits) != 1:
                raise SystemExit(f"--scene {want!r} matches {len(hits)} scenes under {root}")
            picked += hits
        scenes = sorted(set(picked))

    report, n_fail, n_warn = {}, 0, 0
    for d in scenes:
        rec = check_scene(d, a.pattern)
        levels = {i["level"] for i in rec["issues"]}
        rec["status"] = "FAIL" if "FAIL" in levels else "WARN" if "WARN" in levels else "ok"
        n_fail += rec["status"] == "FAIL"
        n_warn += rec["status"] == "WARN"
        report[d.name] = rec
        s = rec["stats"]
        print(f"[static] {rec['status']:4} {d.name}  {s.get('layers', 0)} layers, {s.get('assets', 0)} assets "
              f"({s.get('glb', 0)} glb), {s.get('colliders', 0)} colliders, {s.get('rigid_bodies', 0)} rigid "
              f"({s.get('kinematic', 0)} kinematic)" + (f", {s['camera_poses']} cam poses" if "camera_poses" in s else ""))
        for i in rec["issues"]:
            if i["level"] != "INFO" or a.verbose:
                print(f"           {i['level']:4} {i['check']}: {i['msg']}")

    print(f"[static] {len(scenes) - n_fail - n_warn} ok, {n_warn} WARN, {n_fail} FAIL of {len(scenes)} scene(s)")
    if a.json:
        Path(a.json).write_text(json.dumps({"root": str(root), "pattern": a.pattern, "scenes": report}, indent=1))
        print(f"[static] report -> {a.json}")
    if n_fail or (a.strict and n_warn):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
