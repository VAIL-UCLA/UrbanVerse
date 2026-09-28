#!/usr/bin/env python3
"""Static sanity check of scenes: pure USD, no Isaac Sim, seconds per scene.

    python scripts/sanity_check_static.py                        # every scene
    python scripts/sanity_check_static.py scene_03 -v --json static.json
    python scripts/sanity_check_static.py --root .../Training-Scenes --pattern World0.usd

Scenes are looked up under --root, by default the urbanverse-scene toolkit's CraftBench folder.
Per scene:

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
  physics   at least one collider; rigid bodies (kinematic or not) are counted.
  camera    cam0_to_world.txt, if there is one, holds finite rigid 4x4 transforms.

Exit status 1 if any scene FAILs. Whether it all loads, collides and renders in Isaac Sim is
for sanity_check_sim.py and sanity_check_render.py to tell.
"""
import json
import os
import struct
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

import convert_scenes_simready as convert
import fix_glb_override_names as fix
import sanity_check_common as common

LAYER_EXTS = {".usd", ".usda", ".usdc", ".usdz", ".glb", ".gltf", ".obj", ".fbx"}
ASSET_TYPES = {Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray}


def check_scene(scene_dir: Path, usd: Path) -> dict:
    rec = {"usd": str(usd), "issues": [], "stats": {}}
    with quiet_stderr():  # USD's C++ diagnostics, e.g. every .glb payload failing to open outside Kit
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(usd))
        stage = Usd.Stage.Open(str(usd), Usd.Stage.LoadAll)
    if stage is None:
        common.add(rec, "FAIL", "stage", f"cannot open {usd}")
        return rec
    rec["stats"].update(layers=len(layers), assets=len(assets), glb=sum(a.lower().endswith(".glb") for a in assets))
    check_deps(rec, stage, scene_dir, layers, assets, unresolved)
    check_glb(rec, stage, scene_dir, assets)
    check_simready(rec, stage, scene_dir, layers)
    check_stage(rec, stage)
    check_physics(rec, stage)
    check_camera(rec, scene_dir)
    return rec


def check_deps(rec: dict, stage, scene_dir: Path, layers: list, assets: list, unresolved: list) -> None:
    top = str(scene_dir.resolve()) + os.sep
    for p in [layer.realPath for layer in layers] + list(assets):
        if not os.path.realpath(p.split("[")[0]).startswith(top):
            common.add(rec, "FAIL", "deps", f"resolves outside the scene dir: {p}")
    builtin = sorted({u for u in unresolved if u.endswith(".mdl") and "/" not in u and "\\" not in u})
    if builtin:
        common.add(rec, "INFO", "deps", f"built-in MDL (resolved by Kit): {', '.join(builtin)}")
    missing = [u for u in unresolved if u not in builtin]
    for u in missing:
        if "://" in u:
            common.add(rec, "FAIL", "deps", f"remote dependency: {u}")
        elif is_layer(u):
            common.add(rec, "FAIL", "deps", f"missing layer: {rel(u, scene_dir)}")
        elif u.endswith(".mdl"):
            common.add(rec, "FAIL", "deps", f"missing MDL module: {rel(u, scene_dir)}")

    # Any other missing asset (a texture) matters only as much as the prims that use it.
    users = defaultdict(list)  # authored asset path -> [(prim path, visible)]
    for prim in Usd.PrimRange.Stage(stage, Usd.PrimAllPrimsPredicate):
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in ASSET_TYPES:
                continue
            value = attr.Get()
            for v in (value if isinstance(value, (list, Sdf.AssetPathArray)) else [value]):
                if v is not None and v.path and not v.resolvedPath and not v.path.endswith(".mdl"):
                    users[v.path].append((str(prim.GetPath()), visible(prim)))
    for path, refs in users.items():
        live = [p for p, shown in refs if shown]
        if live:
            common.add(rec, "WARN", "deps", f"missing asset {path} used by {len(live)} visible prim(s), "
                                            f"e.g. {live[0]}")
        else:
            common.add(rec, "INFO", "deps", f"missing asset {os.path.basename(path)} only on invisible/inactive "
                                            f"prims: {', '.join(p for p, _ in refs[:3])}")
    used = {os.path.basename(path) for path in users}
    for u in missing:
        if "://" not in u and not is_layer(u) and not u.endswith(".mdl") \
                and os.path.basename(u.split("[")[-1].rstrip("]")) not in used:
            common.add(rec, "INFO", "deps", f"missing asset {rel(u, scene_dir)} is not reached by any composed prim")


def check_glb(rec: dict, stage, scene_dir: Path, assets: list) -> None:
    for a in assets:
        if a.lower().endswith(".glb") and not glb_ok(a):
            common.add(rec, "FAIL", "glb", f"bad glTF header or length: {rel(a, scene_dir)}")
    stale = fix.stale_overrides(stage)
    if stale:
        prim, new = stale[0]
        common.add(rec, "FAIL", "glb", f"{len(stale)} override(s) target a .glb node by its Isaac Sim 4.5 name, "
                                       f"which Isaac Sim 5 does not match (fix_glb_override_names.py renames them), "
                                       f"e.g. {prim.GetPath()} -> {new}")


def check_simready(rec: dict, stage, scene_dir: Path, layers: list) -> None:
    # Judged on the composed stage, which is what Isaac Sim renders, and by the value: some are
    # declared float2 and still hold a scalar ('float2 inputs:texture_scale = 1000').
    scalar = [attr for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
              if (attr := prim.GetAttribute("inputs:texture_scale")) and attr.HasAuthoredValue()
              and not isinstance(attr.Get(), (Gf.Vec2f, Gf.Vec2d, Gf.Vec2h))]
    if scalar:
        common.add(rec, "FAIL", "simready", f"{len(scalar)} composed inputs:texture_scale hold a scalar, which "
                                            f"renders black/white in Isaac Sim 5, e.g. {scalar[0].GetPath()} = "
                                            f"{scalar[0].GetTypeName()} {scalar[0].Get()!r}")
    # Layers the scene never composes (e.g. unused ones inside a .usdz) can still hold scalars.
    composed = {layer.identifier for layer in stage.GetUsedLayers()}
    unused = {layer.identifier: n for layer in layers if layer.identifier not in composed
              and (n := convert.scalar_texture_scale_count(layer.identifier))}
    if unused:
        common.add(rec, "INFO", "simready", f"{sum(unused.values())} scalar texture_scale only in layers the "
                                            f"scene never composes: {', '.join(rel(u, scene_dir) for u in unused)}")


def check_stage(rec: dict, stage) -> None:
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        common.add(rec, "FAIL", "stage", f"upAxis is {UsdGeom.GetStageUpAxis(stage)}, expected Z")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-6:
        common.add(rec, "FAIL", "stage", f"metersPerUnit is {UsdGeom.GetStageMetersPerUnit(stage)}, expected 1")
    default = stage.GetDefaultPrim()
    if not default:
        common.add(rec, "WARN", "stage", "no defaultPrim")
    elif default.IsA(UsdGeom.Xformable):
        xf = UsdGeom.Xformable(default).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if xf != Gf.Matrix4d(1.0):  # anything spawned under it by local coordinates lands elsewhere
            t = xf.ExtractTranslation()
            common.add(rec, "INFO", "stage", f"defaultPrim {default.GetPath()} is transformed (translate "
                                             f"{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}): place robots/probes in world "
                                             f"coords")


def check_physics(rec: dict, stage) -> None:
    colliders = rigid = kinematic = scenes = 0
    for prim in stage.Traverse():
        colliders += prim.HasAPI(UsdPhysics.CollisionAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid += 1
            kinematic += bool(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())
        scenes += prim.IsA(UsdPhysics.Scene)
    rec["stats"].update(colliders=colliders, rigid_bodies=rigid, kinematic=kinematic, physics_scenes=scenes)
    if not colliders:
        common.add(rec, "FAIL", "physics", "no prim has CollisionAPI - robots fall through")
    if not scenes:
        common.add(rec, "INFO", "physics", "no PhysicsScene authored (Isaac Sim adds a default one)")


def check_camera(rec: dict, scene_dir: Path) -> None:
    """The preview flythrough CraftBench ships: a frame number and a row-major 4x4 matrix per line."""
    path = scene_dir / "cam0_to_world.txt"
    if not path.is_file():
        return
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    bad = sum(not rigid_transform(row[1:]) for row in rows)
    rec["stats"]["camera_poses"] = len(rows)
    if not rows or bad:
        common.add(rec, "FAIL", "camera", f"{bad} of {len(rows)} poses in cam0_to_world.txt are not rigid "
                                          f"transforms")


def rigid_transform(values: list) -> bool:
    try:
        m = np.array(values, dtype=float).reshape(4, 4)
    except ValueError:
        return False
    rot = m[:3, :3]
    return bool(np.isfinite(m).all() and np.allclose(rot @ rot.T, np.eye(3), atol=1e-3)
                and abs(np.linalg.det(rot) - 1) < 1e-3 and np.allclose(m[3], [0, 0, 0, 1]))


def visible(prim) -> bool:
    return prim.IsActive() and not (prim.IsA(UsdGeom.Imageable) and
                                    UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible)


def is_layer(path: str) -> bool:
    """Whether an asset path names a layer, also inside a package ('scene.usdz[SubUSDs/a.usd]')."""
    return os.path.splitext(path.split("[")[-1].rstrip("]"))[1].lower() in LAYER_EXTS


def glb_ok(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(12)
    if len(head) < 12:
        return False
    magic, version, length = struct.unpack("<4sII", head)
    return magic == b"glTF" and version == 2 and length == os.path.getsize(path)


def rel(path: str, scene_dir: Path) -> str:
    return os.path.relpath(path, scene_dir) if path.startswith(str(scene_dir)) else path


@contextmanager
def quiet_stderr():
    sys.stderr.flush()
    saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def main() -> None:
    p = common.parser(__doc__)
    p.add_argument("--json", help="Write the full report here.")
    p.add_argument("-v", "--verbose", action="store_true", help="Also print INFO lines.")
    a = p.parse_args()

    report = {}
    for name, scene_dir, usd in common.find_scenes(a):
        rec = report[name] = check_scene(scene_dir, usd)
        rec["status"] = common.status(rec)
        s = rec["stats"]
        print(f"[static] {rec['status']:5} {name}  {s.get('layers', 0)} layers, {s.get('assets', 0)} assets "
              f"({s.get('glb', 0)} glb), {s.get('colliders', 0)} colliders, {s.get('rigid_bodies', 0)} rigid "
              f"({s.get('kinematic', 0)} kinematic)"
              + (f", {s['camera_poses']} cam poses" if "camera_poses" in s else ""))
        common.print_issues(rec, 11, info=a.verbose)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    common.summary("static", [rec["status"] for rec in report.values()], f"report in {a.json}" if a.json else "")


if __name__ == "__main__":
    main()
