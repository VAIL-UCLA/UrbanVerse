#!/usr/bin/env python3
"""Runtime sanity check of scenes in Isaac Sim 5.x: one headless Kit process per scene, nothing rendered.

    python scripts/sanity_check_sim.py                    # every scene, one by one
    python scripts/sanity_check_sim.py scene_03           # one scene, by id or name
    python scripts/sanity_check_sim.py /path/to/scene     # one scene, by its dir or root .usd

Scenes are looked up under --root, by default the urbanverse-scene toolkit's CraftBench folder.
Per scene:

  load       the stage opens and loads; every .glb model yields geometry, unless the scene
             switched it off (active = false, INFO).
  overrides  every scene override into a .glb model lands on a prim there. One that does not
             (e.g. Isaac Sim 5 names non-ASCII glTF nodes differently from 4.5) loses the
             placement, collision or active = false it carries.
  colliders  every other collider is real geometry, else that object is walk-through.
  log        Kit logged no missing-file, file-format or USD->MDL type errors; PhysX errors WARN.
  ground     rays cast straight down from 12 poses along the scene's camera path
             (cam0_to_world.txt) hit static ground; objects on the way are looked through. The
             hit prim types tell the road mesh from an infinite fallback Plane. A pose with no
             ground below only WARNs: the camera path need not be walkable.
  drop       a 20 cm cube dropped from 0.5 m, wherever nothing stands on that ground, must not
             fall straight through it (FAIL; rolling off an edge WARNs), and the scene's own
             dynamic rigid bodies must not fall more than 1 m.

Writes <out>/<scene>.{log,json}. A scene is FAIL when it is broken, ERROR when the check could
not finish (Kit crashed or hung: rerun it); exit 1 on either. How a scene looks is
sanity_check_render.py's job.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import sanity_check_common as common

PROBES = 12  # camera poses to cast rays down from and drop cubes at
DT = 1 / 120  # physics step, s
DROP_S = 2.0  # simulated seconds for the cubes to fall


# ── inside Kit: one scene ────────────────────────────────────────────────────────────────────

def child(job: dict) -> None:
    from isaaclab.app import AppLauncher

    # the .glb file-format plugin has to be there at startup
    app = AppLauncher(headless=True, kit_args="--enable omni.kit.asset_converter").app
    import omni.usd

    rec = {"issues": [], "stats": {}}
    try:
        ctx = omni.usd.get_context()
        if not ctx.open_stage(job["usd"]):
            common.add(rec, "FAIL", "load", f"could not open {job['usd']}")
            return
        common.wait_for_load(app.update, ctx, rec)
        stage = ctx.get_stage()
        orphans = check_models(stage, rec)
        check_colliders(stage, orphans, rec)
        check_ground(stage, camera_path(Path(job["scene_dir"])), rec)
    except Exception as e:  # noqa: BLE001
        common.add(rec, "ERROR", "run", f"{type(e).__name__}: {e}")
    finally:
        common.finish(rec, job["result"])


def check_models(stage, rec: dict) -> set:
    """The load and overrides checks. Returns the paths of the overrides that land on nothing."""
    from pxr import Usd, UsdGeom, UsdPhysics

    models = [p for p in stage.Traverse() if is_glb_model(p)]
    empty = [p for p in models if not any(q.IsA(UsdGeom.Mesh) for q in Usd.PrimRange(p))]
    # scene_03 ("..._clean") switches most of its models off with active = false on their nodes
    off = [p for p in empty if any(not q.IsActive() for q in Usd.PrimRange(p, Usd.PrimAllPrimsPredicate))]
    broken = [p for p in empty if p not in off]
    rec["stats"].update(glb_payloads=len(models), glb_switched_off=len(off))
    if off:
        common.add(rec, "INFO", "load", f"{len(off)} of {len(models)} .glb models are switched off by the scene "
                                        f"(active = false), e.g. {off[0].GetPath()}")
    if broken:
        common.add(rec, "FAIL", "load", f"{len(broken)} of {len(models)} .glb payloads loaded no mesh, "
                                        f"e.g. {broken[0].GetPath()}")

    # The scene layers author 'over <node>' into a model. One whose node is not there stays a
    # topmost override with nothing defined in it.
    orphans = [q for p in models for q in Usd.PrimRange(p, Usd.PrimAllPrimsPredicate)
               if not q.IsDefined() and q.GetParent().IsDefined()]
    rec["stats"]["orphan_overrides"] = len(orphans)
    if orphans:
        lost = {"placement": sum(bool(UsdGeom.Xformable(q).GetOrderedXformOps()) for q in orphans),
                "collision": sum(q.HasAPI(UsdPhysics.CollisionAPI) for q in orphans),
                "active = false": sum(not q.IsActive() for q in orphans)}
        common.add(rec, "FAIL", "overrides",
                   f"{len(orphans)} scene override(s) into .glb models match no prim (the node has another name in "
                   f"this Isaac Sim), losing {', '.join(f'{what} x{n}' for what, n in lost.items() if n)}: the model "
                   f"is misplaced, walk-through or back on, e.g. {orphans[0].GetPath()}")
    return {q.GetPath() for q in orphans}


def is_glb_model(prim) -> bool:
    payload = prim.GetMetadata("payload")
    return bool(payload) and any(i.assetPath.lower().endswith(".glb") for i in payload.GetAddedOrExplicitItems())


def check_colliders(stage, orphans: set, rec: dict) -> None:
    from pxr import UsdGeom, UsdPhysics

    colliders = [p for p in stage.TraverseAll() if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsActive()]
    hollow = [p for p in colliders if p.GetPath() not in orphans and (
        not p.IsA(UsdGeom.Gprim) or (p.IsA(UsdGeom.Mesh) and not UsdGeom.Mesh(p).GetPointsAttr().Get()))]
    rec["stats"]["colliders"] = len(colliders)
    if hollow:
        common.add(rec, "FAIL", "colliders", f"{len(hollow)} of {len(colliders)} colliders have no geometry "
                                             f"(walk-through), e.g. {hollow[0].GetPath()}")


def camera_path(scene_dir: Path) -> list:
    """The camera position in each frame of the scene's preview flythrough: cam0_to_world.txt holds a
    frame number and a row-major 4x4 camera-to-world matrix per line."""
    path = scene_dir / "cam0_to_world.txt"
    rows = [line.split()[1:] for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    return [(float(m[3]), float(m[7]), float(m[11])) for m in rows]


def check_ground(stage, camera: list, rec: dict) -> None:
    """The ground and drop checks, at PROBES poses spread along the camera path."""
    import isaaclab.sim as sim_utils
    import numpy as np
    from omni.physx import get_physx_interface, get_physx_scene_query_interface

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cpu", enable_scene_query_support=True))
    sim.reset()
    poses = [(k, camera[k]) for k in np.linspace(0, len(camera) - 1, PROBES).round().astype(int)] if camera else []
    ground = find_ground(stage, poses, get_physx_scene_query_interface(), rec)
    drop_cubes(stage, ground, sim, get_physx_interface(), rec)


def find_ground(stage, poses: list, scene_query, rec: dict) -> list:
    """Cast a ray straight down from each (index, position) pose: the first hit that is not an
    object is the ground. Returns (x, y, ground z, ground prim, clear) for each pose with ground
    below, where clear means nothing stands on the ground there."""
    from pxr import UsdPhysics

    def is_object(path: str) -> bool:  # bins, cars, ... are (kinematic) rigid bodies; the ground is static
        prim = stage.GetPrimAtPath(path) if path.startswith("/") else None
        while prim and prim.IsValid() and not prim.IsPseudoRoot():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return True
            prim = prim.GetParent()
        return False

    found = []

    def report(hit) -> bool:
        found.append((hit.distance, float(hit.position[2]), hit.collision))
        return True  # go on: every hit along the ray

    hits, misses = [], []
    for k, (x, y, z) in poses:
        found.clear()
        scene_query.raycast_all([x, y, z], [0.0, 0.0, -1.0], abs(z) + 100.0, report)  # some paths end 190 m up
        found.sort()
        ground = next(((gz, path) for _, gz, path in found if not is_object(path)), None)
        if ground:
            hits.append((x, y, *ground, found[0][2] == ground[1]))
        else:
            misses.append(f"pose {k} at ({x:.1f}, {y:.1f})")
    types = Counter(stage.GetPrimAtPath(path).GetTypeName() if path.startswith("/") else "?"
                    for _, _, _, path, _ in hits)
    rec["stats"]["ground"] = {"probes": len(poses), "hits": len(hits), "hit_types": dict(types),
                              "z": [round(min(h[2] for h in hits), 2), round(max(h[2] for h in hits), 2)]
                              if hits else None}
    if misses:
        common.add(rec, "WARN", "ground", f"{len(misses)} of {len(poses)} camera poses have no static ground below: "
                                          f"{', '.join(misses[:3])}")
    return hits


def drop_cubes(stage, ground: list, sim, physx, rec: dict) -> None:
    """Drop a cube wherever nothing stands on the ground, and let the scene's dynamic rigid bodies
    fall with them. A cube that ends up below the ground went straight through it (not solid) or
    rolled off an edge (sideways)."""
    import isaaclab.sim as sim_utils
    import numpy as np
    from pxr import UsdPhysics

    dynamic = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
               and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False
               and not UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get()]
    cube = sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                               mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                               collision_props=sim_utils.CollisionPropertiesCfg())
    cubes = []
    for i, (x, y, gz, _, clear) in enumerate(ground):
        if clear:
            path = f"/SanityProbe_{i:02d}"  # root level: CraftBench's /World is translated (-735.84, 490.36, 0)
            cube.func(path, cube, translation=(x, y, gz + 0.5))
            cubes.append((path, x, y, gz))
    if not cubes and not dynamic:
        return
    sim.reset()
    start = {}
    for path in dynamic:  # bodies PhysX did not create (e.g. hollow ones) have no transform
        t = physx.get_rigidbody_transformation(path)
        if t["ret_val"]:
            start[path] = t["position"][2]
    for _ in range(round(DROP_S / DT)):
        sim.step(render=False)

    through, off_edge = [], []
    for path, x, y, gz in cubes:
        cx, cy, cz = physx.get_rigidbody_transformation(path)["position"]
        if cz < gz - 0.3:
            moved = float(np.hypot(cx - x, cy - y))
            (through if moved < 0.3 else off_edge).append(
                f"{path} at ({x:.1f}, {y:.1f}) ended at z={cz:.2f}, {moved:.1f} m away, ground z={gz:.2f}")
    rec["stats"]["drop"] = {"cubes": len(cubes), "fell_through": len(through), "off_edge": len(off_edge),
                            "dynamic_bodies": len(start)}
    if through:
        common.add(rec, "FAIL", "drop", f"{len(through)} of {len(cubes)} dropped cubes fell through the ground: "
                                        f"{through[0]}")
    if off_edge:
        common.add(rec, "WARN", "drop", f"{len(off_edge)} of {len(cubes)} dropped cubes rolled off an edge and fell "
                                        f">0.3 m: {off_edge[0]}")
    fell = [path for path, z in start.items() if physx.get_rigidbody_transformation(path)["position"][2] < z - 1.0]
    if fell:
        common.add(rec, "FAIL", "drop", f"{len(fell)} of {len(start)} scene rigid bodies fell >1 m, e.g. {fell[0]}")


# ── parent: one Kit process per scene ────────────────────────────────────────────────────────

def parent(a) -> None:
    scenes = common.find_scenes(a)
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    statuses = []
    for name, scene_dir, usd in scenes:
        t0 = time.perf_counter()
        job = {"usd": str(usd), "scene_dir": str(scene_dir), "result": str(out / f"{name}.json")}
        rec = common.run_kit(__file__, job, out / f"{name}.log", out / ".kit_tmp")
        Path(job["result"]).write_text(json.dumps(rec, indent=1))
        statuses.append(common.status(rec))
        s, g = rec["stats"], rec["stats"].get("ground", {})
        print(f"[sim] {statuses[-1]:5} {name}  {time.perf_counter() - t0:.0f}s, {s['peak_ram_gb']} GB RAM, "
              f"load {s.get('load_s', '?')}s, {s.get('glb_payloads', '?')} glb, {s.get('colliders', '?')} colliders, "
              f"ground {g.get('hits', '?')}/{g.get('probes', '?')} z={g.get('z')} {g.get('hit_types', '')}, "
              f"{s['log_errors']} Kit errors", flush=True)
        common.print_issues(rec, 8)
    common.summary("sim", statuses, f"logs and records in {out}")


def main() -> None:
    if sys.argv[1:2] == ["--one"]:  # inside Kit, started by common.run_kit()
        sys.stdout.reconfigure(line_buffering=True)
        child(json.loads(sys.argv[2]))
        return
    p = common.parser(__doc__)
    p.add_argument("--out", default="sanity_sim", help="Directory for the logs and records (default ./sanity_sim).")
    parent(p.parse_args())


if __name__ == "__main__":
    main()
