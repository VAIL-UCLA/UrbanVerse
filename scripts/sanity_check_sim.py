#!/usr/bin/env python3
"""Runtime sanity check of downloaded scenes in Isaac Sim 5.x, headless, one Kit process per scene.

    python scripts/sanity_check_sim.py --root /path/to/urbanverse_scenes/CraftBench --out /tmp/sanity_sim
    python scripts/sanity_check_sim.py --root ... --out ... --scene scene_03 --go2

Per scene, in a fresh Kit process (the .glb file-format plugin has to be enabled at
startup, see render_scene_preview.py):

  load       the stage opens and the loader drains; every .glb payload yields geometry
             unless the scene switched the model off (active = false, INFO).
  overrides  every scene-layer override into a .glb model matches a prim there. One that
             does not (the .glb node got another name, e.g. Isaac Sim 5 names non-ASCII glTF
             nodes differently from 4.5) loses its placement, collision or active = false:
             the model shows up where the scene did not put it, without collision.
  colliders  every other CollisionAPI prim is real geometry, else that object is walk-through.
  log        Kit logged no missing-file, file-format or USD->MDL type errors (the latter is
             exactly what the sim-ready texture_scale fix removed). PhysX errors WARN.
  render     an RTX frame from cam0_to_world.txt pose 0 (the shipped preview_front.png
             pose) is saved with its brightness, share of pure black/white pixels and its
             correlation with preview_front.png. --no-render skips it.
  ground     PhysX raycasts straight down from --probes poses along the camera path reach
             static ground (objects on the way, e.g. a bin, are looked through). The hit prim
             types show whether that ground is the road mesh or an infinite fallback Plane. A
             pose with no ground below only WARNs: the preview camera path need not be walkable.
  drop       where nothing stands on that ground, a 20 cm cube is dropped from 0.5 m: falling
             straight through FAILs (ray-visible but not solid), rolling off an edge WARNs.
             Dynamic rigid bodies of the scene must not fall >1 m.
  go2        (--go2) physics_check_go2.py at the first ground hit.

Writes <out>/<scene>.{log,json} and <scene>_render.png; Kit's temporary files go to
<out>/.kit_tmp/<scene> and are deleted after each scene. A scene is FAIL when it is broken,
ERROR when the check itself could not finish (e.g. Kit hung after the GPU ran out of memory:
rerun it, with --no-render if the GPU is busy); exit 1 on either.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PATTERN = "Collected_export_version/export_version.usd"
_HERE = Path(__file__).resolve().parent

# Kit log lines that mean a scene is not loading as authored.
LOG_FAIL = {
    "usd->mdl type error": re.compile(r"Tried to assign a '\w+'\(USD\) to a '\w+'\(MDL\)"),
    "file format": re.compile(r"Cannot determine file format"),
    "missing file": re.compile(r"Could not open asset|Failed to open layer|Failed to resolve|Unresolved asset|"
                               r"Could not find file|file not found", re.I),
}
# The GPU ran out of memory: a property of this machine, not of the scene. It shows up as
# failed texture uploads and material snapshots, which only count against the scene without it.
LOG_GPU_OOM = re.compile(r"OUT_OF_DEVICE_MEMORY|Out of GPU memory|vkAllocateMemory failed|Unable to allocate buffer")
LOG_TEXTURE = re.compile(r"Texture upload failed|Failed texture loads reported|Failed to create source texture")
# Colliders PhysX rejected, e.g. a mesh without points or contactOffset <= restOffset.
LOG_PHYSX = re.compile(r"\[Error\] \[omni\.(physx|physicsschema)[\w.]*\]")
LOG_ERROR = re.compile(r"\[Error\] \[([\w.]+)\] (.*)")


def add(rec: dict, level: str, check: str, msg: str) -> None:
    rec["issues"].append({"level": level, "check": check, "msg": msg})


# ── child: one scene inside Kit ───────────────────────────────────────────────────────────────

def child(a) -> None:
    from isaaclab.app import AppLauncher

    a.headless = True
    a.enable_cameras = not a.no_render
    # glb plugin at startup; cap streamed textures so ~1 Gpx of 4K ground textures fit on a 16 GB card
    a.kit_args = ("--enable omni.kit.asset_converter --/rtx-transient/resourcemanager/enableTextureStreaming=true "
                  f"--/rtx-transient/resourcemanager/texturestreaming/memoryBudget={a.tex_budget}")
    app = AppLauncher(a).app

    import numpy as np
    import omni.usd
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    scene_dir = Path(a.one)
    out = Path(a.out)
    rec = {"issues": [], "stats": {}}
    try:
        cams = []
        cam_file = scene_dir / "cam0_to_world.txt"
        if cam_file.is_file():
            for line in cam_file.read_text().splitlines():
                if line.strip():
                    # row-major, translation in the 4th column; Gf wants it in the 4th row
                    cams.append(np.array(line.split()[1:], dtype=float).reshape(4, 4).T)

        ctx = omni.usd.get_context()
        t0 = time.perf_counter()
        if not ctx.open_stage(str(scene_dir / a.pattern)):
            add(rec, "FAIL", "load", f"could not open {a.pattern}")
            return
        stage = ctx.get_stage()
        while True:
            app.update()
            _, loaded, total = ctx.get_stage_loading_status()
            waited = time.perf_counter() - t0
            if loaded >= total and waited > 10:  # 0/0 once idle; give payload requests time to queue
                break
            if waited > a.max_wait:
                add(rec, "FAIL", "load", f"still loading {loaded}/{total} files after {a.max_wait:.0f}s")
                break
        rec["stats"]["load_s"] = round(time.perf_counter() - t0, 1)

        glb_roots = [p for p in stage.Traverse() if p.GetMetadata("payload") and any(
            i.assetPath.lower().endswith(".glb") for i in p.GetMetadata("payload").GetAddedOrExplicitItems())]
        empty = [p for p in glb_roots if not any(q.IsA(UsdGeom.Mesh) for q in Usd.PrimRange(p))]
        # scene_03 ("..._clean") switches most of its models off with active = false on their nodes
        off = [p for p in empty if any(not q.IsActive() for q in Usd.PrimRange(p, Usd.PrimAllPrimsPredicate))]
        broken = [str(p.GetPath()) for p in empty if p not in off]
        rec["stats"]["glb_payloads"] = len(glb_roots)
        rec["stats"]["glb_switched_off"] = len(off)
        if off:
            add(rec, "INFO", "load", f"{len(off)} of {len(glb_roots)} .glb models are switched off by the scene "
                                     f"(active = false), e.g. {off[0].GetPath()}")
        if broken:
            add(rec, "FAIL", "load", f"{len(broken)} of {len(glb_roots)} .glb payloads loaded no mesh, e.g. {broken[0]}")

        # topmost overrides with nothing under them: the scene layers author 'over <node>' into the model
        orphans = [q for p in glb_roots for q in Usd.PrimRange(p, Usd.PrimAllPrimsPredicate)
                   if not q.IsDefined() and q.GetParent().IsDefined()]
        rec["stats"]["orphan_overrides"] = len(orphans)
        if orphans:
            lost = [f"{what} x{n}" for what, f in (
                ("placement", lambda q: bool(UsdGeom.Xformable(q).GetOrderedXformOps())),
                ("collision", lambda q: q.HasAPI(UsdPhysics.CollisionAPI)),
                ("active = false", lambda q: not q.IsActive())) if (n := sum(map(f, orphans)))]
            add(rec, "FAIL", "overrides", f"{len(orphans)} scene override(s) into .glb models match no prim (the node "
                                          f"has another name in this Isaac Sim), losing {', '.join(lost)}: the model is "
                                          f"misplaced, walk-through or back on, e.g. {orphans[0].GetPath()}")

        orphan_paths = {q.GetPath() for q in orphans}
        colliders = [p for p in stage.TraverseAll() if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsActive()]
        hollow = [str(p.GetPath()) for p in colliders if p.GetPath() not in orphan_paths and (
            not p.IsA(UsdGeom.Gprim) or (p.IsA(UsdGeom.Mesh) and not UsdGeom.Mesh(p).GetPointsAttr().Get()))]
        rec["stats"]["colliders"] = len(colliders)
        if hollow:
            add(rec, "FAIL", "colliders", f"{len(hollow)} of {len(colliders)} colliders have no geometry "
                                          f"(walk-through), e.g. {hollow[0]}")

        if not a.no_render and cams:
            import omni.replicator.core as rep

            cam = UsdGeom.Camera.Define(stage, "/SanityCam")
            cam.GetFocalLengthAttr().Set(a.focal_mm)
            cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100000.0))
            xf = UsdGeom.Xformable(cam)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d(*cams[0].flatten().tolist()))
            rp = rep.create.render_product("/SanityCam", (1280, 720))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach(rp)
            t1 = time.perf_counter()
            while time.perf_counter() - t1 < a.settle:  # async MDL compile + texture streaming
                app.update()
            img = np.asarray(rgb.get_data())[..., :3].astype(np.uint8)
            Image.fromarray(img).save(out / f"{scene_dir.name}_render.png")
            gray = img.mean(axis=2)
            stats = {"mean": round(float(gray.mean()), 1),
                     "black": round(float((gray < 8).mean()), 3), "white": round(float((gray > 247).mean()), 3)}
            preview = scene_dir / "preview_front.png"
            if preview.is_file():
                ref = np.asarray(Image.open(preview).convert("L").resize((img.shape[1], img.shape[0])), dtype=float)
                stats["corr_preview"] = round(float(np.corrcoef(gray.ravel(), ref.ravel())[0, 1]), 3)
            rec["stats"]["render"] = stats
            if stats["black"] + stats["white"] > 0.5 or stats["mean"] < 10:
                add(rec, "FAIL", "render", f"frame is mostly flat black/white: {stats}")
            elif stats.get("corr_preview", 1.0) < a.min_corr:
                add(rec, "WARN", "render", f"frame differs from preview_front.png: {stats}")

        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationCfg, SimulationContext
        from omni.physx import get_physx_interface, get_physx_scene_query_interface

        sim = SimulationContext(SimulationCfg(dt=1 / 120, device="cpu", enable_scene_query_support=True))
        sim.reset()
        sq = get_physx_scene_query_interface()

        def is_object(path: str) -> bool:  # bins, cars, ... are (kinematic) rigid bodies; ground is static
            p = stage.GetPrimAtPath(path) if path.startswith("/") else None
            while p and p.IsValid() and not p.IsPseudoRoot():
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    return True
                p = p.GetParent()
            return False

        pose_ids = np.linspace(0, len(cams) - 1, a.probes).round().astype(int) if cams else []
        hits, misses = [], []
        for k in pose_ids:
            x, y, z = (float(v) for v in cams[k][3, :3])
            found = []

            def report(h, found=found):
                found.append((h.distance, float(h.position[2]), h.collision))
                return True

            sq.raycast_all([x, y, z], [0.0, 0.0, -1.0], abs(z) + 100.0, report)  # some paths end 190 m up
            found.sort()
            ground = next(((hz, col) for _, hz, col in found if not is_object(col)), None)
            if ground:  # clear = nothing stands on the ground here, so a dropped cube meets the ground
                hits.append((x, y, *ground, found[0][2] == ground[1]))
            else:
                misses.append(f"pose {k} at ({x:.1f}, {y:.1f})")
        hit_types = {}
        for _, _, _, path, _ in hits:  # an infinite UsdGeom.Plane fallback vs the actual road/sidewalk mesh
            t = stage.GetPrimAtPath(path).GetTypeName() if path.startswith("/") else "?"
            hit_types[t] = hit_types.get(t, 0) + 1
        rec["stats"]["ground"] = {"probes": len(pose_ids), "hits": len(hits), "hit_types": hit_types,
                                  "z": [round(min(h[2] for h in hits), 2), round(max(h[2] for h in hits), 2)] if hits else None}
        rec["ground_hits"] = [list(h) for h in hits]
        if misses:  # the preview camera path need not be walkable, so a gap under it is only suspicious
            add(rec, "WARN", "ground", f"{len(misses)} of {len(pose_ids)} camera poses have no static ground "
                                       f"below: {', '.join(misses[:3])}")

        dynamic = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
                   and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False
                   and not UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get()]
        cube = sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                   mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                   collision_props=sim_utils.CollisionPropertiesCfg())
        probes = []
        for i, (x, y, gz, _, clear) in enumerate(hits):
            if not clear:
                continue
            path = f"/SanityProbe_{i:02d}"  # root level: CraftBench's /World is translated (-735.84, 490.36, 0)
            cube.func(path, cube, translation=(x, y, gz + 0.5))
            probes.append((path, x, y, gz))
        if probes or dynamic:
            sim.reset()
            px = get_physx_interface()
            start = {}
            for p in dynamic:  # bodies PhysX did not create (e.g. hollow) have no transform
                t = px.get_rigidbody_transformation(p)
                if t["ret_val"]:
                    start[p] = t["position"][2]
            dynamic = list(start)
            for _ in range(int(a.drop_s / (1 / 120))):
                sim.step(render=False)
            through, off_edge = [], []
            for path, x, y, gz in probes:
                px_, py_, pz_ = px.get_rigidbody_transformation(path)["position"]
                if pz_ < gz - 0.3:  # straight down = the ground is not solid; sideways = rolled off an edge
                    moved = float(np.hypot(px_ - x, py_ - y))
                    (through if moved < 0.3 else off_edge).append(
                        f"{path} at ({x:.1f}, {y:.1f}) ended at z={pz_:.2f}, {moved:.1f} m away, ground z={gz:.2f}")
            rec["stats"]["drop"] = {"cubes": len(probes), "fell_through": len(through), "off_edge": len(off_edge),
                                    "dynamic_bodies": len(dynamic)}
            if through:
                add(rec, "FAIL", "drop", f"{len(through)} of {len(probes)} dropped cubes fell through the ground: "
                                         f"{through[0]}")
            if off_edge:
                add(rec, "WARN", "drop", f"{len(off_edge)} of {len(probes)} dropped cubes rolled off an edge and fell "
                                         f">0.3 m: {off_edge[0]}")
            fell = [p for p in dynamic if px.get_rigidbody_transformation(p)["position"][2] < start[p] - 1.0]
            if fell:
                add(rec, "FAIL", "drop", f"{len(fell)} of {len(dynamic)} scene rigid bodies fell >1 m, e.g. {fell[0]}")
    except Exception as e:  # noqa: BLE001
        add(rec, "ERROR", "run", f"{type(e).__name__}: {e}")
    finally:
        (out / f"{scene_dir.name}.json").write_text(json.dumps(rec, indent=1))
        print("[sanity-child] result written", flush=True)
        # Nothing to save, and Kit's shutdown can hang on Linux with texture streaming on
        # (seen here: minutes of per-frame texture-upload errors inside app.close()).
        os._exit(0)


# ── parent: loop scenes, one child each ──────────────────────────────────────────────────────

def scan_log(text: str, rec: dict) -> None:
    lines = text.splitlines()
    for name, rx in LOG_FAIL.items():
        hits = [line for line in lines if rx.search(line)]
        if hits:
            add(rec, "FAIL", "log", f"{len(hits)} '{name}' line(s), e.g. {hits[0].strip()[:220]}")
    oom = sum(bool(LOG_GPU_OOM.search(line)) for line in lines)
    tex = [line for line in lines if LOG_TEXTURE.search(line)]
    if oom:
        add(rec, "WARN", "env", f"GPU ran out of memory while rendering ({oom} lines, {len(tex)} texture "
                                f"failures): the render is incomplete; lower --tex-budget or free the GPU")
    elif tex:
        add(rec, "FAIL", "log", f"{len(tex)} texture load failure(s), e.g. {tex[0].strip()[:220]}")
    physx = [line for line in lines if LOG_PHYSX.search(line)]
    if physx:
        add(rec, "WARN", "log", f"{len(physx)} PhysX error(s), e.g. {physx[0].strip()[:220]}")
    errors = {}
    for m in LOG_ERROR.finditer(text):
        key = f"[{m.group(1)}] {m.group(2)[:160]}"
        errors[key] = errors.get(key, 0) + 1
    rec["stats"]["log_errors"] = sum(errors.values())
    rec["log_errors"] = dict(sorted(errors.items(), key=lambda kv: -kv[1])[:40])


def run_child(cmd: list, log: Path, tmp: Path, a) -> tuple:
    """Run one Kit child, logging to `log`. Returns (exit status, whether it was killed)."""
    # The .glb importer extracts every model's textures to $TMPDIR/<hash>/textures and only
    # removes them on a clean shutdown, which the child skips: give it a TMPDIR we delete.
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t0 = time.perf_counter()
    try:
        with open(log, "w") as f:
            # After a GPU out-of-memory, a single app.update() can block forever, so the child
            # cannot time itself out: kill it when its log stalls or the scene runs too long.
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "TMPDIR": str(tmp)})
            size, last = 0, t0
            while (rc := proc.poll()) is None:
                time.sleep(2)
                now = time.perf_counter()
                if log.stat().st_size != size:
                    size, last = log.stat().st_size, now
                if now - last > a.stall or now - t0 > a.timeout:
                    proc.kill()
                    proc.wait()
                    return (f"killed, no output for {now - last:.0f}s" if now - last > a.stall
                            else f"killed after {now - t0:.0f}s"), True
        return rc, False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def parent(a) -> None:
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
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    n_fail = n_error = n_warn = 0
    for d in scenes:
        log, res = out / f"{d.name}.log", out / f"{d.name}.json"
        res.unlink(missing_ok=True)
        cmd = [sys.executable, __file__, "--one", str(d), "--pattern", a.pattern, "--out", str(out),
               "--probes", str(a.probes), "--settle", str(a.settle), "--focal-mm", str(a.focal_mm),
               "--min-corr", str(a.min_corr), "--drop-s", str(a.drop_s), "--max-wait", str(a.max_wait),
               "--tex-budget", str(a.tex_budget)]
        cmd += ["--no-render"] if a.no_render else []
        t0 = time.perf_counter()
        for attempt in (1, 2):  # Kit now and then crashes while starting up (seen: in its telemetry thread)
            rc, killed = run_child(cmd, log, out / ".kit_tmp" / d.name, a)
            if res.exists() or killed:
                break
        rec = json.loads(res.read_text()) if res.exists() else \
            {"issues": [{"level": "ERROR", "check": "run", "msg": f"Kit process ended ({rc}) without a result"}],
             "stats": {}}
        if attempt > 1:
            add(rec, "INFO", "run", "Kit crashed without a result on the first try; this is the second")
        scan_log(log.read_text(errors="replace"), rec)
        clear = [h for h in rec.get("ground_hits", []) if h[4]]
        if a.go2 and clear:
            x, y, gz, _, _ = clear[0]
            go2 = subprocess.run([sys.executable, str(_HERE / "physics_check_go2.py"), "--usd", str(d / a.pattern),
                                  "--spawn", f"{x},{y},{gz}", "--headless"],
                                 capture_output=True, text=True, timeout=a.timeout)
            (out / f"{d.name}_go2.log").write_text(go2.stdout + go2.stderr)
            result = re.search(r"\[go2\] RESULT: (\w+)", go2.stdout)
            rec["stats"]["go2"] = result.group(1) if result else "no result"
            if not result or result.group(1) != "PASS":
                add(rec, "FAIL", "go2", f"Go2 stand/shove check: {rec['stats']['go2']} (see {d.name}_go2.log)")
        res.write_text(json.dumps(rec, indent=1))

        levels = {i["level"] for i in rec["issues"]}
        status = next((lv for lv in ("FAIL", "ERROR", "WARN") if lv in levels), "ok")
        n_fail += status == "FAIL"
        n_error += status == "ERROR"
        n_warn += status == "WARN"
        s, g, r = rec["stats"], rec["stats"].get("ground", {}), rec["stats"].get("render", {})
        print(f"[sim] {status:4} {d.name}  {time.perf_counter() - t0:.0f}s, load {s.get('load_s', '?')}s, "
              f"{s.get('glb_payloads', '?')} glb, {s.get('colliders', '?')} colliders, "
              f"ground {g.get('hits', '?')}/{g.get('probes', '?')} z={g.get('z')} {g.get('hit_types', '')}, "
              + (f"render mean={r['mean']} black={r['black']} corr={r.get('corr_preview')}, " if r else "")
              + f"{s.get('log_errors', 0)} Kit errors", flush=True)
        for i in rec["issues"]:
            print(f"        {i['level']:5} {i['check']}: {i['msg']}", flush=True)
    print(f"[sim] {len(scenes) - n_fail - n_error - n_warn} ok, {n_warn} WARN, {n_error} ERROR, {n_fail} FAIL "
          f"of {len(scenes)} scene(s); logs, renders and JSON in {out}")
    if n_fail or n_error:
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="Directory with one sub-directory per scene.")
    p.add_argument("--out", required=True, help="Directory for logs, renders and per-scene JSON.")
    p.add_argument("--pattern", default=PATTERN, help=f"Root layer relative to each scene dir (default {PATTERN}).")
    p.add_argument("--scene", action="append", default=[], help="Scene name or id prefix (scene_03); repeatable.")
    p.add_argument("--probes", type=int, default=12, help="Camera-path poses to raycast and drop cubes at.")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--settle", type=float, default=60.0, help="Seconds of rendering before the capture.")
    p.add_argument("--focal-mm", type=float, default=18.0)
    p.add_argument("--min-corr", type=float, default=0.5, help="WARN below this correlation with preview_front.png.")
    p.add_argument("--drop-s", type=float, default=2.0, help="Simulated seconds for the cube drop.")
    p.add_argument("--tex-budget", type=float, default=0.3,
                   help="Texture streaming budget as a fraction of GPU memory (default 0.3).")
    p.add_argument("--max-wait", type=float, default=900.0, help="FAIL if the stage is still loading after this.")
    p.add_argument("--timeout", type=float, default=1200.0, help="Kill a scene's Kit process after this many seconds.")
    p.add_argument("--stall", type=float, default=300.0,
                   help="Kill a scene's Kit process after this many seconds without log output (hung).")
    p.add_argument("--go2", action="store_true", help="Also run physics_check_go2.py at the first ground hit.")
    p.add_argument("--one", help=argparse.SUPPRESS)  # internal: run one scene inside Kit
    if "--one" in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
        sys.stdout.reconfigure(line_buffering=True)
        child(p.parse_args())
    else:
        p.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)  # always headless; accepted
        a = p.parse_args()
        if not a.root:
            p.error("--root is required")
        parent(a)


if __name__ == "__main__":
    main()
