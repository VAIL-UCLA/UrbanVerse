#!/usr/bin/env python3
"""Visual sanity check: one image per scene, rendered in Isaac Sim 5.x from its canonical view.

    python scripts/sanity_check_render.py                    # every scene, one by one
    python scripts/sanity_check_render.py scene_03           # one scene, by id or name
    python scripts/sanity_check_render.py /path/to/scene     # one scene, by path (its dir or root .usd)

Scenes are looked up under --root, by default the CraftBench folder of the urbanverse-scene
toolkit (what ``uvs.set(...)`` or ``download_craftbench.py --root`` filled). Each scene gets
a fresh headless Kit process: the stage is opened, a camera is spawned at the scene's
canonical view, and once the stage has loaded and the frame has stopped changing it is
saved as <out>/<scene>.png. One image per scene; logs go to <out>/logs/. Kit's temporary
files and texture cache (1-2 GB a scene) go to <out>/.kit_tmp and are deleted after each
scene, so nothing is left in /tmp or ~/.cache/ov. Isaac Sim needs a lot of RAM for these
scenes, 13-25 GB at peak for CraftBench (each scene's line reports it): on a 32 GB machine,
close other big programs first.

The canonical view of a scene is, in this order:

  --eye / --target    given on the command line (one scene only);
  the views file      scenes/craftbench_canonical_views.json: per scene, the camera at
                      ``eye`` looks at ``target`` (world coordinates, Z up, meters);
  cam0_to_world.txt   the first pose of the scene's preview flythrough, which is what the
                      views file holds for the CraftBench scenes;
  the scene bounds    the whole scene framed from the front-top (WARN: give it a view).

Frames are rendered with RTX translucency on. Isaac Lab launches Kit with it off (as its
'balanced' and 'performance' rendering modes do; 'quality' turns it on), and then a model
with a transmissive glass material is invisible to cameras as a whole: in CraftBench that is
many cars, buses and building fronts. --no-translucency shows what such a camera sees.

The images are for you to look at, best against the preview_front.png the scene ships with,
a slightly different viewpoint rendered in Isaac Sim 4.5 (--compare saves the two side by
side). What the script can tell by itself, per scene:

  frame    FAIL if the frame is mostly flat black/white, WARN if a good part of it is: a
           surface that lost its material renders like that, e.g. a road with a scalar
           texture_scale in Isaac Sim 5. It does not notice a missing object.
  log      Kit logged no missing-file, file-format or USD->MDL type errors.

A scene is FAIL when it renders wrong, ERROR when no image could be made (e.g. Kit hung
after the GPU ran out of memory: free the GPU or lower --tex-budget); exit 1 on either.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sim", _HERE / "sanity_check_sim.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

PATTERN = "Collected_export_version/export_version.usd"
VIEWS = _HERE.parent / "scenes" / "craftbench_canonical_views.json"
PREVIEW = "preview_front.png"
FOCAL_MM = 15.0  # 70 deg horizontal with Kit's 20.955 mm aperture: what the shipped previews were rendered with
CAMERA = "/SanityCam"  # root level: CraftBench's /World is translated, the views are world poses


# ── views ────────────────────────────────────────────────────────────────────────────────────

def default_root() -> Path:
    """<cache>/CraftBench of the urbanverse-scene toolkit: where uvs.set() pointed it, else its default."""
    try:
        cache = Path(json.loads((Path.home() / ".cache" / "urbanverse_scenes_config.json").read_text())["cache_dir"])
    except (OSError, ValueError, KeyError):
        cache = Path.home() / ".cache" / "urbanverse_scenes"
    return cache / "CraftBench"


def canonical_view(name: str, scene_dir: Path, views: dict) -> dict | None:
    """The scene's view from the views file, else the first pose of its cam0_to_world.txt, else None."""
    view = views.get("views", {}).get(name)
    if view:
        return {"focal_mm": views.get("focal_mm", FOCAL_MM), **view, "source": "the views file"}
    cam = scene_dir / "cam0_to_world.txt"
    if cam.is_file():
        row = next((line.split() for line in cam.read_text().splitlines() if line.strip()), None)
        if row and len(row) == 17:
            m = [float(v) for v in row[1:]]  # row-major camera-to-world; the camera looks down its -Z
            eye = [m[3], m[7], m[11]]
            return {"eye": eye, "target": [e - 20.0 * z for e, z in zip(eye, (m[2], m[6], m[10]))],
                    "focal_mm": FOCAL_MM, "source": "cam0_to_world.txt"}
    return None


def framing_view(stage) -> dict:
    """The whole scene from the front-top, for a scene nobody gave a view."""
    from pxr import Usd, UsdGeom

    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]) \
        .ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    lo, hi = box.GetMin(), box.GetMax()
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    return {"eye": [(lo[0] + hi[0]) / 2, lo[1] - 0.6 * span, hi[2] + 0.5 * span],
            "target": [(lo[i] + hi[i]) / 2 for i in range(3)], "focal_mm": FOCAL_MM, "source": "the scene bounds"}


# ── inside Kit ───────────────────────────────────────────────────────────────────────────────

def spawn_camera(stage, view: dict, path: str = CAMERA):
    """A camera prim at view['eye'] looking at view['target'], Z up."""
    from pxr import Gf, UsdGeom

    eye, target = Gf.Vec3d(*view["eye"]), Gf.Vec3d(*view["target"])
    forward = (target - eye).GetNormalized()
    up = Gf.Vec3d(0, 1, 0) if abs(forward[2]) > 0.999 else Gf.Vec3d(0, 0, 1)  # straight down: north is up
    cam = UsdGeom.Camera.Define(stage, path)
    cam.GetFocalLengthAttr().Set(float(view.get("focal_mm", FOCAL_MM)))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100000.0))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse())
    return cam


def frame_stats(frame) -> dict:
    """Brightness, and the share of the frame that is flat black / flat white: 8x8 px blocks that
    are near-black (near-white) and show no texture at all."""
    gray = frame.mean(axis=2)
    h, w = gray.shape[0] // 8, gray.shape[1] // 8
    blocks = gray[:h * 8, :w * 8].reshape(h, 8, w, 8).swapaxes(1, 2).reshape(h, w, 64)
    mean, plain = blocks.mean(axis=2), blocks.max(axis=2) - blocks.min(axis=2) <= 2
    return {"mean": round(float(gray.mean()), 1), "flat_black": round(float((plain & (mean < 32)).mean()), 3),
            "flat_white": round(float((plain & (mean > 232)).mean()), 3)}


def blank(stats: dict) -> bool:
    """Mostly flat black/white: no picture, or not yet."""
    return stats["flat_black"] + stats["flat_white"] > 0.5 or stats["mean"] < 10


def render(app, ctx, camera: str, res: tuple, a, rec: dict, settings: dict):
    """The camera's frame once the stage has loaded and the frame has stopped changing (HxWx3 uint8),
    rendered with the given Kit settings."""
    import carb
    import numpy as np
    import omni.replicator.core as rep

    kit = carb.settings.get_settings()

    def hold() -> None:  # opening a stage puts the render settings back to the Kit experience's
        for key, value in settings.items():
            if kit.get(key) != value:
                kit.set(key, value)

    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rep.create.render_product(camera, res))
    t0 = said = time.perf_counter()
    while True:  # plain app updates: rep.orchestrator.step() can block forever when headless
        hold()
        app.update()
        _, loaded, total = ctx.get_stage_loading_status()
        now = time.perf_counter()
        if loaded >= total and now - t0 > 10:  # 0/0 once idle; give payload requests time to queue
            break
        if now - t0 > a.max_wait:
            sim.add(rec, "FAIL", "load", f"still loading {loaded}/{total} files after {a.max_wait:.0f}s")
            break
        if now - said > 10:  # also keeps the log growing, which is how the parent tells busy from hung
            said = now
            print(f"[sanity-child] loading {loaded}/{total} files, {now - t0:.0f}s", flush=True)
    rec["stats"]["load_s"] = round(time.perf_counter() - t0, 1)

    # Materials compile and textures stream in after that, and a finished frame is static: render
    # until 3 looks, 5 s apart, are the same. A blank frame may be a placeholder, so it gets all of
    # --max-settle to turn into a picture.
    t1, frame, same = time.perf_counter(), None, 0
    while True:
        t = time.perf_counter()
        while time.perf_counter() - t < 5:
            hold()
            app.update()
        data, last = rgb.get_data(), frame
        frame = None if data is None or not np.asarray(data).size else np.asarray(data)[..., :3].astype(np.uint8)
        change = None if frame is None or last is None else float(np.abs(frame.astype(np.int16) - last).mean())
        same = same + 1 if change is not None and change < 0.1 else 0
        spent = time.perf_counter() - t1
        print(f"[sanity-child] rendered {spent:.0f}s, frame " + ("is the first" if change is None else
              f"changed by {change:.2f} of 255"), flush=True)
        if same >= 2 and spent >= (a.max_settle if frame is None or blank(frame_stats(frame)) else a.settle):
            break
        if spent >= a.max_settle:
            if same < 2:
                sim.add(rec, "WARN", "frame", f"still changing after {spent:.0f}s of rendering, saved as it was")
            break
    rec["stats"]["settle_s"] = round(time.perf_counter() - t1, 1)
    rec["stats"]["settings"] = {key: kit.get(key) for key in settings}
    lost = [key for key, value in settings.items() if kit.get(key) != value]
    if lost:
        sim.add(rec, "ERROR", "frame", f"Kit did not keep {', '.join(lost)}: not rendered as asked")
    return frame


def beside_preview(frame, preview):
    """The preview, scaled to the frame's width, left of the frame; both cut to the rows they share
    (Kit keeps the horizontal field of view, so another aspect ratio shows more or fewer rows)."""
    import numpy as np
    from PIL import Image

    h, w = frame.shape[:2]
    ref = Image.open(preview).convert("RGB")
    ref = ref.resize((w, round(ref.height * w / ref.width)))
    rows = min(h, ref.height)
    top, ref_top = (h - rows) // 2, (ref.height - rows) // 2
    return np.hstack([np.asarray(ref.crop((0, ref_top, w, ref_top + rows))), frame[top:top + rows]])


def child(a) -> None:
    from isaaclab.app import AppLauncher

    job = json.loads(a.one)
    a.headless = True
    a.enable_cameras = True
    a.kit_args = "--enable omni.kit.asset_converter"  # the .glb plugin has to be there at startup
    if a.tex_budget:  # cap streamed textures: the 4K ground textures alone can fill a small or busy GPU
        a.kit_args += (" --/rtx-transient/resourcemanager/enableTextureStreaming=true"
                       f" --/rtx-transient/resourcemanager/texturestreaming/memoryBudget={a.tex_budget}")
    if os.environ.get("TMPDIR"):
        # Kit caches every texture it loads in ~/.cache/ov/texturecache, keyed by file time. The .glb
        # textures are extracted anew each run, so theirs pile up there, ~0.5 GB a scene: cache in the
        # parent's per-scene folder instead. Not in kit_args, which are split at spaces.
        sys.argv.append(f"--/rtx-transient/resourcemanager/localTextureCachePath={os.environ['TMPDIR']}/texturecache")
    app = AppLauncher(a).app

    import omni.usd
    from PIL import Image

    rec = {"issues": [], "stats": {}}
    try:
        ctx = omni.usd.get_context()
        if not ctx.open_stage(job["usd"]):
            sim.add(rec, "FAIL", "load", f"could not open {job['usd']}")
            return
        stage = ctx.get_stage()
        view = job["view"]
        if view is None:
            view = framing_view(stage)
            sim.add(rec, "WARN", "view", "no canonical view for this scene, framed it from its bounds: "
                                         "give --eye and --target, or add it to the views file")
        rec["view"] = view
        spawn_camera(stage, view)
        # Isaac Lab's Kit experience has translucency off, which hides every model with a transmissive material
        frame = render(app, ctx, CAMERA, tuple(job["res"]), a, rec,
                       {"/rtx/translucency/enabled": not a.no_translucency})
        if frame is None:
            sim.add(rec, "ERROR", "frame", "the renderer returned no frame")
            return
        stats = rec["stats"]["frame"] = frame_stats(frame)
        if blank(stats):
            sim.add(rec, "FAIL", "frame", f"mostly flat black/white: {stats}")
        for tone in ("black", "white"):
            if not blank(stats) and stats[f"flat_{tone}"] > a.max_flat:
                sim.add(rec, "WARN", "frame", f"{stats[f'flat_{tone}']:.0%} of the frame is flat {tone}: a surface "
                                              f"may have lost its material, compare with {PREVIEW}")
        Image.fromarray(beside_preview(frame, job["preview"]) if job["preview"] else frame).save(job["image"])
    except Exception as e:  # noqa: BLE001
        sim.add(rec, "ERROR", "run", f"{type(e).__name__}: {e}")
    finally:
        Path(job["result"]).write_text(json.dumps(rec, indent=1))
        print("[sanity-child] result written", flush=True)
        os._exit(0)  # nothing to save, and Kit's shutdown can hang (see sanity_check_sim.py)


# ── parent: loop scenes, one child each ──────────────────────────────────────────────────────

def find_scenes(a) -> list:
    """(name, scene dir, root layer) of each scene argument, or of every scene under --root."""
    root = Path(a.root).expanduser().resolve() if a.root else default_root()
    known = sorted(d for d in root.iterdir() if d.is_dir() and (d / a.pattern).is_file()) if root.is_dir() else []
    if not a.scene:
        if not known:
            raise SystemExit(f"no scene under {root} has {a.pattern}: pass --root, or a scene's path")
        return [(d.name, d, d / a.pattern) for d in known]
    scenes = []
    for want in a.scene:
        p = Path(want).expanduser()
        if p.is_file():  # the root layer itself
            p = p.resolve()
            d = p.parents[len(Path(a.pattern).parts) - 1] if p.as_posix().endswith("/" + a.pattern) else p.parent
            scenes.append((d.name, d, p))
        elif p.is_dir():
            if not (p / a.pattern).is_file():
                raise SystemExit(f"{p} has no {a.pattern}: pass the root .usd itself, or --pattern")
            scenes.append((p.resolve().name, p.resolve(), p.resolve() / a.pattern))
        else:
            hits = [d for d in known if d.name == want or d.name.startswith(want + "_")]
            if len(hits) != 1:
                raise SystemExit(f"{want!r} is not a path and matches {len(hits)} scenes under {root}")
            scenes.append((hits[0].name, hits[0], hits[0] / a.pattern))
    return scenes


def parent(a) -> None:
    scenes = find_scenes(a)
    names = [name for name, _, _ in scenes]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise SystemExit(f"{', '.join(twice)}: given more than once (a scene is named after its directory)")
    given = None
    if a.eye or a.target:
        if len(scenes) != 1 or not (a.eye and a.target):
            raise SystemExit("--eye and --target go together, for one scene")
        given = {"eye": a.eye, "target": a.target, "focal_mm": FOCAL_MM, "source": "the command line"}
    views = json.loads(Path(a.views).read_text()) if a.views and Path(a.views).is_file() else {}
    out = Path(a.out).expanduser().resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    print(f"[render] {len(scenes)} scene(s) -> {out}", flush=True)

    n = {"FAIL": 0, "ERROR": 0, "WARN": 0, "ok": 0}
    for name, d, usd in scenes:
        image, log, res = out / f"{name}.png", out / "logs" / f"{name}.log", out / "logs" / f"{name}.json"
        image.unlink(missing_ok=True)  # never leave an earlier run's image standing for this one
        res.unlink(missing_ok=True)
        view = given or canonical_view(name, d, views)
        if view and a.focal_mm:
            view["focal_mm"] = a.focal_mm
        job = {"usd": str(usd), "view": view, "res": a.res, "image": str(image), "result": str(res),
               "preview": str(d / PREVIEW) if a.compare and (d / PREVIEW).is_file() else None}
        cmd = [sys.executable, __file__, "--one", json.dumps(job), "--settle", str(a.settle),
               "--max-settle", str(a.max_settle), "--max-wait", str(a.max_wait), "--max-flat", str(a.max_flat),
               "--tex-budget", str(a.tex_budget)] + (["--no-translucency"] if a.no_translucency else [])
        t0 = time.perf_counter()
        for attempt in (1, 2):  # Kit now and then crashes while starting up
            rc, killed, peak = sim.run_child(cmd, log, out / ".kit_tmp" / name, a)
            if res.exists() or killed:
                break
        rec = json.loads(res.read_text()) if res.exists() else \
            {"issues": [{"level": "ERROR", "check": "run", "msg": f"Kit process ended ({rc}) without a result"}],
             "stats": {}}
        rec["stats"]["peak_ram_gb"] = round(peak, 1)
        sim.scan_log(log.read_text(errors="replace"), rec)
        if not image.is_file() and not any(i["level"] in ("FAIL", "ERROR") for i in rec["issues"]):
            sim.add(rec, "ERROR", "frame", "no image was saved")
        res.write_text(json.dumps(rec, indent=1))

        levels = {i["level"] for i in rec["issues"]}
        status = next((lv for lv in ("FAIL", "ERROR", "WARN") if lv in levels), "ok")
        n[status] += 1
        source = rec.get("view", view or {}).get("source")
        print(f"[render] {status:5} {name}  {time.perf_counter() - t0:.0f}s, {peak:.1f} GB RAM"
              + (f", view from {source}" if source else "")
              + (f"\n         -> {image}" if image.is_file() else ", no image"), flush=True)
        for i in rec["issues"]:
            if i["level"] != "INFO":
                print(f"         {i['level']:5} {i['check']}: {i['msg']}", flush=True)
    try:
        (out / ".kit_tmp").rmdir()
    except OSError:
        pass
    print(f"[render] {n['ok']} ok, {n['WARN']} WARN, {n['ERROR']} ERROR, {n['FAIL']} FAIL of {len(scenes)} scene(s); "
          f"images in {out}")
    if n["ok"] or n["WARN"]:
        print("[render] ok = an image was rendered and nothing in it or in Kit's log is plainly broken. Look at the "
              "images" + ("." if a.compare else f", against each scene's {PREVIEW} (--compare)."))
    if n["FAIL"] or n["ERROR"]:
        raise SystemExit(1)


def _xyz(text: str) -> list:
    v = [float(x) for x in text.replace(",", " ").split()]
    if len(v) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return v


def _res(text: str) -> list:
    v = [int(x) for x in text.lower().split("x")]
    if len(v) != 2:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT")
    return v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", nargs="*", help="Scene id (scene_03), name, directory or root .usd. Default: every "
                                            "scene under --root, one by one.")
    p.add_argument("--root", help="Directory with one sub-directory per scene. Default: the urbanverse-scene "
                                  "toolkit's CraftBench folder.")
    p.add_argument("--out", default="sanity_render", help="Directory for the images (default ./sanity_render).")
    p.add_argument("--pattern", default=PATTERN, help=f"Root layer relative to each scene dir (default {PATTERN}).")
    p.add_argument("--views", default=str(VIEWS), help="Canonical views file (default scenes/craftbench_canonical_"
                                                       "views.json).")
    p.add_argument("--eye", type=_xyz, help="Camera position x,y,z for the one scene given, with --target.")
    p.add_argument("--target", type=_xyz, help="Point x,y,z the camera looks at.")
    p.add_argument("--focal-mm", type=float, help=f"Focal length, instead of the view's (default {FOCAL_MM:g}).")
    p.add_argument("--res", type=_res, default=[1280, 720], help="Image size (default 1280x720).")
    p.add_argument("--compare", action="store_true",
                   help=f"Save the scene's {PREVIEW} and the frame side by side, the preview on the left.")
    p.add_argument("--no-translucency", action="store_true",
                   help="Leave RTX translucency off, as Isaac Lab launches Kit: what a camera sees there, "
                        "without the models that have glass.")
    p.add_argument("--max-flat", type=float, default=0.07,
                   help="WARN if more than this share of the frame is flat black, or flat white (default 0.07).")
    p.add_argument("--settle", type=float, default=20.0,
                   help="Render at least this many seconds after the stage loaded (default 20), then until the "
                        "frame is static.")
    p.add_argument("--max-settle", type=float, default=180.0, help="Stop waiting for a static frame after this.")
    p.add_argument("--tex-budget", type=float, default=0.0,
                   help="Texture streaming budget as a fraction of GPU memory, e.g. 0.3 on a small or busy GPU "
                        "(default 0: Kit's own).")
    p.add_argument("--max-wait", type=float, default=900.0, help="FAIL if the stage is still loading after this.")
    p.add_argument("--timeout", type=float, default=1800.0, help="Kill a scene's Kit process after this many seconds.")
    p.add_argument("--stall", type=float, default=300.0,
                   help="Kill a scene's Kit process after this many seconds without log output (hung).")
    p.add_argument("--one", help=argparse.SUPPRESS)  # internal: render one scene inside Kit
    if "--one" in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
        sys.stdout.reconfigure(line_buffering=True)
        child(p.parse_args())
    else:
        p.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)  # always headless; accepted
        parent(p.parse_args())


if __name__ == "__main__":
    main()
