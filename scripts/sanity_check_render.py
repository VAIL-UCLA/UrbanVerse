#!/usr/bin/env python3
"""Visual sanity check: one image per scene, rendered in Isaac Sim 5.x from its canonical view.

    python scripts/sanity_check_render.py                    # every scene, one by one
    python scripts/sanity_check_render.py scene_03           # one scene, by id or name
    python scripts/sanity_check_render.py /path/to/scene     # one scene, by its dir or root .usd

Scenes are looked up under --root, by default the urbanverse-scene toolkit's CraftBench folder
(what ``uvs.set(...)`` or ``download_craftbench.py --root`` filled). Each scene gets a fresh
headless Kit process: the stage is opened, a camera is spawned at the scene's canonical view,
and once the stage has loaded and the frame has stopped changing it is saved as
<out>/<scene>.png, with Kit's log and the scene's record in <out>/logs/. Rendering takes a lot
of RAM: 8-25 GB at peak per CraftBench scene (each scene's line reports it; loading without
rendering is 3-7 GB), so close other big programs first.

The canonical view of a scene is --eye/--target if given (one scene only), else its entry in
the views file (scenes/craftbench_canonical_views.json: the camera at ``eye`` looks at
``target``, world coordinates, Z up, meters), else the whole scene from the front-top (WARN).

Frames are rendered with RTX translucency on. Isaac Lab launches Kit with it off (as its
'balanced' and 'performance' rendering modes do; 'quality' turns it on), and then a model with
a transmissive glass material is invisible to cameras as a whole: in CraftBench that is many
cars, buses and building fronts. --no-translucency shows what such a camera sees.

The images are for you to look at, best against the preview_front.png each scene ships with, a
slightly different viewpoint rendered in Isaac Sim 4.5 (--compare saves the two side by side).
What the script can tell by itself, per scene:

  frame    FAIL if the frame is mostly flat black/white, WARN if a good part of it is: a
           surface that lost its material renders like that, e.g. a road with a scalar
           texture_scale in Isaac Sim 5. It does not notice a missing object.
  log      Kit logged no missing-file, file-format or USD->MDL type errors.

A scene is FAIL when it renders wrong, ERROR when no image could be made (e.g. Kit hung after
the GPU ran out of memory: free the GPU or lower --tex-budget); exit 1 on either.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import sanity_check_common as common

VIEWS = Path(__file__).resolve().parents[1] / "scenes" / "craftbench_canonical_views.json"
PREVIEW = "preview_front.png"
CAMERA = "/SanityCam"  # root level: CraftBench's /World is translated, and the views are world poses
FOCAL_MM = 15.0  # 70 deg horizontally with Kit's 20.955 mm aperture, as the shipped previews were rendered
TRANSLUCENCY = "/rtx/translucency/enabled"
SETTLE = 20.0  # s: render at least this long after the stage has loaded, then until the frame is static
MAX_SETTLE = 180.0  # s: stop waiting for a static frame after this
MAX_FLAT = 0.07  # WARN if more of the frame than this is flat black, or flat white


# ── inside Kit: one scene ────────────────────────────────────────────────────────────────────

def child(job: dict) -> None:
    kit_args = "--enable omni.kit.asset_converter"  # the .glb file-format plugin has to be there at startup
    if job["tex_budget"]:  # cap streamed textures: the 4K ground textures alone can fill a small or busy GPU
        kit_args += (" --/rtx-transient/resourcemanager/enableTextureStreaming=true"
                     f" --/rtx-transient/resourcemanager/texturestreaming/memoryBudget={job['tex_budget']}")
    # Kit caches every texture it loads in ~/.cache/ov/texturecache, keyed by file time. The .glb
    # textures are extracted anew each run, so theirs pile up there, ~0.5 GB a scene: cache them in
    # $TMPDIR, which the parent deletes, instead. Not in kit_args, which are split at spaces.
    sys.argv.append(f"--/rtx-transient/resourcemanager/localTextureCachePath={os.environ['TMPDIR']}/texturecache")
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, kit_args=kit_args).app

    import carb
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image

    settings = carb.settings.get_settings()

    def update() -> None:  # opening a stage puts the render settings back to the Kit experience's
        if settings.get(TRANSLUCENCY) != job["translucency"]:
            settings.set(TRANSLUCENCY, job["translucency"])
        app.update()

    rec = {"issues": [], "stats": {}}
    try:
        ctx = omni.usd.get_context()
        if not ctx.open_stage(job["usd"]):
            common.add(rec, "FAIL", "load", f"could not open {job['usd']}")
            return
        stage = ctx.get_stage()
        rec["view"] = job["view"] or framing_view(stage)
        if not job["view"]:
            common.add(rec, "WARN", "view", "no canonical view for this scene, framed it from its bounds: give --eye "
                                            "and --target, or add it to the views file")
        spawn_camera(stage, rec["view"])
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(rep.create.render_product(CAMERA, tuple(job["res"])))
        common.wait_for_load(update, ctx, rec)
        frame = settle(update, rgb, rec)
        rec["stats"]["translucency"] = settings.get(TRANSLUCENCY)
        if frame is None:
            common.add(rec, "ERROR", "frame", "the renderer returned no frame")
            return
        judge(frame, rec)
        Image.fromarray(beside_preview(frame, job["preview"]) if job["preview"] else frame).save(job["image"])
    except Exception as e:  # noqa: BLE001
        common.add(rec, "ERROR", "run", f"{type(e).__name__}: {e}")
    finally:
        common.finish(rec, job["result"])


def framing_view(stage) -> dict:
    """The whole scene from the front-top, for a scene nobody gave a view."""
    from pxr import Usd, UsdGeom

    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]) \
        .ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    lo, hi = box.GetMin(), box.GetMax()
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    return {"eye": [(lo[0] + hi[0]) / 2, lo[1] - 0.6 * span, hi[2] + 0.5 * span],
            "target": [(lo[i] + hi[i]) / 2 for i in range(3)]}


def spawn_camera(stage, view: dict) -> None:
    """A camera at view['eye'] looking at view['target'], Z up."""
    from pxr import Gf, UsdGeom

    eye, target = Gf.Vec3d(*view["eye"]), Gf.Vec3d(*view["target"])
    forward = (target - eye).GetNormalized()
    up = Gf.Vec3d(0, 1, 0) if abs(forward[2]) > 0.999 else Gf.Vec3d(0, 0, 1)  # straight down: north is up
    cam = UsdGeom.Camera.Define(stage, CAMERA)
    cam.GetFocalLengthAttr().Set(FOCAL_MM)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100000.0))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse())


def settle(update, rgb, rec: dict):
    """The camera's frame (HxWx3 uint8) once it has stopped changing, or None if the renderer gives
    none. Materials compile and textures stream in after the stage has loaded, and a finished frame
    is static: render until 3 looks, 5 s apart, are the same, and for at least SETTLE seconds. A
    blank frame may be a placeholder, so it gets MAX_SETTLE seconds to turn into a picture."""
    import numpy as np

    t0, frame, same = time.perf_counter(), None, 0
    while True:
        t = time.perf_counter()
        while time.perf_counter() - t < 5:
            update()
        data, last = rgb.get_data(), frame
        frame = None if data is None or not np.asarray(data).size else np.asarray(data)[..., :3].astype(np.uint8)
        change = None if frame is None or last is None else float(np.abs(frame.astype(np.int16) - last).mean())
        same = same + 1 if change is not None and change < 0.1 else 0
        spent = time.perf_counter() - t0
        what = "is the first" if change is None else f"changed by {change:.2f} of 255"
        print(f"[sanity-child] rendered {spent:.0f}s, frame {what}", flush=True)
        static = same >= 2
        if spent >= MAX_SETTLE or (static and spent >= SETTLE and not blank(frame_stats(frame))):
            if not static:
                common.add(rec, "WARN", "frame", f"still changing after {spent:.0f}s of rendering, saved as it was")
            break
    rec["stats"]["settle_s"] = round(time.perf_counter() - t0, 1)
    return frame


def judge(frame, rec: dict) -> None:
    """FAIL a frame that is mostly flat black/white, WARN one that is so in good part."""
    stats = rec["stats"]["frame"] = frame_stats(frame)
    if blank(stats):
        common.add(rec, "FAIL", "frame", f"mostly flat black/white: {stats}")
        return
    for tone in ("black", "white"):
        if stats[f"flat_{tone}"] > MAX_FLAT:
            common.add(rec, "WARN", "frame", f"{stats[f'flat_{tone}']:.0%} of the frame is flat {tone}: a surface "
                                             f"may have lost its material, compare with {PREVIEW}")


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


def beside_preview(frame, preview: str):
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


# ── parent: one Kit process per scene ────────────────────────────────────────────────────────

def parent(a) -> None:
    scenes = common.find_scenes(a)
    if (a.eye or a.target) and not (a.eye and a.target and len(scenes) == 1):
        raise SystemExit("--eye and --target go together, for one scene")
    views = json.loads(Path(a.views).read_text())["views"]
    out = Path(a.out).expanduser().resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    print(f"[render] {len(scenes)} scene(s) -> {out}", flush=True)

    statuses = []
    for name, scene_dir, usd in scenes:
        t0 = time.perf_counter()
        image = out / f"{name}.png"
        image.unlink(missing_ok=True)  # never leave an earlier run's image standing for this one
        preview = scene_dir / PREVIEW
        job = {"usd": str(usd), "view": {"eye": a.eye, "target": a.target} if a.eye else views.get(name),
               "res": a.res, "translucency": not a.no_translucency, "tex_budget": a.tex_budget,
               "preview": str(preview) if a.compare and preview.is_file() else None,
               "image": str(image), "result": str(out / "logs" / f"{name}.json")}
        rec = common.run_kit(__file__, job, out / "logs" / f"{name}.log", out / ".kit_tmp")
        if not image.is_file() and common.status(rec) not in ("FAIL", "ERROR"):
            common.add(rec, "ERROR", "frame", "no image was saved")
        Path(job["result"]).write_text(json.dumps(rec, indent=1))
        statuses.append(common.status(rec))
        ram = rec["stats"]["peak_ram_gb"]
        print(f"[render] {statuses[-1]:5} {name}  {time.perf_counter() - t0:.0f}s, {ram} GB RAM"
              + (f"\n         -> {image}" if image.is_file() else ", no image"), flush=True)
        common.print_issues(rec, 9)
    common.summary("render", statuses, f"images in {out}: look at them"
                   + ("" if a.compare else f", beside each scene's {PREVIEW} (--compare)"))


def xyz(text: str) -> list:
    v = [float(x) for x in text.replace(",", " ").split()]
    if len(v) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return v


def resolution(text: str) -> list:
    v = [int(x) for x in text.lower().split("x")]
    if len(v) != 2:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT")
    return v


def main() -> None:
    if sys.argv[1:2] == ["--one"]:  # inside Kit, started by common.run_kit()
        sys.stdout.reconfigure(line_buffering=True)
        child(json.loads(sys.argv[2]))
        return
    p = common.parser(__doc__)
    p.add_argument("--out", default="sanity_render", help="Directory for the images (default ./sanity_render).")
    p.add_argument("--views", default=str(VIEWS), help="Canonical views file (default scenes/craftbench_canonical_"
                                                       "views.json).")
    p.add_argument("--eye", type=xyz, help="Camera position x,y,z for the one scene given, with --target.")
    p.add_argument("--target", type=xyz, help="Point x,y,z the camera looks at.")
    p.add_argument("--res", type=resolution, default=[1280, 720], help="Image size (default 1280x720).")
    p.add_argument("--compare", action="store_true",
                   help=f"Save the scene's {PREVIEW} and the frame side by side, the preview on the left.")
    p.add_argument("--no-translucency", action="store_true",
                   help="Leave RTX translucency off, as Isaac Lab launches Kit: what a camera sees there, without "
                        "the models that have glass.")
    p.add_argument("--tex-budget", type=float, default=0.0,
                   help="Texture streaming budget as a fraction of GPU memory, e.g. 0.3 on a small or busy GPU "
                        "(default 0: Kit's own).")
    parent(p.parse_args())


if __name__ == "__main__":
    main()
