#!/usr/bin/env python3
"""Render a still of a USD scene headlessly with Isaac Sim's RTX renderer.

Used to eyeball scenes after the Isaac Sim 5 material patch, e.g. against the
4.5-era preview_front.png that ships with CraftBench::

    python scripts/render_scene_preview.py \
        --usd  /path/to/scene/Collected_export_version/export_version.usd \
        --out  /tmp/scene_front.png \
        --cam-to-world "$(sed -n '1p' /path/to/scene/cam0_to_world.txt | cut -d' ' -f2-)" \
        --headless

``--cam-to-world`` takes the 16 row-major floats of a camera-to-world matrix
(USD camera convention: looks down -Z, +Y up). Without it the camera is
auto-placed to frame the whole scene from the front-top.
"""
import argparse
import sys
import time

from isaaclab.app import AppLauncher

# Isaac Sim's shutdown can bypass interpreter cleanup; keep progress lines flowing.
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--usd", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--cam-to-world", type=str, default=None,
                    help="16 row-major floats (translation in the 4th column).")
parser.add_argument("--focal-mm", type=float, default=18.0, help="Focal length (default 18 = ~60 deg hfov).")
parser.add_argument("--res", type=str, default="1280x720")
parser.add_argument("--settle", type=float, default=90.0,
                    help="Seconds to keep rendering after the stage reports fully loaded, so async "
                         "MDL compilation and texture streaming finish before capture.")
parser.add_argument("--max-wait", type=float, default=900.0, help="Give up waiting for stage load after this.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # rendering kit experience, even when headless
# UrbanVerse scenes payload .glb files directly. The USD file-format plugin
# that opens them ships in omni.kit.asset_converter, and USD's format registry
# only sees plugins present when it first initialises - enabling the extension
# after launch is too late (the payloads fail with "Cannot determine file
# format"), so it has to go in at Kit startup.
_GLB_EXT = "--enable omni.kit.asset_converter"
args.kit_args = f"{args.kit_args} {_GLB_EXT}".strip() if args.kit_args else _GLB_EXT

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402


def main() -> None:
    if Sdf.FileFormat.FindByExtension("glb", "usd") is None:
        raise SystemExit("glb file format not registered - omni.kit.asset_converter did not start")

    ctx = omni.usd.get_context()
    ok = ctx.open_stage(args.usd)
    if not ok:
        raise SystemExit(f"could not open {args.usd}")
    stage = ctx.get_stage()
    for _ in range(4):
        simulation_app.update()

    cam = UsdGeom.Camera.Define(stage, "/RenderCam")
    cam.GetFocalLengthAttr().Set(args.focal_mm)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100000.0))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()

    if args.cam_to_world:
        vals = [float(v) for v in args.cam_to_world.replace(",", " ").split()]
        if len(vals) != 16:
            raise SystemExit("--cam-to-world needs 16 floats")
        # File is row-major with translation in the last column; Gf.Matrix4d
        # is row-vector convention (translation in the last row) -> transpose.
        m = np.array(vals).reshape(4, 4).T
        xf.AddTransformOp().Set(Gf.Matrix4d(*m.flatten().tolist()))
    else:
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]) \
            .ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        lo, hi = np.array(bbox.GetMin()), np.array(bbox.GetMax())
        c, size = (lo + hi) / 2, hi - lo
        eye = Gf.Vec3d(c[0], lo[1] - 0.6 * max(size[0], size[1]), hi[2] + 0.5 * max(size[0], size[1]))
        m = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(*c), Gf.Vec3d(0, 0, 1)).GetInverse()
        xf.AddTransformOp().Set(m)

    w, h = (int(v) for v in args.res.lower().split("x"))
    rp = rep.create.render_product("/RenderCam", (w, h))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)

    # Plain app updates (what Isaac Lab's Camera sensor does) rather than
    # rep.orchestrator.step(), which can block indefinitely headless.
    # Payloads, MDL compilation and textures all load asynchronously; the first
    # frames come back as flat black/white placeholders. Wait for the stage
    # loader to report complete, then keep rendering for --settle seconds.
    t0 = time.perf_counter()
    last = -1.0
    while True:
        simulation_app.update()
        msg, loaded, total = ctx.get_stage_loading_status()
        now = time.perf_counter() - t0
        if now - last >= 10:
            print(f"[render] loading {loaded}/{total} files, {now:.0f}s  {msg}")
            last = now
        # The status reports 0/0 once nothing is queued (and for scenes that
        # payload nothing), so an idle loader after a short grace period counts.
        if loaded >= total and now > 10:
            break
        if now > args.max_wait:
            print("[render] WARNING: stage still loading at --max-wait, capturing anyway")
            break
    print(f"[render] stage loaded after {time.perf_counter() - t0:.0f}s; settling {args.settle:.0f}s")
    t1 = time.perf_counter()
    n = 0
    while time.perf_counter() - t1 < args.settle:
        simulation_app.update()
        n += 1
    print(f"[render] settled over {n} frames")
    data = rgb.get_data()
    img = np.asarray(data)[..., :3]
    Image.fromarray(img.astype(np.uint8)).save(args.out)
    print(f"[render] wrote {args.out}  {img.shape[1]}x{img.shape[0]}  mean={img.mean():.1f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
