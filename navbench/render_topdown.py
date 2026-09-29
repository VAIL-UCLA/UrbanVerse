#!/usr/bin/env python3
"""Step 1: the top-down map of each scene, from its geometry. No Isaac Sim, no GPU.

    python navbench/render_topdown.py                    # every scene
    python navbench/render_topdown.py scene_03           # one scene, by id or name
    python navbench/render_topdown.py /path/to/scene     # one scene, by its dir or root .usd

Per scene, in maps/<scene>/:

  topdown.png   the scene from straight above
  cut.png       the same, without what is above head height: roofs and tree crowns are gone,
                walls show as lines
  layers.npz    per cell: ground (height of the ground), clearance (free height above it),
                top (height of the highest thing), kind (road, sidewalk, ...)
  map.json      where the map is in the world, and how it was made

A cell is --res meters (default 0.05). Column 0 is at x_min and row 0 at y_max, so +X is right
and +Y is up in the images; map.json has x_min, y_max and res.

The map covers what is around the scene's preview camera path (cam0_to_world.txt), --margin
meters to each side, cut down to where the scene has anything. --bounds sets it by hand.
docs/01_topdown.md explains the rest.
"""
import sys
import time

import numpy as np
from PIL import Image

from navbench import cli, topdown
from navbench.maps import KINDS, Grid
from navbench.scene import Scene


def window(scene: Scene, scene_dir, margin: float) -> tuple:
    """(x_min, y_min, x_max, y_max) to look at: around the camera path, else around most of the meshes."""
    path = scene_dir / "cam0_to_world.txt"
    if path.is_file():  # a frame number and a row-major 4x4 camera-to-world matrix per line
        rows = [line.split()[1:] for line in path.read_text().splitlines() if line.strip()]
        xy = np.array([(float(m[3]), float(m[7])) for m in rows])
    else:  # the middles of the meshes, without those that lie far off
        xy = np.array([s.box.mean(axis=0)[:2] for s in scene.sources()])
        far = np.linalg.norm(xy - np.median(xy, axis=0), axis=1)
        xy = xy[far <= max(30.0, 4 * np.median(far))]
    return tuple(xy.min(axis=0) - margin) + tuple(xy.max(axis=0) + margin)


def main() -> None:
    p = cli.parser(__doc__)
    p.add_argument("--out", default=str(cli.MAPS), help="Directory for the maps (default navbench/maps).")
    p.add_argument("--res", type=float, default=0.05, help="Cell size in meters (default 0.05).")
    p.add_argument("--margin", type=float, default=40.0,
                   help="Meters to map around the scene's camera path (default 40).")
    p.add_argument("--bounds", type=lambda t: tuple(float(v) for v in t.split(",")), metavar="X0,Y0,X1,Y1",
                   help="Map exactly this part of the world, for the one scene given.")
    p.add_argument("--level", type=float, help="Street level in meters, if the script gets it wrong.")
    p.add_argument("--workers", type=int, default=0,
                   help="Processes to use (default: one per core, as many as free memory holds at ~4 GB each).")
    a = p.parse_args()
    scenes = cli.find_scenes(a)
    if a.bounds and (len(a.bounds) != 4 or len(scenes) != 1):
        raise SystemExit("--bounds takes x_min,y_min,x_max,y_max, for one scene")

    for name, scene_dir, usd in scenes:
        t0 = time.perf_counter()
        scene = Scene(usd)
        level, content = topdown.survey(scene, a.bounds or window(scene, scene_dir, a.margin))
        grid = Grid.around(*(a.bounds or content), a.res)
        level = level if a.level is None else a.level
        print(f"[topdown] {name}: {len(scene.sources())} meshes, street level {level:.2f} m, "
              f"{grid.width} x {grid.height} cells of {a.res:g} m", flush=True)
        done = lambda n, of: print(f"\r[topdown]   tile {n}/{of}", end="", file=sys.stderr, flush=True)  # noqa: E731
        m, images = topdown.build(scene, name, grid, level, a.workers, done)
        print(file=sys.stderr)
        out = cli.Path(a.out).expanduser().resolve() / name
        m.save(out)
        for key, image in images.items():
            Image.fromarray(image).save(out / f"{key}.png")
        has = np.isfinite(m.ground)
        kinds = ", ".join(f"{k} {100 * np.mean(m.kind[has] == i):.0f}%" for i, k in enumerate(KINDS)
                          if (m.kind[has] == i).any())
        print(f"[topdown]   ground on {has.mean():.0%} of the map ({kinds}), {time.perf_counter() - t0:.0f}s "
              f"-> {out}", flush=True)
        for note in scene.notes:
            print(f"[topdown]   note: {note}")


if __name__ == "__main__":
    main()
