#!/usr/bin/env python3
"""Step 4: plan every annotated route and draw the route maps.

    python navbench/generate_routes.py                      # every scene with annotations
    python navbench/generate_routes.py scene_03             # one scene, by id or name
    python navbench/generate_routes.py --routes navbench/routes/examples

Reads routes/<scene>.json (made with annotate.py) and maps/<scene>/ (made with render_topdown.py).
Writes, per scene, in routes/<scene>/:

  <id>.npy       waypoints of the route, N x 4: x, y, z, yaw (world meters, radians)
  routes.json    start pose, goal, path corners and facts of every route
  routes.txt     the same as a table, one line per route
  overview.png   all routes over the scene
  <id>.png       each route on its own, close up

and routes/all_routes.txt: the tables of all scenes in one. A route that cannot be planned (start
or goal blocked, goal unreachable) is reported, kept in routes.json with its reason, and has no
waypoints. docs/04_routes.md explains the planner.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from navbench import cli, draw
from navbench import routes as files
from navbench.maps import Map
from navbench.planner import ROBOTS, NoRoute, Planner

OVERVIEW = 2400  # px: longest side of overview.png
CLOSE_UP = 1600  # px: longest side of <id>.png
MARGIN = 8.0  # m of map around a route in its close-up


def plan_scene(folder: Path, routes: Path, robot, backdrop: str, images: bool) -> list:
    m = Map.load(folder)
    notes = files.load_annotations(routes, m.scene)
    if not notes["routes"]:
        return []
    planner = Planner(m, robot)
    planned = []
    for r in notes["routes"]:
        try:
            route = planner.route([r["start"], *r.get("via", []), r["goal"]])
            planned.append({"id": r["id"], "note": r.get("note", ""), "route": route})
        except NoRoute as e:
            planned.append({"id": r["id"], "note": r.get("note", ""), "error": str(e)})
    out = files.save_planned(routes, m.scene, planned, vars(robot), {"folder": folder.name, **vars(m.grid)})
    if images:
        draw_maps(out, m, planner, planned, backdrop)
    return planned


def draw_maps(out: Path, m: Map, planner: Planner, planned: list, backdrop: str) -> None:
    for old in out.glob("*.png"):
        old.unlink()
    base = draw.backdrop(np.asarray(Image.open(cli.MAPS / m.scene / f"{backdrop}.png").convert("RGB")), planner)
    overview = base.copy()
    scale = max(overview.size) / OVERVIEW  # lines stay visible once the image is shrunk
    ok = [p for p in planned if "route" in p]
    for i, p in enumerate(ok):
        r = p["route"]
        draw.route(overview, m.grid, r.path, r.points, draw.COLORS[i % len(draw.COLORS)], p["id"], max(1.0, scale))
    draw.shrink(overview, OVERVIEW).save(out / "overview.png")
    for i, p in enumerate(ok):
        r = p["route"]
        im = base.copy()
        draw.route(im, m.grid, r.path, r.points, draw.COLORS[i % len(draw.COLORS)], p["id"])
        draw.shrink(draw.crop(im, m.grid, r.path, MARGIN), CLOSE_UP).save(out / f"{p['id']}.png")


def write_table(routes: Path) -> Path:
    """routes/all_routes.txt: the routes.txt of every planned scene, with the scene in front."""
    lines = ["scene\t" + "\t".join(files.TXT_COLUMNS)]
    for table in sorted(routes.glob("*/routes.txt")):
        lines += [f"{table.parent.name}\t{line}" for line in table.read_text().splitlines()[1:]]
    path = routes / "all_routes.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", nargs="*", help="Scene id (scene_03) or name. Default: every scene with annotations.")
    p.add_argument("--routes", default=str(cli.ROUTES), help="Folder of the annotations (default navbench/routes).")
    p.add_argument("--robot", default="delivery", choices=sorted(ROBOTS), help="Robot to plan for (default delivery).")
    p.add_argument("--backdrop", default="cut", choices=["cut", "topdown"],
                   help="Image to draw on: cut (without roofs and tree crowns, default) or topdown.")
    p.add_argument("--no-images", action="store_true", help="Only the files, no route maps.")
    a = p.parse_args()
    routes = Path(a.routes).expanduser().resolve()
    folders = cli.find_maps(a.scene, cli.MAPS)
    if not a.scene:
        folders = [f for f in folders if files.annotation_path(routes, f.name).is_file()]
        if not folders:
            raise SystemExit(f"no annotations in {routes}: make some with annotate.py")

    failed = 0
    for folder in folders:
        t0 = time.perf_counter()
        planned = plan_scene(folder, routes, ROBOTS[a.robot], a.backdrop, not a.no_images)
        good = [q for q in planned if "route" in q]
        length = sum(q["route"].facts["length_m"] for q in good)
        print(f"[routes] {folder.name}: {len(good)}/{len(planned)} routes planned, {length:.0f} m in all, "
              f"{time.perf_counter() - t0:.1f}s -> {routes / folder.name}")
        for q in planned:
            if "error" in q:
                failed += 1
                print(f"[routes]   {q['id']}: {q['error']}")
    print(f"[routes] table of all routes: {write_table(routes)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
