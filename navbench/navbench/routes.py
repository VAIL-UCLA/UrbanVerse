"""The route files, and the loader evaluation code uses. docs/03_format.md describes the files.

  routes/<scene>.json            what was annotated: start, via points, goal of every route
  routes/<scene>/routes.json     what generate_routes.py planned from it: poses, path, facts
  routes/<scene>/<id>.npy        the waypoints of one route: N x 4 float64, x y z yaw
  routes/<scene>/routes.txt      one line per route, for reading and for tools that want text
  routes/<scene>/*.png           the route maps

Positions are world meters in the scene's own frame (Z up), the frame Isaac Sim loads it in.
"""
import json
import time
from pathlib import Path

import numpy as np

FORMAT = 1  # version of the files; raised if they change in a way old readers cannot follow
TXT_COLUMNS = ["id", "start_x", "start_y", "start_z", "start_yaw", "goal_x", "goal_y", "goal_z", "length_m",
               "straight_m", "road_crossings", "curbs", "min_clearance_m", "waypoints", "note"]


# ── annotations: routes/<scene>.json ─────────────────────────────────────────────────────────

def annotation_path(folder, scene: str) -> Path:
    return Path(folder) / f"{scene}.json"


def load_annotations(folder, scene: str) -> dict:
    """The annotations of a scene; an empty set if there are none yet."""
    path = annotation_path(folder, scene)
    if not path.is_file():
        return {"format": FORMAT, "scene": scene, "routes": []}
    data = json.loads(path.read_text())
    if data.get("format", FORMAT) > FORMAT:
        raise SystemExit(f"{path}: format {data['format']} is newer than this code reads ({FORMAT})")
    return data


def save_annotations(folder, data: dict) -> Path:
    """Write the annotations, through a temporary file: a crash never leaves half a file."""
    path = annotation_path(folder, data["scene"])
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"format": FORMAT, "scene": data["scene"], "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "routes": [clean_route(r) for r in data["routes"]]}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    tmp.replace(path)
    return path


def clean_route(r: dict) -> dict:
    """An annotated route with only the keys the format has, numbers rounded to the millimeter."""
    def xy(p):
        return [round(float(p[0]), 3), round(float(p[1]), 3)]
    return {"id": str(r["id"]), "start": xy(r["start"]), "via": [xy(p) for p in r.get("via", [])],
            "goal": xy(r["goal"]), "note": str(r.get("note", ""))}


def next_id(routes: list) -> str:
    """r001, r002, ...: one more than the highest number used so far."""
    numbers = [int(r["id"][1:]) for r in routes if r["id"][:1] == "r" and r["id"][1:].isdigit()]
    return f"r{max(numbers, default=0) + 1:03d}"


# ── planned routes: routes/<scene>/ ──────────────────────────────────────────────────────────

def save_planned(folder, scene: str, planned: list, robot: dict, map_info: dict) -> Path:
    """`planned` = [{"id", "note", "route": planner.Route}] or [{"id", "note", "error"}] for those that failed."""
    out = Path(folder) / scene
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.npy"):
        old.unlink()  # routes deleted from the annotations go too
    entries, lines = [], ["\t".join(TXT_COLUMNS)]
    for p in planned:
        if "error" in p:
            entries.append({"id": p["id"], "note": p["note"], "error": p["error"]})
            continue
        r = p["route"]
        waypoints = np.column_stack([r.waypoints, r.yaw])
        np.save(out / f"{p['id']}.npy", waypoints)
        start, goal = waypoints[0], waypoints[-1]
        entries.append({
            "id": p["id"], "note": p["note"],
            "start": {"x": start[0], "y": start[1], "z": start[2], "yaw": start[3]},
            "goal": {"x": goal[0], "y": goal[1], "z": goal[2]},
            "points": r.points,
            "path": r.path.round(3).tolist(),
            "waypoints": f"{p['id']}.npy",
            "facts": {k: v for k, v in r.facts.items() if k != "robot"},
        })
        f = r.facts
        lines.append("\t".join(str(v) for v in [
            p["id"], *(f"{v:.3f}" for v in start), *(f"{v:.3f}" for v in goal[:3]), f["length_m"],
            f["straight_m"], f["road_crossings"], f["curbs"], f["min_clearance_m"], len(waypoints),
            p["note"].replace("\t", " ").replace("\n", " ")]))
    data = {"format": FORMAT, "scene": scene, "frame": "scene world, meters, Z up; yaw in radians from +X",
            "robot": robot, "map": map_info, "routes": _rounded(entries)}
    (out / "routes.json").write_text(json.dumps(data, indent=1) + "\n")
    (out / "routes.txt").write_text("\n".join(lines) + "\n")
    return out


def _rounded(o):
    if isinstance(o, dict):
        return {k: _rounded(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_rounded(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return round(float(o), 4)
    return o


def load_routes(scene: str, folder=None) -> list:
    """The planned routes of a scene, for evaluation code:

        [{"id", "note", "start": {"x", "y", "z", "yaw"}, "goal": {"x", "y", "z"},
          "waypoints": N x 4 array (x, y, z, yaw), "facts": {...}}, ...]

    `scene` is a scene's full name or its id (scene_03). `folder` is where routes/ is (default:
    the one next to this package). Routes that could not be planned are left out."""
    from .cli import ROUTES

    base = Path(folder) if folder else ROUTES
    hits = [d for d in base.iterdir() if d.is_dir() and (d.name == scene or d.name.startswith(scene + "_"))] \
        if base.is_dir() else []
    if len(hits) != 1:
        raise FileNotFoundError(f"{scene!r} matches {len(hits)} planned scenes in {base}: run generate_routes.py")
    data = json.loads((hits[0] / "routes.json").read_text())
    routes = []
    for r in data["routes"]:
        if "error" not in r:
            routes.append({**r, "waypoints": np.load(hits[0] / r["waypoints"])})
    return routes
