# Step 3: the route files

Two kinds of files: what you annotated (the source of truth, written by `annotate.py`), and what
`generate_routes.py` planned from it (rebuilt from the annotations and the maps at any time).

```
navbench/routes/
├── <scene>.json          annotations: start, via points, goal of every route
├── <scene>/
│   ├── routes.json       planned: start pose, goal, path, facts of every route
│   ├── routes.txt        the same as a table
│   ├── r001.npy          waypoints of route r001: N x 4
│   ├── overview.png      all routes over the scene
│   └── r001.png          route r001, close up
├── all_routes.txt        routes.txt of all scenes in one table
└── examples/             my test routes, same layout; not yours (only their .json in git)
```

All positions are **world meters in the scene's own frame**, Z up: the frame Isaac Sim loads
`export_version.usd` in. Yaw is in radians, counter-clockwise from +X. Nothing is in pixels, so
the files stay valid if the maps are rebuilt at another resolution.

## Annotations: `routes/<scene>.json`

```json
{
 "format": 1,
 "scene": "scene_10_cbd_cross_intersection_diverse_obstacles",
 "saved": "2026-09-28T16:43:12",
 "routes": [
  {"id": "r001", "start": [-580.09, 510.11], "via": [], "goal": [-550.19, 553.11], "note": "example: along the sidewalks"},
  {"id": "r002", "start": [-610.19, 442.81], "via": [], "goal": [-657.89, 478.51], "note": "example: across a street"}
 ]
}
```

| Key | |
| --- | --- |
| `format` | 1. Raised if the files change in a way old readers cannot follow; readers refuse newer formats. |
| `id` | `r001`, `r002`, ... unique within the scene |
| `start`, `goal` | `[x, y]`, where you clicked (mm precision) |
| `via` | `[[x, y], ...]` points the route has to pass, in order; usually empty |
| `note` | free text |

Heights are not stored: they come from the map, so a start is always on the ground.

## Planned routes: `routes/<scene>/`

**`<id>.npy`**: the waypoints, a float64 array of N rows `x, y, z, yaw`, 0.5 m apart along the
path. `z` is the ground height there; `yaw` points to the next waypoint (the last repeats the one
before). Row 0 is the start pose, row N-1 the goal.

**`routes.json`**:

```json
{
 "format": 1, "scene": "scene_10_cbd_cross_intersection_diverse_obstacles",
 "frame": "scene world, meters, Z up; yaw in radians from +X",
 "robot": {"name": "delivery", "radius": 0.35, "height": 1.0, "step": 0.22},
 "map": {"folder": "scene_10_cbd_cross_intersection_diverse_obstacles", "x_min": -739.440369, "y_max": 596.46344,
         "res": 0.05, "width": 3960, "height": 3890},
 "routes": [
  {"id": "r002", "note": "example: across a street",
   "start": {"x": -610.1904, "y": 442.8134, "z": 1.105, "yaw": -1.9513},
   "goal": {"x": -657.8904, "y": 478.5134, "z": 1.105},
   "points": [[-610.19, 442.813], [-657.89, 478.513]],
   "path": [[-610.19, 442.813], [-610.79, 441.313], ...],
   "waypoints": "r002.npy",
   "facts": {"length_m": 74.8, "straight_m": 59.58, "detour": 1.256,
             "by_kind_m": {"road": 11.59, "road_marking": 0.7, "sidewalk": 62.56},
             "road_crossings": 1, "curbs": 2, "min_clearance_m": 0.35, "turns": 10, "climb_m": 0.21}},
  {"id": "r004", "note": "", "error": "the goal cannot be reached: no walkable way for the delivery robot"}
 ]
}
```

(`r004` is made up, to show a failed route.)

`points` are start, via points and goal after snapping: a point clicked on blocked ground is
moved to the nearest walkable cell within 1 m. `path` holds the corners of the planned path;
the waypoints are resampled from it. A route that could not be planned keeps its `error` and has
no other keys and no `.npy`. The facts are explained in [step 4](04_routes.md).

**`routes.txt`**: one tab-separated line per planned route, with a header:

```
id    start_x   start_y  start_z start_yaw goal_x    goal_y   goal_z length_m straight_m road_crossings curbs min_clearance_m waypoints note
r001  -580.090  510.113  1.105   0.015     -550.190  553.113  1.105  76.16    52.37      0              0     0.43            154       example: along the sidewalks
r002  -610.190  442.813  1.105   -1.951    -657.890  478.513  1.105  74.8     59.58      1              2     0.35            151       example: across a street
```

`all_routes.txt` has the same columns for all scenes, with `scene` in front.

## Reading them

```python
import sys; sys.path.insert(0, "navbench")
from navbench.routes import load_routes

for r in load_routes("scene_07"):                  # id or full name
    x, y, z, yaw = r["start"]["x"], r["start"]["y"], r["start"]["z"], r["start"]["yaw"]
    goal = (r["goal"]["x"], r["goal"]["y"], r["goal"]["z"])
    waypoints = r["waypoints"]                     # N x 4 numpy array: x, y, z, yaw
    print(r["id"], r["facts"]["length_m"], len(waypoints))
```

`load_routes(scene, folder)` reads another folder, e.g. `navbench/routes/examples`. It leaves out
routes that could not be planned. Without Python: `routes.txt` and the `.npy` files
(`numpy.load`) are all a reader needs.

To spawn a robot for a route, put it at `start` (x, y, and z plus the robot's own base height),
turned to `yaw`. The route is done when the robot is within the success radius of `goal`; the
waypoints are the reference path for route-following metrics.

## Tested

`tests/test_planner.py` (`test_files`) writes annotations, reads them back, plans them, writes the
planned files and loads them with `load_routes`: the waypoints read back are the ones planned,
the failed route is left out, and the table has one line per planned route.
