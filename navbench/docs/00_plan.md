# CraftBench-Nav: plan

A sidewalk-robot navigation benchmark on the 12 CraftBench scenes. A scene alone is
not a benchmark: it also needs tasks. Here a task is a **route**: a scene, a start
point, a goal point, and the reference path between them along walkable ground.

This file is the plan, written before the work and brought up to date after it ([status](#status)). Each step has its own page: [01 top-down maps](01_topdown.md),
[02 annotation](02_annotate.md), [03 files](03_format.md), [04 routes](04_routes.md).

## Your proposal

1. A script renders the top-down view of each scene.
2. You annotate each scene with start and end points, saved automatically.
3. Files (txt, npy, ...) hold and describe these.
4. A route generation algorithm renders the route map from them.

## The proposal, refined

| Step | You asked for | What gets built | Why the change |
| --- | --- | --- | --- |
| 1 | a top-down render | a **metric map** per scene: an orthographic top-down image where every pixel has a known world position, plus the layers a planner needs (ground height, surface class, obstacles, walkable mask) | A picture alone cannot turn a click into a world position or tell sidewalk from road. The `preview_topdown.png` the scenes ship with is a perspective shot with no calibration, and its buildings lean over the sidewalks. |
| 2 | click start and end, auto-save | a page in your browser, served by one Python script: click start, click goal, see the planned route at once, saved after every change | Seeing the route while annotating catches a bad pair (blocked, unreachable, a detour) on the spot instead of at step 4. |
| 3 | txt / npy files | one JSON per scene as the source of truth, and exports for code: `.npy` waypoints, a `.txt` table of all routes | JSON is readable and diffable; npy and txt are what evaluation code loads. All positions are world meters in the scene's own frame, so they stay valid if a map is re-rendered at another resolution. |
| 4 | route generation, route map | a shortest path over a cost map (prefer sidewalk, cross roads at crosswalks, keep clear of obstacles, never over a step the robot cannot climb), straightened and resampled to waypoints; drawn over the top-down image | Same algorithm as the live preview of step 2, so what you saw while annotating is what gets saved. |

Added, because a benchmark needs them:

- **Checks on every route**: start and goal on walkable ground (or moved onto it, up to 1 m), goal reachable; a route that fails says why.
- **Route facts** for picking and grouping routes: length, road crossings, narrowest clearance, turns.
- **A loader** (`navbench.load_routes`) so evaluation code gets start pose, goal and waypoints in one call.

## Decisions

**Maps come from the scene geometry, in plain Python; Isaac Sim is not needed.**
The scene's USD layers are read with `usd-core`, its `.glb` models directly, and
everything is rasterized from above with numpy. Reasons:

- The Isaac Sim 5.1 environment (`dev`) is gone and the GPU is running your GS job,
  which I will not disturb. Nothing here needs either.
- Anyone can rebuild the maps on a laptop, and the result is the same on every machine.
- Geometry gives what a render cannot: the ground height under a roof or a tree, and
  which surface is sidewalk.

The cost: ground surfaces get flat colors by kind, not their MDL textures (those come
from a library inside Isaac Sim). Models keep their own textures. An RTX top-down
image in the same map frame can be added later as another backdrop; nothing depends on it.

**What usd-core does not do, the reader does.** usd-core has no glTF plugin, so the
`.glb` payloads stay empty. The reader parses each `.glb` itself and places its nodes
the way Isaac Sim 5 does: under the prim the scene put it on, with the transform the
scene authored for that node. 97% of the 548 models are a single node; none use
compressed geometry.

**Checked against Isaac Sim, from the runs already on disk.** No new Isaac Sim run is
needed for this:

- the 12 images `sanity_check_render.py` made: the reconstructed scene, drawn through
  the same camera, has to line up with them;
- the world bounding boxes Isaac Sim 5.1 reported for benches and buildings;
- the ground hits of `sanity_check_sim.py` (12 per scene): the map's ground height
  has to match them.

**Walkable is decided by geometry, not by names.** A cell is walkable if it has a
ground surface, nothing stands in the robot's height range above it, and its
neighbors are within a step the robot can climb. Names only say what *kind* of ground
it is (sidewalk, road, crosswalk), which sets the cost of walking there. The robot
(radius, height, step) is a parameter, with a sidewalk delivery robot as the default.

**Annotation runs in the browser.** One script, no extra packages, works over SSH
with a forwarded port. OpenCV and matplotlib windows depend on how those packages
were built.

## Layout

```
navbench/
├── README.md               quick start
├── docs/                   this plan and one page per step
├── render_topdown.py       step 1: scene -> maps/<scene>/
├── annotate.py             step 2: the annotation page and its server
├── generate_routes.py      step 4: routes/<scene>.json -> waypoints and route maps
├── navbench/               shared code: scene reader, rasterizer, map, planner, files, the annotation page
├── tests/                  one test per part, and the example-route maker
├── maps/<scene>/           built by step 1 (not in git: large, and rebuilt by one command)
└── routes/                 annotations and generated routes (in git)
```

## Steps

| # | Step | Done when |
| --- | --- | --- |
| 0 | This plan | saved |
| 1 | Scene reader and rasterizer; `render_topdown.py`; checks against Isaac Sim | maps of all 12 scenes built; the three checks above pass |
| 2 | `annotate.py` | a route can be added, moved, deleted in the browser; every change is on disk; tested in headless Chrome |
| 3 | File format, exports, loader | files written and read back identically; format page written |
| 4 | Planner and route maps; `generate_routes.py` | example routes in every scene planned and drawn; blocked and unreachable pairs reported |
| 5 | Quick start, a test for each part, clean-up | one command per step, from a fresh clone |

## Your part

Step 2 is yours: the start and goal of each route. I test everything with example
routes of my own, kept apart under `routes/examples/` so they are never mistaken for
yours.

## Risks

| Risk | Handling |
| --- | --- |
| A model placed differently than in Isaac Sim | the three checks above; any model that fails them is listed, not silently drawn |
| 140 M triangles in total (28 M in scene_09) | sampled rasterization in chunks; maps are built once and cached |
| Ground under something that is not named as ground (a plaza floor that is part of a model) | walkable is decided by geometry; per scene, the map page shows what was taken as ground |
| Two levels above each other (a bridge, a roofed passage) | one ground level per cell: the one open to the robot. Reported if a scene has more |

## Not in this plan

- Running a policy in Isaac Sim and scoring it (success rate, route completion, collisions). The route files are what that code will read.
- Pedestrians and moving vehicles.
- The 375 training scenes. The tools take any scene directory, but only CraftBench is checked.

## Status

All steps are done. What changed from the plan while building:

- **Dijkstra, not A\***: scikit-image's `MCP_Geometric` (C code, little memory) searches a
  window around the points first, then the whole map if needed; fast enough for the live preview.
- **Model floors**: a model's flat surface counts as ground only up to 12 cm over the road
  network under it. Found by the Isaac Sim check: car hoods and bench seats had counted as ground.
- **Instanced prims**: scene_10's crossing is an instanceable road block that a plain USD
  traversal skips; found by the same check.
- **Memory**: map building runs as many workers as free memory holds (~4 GB each), after the
  first all-cores run ran the machine out of memory.
- **Robots**: three presets (delivery, quadruped, wheelchair); curbs in CraftBench are up to
  21 cm, so the default robot climbs 22 cm.
