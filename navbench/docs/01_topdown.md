# Step 1: top-down maps

```bash
python navbench/render_topdown.py              # all 12 scenes, ~10 min
python navbench/render_topdown.py scene_07     # one scene
```

Each scene gets a folder `navbench/maps/<scene>/` (not in git; rebuilt by the command above):

| File | What it is |
| --- | --- |
| `topdown.png` | the scene from straight above, 1 pixel = 1 cell |
| `cut.png` | the same without what is higher than street level + 2.4 m: roofs, awnings and tree crowns are gone, walls show as lines. The route maps are drawn on this one. |
| `layers.npz` | four arrays, one value per cell (below) |
| `map.json` | where the map is in the world, and how it was made |

## The map frame

A cell is 5 cm (`--res`). `map.json` has `grid: {x_min, y_max, res, width, height}`:
column 0 is at `x_min`, row 0 at `y_max`, so +X is right and +Y is up in the images.
The center of cell (col, row) is

```
x = x_min + (col + 0.5) * res
y = y_max - (row + 0.5) * res
```

x and y are world meters in the scene's own frame (Z up), the frame Isaac Sim loads the scene
in. Routes are stored in these world coordinates, not in pixels, so they stay valid if a map
is rebuilt at another resolution or extent. `navbench.maps.Grid` does the conversion.

The map covers the scene's preview camera path (`cam0_to_world.txt`) plus 40 m to each side
(`--margin`), cut down to where the scene has anything. `--bounds x0,y0,x1,y1` sets it by hand.

## The layers

| Layer | Type | Meaning |
| --- | --- | --- |
| `ground` | float32, m | height of the ground: the highest flat surface (≤ 35° slope) within 0.6 m of street level. NaN where there is none. |
| `kind` | uint8 | what that ground is, an index into `map.json`'s `kinds`: `none, road, road_marking, crosswalk, sidewalk, ground` |
| `clearance` | float32, m | free height above the ground, up to the first thing above it; 9.99 if nothing is within 6.4 m |
| `top` | float32, m | height of the highest thing in the cell, what one sees from above |

None of this depends on the robot. What a robot can do with a cell (its size, the step it can
climb) is the planner's business ([step 4](04_routes.md)).

**Street level** is found per scene: the height where most of the flat road network lies
(1.10–1.26 m in CraftBench). `--level` overrides it.

**Kind** comes from the names of the road network's meshes: `Lane` → road, `WhiteLine` and
`YellowLine` → road_marking, `crosswalk` → crosswalk, `Sidewalk`, `NearRoad`, `NearBuffer` →
sidewalk, any other piece of the road network → ground. Where there is no road network, but a model's floor (a park's lawn, a plaza's paving),
the kind is `ground`.

**A model's surface is ground only if it is low**: at most 12 cm over the road network under
it (or over street level where there is none). A speed bump or a plaza's paving is ground; a
car's hood, a bench's seat or a planter's rim is not, and the ground under it stays the road,
with the car above it as something that leaves no clearance.

**Hollow buildings.** Most buildings are shells with no floor of their own; the street's ground
runs through them. Their walls block them off, so the planner sees their inside as a sealed-off
area (orange on the route maps) that no route can reach.

## How the maps are made

Isaac Sim is not needed, nor a GPU. The scene is read with `usd-core` and drawn from above with
numpy:

1. **The scene's own meshes** (roads, sidewalks, and copies of model meshes the scene keeps in
   its layers) are read with usd-core, including those inside instanced prims (scene_10's
   crossing is one).
2. **The `.glb` models.** usd-core has no glTF plugin, so the payloads come up empty.
   `navbench/glb.py` reads each `.glb` and `navbench/scene.py` places its nodes the way Isaac
   Sim 5 does (below).
3. **Drawing by sampling.** Every triangle gets points no further apart than half a cell, and
   in every cell the highest point wins (`navbench/raster.py`). Along the way every point marks
   its 5 cm level in a column of 128 levels per cell; `clearance` is read from those columns.
4. **In tiles, in parallel.** The map is cut into 256 × 256 cell tiles, drawn by as many worker
   processes as the machine has cores **and free memory** (each worker loads the whole scene and
   takes up to ~4 GB; `--workers` sets the count). The result does not depend on the tiling.

A scene takes 20 s to 5 min with 3 workers (all 12: ~25 min); scene_09 (28 M triangles) is the
largest.

### How Isaac Sim 5 places a `.glb`, and how that was found

Every model is a payload on a prim of the scene, and the scene overrides some of the model's
prims (to move or hide them). To draw the scene as Isaac Sim does, the reader has to give every
glb node the prim path Isaac Sim gives it, or those overrides land nowhere. From the overrides in
the 12 scenes and Isaac Sim 5.1's own output:

- the payload prim is the glb's root node; every other node is a child prim named after the
  node, nested as in the glb;
- a node's mesh is a child prim of the node, named after the mesh; a skinned mesh hangs under
  the skeleton's root joint instead;
- names are made valid per character (anything else becomes `_`); a name that starts with a
  digit gets `node_` (nodes) or `mesh_` (meshes) in front, e.g. mesh `1_Mat.1_0` →
  `mesh__Mat_1_0`; siblings with the same name get a number;
- a prim's transform is the scene's `xformOp`s if the scene authors them, else the node's own
  TRS or matrix; the glb's points are used as they are.

Checks: every scene override of a model prim lands on a prim the reader made (0 unmatched in all
12 scenes); a bench in scene_09 gets exactly the world box Isaac Sim 5.1 reported,
(-636.60, 497.45, 1.10)..(-634.81, 498.02, 1.95); and scene_07 drawn through the camera of
`sanity_check_render.py` lines up with Isaac Sim's RTX image.

The 548 models are 97% single-node, none use Draco compression, one (the pigeon of scene_12) is
skinned and is drawn in its bind pose.

## Checked against Isaac Sim

`sanity_check_sim.py` cast 12 rays straight down in each scene in Isaac Sim 5.1 and recorded
where they hit the ground. `tests/test_maps.py` compares the map's ground height in those cells:

```bash
python navbench/tests/test_maps.py        # reads ~/urbanverse_sanity/sim_norender/<scene>.json
```

| Scene | Hits | Match (±3 cm) | Explained |
| --- | --- | --- | --- |
| 01 – 05 | 11 each | all | 1 each outside the map |
| 06, 10, 11 | 12 each | all | |
| 07, 08 | 12 each | 10 | 2 on a model's floor |
| 09 | 12 | 11 | 1 below the street |
| 12 | 12 | 9 | 3 on a model's floor |

139 hits in the maps: 131 match within 3 cm and 8 are explained. None is off. The two
explanations, each checked by hand:

- **On a model's floor** (7): the map's ground is 5–11 cm above the hit, on a building's plinth
  or floor (a skyscraper's base in scene_08, a cathedral's floor in scene_07 and scene_12). Isaac
  Sim's ray passed through it to the sidewalk underneath, because that model has no collider
  there. The map is right to show the raised floor: that is what a robot's cameras see and
  what it drives up onto. It also means those floors are not solid in Isaac Sim; worth
  knowing when a robot is spawned on one.
- **Below the street** (1): in scene_09 one ray hit the infinite collision plane at −0.30 m,
  1.4 m below the street, where the scene has no ground of its own. The map leaves that cell
  empty.

Two bugs this found, both fixed: the instanced crossing of scene_10 was missing (a plain USD
traversal skips instance proxies), and car hoods and bench seats counted as ground (Isaac Sim's
rays pass through them: they have no colliders).

## What to know

- Road surfaces get flat colors by kind, not their MDL textures (those live inside Isaac Sim).
  Models keep their own textures.
- One ground level per cell. None of the 12 scenes has a bridge or a walkable second level.
- scene_03 has road blocks that the scene switches off; they show as black gaps (no ground).
- The top-down images are not photos: no shadows or lighting beyond a fixed light from above.
