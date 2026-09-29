# Step 4: planning routes and drawing route maps

```bash
python navbench/generate_routes.py                 # every scene with annotations
python navbench/generate_routes.py scene_07        # one scene
python navbench/generate_routes.py --routes navbench/routes/examples
```

For every annotated route: plan it, write its waypoints and facts ([step 3](03_format.md)),
and draw it. It prints one line per scene and one per route that failed, and exits with 1 if
any did, so a script can tell.

![](route_example.png)

## The robot

The map says what is where; the robot says what of it is usable. Three are built in (`--robot`):

| Robot | Radius | Needs height | Climbs | For |
| --- | --- | --- | --- | --- |
| `delivery` (default) | 0.35 m | 1.0 m | 0.22 m | a six-wheeled sidewalk delivery robot, which climbs curbs |
| `quadruped` | 0.35 m | 0.5 m | 0.25 m | e.g. a Unitree Go2 |
| `wheelchair` | 0.45 m | 1.4 m | 0.05 m | needs curb ramps |

CraftBench's curbs are 4 to 21 cm high (road 0.90 m and sidewalk 1.105 m in scene_10), and its
crosswalks are at road level, so a robot has to climb curbs to cross a street at all. Others:
`navbench.planner.Robot(name, radius, height, step)`.

## Walkable

A map cell is walkable for the robot if

1. it has ground,
2. its clearance is at least the robot's height (nothing hangs lower: a table top, a car body,
   an awning at 0.8 m),
3. no neighbor's ground is more than the robot's step higher or lower (a wall's foot, a
   high curb, a planter's edge), and
4. no cell failing 1–3 is closer than the robot's radius (the robot is a disc).

Walkable cells that do not connect to the largest walkable area (inside hollow buildings,
fenced-off yards) are reported as sealed off: a start or goal there has no route.

## Cost

Every walkable meter costs, by the kind of ground:

| sidewalk | ground (park, plaza) | crosswalk | road, road marking |
| --- | --- | --- | --- |
| 1.0 | 1.2 | 1.5 | 6.0 |

A zebra crossing's stripes are crosswalk and the road between them road; gaps up to 1.2 m
between stripes count as crosswalk too, or routes would weave from stripe to stripe.

Plus up to 1.0 more within 0.5 m of the robot's radius from an obstacle (routes keep a little
away from walls and cars), plus the cost of 5 m of sidewalk for every curb climbed (a height
change of more than 3 cm between neighbors).

So a route stays on the sidewalk, crosses the street at a crosswalk if one is within a few times
the street's width, and otherwise crosses straight over. To make it cross elsewhere, give it a
via point.

## Search

The map is 5 cm per cell; the search runs on 10 cm cells, each as dear as its dearest map cell
and blocked if any of them is (so a gap narrower than the robot stays shut). It is a shortest
path over 8-connected cells (scikit-image's `MCP_Geometric`, Dijkstra in C), first within 25 m
around the points and then over the whole map if the path needs more. A route with via points
is planned leg by leg.

The path is then **straightened**: a stretch of the 8-connected chain is replaced by a straight
line wherever that line crosses only walkable cells and costs no more (so it does not cut across
a road to save a few meters). Last, it is **resampled** to a waypoint every 0.5 m, with the
ground height as z and the heading to the next waypoint as yaw.

A start or goal on blocked ground (inside the red band around a wall, say) moves to the nearest
walkable cell within 1 m; farther than that, the route fails with a message saying which point
and why.

The cost map of a scene takes about 3 s to make for the largest maps (scene_09, scene_10; 4 M
search cells, ~1.3 GB of memory at the peak). A route then plans in 0.01 to 0.1 s within the
25 m window, and in at most 0.75 s over the whole of the largest maps (20 random long routes
each), which is what makes the live preview of [step 2](02_annotate.md) possible.

The picture above (scene_10, example r002) is a case in point: the crosswalk is blocked, a
police car on one end and a bus over the other, so the route crosses mid-block between parked
cars, climbing down one curb and up the other.

## Facts

For picking, grouping and reporting routes (`routes.json` → `facts`):

| Fact | Meaning |
| --- | --- |
| `length_m` | length of the path |
| `straight_m`, `detour` | start-to-goal distance, and length / that |
| `by_kind_m` | meters on each kind of ground |
| `road_crossings` | stretches of 1 m or more on road, road marking or crosswalk |
| `curbs` | curbs climbed up or down |
| `min_clearance_m` | least distance from the path to an obstacle (at 10 cm resolution) |
| `turns` | corners of the path sharper than 30° |
| `climb_m` | highest minus lowest ground along it |

## Route maps

`overview.png` shows all of a scene's routes over `cut.png` (`--backdrop topdown` for the view
with roofs), each in its own color, labeled with its id; start green, goal red, via points
yellow. Red shade: blocked for the robot; orange: sealed off. `<id>.png` shows one route with 8 m
around it at full resolution.

## Tested

`tests/test_planner.py` builds a 30 × 20 m map by hand (sidewalks at 15 cm on both sides of a
road, a crosswalk, a wall with a 1.5 m gap) and checks that a route goes through the gap and not
through the wall, crosses at the crosswalk rather than straight over the road, crosses straight
when a via point says so, fails for a robot that cannot climb the curb, and that a start inside
the wall is moved off it while one outside the map is refused.
