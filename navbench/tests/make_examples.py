"""Example routes for every scene, to try the tools on: navbench/routes/examples/<scene>.json.

    python navbench/tests/make_examples.py [scene ...]
    python navbench/generate_routes.py --routes navbench/routes/examples

These are not benchmark routes, only test data: random starts and goals on the sidewalks (or
the ground, in a park), 25 to 70 m apart, picked so that each scene has one route along its
sidewalks, one across a street if the scene has one, and the tightest one that could be found;
all with a detour of at most 1.5 if any such route turned up.
Seeded: the same maps give the same routes.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navbench import cli  # noqa: E402
from navbench import routes as files  # noqa: E402
from navbench.maps import KINDS, Map  # noqa: E402
from navbench.planner import NoRoute, Planner  # noqa: E402

EXAMPLES = cli.ROUTES / "examples"
TRIES = 60
DETOUR = 1.5  # longest detour (path length / straight distance) an example should have, if it can


def examples(planner: Planner) -> list:
    rng = np.random.default_rng(0)
    on = np.isin(planner.kind, [KINDS.index("sidewalk"), KINDS.index("ground")])
    cells = np.argwhere(planner.main & on & (planner.clear > planner.robot.radius + 0.3))
    found = []
    for _ in range(TRIES):
        a, b = planner.world(*cells[rng.integers(len(cells), size=2)].T)
        if not 25.0 <= np.hypot(*(a - b)) <= 70.0:
            continue
        try:
            found.append((a.round(2).tolist(), b.round(2).tolist(), planner.route([a, b]).facts))
        except NoRoute:
            continue
    if not found:
        return []
    picks = []
    direct = [f for f in found if f[2]["detour"] <= DETOUR] or found  # not all the way around a block
    along = [f for f in direct if f[2]["road_crossings"] == 0]
    across = [f for f in direct if f[2]["road_crossings"] > 0]
    if along:
        picks.append((max(along, key=lambda f: f[2]["length_m"]), "along the sidewalks"))
    if across:
        picks.append((max(across, key=lambda f: f[2]["length_m"]), "across a street"))
    rest = [f for f in direct if all(f is not p for p, _ in picks)]
    if rest:
        picks.append((min(rest, key=lambda f: f[2]["min_clearance_m"]), "the tightest found"))
    return [{"id": f"r{i:03d}", "start": a, "via": [], "goal": b, "note": f"example: {why}"}
            for i, ((a, b, _), why) in enumerate(picks, 1)]


def main() -> None:
    for folder in cli.find_maps(sys.argv[1:], cli.MAPS):
        m = Map.load(folder)
        routes = examples(Planner(m))
        path = files.save_annotations(EXAMPLES, {"scene": m.scene, "routes": routes})
        print(f"{m.scene}: {len(routes)} example routes -> {path}")


if __name__ == "__main__":
    main()
