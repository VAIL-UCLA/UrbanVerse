"""The planner and the route files, on a small made-up map: no scene needed.

    python navbench/tests/test_planner.py

The map, 30 x 20 m at 5 cm: sidewalk at 0.15 m on both sides of a road at 0 m, a zebra
crossing across the road, and a wall across the lower sidewalk with a 1.5 m gap in it.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navbench import routes as files  # noqa: E402
from navbench.maps import KINDS, NO_CLEARANCE_LIMIT, Grid, Map  # noqa: E402
from navbench.planner import NoRoute, Planner, Robot  # noqa: E402

RES = 0.05


def made_up_map() -> Map:
    grid = Grid.around(0.0, 0.0, 30.0, 20.0, RES)
    shape = (grid.height, grid.width)
    x, y = grid.world(*np.meshgrid(np.arange(grid.width), np.arange(grid.height)))
    road = (y > 8) & (y < 14)
    ground = np.where(road, 0.0, 0.15).astype(np.float32)
    kind = np.where(road, KINDS.index("road"), KINDS.index("sidewalk")).astype(np.uint8)
    stripe = road & (x > 20) & (x < 23) & ((y * 2).astype(int) % 2 == 0)  # a zebra: 0.5 m stripes, 0.5 m gaps
    kind[stripe] = KINDS.index("crosswalk")
    clearance = np.full(shape, NO_CLEARANCE_LIMIT, np.float32)
    wall = (x > 10) & (x < 10.3) & (y < 8) & ~((y > 3) & (y < 4.5))  # the gap: 3 m to 4.5 m
    clearance[wall] = 0.0
    top = np.where(wall, 2.5, ground).astype(np.float32)
    return Map("made_up", grid, ground, clearance, top, kind, {})


def test_planner() -> None:
    m = made_up_map()
    p = Planner(m, Robot(radius=0.35, height=1.0, step=0.22))

    # Along the lower sidewalk, through the gap in the wall.
    r = p.route([(2.0, 2.0), (18.0, 2.0)])
    assert r.facts["by_kind_m"].get("road", 0) == 0, r.facts
    ys = np.interp(10.15, r.path[:, 0], r.path[:, 1])
    assert 3.0 + 0.35 <= ys <= 4.5 - 0.35, f"passes the wall at y = {ys:.2f}, not through the gap"
    assert np.allclose(r.waypoints[:, 2], 0.15), "waypoints are at the sidewalk's height"
    assert np.all(np.hypot(*np.diff(r.waypoints[:, :2], axis=0).T) <= 0.5 + 1e-6)

    # Across the street: over the crosswalk, not straight over the road (road costs 6, crosswalk 1.5).
    r = p.route([(15.0, 5.0), (15.0, 17.0)])
    assert r.facts["road_crossings"] == 1 and "crosswalk" in r.facts["by_kind_m"], r.facts
    assert r.facts["by_kind_m"].get("road", 0) < 1.0, r.facts
    assert r.facts["curbs"] == 2, r.facts
    assert r.facts["turns"] <= 4, f"weaves over the zebra's stripes: {r.facts}"

    # A via point forces the way.
    r = p.route([(15.0, 5.0), (15.0, 11.0), (15.0, 17.0)])
    assert r.facts["by_kind_m"]["road"] > 4.0, r.facts

    # A robot that cannot climb the curb cannot cross at all.
    low = Planner(m, Robot(radius=0.35, height=1.0, step=0.05))
    try:
        low.route([(15.0, 5.0), (15.0, 17.0)])
        raise AssertionError("a 5 cm step robot crossed a 15 cm curb")
    except NoRoute as e:
        assert "cannot be reached" in str(e)

    # A start inside the wall moves off it; one far from anything walkable is refused.
    r = p.route([(10.15, 6.0), (2.0, 6.0)])
    assert abs(r.points[0][0] - 10.15) <= 1.0
    for bad in ([(-5.0, 2.0), (2.0, 2.0)], [(2.0, 2.0)]):
        try:
            p.route(bad)
            raise AssertionError(f"planned {bad}")
        except NoRoute:
            pass
    print("PASS: planner")


def test_files() -> None:
    m = made_up_map()
    p = Planner(m)
    with tempfile.TemporaryDirectory() as tmp:
        notes = {"scene": m.scene, "routes": [
            {"id": "r001", "start": [2.0, 2.0], "goal": [18.0, 2.0], "note": "through the gap"},
            {"id": "r002", "start": [15.0, 5.0], "via": [[15.0, 11.0]], "goal": [15.0, 17.0]},
            {"id": "r003", "start": [-5.0, 2.0], "goal": [2.0, 2.0]}]}
        files.save_annotations(tmp, notes)
        back = files.load_annotations(tmp, m.scene)
        assert [r["id"] for r in back["routes"]] == ["r001", "r002", "r003"]
        assert back["routes"][1]["via"] == [[15.0, 11.0]] and back["routes"][0]["via"] == []
        assert files.next_id(back["routes"]) == "r004"

        planned = []
        for r in back["routes"]:
            try:
                planned.append({"id": r["id"], "note": r["note"],
                                "route": p.route([r["start"], *r["via"], r["goal"]])})
            except NoRoute as e:
                planned.append({"id": r["id"], "note": r["note"], "error": str(e)})
        files.save_planned(tmp, m.scene, planned, {"name": "delivery"}, {})
        loaded = files.load_routes(m.scene, tmp)
        assert [r["id"] for r in loaded] == ["r001", "r002"], "the route that failed is left out"
        for r, q in zip(loaded, planned[:2], strict=True):
            w = r["waypoints"]
            assert w.shape[1] == 4 and np.allclose(w[:, :3], q["route"].waypoints)
            assert np.isclose(r["start"]["x"], w[0, 0], atol=1e-4) and np.isclose(r["goal"]["y"], w[-1, 1], atol=1e-4)
        table = (Path(tmp) / m.scene / "routes.txt").read_text().splitlines()
        assert table[0].split("\t") == files.TXT_COLUMNS and len(table) == 3
    print("PASS: route files")


if __name__ == "__main__":
    test_planner()
    test_files()
