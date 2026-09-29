"""The maps against Isaac Sim: the ground height of the map where Isaac Sim 5.1 found the ground.

    python navbench/tests/test_maps.py [--records DIR] [scene ...]

sanity_check_sim.py casts 12 rays straight down in every scene and records where each one hit
(DIR/<scene>.json, "ground_hits": [x, y, z, prim, ...]). The map's ground in that cell has to be
within TOLERANCE of the hit. Two kinds of difference are expected and counted, not failed:

  a model's floor   the map's ground is a model's surface up to OBJECT_ABOVE over the hit (a
                    building's plinth): Isaac Sim's ray passes through models without colliders
  below the street  the ray hit something far below street level (the scene's infinite ground
                    plane, where the scene has no ground); the map has no ground there

Rays outside the map are counted too.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navbench import cli  # noqa: E402
from navbench.maps import KINDS, Map  # noqa: E402
from navbench.topdown import BAND, OBJECT_ABOVE  # noqa: E402

TOLERANCE = 0.03  # m
RECORDS = Path.home() / "urbanverse_sanity" / "sim_norender"


def check(folder: Path, record: Path) -> dict:
    """{"match": n, "model floor": n, "below the street": n, "outside": n, "off": [what is off]}."""
    m = Map.load(folder)
    level = m.info["street_level"]
    out = {"match": 0, "model floor": 0, "below the street": 0, "outside": 0, "off": []}
    for x, y, z, prim, *_ in json.loads(record.read_text())["ground_hits"]:
        col, row = m.grid.cell(x, y)
        if not m.grid.inside(col, row):
            out["outside"] += 1
            continue
        g = float(m.ground[row, col])
        if abs(g - z) <= TOLERANCE:
            out["match"] += 1
        elif TOLERANCE < g - z <= OBJECT_ABOVE + TOLERANCE:
            out["model floor"] += 1
        elif z < level - BAND and g != g:  # NaN: no ground in the map
            out["below the street"] += 1
        else:
            out["off"].append(f"({x:.1f}, {y:.1f}): Isaac Sim {z:.3f} on {prim.split('/')[-3]}, "
                              f"map {g:.3f} ({KINDS[m.kind[row, col]]})")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", nargs="*")
    p.add_argument("--records", default=str(RECORDS), help=f"sanity_check_sim.py's output (default {RECORDS}).")
    a = p.parse_args()
    records, failed, total = Path(a.records).expanduser(), 0, 0
    for folder in cli.find_maps(a.scene, cli.MAPS):
        record = records / f"{folder.name}.json"
        if not record.is_file():
            print(f"SKIP {folder.name}: no {record}")
            continue
        c = check(folder, record)
        counted = c["match"] + c["model floor"] + c["below the street"] + len(c["off"])
        total += counted
        failed += len(c["off"])
        extra = ", ".join(f"{c[k]} {k}" for k in ("model floor", "below the street", "outside") if c[k])
        print(f"{'FAIL' if c['off'] else 'PASS'} {folder.name}: {c['match']} of {counted} hits match"
              + (f" ({extra})" if extra else ""))
        for line in c["off"]:
            print(f"     {line}")
    print(f"{total - failed} of {total} ground hits match or are explained; {failed} off")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
