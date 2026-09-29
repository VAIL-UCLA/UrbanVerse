"""annotate.py in a real browser: headless Chrome clicks, drags, deletes and undoes on the page
(navbench/web/selftest.js), and every change has to be on disk the way the page shows it.

    python navbench/tests/test_annotate.py [scene_01]

Uses its own temporary Chrome profile and routes folder; your annotations are not touched.
Needs the scene's map, and Chrome or Chromium.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from html import unescape
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import annotate  # noqa: E402
from navbench import cli  # noqa: E402
from navbench import routes as files  # noqa: E402
from navbench.maps import Map  # noqa: E402
from navbench.planner import Planner  # noqa: E402

CHROME = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]


def three_points(planner: Planner) -> list:
    """Three walkable points of the streets, 10 to 25 m apart."""
    rng = np.random.default_rng(7)
    cells = np.argwhere(planner.main & (planner.clear > planner.robot.radius + 0.5))
    while True:
        pts = planner.world(*cells[rng.integers(len(cells), size=3)].T)
        d = np.hypot(*(pts[[0, 0, 1]] - pts[[1, 2, 2]]).T)
        if (d > 10).all() and (d < 25).all():
            return [p.round(3).tolist() for p in pts]


def test_annotate(scene: str = "scene_01") -> None:
    chrome = next((c for c in CHROME if shutil.which(c)), None)
    if chrome is None:
        print("SKIP: no Chrome or Chromium")
        return
    folder = cli.find_maps([scene], cli.MAPS)[0]
    points = three_points(Planner(Map.load(folder)))
    with tempfile.TemporaryDirectory() as tmp:
        routes = Path(tmp) / "routes"
        annotate.Handler.scenes = annotate.Scenes(cli.MAPS, routes, annotate.ROBOTS["delivery"])
        server = ThreadingHTTPServer(("127.0.0.1", 0), annotate.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = (f"http://127.0.0.1:{server.server_port}/?selftest=1&scene={folder.name}&"
               + "&".join(f"{k}={x},{y}" for k, (x, y) in zip("abc", points, strict=True)))
        try:
            dom = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-first-run",
                                  f"--user-data-dir={tmp}/chrome", "--window-size=1400,900",
                                  "--virtual-time-budget=120000", "--dump-dom", url],
                                 capture_output=True, text=True, timeout=300).stdout
        finally:
            server.shutdown()
        found = re.search(r'<pre id="selftest">(.*?)</pre>', dom, re.S)
        assert found, "the page never finished its self-test:\n" + dom[-2000:]
        result = json.loads(unescape(found.group(1)))
        for step in result["steps"]:
            print("  " + step)
        assert result["ok"], result.get("error")
        saved = files.load_annotations(routes, folder.name)["routes"]
        assert saved == [files.clean_route(r) for r in result["routes"]], (saved, result["routes"])
        print(f"PASS: {len(result['steps'])} steps; on disk: {json.dumps(saved)}")
        print(f"      route facts: {result['facts']}")


if __name__ == "__main__":
    test_annotate(*sys.argv[1:])
