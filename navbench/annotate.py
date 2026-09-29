#!/usr/bin/env python3
"""Step 2: annotate routes in the browser. Every change is saved at once to routes/<scene>.json.

    python navbench/annotate.py                  # then open http://localhost:8765
    python navbench/annotate.py --port 9000 --robot quadruped

On a remote machine, forward the port first: ssh -L 8765:localhost:8765 <host>.

In the page: click the start, then the goal. The route is planned and drawn as soon as both are
there. Shift+click adds a via point to the selected route, dragging moves a point, Delete removes
the selected route, Ctrl+Z undoes. docs/02_annotate.md has the whole list.

Needs the maps of step 1 (render_topdown.py). Nothing but the Python standard library and the
navbench package; the page itself is navbench/web/annotate.html.
"""
import argparse
import io
import json
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

from navbench import cli, draw
from navbench import routes as files
from navbench.maps import Map
from navbench.planner import ROBOTS, NoRoute, Planner

PAGE = Path(__file__).resolve().parent / "navbench" / "web" / "annotate.html"
KEEP = 3  # planners kept in memory at once (one per scene, up to a few hundred MB each)


class Scenes:
    """The maps, their planners (made when first needed), and the annotation files."""

    def __init__(self, maps: Path, routes: Path, robot):
        self.maps, self.routes, self.robot = maps, routes, robot
        self.planners = OrderedDict()
        self.lock = threading.Lock()  # one planner is built, one file written, at a time

    def names(self) -> list:
        return [d.name for d in cli.find_maps([], self.maps)]

    def folder(self, name: str) -> Path:
        if name not in self.names():
            raise KeyError(name)
        return self.maps / name

    def planner(self, name: str) -> Planner:
        with self.lock:
            if name not in self.planners:
                self.planners[name] = Planner(Map.load(self.folder(name)), self.robot)
                while len(self.planners) > KEEP:
                    self.planners.popitem(last=False)
            self.planners.move_to_end(name)
            return self.planners[name]

    def info(self, name: str) -> dict:
        m = json.loads((self.folder(name) / "map.json").read_text())
        notes = files.load_annotations(self.routes, name)
        return {"name": name, "grid": m["grid"], "street_level": m["street_level"], "robot": vars(self.robot),
                "routes": notes["routes"], "file": str(files.annotation_path(self.routes, name))}

    def plan(self, name: str, points: list) -> dict:
        planner = self.planner(name)
        try:
            r = planner.route(points)
        except NoRoute as e:
            return {"error": str(e)}
        facts = {k: v for k, v in r.facts.items() if k != "robot"}
        return {"path": r.path.round(3).tolist(), "points": r.points, "facts": facts}

    def save(self, name: str, routes: list) -> dict:
        self.folder(name)
        with self.lock:
            path = files.save_annotations(self.routes, {"scene": name, "routes": routes})
        return {"file": str(path), "routes": len(routes)}

    def overlay(self, name: str) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(draw.overlay(self.planner(name))).save(buf, "PNG")
        return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    scenes: Scenes = None

    def log_message(self, fmt, *args):  # only the requests that change something
        if self.command == "POST" and "/save" in self.path:
            print(f"[annotate] {self.address_string()} {fmt % args}", flush=True)

    def reply(self, body, kind="application/json", status=200) -> None:
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.strip("/").split("/")]
        try:
            if url.path in ("/", "/index.html"):
                self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/selftest.js":  # used by tests/test_annotate.py
                self.reply((PAGE.parent / "selftest.js").read_bytes(), "text/javascript")
            elif parts == ["api", "scenes"]:
                self.reply([{"name": n, "routes": len(files.load_annotations(self.scenes.routes, n)["routes"])}
                            for n in self.scenes.names()])
            elif parts == ["api", "scene"]:
                self.reply(self.scenes.info(parse_qs(url.query)["name"][0]))
            elif len(parts) == 3 and parts[0] == "map" and parts[2] in ("cut.png", "topdown.png"):
                self.reply((self.scenes.folder(parts[1]) / parts[2]).read_bytes(), "image/png")
            elif len(parts) == 2 and parts[0] == "overlay" and parts[1].endswith(".png"):
                self.reply(self.scenes.overlay(parts[1][:-4]), "image/png")
            else:
                self.reply({"error": "not found"}, status=404)
        except (KeyError, FileNotFoundError, SystemExit) as e:
            self.reply({"error": f"not found: {e}"}, status=404)

    def do_POST(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/plan":
                self.reply(self.scenes.plan(body["name"], body["points"]))
            elif self.path == "/api/save":
                self.reply(self.scenes.save(body["name"], body["routes"]))
            else:
                self.reply({"error": "not found"}, status=404)
        except (KeyError, ValueError, TypeError, SystemExit) as e:
            self.reply({"error": f"bad request: {e!r}"}, status=400)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765, help="Port to serve the page on (default 8765).")
    p.add_argument("--host", default="127.0.0.1", help="Address to listen on (default 127.0.0.1: this machine only).")
    p.add_argument("--routes", default=str(cli.ROUTES), help="Folder for the annotations (default navbench/routes).")
    p.add_argument("--robot", default="delivery", choices=sorted(ROBOTS),
                   help="Robot the preview plans for (default delivery).")
    a = p.parse_args()
    Handler.scenes = Scenes(cli.MAPS, Path(a.routes).expanduser().resolve(), ROBOTS[a.robot])
    Handler.scenes.names()  # fail now if there are no maps
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[annotate] open http://localhost:{a.port} - annotations go to {Handler.scenes.routes}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
