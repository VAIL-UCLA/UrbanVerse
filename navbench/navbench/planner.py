"""Plan a route over a scene's map: the cheapest way from start to goal that the robot fits through.

The map says what is where (maps.py); this file decides what a robot can do with it:

  walkable   there is ground, nothing above it lower than the robot, no step higher than it climbs
             between neighboring cells, and no obstacle closer than its radius
  cost       per meter, by the kind of ground (sidewalk cheapest, road dearest), plus a little
             near obstacles, plus a fixed price for every curb climbed

The search runs on a coarser grid than the map (PLAN_RES), 8-connected, with scikit-image's
MCP_Geometric. The path is then straightened where a straight line is no dearer, and resampled
to waypoints every WAYPOINT_SPACING meters, with the ground height as z.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric

from .maps import KINDS, Map

PLAN_RES = 0.1  # m: cell size of the search
WAYPOINT_SPACING = 0.5  # m between waypoints
SNAP = 1.0  # m: a start or goal on blocked ground moves to the nearest walkable cell this close
CROP = 25.0  # m: first search only this far around the points; the whole map if the path needs more
COST = {"road": 6.0, "road_marking": 6.0, "crosswalk": 1.5, "sidewalk": 1.0, "ground": 1.2}  # per meter
NEAR = 0.5  # m: within this much of the robot's radius from an obstacle, walking costs up to NEAR_COST more
NEAR_COST = 1.0
CURB = 0.03  # m: a height change between neighboring cells above this is a curb (below it, a slope or seam)
CURB_COST = 5.0  # m of sidewalk that climbing one curb is worth
CROSSING = 1.0  # m of road in a row that make a road crossing
ZEBRA = 0.6  # m: gaps between a crosswalk's stripes up to twice this are part of the crosswalk


@dataclass
class Robot:
    """What the planner needs to know of a robot. Sizes in meters."""
    name: str = "delivery"
    radius: float = 0.35  # of the circle around its footprint
    height: float = 1.0  # free height it needs
    step: float = 0.22  # highest step it climbs: a curb


ROBOTS = {
    "delivery": Robot(),  # a six-wheeled sidewalk delivery robot, which climbs curbs
    "quadruped": Robot("quadruped", radius=0.35, height=0.5, step=0.25),  # e.g. Unitree Go2
    "wheelchair": Robot("wheelchair", radius=0.45, height=1.4, step=0.05),  # needs curb ramps
}


@dataclass
class Route:
    points: list  # the start, the via points and the goal, as planned from (after snapping), [[x, y], ...]
    path: np.ndarray  # Kx2: corners of the straightened path, in world meters
    waypoints: np.ndarray  # Nx3: x, y, z every WAYPOINT_SPACING meters along the path
    yaw: np.ndarray  # N: heading at each waypoint, radians from +X
    facts: dict = field(default_factory=dict)  # length, crossings, ... (see Planner.facts)


class NoRoute(Exception):
    """A point is blocked, or the goal cannot be reached."""


class Planner:
    """The cost map of one scene for one robot; `route()` then plans as many routes as wanted."""

    def __init__(self, m: Map, robot: Robot = None):
        self.map, self.robot = m, robot or Robot()
        self.factor = max(1, int(round(PLAN_RES / m.grid.res)))
        self.res = m.grid.res * self.factor
        fine = self._fine()
        # The search grid: a cell is as dear as its dearest map cell, blocked if any of them is.
        self.cost = self._coarse(fine["cost"], np.max, np.inf)
        self.walkable = np.isfinite(self.cost)
        self.kind = self._coarse(fine["kind"], self._middle, 0)
        self.clear = self._coarse(fine["clear"], np.min, 0.0)  # m from the cell to the nearest obstacle
        self.ground = self._coarse(np.where(np.isfinite(fine["ground"]), fine["ground"], -np.inf), np.max, -np.inf)
        self.ground[np.isinf(self.ground)] = np.nan
        self.curb = self._coarse(fine["curb"], np.max, False)
        # Walkable cells that reach each other: a building's hollow inside is walkable, but sealed off.
        self.region, _ = ndimage.label(self.walkable, structure=np.ones((3, 3)))
        sizes = np.bincount(self.region.ravel())
        sizes[0] = 0
        self.main = self.region == np.argmax(sizes)  # the largest region: the streets

    # ── the cost map ─────────────────────────────────────────────────────────────────────────

    def _fine(self) -> dict:
        """Walkability and cost at the map's own resolution."""
        m, r = self.map, self.robot
        ground = m.ground
        has = np.isfinite(ground)
        step = np.zeros(ground.shape, np.float32)  # the largest height change to a neighbor
        padded = np.pad(np.where(has, ground, np.nan), 1, constant_values=np.nan)
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            there = padded[1 + dr:1 + dr + ground.shape[0], 1 + dc:1 + dc + ground.shape[1]]
            back = padded[1 - dr:1 - dr + ground.shape[0], 1 - dc:1 - dc + ground.shape[1]]
            for other in (there, back):
                diff = np.abs(other - ground)
                step = np.fmax(step, np.where(np.isfinite(other), diff, 0.0))
        blocked = ~has | (m.clearance < r.height) | (step > r.step)
        clear = ndimage.distance_transform_edt(~blocked).astype(np.float32) * m.grid.res
        kind = self._whole_crosswalks(m.kind)
        cost = np.full(ground.shape, np.inf, np.float32)
        for name, c in COST.items():
            cost[kind == KINDS.index(name)] = c
        cost += NEAR_COST * np.clip(1.0 - (clear - r.radius) / NEAR, 0.0, 1.0)
        curb = (step > CURB) & ~blocked
        cost[curb] += CURB_COST / (2 * self.res)  # a curb is about two search cells across
        cost[clear <= r.radius] = np.inf
        return {"cost": cost, "kind": kind, "clear": clear, "ground": ground, "curb": curb}

    def _whole_crosswalks(self, kind: np.ndarray) -> np.ndarray:
        """A zebra crossing's stripes are crosswalk and the road between them road; make the road
        between the stripes crosswalk too (a morphological closing), or routes weave stripe to stripe."""
        stripes = kind == KINDS.index("crosswalk")
        if not stripes.any():
            return kind
        res = self.map.grid.res
        near = ndimage.distance_transform_edt(~stripes) * res <= ZEBRA
        whole = ndimage.distance_transform_edt(near) * res > ZEBRA
        road = np.isin(kind, [KINDS.index("road"), KINDS.index("road_marking")])
        return np.where(whole & road, np.uint8(KINDS.index("crosswalk")), kind)

    def _coarse(self, a: np.ndarray, reduce, fill) -> np.ndarray:
        """`a` on the search grid: `reduce` over each block of factor x factor map cells.
        `fill` pads the last row and column of blocks where the map ends."""
        f = self.factor
        if f == 1:
            return a
        h, w = -(-a.shape[0] // f) * f, -(-a.shape[1] // f) * f
        padded = np.full((h, w), fill, dtype=a.dtype)
        padded[:a.shape[0], :a.shape[1]] = a
        return reduce(padded.reshape(h // f, f, w // f, f), axis=(1, 3))

    def _middle(self, blocks: np.ndarray, axis) -> np.ndarray:
        """Of every block, the map cell in its middle."""
        return blocks[:, self.factor // 2, :, self.factor // 2]

    # ── cells and world points ───────────────────────────────────────────────────────────────

    def cell(self, x: float, y: float) -> tuple:
        g = self.map.grid
        return int(np.floor((g.y_max - y) / self.res)), int(np.floor((x - g.x_min) / self.res))

    def world(self, rows, cols) -> np.ndarray:
        g = self.map.grid
        return np.stack([g.x_min + (np.asarray(cols) + 0.5) * self.res,
                         g.y_max - (np.asarray(rows) + 0.5) * self.res], axis=-1)

    def snap(self, x: float, y: float, what: str = "point") -> tuple:
        """The walkable cell at (x, y), or the nearest one within SNAP meters."""
        r, c = self.cell(x, y)
        h, w = self.cost.shape
        if not (0 <= r < h and 0 <= c < w):
            raise NoRoute(f"{what} ({x:.2f}, {y:.2f}) is outside the map")
        if self.walkable[r, c]:
            return r, c
        n = int(np.ceil(SNAP / self.res))
        r0, c0 = max(0, r - n), max(0, c - n)
        rows, cols = np.nonzero(self.walkable[r0:r + n + 1, c0:c + n + 1])
        if not len(rows):
            raise NoRoute(f"{what} ({x:.2f}, {y:.2f}) is blocked for the {self.robot.name} robot, "
                          f"and nothing within {SNAP:g} m is walkable")
        d = np.hypot(rows + r0 - r, cols + c0 - c)
        i = int(np.argmin(d))
        return int(rows[i] + r0), int(cols[i] + c0)

    # ── routes ───────────────────────────────────────────────────────────────────────────────

    def route(self, points) -> Route:
        """The route through `points` = [(x, y) start, (x, y) via, ..., (x, y) goal], in world meters."""
        if len(points) < 2:
            raise NoRoute("a route needs a start and a goal")
        names = ["start"] + [f"via point {i}" for i in range(1, len(points) - 1)] + ["goal"]
        cells = [self.snap(float(x), float(y), what) for (x, y), what in zip(points, names, strict=True)]
        path = [cells[0]]
        for i in range(len(cells) - 1):
            leg = self._leg(cells[i], cells[i + 1], names[i + 1])
            path += self.straighten(leg)[1:]
        path = np.array(path)
        corners = self.world(path[:, 0], path[:, 1])
        waypoints, yaw = self._resample(corners)
        route = Route([self.world(*c).round(3).tolist() for c in cells], corners, waypoints, yaw)
        route.facts = self.facts(route, cells)
        return route

    def _leg(self, a: tuple, b: tuple, what: str) -> list:
        """The cheapest 8-connected chain of cells from a to b: first near them, then over the whole map."""
        h, w = self.cost.shape
        pad = int(np.ceil(max(CROP, 0.5 * self.res * np.hypot(a[0] - b[0], a[1] - b[1])) / self.res))
        near = (max(0, min(a[0], b[0]) - pad), max(0, min(a[1], b[1]) - pad),
                min(h, max(a[0], b[0]) + pad + 1), min(w, max(a[1], b[1]) + pad + 1))
        for window in dict.fromkeys([near, (0, 0, h, w)]):  # once if the two are the same
            leg = self._search(a, b, window)
            if leg is None:
                continue
            y0, x0, y1, x1 = window
            rows, cols = np.array(leg).T
            cut = (((rows == y0) & (y0 > 0)) | ((rows == y1 - 1) & (y1 < h))
                   | ((cols == x0) & (x0 > 0)) | ((cols == x1 - 1) & (x1 < w)))
            if not cut.any():  # a path along the edge of the window may have been cut short by it
                return leg
        raise NoRoute(f"the {what} cannot be reached: no walkable way for the {self.robot.name} robot")

    def _search(self, a: tuple, b: tuple, window: tuple):
        y0, x0, y1, x1 = window
        mcp = MCP_Geometric(self.cost[y0:y1, x0:x1], fully_connected=True)
        costs, _ = mcp.find_costs([(a[0] - y0, a[1] - x0)], [(b[0] - y0, b[1] - x0)])
        if not np.isfinite(costs[b[0] - y0, b[1] - x0]):
            return None
        return [(r + y0, c + x0) for r, c in mcp.traceback((b[0] - y0, b[1] - x0))]

    def straighten(self, leg: list) -> list:
        """Of a chain of cells, the corners left when every stretch that a straight line covers,
        through walkable cells and at no more cost, is replaced by that line."""
        leg = np.array(leg)
        step = np.r_[0.0, np.hypot(*np.diff(leg, axis=0).T)]
        along = np.cumsum(self.cost[leg[:, 0], leg[:, 1]] * step)  # cost to each cell of the chain
        keep, i = [0], 0
        while i < len(leg) - 1:
            j = i + 1
            while j + 1 < len(leg) and self._line_cost(leg[i], leg[j + 1]) <= (along[j + 1] - along[i]) * 1.001 + 1e-9:
                j += 1
            keep.append(j)
            i = j
        return [tuple(leg[k]) for k in keep]

    def _line_cost(self, a, b) -> float:
        n = int(np.ceil(2 * np.hypot(b[0] - a[0], b[1] - a[1]))) + 1
        t = np.linspace(0.0, 1.0, n)
        rows = np.rint(a[0] + t * (b[0] - a[0])).astype(int)
        cols = np.rint(a[1] + t * (b[1] - a[1])).astype(int)
        c = self.cost[rows, cols]
        return float(np.inf) if not np.isfinite(c).all() else float(c.mean() * np.hypot(b[0] - a[0], b[1] - a[1]))

    def _resample(self, corners: np.ndarray) -> tuple:
        seg = np.hypot(*np.diff(corners, axis=0).T)
        at = np.r_[0.0, np.cumsum(seg)]
        n = max(2, int(np.ceil(at[-1] / WAYPOINT_SPACING)) + 1)
        s = np.linspace(0.0, at[-1], n)
        xy = np.stack([np.interp(s, at, corners[:, 0]), np.interp(s, at, corners[:, 1])], axis=1)
        yaw = np.arctan2(*np.diff(xy, axis=0).T[::-1]) if len(xy) > 1 else np.zeros(0)
        yaw = np.r_[yaw, yaw[-1:]] if len(yaw) else np.zeros(len(xy))
        return np.column_stack([xy, self.height(xy)]), yaw

    def height(self, xy: np.ndarray) -> np.ndarray:
        """Ground height under world points: of the map cell, else the highest of its search cell."""
        g, m = self.map.grid, self.map
        col, row = g.cell(xy[:, 0], xy[:, 1])
        ok = g.inside(col, row)
        z = np.full(len(xy), np.nan)
        z[ok] = m.ground[row[ok], col[ok]]
        missing = np.isnan(z)
        if missing.any():
            r, c = np.array([self.cell(x, y) for x, y in xy[missing]]).T
            z[missing] = self.ground[np.clip(r, 0, self.ground.shape[0] - 1), np.clip(c, 0, self.ground.shape[1] - 1)]
        return z

    # ── what the route is like ───────────────────────────────────────────────────────────────

    def facts(self, route: Route, cells: list) -> dict:
        """Numbers to pick and group routes by. All lengths in meters."""
        corners = route.path
        seg = np.hypot(*np.diff(corners, axis=0).T)
        length = float(seg.sum())
        # Walk the path in small steps and see what is under it.
        n = max(2, int(np.ceil(length / (self.res / 2))) + 1)
        at = np.r_[0.0, np.cumsum(seg)]
        s = np.linspace(0.0, length, n)
        xy = np.stack([np.interp(s, at, corners[:, 0]), np.interp(s, at, corners[:, 1])], axis=1)
        rc = np.array([self.cell(x, y) for x, y in xy])
        rc[:, 0] = np.clip(rc[:, 0], 0, self.cost.shape[0] - 1)
        rc[:, 1] = np.clip(rc[:, 1], 0, self.cost.shape[1] - 1)
        kind = np.array(KINDS)[self.kind[rc[:, 0], rc[:, 1]]]
        ds = length / (n - 1)
        by_kind = {k: round(float((kind == k).sum() * ds), 2) for k in KINDS[1:] if (kind == k).any()}
        on_road = np.isin(kind, ["road", "road_marking", "crosswalk"])
        runs = np.flatnonzero(np.diff(np.r_[0, on_road.astype(int), 0]))
        crossings = int(sum((end - start) * ds >= CROSSING for start, end in zip(runs[::2], runs[1::2], strict=True)))
        curb = self.curb[rc[:, 0], rc[:, 1]]
        heading = np.arctan2(*np.diff(corners, axis=0).T[::-1])
        turn = np.abs((np.diff(heading) + np.pi) % (2 * np.pi) - np.pi)
        start, goal = route.points[0], route.points[-1]
        straight = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
        return {
            "length_m": round(length, 2),
            "straight_m": round(straight, 2),
            "detour": round(length / straight, 3) if straight > 0 else None,
            "by_kind_m": by_kind,
            "road_crossings": crossings,
            "curbs": int(np.count_nonzero(np.diff(np.r_[0, curb.astype(int)]) == 1)),
            "min_clearance_m": round(float(self.clear[rc[:, 0], rc[:, 1]].min()), 2),
            "turns": int((turn > np.radians(30)).sum()),
            "climb_m": round(float(np.nanmax(route.waypoints[:, 2]) - np.nanmin(route.waypoints[:, 2])), 2),
            "robot": asdict(self.robot),
        }
