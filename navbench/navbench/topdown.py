"""Build a scene's map: look straight down at it, and keep what a planner needs of every cell.

  ground     height of the ground: the highest flat surface near street level; of a model (a plaza's
             paving, a speed bump) only up to OBJECT_ABOVE over the road network under it, so that
             a car's hood or a bench's seat is not ground
  kind       what that ground is: road, sidewalk, crosswalk, ... by the name of the road network's
             surface there; "ground" where only a model's floor is
  clearance  free height above the ground, up to the first thing above it
  top        height of the highest thing in the cell: what one sees from above
  two images of the scene from above: all of it, and cut off above head height

Nothing here depends on the robot. What a given robot can do with a cell is planner.py's call.

The map is built tile by tile, in as many processes as the machine has cores. Every point of
every surface lands in the same cell whatever the tiles are, so the map does not depend on them.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .maps import KINDS, NO_CLEARANCE_LIMIT, Grid, Map
from .raster import Canvas, colors, nearest, normals, samples
from .scene import Scene

FLAT = 0.82  # |normal z| of a surface that can be ground: up to 35 degrees of slope
DZ = 0.05  # m: height of one level of the columns
LEVELS = 128  # levels of a column: 6.4 m
BAND = 0.6  # m: ground is looked for this far below and above street level
OBJECT_ABOVE = 0.12  # m: how far above street level a model's surface can be ground (a floor, not a car hood)
HEAD = 2.4  # m above street level: where the cut view is cut off
LOOSE = 1  # levels right above the ground that do not count as something above it (up to 2 * DZ)
TILE = 256  # cells along the side of a tile
WORKER_GB = 4.0  # memory one worker may take (a scene loaded, a tile being drawn): up to 3.8 GB on scene_12
SPARE_GB = 4.0  # memory left to the rest of the machine
SURVEY = 0.5  # m: cell size of the first look at a scene


class Columns:
    """For every cell, which levels of a range of heights have something in them."""

    def __init__(self, cells: int, bottom: float):
        self.bottom = bottom
        self.bits = np.zeros(cells * (LEVELS // 64), dtype=np.uint64)

    def level(self, z: np.ndarray) -> np.ndarray:
        return np.floor((z - self.bottom) / DZ).astype(np.int64)

    def add(self, cell: np.ndarray, z: np.ndarray) -> None:
        level = self.level(z)
        ok = (level >= 0) & (level < LEVELS)
        key = np.unique(cell[ok] * LEVELS + level[ok])  # sorted: by cell, then by level
        if not len(key):
            return
        word = key // 64  # = cell * (LEVELS // 64) + level // 64
        bit = np.uint64(1) << (key % 64).astype(np.uint64)
        start = np.flatnonzero(np.r_[True, word[1:] != word[:-1]])
        self.bits[word[start]] |= np.bitwise_or.reduceat(bit, start)

    def lowest_above(self, level: np.ndarray) -> np.ndarray:
        """For every cell the lowest level above `level` with something in it; LEVELS if there is none."""
        lowest = np.full(len(level), LEVELS, dtype=np.int64)
        words = self.bits.reshape(len(level), -1)
        for w in range(words.shape[1] - 1, -1, -1):  # from the top word down: a lower one overrules
            first = np.clip(level + 1 - 64 * w, 0, 64)  # levels of this word below `first` do not count
            bits = np.where(first < 64, (words[:, w] >> first.astype(np.uint64)) << first.astype(np.uint64),
                            np.uint64(0))
            some = bits != 0
            lone = bits[some] & (~bits[some] + np.uint64(1))  # the lowest bit alone: a power of two
            lowest[some] = 64 * w + np.log2(lone.astype(np.float64)).astype(np.int64)
        return lowest


def keep_inside(grid: Grid, z_min: float = -np.inf, z_max: float = np.inf):
    """Triangles whose box touches the grid, between two heights."""
    x0, y0, x1, y1 = grid.bounds
    low, high = np.array([x0, y0, z_min]), np.array([x1, y1, z_max])

    def keep(corners: np.ndarray) -> np.ndarray:
        return ((corners.max(axis=1) >= low) & (corners.min(axis=1) <= high)).all(axis=1)
    return keep


def survey(scene: Scene, window: tuple) -> tuple:
    """A first, coarse look at the part of the scene inside `window` = (x_min, y_min, x_max, y_max):
    (street level, the part of the window that has anything in it).

    Street level is the height where most of the flat surface one sees from above lies: of the
    named ground (roads, sidewalks) if the scene has any, else of everything."""
    grid = Grid.around(*window, SURVEY)
    top = np.full(grid.width * grid.height, -np.inf)
    flat = np.zeros(len(top), bool)
    named = np.zeros(len(top), bool)
    for s in scene.surfaces(window):
        normal = normals(s)
        for points, face, _ in samples(s, SURVEY / 2, keep_inside(grid), flat=True):
            col, row = grid.cell(points[:, 0], points[:, 1])
            ok = grid.inside(col, row)
            cell, z, face = (row * grid.width + col)[ok], points[ok, 2], face[ok]
            won = nearest(cell, -z)
            won = won[z[won] > top[cell[won]]]
            top[cell[won]], flat[cell[won]] = z[won], np.abs(normal[face[won], 2]) >= FLAT
            named[cell[won]] = s.kind != "object"
    seen = flat & (named if (flat & named).sum() > 100 else True)
    if not seen.any():
        raise SystemExit(f"{scene.usd}: no flat surface inside {window}")
    count, edge = np.histogram(top[seen], bins=np.arange(top[seen].min() - 0.1, top[seen].max() + 0.2, 0.1))
    best = np.argmax(np.convolve(count, np.ones(3), "same"))  # the fullest 30 cm
    level = float(np.median(top[seen & (np.abs(top - edge[best] - 0.05) <= 0.2)]))

    full = np.isfinite(top).reshape(grid.height, grid.width)
    cols, rows = np.flatnonzero(full.any(axis=0)), np.flatnonzero(full.any(axis=1))
    x0, y1 = grid.x_min + cols[0] * SURVEY, grid.y_max - rows[0] * SURVEY
    return level, (x0, y1 - (rows[-1] - rows[0] + 1) * SURVEY, x0 + (cols[-1] - cols[0] + 1) * SURVEY, y1)


def build_tile(scene: Scene, grid: Grid, level: float) -> dict:
    """The layers and images of one tile, as flat arrays of its cells, and what went into them."""
    cells = grid.width * grid.height
    top, cut = Canvas(grid.width, grid.height), Canvas(grid.width, grid.height)
    # The highest flat surface near street level of the road network (with its kind), and of the models.
    road, road_kind = np.full(cells, -np.inf, dtype=np.float32), np.zeros(cells, dtype=np.uint8)
    floor = np.full(cells, -np.inf, dtype=np.float32)
    columns = Columns(cells, bottom=level - BAND)
    ceiling = columns.bottom + LEVELS * DZ
    count = {"triangles": 0, "points": 0}

    def inside(points, face, weights):
        col, row = grid.cell(points[:, 0], points[:, 1])
        ok = grid.inside(col, row)
        count["points"] += int(ok.sum())
        return (row * grid.width + col)[ok], points[ok, 2], face[ok], weights[ok]

    for s in scene.surfaces(grid.bounds):
        normal = normals(s)
        count["triangles"] += int(keep_inside(grid)(s.points[s.faces]).sum())
        # What is within the height of the columns: points close enough for every cell and level.
        for chunk in samples(s, min(grid.res / 2, DZ), keep_inside(grid, columns.bottom, ceiling)):
            cell, z, face, weights = inside(*chunk)
            if not len(cell):
                continue

            def rgb(won, m=slice(None), s=s, face=face, weights=weights, normal=normal):
                return colors(s, face[m][won], weights[m][won], normal)

            top.put(cell, -z, rgb)
            low = z <= level + HEAD
            cut.put(cell[low], -z[low], lambda won, low=low, rgb=rgb: rgb(won, low))
            columns.add(cell, z)
            flat = (np.abs(normal[face, 2]) >= FLAT) & (z >= level - BAND) \
                & (z <= level + (OBJECT_ABOVE if s.kind == "object" else BAND))
            won = np.flatnonzero(flat)[nearest(cell[flat], -z[flat])]
            highest = floor if s.kind == "object" else road
            won = won[z[won] > highest[cell[won]]]
            highest[cell[won]] = z[won]
            if s.kind != "object":
                road_kind[cell[won]] = KINDS.index(s.kind)
        # What is above that only shows in the picture from above: points close enough as seen from there.
        for chunk in samples(s, grid.res / 2, keep_inside(grid, z_min=ceiling), flat=True):
            cell, z, face, weights = inside(*chunk)
            if len(cell):
                top.put(cell, -z, lambda won, s=s, face=face, weights=weights, normal=normal:
                        colors(s, face[won], weights[won], normal))

    # A model's floor is ground if it is at most OBJECT_ABOVE over the road network under it (a speed
    # bump, a plaza's paving; not a car's hood), and of the kind of that road network.
    use = np.isfinite(floor) & (floor <= np.where(np.isfinite(road), road, level) + OBJECT_ABOVE)
    ground = np.where(use, np.maximum(floor, road), road)
    kind = np.where(np.isfinite(road), road_kind, np.where(use, KINDS.index("ground"), 0)).astype(np.uint8)
    has = np.isfinite(ground)
    above = columns.lowest_above(np.where(has, columns.level(np.where(has, ground, level)) + LOOSE, LEVELS))
    clearance = np.where(above < LEVELS, columns.bottom + above * DZ - ground, NO_CLEARANCE_LIMIT)
    clearance = np.where(has, np.clip(clearance, 0.0, NO_CLEARANCE_LIMIT), 0.0).astype(np.float32)
    ground[~has] = np.nan
    return {"ground": ground, "kind": kind, "clearance": clearance,
            "top": np.where(np.isfinite(top.depth), -top.depth, np.nan).astype(np.float32),
            "topdown": top.rgb, "cut": cut.rgb, "count": count}


def default_workers() -> int:
    """One per core, but no more than the free memory holds: every worker loads the whole scene."""
    try:
        with open("/proc/meminfo") as f:
            free_gb = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable:")) / 2**20
    except (OSError, StopIteration):
        free_gb = 16.0
    return max(1, min(os.cpu_count() or 1, int((free_gb - SPARE_GB) / WORKER_GB)))


_scene = None  # the scene of this worker process


def _open(usd: str) -> None:
    global _scene
    _scene = Scene(usd)


def _tile(job: tuple) -> tuple:
    col, row, grid, level = job
    return col, row, grid, build_tile(_scene, grid, level)


def build(scene: Scene, name: str, grid: Grid, level: float, workers: int = 0, progress=None) -> tuple:
    """(the scene's Map, {"topdown": image of all of it, "cut": image of what is below head height}).
    `progress(tiles done, tiles)` is called as the tiles come in."""
    jobs = [(col, row, Grid(grid.x_min + col * grid.res, grid.y_max - row * grid.res, grid.res,
                            min(TILE, grid.width - col), min(TILE, grid.height - row)), level)
            for row in range(0, grid.height, TILE) for col in range(0, grid.width, TILE)]
    shape = (grid.height, grid.width)
    layers = {"ground": np.empty(shape, np.float32), "clearance": np.empty(shape, np.float32),
              "top": np.empty(shape, np.float32), "kind": np.empty(shape, np.uint8),
              "topdown": np.empty(shape + (3,), np.uint8), "cut": np.empty(shape + (3,), np.uint8)}
    count = {"triangles": 0, "points": 0}

    def paste(done: tuple) -> None:
        col, row, tile, part = done
        for key, layer in layers.items():
            layer[row:row + tile.height, col:col + tile.width] = part[key].reshape(tile.height, tile.width, -1) \
                if layer.ndim == 3 else part[key].reshape(tile.height, tile.width)
        for key in count:
            count[key] += part["count"][key]

    workers = min(workers or default_workers(), len(jobs))
    if workers == 1:
        results = ((col, row, tile, build_tile(scene, tile, level)) for col, row, tile, level in jobs)
        for n, done in enumerate(results, 1):
            paste(done)
            if progress:
                progress(n, len(jobs))
    else:
        with ProcessPoolExecutor(workers, initializer=_open, initargs=(str(scene.usd),)) as pool:
            for n, done in enumerate(pool.map(_tile, jobs), 1):
                paste(done)
                if progress:
                    progress(n, len(jobs))
    info = {"usd": str(scene.usd), "street_level": round(level, 3), "band": BAND, "object_above": OBJECT_ABOVE,
            "head": HEAD,
            "clearance_step": DZ, "ground_planes": sorted({round(h, 3) for h in scene.collision_planes()}),
            "count": {"meshes": len(scene.sources()), **count}, "notes": scene.notes}
    images = {"topdown": layers.pop("topdown"), "cut": layers.pop("cut")}
    return Map(name, grid, info=info, **layers), images
