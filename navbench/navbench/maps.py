"""A scene's map: a grid over the ground with a known place in the world, and its layers."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# What a cell's ground is. The numbers are what the `kind` layer holds.
KINDS = ["none", "road", "road_marking", "crosswalk", "sidewalk", "ground"]
NO_CLEARANCE_LIMIT = 9.99  # m: clearance of a cell with nothing above it, in the range the map looks at


@dataclass
class Grid:
    """Cells of `res` meters. Column 0 is at x_min, row 0 at y_max: +X is right, +Y is up in the image."""
    x_min: float
    y_max: float
    res: float
    width: int
    height: int

    @classmethod
    def around(cls, x_min: float, y_min: float, x_max: float, y_max: float, res: float) -> "Grid":
        return cls(float(x_min), float(y_max), res, int(np.ceil(round((x_max - x_min) / res, 6))),
                   int(np.ceil(round((y_max - y_min) / res, 6))))

    @property
    def bounds(self) -> tuple:
        """(x_min, y_min, x_max, y_max) in world meters."""
        return self.x_min, self.y_max - self.height * self.res, self.x_min + self.width * self.res, self.y_max

    def cell(self, x, y) -> tuple:
        """(column, row) of world points; outside the grid they are out of range."""
        return (np.floor((np.asarray(x) - self.x_min) / self.res).astype(np.int64),
                np.floor((self.y_max - np.asarray(y)) / self.res).astype(np.int64))

    def inside(self, col, row):
        return (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)

    def world(self, col, row) -> tuple:
        """(x, y) of cell centers. Fractions are fine: world(col + 0.5, row) is on the cell's right edge."""
        return (self.x_min + (np.asarray(col) + 0.5) * self.res,
                self.y_max - (np.asarray(row) + 0.5) * self.res)


@dataclass
class Map:
    scene: str
    grid: Grid
    ground: np.ndarray  # HxW float32, m: height of the ground, NaN where there is none
    clearance: np.ndarray  # HxW float32, m: free height above the ground, NO_CLEARANCE_LIMIT if nothing is above
    top: np.ndarray  # HxW float32, m: height of the highest thing in the cell, NaN where there is nothing
    kind: np.ndarray  # HxW uint8: index into KINDS
    info: dict  # how the map was made

    def save(self, folder) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "layers.npz", ground=self.ground, clearance=self.clearance, top=self.top,
                            kind=self.kind)
        (folder / "map.json").write_text(json.dumps(
            {"scene": self.scene, "grid": asdict(self.grid), "kinds": KINDS, **self.info}, indent=1))

    @classmethod
    def load(cls, folder) -> "Map":
        folder = Path(folder)
        info = json.loads((folder / "map.json").read_text())
        info.pop("kinds")
        with np.load(folder / "layers.npz") as layers:
            return cls(info.pop("scene"), Grid(**info.pop("grid")), layers["ground"], layers["clearance"],
                       layers["top"], layers["kind"], info)
