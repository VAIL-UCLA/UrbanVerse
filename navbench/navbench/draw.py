"""Route maps: routes drawn over a scene's top-down image."""
import numpy as np
from PIL import Image, ImageDraw

from .maps import Grid

COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
          (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128)]
START, GOAL, VIA = (40, 200, 60), (220, 30, 30), (250, 200, 0)


def overlay(planner) -> np.ndarray:
    """RGBA on the search grid: blocked cells red, walkable cells sealed off from the streets orange,
    the rest clear. What the robot of `planner` cannot use, at a glance."""
    rgba = np.zeros(planner.cost.shape + (4,), np.uint8)
    rgba[~planner.walkable] = (200, 30, 30, 110)
    rgba[planner.walkable & ~planner.main] = (255, 140, 0, 90)
    return rgba


def backdrop(image: np.ndarray, planner=None) -> Image.Image:
    """The top-down image, with the planner's overlay laid over it if one is given."""
    im = Image.fromarray(image).convert("RGBA")
    if planner is not None:
        im.alpha_composite(Image.fromarray(overlay(planner)).resize(im.size, Image.NEAREST))
    return im.convert("RGB")


def pixels(grid: Grid, xy) -> list:
    """Image positions (column, row) of world points, as floats."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return list(zip((xy[:, 0] - grid.x_min) / grid.res, (grid.y_max - xy[:, 1]) / grid.res, strict=True))


def route(im: Image.Image, grid: Grid, path, points, color, label: str = "", scale: float = 1.0) -> None:
    """Draw one route: its path, its start (green), via points (yellow), goal (red) and its id."""
    d = ImageDraw.Draw(im)
    w = max(2, int(round(4 * scale)))
    d.line(pixels(grid, path), fill=color, width=w, joint="curve")
    r = 3 * w
    for i, (x, y) in enumerate(pixels(grid, points)):
        fill = START if i == 0 else GOAL if i == len(points) - 1 else VIA
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=(0, 0, 0), width=max(1, w // 2))
    if label:
        x, y = pixels(grid, points[:1])[0]
        d.text((x + r + 2, y - r - 2), label, fill=(255, 255, 255), stroke_width=max(1, w // 2),
               stroke_fill=(0, 0, 0), font_size=max(12, int(14 * scale)))


def crop(im: Image.Image, grid: Grid, xy, margin: float) -> Image.Image:
    """The part of the image around world points, `margin` meters to each side."""
    px = np.array(pixels(grid, xy))
    m = margin / grid.res
    x0, y0 = np.maximum(px.min(axis=0) - m, 0).astype(int)
    x1, y1 = np.minimum(px.max(axis=0) + m, im.size).astype(int)
    return im.crop((x0, y0, x1, y1))


def shrink(im: Image.Image, longest: int) -> Image.Image:
    if max(im.size) > longest:
        im = im.copy()
        im.thumbnail((longest, longest), Image.LANCZOS)
    return im
