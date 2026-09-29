"""Draw surfaces by sampling them: every triangle gets points no further apart than a pixel,
and in every pixel the nearest point wins. Plain numpy, no GPU.

Two views share the sampling: the map from straight above (topdown.py), and a pinhole camera
(`Camera`), which is how the reconstructed scene is compared with Isaac Sim's renders.
"""
import numpy as np

from .scene import Surface

MAX_SPLIT = 32  # a triangle gets at most MAX_SPLIT rows of points at once; longer ones are halved first
BLOCK = 300_000  # triangles handled at once
CHUNK = 2_000_000  # points handled at once
LIGHT = np.array([0.35, 0.25, 0.9]) / np.linalg.norm([0.35, 0.25, 0.9])  # from above, slightly to one side
_PATTERNS = {}


def pattern(rows: int, across: int) -> np.ndarray:
    """Weights of the corners (K x 3) for points spread over a triangle: `rows` rows from corner 0
    to the edge opposite, the last of them with `across` points, the others with fewer as the
    triangle narrows."""
    if (rows, across) not in _PATTERNS:
        points = []
        for i in range(rows):
            s = (i + 0.5) / rows
            m = max(1, int(np.ceil(s * across)))
            points += [(1 - s, s - s * (j + 0.5) / m, s * (j + 0.5) / m) for j in range(m)]
        _PATTERNS[rows, across] = np.array(points)
    return _PATTERNS[rows, across]


def samples(s: Surface, spacing, keep=None, flat: bool = False):
    """Points on the surface, in chunks: (points Kx3, face of each point, its barycentric
    coordinates in that face Kx3).

    `spacing` is how far apart the points may be, in meters: a number, or a function of the
    triangles (T x 3 corners x 3) that gives a number per triangle.
    `keep`, a function of the triangles that gives a bool per triangle, drops triangles early,
    e.g. those outside the picture.
    `flat` measures the spacing as seen from above: a wall then gets points along its length only."""
    for start in range(0, len(s.faces), BLOCK):
        face = np.arange(start, min(start + BLOCK, len(s.faces)))
        corners = s.points[s.faces[face]]
        bary = np.broadcast_to(np.eye(3), (len(face), 3, 3))  # the corners, in barycentric coordinates
        while len(face):
            if keep is not None:
                ok = keep(corners)
                corners, face, bary = corners[ok], face[ok], bary[ok]
            space = spacing(corners) if callable(spacing) else np.full(len(face), float(spacing))
            edge = edges(corners, flat)
            big = edge.max(axis=1) > MAX_SPLIT * space
            yield from spread(corners[~big], face[~big], bary[~big], edge[~big], space[~big])
            if not big.any():
                break
            first = edge[big].argmax(axis=1)  # halve at the middle of the longest edge
            corners, bary = halves(turn(corners[big], first)), halves(turn(bary[big], first))
            face = np.tile(face[big], 2)


def edges(corners: np.ndarray, flat: bool) -> np.ndarray:
    """Lengths (T x 3) of the edges 01, 12, 20; as seen from above if `flat`."""
    c = corners[:, :, :2] if flat else corners
    return np.linalg.norm(np.roll(c, -1, axis=1) - c, axis=2)


def turn(c: np.ndarray, first: np.ndarray) -> np.ndarray:
    """The triangles (T x 3 corners x D) with corner `first` (T) as their corner 0, in the same sense."""
    order = (first[:, None] + np.arange(3)) % 3
    return np.take_along_axis(c, order[:, :, None], axis=1)


def halves(c: np.ndarray) -> np.ndarray:
    """The two triangles each triangle splits into at the middle of its edge 01."""
    m = (c[:, 0] + c[:, 1]) / 2
    return np.concatenate([np.stack([c[:, 0], m, c[:, 2]], axis=1), np.stack([m, c[:, 1], c[:, 2]], axis=1)])


def spread(corners, face, bary, edge, space):
    """Points on triangles that are small enough: rows of them, from the corner opposite the
    shortest edge to that edge."""
    first = (edge.argmin(axis=1) + 2) % 3  # edge i runs from corner i: the corner opposite it is i + 2
    corners, bary = turn(corners, first), turn(bary, first)
    edge = np.take_along_axis(edge, (first[:, None] + np.arange(3)) % 3, axis=1)  # now 01, 12, 20; 12 the shortest
    rows = np.clip(np.ceil(np.maximum(edge[:, 0], edge[:, 2]) / space), 1, MAX_SPLIT).astype(int)
    across = np.clip(np.ceil(edge[:, 1] / space), 1, MAX_SPLIT).astype(int)
    key = rows * (MAX_SPLIT + 1) + across
    for k in np.unique(key):
        picked = np.flatnonzero(key == k)
        weights = pattern(int(k) // (MAX_SPLIT + 1), int(k) % (MAX_SPLIT + 1))
        step = max(1, CHUNK // len(weights))
        for start in range(0, len(picked), step):
            part = picked[start:start + step]
            yield (np.einsum("kc,tcd->tkd", weights, corners[part]).reshape(-1, 3),
                   np.repeat(face[part], len(weights)),
                   np.einsum("kc,tcd->tkd", weights, bary[part]).reshape(-1, 3))


def normals(s: Surface) -> np.ndarray:
    """Unit normal of every face, turned to point up."""
    c = s.points[s.faces]
    n = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    n[n[:, 2] < 0] *= -1
    return n


def colors(s: Surface, face: np.ndarray, weights: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """RGB (uint8) of sample points: the face's color, its texture, and a fixed light from above."""
    rgb = np.broadcast_to(np.atleast_2d(s.color), (len(s.faces), 3))[face].astype(np.float32)
    if s.textures and s.uv is not None:
        uv = np.einsum("kc,kcd->kd", weights, s.uv[s.faces[face]])
        which = s.face_texture[face]
        for index, (image, scale, offset) in enumerate(s.textures):
            m = which == index
            if not m.any():
                continue
            at = uv[m] * (1.0 if scale is None else scale) + (0.0 if offset is None else offset)
            at -= np.floor(at)  # textures repeat
            h, w = image.shape[:2]
            col = np.minimum((at[:, 0] * w).astype(int), w - 1)
            row = np.minimum((at[:, 1] * h).astype(int), h - 1)
            rgb[m] *= image[row, col] / 255.0
    rgb *= (0.55 + 0.45 * np.clip(normal[face] @ LIGHT, 0.0, 1.0))[:, None]
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def nearest(pixel: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Of points that fall into pixels, the indices of the nearest one of every pixel."""
    order = np.lexsort((depth, pixel))
    first = np.ones(len(order), bool)
    first[1:] = pixel[order[1:]] != pixel[order[:-1]]
    return order[first]


class Canvas:
    """An image with a depth per pixel: of all the points put into a pixel, the nearest stays."""

    def __init__(self, width: int, height: int, background=(24, 26, 30)):
        self.width, self.height = width, height
        self.depth = np.full(width * height, np.inf, dtype=np.float32)
        self.rgb = np.empty((width * height, 3), dtype=np.uint8)
        self.rgb[:] = background

    def put(self, pixel: np.ndarray, depth: np.ndarray, rgb) -> None:
        """`rgb(indices)` gives the colors of the points that turn out to be needed."""
        won = nearest(pixel, depth)
        won = won[depth[won] < self.depth[pixel[won]]]
        if len(won):
            self.depth[pixel[won]] = depth[won]
            self.rgb[pixel[won]] = rgb(won)

    def image(self) -> np.ndarray:
        return self.rgb.reshape(self.height, self.width, 3)


class Camera:
    """A pinhole camera at `eye` looking at `target`, Z up, as sanity_check_render.py spawns it."""

    NEAR = 0.1  # m

    def __init__(self, eye, target, width: int, height: int, focal_mm: float = 15.0, aperture_mm: float = 20.955):
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        self.eye, self.width, self.height = eye, width, height
        self.forward = (target - eye) / np.linalg.norm(target - eye)
        up = np.array([0.0, 1.0, 0.0]) if abs(self.forward[2]) > 0.999 else np.array([0.0, 0.0, 1.0])
        self.right = np.cross(self.forward, up)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        self.scale = focal_mm / aperture_mm * width  # pixels per unit of x / depth
        self.canvas = Canvas(width, height)

    def view(self, points: np.ndarray) -> tuple:
        """(right, up, depth) of world points, in meters from the camera."""
        rel = points - self.eye
        return rel @ self.right, rel @ self.up, rel @ self.forward

    def keep(self, corners: np.ndarray) -> np.ndarray:
        """Triangles that may be in the picture: not wholly behind the camera or beyond one of its edges."""
        x, y, depth = self.view(corners)
        half_w, half_h = self.width / 2 / self.scale, self.height / 2 / self.scale
        d = np.maximum(depth, self.NEAR)
        return ((depth > self.NEAR).any(axis=1) & ~(x > half_w * d).all(axis=1) & ~(x < -half_w * d).all(axis=1)
                & ~(y > half_h * d).all(axis=1) & ~(y < -half_h * d).all(axis=1))

    def spacing(self, corners: np.ndarray) -> np.ndarray:
        """Three quarters of what a pixel covers at the triangle's nearest corner."""
        depth = self.view(corners)[2].min(axis=1)
        return 0.75 * np.maximum(depth, self.NEAR) / self.scale

    def add(self, s: Surface) -> None:
        normal = normals(s)
        for points, face, weights in samples(s, self.spacing, self.keep):
            x, y, depth = self.view(points)
            ok = depth > self.NEAR
            depth = np.where(ok, depth, 1.0)
            col = np.floor(self.width / 2 + self.scale * x / depth).astype(np.int64)
            row = np.floor(self.height / 2 - self.scale * y / depth).astype(np.int64)
            ok &= (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
            if ok.any():
                face, weights = face[ok], weights[ok]
                self.canvas.put(row[ok] * self.width + col[ok], depth[ok],
                                lambda won, f=face, w=weights: colors(s, f[won], w[won], normal))
