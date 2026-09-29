"""Read a scene's geometry without Isaac Sim: every visible mesh, in world coordinates.

usd-core opens the scene's USD layers but has no glTF plugin, so the .glb models the scene
places (as payloads) stay empty. They are read here and placed the way Isaac Sim 5 places them:

  - the prim with the payload is the .glb's root node;
  - every other node is a child prim named after the node, nested as in the .glb;
  - a node's mesh is a child prim of the node, named after the mesh (a skinned mesh: of the
    root joint of its skeleton);
  - a prim's transform is what the scene authors on it, else what the .glb gives the node;
  - what the scene switches off (active = false, visibility = invisible) is left out.

docs/01_topdown.md tells how these rules were found and checked.
"""
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from .glb import Glb

GROUND_ROOTS = re.compile(r"^/World/(Big_Block|crosswalk)", re.I)  # the road network the scenes are built on
# Ground kinds by the name of the road piece. The three sidewalk zones run from the curb outwards.
GROUND_KINDS = {"lane": "road", "whiteline": "road_marking", "yellowline": "road_marking", "crosswalk": "crosswalk",
                "nearbuffer": "sidewalk", "nearroad": "sidewalk", "sidewalk": "sidewalk"}
# The ground's materials come from a library inside Isaac Sim, so the map colors ground by its name.
GROUND_COLORS = {"lane": (0.30, 0.30, 0.32), "whiteline": (0.92, 0.92, 0.90), "yellowline": (0.93, 0.76, 0.18),
                 "crosswalk": (0.95, 0.95, 0.95), "nearbuffer": (0.60, 0.58, 0.54), "nearroad": (0.66, 0.63, 0.58),
                 "sidewalk": (0.72, 0.69, 0.64), None: (0.55, 0.55, 0.50)}


@dataclass
class Surface:
    """Triangles of one mesh, in world coordinates."""
    path: str  # the prim the mesh is on
    kind: str  # a ground kind (road, sidewalk, ...) or "object"
    points: np.ndarray  # Nx3 float64
    faces: np.ndarray  # Mx3 int
    uv: np.ndarray | None = None  # Nx2, v down (row = v * height)
    color: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8]))  # RGB 0..1, per face or one
    textures: list = field(default_factory=list)  # [(HxWx3 uint8, uv scale, uv offset)]; a texture multiplies color
    face_texture: np.ndarray | None = None  # per face: index into textures, -1 for none


@dataclass
class Source:
    """A mesh of the scene that has not been read yet."""
    path: str
    kind: str
    box: np.ndarray  # 2x3: the lowest and the highest corner of its box in the world
    read: object  # () -> Surface, or None if the mesh turns out to be empty

    def touches(self, box) -> bool:
        """Whether the mesh may reach into `box` = (x_min, y_min, x_max, y_max)."""
        return bool(self.box[1, 0] >= box[0] and self.box[0, 0] <= box[2]
                    and self.box[1, 1] >= box[1] and self.box[0, 1] <= box[3])


@contextmanager
def quiet_stderr():
    """Mute USD's C++ diagnostics: every .glb payload fails to open outside Isaac Sim."""
    sys.stderr.flush()
    saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def prim_name(name: str, what: str = "node") -> str:
    """The prim name Isaac Sim 5 gives a glTF node or mesh: every character that cannot be in a
    prim name becomes '_', and a leading digit is replaced by 'node_' or 'mesh_'."""
    name = "".join(c if c.isascii() and (c.isalnum() or c == "_") else "_" for c in name)
    return f"{what}_{name[1:]}" if not name or name[0].isdigit() else name


class Scene:
    def __init__(self, usd):
        self.usd = Path(usd)
        with quiet_stderr():
            self.stage = Usd.Stage.Open(str(self.usd), Usd.Stage.LoadAll)
        if self.stage is None:
            raise SystemExit(f"cannot open {self.usd}")
        if UsdGeom.GetStageUpAxis(self.stage) != "Z" or abs(UsdGeom.GetStageMetersPerUnit(self.stage) - 1.0) > 1e-6:
            raise SystemExit(f"{self.usd}: expected a Z-up stage in meters")
        self._world = {}
        self._glb_local = {}  # prim path -> the transform the .glb gives that node
        self.models = []  # (payload prim, Glb, [(mesh prim path, node index)])
        self.notes = []  # what a reader of the map should know
        for prim in self.all_prims():
            if prim.IsActive() and prim.HasAuthoredPayloads():
                self._add_model(prim)
        self._model_prims = set(self._glb_local) | {path for _, _, meshes in self.models for path, _ in meshes}
        self._sources = None

    def all_prims(self):
        """Every prim: also inactive ones, those only a model defines, and those inside instances
        (scene_10's crossing is an instanceable road block, which a plain traversal skips)."""
        return self.stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))

    # ── the .glb models ──────────────────────────────────────────────────────────────────────

    def _add_model(self, prim) -> None:
        layer = next(s.layer for s in prim.GetPrimStack() if s.HasInfo("payload"))
        for item in prim.GetMetadata("payload").GetAddedOrExplicitItems():
            path = layer.ComputeAbsolutePath(item.assetPath)
            if not path.lower().endswith(".glb"):
                continue  # a USD payload: usd-core has composed it
            if not os.path.isfile(path):
                self.notes.append(f"missing model {path}")
                continue
            glb = Glb(path)
            if glb.json.get("skins"):
                self.notes.append(f"{prim.GetPath()}: skinned model, drawn in its bind pose")
            self.models.append((prim, glb, self._place(glb, str(prim.GetPath()))))

    def _place(self, glb: Glb, payload: str) -> list:
        """Give the model's nodes and meshes their prims under the prim at `payload`.
        Returns [(mesh prim path, node index)]."""
        taken, at = set(), {}

        def child(parent: str, name: str) -> str:
            n = 0
            while (path := f"{parent}/{name}{n or ''}") in taken:  # siblings of the same name
                n += 1
            taken.add(path)
            return path

        def walk(node: int) -> None:
            for c in glb.children(node):
                at[c] = child(at[node], prim_name(glb.name(c)))
                self._glb_local[at[c]] = glb.local_matrix(c)
                walk(c)

        for root in glb.roots():
            at[root] = payload
            self._glb_local[payload] = glb.local_matrix(root)
            walk(root)
        meshes = []
        for node in sorted(at):
            nd = glb.nodes[node]
            if "mesh" in nd:
                skeleton = glb.json["skins"][nd["skin"]].get("skeleton") if "skin" in nd else None
                name = glb.json["meshes"][nd["mesh"]].get("name", f"mesh_{nd['mesh']}")
                meshes.append((child(at.get(skeleton, at[node]), prim_name(name, "mesh")), node))
        return meshes

    # ── transforms and visibility, for prims of the stage and of the models alike ───────────

    def world(self, path: str) -> np.ndarray:
        """Local-to-world of the prim at `path`, as a 4x4 for row vectors (p' = p @ M)."""
        if path in ("/", ""):
            return np.eye(4)
        if path not in self._world:
            prim = self.stage.GetPrimAtPath(path)
            # By what is authored, not by the prim's type: the payload prims have none in usd-core.
            if prim and prim.GetAttribute("xformOpOrder").HasAuthoredValue():
                local = np.array(UsdGeom.Xformable(prim).GetLocalTransformation()).reshape(4, 4)
            else:
                local = self._glb_local.get(path, np.eye(4))
            self._world[path] = local @ self.world(path.rsplit("/", 1)[0])
        return self._world[path]

    def shown(self, path: str) -> bool:
        """Whether the prim at `path` is there, active and visible, by what the scene authors on it
        and above it. A prim the scene only overrides is there if a model provides it."""
        while path not in ("/", ""):
            prim = self.stage.GetPrimAtPath(path)
            if prim:
                if not prim.HasDefiningSpecifier() and path not in self._model_prims:
                    return False
                if not prim.IsActive():
                    return False
                vis = prim.GetAttribute("visibility")
                if vis and vis.HasAuthoredValue() and vis.Get() == UsdGeom.Tokens.invisible:
                    return False
                purpose = prim.GetAttribute("purpose")
                if purpose and purpose.HasAuthoredValue() and purpose.Get() in ("guide", "proxy"):
                    return False
            path = path.rsplit("/", 1)[0]
        return True

    def ground_name(self, path: str):
        """The name of the road piece the prim at `path` belongs to: 'lane', 'sidewalk', ... None if it
        is ground of no known name, 'object' if it is not part of the road network."""
        if not GROUND_ROOTS.match(path):
            return "crosswalk" if "crosswalk" in path.lower() else "object"
        names = (re.sub(r"[_\d]+$", "", part) for part in path.lower().split("/")[2:])
        return next((name for name in names if name in GROUND_KINDS), None)

    def kind(self, path: str) -> str:
        name = self.ground_name(path)
        return name if name == "object" else GROUND_KINDS.get(name, "ground")

    # ── surfaces ─────────────────────────────────────────────────────────────────────────────

    def sources(self) -> list:
        """Every visible mesh of the scene: the scene's own meshes, then the models'."""
        if self._sources is None:
            self._sources = list(self._stage_sources()) + list(self._model_sources())
        return self._sources

    def surfaces(self, box=None):
        """The meshes as Surfaces; with `box` = (x_min, y_min, x_max, y_max), those that may reach into it."""
        for source in self.sources():
            if box is None or source.touches(box):
                surface = source.read()
                if surface is not None:
                    yield surface

    def _stage_sources(self):
        models = {str(prim.GetPath()): glb for prim, glb, _ in self.models}
        for prim in self.all_prims():
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasDefiningSpecifier():
                continue
            path = str(prim.GetPath())
            if not self.shown(path):
                continue
            extent = UsdGeom.Mesh(prim).GetExtentAttr().Get()
            if not extent:
                points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if not points:
                    continue
                points = np.asarray(points)
                extent = [points.min(axis=0), points.max(axis=0)]
            glb = next((g for p, g in models.items() if path.startswith(p + "/")), None)
            yield Source(path, self.kind(path), world_box(extent, self.world(path)),
                         lambda prim=prim, glb=glb: self._read_stage_mesh(prim, glb))

    def _read_stage_mesh(self, prim, glb):
        path = str(prim.GetPath())
        mesh = UsdGeom.Mesh(prim)
        points, counts = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get()
        if not points or not counts:
            return None
        points, counts = np.asarray(points, dtype=np.float64), np.asarray(counts)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        corners, face_of = triangulate(counts, indices, len(points))
        if not len(corners):
            return None
        s = Surface(path, self.kind(path), transform(points, self.world(path)), indices[corners])
        if glb is not None:  # a copy of a model's mesh that the scene keeps in its own layers
            self._copy_colors(s, prim, glb, counts, indices, corners, face_of)
        elif s.kind == "object":
            s.color = self._material_color(prim)
        else:
            s.color = np.array(GROUND_COLORS[self.ground_name(path)])
        return s

    def _copy_colors(self, s: Surface, prim, glb: Glb, counts, indices, corners, face_of) -> None:
        names = {prim_name(m.get("name", "")): i for i, m in enumerate(glb.json.get("materials", []))}
        st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if st and st.Get() is not None:
            uv = np.asarray(st.Get(), dtype=np.float32)
            if st.GetInterpolation() == UsdGeom.Tokens.faceVarying and len(uv) == len(indices):
                # one texture coordinate per face corner: give every corner its own point
                s.points, s.faces = s.points[s.faces].reshape(-1, 3), np.arange(corners.size).reshape(-1, 3)
                uv = uv[corners].reshape(-1, 2)
            if len(uv) == len(s.points):
                s.uv = np.c_[uv[:, 0], 1.0 - uv[:, 1]]  # USD has v up, images have rows down
        material = np.full(len(s.faces), names.get(bound_material(prim), -1))
        for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim)):
            picked = np.zeros(len(counts), bool)
            picked[np.asarray(subset.GetIndicesAttr().Get() or [], dtype=int)] = True
            material[picked[face_of]] = names.get(bound_material(subset.GetPrim()), -1)
        colors(s, glb, material)

    def _material_color(self, prim) -> np.ndarray:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            for shader in Usd.PrimRange(material.GetPrim()):
                for name in ("inputs:diffuse_color_constant", "inputs:diffuseColor", "inputs:base_color",
                             "inputs:diffuse_tint"):
                    attr = shader.GetAttribute(name)
                    if attr and attr.HasAuthoredValue() and attr.Get() is not None:
                        return np.asarray(attr.Get(), dtype=np.float64)[:3]
        return np.array([0.6, 0.6, 0.6])

    def _model_sources(self):
        for _, glb, meshes in self.models:
            for path, node in meshes:
                extent = glb.extent(node)
                if extent is not None and self.shown(path):
                    yield Source(path, self.kind(path), world_box(extent, self.world(path)),
                                 lambda glb=glb, path=path, node=node: self._read_model_mesh(glb, path, node))

    def _read_model_mesh(self, glb: Glb, path: str, node: int):
        parts = glb.mesh(node)
        if not parts:
            return None
        offsets = np.cumsum([0] + [len(p["points"]) for p in parts])
        points = np.concatenate([p["points"] for p in parts]).astype(np.float64)
        faces = np.concatenate([p["faces"] + o for p, o in zip(parts, offsets[:-1], strict=True)])
        s = Surface(path, self.kind(path), transform(points, self.world(path)), faces)
        if all(p["uv"] is not None for p in parts):
            s.uv = np.concatenate([p["uv"] for p in parts])
        material = np.concatenate([np.full(len(p["faces"]), -1 if p["material"] is None else p["material"])
                                   for p in parts])
        colors(s, glb, material)
        return s

    # ── what else the map needs ──────────────────────────────────────────────────────────────

    def collision_planes(self) -> list:
        """Heights of the infinite ground planes PhysX collides with (UsdGeom.Plane with CollisionAPI)."""
        heights = []
        for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Plane) and prim.HasAPI(UsdPhysics.CollisionAPI) \
                    and UsdGeom.Plane(prim).GetAxisAttr().Get() == "Z":
                heights.append(float(self.world(str(prim.GetPath()))[3, 2]))
        return heights

    def unmatched_overrides(self) -> list:
        """Scene overrides of a model's node that land on none of the prims the model has here.
        Either Isaac Sim loses them too, or it names that prim differently than this reader."""
        lost = []
        for prim, _, _ in self.models:
            for q in Usd.PrimRange(prim, Usd.PrimAllPrimsPredicate):
                path = str(q.GetPath())
                if not q.HasDefiningSpecifier() and path not in self._model_prims and q.GetName() != "Looks" \
                        and str(q.GetParent().GetPath()) in self._glb_local:
                    lost.append(path)
        return lost


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3] + matrix[3, :3]


def world_box(extent, matrix: np.ndarray) -> np.ndarray:
    """The box in the world around a box in a mesh's own coordinates."""
    lo, hi = np.asarray(extent[0], dtype=np.float64), np.asarray(extent[1], dtype=np.float64)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    corners = transform(corners, matrix)
    return np.stack([corners.min(axis=0), corners.max(axis=0)])


def corner_indices(counts: np.ndarray) -> np.ndarray:
    """For polygons with `counts` corners each, the corner numbers of their triangles (a fan per polygon)."""
    start = np.cumsum(counts) - counts
    fans = [np.stack([start[counts >= k + 3], start[counts >= k + 3] + k + 1, start[counts >= k + 3] + k + 2], axis=1)
            for k in range(int(counts.max()) - 2)] if len(counts) else []
    return np.concatenate(fans) if fans else np.zeros((0, 3), int)


def triangulate(counts: np.ndarray, indices: np.ndarray, n_points: int) -> tuple:
    """(the triangles of the polygons, as corner numbers; the polygon each triangle came from)."""
    corners = corner_indices(counts)
    corners = corners[(corners < len(indices)).all(axis=1)]
    corners = corners[(indices[corners] < n_points).all(axis=1)]
    return corners, np.searchsorted(np.cumsum(counts), corners[:, 0], side="right")


def bound_material(prim) -> str:
    """Name of the material bound to the prim itself, '' if none."""
    targets = prim.GetRelationship("material:binding").GetTargets() if prim.HasRelationship("material:binding") else []
    return targets[0].name if targets else ""


def colors(s: Surface, glb: Glb, material: np.ndarray) -> None:
    """Color the surface's faces by their glTF material: base color, and base color texture."""
    s.color = np.full((len(s.faces), 3), 0.8)
    s.face_texture = np.full(len(s.faces), -1)
    for index in np.unique(material):
        m = glb.material(None if index < 0 else int(index))
        picked = material == index
        s.color[picked] = m["color"]
        if m["texture"] is not None and s.uv is not None:
            s.face_texture[picked] = len(s.textures)
            s.textures.append((m["texture"], m.get("uv_scale"), m.get("uv_offset")))
        elif m["texture"] is not None:
            s.color[picked] = m["color"] * m["texture"].reshape(-1, 3).mean(axis=0) / 255.0
