"""A minimal .glb (binary glTF 2.0) reader: the triangles of every node, with their colors.

Only what a top-down map needs: positions, triangle indices, texture coordinates, and the
base color (factor and texture) of each material. No skinning, morphing or animation: a
skinned mesh is read in its bind pose.
"""
import io
import json
import struct
from pathlib import Path

import numpy as np

MAX_TEXTURES = 64  # textures a .glb keeps in memory, of 256 x 256 pixels at most
COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class Glb:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            magic, version, _ = struct.unpack("<4sII", f.read(12))
            if magic != b"glTF" or version != 2:
                raise ValueError(f"{self.path}: not a glTF 2.0 binary")
            length, kind = struct.unpack("<I4s", f.read(8))
            if kind != b"JSON":
                raise ValueError(f"{self.path}: the first chunk is not JSON")
            self.json = json.loads(f.read(length))
            header = f.read(8)
            self.bin_start = f.tell() if len(header) == 8 and header[4:] == b"BIN\0" else None
        self.nodes = self.json.get("nodes", [])
        self._textures = {}  # the last MAX_TEXTURES that were read

    # ── the node tree ────────────────────────────────────────────────────────────────────────

    def roots(self) -> list:
        """Indices of the root nodes of the default scene."""
        scenes = self.json.get("scenes") or [{"nodes": list(range(len(self.nodes)))}]
        return scenes[self.json.get("scene", 0)].get("nodes", [])

    def children(self, node: int) -> list:
        return self.nodes[node].get("children", [])

    def name(self, node: int) -> str:
        return self.nodes[node].get("name", f"node_{node}")

    def local_matrix(self, node: int) -> np.ndarray:
        """The node's transform to its parent as a 4x4 for row vectors (p' = p @ M), as USD has it."""
        nd = self.nodes[node]
        if "matrix" in nd:
            return np.array(nd["matrix"], dtype=np.float64).reshape(4, 4)  # column-major = row-vector form
        m = np.eye(4)
        x, y, z, w = nd.get("rotation", (0.0, 0.0, 0.0, 1.0))
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
                        [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
                        [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]])
        m[:3, :3] = np.asarray(nd.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)[:, None] * rot
        m[3, :3] = nd.get("translation", (0.0, 0.0, 0.0))
        return m

    # ── geometry ─────────────────────────────────────────────────────────────────────────────

    def accessor(self, index: int) -> np.ndarray:
        acc = self.json["accessors"][index]
        dtype, width = np.dtype(COMPONENT[acc["componentType"]]), WIDTH[acc["type"]]
        if "bufferView" not in acc:
            data = np.zeros((acc["count"], width), dtype)
        else:
            view = self.json["bufferViews"][acc["bufferView"]]
            if view.get("buffer", 0) != 0 or self.bin_start is None:
                raise ValueError(f"{self.path}: data outside the binary chunk")
            start = self.bin_start + view.get("byteOffset", 0) + acc.get("byteOffset", 0)
            item = dtype.itemsize * width
            stride = view.get("byteStride") or item
            with open(self.path, "rb") as f:
                f.seek(start)
                raw = np.frombuffer(f.read(stride * (acc["count"] - 1) + item), dtype=np.uint8)
            if stride == item:
                data = raw.view(dtype).reshape(acc["count"], width)
            else:  # interleaved with other attributes
                rows = np.lib.stride_tricks.as_strided(raw, (acc["count"], item), (stride, 1))
                data = np.ascontiguousarray(rows).view(dtype).reshape(acc["count"], width)
        if "sparse" in acc:
            raise ValueError(f"{self.path}: sparse accessors are not supported")
        if acc.get("normalized") and dtype.kind in "iu":
            data = np.maximum(data / np.iinfo(dtype).max, -1.0).astype(np.float32)
        return data

    def extent(self, node: int):
        """(lowest corner, highest corner) of the box around the node's mesh, in the mesh's own
        coordinates; None if the node has no triangles. Read from the file's index, not its data."""
        nd = self.nodes[node]
        if "mesh" not in nd:
            return None
        boxes = [self.json["accessors"][p["attributes"]["POSITION"]]
                 for p in self.json["meshes"][nd["mesh"]]["primitives"]
                 if p.get("mode", 4) == 4 and "POSITION" in p["attributes"]]
        if not boxes:
            return None
        if not all("min" in b and "max" in b for b in boxes):  # the format requires them; read if missing
            points = np.concatenate([p["points"] for p in self.mesh(node)])
            return points.min(axis=0), points.max(axis=0)
        return np.min([b["min"] for b in boxes], axis=0), np.max([b["max"] for b in boxes], axis=0)

    def mesh(self, node: int) -> list:
        """The node's mesh as a list of parts, one per glTF primitive:
        {"points": Nx3 float32, "faces": Mx3 int, "uv": Nx2 float32 or None, "material": index or None}."""
        nd = self.nodes[node]
        if "mesh" not in nd:
            return []
        parts = []
        for prim in self.json["meshes"][nd["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4 or "POSITION" not in prim["attributes"]:
                continue  # points and lines draw nothing on a map
            points = self.accessor(prim["attributes"]["POSITION"]).astype(np.float32)
            if "indices" in prim:
                faces = self.accessor(prim["indices"]).astype(np.int64).reshape(-1, 3)
            else:
                faces = np.arange(len(points) // 3 * 3, dtype=np.int64).reshape(-1, 3)
            uv = self.accessor(prim["attributes"]["TEXCOORD_0"]).astype(np.float32) \
                if "TEXCOORD_0" in prim["attributes"] else None
            parts.append({"points": points, "faces": faces, "uv": uv, "material": prim.get("material")})
        return parts

    # ── color ────────────────────────────────────────────────────────────────────────────────

    def material(self, index) -> dict:
        """{"name", "color": RGB 0..1, "texture": HxWx3 uint8 or None, "uv_scale", "uv_offset"} of a material."""
        if index is None:
            return {"name": None, "color": np.array([0.8, 0.8, 0.8]), "texture": None}
        mat = self.json["materials"][index]
        pbr = mat.get("pbrMetallicRoughness", {})
        factor, tex = pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0)), pbr.get("baseColorTexture")
        old = mat.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness")
        if old:
            factor, tex = old.get("diffuseFactor", factor), old.get("diffuseTexture", tex)
        out = {"name": mat.get("name"), "color": np.asarray(factor[:3], dtype=np.float64), "texture": None}
        if tex is not None:
            out["texture"] = self.texture(tex["index"])
            move = tex.get("extensions", {}).get("KHR_texture_transform", {})
            if move.get("rotation"):
                out["texture"] = None  # rare; fall back to the texture's mean color
                out["color"] = out["color"] * self.texture(tex["index"]).reshape(-1, 3).mean(axis=0) / 255.0
            out["uv_scale"] = np.asarray(move.get("scale", (1.0, 1.0)), dtype=np.float32)
            out["uv_offset"] = np.asarray(move.get("offset", (0.0, 0.0)), dtype=np.float32)
        return out

    def texture(self, index: int, size: int = 256):
        """A texture's image as HxWx3 uint8, at most `size` pixels wide or high; None if it cannot be read."""
        if index not in self._textures:
            if len(self._textures) >= MAX_TEXTURES:
                self._textures.pop(next(iter(self._textures)))
            self._textures[index] = self._read_texture(index, size)
        return self._textures[index]

    def _read_texture(self, index: int, size: int):
        from PIL import Image

        tex = self.json["textures"][index]
        source = tex.get("source", tex.get("extensions", {}).get("KHR_texture_basisu", {}).get("source"))
        if source is None:
            return None
        image = self.json["images"][source]
        try:
            if "bufferView" in image:
                view = self.json["bufferViews"][image["bufferView"]]
                with open(self.path, "rb") as f:
                    f.seek(self.bin_start + view.get("byteOffset", 0))
                    im = Image.open(io.BytesIO(f.read(view["byteLength"])))
            else:
                im = Image.open(self.path.parent / image["uri"])
            im.draft("RGB", (size, size))  # JPEG: decode at a reduced size
            im = im.convert("RGB")
            im.thumbnail((size, size))
            return np.asarray(im)
        except Exception:  # noqa: BLE001 - an unreadable texture costs the color, not the map
            return None
