"""
UrbanVerse-100K

Developer: Mingxuan Liu
Affiliation: University of Trento
Email: mingxuan.liu@unitn.it
Website: https://oatmealliu.github.io/

Description:
Utility package for accessing, exploring, and using the UrbanVerse-100K asset database.

Copyright (c) 2026, Mingxuan Liu.
"""

"""
USD-to-GLB conversion utility for vegetation asset previews.

Uses Pixar's OpenUSD (``pxr``) to read geometry and material bindings,
then ``trimesh`` to assemble and export a GLB file.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_TEXTURE_2D_RE = re.compile(
    r'(\w+)\s*:\s*texture_2d\(\s*"([^"]+)"',
)


def convert_usd_to_glb(usd_path: Path, glb_path: Path) -> Path:
    """Convert a USD file to GLB for Three.js preview.

    Reads mesh geometry, UVs, and diffuse/normal textures from the USD
    stage and exports a self-contained GLB via trimesh.

    Args:
        usd_path: Path to the source ``World0.usd`` file.
        glb_path: Destination path for the output ``.glb`` file.

    Returns:
        *glb_path* on success.

    Raises:
        ImportError: If ``pxr`` or ``trimesh`` is not installed.
        FileNotFoundError: If *usd_path* does not exist.
    """
    try:
        from pxr import Usd, UsdGeom, UsdShade, Sdf
    except ImportError:
        raise ImportError(
            "The 'pxr' (OpenUSD) package is required for USD→GLB conversion.\n"
            "Install it via your NVIDIA Isaac Sim / Omniverse environment."
        ) from None

    try:
        import trimesh
        import trimesh.visual
        import trimesh.visual.texture
    except ImportError:
        raise ImportError(
            "trimesh is required for GLB export: pip install trimesh"
        ) from None

    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path))
    scene = trimesh.Scene()
    asset_dir = usd_path.parent

    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.GetTypeName() != "Mesh":
            continue

        mesh_geom = UsdGeom.Mesh(prim)
        pts_raw = mesh_geom.GetPointsAttr().Get()
        fvi_raw = mesh_geom.GetFaceVertexIndicesAttr().Get()
        fvc_raw = mesh_geom.GetFaceVertexCountsAttr().Get()

        if not pts_raw or not fvi_raw or not fvc_raw:
            continue

        pts = np.array(pts_raw, dtype=np.float64)
        fvi = np.array(fvi_raw, dtype=np.int64)
        fvc = np.array(fvc_raw, dtype=np.int64)

        faces = _triangulate(fvi, fvc)
        if faces is None or len(faces) == 0:
            continue

        uv_data = _get_uvs(prim)
        diffuse_path, normal_path = _get_texture_paths(prim, asset_dir, Sdf, UsdShade)

        visual = _build_visual(uv_data, diffuse_path, normal_path, trimesh)

        tm = trimesh.Trimesh(vertices=pts, faces=faces, visual=visual)
        scene.add_geometry(tm, node_name=str(prim.GetPath()))

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_data = scene.export(file_type="glb")
    glb_path.write_bytes(glb_data)

    return glb_path


def _triangulate(
    fvi: np.ndarray, fvc: np.ndarray
) -> Optional[np.ndarray]:
    """Fan-triangulate faces of arbitrary vertex count."""
    tris: List[List[int]] = []
    idx = 0
    for c in fvc:
        c = int(c)
        if c < 3:
            idx += c
            continue
        for i in range(1, c - 1):
            tris.append([int(fvi[idx]), int(fvi[idx + i]), int(fvi[idx + i + 1])])
        idx += c
    if not tris:
        return None
    return np.array(tris, dtype=np.int64)


def _get_uvs(prim) -> Optional[np.ndarray]:
    """Extract the 'st' primvar (UV coordinates) from a mesh prim."""
    from pxr import UsdGeom

    pvapi = UsdGeom.PrimvarsAPI(prim)
    st = pvapi.GetPrimvar("st")
    if st and st.IsDefined():
        data = st.Get()
        if data is not None and len(data) > 0:
            return np.array(data, dtype=np.float64)
    return None


def _get_texture_paths(
    prim, asset_dir: Path, Sdf, UsdShade
) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve diffuse and normal texture paths from a mesh's bound material.

    Tries direct shader input asset-paths first.  When none are found
    (common for assets authored in NVIDIA MDL), falls back to parsing
    the external ``.mdl`` file referenced by the shader for
    ``texture_2d("...")`` declarations.
    """
    binding = UsdShade.MaterialBindingAPI(prim)
    mat, _ = binding.ComputeBoundMaterial()
    if not mat:
        return None, None

    diffuse: Optional[Path] = None
    normal: Optional[Path] = None

    for child in mat.GetPrim().GetAllChildren():
        if child.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(child)

        # --- Strategy 1: direct asset-path inputs on the shader ---
        for inp in shader.GetInputs():
            val = inp.Get()
            if not isinstance(val, Sdf.AssetPath):
                continue
            resolved = val.resolvedPath or val.path
            if not resolved:
                continue
            p = Path(resolved)
            if not p.is_absolute():
                p = asset_dir / p
            if not p.exists():
                continue

            name = inp.GetBaseName().lower()
            if "diffuse" in name or "albedo" in name or "basecolor" in name:
                diffuse = diffuse or p
            elif "normal" in name:
                normal = normal or p

        if diffuse:
            continue

        # --- Strategy 2: parse the referenced .mdl file ---
        mdl_asset = shader.GetSourceAsset(sourceType="mdl")
        if not mdl_asset:
            continue
        mdl_resolved = mdl_asset.resolvedPath or mdl_asset.path
        if not mdl_resolved:
            continue
        mdl_path = Path(mdl_resolved)
        if not mdl_path.is_absolute():
            mdl_path = asset_dir / mdl_path
        if not mdl_path.exists():
            continue

        d, n = _parse_mdl_textures(mdl_path)
        diffuse = diffuse or d
        normal = normal or n

    return diffuse, normal


def _parse_mdl_textures(
    mdl_path: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Extract diffuse and normal texture paths from a ``.mdl`` file.

    Uses the MDL parameter name (e.g. ``diffuse_texture:``) to classify
    textures, with filename keywords as a fallback.
    """
    try:
        content = mdl_path.read_text(errors="replace")
    except OSError:
        return None, None

    diffuse: Optional[Path] = None
    normal: Optional[Path] = None
    mdl_dir = mdl_path.parent

    for param_name, ref in _TEXTURE_2D_RE.findall(content):
        p = (mdl_dir / ref).resolve()
        if not p.exists():
            continue

        pname = param_name.lower()
        fname = p.name.lower()

        is_diffuse = (
            "diffuse" in pname or "albedo" in pname
            or "diffuse" in fname or "albedo" in fname or "basecolor" in fname
        )
        is_normal = "normal" in pname or "normal" in fname

        if is_diffuse:
            diffuse = diffuse or p
        elif is_normal:
            normal = normal or p

    return diffuse, normal


def _build_visual(uv_data, diffuse_path, normal_path, trimesh_mod):
    """Build a trimesh TextureVisuals from UV data and texture paths."""
    if uv_data is None or diffuse_path is None:
        return None

    try:
        from PIL import Image

        img = Image.open(diffuse_path)
        material = trimesh_mod.visual.texture.SimpleMaterial(image=img)
        return trimesh_mod.visual.TextureVisuals(uv=uv_data, material=material)
    except Exception:
        return None
