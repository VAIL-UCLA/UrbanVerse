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
urbanverse_asset.viewer
────────────────────────
Browser-based interactive viewers for UrbanVerse dataset assets.

All viewer functions follow the same three-step pattern:
  1. Download missing files to the local cache.
  2. Spin up a localhost HTTP server serving the cache directory.
  3. Generate a self-contained HTML page and open it in the default browser.

The HTTP server runs in a daemon thread and exits automatically when Python
exits. It is reused across calls in the same session.

Usage::

    import urbanverse_asset as uva

    uva.viewer.object_distribution()                      # interactive dataset sunburst
    uva.viewer.object_show(["uid1", "uid2"])               # GLB + thumbnail + renders + annotation
    uva.viewer.object_assets(["uid1"])                     # 3D GLB viewer only
    uva.viewer.object_thumbnails(["uid1", "uid2"])         # thumbnail grid
    uva.viewer.object_renders(["uid1"], angles=(0,90))     # render image strip
    uva.viewer.object_annotations(["uid1"])                # annotation table
    uva.viewer.sky_show()                                  # IBL HDRI preview
    uva.viewer.road_show(["Concrete007_2K_PNG"])
"""

import json
import os
import re
import socket
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import _core
from . import sky as _sky
from . import road as _road
from . import sidewalk as _sidewalk
from . import terrain as _terrain
from . import plant as _plant
from . import shrub as _shrub
from . import tree as _tree

# ── Template directory ────────────────────────────────────────────────────────
_TEMPLATES_DIR = Path(__file__).parent / "_viewer"

# ── Module-level HTTP server state ────────────────────────────────────────────
_server_instance: Optional[HTTPServer] = None
_server_port: int = 0
_server_dir: Optional[Path] = None
_server_lock = threading.Lock()

_RENDER_ANGLES = (0, 90, 180, 270)


# ══════════════════════════════════════════════════════════════════════════════
# Internal: server management
# ══════════════════════════════════════════════════════════════════════════════

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _ensure_server() -> Tuple[str, int]:
    """Return (host, port) of a running localhost server serving the cache dir.

    Creates or restarts the server if needed (e.g. after ``uva.set()``).
    """
    global _server_instance, _server_port, _server_dir

    cache_dir = _core.get_cache_dir()
    with _server_lock:
        if _server_instance is not None and _server_dir == cache_dir:
            return "127.0.0.1", _server_port

        if _server_instance is not None:
            _server_instance.shutdown()
            _server_instance = None

        port = _find_free_port()

        class _QuietHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(cache_dir), **kwargs)

            def log_message(self, fmt, *args):  # suppress access logs
                pass

        server = HTTPServer(("127.0.0.1", port), _QuietHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        _server_instance = server
        _server_port = port
        _server_dir = cache_dir

    time.sleep(0.05)  # brief warm-up
    return "127.0.0.1", port


# ══════════════════════════════════════════════════════════════════════════════
# Internal: HTML generation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


def _inject_config(template: str, config: Dict[str, Any]) -> str:
    return template.replace(
        "/* __VIEWER_CONFIG__ */",
        json.dumps(config, indent=2, default=str),
    )


def _open_viewer(html_content: str, slug: str, url: str) -> None:
    """Write rendered HTML to the cache dir and open it via the local server."""
    cache_dir = _core.get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"_uva_{slug}.html"
    out_path = cache_dir / filename
    out_path.write_text(html_content, encoding="utf-8")
    webbrowser.open(url)
    print(f"[urbanverse] Viewer opened → {url}")
    print(f"[urbanverse]  HTML saved  → {out_path}")


def _file_url(base: str, rel_path: str) -> str:
    return f"{base}/{rel_path}"


# ══════════════════════════════════════════════════════════════════════════════
# Internal: per-asset download helpers (avoids circular import with __init__)
# ══════════════════════════════════════════════════════════════════════════════

def _download_std_glb(uids: List[str], num_workers: int) -> None:
    paths = [
        rp for uid in uids
        if (rp := _core.get_bucket_path(uid, "glb")) is not None
    ]
    _core._download_files_parallel(paths, num_workers, desc="GLB models")


def _download_thumbnails(uids: List[str], num_workers: int) -> None:
    paths = [
        rp for uid in uids
        if (rp := _core.get_bucket_path(uid, "thumbnail")) is not None
    ]
    _core._download_files_parallel(paths, num_workers, desc="Thumbnails")


def _download_renders(
    uids: List[str],
    angles: Tuple[int, ...],
    num_workers: int,
) -> None:
    _core._download_and_extract_renders(uids, num_workers)


def _download_annotations(uids: List[str], num_workers: int) -> None:
    ok = _core._ensure_per_asset_annotations()
    if not ok:
        paths = [
            rp for uid in uids
            if (rp := _core.get_bucket_path(uid, "annotation")) is not None
        ]
        _core._download_files_parallel(paths, num_workers, desc="Annotations")


def _read_annotation(uid: str) -> Optional[Dict]:
    cache = _core.get_cache_dir()
    # Check bucket path first, then flat path (from annotation bundle)
    rp = _core.get_bucket_path(uid, "annotation")
    p = cache / rp if rp else None
    if p is None or not p.exists():
        p = cache / f"assets_std_annotation_flat/std_{uid}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _all_uids() -> List[str]:
    ann = _core.load_master_annotation()
    return [
        uid
        for entry in ann["annotation"].values()
        for uid in entry["asset_uids"]
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Internal: config dict builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_asset_config(
    uids: List[str],
    mode: str,
    base: str,
    angles: Tuple[int, ...] = _RENDER_ANGLES,
) -> Dict[str, Any]:
    """Build the VIEWER_CONFIG JSON that is embedded in asset_viewer.html."""
    cache = _core.get_cache_dir()
    assets_data: List[Dict] = []

    for uid in uids:
        entry: Dict[str, Any] = {
            "uid": uid,
            "saved_path": str(cache),
        }

        # GLB URL (bucket path)
        if mode in ("full", "glb"):
            rp = _core.get_bucket_path(uid, "glb")
            p = cache / rp if rp else None
            entry["glb_url"] = _file_url(base, rp) if rp and p and p.exists() else None
        else:
            entry["glb_url"] = None

        # Thumbnail URL (bucket path)
        if mode in ("full", "thumbnails", "images"):
            rp = _core.get_bucket_path(uid, "thumbnail")
            p = cache / rp if rp else None
            entry["thumbnail_url"] = _file_url(base, rp) if rp and p and p.exists() else None
        else:
            entry["thumbnail_url"] = None

        # Render URLs (extracted into bucket folders)
        if mode in ("full", "renders", "images"):
            render_urls: Dict[str, Optional[str]] = {}
            render_dir = _core.render_extract_dir(uid)
            for a in angles:
                if render_dir:
                    jpg = render_dir / f"render_{float(a)}.jpg"
                    if jpg.exists():
                        rel = str(jpg.relative_to(cache))
                        render_urls[str(a)] = _file_url(base, rel)
                    else:
                        render_urls[str(a)] = None
                else:
                    render_urls[str(a)] = None
            entry["render_urls"] = render_urls
        else:
            entry["render_urls"] = {}

        # Annotation
        if mode in ("full", "annotation"):
            entry["annotation"] = _read_annotation(uid)
        else:
            entry["annotation"] = None

        assets_data.append(entry)

    titles = {
        "full":        "UrbanVerse Asset Viewer",
        "glb":         "UrbanVerse Object Viewer",
        "original":    "UrbanVerse Original Object Viewer",
        "thumbnails":  "UrbanVerse Thumbnail Viewer",
        "renders":     "UrbanVerse Render Viewer",
        "images":      "UrbanVerse Image Viewer",
        "annotation":  "UrbanVerse Annotation Viewer",
    }
    return {
        "title":  titles.get(mode, "UrbanVerse Viewer"),
        "mode":   mode,
        "assets": assets_data,
        "defaultExposure": 1,
    }


def _build_sky_config(
    descriptions: List[str],
    file_map: Dict[str, Dict],
    base: str,
) -> Dict[str, Any]:
    cache = _core.get_cache_dir()
    materials: List[Dict] = []
    for desc in descriptions:
        entry = file_map.get(desc, {})
        main_files = entry.get("main_files", [])
        hdr_rel   = main_files[0] if main_files else ""
        thumb_rel = entry.get("thumbnail", "")
        hdr_local = cache / hdr_rel if hdr_rel else None
        materials.append({
            "description":  desc,
            "hdr_url":      _file_url(base, hdr_rel) if hdr_rel and hdr_local and hdr_local.exists() else None,
            "thumbnail_url":_file_url(base, thumb_rel) if thumb_rel and (cache / thumb_rel).exists() else None,
            "saved_path":   str(hdr_local) if hdr_local else "",
        })
    return {
        "title":     "UrbanVerse Sky Materials Viewer",
        "materials": materials,
    }


def _build_grid_config(
    descriptions: List[str],
    file_map: Dict[str, Dict],
    base: str,
    title: str,
) -> Dict[str, Any]:
    cache = _core.get_cache_dir()
    materials: List[Dict] = []
    for desc in descriptions:
        entry   = file_map.get(desc, {})
        rel     = entry.get("thumbnail", "")
        local   = cache / rel if rel else None
        materials.append({
            "description":  desc,
            "thumbnail_url": _file_url(base, rel) if rel and local and local.exists() else None,
            "saved_path":   str(local) if local else "",
        })
    return {"title": title, "materials": materials}


# ── MDL texture parsing (for road / sidewalk / terrain PBR viewer) ────────

_TEXTURE_REF_RE = re.compile(r'texture_2d\(\s*"([^"]+)"')

_CHANNEL_KEYWORDS = {
    "diff":   "diffuse",
    "norm":   "normal",
    "rough":  "roughness",
    "height": "height",
    "ao":     "ao",
    "metal":  "metallic",
    "mask":   "mask",
    "multi":  "multi",
    "dis":    "height",
    "ref":    "metallic",
    "rou":    "roughness",
    "curv":   "ao",
    "ORM":    "orm",
    "grunge": "mask",
}


def _classify_texture(filename: str) -> str:
    """Map a texture filename to a PBR channel based on suffix keywords."""
    base = os.path.splitext(filename)[0]
    parts = base.lower().split("_")
    for part in reversed(parts):
        for keyword, channel in _CHANNEL_KEYWORDS.items():
            if keyword.lower() == part:
                return channel
    if "orm" in base.lower():
        return "orm"
    return "unknown"


def _parse_mdl_textures(
    mdl_path: Path,
    folder_main: str,
    base_url: str,
) -> Dict[str, str]:
    """Parse an .mdl file and return {channel: http_url} for its PBR textures."""
    try:
        content = mdl_path.read_text(errors="replace")
    except Exception:
        return {}

    refs = _TEXTURE_REF_RE.findall(content)
    seen: set = set()
    unique_refs: List[str] = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            unique_refs.append(r)

    cache = _core.get_cache_dir()
    textures: Dict[str, str] = {}

    for ref in unique_refs:
        filename = os.path.basename(ref)
        channel = _classify_texture(filename)

        clean_ref = ref.lstrip("./")
        repo_rel = f"{folder_main}/{clean_ref}"
        local_path = cache / repo_rel

        if not local_path.exists():
            continue

        url = _file_url(base_url, repo_rel)

        if channel == "orm":
            textures.setdefault("roughness", url)
            textures.setdefault("ao", url)
        elif channel == "multi":
            name_lower = filename.lower()
            if "rough" in name_lower:
                textures.setdefault("roughness", url)
            if "ao" in name_lower:
                textures.setdefault("ao", url)
            if "height" in name_lower:
                textures.setdefault("height", url)
        elif channel not in ("unknown", "mask"):
            textures.setdefault(channel, url)

    return textures


def _build_mdl_config(
    descriptions: List[str],
    file_map: Dict[str, Dict],
    folder_main: str,
    base: str,
    title: str,
) -> Dict[str, Any]:
    """Build VIEWER_CONFIG for the MDL material PBR viewer."""
    cache = _core.get_cache_dir()
    mat_list: List[Dict] = []

    for desc in descriptions:
        entry = file_map.get(desc, {})
        main_files = entry.get("main_files", [])

        mdl_file = next((f for f in main_files if f.endswith(".mdl")), None)
        mdl_path = cache / mdl_file if mdl_file else None

        textures: Dict[str, str] = {}
        if mdl_path and mdl_path.exists():
            textures = _parse_mdl_textures(mdl_path, folder_main, base)

        thumb_rel = entry.get("thumbnail", "")
        thumb_url = (
            _file_url(base, thumb_rel)
            if thumb_rel and (cache / thumb_rel).exists()
            else None
        )

        mat_list.append({
            "name":          desc,
            "textures":      textures,
            "thumbnail_url": thumb_url,
            "saved_path":    str(mdl_path) if mdl_path else "",
        })

    return {"title": title, "materials": mat_list}


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def object_distribution() -> None:
    """Open the interactive UrbanVerse dataset distribution visualization.

    Downloads ``UrbanVerse-100K_Interactive_Distribution.html`` from the HF
    repo on first call (cached locally afterwards) and opens it in the browser.

    Example::

        uva.viewer.object_distribution()
    """
    dist_repo = "UrbanVerse-100K_Interactive_Distribution.html"
    try:
        _core._download_file(dist_repo)
    except Exception as e:
        print(f"[urbanverse] Could not download distribution plot: {e}")
        return

    host, port = _ensure_server()
    url = f"http://{host}:{port}/{dist_repo}"
    webbrowser.open(url)
    print(f"[urbanverse] Distribution viewer → {url}")


def object_show(
    uids: Optional[List[str]] = None,
    num_workers: int = 8,
    angles: Tuple[int, ...] = (0, 90, 180, 270),
) -> Dict[str, Any]:
    """Download and open a viewer for assets (GLB + thumbnail + renders + annotation).

    Each asset is displayed on its own page with:
      - An interactive 3D GLB viewer
      - A horizontal thumbnail & render image strip
      - A structured annotation table

    Navigate between assets using the ← / → buttons or arrow keys.

    Args:
        uids:        UIDs to view. ``None`` → all assets (**very large** download).
        num_workers: Parallel download threads (default 8).
        angles:      Render angles to include. Subset of ``(0, 90, 180, 270)``.

    Returns:
        ``{uid: {"std_glb": Path, "thumbnail": Path, "render": {...},
                 "std_annotation": Path}}``

    Example::

        uva.viewer.object_show(["uid1", "uid2"])
    """
    if uids is None:
        uids = _all_uids()
    if not uids:
        print("[urbanverse] No UIDs to view.")
        return {}

    _download_std_glb(uids, num_workers)
    _download_thumbnails(uids, num_workers)
    _download_renders(uids, angles, num_workers)
    _download_annotations(uids, num_workers)

    host, port = _ensure_server()
    base = f"http://{host}:{port}"
    config = _build_asset_config(uids, "full", base, angles)
    html = _inject_config(_load_template("asset_viewer.html"), config)
    url = f"{base}/_uva_assets.html"
    _open_viewer(html, "assets", url)

    cache = _core.get_cache_dir()
    result_dict: Dict[str, Any] = {}
    for uid in uids:
        glb_rp = _core.get_bucket_path(uid, "glb")
        thumb_rp = _core.get_bucket_path(uid, "thumbnail")
        ann_rp = _core.get_bucket_path(uid, "annotation")
        render_dir = _core.render_extract_dir(uid)
        result_dict[uid] = {
            "std_glb":        cache / glb_rp if glb_rp else None,
            "thumbnail":      cache / thumb_rp if thumb_rp else None,
            "render":         {a: render_dir / f"render_{float(a)}.jpg" for a in angles} if render_dir else {},
            "std_annotation": cache / ann_rp if ann_rp else None,
        }
    return result_dict


def object_annotations(
    uids: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Optional[Dict]]:
    """Download and open a viewer showing only annotation tables (no 3D or images).

    Args:
        uids:        UIDs to view. ``None`` → all assets.
        num_workers: Parallel download threads (default 8).

    Returns:
        ``{uid: annotation_dict | None}``

    Example::

        uva.viewer.object_annotations(["uid1", "uid2"])
    """
    if uids is None:
        uids = _all_uids()
    if not uids:
        print("[urbanverse] No UIDs to view.")
        return {}

    _download_annotations(uids, num_workers)

    host, port = _ensure_server()
    base = f"http://{host}:{port}"
    config = _build_asset_config(uids, "annotation", base)
    html = _inject_config(_load_template("asset_viewer.html"), config)
    url = f"{base}/_uva_annotations.html"
    _open_viewer(html, "annotations", url)

    return {uid: _read_annotation(uid) for uid in uids}


def object_assets(
    uids: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Optional[Path]]:
    """Download and open an interactive 3D viewer for metric-scale GLB objects.

    Args:
        uids:        UIDs to view. ``None`` → all assets.
        num_workers: Parallel download threads (default 8).

    Returns:
        ``{uid: Path | None}``  — paths to ``.glb`` files on disk.

    Example::

        uva.viewer.object_assets(["uid1", "uid2"])
    """
    if uids is None:
        uids = _all_uids()
    if not uids:
        print("[urbanverse] No UIDs to view.")
        return {}

    _download_std_glb(uids, num_workers)

    host, port = _ensure_server()
    base = f"http://{host}:{port}"
    config = _build_asset_config(uids, "glb", base)
    html = _inject_config(_load_template("asset_viewer.html"), config)
    url = f"{base}/_uva_objects.html"
    _open_viewer(html, "objects", url)

    result_dict: Dict[str, Optional[Path]] = {}
    for uid in uids:
        rp = _core.get_bucket_path(uid, "glb")
        p = _core.get_cache_dir() / rp if rp else None
        result_dict[uid] = p if p and p.exists() else None
    return result_dict


def object_thumbnails(
    uids: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Optional[Path]]:
    """Download and open a thumbnail image viewer.

    Args:
        uids:        UIDs to view. ``None`` → all assets.
        num_workers: Parallel download threads (default 8).

    Returns:
        ``{uid: Path | None}``

    Example::

        uva.viewer.object_thumbnails(["uid1", "uid2"])
    """
    if uids is None:
        uids = _all_uids()
    if not uids:
        print("[urbanverse] No UIDs to view.")
        return {}

    _download_thumbnails(uids, num_workers)

    host, port = _ensure_server()
    base = f"http://{host}:{port}"
    config = _build_asset_config(uids, "thumbnails", base)
    html = _inject_config(_load_template("asset_viewer.html"), config)
    url = f"{base}/_uva_thumbnails.html"
    _open_viewer(html, "thumbnails", url)

    result_dict: Dict[str, Optional[Path]] = {}
    for uid in uids:
        rp = _core.get_bucket_path(uid, "thumbnail")
        p = _core.get_cache_dir() / rp if rp else None
        result_dict[uid] = p if p and p.exists() else None
    return result_dict


def object_renders(
    uids: Optional[List[str]] = None,
    angles: Tuple[int, ...] = (0, 90, 180, 270),
    num_workers: int = 8,
) -> Dict[str, Dict[int, Optional[Path]]]:
    """Download and open a render image viewer.

    Args:
        uids:        UIDs to view. ``None`` → all assets.
        angles:      Angles to download/view. Subset of ``(0, 90, 180, 270)``.
        num_workers: Parallel download threads (default 8).

    Returns:
        ``{uid: {angle: Path | None}}``

    Example::

        uva.viewer.object_renders(["uid1"], angles=(0, 90))
    """
    valid = frozenset(_RENDER_ANGLES)
    invalid = frozenset(angles) - valid
    if invalid:
        raise ValueError(
            f"Invalid angle(s): {sorted(invalid)}. Must be subset of {sorted(valid)}."
        )

    if uids is None:
        uids = _all_uids()
    if not uids:
        print("[urbanverse] No UIDs to view.")
        return {}

    _download_renders(uids, angles, num_workers)

    host, port = _ensure_server()
    base = f"http://{host}:{port}"
    config = _build_asset_config(uids, "renders", base, angles)
    html = _inject_config(_load_template("asset_viewer.html"), config)
    url = f"{base}/_uva_renders.html"
    _open_viewer(html, "renders", url)

    result_dict: Dict[str, Dict[int, Optional[Path]]] = {}
    for uid in uids:
        render_dir = _core.render_extract_dir(uid)
        result_dict[uid] = {
            a: (p if render_dir and (p := render_dir / f"render_{float(a)}.jpg").exists() else None)
            for a in angles
        }
    return result_dict


def sky_show(
    descriptions: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Dict[str, Optional[Path]]]:
    """Download and open an IBL sky-map viewer.

    Shows three PBR reference spheres (chrome, adjustable roughness, matte)
    lit by the selected HDRI environment map.  Navigate between sky maps using
    the ← / → buttons.

    Args:
        descriptions: Material names to view. ``None`` → all available.
                      Call ``uva.sky.get_descriptions()`` for valid names.
        num_workers:  Parallel download threads (default 8).

    Returns:
        ``{description: {"hdr": Path | None, "thumbnail": Path | None}}``

    Example::

        uva.viewer.sky_show(["autumn_field_puresky", "venice_sunset"])
    """
    file_map = _sky._Sky._get_file_map()

    if descriptions is None:
        descriptions = sorted(file_map.keys())
    else:
        unknown = set(descriptions) - set(file_map.keys())
        if unknown:
            raise ValueError(
                f"Unknown sky material(s): {sorted(unknown)}\n"
                f"  Call uva.sky.get_descriptions() to see valid names."
            )

    result = _sky._Sky.load_materials(descriptions, num_workers=num_workers)

    host, port = _ensure_server()
    base = f"http://{host}:{port}"
    config = _build_sky_config(descriptions, file_map, base)
    html = _inject_config(_load_template("sky_viewer.html"), config)
    url = f"{base}/_uva_sky_show.html"
    _open_viewer(html, "sky_show", url)

    return result


def road_show(
    descriptions: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Dict[str, Optional[Path]]]:
    """Download and open a 3D PBR material viewer for road materials.

    Parses ``.mdl`` files to extract PBR texture references and renders
    them on a 3D plane using Three.js with physically-based shading.

    Args:
        descriptions: Material names to view. ``None`` → all available.
                      Call ``uva.road.get_descriptions()`` for valid names.
        num_workers:  Parallel download threads (default 8).

    Returns:
        ``{description: {"mdl": Path | None, "thumbnail": Path | None}}``

    Example::

        uva.viewer.road_show()
    """
    file_map = _road._Road._get_file_map()

    if descriptions is None:
        descriptions = sorted(file_map.keys())
    else:
        unknown = set(descriptions) - set(file_map.keys())
        if unknown:
            raise ValueError(
                f"Unknown road material(s): {sorted(unknown)}\n"
                f"  Call uva.road.get_descriptions() to see valid names."
            )

    result = _road._Road.load_materials(descriptions, num_workers=num_workers)

    host, port = _ensure_server()
    base = f"http://{host}:{port}"
    config = _build_mdl_config(
        descriptions, file_map, "material_road_mdl", base, "UrbanVerse Road Materials",
    )
    html = _inject_config(_load_template("mdl_material_viewer.html"), config)
    url = f"{base}/_uva_road_show.html"
    _open_viewer(html, "road_show", url)

    return result


def sidewalk_show(
    descriptions: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Dict[str, Optional[Path]]]:
    """Download and open a 3D PBR material viewer for sidewalk materials.

    Parses ``.mdl`` files to extract PBR texture references and renders
    them on a 3D plane using Three.js with physically-based shading.

    Args:
        descriptions: Material names to view. ``None`` → all available.
                      Call ``uva.sidewalk.get_descriptions()`` for valid names.
        num_workers:  Parallel download threads (default 8).

    Returns:
        ``{description: {"mdl": Path | None, "thumbnail": Path | None}}``

    Example::

        uva.viewer.sidewalk_show()
    """
    file_map = _sidewalk._Sidewalk._get_file_map()

    if descriptions is None:
        descriptions = sorted(file_map.keys())
    else:
        unknown = set(descriptions) - set(file_map.keys())
        if unknown:
            raise ValueError(
                f"Unknown sidewalk material(s): {sorted(unknown)}\n"
                f"  Call uva.sidewalk.get_descriptions() to see valid names."
            )

    result = _sidewalk._Sidewalk.load_materials(descriptions, num_workers=num_workers)

    host, port = _ensure_server()
    base = f"http://{host}:{port}"
    config = _build_mdl_config(
        descriptions, file_map, "material_sidewalk_mdl", base, "UrbanVerse Sidewalk Materials",
    )
    html = _inject_config(_load_template("mdl_material_viewer.html"), config)
    url = f"{base}/_uva_sidewalk_show.html"
    _open_viewer(html, "sidewalk_show", url)

    return result


def terrain_show(
    descriptions: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Dict[str, Optional[Path]]]:
    """Download and open a 3D PBR material viewer for terrain materials.

    Parses ``.mdl`` files to extract PBR texture references and renders
    them on a 3D plane using Three.js with physically-based shading.

    Args:
        descriptions: Material names to view. ``None`` → all available.
                      Call ``uva.terrain.get_descriptions()`` for valid names.
        num_workers:  Parallel download threads (default 8).

    Returns:
        ``{description: {"mdl": Path | None, "thumbnail": Path | None}}``

    Example::

        uva.viewer.terrain_show()
    """
    file_map = _terrain._Terrain._get_file_map()

    if descriptions is None:
        descriptions = sorted(file_map.keys())
    else:
        unknown = set(descriptions) - set(file_map.keys())
        if unknown:
            raise ValueError(
                f"Unknown terrain material(s): {sorted(unknown)}\n"
                f"  Call uva.terrain.get_descriptions() to see valid names."
            )

    result = _terrain._Terrain.load_materials(descriptions, num_workers=num_workers)

    host, port = _ensure_server()
    base = f"http://{host}:{port}"
    config = _build_mdl_config(
        descriptions, file_map, "material_terrain_mdl", base, "UrbanVerse Terrain Materials",
    )
    html = _inject_config(_load_template("mdl_material_viewer.html"), config)
    url = f"{base}/_uva_terrain_show.html"
    _open_viewer(html, "terrain_show", url)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Vegetation viewers (plant / shrub / tree — USD→GLB conversion)
# ══════════════════════════════════════════════════════════════════════════════

_PREVIEW_GLB = "_preview.glb"


def _ensure_preview_glb(
    descriptions: List[str],
    load_result: Dict[str, Dict],
) -> Dict[str, Optional[Path]]:
    """Convert USD assets to GLB previews, caching alongside the extracted folder."""
    from ._usd_to_glb import convert_usd_to_glb

    glb_paths: Dict[str, Optional[Path]] = {}
    for desc in descriptions:
        entry = load_result.get(desc, {})
        usd_path = entry.get("usd")
        folder = entry.get("folder")

        if not usd_path or not folder:
            glb_paths[desc] = None
            continue

        glb_path = folder / _PREVIEW_GLB
        if not glb_path.exists():
            try:
                convert_usd_to_glb(usd_path, glb_path)
            except Exception as e:
                print(f"[urbanverse] USD→GLB conversion failed for {desc}: {e}")
                glb_paths[desc] = None
                continue

        glb_paths[desc] = glb_path
    return glb_paths


def _build_vegetation_config(
    descriptions: List[str],
    glb_paths: Dict[str, Optional[Path]],
    folder_name: str,
    base: str,
    title: str,
) -> Dict[str, Any]:
    """Build VIEWER_CONFIG for the asset_viewer.html template."""
    cache = _core.get_cache_dir()
    assets_data: List[Dict] = []

    for desc in descriptions:
        glb = glb_paths.get(desc)
        glb_url = None
        if glb and glb.exists():
            rel = glb.relative_to(cache)
            glb_url = _file_url(base, str(rel))

        assets_data.append({
            "uid": desc,
            "saved_path": str(glb) if glb else "",
            "glb_url": glb_url,
            "thumbnail_url": None,
            "render_urls": {},
            "annotation": None,
        })

    return {
        "title": title,
        "mode": "glb",
        "assets": assets_data,
        "defaultExposure": 4,
    }


def plant_show(
    descriptions: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Dict]:
    """Download, convert, and view plant assets in a 3D viewer.

    Downloads ``.tar.gz`` archives from HuggingFace, extracts them,
    converts USD to GLB for Three.js rendering, and opens a 3D viewer.

    Args:
        descriptions: Plant names to view.  ``None`` → all available.
                      Call ``uva.plant.get_descriptions()`` for valid names.
        num_workers:  Parallel download threads (default 8).

    Returns:
        ``{name: {"usd": Path | None, "folder": Path | None}}``

    Example::

        uva.viewer.plant_show(["Plant_02"])
    """
    if descriptions is None:
        descriptions = _plant.get_descriptions()
    else:
        valid = set(_plant._Plant._get_file_map().keys())
        unknown = set(descriptions) - valid
        if unknown:
            raise ValueError(
                f"Unknown plant(s): {sorted(unknown)}\n"
                f"  Call uva.plant.get_descriptions() to see valid names."
            )

    if not descriptions:
        print("[urbanverse] No plant assets to view.")
        return {}

    result = _plant.load_materials(descriptions, num_workers=num_workers)
    glb_paths = _ensure_preview_glb(descriptions, result)

    host, port = _ensure_server()
    base = f"http://{host}:{port}"
    config = _build_vegetation_config(
        descriptions, glb_paths, "collected_plants", base, "UrbanVerse Plant Viewer",
    )
    html = _inject_config(_load_template("asset_viewer.html"), config)
    url = f"{base}/_uva_plant_show.html"
    _open_viewer(html, "plant_show", url)

    return result


def shrub_show(
    descriptions: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Dict]:
    """Download, convert, and view shrub assets in a 3D viewer.

    Downloads ``.tar.gz`` archives from HuggingFace, extracts them,
    converts USD to GLB for Three.js rendering, and opens a 3D viewer.

    Args:
        descriptions: Shrub names to view.  ``None`` → all available.
                      Call ``uva.shrub.get_descriptions()`` for valid names.
        num_workers:  Parallel download threads (default 8).

    Returns:
        ``{name: {"usd": Path | None, "folder": Path | None}}``

    Example::

        uva.viewer.shrub_show(["Acacia"])
    """
    if descriptions is None:
        descriptions = _shrub.get_descriptions()
    else:
        valid = set(_shrub._Shrub._get_file_map().keys())
        unknown = set(descriptions) - valid
        if unknown:
            raise ValueError(
                f"Unknown shrub(s): {sorted(unknown)}\n"
                f"  Call uva.shrub.get_descriptions() to see valid names."
            )

    if not descriptions:
        print("[urbanverse] No shrub assets to view.")
        return {}

    result = _shrub.load_materials(descriptions, num_workers=num_workers)
    glb_paths = _ensure_preview_glb(descriptions, result)

    host, port = _ensure_server()
    base = f"http://{host}:{port}"
    config = _build_vegetation_config(
        descriptions, glb_paths, "collected_shrubs", base, "UrbanVerse Shrub Viewer",
    )
    html = _inject_config(_load_template("asset_viewer.html"), config)
    url = f"{base}/_uva_shrub_show.html"
    _open_viewer(html, "shrub_show", url)

    return result


def tree_show(
    descriptions: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Dict]:
    """Download, convert, and view tree assets in a 3D viewer.

    Downloads ``.tar.gz`` archives from HuggingFace, extracts them,
    converts USD to GLB for Three.js rendering, and opens a 3D viewer.

    Args:
        descriptions: Tree names to view.  ``None`` → all available.
                      Call ``uva.tree.get_descriptions()`` for valid names.
        num_workers:  Parallel download threads (default 8).

    Returns:
        ``{name: {"usd": Path | None, "folder": Path | None}}``

    Example::

        uva.viewer.tree_show(["American_Beech"])
    """
    if descriptions is None:
        descriptions = _tree.get_descriptions()
    else:
        valid = set(_tree._Tree._get_file_map().keys())
        unknown = set(descriptions) - valid
        if unknown:
            raise ValueError(
                f"Unknown tree(s): {sorted(unknown)}\n"
                f"  Call uva.tree.get_descriptions() to see valid names."
            )

    if not descriptions:
        print("[urbanverse] No tree assets to view.")
        return {}

    result = _tree.load_materials(descriptions, num_workers=num_workers)
    glb_paths = _ensure_preview_glb(descriptions, result)

    host, port = _ensure_server()
    base = f"http://{host}:{port}"
    config = _build_vegetation_config(
        descriptions, glb_paths, "collected_trees", base, "UrbanVerse Tree Viewer",
    )
    html = _inject_config(_load_template("asset_viewer.html"), config)
    url = f"{base}/_uva_tree_show.html"
    _open_viewer(html, "tree_show", url)

    return result
