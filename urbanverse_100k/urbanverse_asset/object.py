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
uva.object — 3D object asset APIs (GLB, annotations, thumbnails, renders).

Usage::

    import urbanverse_asset as uva

    # Discover UIDs
    all_uids = uva.object.get_uids_all()
    uids = uva.object.get_uids_conditioned(categories=["vehicle"], movable=True)

    # Download everything for selected UIDs
    result = uva.object.load(uids[:5])
    glb = result["<uid>"]["std_glb"]

    # Download only GLB assets
    paths = uva.object.load_assets(["uid1"])
    glb = paths["uid1"]
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import _core
from ._filter import get_uids_conditioned  # re-exported as uva.object.get_uids_conditioned

_ASSET_WHAT = ("std_glb", "std_annotation", "thumbnail", "render")
_RENDER_ANGLES = (0, 90, 180, 270)


# ══════════════════════════════════════════════════════════════════════════════
# UID discovery & category browsing
# ══════════════════════════════════════════════════════════════════════════════

def get_uids_all() -> List[str]:
    """Return a list of all 102,444 asset UIDs in the dataset.

    Downloads the master annotation on first call; subsequent calls are instant.

    Returns:
        List of UID strings (32-char hex).

    Example::

        uids = uva.object.get_uids_all()
        print(len(uids))     # 102444
        print(uids[0])       # "06b6e2b916584faab430bd03c0c19499"
    """
    ann = _core.load_master_annotation()
    uids = [
        uid
        for entry in ann["annotation"].values()
        for uid in entry["asset_uids"]
    ]
    return uids


def categories(level: Optional[int] = None) -> Dict[str, Any]:
    """List the category hierarchy and the UIDs belonging to each class.

    Args:
        level: ``1``, ``2``, or ``3`` to get one level of the hierarchy.
               ``None`` (default) returns the full per-L3-class annotation
               dict (includes UIDs, parent class names, and asset count).

    Returns:
        *level is None* →
            ``{l3_class_name: {class_name_l1, class_name_l2, class_name_l3,
            class_id_l1/l2/l3, asset_uids: [...], asset_count: int}}``

        *level == 1 / 2 / 3* →
            ``{class_name: [uid, uid, ...]}``

    Raises:
        ValueError: If ``level`` is not 1, 2, 3, or ``None``.

    Example::

        l1 = uva.object.categories(level=1)
        vehicle_uids = l1["vehicle"]

        full = uva.object.categories()
        bench_info = full["bench"]
        print(bench_info["asset_count"])
    """
    if level is not None and level not in (1, 2, 3):
        raise ValueError(f"`level` must be 1, 2, 3, or None — got {level!r}")

    ann = _core.load_master_annotation()
    annotation = ann["annotation"]

    if level is None:
        return annotation

    out: Dict[str, List[str]] = {}
    key = f"class_name_l{level}"
    for entry in annotation.values():
        cat = entry[key]
        out.setdefault(cat, []).extend(entry["asset_uids"])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _uid_to_repo_paths(uid: str, what: tuple) -> List[str]:
    """Resolve UID → bucketed HF repo paths using the bucket index files."""
    paths: List[str] = []
    if "std_glb" in what:
        rp = _core.get_bucket_path(uid, "glb")
        if rp:
            paths.append(rp)
    if "std_annotation" in what:
        rp = _core.get_bucket_path(uid, "annotation")
        if rp:
            paths.append(rp)
    if "thumbnail" in what:
        rp = _core.get_bucket_path(uid, "thumbnail")
        if rp:
            paths.append(rp)
    return paths


def _uid_local_path(uid: str, asset_type: str) -> Optional[Path]:
    """Return the local cache path for a UID and asset type using bucket index."""
    rp = _core.get_bucket_path(uid, asset_type)
    if rp is None:
        return None
    return _core.get_cache_dir() / rp


def _build_asset_result(uid: str, what: tuple, cache: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if "std_glb" in what:
        p = _uid_local_path(uid, "glb")
        result["std_glb"] = p if p and p.exists() else None
    if "std_annotation" in what:
        p = _uid_local_path(uid, "annotation")
        if p and p.exists():
            result["std_annotation"] = p
        else:
            p_flat = cache / f"assets_std_annotation_flat/std_{uid}.json"
            result["std_annotation"] = p_flat if p_flat.exists() else None
    if "thumbnail" in what:
        p = _uid_local_path(uid, "thumbnail")
        result["thumbnail"] = p if p and p.exists() else None
    if "render" in what:
        renders: Dict[int, Optional[Path]] = {}
        render_dir = _core.render_extract_dir(uid)
        for a in _RENDER_ANGLES:
            p = render_dir / f"render_{float(a)}.jpg" if render_dir else None
            renders[a] = p if p and p.exists() else None
        result["render"] = renders
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ══════════════════════════════════════════════════════════════════════════════

def load(
    uids: Optional[List[str]] = None,
    num_workers: int = 8,
    what: tuple = ("std_glb", "std_annotation", "thumbnail", "render"),
) -> Dict[str, Dict[str, Any]]:
    """Download all asset data for the given UIDs and return their local paths.

    This is the all-in-one loader: GLB models, annotations, thumbnails,
    and multi-angle renders in a single call.  Already-cached files are
    **not** re-downloaded.

    Args:
        uids:        List of UID strings.  ``None`` → all 102,444 assets.
        num_workers: Parallel download threads (default 8).
        what:        Data types to download/include in the result dict.
                     Valid values: ``"std_glb"``, ``"std_annotation"``,
                     ``"thumbnail"``, ``"render"``.

    Returns:
        ``{uid: {
            "std_glb":        Path | None,
            "std_annotation": Path | None,
            "thumbnail":      Path | None,
            "render":         {0: Path, 90: Path, 180: Path, 270: Path} | None,
        }}``

        Only keys requested in ``what`` are present.  A value of ``None``
        means the file failed to download.

    Raises:
        ValueError: If ``what`` contains unknown values.

    Example::

        result = uva.object.load(["uid1", "uid2"])
        glb = result["uid1"]["std_glb"]
        ann = result["uid1"]["std_annotation"]
    """
    _core.validate_what(what, _ASSET_WHAT)

    if uids is None:
        uids = get_uids_all()

    if not uids:
        return {}

    if "std_annotation" in what:
        bundle_ok = _core._ensure_per_asset_annotations()
        non_ann_what = tuple(w for w in what if w != "std_annotation")
    else:
        bundle_ok = False
        non_ann_what = what

    non_render_what = tuple(w for w in non_ann_what if w != "render")
    if non_render_what:
        repo_paths = [p for uid in uids for p in _uid_to_repo_paths(uid, non_render_what)]
        _core._download_files_parallel(repo_paths, num_workers, desc="Assets")

    if "render" in what:
        _core._download_and_extract_renders(uids, num_workers)

    if "std_annotation" in what and not bundle_ok:
        ann_paths = [
            rp for uid in uids
            if (rp := _core.get_bucket_path(uid, "annotation")) is not None
        ]
        _core._download_files_parallel(ann_paths, num_workers, desc="Annotations")

    cache = _core.get_cache_dir()
    return {uid: _build_asset_result(uid, what, cache) for uid in uids}


def load_assets(
    uids: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Optional[Path]]:
    """Download metric-scale .glb files and return their local paths.

    These are the **standardised, metric-scaled** 3D assets (``std_*.glb``),
    suitable for simulation.

    Args:
        uids:        List of UIDs.  ``None`` → all assets.
        num_workers: Parallel download threads (default 8).

    Returns:
        ``{uid: Path | None}``

    Example::

        paths = uva.object.load_assets(["uid1"])
        glb   = paths["uid1"]
    """
    if uids is None:
        uids = get_uids_all()

    if not uids:
        return {}

    repo_paths = [
        rp for uid in uids
        if (rp := _core.get_bucket_path(uid, "glb")) is not None
    ]
    _core._download_files_parallel(repo_paths, num_workers, desc="GLB (std)")

    result: Dict[str, Optional[Path]] = {}
    for uid in uids:
        p = _uid_local_path(uid, "glb")
        result[uid] = p if p and p.exists() else None
    return result


def load_annotations(
    uids: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Download per-asset annotation JSONs and return them as dicts.

    On first call, automatically downloads and extracts the full annotation
    bundle (``full_per_asset_annotations.tar.gz``) from the HF repo so that
    every annotation is available locally in one shot.  Subsequent calls read
    directly from disk — no network traffic needed.

    If the bundle is not available in the repo (e.g. toy dataset), individual
    JSON files are downloaded in parallel as a fallback.

    Args:
        uids:        List of UIDs.  ``None`` → all assets.
        num_workers: Parallel threads used for the per-file fallback (default 8).

    Returns:
        ``{uid: annotation_dict | None}``
        ``None`` means the file was not found or failed to download.

    Example::

        anns = uva.object.load_annotations(["uid1", "uid2"])
        print(anns["uid1"]["category"])     # "bench"
        print(anns["uid1"]["height"])       # 0.85
    """
    if uids is None:
        uids = get_uids_all()

    if not uids:
        return {}

    bundle_ok = _core._ensure_per_asset_annotations()
    if not bundle_ok:
        ann_paths = [
            rp for uid in uids
            if (rp := _core.get_bucket_path(uid, "annotation")) is not None
        ]
        _core._download_files_parallel(ann_paths, num_workers, desc="Annotations")

    cache = _core.get_cache_dir()
    out: Dict[str, Optional[Dict]] = {}
    for uid in uids:
        p = _uid_local_path(uid, "annotation")
        if p is None or not p.exists():
            p = cache / f"assets_std_annotation_flat/std_{uid}.json"
        if p.exists():
            with open(p) as f:
                out[uid] = json.load(f)
        else:
            out[uid] = None
    return out


def load_thumbnails(
    uids: Optional[List[str]] = None,
    num_workers: int = 8,
) -> Dict[str, Optional[Path]]:
    """Download thumbnail images and return their local paths.

    Args:
        uids:        List of UIDs.  ``None`` → all assets.
        num_workers: Parallel download threads (default 8).

    Returns:
        ``{uid: Path | None}``  (paths point to ``.png`` files)

    Example::

        thumbs = uva.object.load_thumbnails(["uid1"])
        img    = thumbs["uid1"]
    """
    if uids is None:
        uids = get_uids_all()

    if not uids:
        return {}

    repo_paths = [
        rp for uid in uids
        if (rp := _core.get_bucket_path(uid, "thumbnail")) is not None
    ]
    _core._download_files_parallel(repo_paths, num_workers, desc="Thumbnails")

    result: Dict[str, Optional[Path]] = {}
    for uid in uids:
        p = _uid_local_path(uid, "thumbnail")
        result[uid] = p if p and p.exists() else None
    return result


def load_renders(
    uids: Optional[List[str]] = None,
    angles: Tuple[int, ...] = (0, 90, 180, 270),
    num_workers: int = 8,
) -> Dict[str, Dict[int, Optional[Path]]]:
    """Download multi-angle render images and return their local paths.

    Args:
        uids:        List of UIDs.  ``None`` → all assets.
        angles:      Angles to download.  Must be a subset of ``(0, 90, 180, 270)``.
        num_workers: Parallel download threads (default 8).

    Returns:
        ``{uid: {0: Path|None, 90: Path|None, 180: Path|None, 270: Path|None}}``
        Only keys in ``angles`` are present.

    Raises:
        ValueError: If ``angles`` contains values outside ``{0, 90, 180, 270}``.

    Example::

        renders = uva.object.load_renders(["uid1"], angles=(0, 180))
        front = renders["uid1"][0]
        back  = renders["uid1"][180]
    """
    valid_angles = frozenset(_RENDER_ANGLES)
    invalid = frozenset(angles) - valid_angles
    if invalid:
        raise ValueError(
            f"Invalid render angle(s): {sorted(invalid)}. "
            f"Must be a subset of {sorted(valid_angles)}."
        )

    if uids is None:
        uids = get_uids_all()

    if not uids:
        return {}

    _core._download_and_extract_renders(uids, num_workers)

    result: Dict[str, Dict[int, Optional[Path]]] = {}
    for uid in uids:
        render_dir = _core.render_extract_dir(uid)
        renders: Dict[int, Optional[Path]] = {}
        for a in angles:
            p = render_dir / f"render_{float(a)}.jpg" if render_dir else None
            renders[a] = p if p and p.exists() else None
        result[uid] = renders
    return result


# ══════════════════════════════════════════════════════════════════════════════
# GLB → USD conversion for IsaacSim
# ══════════════════════════════════════════════════════════════════════════════

_GLB_TO_USD_SCRIPT = Path(__file__).parent / "_glb_to_usd.py"


def convert_glb_to_usd(
    uids: Optional[List[str]] = None,
    num_workers: int = 8,
    headless: bool = True,
    collision_approximation: str = "convexDecomposition",
    make_instanceable: bool = False,
    mass: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert standardised GLB assets to IsaacSim-ready USD format.

    **Requires Isaac Lab / Isaac Sim** to be installed.

    For each UID the function:

    1. Downloads the ``.glb`` file if it is not already cached.
    2. Converts it using Isaac Lab's ``MeshConverter``.
    3. Saves the result under
       ``assets_std_usd_flat/std_{UID}/std_{UID}.usd``.

    Already-converted assets (where the ``.usd`` file exists) are skipped.

    Args:
        uids:        UIDs to convert.  ``None`` → all assets.
        num_workers: Parallel download threads for fetching GLBs (default 8).
        headless:    Run Isaac Sim in headless mode (default ``True``).
        collision_approximation: Collision mesh method
                     (default ``"convexDecomposition"``).
        make_instanceable: Make assets instanceable (default ``False``).
        mass:        Mass in kg to assign to assets (default ``None``).

    Returns:
        ``{uid: Path | None}`` — absolute path to ``.usd`` or ``None``
        on failure.

    Example::

        result = uva.object.convert_glb_to_usd(["uid1", "uid2"])
        usd_path = result["uid1"]
    """
    import importlib
    import shutil
    import subprocess
    import sys

    _has_isaacsim = importlib.util.find_spec("isaacsim") is not None
    _has_isaaclab = importlib.util.find_spec("isaaclab") is not None
    _isaaclab_cli = shutil.which("isaaclab")

    if _has_isaacsim and _has_isaaclab:
        _launch_mode = "python"
    elif _isaaclab_cli is not None:
        _launch_mode = "cli"
    else:
        _missing = []
        if not _has_isaacsim:
            _missing.append("isaacsim")
        if not _has_isaaclab:
            _missing.append("isaaclab")
        print(
            f"[urbanverse] GLB->USD conversion requires Isaac Sim + Isaac Lab,\n"
            f"  but the following were not found: {', '.join(_missing)}\n"
            f"  (and no 'isaaclab' CLI wrapper was found on PATH)\n"
            f"\n"
            f"  To install, run:\n"
            f"    pip install --upgrade pip\n"
            f"    pip install 'isaacsim[all,extscache]==4.5.0'"
            f" --extra-index-url https://pypi.nvidia.com\n"
            f"\n"
            f"    cd urbanverse_asset/IsaacLab\n"
            f"    ./isaaclab.sh --install\n"
        )
        return {}

    if uids is None:
        uids = get_uids_all()
    if not uids:
        return {}

    glb_paths = load_assets(uids, num_workers=num_workers)
    cache = _core.get_cache_dir()
    usd_base = cache / "assets_std_usd_flat"

    tasks = []
    pre_existing: Dict[str, Path] = {}
    missing_glb: List[str] = []

    for uid in uids:
        glb = glb_paths.get(uid)
        if glb is None or not glb.exists():
            missing_glb.append(uid)
            continue

        usd_dir = usd_base / f"std_{uid}"
        usd_name = f"std_{uid}.usd"
        usd_file = usd_dir / usd_name

        if usd_file.exists():
            pre_existing[uid] = usd_file
        else:
            tasks.append({
                "uid": uid,
                "glb": str(glb.resolve()),
                "usd_dir": str(usd_dir.resolve()),
                "usd_name": usd_name,
            })

    if missing_glb:
        print(
            f"[urbanverse] GLB not found for {len(missing_glb)} UID(s), "
            f"skipping conversion"
        )

    if not tasks:
        print(
            f"[urbanverse] GLB->USD: 0 to convert, "
            f"{len(pre_existing)} cached, "
            f"{len(missing_glb)} missing (of {len(uids)} total)"
        )
        result: Dict[str, Any] = {uid: pre_existing.get(uid) for uid in uids}
        return result

    tasks_file = cache / "_glb_to_usd_tasks.json"
    task_data = {"tasks": tasks}
    tasks_file.write_text(json.dumps(task_data, indent=2))

    if _launch_mode == "python":
        cmd = [sys.executable, str(_GLB_TO_USD_SCRIPT)]
    else:
        cmd = [_isaaclab_cli, "-p", str(_GLB_TO_USD_SCRIPT)]

    cmd.extend(["--tasks", str(tasks_file)])
    cmd.extend(["--collision-approximation", collision_approximation])
    if headless:
        cmd.append("--headless")
    if make_instanceable:
        cmd.append("--make-instanceable")
    if mass is not None:
        cmd.extend(["--mass", str(mass)])

    print(f"[urbanverse] Launching Isaac Lab converter for {len(tasks)} asset(s)...")
    print(f"[urbanverse] Command: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, check=False, capture_output=False)
        if proc.returncode != 0:
            print(f"[urbanverse] Conversion process exited with code {proc.returncode}")
    except FileNotFoundError:
        print(
            "[urbanverse] Could not launch the conversion subprocess.\n"
            f"  The tasks file has been saved — you can run it manually:\n"
            f"    isaaclab -p {_GLB_TO_USD_SCRIPT} "
            f"--tasks {tasks_file} --headless"
        )
        return {uid: pre_existing.get(uid) for uid in uids}

    try:
        result_data = json.loads(tasks_file.read_text())
        results_list = result_data.get("results", [])
    except Exception:
        results_list = []

    converted = 0
    failed = 0
    uid_to_usd: Dict[str, Any] = {}

    for r in results_list:
        glb_path = r.get("glb", "")
        usd_path_str = r.get("usd")
        ok = r.get("ok", False)

        uid_match = None
        for t in tasks:
            if t["glb"] == glb_path:
                uid_match = t["uid"]
                break

        if uid_match is None:
            continue

        if ok and usd_path_str:
            p = Path(usd_path_str)
            if p.exists():
                uid_to_usd[uid_match] = p
                converted += 1
                continue
        uid_to_usd[uid_match] = None
        failed += 1

    result = {}
    for uid in uids:
        if uid in pre_existing:
            result[uid] = pre_existing[uid]
        elif uid in uid_to_usd:
            result[uid] = uid_to_usd[uid]
        else:
            result[uid] = None

    print(
        f"[urbanverse] GLB->USD: {converted} converted, "
        f"{len(pre_existing)} cached, "
        f"{failed + len(missing_glb)} failed/missing "
        f"(of {len(uids)} total)"
    )

    try:
        tasks_file.unlink()
    except OSError:
        pass

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Annotation schema explanation
# ══════════════════════════════════════════════════════════════════════════════

_ANNOTATION_SCHEMA: Dict[str, str] = {
    "description_long":
        "A very detailed visual description of this object that is no more "
        "than 6 sentences, focusing on geometry, texture, and state without "
        "mentioning orientation.",
    "description":
        "A 1-2 sentence summary of the long description that remains rich "
        "and visual.",
    "description_view_0":
        "A short description of this object from view 0, highlighting "
        "features unique to this perspective.",
    "description_view_1":
        "A short description of this object from view 1, highlighting "
        "features unique to this perspective.",
    "description_view_2":
        "A short description of this object from view 2, highlighting "
        "features unique to this perspective.",
    "description_view_3":
        "A short description of this object from view 3, highlighting "
        "features unique to this perspective.",
    "category":
        "The specific L3 (leaf) category name under the UrbanVerse ontology.",
    "height":
        "The approximate height of this object in meters (m).",
    "max_dimension":
        "The approximate maximum dimension of this object in meters (m).",
    "length":
        "The approximate length of this object in meters (m).",
    "width":
        "The approximate width of this object in meters (m).",
    "materials":
        "A Python list of the materials that this object appears to be made "
        "of, including 'air' for non-solid interiors.",
    "materials_composition":
        "A Python list of floats representing the volume mixture of the "
        "materials (must sum to 1.0).",
    "mass":
        "The approximate mass of this object in kilograms (kg).",
    "receptacle":
        "A boolean indicating whether or not this object can contain other "
        "items (True/False).",
    "frontView":
        "The integer render image index of the view (0-3) that represents "
        "the functional front of the object.",
    "quality":
        "A number from 0-10 indicating the visual and geometric quality of "
        "the 3D asset.",
    "movable":
        "A boolean indicating whether this object is physically movable or "
        "fixed in the environment.",
    "required_force":
        "The approximate force required to move or push this object in "
        "Newtons (N).",
    "walkable":
        "A boolean indicating whether agents can walk or stand on this "
        "object.",
    "enterable":
        "A boolean indicating whether agents can physically enter the "
        "interior of this object.",
    "affordances":
        "A Python list of high-level functional affordances such as "
        "['sittable', 'openable', 'drivable'].",
    "support_surface":
        "A boolean indicating whether this object can physically support "
        "other objects placed on it.",
    "interactive_parts":
        "A Python list of distinct functional components that can be "
        "interacted with (e.g., ['handle', 'button']).",
    "traversability":
        "A string label describing traversal logic: 'pass through', "
        "'push through', or 'obstacle'.",
    "traversable_by":
        "A Python list of agent types (e.g., ['person', 'drone']) that can "
        "navigate through the object.",
    "colors":
        "A Python list of the dominant visible colors of this object.",
    "colors_composition":
        "A Python list of floats representing the approximate volume "
        "composition of the visible colors (must sum to 1.0).",
    "surface_hardness":
        "A string describing the tactile hardness: 'soft', 'semi-soft', or "
        "'hard'.",
    "surface_roughness":
        "A float in the range [0, 1] indicating the micro-texture roughness "
        "(0 is smooth, 1 is coarse).",
    "surface_finish":
        "A string describing the surface quality: 'rough', 'matte', "
        "'smooth', 'glossy', 'sleek', or 'grippy'.",
    "reflectivity":
        "A float in the range [0, 1] representing the amount of light "
        "reflected (1.0 is mirror-like).",
    "index_of_refraction":
        "A float representing the optical index of refraction (IOR) of the "
        "surface material.",
    "youngs_modulus":
        "The approximate material stiffness of this object in Megapascals "
        "(MPa).",
    "friction_coefficient":
        "A positive float representing the estimated friction coefficient "
        "based on material and finish.",
    "bounciness":
        "A float in the range [0, 1] representing the expected elasticity "
        "of the object upon impact.",
    "recommended_clearance":
        "The approximate safe buffer distance surrounding this object in "
        "meters (m).",
    "asset_composition_type":
        "The structural nature of the asset: 'single', 'group', or 'scene'.",
    "attribute_<class_specific>":
        "One of five category-specific annotations that capture distinct "
        "physical, functional, or structural characteristics unique to the "
        "object's CLASS (e.g., 'backrest_presence' for seating or "
        "'mounting_type' for fixtures).",
    "uid":
        "The unique identifier string for this specific asset.",
    "near_synsets":
        "A Python dictionary mapping related WordNet synsets (keys) to their "
        "semantic similarity scores (values).",
    "synset":
        "The primary WordNet synset ID associated with this object such as "
        "'workbench.n.01'.",
    "wn_version":
        "The version of WordNet used for the synset mapping.",
    "annotation_info":
        "The textual and visual foundation model names used for the "
        "annotation process, saved in a dict.",
    "license_info":
        "Specific license information of the 3D asset, saved in a dict.",
    "filename":
        "The exact name of the source .glb file for this asset.",
}


def _print_schema(schema: Dict[str, str]) -> None:
    """Pretty-print annotation field explanations to stdout."""
    header = "UrbanVerse Per-Asset Annotation Schema"
    print(f"\n{'=' * 72}")
    print(f"  {header}")
    print(f"{'=' * 72}")
    print(f"  Total fields: {len(schema)}\n")

    max_key = max(len(k) for k in schema)
    for key, desc in schema.items():
        print(f"  {key:<{max_key}}  —  {desc}")

    print(f"\n{'=' * 72}\n")


def explain_annotation(field: Optional[str] = None) -> Dict[str, str]:
    """Print and return an explanation of each field in the per-asset annotation.

    When called with no arguments, prints an overview of every annotation
    field and returns the full schema dictionary.  When ``field`` is given,
    prints only that field's explanation and returns a single-entry dict.

    Args:
        field: A specific annotation field name (e.g. ``"mass"``).
               ``None`` (default) prints all fields.

    Returns:
        ``{field_name: explanation_string, ...}``

    Raises:
        ValueError: If ``field`` is not a recognized annotation key.

    Example::

        uva.object.explain_annotation()
        uva.object.explain_annotation("mass")
    """
    if field is not None:
        matched = {k: v for k, v in _ANNOTATION_SCHEMA.items() if k == field}
        if not matched:
            if field.startswith("attribute_"):
                matched = {field: _ANNOTATION_SCHEMA["attribute_<class_specific>"]}
            else:
                raise ValueError(
                    f"Unknown annotation field: {field!r}\n"
                    f"  Valid fields: {sorted(_ANNOTATION_SCHEMA.keys())}"
                )
        _print_schema(matched)
        return matched

    _print_schema(_ANNOTATION_SCHEMA)
    return dict(_ANNOTATION_SCHEMA)
