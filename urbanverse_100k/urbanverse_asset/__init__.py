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
urbanverse_asset — Python package for the UrbanVerse-100K 3D asset dataset.

Hosted at: ``Oatmealliu/UrbanVerse-100K`` (HuggingFace Datasets)

Quick-start
-----------
::

    import urbanverse_asset as uva

    # (Optional) override where files are saved
    uva.set("~/my_data/urbanverse")

    # Explore the dataset
    uva.info()
    uva.object.categories(level=1)

    # Get UIDs
    all_uids = uva.object.get_uids_all()

    # Filter by attributes
    uids = uva.object.get_uids_conditioned(
        categories=["vehicle"],
        height_range=(1.5, 3.0),
        movable=True,
    )

    # Download assets
    result = uva.object.load(uids[:5])
    glb    = result["<uid>"]["std_glb"]   # Path to .glb on disk

    # Materials
    uva.sky.get_descriptions()
    uva.sky.load_materials()

Notes
-----
- Files are cached in ``~/.cache/urbanverse/`` by default. Call ``uva.set(path)``
  to change this before any download.
- Set the ``HF_TOKEN`` environment variable (or pass ``token=`` to any call)
  to authenticate with HuggingFace for private repos.
- All download functions skip files that are already cached — re-running is
  always safe.
"""

from typing import Any, Dict, List

from . import _core, object, sky, road, sidewalk, terrain, plant, shrub, tree, viewer

__version__ = "0.1.0"
__all__ = [
    "set", "info", "download_all", "check_integrity", "repair",
    "object",
    "sky", "road", "sidewalk", "terrain",
    "plant", "shrub", "tree",
    "viewer",
]


# ══════════════════════════════════════════════════════════════════════════════
# Configuration & dataset metadata
# ══════════════════════════════════════════════════════════════════════════════

def set(path: str) -> None:  # noqa: A001  (shadows built-in set intentionally)
    """Set the local directory where all UrbanVerse data will be saved.

    The path is persisted in ``~/.cache/urbanverse_config.json`` so it survives
    Python restarts.  If never called, defaults to ``~/.cache/urbanverse/``.

    Args:
        path: Absolute or ``~``-relative path to the desired cache directory.
              The directory is created if it does not exist.

    Example::

        uva.set("~/datasets/urbanverse")
        uva.set("/mnt/ssd/urbanverse")
    """
    _core.set_cache_dir(path)
    print(f"[urbanverse] Cache directory → {_core.get_cache_dir()}")


def info() -> Dict[str, Any]:
    """Return dataset-level metadata (version, counts, license, description).

    Downloads ``urbanverse_master_annotation.json`` on first call; subsequent
    calls are instant (in-memory cache).

    Returns:
        dict with keys ``version``, ``date_created``, ``description``,
        ``license``, ``contributor``, ``url``, ``citation``, and
        ``statistics`` (``number_of_assets``, ``number_of_classes_l1/l2/l3``).

    Example::

        meta = uva.info()
        print(meta["statistics"]["number_of_assets"])  # 102444
        print(meta["license"])                         # "CC BY-NC-SA 4.0"
    """
    ann = _core.load_master_annotation()
    return {**ann["about"], "statistics": ann["statistics"]}


# ══════════════════════════════════════════════════════════════════════════════
# Bulk download — everything in the HuggingFace repo
# ══════════════════════════════════════════════════════════════════════════════

def download_all(num_workers: int = 8) -> None:
    """Download the entire UrbanVerse-100K dataset to the local cache.

    Downloads **everything** in the HuggingFace repo: 3D object assets
    (GLB, annotations, thumbnails, renders), sky/road/sidewalk/terrain
    materials, and plant/shrub/tree vegetation assets.

    Args:
        num_workers: Parallel download threads (default 8).

    Example::

        uva.download_all()
        uva.download_all(num_workers=16)
    """
    print("[urbanverse] Starting full dataset download (all content)…")

    uids = object.get_uids_all()

    # Annotations — download bundle tar.gz and extract (much faster than
    # downloading ~102k individual small JSON files)
    _core._ensure_per_asset_annotations()

    # GLB files
    glb_paths = [
        rp for uid in uids
        if (rp := _core.get_bucket_path(uid, "glb")) is not None
    ]
    if glb_paths:
        print(f"[urbanverse] {len(glb_paths)} GLB file(s) queued…")
        _core._download_files_parallel(glb_paths, num_workers, desc="GLB models")

    # Thumbnails
    thumb_paths = [
        rp for uid in uids
        if (rp := _core.get_bucket_path(uid, "thumbnail")) is not None
    ]
    if thumb_paths:
        print(f"[urbanverse] {len(thumb_paths)} thumbnail(s) queued…")
        _core._download_files_parallel(thumb_paths, num_workers, desc="Thumbnails")

    # Render archives
    _core._download_and_extract_renders(uids, num_workers)

    # Materials
    sky.load_materials(num_workers=num_workers)
    road.load_materials(num_workers=num_workers)
    sidewalk.load_materials(num_workers=num_workers)
    terrain.load_materials(num_workers=num_workers)

    # Vegetation
    plant.load_materials(num_workers=num_workers)
    shrub.load_materials(num_workers=num_workers)
    tree.load_materials(num_workers=num_workers)

    print("[urbanverse] Download complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Integrity checking & repair
# ══════════════════════════════════════════════════════════════════════════════

def _check_annotation_status() -> tuple:
    """Return ``(expected, downloaded, missing_uids)`` for per-asset annotations."""
    index = _core._load_bucket_index("annotation")
    cache = _core.get_cache_dir()
    expected = len(index)
    missing_uids: List[str] = []
    for uid, repo_path in index.items():
        if (cache / repo_path).exists():
            continue
        if (cache / f"assets_std_annotation_flat/std_{uid}.json").exists():
            continue
        missing_uids.append(uid)
    return expected, expected - len(missing_uids), missing_uids


def _check_material_status(source_cls: type) -> tuple:
    """Return ``(expected, downloaded, missing_descriptions)`` for a material type."""
    file_map = source_cls._get_file_map()
    cache = _core.get_cache_dir()
    expected = len(file_map)
    missing_descs: List[str] = []
    for desc, entry in file_map.items():
        main_files = entry.get("main_files", [])
        thumb = entry.get("thumbnail")
        all_present = True
        for f in main_files:
            if not (cache / f).exists():
                all_present = False
                break
        if all_present and thumb and not (cache / thumb).exists():
            all_present = False
        if not all_present:
            missing_descs.append(desc)
    return expected, expected - len(missing_descs), missing_descs


def _check_vegetation_status(source_cls: type) -> tuple:
    """Return ``(expected, downloaded, missing_descriptions)`` for a vegetation type."""
    file_map = source_cls._get_file_map()
    cache = _core.get_cache_dir()
    expected = len(file_map)
    missing_descs: List[str] = []
    for desc in file_map:
        extract_dir = cache / source_cls._folder / desc
        if not (extract_dir.exists() and extract_dir.is_dir()):
            missing_descs.append(desc)
    return expected, expected - len(missing_descs), missing_descs


def _build_integrity_report() -> tuple:
    """Scan the local cache and build a full integrity report.

    Returns:
        ``(report_dict, missing_dict)`` — *report_dict* is the user-facing
        summary; *missing_dict* contains lists needed by :func:`repair`.
    """
    cache = _core.get_cache_dir()

    # ── Object assets ────────────────────────────────────────────────────────
    ann_exp, ann_dl, ann_missing_uids = _check_annotation_status()

    glb_index = _core._load_bucket_index("glb")
    glb_exp, glb_dl, glb_missing_paths = _core._count_local_bucket_files(glb_index)

    thumb_index = _core._load_bucket_index("thumbnail")
    thumb_exp, thumb_dl, thumb_missing_paths = _core._count_local_bucket_files(thumb_index)

    render_index = _core._load_bucket_index("render")
    render_exp, render_dl, render_missing_uids = _core._count_local_render_uids(render_index)

    # ── Materials ────────────────────────────────────────────────────────────
    sky_exp, sky_dl, sky_missing = _check_material_status(sky._Sky)
    road_exp, road_dl, road_missing = _check_material_status(road._Road)
    sw_exp, sw_dl, sw_missing = _check_material_status(sidewalk._Sidewalk)
    ter_exp, ter_dl, ter_missing = _check_material_status(terrain._Terrain)

    # ── Vegetation ───────────────────────────────────────────────────────────
    pl_exp, pl_dl, pl_missing = _check_vegetation_status(plant._Plant)
    sh_exp, sh_dl, sh_missing = _check_vegetation_status(shrub._Shrub)
    tr_exp, tr_dl, tr_missing = _check_vegetation_status(tree._Tree)

    # ── Assemble report ─────────────────────────────────────────────────────
    def _entry(exp: int, dl: int) -> Dict[str, int]:
        return {"expected": exp, "downloaded": dl, "missing": exp - dl}

    total_missing = (
        (ann_exp - ann_dl) + (glb_exp - glb_dl) + (thumb_exp - thumb_dl)
        + (render_exp - render_dl)
        + (sky_exp - sky_dl) + (road_exp - road_dl) + (sw_exp - sw_dl) + (ter_exp - ter_dl)
        + (pl_exp - pl_dl) + (sh_exp - sh_dl) + (tr_exp - tr_dl)
    )

    report: Dict[str, Any] = {
        "cache_dir": str(cache),
        "object": {
            "annotations": _entry(ann_exp, ann_dl),
            "glb":         _entry(glb_exp, glb_dl),
            "thumbnails":  _entry(thumb_exp, thumb_dl),
            "renders":     _entry(render_exp, render_dl),
        },
        "sky":       _entry(sky_exp, sky_dl),
        "road":      _entry(road_exp, road_dl),
        "sidewalk":  _entry(sw_exp, sw_dl),
        "terrain":   _entry(ter_exp, ter_dl),
        "plant":     _entry(pl_exp, pl_dl),
        "shrub":     _entry(sh_exp, sh_dl),
        "tree":      _entry(tr_exp, tr_dl),
        "complete":       total_missing == 0,
        "total_missing":  total_missing,
    }

    missing = {
        "annotation_uids":   ann_missing_uids,
        "glb_paths":         glb_missing_paths,
        "thumbnail_paths":   thumb_missing_paths,
        "render_uids":       render_missing_uids,
        "sky_descs":         sky_missing,
        "road_descs":        road_missing,
        "sidewalk_descs":    sw_missing,
        "terrain_descs":     ter_missing,
        "plant_descs":       pl_missing,
        "shrub_descs":       sh_missing,
        "tree_descs":        tr_missing,
    }

    return report, missing


def _print_integrity_report(report: Dict[str, Any]) -> None:
    """Print a formatted integrity summary to stdout."""
    print(f"\n[urbanverse] Integrity report for {report['cache_dir']}")

    rows = [
        ("Object annotations", report["object"]["annotations"]),
        ("Object GLB",         report["object"]["glb"]),
        ("Object thumbnails",  report["object"]["thumbnails"]),
        ("Object renders",     report["object"]["renders"]),
        ("Sky materials",      report["sky"]),
        ("Road materials",     report["road"]),
        ("Sidewalk materials", report["sidewalk"]),
        ("Terrain materials",  report["terrain"]),
        ("Plant assets",       report["plant"]),
        ("Shrub assets",       report["shrub"]),
        ("Tree assets",        report["tree"]),
    ]

    label_width = max(len(label) for label, _ in rows)
    exp_width = max(len(str(e["expected"])) for _, e in rows)

    for label, entry in rows:
        dl  = entry["downloaded"]
        exp = entry["expected"]
        m   = entry["missing"]
        status = "OK" if m == 0 else f"({m} missing)"
        print(
            f"  {label:<{label_width}}  "
            f"{dl:>{exp_width}} / {exp:<{exp_width}}  {status}"
        )

    total = report["total_missing"]
    if total == 0:
        print(f"\n  Overall: all files present — dataset is complete.")
    else:
        print(f"\n  Overall: {total} file(s) missing — run uva.repair() to fix.")
    print()


def check_integrity() -> Dict[str, Any]:
    """Scan the local cache and report download completeness per asset type.

    Compares files on disk against the expected file lists from the
    HuggingFace repo.  **No files are downloaded** (except for the small
    bucket-index JSONs and master annotation, which are needed to know
    what to check).

    Returns:
        A dict with per-category counts (``expected``, ``downloaded``,
        ``missing``), a boolean ``complete`` flag, and a ``total_missing``
        count.  See :func:`repair` to automatically fix missing files.

    Example::

        report = uva.check_integrity()
        print(report["object"]["glb"])  # {"expected": 102444, "downloaded": 102444, "missing": 0}
        print(report["complete"])       # True / False
    """
    report, _ = _build_integrity_report()
    _print_integrity_report(report)
    return report


def repair(num_workers: int = 8) -> Dict[str, Any]:
    """Download only the missing files identified by :func:`check_integrity`.

    Runs an integrity scan first.  If all files are already present, prints
    a confirmation and returns immediately.  Otherwise, selectively downloads
    the missing items and re-checks integrity afterwards.

    Args:
        num_workers: Parallel download threads (default 8).

    Returns:
        The post-repair integrity report dict (same structure as
        :func:`check_integrity`).

    Example::

        report = uva.repair()
        assert report["complete"]
    """
    print("[urbanverse] Checking integrity before repair…")
    report, missing = _build_integrity_report()
    _print_integrity_report(report)

    if report["complete"]:
        print("[urbanverse] Nothing to repair — dataset is complete.")
        return report

    print(f"[urbanverse] Repairing {report['total_missing']} missing file(s)…\n")

    # ── Object annotations ───────────────────────────────────────────────────
    if missing["annotation_uids"]:
        _core._ensure_per_asset_annotations()

    # ── Object GLB ───────────────────────────────────────────────────────────
    if missing["glb_paths"]:
        print(f"[urbanverse] {len(missing['glb_paths'])} GLB file(s) to repair…")
        _core._download_files_parallel(
            missing["glb_paths"], num_workers, desc="Repair GLB"
        )

    # ── Object thumbnails ────────────────────────────────────────────────────
    if missing["thumbnail_paths"]:
        print(f"[urbanverse] {len(missing['thumbnail_paths'])} thumbnail(s) to repair…")
        _core._download_files_parallel(
            missing["thumbnail_paths"], num_workers, desc="Repair thumbnails"
        )

    # ── Object renders ───────────────────────────────────────────────────────
    if missing["render_uids"]:
        print(f"[urbanverse] {len(missing['render_uids'])} render(s) to repair…")
        _core._download_and_extract_renders(missing["render_uids"], num_workers)

    # ── Materials ────────────────────────────────────────────────────────────
    if missing["sky_descs"]:
        sky.load_materials(missing["sky_descs"], num_workers=num_workers)
    if missing["road_descs"]:
        road.load_materials(missing["road_descs"], num_workers=num_workers)
    if missing["sidewalk_descs"]:
        sidewalk.load_materials(missing["sidewalk_descs"], num_workers=num_workers)
    if missing["terrain_descs"]:
        terrain.load_materials(missing["terrain_descs"], num_workers=num_workers)

    # ── Vegetation ───────────────────────────────────────────────────────────
    if missing["plant_descs"]:
        plant.load_materials(missing["plant_descs"], num_workers=num_workers)
    if missing["shrub_descs"]:
        shrub.load_materials(missing["shrub_descs"], num_workers=num_workers)
    if missing["tree_descs"]:
        tree.load_materials(missing["tree_descs"], num_workers=num_workers)

    # ── Re-check ─────────────────────────────────────────────────────────────
    print("\n[urbanverse] Repair complete. Re-checking integrity…")
    final_report, _ = _build_integrity_report()
    _print_integrity_report(final_report)

    if final_report["complete"]:
        print("[urbanverse] All files present — dataset is complete.")
    else:
        remaining = final_report["total_missing"]
        print(
            f"[urbanverse] {remaining} file(s) still missing after repair.\n"
            f"  This may be due to network errors. Run uva.repair() again to retry."
        )

    return final_report
