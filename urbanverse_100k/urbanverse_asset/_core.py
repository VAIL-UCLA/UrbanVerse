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
Shared state, HF configuration, cache management, and download utilities.

All other modules import from here — this is the single source of truth for:
  - Which HF repo to use
  - Where files are cached locally
  - How to download files (parallel, rate-limit-aware, resumable)
  - The in-memory master annotation cache
"""

import json
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# HF repo — change this constant when switching from toy → full dataset
# ──────────────────────────────────────────────────────────────────────────────
REPO_ID   = "Oatmealliu/UrbanVerse-100K"
REPO_TYPE = "dataset"

# ──────────────────────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_CACHE = Path.home() / ".cache" / "urbanverse"
_CONFIG_FILE   = Path.home() / ".cache" / "urbanverse_config.json"

# ──────────────────────────────────────────────────────────────────────────────
# Bucket index filenames (UID → bucketed repo path)
# ──────────────────────────────────────────────────────────────────────────────
_BUCKET_INDEX_FILES = {
    "glb":        "bucket_index_glb.json",
    "annotation": "bucket_index_annotation.json",
    "thumbnail":  "bucket_index_thumbnail.json",
    "render":     "bucket_index_render.json",
}

# ──────────────────────────────────────────────────────────────────────────────
# Mutable session state (module-level singleton)
# ──────────────────────────────────────────────────────────────────────────────
_state: Dict = {
    "cache_dir":        None,   # resolved lazily
    "token":            None,   # HF auth token (optional for public repos)
    "_master_ann":      None,   # in-memory cache of master annotation dict
    "_all_uids":        None,   # in-memory cache of all UIDs
    "_per_asset_ann":   {},     # uid → annotation dict  (lazily populated)
    "_repo_files":      None,   # cached list of all HF repo file paths
    "_bucket_indices":  {},     # key → {uid: repo_path}  (lazily populated)
}
_state_lock = Lock()


# ──────────────────────────────────────────────────────────────────────────────
# Cache directory management
# ──────────────────────────────────────────────────────────────────────────────

def get_cache_dir() -> Path:
    """Return the current local cache directory, creating it if needed."""
    with _state_lock:
        if _state["cache_dir"] is not None:
            return _state["cache_dir"]

    # Try loading from persisted config
    if _CONFIG_FILE.exists():
        try:
            cfg = json.loads(_CONFIG_FILE.read_text())
            if "cache_dir" in cfg:
                p = Path(cfg["cache_dir"])
                with _state_lock:
                    _state["cache_dir"] = p
                return p
        except Exception:
            pass

    # Fall back to default
    with _state_lock:
        _state["cache_dir"] = _DEFAULT_CACHE
    _DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_CACHE


def set_cache_dir(path: Union[str, Path]) -> None:
    """Set and persist the local cache directory."""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    with _state_lock:
        _state["cache_dir"] = p
        # Invalidate in-memory file references (paths changed)
        _state["_master_ann"]      = None
        _state["_all_uids"]        = None
        _state["_per_asset_ann"]   = {}
        _state["_repo_files"]      = None
        _state["_bucket_indices"]  = {}
        # Clear per-folder file caches
        for k in [k for k in _state if k.startswith("_folder_files:")]:
            del _state[k]

    # Persist to config file
    try:
        cfg: Dict = {}
        if _CONFIG_FILE.exists():
            cfg = json.loads(_CONFIG_FILE.read_text())
        cfg["cache_dir"] = str(p)
        _CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass  # non-fatal — cache dir still updated in memory


def get_token() -> Optional[str]:
    """Return HF auth token from session state or HF_TOKEN env var."""
    return _state["token"] or os.environ.get("HF_TOKEN")


# ──────────────────────────────────────────────────────────────────────────────
# Rate limiter — shared across all worker threads
# ──────────────────────────────────────────────────────────────────────────────

class _RateLimiter:
    def __init__(self) -> None:
        self._lock           = Lock()
        self._cooldown_until = 0.0

    def set_cooldown(self, seconds: float) -> None:
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, time.time() + seconds)

    def wait(self) -> None:
        while True:
            with self._lock:
                remaining = self._cooldown_until - time.time()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 1.0))


_rate_limiter = _RateLimiter()


# ──────────────────────────────────────────────────────────────────────────────
# Per-file download locks
#
# Prevents two threads from downloading the same file simultaneously.
# Scenario: user calls load_assets() and load_thumbnails() concurrently for
# overlapping UIDs — both would pass the local_path.exists() check at the same
# time and attempt redundant (or conflicting) downloads of the same file.
#
# With these locks, the second thread waits, then finds the file already on
# disk and returns immediately — no duplicate network traffic, no conflicts.
# ──────────────────────────────────────────────────────────────────────────────

_file_locks: Dict[str, Lock] = {}
_file_locks_lock = Lock()


def _get_file_lock(repo_path: str) -> Lock:
    with _file_locks_lock:
        if repo_path not in _file_locks:
            _file_locks[repo_path] = Lock()
        return _file_locks[repo_path]


# ──────────────────────────────────────────────────────────────────────────────
# Single-file download
# ──────────────────────────────────────────────────────────────────────────────

def _download_file(repo_path: str, max_retries: int = 5) -> Path:
    """Download one file from HF → local cache. No-ops if already present.

    Thread-safe: a per-file lock ensures only one thread downloads a given
    path at a time. hf_hub_download writes to a temp file first, then renames
    atomically, so a killed process never leaves a partial file on disk —
    the existence check on the final path is always safe for resume.

    Returns the local Path. Raises on permanent errors (404, auth failure).
    """
    cache_dir  = get_cache_dir()
    local_path = cache_dir / repo_path

    # Fast path — no lock needed for already-cached files
    if local_path.exists():
        return local_path

    # Acquire per-file lock before downloading so concurrent calls for the
    # same path don't race past the existence check above.
    file_lock = _get_file_lock(repo_path)
    with file_lock:
        # Re-check inside the lock: another thread may have finished while we
        # were waiting.
        if local_path.exists():
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        delay = 2.0
        for attempt in range(1, max_retries + 1):
            _rate_limiter.wait()
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=repo_path,
                    repo_type=REPO_TYPE,
                    local_dir=str(cache_dir),
                    token=get_token(),
                )
                return local_path
            except HfHubHTTPError as e:
                if e.response is not None:
                    status = e.response.status_code
                    if status == 429:
                        retry_after = float(e.response.headers.get("Retry-After", 60))
                        print(
                            f"[urbanverse] rate limit — 429 received, "
                            f"cooling down {retry_after:.0f}s (all workers paused)"
                        )
                        _rate_limiter.set_cooldown(retry_after)
                        continue   # don't count as a retry attempt
                    if 400 <= status < 500:
                        raise      # permanent failure (404, 401, 403 …)
                if attempt == max_retries:
                    raise
                print(f"[urbanverse] retry {attempt}/{max_retries} — {repo_path}: {e}")
                time.sleep(delay)
                delay = min(delay * 2, 60)
            except Exception as e:
                if attempt == max_retries:
                    raise
                print(f"[urbanverse] retry {attempt}/{max_retries} — {repo_path}: {e}")
                time.sleep(delay)
                delay = min(delay * 2, 60)

    return local_path  # unreachable, but satisfies type checkers


# ──────────────────────────────────────────────────────────────────────────────
# Parallel batch download
# ──────────────────────────────────────────────────────────────────────────────

def _download_files_parallel(
    repo_paths: List[str],
    num_workers: int = 8,
    desc: str = "Downloading",
) -> Dict[str, Union[Path, Exception]]:
    """Download a list of repo paths in parallel.

    Already-cached files are skipped instantly (no network request).

    Returns:
        {repo_path: local Path}  on success
        {repo_path: Exception}   on failure
    """
    # Pre-filter paths that already exist to avoid spinning up pool workers
    pending  = [rp for rp in repo_paths if not (get_cache_dir() / rp).exists()]
    results: Dict[str, Union[Path, Exception]] = {
        rp: get_cache_dir() / rp
        for rp in repo_paths
        if rp not in pending
    }

    if not pending:
        return results

    failed  = 0
    success = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_download_file, rp): rp for rp in pending}
        completed_iter: Any = as_completed(futures)
        if _TQDM_AVAILABLE:
            completed_iter = tqdm(
                completed_iter, total=len(futures),
                desc=desc, unit="file", dynamic_ncols=True,
            )
        for future in completed_iter:
            rp = futures[future]
            try:
                results[rp] = future.result()
                success += 1
            except Exception as e:
                results[rp] = e
                failed += 1

    if failed:
        print(f"[urbanverse] ⚠  {failed} file(s) failed to download. Re-run to retry.")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Master annotation
# ──────────────────────────────────────────────────────────────────────────────

MASTER_ANN_REPO_PATH = "urbanverse_master_annotation.json"


def load_master_annotation() -> Dict:
    """Download (once) and return the full master annotation dict.

    Result is cached in memory for the session lifetime.
    """
    with _state_lock:
        if _state["_master_ann"] is not None:
            return _state["_master_ann"]

    local_path = _download_file(MASTER_ANN_REPO_PATH)
    with open(local_path) as f:
        ann = json.load(f)

    with _state_lock:
        _state["_master_ann"] = ann
    return ann


def invalidate_master_annotation() -> None:
    """Force re-download of master annotation on next access (e.g. after a dataset update)."""
    with _state_lock:
        _state["_master_ann"] = None
        local = get_cache_dir() / MASTER_ANN_REPO_PATH
    if local.exists():
        local.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# Per-asset annotation bundle
#
# Downloading 100K individual JSON files is slow even in parallel.  The HF repo
# ships a single compressed archive, full_per_asset_annotations.tar.gz, that
# contains every std_*.json file.  We download it once per cache directory,
# extract it, and write a marker file so subsequent calls are instant.
#
# Fallback: if the bundle does not exist in the repo (e.g. during development
# with the toy dataset), we fall back to individual per-file downloads so the
# API still works correctly — just more slowly.
# ──────────────────────────────────────────────────────────────────────────────

ANNOTATIONS_BUNDLE_REPO_PATH = "full_per_asset_annotations.tar.gz"
_ANNOTATIONS_MARKER_RELPATH  = "assets_std_annotation_flat/.bundle_extracted"

# Ensures only one thread runs the download+extract at a time
_bundle_lock = Lock()


def _ensure_per_asset_annotations() -> bool:
    """Guarantee all per-asset annotation JSONs are present in the cache.

    On first call for a given cache directory:
      1. Downloads ``full_per_asset_annotations.tar.gz`` from the HF repo.
      2. Extracts every ``std_*.json`` into ``assets_std_annotation_flat/``.
      3. Writes a marker file so the next call is a single stat() check.

    Automatically re-triggers after ``uva.set(path)`` because the new
    directory will not contain the marker file.

    Returns:
        ``True``  — bundle was (or is now) extracted; all JSONs are on disk.
        ``False`` — bundle is not available in this repo (404); caller should
                    fall back to individual per-file downloads.
    """
    cache_dir = get_cache_dir()
    marker    = cache_dir / _ANNOTATIONS_MARKER_RELPATH

    # Fast path — marker exists, nothing to do
    if marker.exists():
        return True

    with _bundle_lock:
        # Re-check inside the lock (another thread may have finished)
        if marker.exists():
            return True

        # ── Step 1: download the bundle ──────────────────────────────────────
        bundle_path = cache_dir / ANNOTATIONS_BUNDLE_REPO_PATH
        if not bundle_path.exists():
            print(
                f"[urbanverse] Downloading annotation bundle "
                f"({ANNOTATIONS_BUNDLE_REPO_PATH}) — one-time download, "
                f"may take a moment …"
            )
            try:
                _download_file(ANNOTATIONS_BUNDLE_REPO_PATH)
            except HfHubHTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    print(
                        "[urbanverse] ⚠  Annotation bundle not found in this repo "
                        "(full_per_asset_annotations.tar.gz). "
                        "Falling back to individual per-file downloads."
                    )
                    return False
                raise

        # ── Step 2: extract ──────────────────────────────────────────────────
        ann_dir = cache_dir / "assets_std_annotation_flat"
        ann_dir.mkdir(parents=True, exist_ok=True)
        print(f"[urbanverse] Extracting annotation bundle → {ann_dir} …")

        with tarfile.open(bundle_path, "r:gz") as tf:
            members = tf.getmembers()

            # Skip macOS AppleDouble metadata files (._*) that get bundled
            # when a tarball is created on a Mac.  They are binary resource-fork
            # stubs that are meaningless on other platforms and would clutter
            # the annotation folder with unreadable files.
            def _is_real_json(m: tarfile.TarInfo) -> bool:
                basename = Path(m.name).name
                return (
                    m.isfile()
                    and m.name.endswith(".json")
                    and not basename.startswith("._")
                )

            # Detect tar structure by inspecting the first real JSON entry:
            #   Case A: files sit at root         → "std_<uid>.json"
            #   Case B: files have folder prefix  → "assets_std_annotation_flat/std_<uid>.json"
            first_real = next((m for m in members if _is_real_json(m)), None)
            has_prefix = first_real is not None and "/" in first_real.name

            # Extract each file manually so we can apply the filter regardless
            # of tar structure — this avoids blindly calling extractall() which
            # would pull in all the ._* junk.
            for member in members:
                if not _is_real_json(member):
                    continue

                if has_prefix:
                    # Keep the relative path under cache_dir
                    # e.g. "assets_std_annotation_flat/std_<uid>.json"
                    target = cache_dir / member.name
                else:
                    # Root-level files → place directly in ann_dir
                    target = ann_dir / Path(member.name).name

                target.parent.mkdir(parents=True, exist_ok=True)
                file_obj = tf.extractfile(member)
                if file_obj is not None:
                    target.write_bytes(file_obj.read())

        # ── Step 3: write marker ─────────────────────────────────────────────
        marker.write_text(str(time.time()))
        try:
            bundle_path.unlink()
        except OSError:
            pass
        print("[urbanverse] Annotation bundle ready.")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Repo file listing
# ──────────────────────────────────────────────────────────────────────────────

def list_repo_files() -> List[str]:
    """Return all file paths in the HF repo (cached in memory for the session)."""
    with _state_lock:
        if _state["_repo_files"] is not None:
            return _state["_repo_files"]

    api   = HfApi()
    files = list(api.list_repo_files(
        repo_id=REPO_ID, repo_type=REPO_TYPE, token=get_token()
    ))
    with _state_lock:
        _state["_repo_files"] = files
    return files


def list_folder_files(folder: str) -> List[str]:
    """Return all file paths under a specific folder in the HF repo.

    Uses ``HfApi.list_repo_tree(path_in_repo=folder)`` so that only the
    requested subtree is fetched — avoiding the full 600k+ file listing
    that ``list_repo_files()`` performs.

    Results are cached per folder for the session lifetime.
    """
    cache_key = f"_folder_files:{folder}"
    with _state_lock:
        cached = _state.get(cache_key)
        if cached is not None:
            return cached

    api = HfApi()
    files: List[str] = []
    for item in api.list_repo_tree(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        path_in_repo=folder, recursive=True, token=get_token(),
    ):
        if hasattr(item, "rfilename"):
            files.append(item.rfilename)

    with _state_lock:
        _state[cache_key] = files
    return files


# ──────────────────────────────────────────────────────────────────────────────
# Bucket index loading
# ──────────────────────────────────────────────────────────────────────────────

_bucket_index_lock = Lock()


def _load_bucket_index(key: str) -> Dict[str, str]:
    """Download (once) and return the bucket index for the given asset type.

    ``key`` must be one of: ``"glb"``, ``"annotation"``, ``"thumbnail"``, ``"render"``.

    Returns:
        ``{uid: relative_repo_path}``  e.g.
        ``{"06b6e2…": "assets_std_glb_flat/bucket_0001/std_06b6e2….glb"}``
    """
    with _state_lock:
        cached = _state["_bucket_indices"].get(key)
        if cached is not None:
            return cached

    if key not in _BUCKET_INDEX_FILES:
        raise ValueError(f"Unknown bucket index key: {key!r}")

    repo_path = _BUCKET_INDEX_FILES[key]
    with _bucket_index_lock:
        # Re-check inside the lock
        with _state_lock:
            cached = _state["_bucket_indices"].get(key)
            if cached is not None:
                return cached

        local = _download_file(repo_path)
        with open(local) as f:
            index = json.load(f)

        with _state_lock:
            _state["_bucket_indices"][key] = index

    return index


def get_bucket_path(uid: str, asset_type: str) -> Optional[str]:
    """Return the bucketed repo path for a UID and asset type, or None."""
    index = _load_bucket_index(asset_type)
    return index.get(uid)


# ──────────────────────────────────────────────────────────────────────────────
# Render tar.gz extraction
# ──────────────────────────────────────────────────────────────────────────────

_render_extract_locks: Dict[str, Lock] = {}
_render_extract_locks_lock = Lock()


def _get_render_extract_lock(uid: str) -> Lock:
    with _render_extract_locks_lock:
        if uid not in _render_extract_locks:
            _render_extract_locks[uid] = Lock()
        return _render_extract_locks[uid]


def render_extract_dir(uid: str) -> Optional[Path]:
    """Return the local directory where extracted render JPGs live for a UID.

    Derives the path from the bucket index so renders stay inside their
    bucket folder: ``assets_render_flat/bucket_NNNNN/{uid}/``.

    Returns ``None`` if the UID is not in the render bucket index.
    """
    rp = get_bucket_path(uid, "render")
    if rp is None:
        return None
    # rp = "assets_render_flat/bucket_00001/06b6e2b9….tar.gz"
    # → extract to "assets_render_flat/bucket_00001/06b6e2b9…/"
    # Path.stem strips only the last suffix (.gz); we need to strip .tar.gz entirely
    tar_path = Path(rp)
    folder_name = tar_path.name
    for suffix in (".tar.gz", ".tgz"):
        if folder_name.endswith(suffix):
            folder_name = folder_name[: -len(suffix)]
            break
    return get_cache_dir() / tar_path.parent / folder_name


def _extract_render_tar(uid: str) -> Path:
    """Extract a render tar.gz into its bucket folder and delete the archive.

    Output directory: ``assets_render_flat/bucket_NNNNN/{uid}/``

    Thread-safe and idempotent — skips extraction if the output directory
    already contains any .jpg files.

    Returns the output directory path.
    """
    cache = get_cache_dir()
    render_repo_path = get_bucket_path(uid, "render")
    if render_repo_path is None:
        raise FileNotFoundError(f"No render bucket entry for UID {uid}")

    tar_local = cache / render_repo_path
    out_dir = render_extract_dir(uid)
    assert out_dir is not None

    # Fast path: already extracted (and tar already deleted)
    if out_dir.exists() and any(out_dir.glob("*.jpg")):
        return out_dir

    lock = _get_render_extract_lock(uid)
    with lock:
        # Re-check inside the lock
        if out_dir.exists() and any(out_dir.glob("*.jpg")):
            return out_dir

        if not tar_local.exists():
            raise FileNotFoundError(
                f"Render tar.gz not found at {tar_local}. Download it first."
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_local, "r:gz") as tf:
            for member in tf.getmembers():
                if member.isfile() and member.name.endswith(".jpg"):
                    basename = Path(member.name).name
                    target = out_dir / basename
                    file_obj = tf.extractfile(member)
                    if file_obj is not None:
                        target.write_bytes(file_obj.read())

        # Remove the tar.gz to free disk space
        try:
            tar_local.unlink()
        except OSError:
            pass

    return out_dir


_RENDER_BATCH_SIZE = 256


def _download_and_extract_renders(
    uids: List[str],
    num_workers: int = 8,
) -> None:
    """Download render tar.gz archives and extract them to per-UID folders.

    Archives are processed in batches of :data:`_RENDER_BATCH_SIZE` (256).
    Each batch is downloaded in parallel, then extracted and deleted before
    the next batch starts.  This keeps peak disk usage from .tar.gz files
    low (~256 archives) and ensures that all completed batches have usable
    JPG renders on disk even if the process is interrupted.
    """
    render_index = _load_bucket_index("render")
    repo_paths = []
    valid_uids = []
    for uid in uids:
        rp = render_index.get(uid)
        if not rp:
            continue
        out = render_extract_dir(uid)
        if out and out.exists() and any(out.glob("*.jpg")):
            continue
        repo_paths.append(rp)
        valid_uids.append(uid)

    total = len(repo_paths)
    num_batches = (total + _RENDER_BATCH_SIZE - 1) // _RENDER_BATCH_SIZE if total else 0

    for i in range(0, total, _RENDER_BATCH_SIZE):
        batch_paths = repo_paths[i : i + _RENDER_BATCH_SIZE]
        batch_uids  = valid_uids[i : i + _RENDER_BATCH_SIZE]
        batch_num   = i // _RENDER_BATCH_SIZE + 1

        _download_files_parallel(
            batch_paths, num_workers,
            desc=f"Renders [{batch_num}/{num_batches}]",
        )

        for uid in batch_uids:
            try:
                _extract_render_tar(uid)
            except Exception as e:
                print(f"[urbanverse] render extraction failed for {uid}: {e}")

    # Defensive sweep: remove any leftover .tar.gz for ALL requested UIDs
    # (covers both newly extracted and already-extracted-but-orphaned archives)
    for uid in uids:
        rp = render_index.get(uid)
        if rp:
            leftover = get_cache_dir() / rp
            if leftover.exists():
                try:
                    leftover.unlink()
                except OSError:
                    pass


# ──────────────────────────────────────────────────────────────────────────────
# Integrity helpers
# ──────────────────────────────────────────────────────────────────────────────

def _count_local_bucket_files(
    index: Dict[str, str],
) -> tuple:
    """Count how many files from a bucket index are present in the local cache.

    Args:
        index: ``{uid: repo_path}`` dict from ``_load_bucket_index()``.

    Returns:
        ``(expected, downloaded, missing_repo_paths)`` where
        *missing_repo_paths* is a list of repo paths that are **not** on disk.
    """
    cache = get_cache_dir()
    expected = len(index)
    missing: List[str] = []
    for repo_path in index.values():
        if not (cache / repo_path).exists():
            missing.append(repo_path)
    return expected, expected - len(missing), missing


def _count_local_render_uids(
    index: Dict[str, str],
) -> tuple:
    """Count how many UIDs have their renders fully extracted.

    Returns:
        ``(expected, downloaded, missing_uids)``
    """
    expected = len(index)
    missing: List[str] = []
    for uid in index:
        out = render_extract_dir(uid)
        if out is None or not out.exists() or not any(out.glob("*.jpg")):
            missing.append(uid)
    return expected, expected - len(missing), missing


# ──────────────────────────────────────────────────────────────────────────────
# Shared validation helper
# ──────────────────────────────────────────────────────────────────────────────

def validate_what(what: tuple, valid: tuple, context: str = "") -> None:
    invalid = set(what) - set(valid)
    if invalid:
        raise ValueError(
            f"Invalid `what` value(s){' for ' + context if context else ''}: {sorted(invalid)}\n"
            f"  Valid choices: {valid}"
        )
