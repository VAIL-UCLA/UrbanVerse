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
Base class for vegetation sub-modules (plant, shrub, tree).

Each sub-module is a thin subclass that sets one class-level constant:
  _folder  — HF repo folder containing .tar.gz archives
              (e.g. "collected_plants", "collected_shrubs", "collected_trees")

The base class handles:
  - Scanning the HF repo for .tar.gz files under ``_folder``.
  - Downloading archives, extracting them, and removing the .tar.gz.
  - Returning {description → {"usd": Path, "folder": Path}} dicts.
  - Semantic text search over the asset names.
"""

import shutil
import tarfile
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from . import _core

_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


class _VegetationSource:
    """Abstract base for plant / shrub / tree asset APIs."""

    _folder: str = ""

    _cached_file_map: Optional[Dict] = None
    _cache_lock: Lock = Lock()

    @classmethod
    def _get_file_map(cls) -> Dict[str, str]:
        """Build ``{description: repo_path}`` for ``.tar.gz`` files.

        Description is the archive stem (e.g. ``"American_Beech"`` from
        ``collected_trees/American_Beech.tar.gz``).
        """
        with cls._cache_lock:
            if cls._cached_file_map is not None:
                return cls._cached_file_map

        prefix = cls._folder + "/"
        all_files = _core.list_folder_files(cls._folder)
        file_map: Dict[str, str] = {}
        for f in all_files:
            if f.startswith(prefix) and f.endswith(".tar.gz"):
                name = f[len(prefix):]
                if "/" in name:
                    continue
                desc = name[: -len(".tar.gz")]
                file_map[desc] = f

        with cls._cache_lock:
            cls._cached_file_map = file_map

        return file_map

    # ── Public API ─────────────────────────────────────────────────────────────

    @classmethod
    def get_descriptions(cls) -> List[str]:
        """Return a sorted list of all available vegetation asset names.

        These names are used as keys in ``load_materials()`` results.

        Example::
·
            names = uva.tree.get_descriptions()
            # ["American_Beech", "Black_Oak", "Colorado_Spruce", ...]
        """
        return sorted(cls._get_file_map().keys())

    @classmethod
    def get_descriptions_conditioned(
        cls,
        query: str,
        top_k: int = 10,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ) -> List[str]:
        """Return up to ``top_k`` asset names ranked by similarity to ``query``.

        Asset names (e.g. ``"Japanese_Maple"``) are embedded with a
        sentence-transformer model and compared to the query via cosine
        similarity.  Results are sorted by descending similarity.

        Args:
            query:           Free-form text, e.g. ``"tall evergreen tree"``.
            top_k:           Maximum results to return (default 10).
            embedding_model: Sentence-transformer model identifier.

        Returns:
            List of asset name strings sorted by descending similarity.

        Example::

            names = uva.tree.get_descriptions_conditioned("palm tree", top_k=5)
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "numpy is required for text search — it installs automatically "
                "with sentence-transformers:  pip install sentence-transformers"
            ) from None

        from . import _filter

        if embedding_model not in _filter._VALID_EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown embedding_model {embedding_model!r}.\n"
                f"  Valid choices: {list(_filter._VALID_EMBEDDING_MODELS)}"
            )

        descriptions = cls.get_descriptions()
        if not descriptions:
            return []

        model = _filter._get_embedding_model(embedding_model)

        readable = [d.replace("_", " ") for d in descriptions]
        all_texts = [query] + readable
        all_embs = model.encode(
            all_texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=(len(descriptions) > 200),
            normalize_embeddings=True,
        )

        query_emb = all_embs[0]
        desc_embs = all_embs[1:]
        scores = desc_embs @ query_emb

        top_k = min(top_k, len(descriptions))
        order = np.argsort(-scores)[:top_k]
        return [descriptions[i] for i in order]

    @classmethod
    def load_materials(
        cls,
        descriptions: Optional[List[str]] = None,
        num_workers: int = 8,
    ) -> Dict[str, Dict[str, Any]]:
        """Download, extract, and return local paths for vegetation assets.

        Each ``.tar.gz`` archive is downloaded from HuggingFace, extracted
        into a folder of the same name, and the archive is then removed.
        Already-extracted assets are skipped.

        Args:
            descriptions: Asset names to download.  ``None`` downloads all.
                          Call :meth:`get_descriptions` to see valid names.
            num_workers:  Parallel download threads (default 8).

        Returns:
            ``{name: {"usd": Path("…/World0.usd"), "folder": Path("…/Name/")}}``

            ``usd`` is ``None`` if the ``World0.usd`` file was not found
            after extraction.

        Raises:
            ValueError: If any description name is unknown.

        Example::

            result = uva.tree.load_materials(["American_Beech"])
            usd = result["American_Beech"]["usd"]
            print(usd)  # ~/.cache/urbanverse/collected_trees/American_Beech/World0.usd
        """
        file_map = cls._get_file_map()

        if descriptions is None:
            descriptions = sorted(file_map.keys())
        else:
            unknown = set(descriptions) - set(file_map.keys())
            if unknown:
                raise ValueError(
                    f"Unknown {cls._folder} asset(s): {sorted(unknown)}\n"
                    f"  Call get_descriptions() to see valid names."
                )

        cache = _core.get_cache_dir()

        to_download: List[str] = []
        for desc in descriptions:
            extract_dir = cache / cls._folder / desc
            if extract_dir.exists() and extract_dir.is_dir():
                continue
            to_download.append(file_map[desc])

        if to_download:
            _core._download_files_parallel(
                to_download, num_workers,
                desc=f"Downloading {cls._folder}",
            )

            for repo_path in to_download:
                archive_path = cache / repo_path
                if not archive_path.exists():
                    continue
                _extract_and_cleanup(archive_path)

        out: Dict[str, Dict[str, Any]] = {}
        for desc in descriptions:
            folder = cache / cls._folder / desc
            usd_path = folder / "World0.usd"
            out[desc] = {
                "usd": usd_path if usd_path.exists() else None,
                "folder": folder if folder.exists() else None,
            }

        return out


def _extract_and_cleanup(archive_path: Path) -> None:
    """Extract a .tar.gz archive into a sibling folder and remove the archive."""
    extract_dir = archive_path.parent / archive_path.name.replace(".tar.gz", "")

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=archive_path.parent, filter="data")
    except (tarfile.TarError, Exception) as e:
        print(f"[urbanverse] Failed to extract {archive_path.name}: {e}")
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        return

    try:
        archive_path.unlink()
    except OSError:
        pass

    print(f"[urbanverse] Extracted {extract_dir.name}/")
