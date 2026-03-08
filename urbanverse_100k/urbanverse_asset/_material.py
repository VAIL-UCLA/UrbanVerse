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
Base class for material sub-modules (sky, road, sidewalk, terrain).

Each sub-module is a thin subclass that sets four class-level constants:
  _folder_main        — HF repo folder for the primary material files
  _folder_thumb       — HF repo folder for thumbnail files
  _main_key           — key name used in the returned dict ("hdr" or "mdl")
  _what_choices       — valid values for the `what` parameter
  _texture_subfolder  — (optional) subfolder name containing shared textures
                        that must be downloaded in full (e.g. "textures")

The base class handles:
  - Building a {description → {main_files, thumbnail}} file map from
    the HF repo file listing (cached in memory).
  - Downloading requested files in parallel (including full texture folders
    for material types that depend on them).
  - Returning {description → {key: local_path}} dicts.
"""

from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from . import _core

_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


class _MaterialSource:
    """Abstract base for sky / road / sidewalk / terrain material APIs."""

    # Subclasses must override these:
    _folder_main:       str   = ""
    _folder_thumb:      str   = ""
    _main_key:          str   = ""           # "hdr" | "mdl"
    _what_choices:      tuple = ()
    _texture_subfolder: str   = ""           # e.g. "textures" for MDL materials

    # Per-subclass cache (set to None in each subclass to avoid sharing)
    _cached_file_map: Optional[Dict] = None
    _cache_lock:      Lock            = Lock()

    # ── File map ──────────────────────────────────────────────────────────────

    @classmethod
    def _get_file_map(cls) -> Dict[str, Dict[str, Any]]:
        """Build {description: {"main_files": [repo_paths], "thumbnail": repo_path}}.

        Only files directly under ``_folder_main/`` are considered material
        files.  Files in subdirectories (e.g. ``textures/``) are excluded
        from the description map — they are handled separately via
        ``_get_texture_files()``.

        Result is cached per subclass for the session lifetime.
        """
        with cls._cache_lock:
            if cls._cached_file_map is not None:
                return cls._cached_file_map

        all_files = (
            _core.list_folder_files(cls._folder_main)
            + _core.list_folder_files(cls._folder_thumb)
        )
        file_map: Dict[str, Dict[str, Any]] = {}

        main_prefix  = cls._folder_main + "/"
        thumb_prefix = cls._folder_thumb + "/"

        for f in all_files:
            if f.startswith(main_prefix):
                rel = f[len(main_prefix):]
                if "/" in rel:
                    continue
                fname = Path(f).name
                desc  = fname.split(".")[0]
                entry = file_map.setdefault(desc, {})
                entry.setdefault("main_files", []).append(f)
            elif f.startswith(thumb_prefix):
                rel = f[len(thumb_prefix):]
                if "/" in rel:
                    continue
                fname = Path(f).name
                desc  = fname.split(".")[0]
                entry = file_map.setdefault(desc, {})
                entry["thumbnail"] = f

        with cls._cache_lock:
            cls._cached_file_map = file_map

        return file_map

    @classmethod
    def _get_texture_files(cls) -> List[str]:
        """Return all repo paths under the material texture subfolder."""
        if not cls._texture_subfolder:
            return []
        texture_folder = f"{cls._folder_main}/{cls._texture_subfolder}"
        prefix = texture_folder + "/"
        return [f for f in _core.list_folder_files(texture_folder) if f.startswith(prefix)]

    # ── Public API ─────────────────────────────────────────────────────────────

    @classmethod
    def get_descriptions(cls) -> List[str]:
        """Return a sorted list of all available material names.

        These names are used as keys in load_materials() results and as valid
        values for the ``descriptions`` filter parameter.

        Example::

            descs = uva.sky.get_descriptions()
            print(descs[:3])  # ["autumn_field_puresky", "autumn_hockey", ...]
        """
        return sorted(cls._get_file_map().keys())

    @classmethod
    def get_descriptions_conditioned(
        cls,
        query: str,
        top_k: int = 10,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ) -> List[str]:
        """Return up to ``top_k`` material names ranked by similarity to ``query``.

        Each material name (the file basename, e.g. ``"Cobblestone_Medieval"``)
        is embedded with a sentence-transformer model and compared to the query
        via cosine similarity (L2-normalised dot product).  The result list is
        sorted by descending similarity.

        Args:
            query:           Free-form text describing the desired material,
                             e.g. ``"dramatic orange sunset sky"`` or
                             ``"rough wet asphalt road surface"``.
            top_k:           Maximum number of results to return (default 10).
                             If larger than the total number of materials, all
                             materials are returned (sorted by similarity).
            embedding_model: Sentence-transformer model identifier.

                             * ``"sentence-transformers/all-mpnet-base-v2"``
                               (default) — higher quality, 420 MB
                             * ``"sentence-transformers/all-MiniLM-L6-v2"``
                               — fast, 80 MB

        Returns:
            List of material name strings sorted by descending cosine similarity
            to ``query``.  Pass the result directly to :meth:`load_materials`.

        Raises:
            ValueError:  If ``embedding_model`` is not a recognised choice.
            ImportError: If ``sentence-transformers`` is not installed.

        Example::

            # Sky maps that look like a sunset
            names = uva.sky.get_descriptions_conditioned(
                "dramatic orange sunset sky", top_k=5
            )
            uva.sky.load_materials(names, what=("thumbnail",))

            # Top-10 cobblestone-like road surfaces
            names = uva.road.get_descriptions_conditioned(
                "cobblestone medieval street", top_k=10
            )
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "numpy is required for text search — it installs automatically "
                "with sentence-transformers:  pip install sentence-transformers"
            ) from None

        # Lazy import: avoids circular imports while sharing the model cache
        # with _filter (so models are not loaded twice across the session).
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

        n = len(descriptions)
        all_texts = [query] + descriptions
        all_embs = model.encode(
            all_texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=(n > 200),
            normalize_embeddings=True,   # unit-norm → dot product == cosine sim
        )

        query_emb = all_embs[0]    # shape (D,)
        desc_embs = all_embs[1:]   # shape (N, D)

        # Vectorised cosine similarity
        scores = desc_embs @ query_emb   # (N,)

        top_k = min(top_k, n)
        order = np.argsort(-scores)[:top_k]
        return [descriptions[i] for i in order]

    @classmethod
    def load_materials(
        cls,
        descriptions: Optional[List[str]] = None,
        num_workers: int = 8,
        what: Optional[tuple] = None,
    ) -> Dict[str, Dict[str, Optional[Path]]]:
        """Download materials and return their local paths.

        For MDL-based material types (road, sidewalk, terrain), the shared
        ``textures/`` subfolder is always downloaded in full because every
        ``.mdl`` / ``.usda`` file may reference textures from that folder.

        Args:
            descriptions: Material names to download. ``None`` downloads all.
                          Call :meth:`get_descriptions` to see valid values.
            num_workers:  Parallel download threads (default 8).
            what:         Tuple of asset types to include. Valid values depend
                          on the sub-module (e.g. ``("hdr", "thumbnail")`` for
                          sky, or ``("mdl", "thumbnail")`` for road/sidewalk/
                          terrain). ``None`` uses the sub-module default.

        Returns:
            ``{description: {main_key: Path | None, "thumbnail": Path | None}}``

        Raises:
            ValueError: If any description name is unknown or ``what`` contains
                        invalid values.
        """
        if what is None:
            what = cls._what_choices

        _core.validate_what(what, cls._what_choices, context=cls._main_key)

        file_map = cls._get_file_map()

        if descriptions is None:
            descriptions = list(file_map.keys())
        else:
            unknown = set(descriptions) - set(file_map.keys())
            if unknown:
                raise ValueError(
                    f"Unknown material name(s): {sorted(unknown)}\n"
                    f"  Call get_descriptions() to see valid names."
                )

        # Collect paths to download
        to_download: List[str] = []
        for desc in descriptions:
            entry = file_map.get(desc, {})
            if cls._main_key in what:
                to_download.extend(entry.get("main_files", []))
            if "thumbnail" in what and "thumbnail" in entry:
                to_download.append(entry["thumbnail"])

        # MDL materials depend on shared textures — download them in full
        if cls._texture_subfolder and cls._main_key in what:
            to_download.extend(cls._get_texture_files())

        _core._download_files_parallel(to_download, num_workers,
                                       desc=f"Downloading {cls._main_key}")

        # Build result dict
        cache = _core.get_cache_dir()
        out: Dict[str, Dict[str, Optional[Path]]] = {}
        for desc in descriptions:
            entry  = file_map.get(desc, {})
            result: Dict[str, Optional[Path]] = {}
            if cls._main_key in what:
                main_files = entry.get("main_files", [])
                # Prefer the file with the canonical extension (.mdl / .hdr)
                preferred = next(
                    (f for f in main_files if f.endswith(f".{cls._main_key}")),
                    main_files[0] if main_files else None,
                )
                if preferred:
                    p = cache / preferred
                    result[cls._main_key] = p if p.exists() else None
            if "thumbnail" in what and "thumbnail" in entry:
                p = cache / entry["thumbnail"]
                result["thumbnail"] = p if p.exists() else None
            out[desc] = result

        return out
