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
get_uids_conditioned — filter asset UIDs by per-asset attribute conditions,
with optional free-form natural language query search.

Design
------
All filter parameters are optional and combined with AND logic.
Within a list-type filter (e.g. categories, affordances), OR logic is used —
the asset matches if it has ANY of the specified values.
For dominant_materials and dominant_colors, only the first (dominant) entry
in the annotation list is compared.

Text query (optional)
---------------------
When ``query`` is provided, per-asset annotations are loaded (via the fast
annotation bundle), each asset's ``description`` and ``description_long``
fields are encoded with a sentence embedding model, cosine similarity is
computed against the query embedding, and the top-k most similar assets
(after applying all other attribute filters) are returned.

Similarity score = average(
    cosine_sim(query, description),
    cosine_sim(query, description_long)
)

Optimisation
------------
If only ``categories`` is given (no attribute filters, no query), UIDs are
resolved directly from the master annotation without loading per-asset JSON
files (fast path).
Per-asset annotations are downloaded when attribute filters or text query
search is active.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from . import _core


# ──────────────────────────────────────────────────────────────────────────────
# Embedding model cache (one instance per model name, shared across calls)
# ──────────────────────────────────────────────────────────────────────────────

_VALID_EMBEDDING_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",   # default — fast, 80 MB
    "sentence-transformers/all-mpnet-base-v2",  # high quality, 420 MB
)

_model_cache: Dict[str, Any] = {}
_model_lock = threading.Lock()


def _detect_device() -> str:
    """Return the best available torch device: 'cuda' > 'mps' > 'cpu'."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    # MPS = Apple Silicon GPU (Metal Performance Shaders), PyTorch >= 1.12
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_embedding_model(model_name: str):
    """Return a cached SentenceTransformer instance, loading it on first use.

    Automatically selects the best available device (CUDA GPU → Apple MPS → CPU)
    and explicitly passes it to the model constructor so the choice is never
    left ambiguous across different sentence-transformers versions.
    """
    with _model_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "The 'sentence-transformers' package is required for text query search.\n"
            "Install it with:  pip install sentence-transformers"
        ) from None

    device = _detect_device()
    device_label = {"cuda": "GPU (CUDA)", "mps": "GPU (Apple MPS)", "cpu": "CPU"}[device]
    print(
        f"[urbanverse] Loading embedding model '{model_name}' "
        f"on {device_label} (first use only)…"
    )
    model = SentenceTransformer(model_name, device=device)

    with _model_lock:
        _model_cache[model_name] = model
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Text similarity helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compute_text_scores(
    query: str,
    uids: List[str],
    annotations: Dict[str, Dict],
    model_name: str,
) -> List[Tuple[str, float]]:
    """Return ``[(uid, score), …]`` sorted by descending similarity.

    Score = average cosine similarity of the query against each asset's
    ``description`` and ``description_long`` fields.

    Assets whose annotations are missing are silently skipped.
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is required for text search. It should install automatically "
            "with sentence-transformers: pip install sentence-transformers"
        ) from None

    # Collect valid UIDs and their texts
    valid_uids: List[str] = []
    short_descs: List[str] = []
    long_descs: List[str] = []

    for uid in uids:
        ann = annotations.get(uid)
        if ann is None:
            continue
        valid_uids.append(uid)
        short_descs.append(ann.get("description", "") or "")
        long_descs.append(ann.get("description_long", "") or "")

    if not valid_uids:
        return []

    model = _get_embedding_model(model_name)

    # Encode all texts in one batched call for maximum efficiency.
    # Layout: [query, desc_0, …, desc_N, long_0, …, long_N]
    n = len(valid_uids)
    show_bar = n > 200
    all_texts = [query] + short_descs + long_descs
    all_embs = model.encode(
        all_texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=show_bar,
        normalize_embeddings=True,   # pre-normalise → dot product == cosine sim
    )

    query_emb    = all_embs[0]           # shape (D,)
    short_embs   = all_embs[1 : n + 1]  # shape (N, D)
    long_embs    = all_embs[n + 1 :]    # shape (N, D)

    # Vectorised cosine similarity (dot product on pre-normalised embeddings)
    sim_short = short_embs @ query_emb   # (N,)
    sim_long  = long_embs  @ query_emb   # (N,)
    scores    = (sim_short + sim_long) / 2.0

    # Sort descending
    order  = np.argsort(-scores)
    return [(valid_uids[i], float(scores[i])) for i in order]


# ──────────────────────────────────────────────────────────────────────────────
# Existing filter helpers (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def _uids_for_categories(categories: List[str]) -> Set[str]:
    """Return UIDs whose L1, L2, or L3 class name matches any value in `categories`."""
    ann        = _core.load_master_annotation()
    cat_set    = set(c.lower() for c in categories)
    matched    : Set[str] = set()

    for entry in ann["annotation"].values():
        if (
            entry["class_name_l1"].lower() in cat_set
            or entry["class_name_l2"].lower() in cat_set
            or entry["class_name_l3"].lower() in cat_set
        ):
            matched.update(entry["asset_uids"])
    return matched


def _load_per_asset_annotations(uids: List[str], num_workers: int) -> Dict[str, Dict]:
    """Load per-asset JSON annotations for the given UIDs.

    Strategy (fast path first):
      1. Try to download and extract ``full_per_asset_annotations.tar.gz``
         from the HF repo (one-time per cache directory).  After extraction
         all JSONs are on disk and no further network calls are needed.
      2. If the bundle is unavailable in this repo (404 — e.g. toy dataset),
         fall back to downloading the individual JSON files via bucket index.
    """
    bundle_ok = _core._ensure_per_asset_annotations()

    if not bundle_ok:
        repo_paths = [
            rp for uid in uids
            if (rp := _core.get_bucket_path(uid, "annotation")) is not None
        ]
        _core._download_files_parallel(repo_paths, num_workers, desc="Fetching annotations")

    cache  = _core.get_cache_dir()
    result = {}
    for uid in uids:
        # Check bucket path first, then flat path (from annotation bundle)
        rp = _core.get_bucket_path(uid, "annotation")
        p = cache / rp if rp else None
        if p is None or not p.exists():
            p = cache / f"assets_std_annotation_flat/std_{uid}.json"
        if p.exists():
            with open(p) as f:
                result[uid] = json.load(f)
    return result


def _in_range(value: Any, rng: Optional[Tuple[float, float]]) -> bool:
    """Return True if value is within [min, max] (inclusive)."""
    if rng is None:
        return True
    if value is None:
        return False
    lo, hi = rng
    return lo <= value <= hi


def _list_match(asset_list: Optional[List], filter_list: Optional[List]) -> bool:
    """Return True if asset_list contains ANY item from filter_list (case-insensitive)."""
    if filter_list is None:
        return True
    if not asset_list:
        return False
    filter_set = set(v.lower() for v in filter_list)
    return any(str(v).lower() in filter_set for v in asset_list)


def _dominant_match(asset_list: Optional[List], filter_list: Optional[List]) -> bool:
    """Return True if the first (dominant) item in asset_list matches ANY item in filter_list."""
    if filter_list is None:
        return True
    if not asset_list:
        return False
    filter_set = set(v.lower() for v in filter_list)
    return str(asset_list[0]).lower() in filter_set


def _exact_match(value: Any, allowed: Optional[List]) -> bool:
    """Return True if value matches ANY item in allowed (case-insensitive)."""
    if allowed is None:
        return True
    if value is None:
        return False
    allowed_set = set(v.lower() for v in allowed)
    return str(value).lower() in allowed_set


def _bool_match(value: Any, required: Optional[bool]) -> bool:
    if required is None:
        return True
    return bool(value) == required


def _passes_filters(ann: Dict, **filters) -> bool:
    """Return True if the per-asset annotation dict satisfies all filter conditions."""
    # ── Range filters ─────────────────────────────────────────────────────────
    if not _in_range(ann.get("height"),                  filters["height_range"]):                return False
    if not _in_range(ann.get("length"),                  filters["length_range"]):                return False
    if not _in_range(ann.get("width"),                   filters["width_range"]):                 return False
    if not _in_range(ann.get("max_dimension"),           filters["max_dimension_range"]):          return False
    if not _in_range(ann.get("mass"),                    filters["mass_range"]):                    return False
    if not _in_range(ann.get("quality"),                 filters["quality_range"]):                 return False
    if not _in_range(ann.get("required_force"),          filters["required_force_range"]):          return False
    if not _in_range(ann.get("surface_roughness"),       filters["surface_roughness_range"]):       return False
    if not _in_range(ann.get("reflectivity"),            filters["reflectivity_range"]):            return False
    if not _in_range(ann.get("index_of_refraction"),     filters["index_of_refraction_range"]):     return False
    if not _in_range(ann.get("youngs_modulus"),          filters["youngs_modulus_range"]):          return False
    if not _in_range(ann.get("friction_coefficient"),    filters["friction_coefficient_range"]):    return False
    if not _in_range(ann.get("bounciness"),              filters["bounciness_range"]):              return False
    if not _in_range(ann.get("recommended_clearance"),   filters["recommended_clearance_range"]):   return False

    # ── Boolean filters ───────────────────────────────────────────────────────
    if not _bool_match(ann.get("receptacle"),       filters["receptacle"]):    return False
    if not _bool_match(ann.get("movable"),          filters["movable"]):       return False
    if not _bool_match(ann.get("walkable"),         filters["walkable"]):      return False
    if not _bool_match(ann.get("enterable"),        filters["enterable"]):     return False
    if not _bool_match(ann.get("support_surface"),  filters["support_surface"]): return False

    # ── Dominant-item filters (first element = highest composition) ──────────
    if not _dominant_match(ann.get("materials"), filters["dominant_materials"]): return False
    if not _dominant_match(ann.get("colors"),    filters["dominant_colors"]):    return False

    # ── List / membership filters ─────────────────────────────────────────────
    if not _list_match(ann.get("affordances"),       filters["affordances"]):       return False
    if not _list_match(ann.get("interactive_parts"), filters["interactive_parts"]): return False

    # ── Exact-match-from-list filters ─────────────────────────────────────────
    if not _exact_match(ann.get("surface_hardness"),       filters["surface_hardness"]):       return False
    if not _exact_match(ann.get("surface_finish"),         filters["surface_finish"]):         return False
    if not _exact_match(ann.get("asset_composition_type"), filters["asset_composition_type"]): return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_uids_conditioned(
    categories: Optional[List[str]] = None,
    height_range: Optional[Tuple[float, float]] = None,
    length_range: Optional[Tuple[float, float]] = None,
    width_range: Optional[Tuple[float, float]] = None,
    max_dimension_range: Optional[Tuple[float, float]] = None,
    mass_range: Optional[Tuple[float, float]] = None,
    quality_range: Optional[Tuple[int, int]] = None,
    dominant_materials: Optional[List[str]] = None,
    receptacle: Optional[bool] = None,
    movable: Optional[bool] = None,
    required_force_range: Optional[Tuple[float, float]] = None,
    walkable: Optional[bool] = None,
    enterable: Optional[bool] = None,
    affordances: Optional[List[str]] = None,
    support_surface: Optional[bool] = None,
    interactive_parts: Optional[List[str]] = None,
    dominant_colors: Optional[List[str]] = None,
    surface_hardness: Optional[List[str]] = None,
    surface_roughness_range: Optional[Tuple[float, float]] = None,
    surface_finish: Optional[List[str]] = None,
    reflectivity_range: Optional[Tuple[float, float]] = None,
    index_of_refraction_range: Optional[Tuple[float, float]] = None,
    youngs_modulus_range: Optional[Tuple[float, float]] = None,
    friction_coefficient_range: Optional[Tuple[float, float]] = None,
    bounciness_range: Optional[Tuple[float, float]] = None,
    recommended_clearance_range: Optional[Tuple[float, float]] = None,
    asset_composition_type: Optional[List[str]] = None,
    # ── Text query (new) ──────────────────────────────────────────────────────
    query: Optional[str] = None,
    top_k: int = 100,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    # ─────────────────────────────────────────────────────────────────────────
    num_workers: int = 8,
) -> List[str]:
    """Return UIDs of assets that satisfy all specified conditions.

    All parameters are optional — unset parameters are not filtered.
    Within list-type parameters (categories, affordances …), OR logic
    applies: an asset qualifies if it matches ANY of the provided values.
    Across parameters, AND logic applies.

    For ``dominant_materials`` and ``dominant_colors``, only the **first**
    (0-indexed) entry in the annotation list is checked — this is always
    the dominant value (highest composition percentage).

    Args:
        categories:               L1, L2, or L3 class names (e.g. ["vehicle", "bench"]).
        height_range:             (min_m, max_m) in metres.
        length_range:             (min_m, max_m) in metres.
        width_range:              (min_m, max_m) in metres.
        max_dimension_range:      (min_m, max_m) in metres.
        mass_range:               (min_kg, max_kg).
        quality_range:            (min, max), integer 1–10.
        dominant_materials:       Dominant material names (e.g. ["steel", "wood"]).
                                  Matches if the asset's primary (0-indexed)
                                  material is any of the given values.
        receptacle:               True/False — can it hold other objects?
        movable:                  True/False — can it be physically moved?
        required_force_range:     (min_N, max_N) force needed to move it.
        walkable:                 True/False — can an agent walk on it?
        enterable:                True/False — can an agent enter/occupy it?
        affordances:              Interaction types (e.g. ["sittable", "drivable"]).
        support_surface:          True/False — can it support placed objects?
        interactive_parts:        Named sub-parts (e.g. ["door", "wheel"]).
        dominant_colors:          Dominant color names (e.g. ["red", "white"]).
                                  Matches if the asset's primary (0-indexed)
                                  color is any of the given values.
        surface_hardness:         Hardness category: "soft", "semi-soft", "hard".
        surface_roughness_range:  (min, max), 0.0–1.0.
        surface_finish:           Finish type: "rough", "matte", "smooth", "glossy", "sleek", or "grippy".
        reflectivity_range:       (min, max), 0.0–1.0.
        index_of_refraction_range:(min, max).
        youngs_modulus_range:     (min_MPa, max_MPa).
        friction_coefficient_range:(min, max).
        bounciness_range:         (min, max).
        recommended_clearance_range:(min_m, max_m).
        asset_composition_type:   Composition type list (e.g. ["single"]).
        query:                    Free-form natural language description of the
                                  desired object (e.g. "a red fire hydrant on a
                                  sidewalk").  When provided, assets are ranked
                                  by text-to-asset similarity and the top ``top_k``
                                  results are returned.  Can be combined with any
                                  other filter — attribute filters are applied
                                  first, then text ranking is applied to the
                                  surviving candidates.
        top_k:                    Number of top results to return when ``query``
                                  is provided (default 100).  Ignored if ``query``
                                  is ``None``.
        embedding_model:          Sentence embedding model to use for text search.
                                  Choices:

                                  ``"sentence-transformers/all-mpnet-base-v2"``
                                      (default) — ~420 MB, best quality.
                                  ``"sentence-transformers/all-MiniLM-L6-v2"``
                                      — fast, ~80 MB, good quality.

                                  Requires ``pip install sentence-transformers``.
                                  The model is loaded once and cached in memory
                                  for the session.
        num_workers:              Parallel threads for annotation download (default 8).

    Returns:
        List of matching UID strings.  When ``query`` is given, the list is
        ordered from most to least similar to the query and capped at ``top_k``.
        Otherwise, order is not guaranteed.

    Raises:
        ValueError: If ``embedding_model`` is not one of the supported values.
        ImportError: If ``query`` is used without ``sentence-transformers``
                     installed.

    Example::

        # Heavy outdoor furniture that an agent can sit on
        uids = uva.get_uids_conditioned(
            categories=["bench"],
            mass_range=(50, 500),
            affordances=["sittable"],
        )

        # All vehicles taller than 2 m
        uids = uva.get_uids_conditioned(
            categories=["vehicle"],
            height_range=(2.0, float("inf")),
        )

        # Natural language search across all assets
        uids = uva.get_uids_conditioned(
            query="a red fire hydrant with a metallic surface",
            top_k=20,
        )

        # Combine text search with attribute filters
        uids = uva.get_uids_conditioned(
            categories=["bench"],
            query="weathered wooden bench with concrete supports",
            top_k=5,
        )
    """
    # ── Validate embedding model choice ───────────────────────────────────────
    if query is not None and embedding_model not in _VALID_EMBEDDING_MODELS:
        raise ValueError(
            f"Invalid embedding_model: {embedding_model!r}\n"
            f"  Valid choices: {_VALID_EMBEDDING_MODELS}"
        )

    # ── Step 1: determine candidate UIDs ──────────────────────────────────────
    if categories is not None:
        candidate_uids = list(_uids_for_categories(categories))
        if not candidate_uids:
            return []
    else:
        ann = _core.load_master_annotation()
        candidate_uids = [
            uid
            for entry in ann["annotation"].values()
            for uid in entry["asset_uids"]
        ]

    # ── Step 2: decide whether per-asset annotations are needed ───────────────
    attribute_filters = {
        "height_range": height_range,
        "length_range": length_range,
        "width_range": width_range,
        "max_dimension_range": max_dimension_range,
        "mass_range": mass_range,
        "quality_range": quality_range,
        "dominant_materials": dominant_materials,
        "receptacle": receptacle,
        "movable": movable,
        "required_force_range": required_force_range,
        "walkable": walkable,
        "enterable": enterable,
        "affordances": affordances,
        "support_surface": support_surface,
        "interactive_parts": interactive_parts,
        "dominant_colors": dominant_colors,
        "surface_hardness": surface_hardness,
        "surface_roughness_range": surface_roughness_range,
        "surface_finish": surface_finish,
        "reflectivity_range": reflectivity_range,
        "index_of_refraction_range": index_of_refraction_range,
        "youngs_modulus_range": youngs_modulus_range,
        "friction_coefficient_range": friction_coefficient_range,
        "bounciness_range": bounciness_range,
        "recommended_clearance_range": recommended_clearance_range,
        "asset_composition_type": asset_composition_type,
    }
    needs_per_asset  = any(v is not None for v in attribute_filters.values())
    needs_text_search = query is not None

    # Fast path: no per-asset data needed at all
    if not needs_per_asset and not needs_text_search:
        return candidate_uids

    # ── Step 3: warn on large candidate sets ──────────────────────────────────
    if len(candidate_uids) > 500:
        reason = "text search" if needs_text_search and not needs_per_asset else "attribute filters"
        print(
            f"[urbanverse] Loading per-asset annotations for "
            f"{len(candidate_uids):,} candidates ({reason})…"
        )

    # ── Step 4: load per-asset annotations ────────────────────────────────────
    per_asset = _load_per_asset_annotations(candidate_uids, num_workers)

    # ── Step 5: apply attribute filters ───────────────────────────────────────
    if needs_per_asset:
        filtered_uids = [
            uid for uid in candidate_uids
            if per_asset.get(uid) is not None
            and _passes_filters(per_asset[uid], **attribute_filters)
        ]
    else:
        # No attribute filters, but we loaded annotations for text search —
        # keep all candidates whose annotations were successfully loaded.
        filtered_uids = [uid for uid in candidate_uids if per_asset.get(uid) is not None]

    # ── Step 6: text similarity ranking (optional) ────────────────────────────
    if not needs_text_search:
        return filtered_uids

    if not filtered_uids:
        return []

    if len(filtered_uids) > 1000:
        print(
            f"[urbanverse] Computing text similarity for "
            f"{len(filtered_uids):,} candidates (model: {embedding_model})…"
        )

    scored = _compute_text_scores(query, filtered_uids, per_asset, embedding_model)
    return [uid for uid, _ in scored[:top_k]]
