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
uva.tree — Vegetation tree assets (USD format).

Usage::

    import urbanverse_asset as uva

    names = uva.tree.get_descriptions()
    # ["American_Beech", "Black_Oak", "Colorado_Spruce", ...]

    results = uva.tree.load_materials(["American_Beech"])
    usd_path = results["American_Beech"]["usd"]      # Path to World0.usd
    folder   = results["American_Beech"]["folder"]    # Path to extracted folder
"""

from threading import Lock
from ._vegetation import _VegetationSource


class _Tree(_VegetationSource):
    _folder = "collected_trees"

    _cached_file_map = None
    _cache_lock = Lock()


get_descriptions = _Tree.get_descriptions
get_descriptions_conditioned = _Tree.get_descriptions_conditioned
load_materials = _Tree.load_materials
