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
uva.shrub — Vegetation shrub assets (USD format).

Usage::

    import urbanverse_asset as uva

    names = uva.shrub.get_descriptions()
    # ["Acacia", "Barberry", "Boxwood", ...]

    results = uva.shrub.load_materials(["Acacia"])
    usd_path = results["Acacia"]["usd"]      # Path to World0.usd
    folder   = results["Acacia"]["folder"]    # Path to extracted folder
"""

from threading import Lock
from ._vegetation import _VegetationSource


class _Shrub(_VegetationSource):
    _folder = "collected_shrubs"

    _cached_file_map = None
    _cache_lock = Lock()


get_descriptions = _Shrub.get_descriptions
get_descriptions_conditioned = _Shrub.get_descriptions_conditioned
load_materials = _Shrub.load_materials
