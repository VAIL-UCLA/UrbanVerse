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
uva.plant — Vegetation plant assets (USD format).

Usage::

    import urbanverse_asset as uva

    names = uva.plant.get_descriptions()
    # ["Agave", "Crane_Lily", "Dagger", "PlantPot", ...]

    results = uva.plant.load_materials(["Plant_02"])
    usd_path = results["Plant_02"]["usd"]      # Path to World0.usd
    folder   = results["Plant_02"]["folder"]    # Path to extracted folder
"""

from threading import Lock
from ._vegetation import _VegetationSource


class _Plant(_VegetationSource):
    _folder = "collected_plants"

    _cached_file_map = None
    _cache_lock = Lock()


get_descriptions = _Plant.get_descriptions
get_descriptions_conditioned = _Plant.get_descriptions_conditioned
load_materials = _Plant.load_materials
