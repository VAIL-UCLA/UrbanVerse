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
uva.road — Road surface materials (MDL format, IsaacSim-ready).

Usage::

    import urbanverse_asset as uva

    names = uva.road.get_descriptions()
    # ["Cobblestone_Medieval", "Concrete007_2K_PNG", ...]

    results = uva.road.load_materials(what=("mdl", "thumbnail"))
    mdl_path   = results["Cobblestone_Medieval"]["mdl"]        # Path to .mdl.png
    thumb_path  = results["Cobblestone_Medieval"]["thumbnail"]  # Path to thumbnail
"""

from threading import Lock
from ._material import _MaterialSource


class _Road(_MaterialSource):
    _folder_main        = "material_road_mdl"
    _folder_thumb       = "material_road_thumbnail"
    _main_key           = "mdl"
    _what_choices       = ("mdl", "thumbnail")
    _texture_subfolder  = "textures"

    _cached_file_map = None
    _cache_lock      = Lock()


get_descriptions             = _Road.get_descriptions
get_descriptions_conditioned = _Road.get_descriptions_conditioned
load_materials               = _Road.load_materials
