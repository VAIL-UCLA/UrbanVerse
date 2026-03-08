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
uva.sidewalk — Sidewalk surface materials (MDL format, IsaacSim-ready).

Usage::

    import urbanverse_asset as uva

    names = uva.sidewalk.get_descriptions()
    # ["Asphalt_Fine", "PavingStones011_2K_PNG", ...]

    results = uva.sidewalk.load_materials()
    mdl_path = results["Asphalt_Fine"]["mdl"]
"""

from threading import Lock
from ._material import _MaterialSource


class _Sidewalk(_MaterialSource):
    _folder_main        = "material_sidewalk_mdl"
    _folder_thumb       = "material_sidewalk_thumbnail"
    _main_key           = "mdl"
    _what_choices       = ("mdl", "thumbnail")
    _texture_subfolder  = "textures"

    _cached_file_map = None
    _cache_lock      = Lock()


get_descriptions             = _Sidewalk.get_descriptions
get_descriptions_conditioned = _Sidewalk.get_descriptions_conditioned
load_materials               = _Sidewalk.load_materials
