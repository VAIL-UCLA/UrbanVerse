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
uva.terrain — Terrain surface materials (MDL format, IsaacSim-ready).

Usage::

    import urbanverse_asset as uva

    names = uva.terrain.get_descriptions()
    # ["Grass008_2K_PNG", "Ground007_2K_PNG", ...]

    results = uva.terrain.load_materials()
    mdl_path = results["Grass008_2K_PNG"]["mdl"]
"""

from threading import Lock
from ._material import _MaterialSource


class _Terrain(_MaterialSource):
    _folder_main        = "material_terrain_mdl"
    _folder_thumb       = "material_terrain_thumbnail"
    _main_key           = "mdl"
    _what_choices       = ("mdl", "thumbnail")
    _texture_subfolder  = "textures"

    _cached_file_map = None
    _cache_lock      = Lock()


get_descriptions             = _Terrain.get_descriptions
get_descriptions_conditioned = _Terrain.get_descriptions_conditioned
load_materials               = _Terrain.load_materials
