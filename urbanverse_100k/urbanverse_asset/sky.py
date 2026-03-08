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
uva.sky — HDR sky / environment maps.

Usage::

    import urbanverse_asset as uva

    names = uva.sky.get_descriptions()
    # ["autumn_field_puresky", "autumn_hockey", "buikslotermeerplein", ...]

    results = uva.sky.load_materials(what=("hdr", "thumbnail"))
    hdr_path  = results["autumn_field_puresky"]["hdr"]        # Path to .hdr
    thumb_path = results["autumn_field_puresky"]["thumbnail"]  # Path to .png
"""

from threading import Lock
from ._material import _MaterialSource


class _Sky(_MaterialSource):
    _folder_main  = "material_background_hdr"
    _folder_thumb = "material_background_thumbnail"
    _main_key     = "hdr"
    _what_choices = ("hdr", "thumbnail")

    # Each subclass gets its own cache slot
    _cached_file_map = None
    _cache_lock      = Lock()


# Expose classmethods as module-level functions so callers write
# ``uva.sky.get_descriptions()`` rather than ``uva.sky._Sky.get_descriptions()``
get_descriptions            = _Sky.get_descriptions
get_descriptions_conditioned = _Sky.get_descriptions_conditioned
load_materials              = _Sky.load_materials
