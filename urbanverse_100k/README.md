# urbanverse_asset

Python package for programmatic access to the **UrbanVerse-100K** 3D asset dataset,
hosted on HuggingFace at [Oatmealliu/UrbanVerse-100K](https://huggingface.co/datasets/Oatmealliu/UrbanVerse-100K).

UrbanVerse-100K contains **102,444 metric-scale 3D object assets** with rich
annotations, plus **646 sky HDRIs**, **98 road / 190 sidewalk / 115 terrain PBR
materials**, and **plant / shrub / tree vegetation assets** — all curated for
autonomous driving and urban simulation.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Architecture](#package-architecture)
- [API Cheatsheet](#api-cheatsheet)
- [API Reference](#api-reference)
  - [Configuration](#configuration)
  - [Object Assets — `uva.object`](#object-assets--uvaobject)
  - [Material Assets — `uva.sky` / `uva.road` / `uva.sidewalk` / `uva.terrain]`](#material-assets)
  - [Vegetation Assets — `uva.plant` / `uva.shrub` / `uva.tree`](#vegetation-assets)
  - [Viewers — `uva.viewer`](#viewers--uvaviewer)
- [Caching & Storage](#caching--storage)
- [Authentication](#authentication)
- [IsaacSim Integration](#isaacsim-integration)

---

## Installation

```bash
pip install urbanverse-asset
```

**Optional — for GLB-to-USD conversion API (requires NVIDIA IsaacSim and IsaacLab):**

```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
git clone https://github.com/OatmealLiu/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install
```

---

## Quick Start

> **Interactive notebook:** For a step-by-step walkthrough with explanations, see [`demo_usage.ipynb`](demo_usage.ipynb).

```python
import urbanverse_asset as uva

# (Optional but recommended) set a custom cache directory
# Default: ~/.cache/urbanverse/
uva.set("~/datasets/urbanverse")

# Dataset metadata
meta = uva.info()
print(meta["statistics"]["number_of_assets"])  # 102444

# Get all UIDs
all_uids = uva.object.get_uids_all()

# Understand per-asset annotation fields
uva.object.explain_annotation()             # prints all fields
uva.object.explain_annotation("mass")       # prints explanation for "mass"

# Filter by attributes + semantic text search
vehicle_uids = uva.object.get_uids_conditioned(
    categories=["vehicle"],
    query="Old yellow Italian style two-door car",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    top_k=10,
)

# Download everything for selected UIDs
result = uva.object.load(vehicle_uids)
glb_path = result[vehicle_uids[0]]["std_glb"]

# Browse in an interactive 3D viewer
uva.viewer.object_show(vehicle_uids)

# Combine attribute filters with text search
barrier_uids = uva.object.get_uids_conditioned(
    categories=["traffic cone", "traffic barricade", "jersey barrier"],
    height_range=(0.0, 1.2),
    query="Yellow obstacle",
    top_k=10,
)
uva.viewer.object_show(barrier_uids)

# Sky materials
sunny = uva.sky.get_descriptions_conditioned(query="A sunny day", top_k=10)
uva.sky.load_materials(sunny)
uva.viewer.sky_show(sunny)

# Road materials
highway = uva.road.get_descriptions_conditioned(query="Highway", top_k=10)
uva.road.load_materials(highway)
uva.viewer.road_show(highway)

# Sidewalk materials
stone = uva.sidewalk.get_descriptions_conditioned(query="Bumpy stone road", top_k=10)
uva.sidewalk.load_materials(stone)
uva.viewer.sidewalk_show(stone)

# Terrain materials
grass = uva.terrain.get_descriptions_conditioned(query="Grassland", top_k=10)
uva.terrain.load_materials(grass)
uva.viewer.terrain_show(grass)

# Download the entire dataset
uva.download_all(num_workers=16)
```

---

## Package Architecture

```
urbanverse_asset/
├── __init__.py        # Top-level: set(), info(), download_all(), check_integrity(), repair()
├── object.py          # 3D object APIs: get_uids_all(), load(), etc.
├── sky.py             # Sky HDRI materials
├── road.py            # Road PBR materials
├── sidewalk.py        # Sidewalk PBR materials
├── terrain.py         # Terrain PBR materials
├── plant.py           # Plant vegetation assets
├── shrub.py           # Shrub vegetation assets
├── tree.py            # Tree vegetation assets
├── viewer.py          # Browser-based viewers for all asset types
├── _core.py           # Shared: cache, HF downloads, bucket indices
├── _filter.py         # Attribute & text-based UID filtering
├── _material.py       # Base class for material modules
├── _vegetation.py     # Base class for vegetation modules
├── _glb_to_usd.py     # GLB → USD conversion script (IsaacSim)
├── _usd_to_glb.py     # USD → GLB preview conversion
└── _viewer/           # HTML templates for browser viewers
```

**Namespace overview:**


| Namespace                                                                                | Purpose                                                  |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `uva.set()`, `uva.info()`, `uva.download_all()`, `uva.check_integrity()`, `uva.repair()` | Global configuration, metadata & maintenance             |
| `uva.object.`*                                                                           | 3D object assets (GLB, annotations, thumbnails, renders) |
| `uva.sky.`*, `uva.road.*`, `uva.sidewalk.*`, `uva.terrain.*`                             | PBR materials                                            |
| `uva.plant.*`, `uva.shrub.*`, `uva.tree.*`                                               | Vegetation assets                                        |
| `uva.viewer.*`                                                                           | Interactive browser-based viewers                        |


---

## API Cheatsheet

```python
import urbanverse_asset as uva
```

### Configuration


| API                               | Description                                            |
| --------------------------------- | ------------------------------------------------------ |
| `uva.set(path)`                   | Set local cache directory                              |
| `uva.info()`                      | Dataset metadata (version, counts, license)            |
| `uva.download_all(num_workers=8)` | Download everything (objects + materials + vegetation) |
| `uva.check_integrity()`           | Scan cache and report completeness per asset type      |
| `uva.repair(num_workers=8)`       | Download only the missing files to complete the cache  |


### Object Assets — `uva.object`


| API                                     | Description                                                       |
| --------------------------------------- | ----------------------------------------------------------------- |
| `uva.object.get_uids_all()`             | All 102,444 UIDs                                                  |
| `uva.object.get_uids_conditioned(...)`  | Filter UIDs by attributes and/or text query                       |
| `uva.object.categories(level=None)`     | Category hierarchy with UIDs per class                            |
| `uva.object.load(uids, what=...)`       | Download all asset types (GLB + annotation + thumbnail + renders) |
| `uva.object.load_assets(uids)`          | Download GLB files only                                           |
| `uva.object.load_annotations(uids)`     | Download annotation JSONs                                         |
| `uva.object.load_thumbnails(uids)`      | Download thumbnail PNGs                                           |
| `uva.object.load_renders(uids, angles)` | Download multi-angle render images                                |
| `uva.object.convert_glb_to_usd(uids)`   | Convert GLB to IsaacSim USD (requires Isaac Lab)                  |
| `uva.object.explain_annotation(field)`  | Explain annotation fields                                         |


#### Filtering — `uva.object.get_uids_conditioned(...)`

All parameters are optional. Attribute filters use AND logic; list values use OR.

**Attribute filters:**
`categories`, `height_range`, `length_range`, `width_range`, `max_dimension_range`,
`mass_range`, `quality_range`, `dominant_materials`, `receptacle`, `movable`,
`required_force_range`, `walkable`, `enterable`, `affordances`, `support_surface`,
`interactive_parts`, `dominant_colors`, `surface_hardness`, `surface_roughness_range`,
`surface_finish`, `reflectivity_range`, `index_of_refraction_range`,
`youngs_modulus_range`, `friction_coefficient_range`, `bounciness_range`,
`recommended_clearance_range`, `asset_composition_type`

**Text query:** `query`, `top_k=100`, `embedding_model="sentence-transformers/all-mpnet-base-v2"`

```python
# Attribute filter
uids = uva.object.get_uids_conditioned(categories=["vehicle"], movable=True)

# Text search
uids = uva.object.get_uids_conditioned(query="Old yellow Italian style two-door car", top_k=10)

# Combined
uids = uva.object.get_uids_conditioned(
    categories=["vehicle"],
    query="Old yellow Italian style two-door car",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    top_k=10,
)
```

### Materials — `uva.sky` / `uva.road` / `uva.sidewalk` / `uva.terrain`

All four modules share the same API:


| API                                                       | Description                                                           |
| --------------------------------------------------------- | --------------------------------------------------------------------- |
| `uva.<mod>.get_descriptions()`                            | List all material names                                               |
| `uva.<mod>.get_descriptions_conditioned(query, top_k=10)` | Search materials by text similarity                                   |
| `uva.<mod>.load_materials(descriptions)`                  | Download materials → `{name: {"hdr"/"mdl": Path, "thumbnail": Path}}` |


```python
descs = uva.sky.get_descriptions()                                              # 646 sky maps
sunny = uva.sky.get_descriptions_conditioned(query="A sunny day", top_k=10)     # text search
result = uva.sky.load_materials(sunny[:3])                                      # download

highway = uva.road.get_descriptions_conditioned(query="Highway", top_k=10)
stone = uva.sidewalk.get_descriptions_conditioned(query="Bumpy stone road", top_k=10)
grass = uva.terrain.get_descriptions_conditioned(query="Grassland", top_k=10)
```

### Vegetation — `uva.plant` / `uva.shrub` / `uva.tree`

All three modules share the same API:


| API                                                       | Description                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| `uva.<mod>.get_descriptions()`                            | List all vegetation asset names                              |
| `uva.<mod>.get_descriptions_conditioned(query, top_k=10)` | Search by text similarity                                    |
| `uva.<mod>.load_materials(descriptions)`                  | Download & extract → `{name: {"usd": Path, "folder": Path}}` |


```python
trees = uva.tree.get_descriptions()
result = uva.tree.load_materials(trees[:3])
```

### Viewers — `uva.viewer`

Open interactive browser-based viewers. Keep the Python process running while viewing.

#### Object Viewers


| API                                       | Description                             |
| ----------------------------------------- | --------------------------------------- |
| `uva.viewer.object_distribution()`        | Category distribution sunburst chart    |
| `uva.viewer.object_show(uids)`            | Full viewer (GLB + images + annotation) |
| `uva.viewer.object_assets(uids)`          | 3D GLB viewer only                      |
| `uva.viewer.object_annotations(uids)`     | Annotation table only                   |
| `uva.viewer.object_thumbnails(uids)`      | Thumbnail gallery                       |
| `uva.viewer.object_renders(uids, angles)` | Render image strip                      |


#### Material Viewers


| API                                      | Description                     |
| ---------------------------------------- | ------------------------------- |
| `uva.viewer.sky_show(descriptions)`      | IBL sky viewer (3 PBR spheres)  |
| `uva.viewer.road_show(descriptions)`     | 3D PBR road material viewer     |
| `uva.viewer.sidewalk_show(descriptions)` | 3D PBR sidewalk material viewer |
| `uva.viewer.terrain_show(descriptions)`  | 3D PBR terrain material viewer  |


#### Vegetation Viewers


| API                                   | Description     |
| ------------------------------------- | --------------- |
| `uva.viewer.plant_show(descriptions)` | 3D plant viewer |
| `uva.viewer.shrub_show(descriptions)` | 3D shrub viewer |
| `uva.viewer.tree_show(descriptions)`  | 3D tree viewer  |


```python
uva.viewer.object_show(uids[:5])
uva.viewer.sky_show(["001_autumn_afternoon_clear_sky_abandoned_outdoor_stage"])
```

**Viewer controls:** Arrow keys / A,D to navigate | Mouse drag to orbit | Scroll to zoom

---

## API Reference

### Configuration

#### `uva.set(path)`

Set the local directory where all UrbanVerse data will be saved. Persisted across
Python sessions in `~/.cache/urbanverse_config.json`.


| Parameter | Type  | Default | Description                                                  |
| --------- | ----- | ------- | ------------------------------------------------------------ |
| `path`    | `str` | —       | Absolute or `~`-relative path. Created if it does not exist. |


```python
uva.set("~/datasets/urbanverse")
uva.set("/mnt/ssd/urbanverse")
```

---

#### `uva.info() -> Dict[str, Any]`

Return dataset-level metadata. Downloads the master annotation on first call;
subsequent calls are instant (in-memory cache).

**Returns:** dict with keys `version`, `date_created`, `description`, `license`,
`contributor`, `url`, `citation`, and `statistics` (containing
`number_of_assets`, `number_of_classes_l1`, `number_of_classes_l2`,
`number_of_classes_l3`).

```python
meta = uva.info()
print(meta["statistics"]["number_of_assets"])  # 102444
print(meta["license"])                         # "CC BY-NC-SA 4.0"
```

---

#### `uva.download_all(num_workers=8)`

Download the **entire** UrbanVerse-100K dataset — all object assets (GLB,
annotations, thumbnails, renders), all materials (sky, road, sidewalk, terrain),
and all vegetation (plant, shrub, tree).


| Parameter     | Type  | Default | Description                |
| ------------- | ----- | ------- | -------------------------- |
| `num_workers` | `int` | `8`     | Parallel download threads. |


```python
uva.download_all()
uva.download_all(num_workers=16)
```

---

#### `uva.check_integrity() -> Dict[str, Any]`

Scan the local cache and report download completeness per asset type. Compares
files on disk against the expected file lists from the HuggingFace repo. **No
data files are downloaded** (only small metadata files like bucket indices are
fetched if not already cached).

**Returns:** a dict with per-category counts (`expected`, `downloaded`,
`missing`), a boolean `complete` flag, and a `total_missing` count.

```python
report = uva.check_integrity()
print(report["object"]["glb"])   # {"expected": 102444, "downloaded": 102444, "missing": 0}
print(report["complete"])        # True / False
print(report["total_missing"])   # 0
```

**Report structure:**

```python
{
    "cache_dir": "/path/to/.cache/urbanverse",
    "object": {
        "annotations": {"expected": ..., "downloaded": ..., "missing": ...},
        "glb":         {"expected": ..., "downloaded": ..., "missing": ...},
        "thumbnails":  {"expected": ..., "downloaded": ..., "missing": ...},
        "renders":     {"expected": ..., "downloaded": ..., "missing": ...},
    },
    "sky":       {"expected": ..., "downloaded": ..., "missing": ...},
    "road":      {"expected": ..., "downloaded": ..., "missing": ...},
    "sidewalk":  {"expected": ..., "downloaded": ..., "missing": ...},
    "terrain":   {"expected": ..., "downloaded": ..., "missing": ...},
    "plant":     {"expected": ..., "downloaded": ..., "missing": ...},
    "shrub":     {"expected": ..., "downloaded": ..., "missing": ...},
    "tree":      {"expected": ..., "downloaded": ..., "missing": ...},
    "complete":      True/False,
    "total_missing": 0,
}
```

---

#### `uva.repair(num_workers=8) -> Dict[str, Any]`

Automatically fix an incomplete download by selectively downloading only the
missing files. Runs `check_integrity()` first; if everything is already present,
returns immediately. After downloading, re-checks integrity and returns the
updated report.


| Parameter     | Type  | Default | Description                |
| ------------- | ----- | ------- | -------------------------- |
| `num_workers` | `int` | `8`     | Parallel download threads. |


```python
report = uva.repair()
assert report["complete"]

# With more threads for faster repair
report = uva.repair(num_workers=16)
```

---

### Object Assets — `uva.object`

#### `uva.object.get_uids_all() -> List[str]`

Return all 102,444 asset UIDs in the dataset.

```python
all_uids = uva.object.get_uids_all()
print(len(all_uids))  # 102444
```

---

#### `uva.object.categories(level=None) -> Dict[str, Any]`

Return the category hierarchy with UIDs per class.


| Parameter | Type            | Default | Description                                                                          |
| --------- | --------------- | ------- | ------------------------------------------------------------------------------------ |
| `level`   | `int` or `None` | `None`  | `1`, `2`, or `3` for a specific level; `None` returns the full annotation hierarchy. |


```python
cats = uva.object.categories(level=1)
print(list(cats.keys()))  # ['vehicle', 'building', 'barrier', ...]
```

---

#### `uva.object.get_uids_conditioned(...) -> List[str]`

Filter UIDs by per-asset attributes and/or a natural-language text query. All
conditions are combined with AND logic. Omitted parameters are not filtered.

**Attribute filters:**


| Parameter                     | Type             | Description                                 |
| ----------------------------- | ---------------- | ------------------------------------------- |
| `categories`                  | `List[str]`      | L1/L2/L3 category names (case-insensitive). |
| `height_range`                | `(float, float)` | Min/max height in meters.                   |
| `length_range`                | `(float, float)` | Min/max length in meters.                   |
| `width_range`                 | `(float, float)` | Min/max width in meters.                    |
| `max_dimension_range`         | `(float, float)` | Min/max of the largest dimension.           |
| `mass_range`                  | `(float, float)` | Min/max estimated mass in kg.               |
| `quality_range`               | `(int, int)`     | Min/max quality score (1–5).                |
| `dominant_materials`          | `List[str]`      | e.g. `["metal", "glass"]`.                  |
| `receptacle`                  | `bool`           | Can the object contain other objects?       |
| `movable`                     | `bool`           | Can a human move this object?               |
| `required_force_range`        | `(float, float)` | Force range in Newtons.                     |
| `walkable`                    | `bool`           | Can a person walk on/over this object?      |
| `enterable`                   | `bool`           | Can a person enter this object?             |
| `affordances`                 | `List[str]`      | e.g. `["sit", "open"]`.                     |
| `support_surface`             | `bool`           | Does it have a flat support surface?        |
| `interactive_parts`           | `List[str]`      | e.g. `["door", "drawer"]`.                  |
| `dominant_colors`             | `List[str]`      | e.g. `["red", "blue"]`.                     |
| `surface_hardness`            | `List[str]`      | e.g. `["hard", "soft"]`.                    |
| `surface_roughness_range`     | `(float, float)` | 0.0–1.0 roughness range.                    |
| `surface_finish`              | `List[str]`      | e.g. `["matte", "glossy"]`.                 |
| `reflectivity_range`          | `(float, float)` | 0.0–1.0 reflectivity range.                 |
| `index_of_refraction_range`   | `(float, float)` | IOR range.                                  |
| `youngs_modulus_range`        | `(float, float)` | Young's modulus range (Pa).                 |
| `friction_coefficient_range`  | `(float, float)` | Friction coefficient range.                 |
| `bounciness_range`            | `(float, float)` | Bounciness / restitution range.             |
| `recommended_clearance_range` | `(float, float)` | Clearance range in meters.                  |
| `asset_composition_type`      | `List[str]`      | e.g. `["single_mesh", "multi_mesh"]`.       |


**Text query parameters:**


| Parameter         | Type  | Default                                     | Description                                                                        |
| ----------------- | ----- | ------------------------------------------- | ---------------------------------------------------------------------------------- |
| `query`           | `str` | `None`                                      | Natural language description to search by semantic similarity.                     |
| `top_k`           | `int` | `100`                                       | Maximum number of results to return when using `query`.                            |
| `embedding_model` | `str` | `"sentence-transformers/all-mpnet-base-v2"` | Sentence embedding model. Alternative: `"sentence-transformers/all-MiniLM-L6-v2"`. |


**Other:**


| Parameter     | Type  | Default | Description                                |
| ------------- | ----- | ------- | ------------------------------------------ |
| `num_workers` | `int` | `8`     | Parallel download threads for annotations. |


```python
# Filter by attributes
uids = uva.object.get_uids_conditioned(
    categories=["vehicle"],
    height_range=(1.5, 3.0),
    movable=True,
)

# Semantic text search
uids = uva.object.get_uids_conditioned(
    query="Old yellow Italian style two-door car",
    top_k=5,
)

# Combine both
uids = uva.object.get_uids_conditioned(
    categories=["vehicle"],
    query="Old yellow Italian style two-door car",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    top_k=10,
)
```

---

#### `uva.object.load(uids=None, num_workers=8, what=("std_glb", "std_annotation", "thumbnail", "render")) -> Dict[str, Dict[str, Any]]`

All-in-one downloader. Downloads the specified asset types for each UID and
returns a nested dict of local paths.


| Parameter     | Type                  | Default                                                | Description                     |
| ------------- | --------------------- | ------------------------------------------------------ | ------------------------------- |
| `uids`        | `List[str]` or `None` | `None`                                                 | UIDs to download. `None` = all. |
| `num_workers` | `int`                 | `8`                                                    | Parallel threads.               |
| `what`        | `tuple`               | `("std_glb", "std_annotation", "thumbnail", "render")` | Which asset types to download.  |


**Returns:** `{uid: {"std_glb": Path | None, "std_annotation": Path | None, "thumbnail": Path | None, "render": {angle: Path | None}}}`.

```python
result = uva.object.load(uids[:5])
glb = result[uids[0]]["std_glb"]       # Path to .glb
ann = result[uids[0]]["std_annotation"] # dict
```

---

#### `uva.object.load_assets(uids=None, num_workers=8) -> Dict[str, Optional[Path]]`

Download only the metric-scale `.glb` files.

```python
paths = uva.object.load_assets(uids[:10])
glb = paths[uids[0]]  # Path to .glb file
```

---

#### `uva.object.load_annotations(uids=None, num_workers=8) -> Dict[str, Optional[Dict]]`

Download per-asset annotation JSON files and return them as parsed dicts.

```python
anns = uva.object.load_annotations(uids[:10])
print(anns[uids[0]]["description"])
```

---

#### `uva.object.load_thumbnails(uids=None, num_workers=8) -> Dict[str, Optional[Path]]`

Download thumbnail `.png` images.

```python
thumbs = uva.object.load_thumbnails(uids[:10])
print(thumbs[uids[0]])  # Path to .png
```

---

#### `uva.object.load_renders(uids=None, angles=(0, 90, 180, 270), num_workers=8) -> Dict[str, Dict[int, Optional[Path]]]`

Download multi-angle render images.


| Parameter     | Type                  | Default             | Description                                         |
| ------------- | --------------------- | ------------------- | --------------------------------------------------- |
| `uids`        | `List[str]` or `None` | `None`              | UIDs to download.                                   |
| `angles`      | `Tuple[int, ...]`     | `(0, 90, 180, 270)` | Render angles in degrees. Valid: `0, 90, 180, 270`. |
| `num_workers` | `int`                 | `8`                 | Parallel threads.                                   |


```python
renders = uva.object.load_renders(uids[:3], angles=(0, 180))
print(renders[uids[0]][0])    # Path to render_0.0.jpg
print(renders[uids[0]][180])  # Path to render_180.0.jpg
```

---

#### `uva.object.convert_glb_to_usd(...) -> Dict[str, Any]`

Convert metric-scale GLB assets to IsaacSim-ready USD format. **Requires NVIDIA
IsaacSim and Isaac Lab.**


| Parameter                 | Type                  | Default                 | Description                  |
| ------------------------- | --------------------- | ----------------------- | ---------------------------- |
| `uids`                    | `List[str]` or `None` | `None`                  | UIDs to convert.             |
| `num_workers`             | `int`                 | `8`                     | Parallel conversion workers. |
| `headless`                | `bool`                | `True`                  | Run IsaacSim headless.       |
| `collision_approximation` | `str`                 | `"convexDecomposition"` | Collision mesh strategy.     |
| `make_instanceable`       | `bool`                | `False`                 | Generate instanceable USD.   |
| `mass`                    | `float` or `None`     | `None`                  | Override mass in kg.         |


```python
result = uva.object.convert_glb_to_usd(uids[:5])
usd_path = result[uids[0]]  # Path to .usd or None
```

---

#### `uva.object.explain_annotation(field=None) -> Dict[str, str]`

Print and return human-readable explanations of per-asset annotation fields.


| Parameter | Type            | Default | Description                                      |
| --------- | --------------- | ------- | ------------------------------------------------ |
| `field`   | `str` or `None` | `None`  | A specific field name, or `None` for all fields. |


```python
uva.object.explain_annotation()             # prints all fields
uva.object.explain_annotation("mass")       # prints explanation for "mass"
```

---

### Material Assets

All four material modules — `uva.sky`, `uva.road`, `uva.sidewalk`, `uva.terrain`
— share the same API pattern:

#### `uva.<module>.get_descriptions() -> List[str]`

Return a sorted list of all available material names.

```python
sky_names = uva.sky.get_descriptions()       # 646 sky maps
road_names = uva.road.get_descriptions()     # 98 road materials
sw_names = uva.sidewalk.get_descriptions()   # 190 sidewalk materials
terr_names = uva.terrain.get_descriptions()  # 115 terrain materials
```

---

#### `uva.<module>.get_descriptions_conditioned(query, top_k=10, embedding_model=...) -> List[str]`

Return material names ranked by semantic similarity to a text query.


| Parameter         | Type  | Default                                     | Description                    |
| ----------------- | ----- | ------------------------------------------- | ------------------------------ |
| `query`           | `str` | —                                           | Natural language search query. |
| `top_k`           | `int` | `10`                                        | Maximum results to return.     |
| `embedding_model` | `str` | `"sentence-transformers/all-mpnet-base-v2"` | Sentence embedding model.      |


```python
uva.sky.get_descriptions_conditioned(query="A sunny day", top_k=10)
uva.road.get_descriptions_conditioned(query="Highway", top_k=10)
uva.sidewalk.get_descriptions_conditioned(query="Bumpy stone road", top_k=10)
uva.terrain.get_descriptions_conditioned(query="Grassland", top_k=10)
```

---

#### `uva.<module>.load_materials(descriptions=None, num_workers=8) -> Dict[str, Dict[str, Optional[Path]]]`

Download material files and return local paths.


| Parameter      | Type                  | Default | Description                               |
| -------------- | --------------------- | ------- | ----------------------------------------- |
| `descriptions` | `List[str]` or `None` | `None`  | Material names to download. `None` = all. |
| `num_workers`  | `int`                 | `8`     | Parallel threads.                         |


**Return keys by module:**


| Module     | Keys in returned dict  |
| ---------- | ---------------------- |
| `sky`      | `"hdr"`, `"thumbnail"` |
| `road`     | `"mdl"`, `"thumbnail"` |
| `sidewalk` | `"mdl"`, `"thumbnail"` |
| `terrain`  | `"mdl"`, `"thumbnail"` |


```python
result = uva.sky.load_materials(["001_autumn_afternoon_clear_sky_abandoned_outdoor_stage"])
hdr_path = result["001_autumn_afternoon_clear_sky_abandoned_outdoor_stage"]["hdr"]

result = uva.road.load_materials(["001_aged_gray_concrete_wall"])
mdl_path = result["001_aged_gray_concrete_wall"]["mdl"]
```

---

### Vegetation Assets

All three vegetation modules — `uva.plant`, `uva.shrub`, `uva.tree` — share the
same API pattern:

#### `uva.<module>.get_descriptions() -> List[str]`

Return a sorted list of available vegetation asset names.

```python
plants = uva.plant.get_descriptions()
shrubs = uva.shrub.get_descriptions()
trees  = uva.tree.get_descriptions()
```

---

#### `uva.<module>.get_descriptions_conditioned(query, top_k=10, embedding_model=...) -> List[str]`

Return vegetation names ranked by semantic similarity to a text query.


| Parameter         | Type  | Default                                     | Description                    |
| ----------------- | ----- | ------------------------------------------- | ------------------------------ |
| `query`           | `str` | —                                           | Natural language search query. |
| `top_k`           | `int` | `10`                                        | Maximum results to return.     |
| `embedding_model` | `str` | `"sentence-transformers/all-mpnet-base-v2"` | Sentence embedding model.      |


```python
uva.tree.get_descriptions_conditioned(query="tall oak tree", top_k=10)
```

---

#### `uva.<module>.load_materials(descriptions=None, num_workers=8) -> Dict[str, Dict[str, Any]]`

Download, extract, and return local paths for vegetation assets.


| Parameter      | Type                  | Default | Description                            |
| -------------- | --------------------- | ------- | -------------------------------------- |
| `descriptions` | `List[str]` or `None` | `None`  | Asset names to download. `None` = all. |
| `num_workers`  | `int`                 | `8`     | Parallel threads.                      |


**Return keys:** `"usd"` (Path to `.usd` file), `"folder"` (Path to extracted asset directory).

```python
result = uva.plant.load_materials(["some_plant_name"])
usd_path = result["some_plant_name"]["usd"]
```

---

### Viewers — `uva.viewer`

All viewer functions download the required assets (if not cached), generate an
HTML page, start a local HTTP server, and open the viewer in your default browser.
The Python process must remain running for the viewer to work (it serves files
via `http://127.0.0.1`).

#### Object Viewers


| Function                                                                | Description                                                          |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `uva.viewer.object_distribution()`                                      | Interactive sunburst chart of the dataset's category distribution.   |
| `uva.viewer.object_show(uids, num_workers=8, angles=(0,90,180,270))`    | Full viewer: 3D GLB model + thumbnails + renders + annotation table. |
| `uva.viewer.object_assets(uids, num_workers=8)`                         | 3D GLB model viewer only.                                            |
| `uva.viewer.object_annotations(uids, num_workers=8)`                    | Annotation table viewer only.                                        |
| `uva.viewer.object_thumbnails(uids, num_workers=8)`                     | Thumbnail image viewer only.                                         |
| `uva.viewer.object_renders(uids, angles=(0,90,180,270), num_workers=8)` | Render image strip viewer.                                           |


All object viewer functions accept:


| Parameter     | Type                  | Default             | Description                                                      |
| ------------- | --------------------- | ------------------- | ---------------------------------------------------------------- |
| `uids`        | `List[str]` or `None` | `None`              | UIDs to view. `None` = all (not recommended for large datasets). |
| `num_workers` | `int`                 | `8`                 | Parallel download threads.                                       |
| `angles`      | `Tuple[int, ...]`     | `(0, 90, 180, 270)` | Render angles (where applicable). Valid: `0, 90, 180, 270`.      |


```python
uids = uva.object.get_uids_conditioned(categories=["vehicle"], query="Old yellow Italian style two-door car", top_k=10)

uva.viewer.object_distribution()        # sunburst chart
uva.viewer.object_show(uids[:5])        # full viewer
uva.viewer.object_assets(uids[:5])      # GLB-only 3D viewer
uva.viewer.object_annotations(uids[:5]) # annotation tables
uva.viewer.object_thumbnails(uids[:5])  # thumbnail gallery
uva.viewer.object_renders(uids[:5], angles=(0, 180))  # render strip
```

**Viewer controls:**

- Arrow keys or A/D to navigate between assets
- Mouse drag to orbit, scroll to zoom (3D viewers)
- GUI panel for wireframe, exposure, auto-rotate, and camera reset

---

#### Material Viewers


| Function                                                | Description                                                             |
| ------------------------------------------------------- | ----------------------------------------------------------------------- |
| `uva.viewer.sky_show(descriptions, num_workers=8)`      | IBL sky viewer with chrome/roughness/matte spheres under HDRI lighting. |
| `uva.viewer.road_show(descriptions, num_workers=8)`     | 3D PBR plane viewer for road materials with texture channels.           |
| `uva.viewer.sidewalk_show(descriptions, num_workers=8)` | 3D PBR plane viewer for sidewalk materials.                             |
| `uva.viewer.terrain_show(descriptions, num_workers=8)`  | 3D PBR plane viewer for terrain materials.                              |



| Parameter      | Type                  | Default | Description                           |
| -------------- | --------------------- | ------- | ------------------------------------- |
| `descriptions` | `List[str]` or `None` | `None`  | Material names to view. `None` = all. |
| `num_workers`  | `int`                 | `8`     | Parallel download threads.            |


```python
sky_descs = uva.sky.get_descriptions()
uva.viewer.sky_show(sky_descs[:5])

road_descs = uva.road.get_descriptions()
uva.viewer.road_show(road_descs[:5])
```

---

#### Vegetation Viewers


| Function                                             | Description                                                    |
| ---------------------------------------------------- | -------------------------------------------------------------- |
| `uva.viewer.plant_show(descriptions, num_workers=8)` | 3D viewer for plant assets (USD converted to GLB for preview). |
| `uva.viewer.shrub_show(descriptions, num_workers=8)` | 3D viewer for shrub assets.                                    |
| `uva.viewer.tree_show(descriptions, num_workers=8)`  | 3D viewer for tree assets.                                     |


```python
plant_descs = uva.plant.get_descriptions()
uva.viewer.plant_show(plant_descs[:3])
```

---

## Caching & Storage

- **Default cache:** `~/.cache/urbanverse/`
- **Override:** call `uva.set("your/path")` before any download. The choice is
persisted in `~/.cache/urbanverse_config.json`.
- All download functions **skip files that already exist locally** — re-running
is always safe and instant for cached files.
- Render archives (`.tar.gz`) are automatically extracted into bucketed folders
and the archives are deleted after extraction to save disk space.
- The annotation bundle (`full_per_asset_annotations.tar.gz`) is automatically
downloaded and extracted on first use of any annotation-dependent API.

---

## Authentication

For private HuggingFace repos, set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN="hf_your_token_here"
```

Or log in via the HuggingFace CLI:

```bash
huggingface-cli login
```

---

## IsaacSim Integration

The `uva.object.convert_glb_to_usd()` function converts metric-scale GLB assets
into IsaacSim-ready USD files with collision meshes. This requires NVIDIA
IsaacSim and Isaac Lab:

```bash
# Install IsaacSim 4.5.0
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab (from within the package directory)
cd urbanverse_asset/IsaacLab
./isaaclab.sh --install
```

```python
result = uva.object.convert_glb_to_usd(
    uids=uids[:5],
    headless=True,
    collision_approximation="convexDecomposition",
)
```

---

## License

The UrbanVerse-100K dataset is released under **CC BY-NC-SA 4.0**.