---
license: cc-by-4.0
task_categories:
  - robotics
tags:
  - urbanverse
  - isaac-sim
  - isaac-lab
  - usd
  - simulation
  - urban-navigation
pretty_name: UrbanVerse Training Scenes (Sim-Ready)
---

# UrbanVerse-Training-Scenes-Sim-Ready

The [UrbanVerse training scenes](https://huggingface.co/datasets/UCLA-VAIL/UrbanVerse-Training-Scenes)
upgraded to load correctly in **NVIDIA Isaac Sim ≥ 5**. Same scenes, same layout, same
`World0.usd` entry point - only the material layers that broke between Isaac Sim 4.5 and
5.x were changed.

Project: [UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos](https://urbanverseproject.github.io/) (ICLR 2026) ·
Code: [VAIL-UCLA/UrbanVerse](https://github.com/VAIL-UCLA/UrbanVerse)

![before/after](https://raw.githubusercontent.com/VAIL-UCLA/UrbanVerse/main/material/simready_scene03_before_after.png)

## What changed, and why

The scenes were exported from Isaac Sim 4.5 with ground materials whose
`inputs:texture_scale` was authored as an `int` while the MDL declares `float2`.

- Isaac Sim **4.5** rejected that value (`[UsdToMdl] Tried to assign a 'int'(USD) to a
  'float2'(MDL)`) and silently fell back to the material's default tile size - that is
  what the scenes looked like when the paper's policies were trained.
- Isaac Sim **5.x** rejects it too, but the surface then renders flat black/white: roads,
  sidewalks and terrain lose their texture.

Every such attribute was re-authored as `float2` with the **MDL parameter's own default**
(e.g. `Mortar` 1.0, `Paving_Stones` 0.5, `Rough_Gravel` 2.0), reproducing the 4.5
appearance in 5.x. Attributes that were already a valid `float2` were left untouched.
Nothing else - geometry, layout, textures, physics - was modified.

Tooling and per-scene provenance (every changed layer with its sha256) live in the code
repository: `scripts/upgrade_scene_for_isaacsim5.py`, `scripts/convert_scenes_simready.py`,
`scenes/training_simready_manifest.json`.

## Layout

```
<scene>/                      e.g. Africa_Egypt_Cairo_walk_02_Cousin_01
├── World0.usd                # open this
├── .collect.mapping.json
└── SubUSDs/
    ├── *.usd                 # per-object layers
    ├── materials/*.mdl
    └── textures/
```

Scene names are `<continent>_<country>_<city>_<walk|drive>_<seq>_Cousin_<k>`; cousins share a
source video and differ in layout / assets.

## Usage

```bash
# Isaac Sim 5.x
isaacsim --exec 'omni.usd.get_context().open_stage("/path/to/<scene>/World0.usd")'
```

or with the [`urbanverse-scene`](https://github.com/VAIL-UCLA/UrbanVerse) toolkit / Isaac Lab's
`UsdFileCfg`.

## Citation

```bibtex
@inproceedings{liu2026urbanverse,
  title={UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos},
  author={Mingxuan Liu and Honglin He and Elisa Ricci and Wayne Wu and Bolei Zhou},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
}
```
