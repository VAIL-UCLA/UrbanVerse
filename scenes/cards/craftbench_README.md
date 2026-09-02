---
license: cc-by-nc-4.0
language:
- en
viewer: false
pretty_name: UrbanVerse-CraftBench (Sim-Ready)
size_categories:
- n<1K
task_categories:
- robotics
- reinforcement-learning
tags:
- 3d
- Robotics
- PhysicalAI
- EmbodiedAI
- UrbanSimulation
- IsaacSim
- IsaacLab
extra_gated_fields:
  Full Name: text
  Email Address: text
  Country: country
  Institution: text
  Sector of Institution:
    type: select
    options:
    - Academic/Education
    - Corporation
    - Startup
    - Government
    - Non-profit Organization
    - Individual
    - Other
  Purpose:
    type: select
    options:
    - Embodied AI
    - Physical AI
    - 3D Generation
    - Reinforcement Learning
    - Imitation Learning
    - Computer Vision
    - Autonomous Driving
    - Generative Models
    - Multimodal Large Language Models
    - Visual Question Answering
  I accept the conditions and licenses of the files contained in this dataset: checkbox
---

# UrbanVerse-CraftBench Scenes (Sim-Ready)

[![Project Page](https://img.shields.io/badge/Project-Page-3c78d8?style=flat-square)](https://urbanverseproject.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat-square)](https://arxiv.org/abs/2510.15018)
[![Code](https://img.shields.io/badge/GitHub-Code-181717?style=flat-square)](https://github.com/VAIL-UCLA/UrbanVerse)
[![CraftBench](https://img.shields.io/badge/CraftBench-Original-ff9900?style=flat-square)](https://huggingface.co/datasets/Oatmealliu/UrbanVerse-CraftBench)

The 12 hand-crafted [UrbanVerse-CraftBench](https://huggingface.co/datasets/Oatmealliu/UrbanVerse-CraftBench)
test scenes, upgraded to load correctly in **NVIDIA Isaac Sim ≥ 5**. Same scenes, same
packaging (`Collected_export_version.tar` per scene, open `Collected_export_version/export_version.usd`),
same license as the original.

![before/after](https://raw.githubusercontent.com/VAIL-UCLA/UrbanVerse/main/material/simready_scene03_before_after.png)

## What changed, and why

The scenes were exported from Isaac Sim 4.5 with road / sidewalk vMaterials whose
`inputs:texture_scale` was authored as an `int` while the MDL declares `float2`.

- Isaac Sim **4.5** rejected that value (`[UsdToMdl] Tried to assign a 'int'(USD) to a
  'float2'(MDL)`) and silently fell back to the material's default tile size - that is what
  the `preview_*.png` images show.
- Isaac Sim **5.x** rejects it too, but the surface then renders flat black/white.

Every such attribute was re-authored as `float2` with the **MDL parameter's own default**
(`Mortar` 1.0, `Paving_Stones` 0.5, `Rough_Gravel` 2.0, ...), reproducing the 4.5 appearance
in 5.x. Attributes that were already a valid `float2` were left untouched; geometry, layout,
assets, textures and physics are unchanged.

Tooling and per-scene provenance (every changed layer with its sha256):
[`scripts/upgrade_isaacsim5.md`](https://github.com/VAIL-UCLA/UrbanVerse/blob/main/scripts/upgrade_isaacsim5.md),
[`scenes/craftbench_simready_manifest.json`](https://github.com/VAIL-UCLA/UrbanVerse/blob/main/scenes/craftbench_simready_manifest.json).

## Layout

```
scene_<id>_<place_type>_<topology>_<attributes>/
├── Collected_export_version.tar   # extract, then open Collected_export_version/export_version.usd
├── cam0_to_world.txt              # per-frame camera-to-world matrices of the preview flythrough
├── preview_front.png / preview_topdown.png / preview_closeup.png
└── preview_video.mp4
```

## Usage

```python
import urbanverse_scene as uvs          # pip install urbanverse-scene
uvs.craftbench.load_scenes(uvs.craftbench.get_descriptions()[:1])
```

or extract a tar and open `export_version.usd` in Isaac Sim 5.x.

## Citation

```bibtex
@inproceedings{liu2026urbanverse,
  title={UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos},
  author={Mingxuan Liu and Honglin He and Elisa Ricci and Wayne Wu and Bolei Zhou},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
}
```
