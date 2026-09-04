---
license: odc-by
task_categories:
  - robotics
tags:
  - urbanverse
  - isaac-sim
  - isaac-lab
  - usd
  - simulation
  - 3d-assets
pretty_name: UrbanVerse Assets (Sim-Ready)
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

# UrbanVerse-Assets-Sim-ready

Simulation-ready USD conversions of the [UrbanVerse-100K](https://huggingface.co/datasets/Oatmealliu/UrbanVerse-100K)
3D asset database, for **NVIDIA Isaac Sim ≥ 5** / **Isaac Lab**.

Project: [UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos](https://urbanverseproject.github.io/) (ICLR 2026) ·
Code: [VAIL-UCLA/UrbanVerse](https://github.com/VAIL-UCLA/UrbanVerse)

## What "sim-ready" means here

Each asset was converted from its metric, ground-aligned GLB
(`adjusted_asset_scaled_bottomed.glb`) with Isaac Lab's `MeshConverter` and validated
against the source mesh:

| property | value |
| --- | --- |
| up axis | **Z** (glTF's Y-up geometry is rotated +90° about X, so the asset stands upright at identity) |
| units | metres, `metersPerUnit = 1` |
| origin | base of the asset on `z = 0` |
| physics | `RigidBodyAPI` (kinematic), `CollisionAPI`, `MeshCollisionAPI` = `convexDecomposition` |
| materials | UsdPreviewSurface + textures under `textures/` |

Every stage's world bounds were checked against the exact vertex bounds of its GLB
after the axis swap (extents equal to 1e-3, base at z = 0).

## Layout

One tar per asset, bucketed by the first two hex characters of its id (256 folders,
~400 files each):

```
usd/
└── <uid[:2]>/
    └── std_<uid>.tar
        └── std_<uid>/
            ├── std_<uid>.usd      # reference this
            ├── config.yaml        # MeshConverterCfg used for the conversion
            └── textures/
```

`<uid>` is the UrbanVerse-100K asset id, so annotations from the
[`urbanverse-asset`](https://github.com/VAIL-UCLA/UrbanVerse/tree/main/urbanverse_100k)
toolkit apply directly. Extracting every tar into one directory gives a flat
`std_<uid>/` folder per asset.

## Usage

Fetch and unpack one asset:

```python
import tarfile
from huggingface_hub import hf_hub_download

uid = "0011e400ad1042cfb6a990376240b091"
tar = hf_hub_download("UCLA-VAIL/UrbanVerse-Assets-Sim-ready", f"usd/{uid[:2]}/std_{uid}.tar",
                      repo_type="dataset")
tarfile.open(tar).extractall("assets")          # -> assets/std_<uid>/std_<uid>.usd
```

Everything (~1.7 TB):

```bash
hf download UCLA-VAIL/UrbanVerse-Assets-Sim-ready --repo-type dataset --local-dir assets_tar
find assets_tar/usd -name '*.tar' -print0 | xargs -0 -n1 -P8 tar -C assets -xf
```

Then, in Isaac Lab:

```python
import isaaclab.sim as sim_utils

cfg = sim_utils.UsdFileCfg(usd_path="assets/std_<uid>/std_<uid>.usd")
cfg.func("/World/asset", cfg, translation=(0.0, 0.0, 0.0))
```

## Status

This repository is filled in batches as the full corpus (102,445 assets) is converted;
see the [live progress](https://github.com/VAIL-UCLA/UrbanVerse#sim-ready-conversion-progress)
in the project README. Conversion tooling: `scripts/profile_glb_to_usd.py` and
`urbanverse_100k/urbanverse_asset/_glb_to_usd.py` in the code repository.

## Citation

```bibtex
@inproceedings{liu2026urbanverse,
  title={UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos},
  author={Mingxuan Liu and Honglin He and Elisa Ricci and Wayne Wu and Bolei Zhou},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
}
```
