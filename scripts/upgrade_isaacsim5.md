# Upgrading UrbanVerse scenes for Isaac Sim 5.x

Scenes exported from Isaac Sim 4.5 (and the older UrbanVerse tooling) render
with missing road, sidewalk, and ground textures when opened in Isaac Sim
5.1+. This directory ships three files to fix that in-place or as sibling
copies, for a single scene or a whole batch.

- `upgrade_scene_for_isaacsim5.py` &mdash; the Python patcher (USD-core based)
- `upgrade_scene_for_isaacsim5.sh` &mdash; single-scene wrapper
- `upgrade_scenes_batch.sh` &mdash; walks a root directory and upgrades every
  scene it finds

---

## Why this is needed

The NVIDIA vMaterials MDL library that UrbanVerse scenes reference (e.g.
`vMaterials_2::Ground::Cobblestone_Big_and_Loose`,
`vMaterials_2::Ground::Paving_Stones`, `vMaterials_2::Concrete::Mortar`, ...)
changed signature between Isaac Sim 4.5 and 5.x:

| Isaac Sim | `texture_scale` MDL type | 4.5-era USD authored as |
| --------- | ------------------------ | ----------------------- |
| 4.5       | tolerated `int` / `float` via silent coercion | `int` |
| 5.1+      | strict `float2` &mdash; mismatch rejected     | still `int` |

On 5.1 you will see warnings / errors similar to:

```
[UsdToMdl] Prim '.../Shader' parameter 'texture_scale':
  Reached invalid assignment. Tried to assign a 'int'(USD) to a 'float2'(MDL).
```

When the assignment is rejected, the shader falls back to the parameter's
default (often zero or an enormous tile), so the **road, sidewalk, and
ground look flat / textureless** while everything else renders fine. Assets
that do not use vMaterials (buildings from `.glb`, vehicles, bollards, etc.)
are unaffected.

The Python patcher walks every USD layer in the composition and, for each
`inputs:texture_scale` attribute authored on a `Shader` prim, rewrites the
scalar `int`/`float` default as `Gf.Vec2f(v, v)` with
`SdfValueTypeNames.Float2`. Metadata (`custom`, `connectability`) is
preserved. Already-`float2` attributes are rescaled in place when you pass
`--scale` / `--value`, so tuning tile density across a whole scene is a
one-liner.

## Dependencies

- Python 3.9+ (we test on 3.11)
- `usd-core` from PyPI &mdash; the bash wrappers auto-install it into the
  active env if `from pxr import Usd` fails. **Isaac Sim's bundled Python is
  not used** &mdash; its `pxr` bindings only load inside a running Kit app,
  which makes offline scene surgery awkward.

If you prefer a different interpreter:

```bash
PYTHON=/path/to/python3 bash upgrade_scene_for_isaacsim5.sh ...
```

## Single-scene usage

```bash
bash upgrade_scene_for_isaacsim5.sh \
    "/path/to/scene/Collected_export_version/export_version.usd" \
    --value 1.0
```

Flags:

| flag | effect |
| ---- | ------ |
| `--value V` | Ignore original values; every `texture_scale` becomes `float2(V, V)`. Good for normalizing tile density across a scene (typical: `0.5`&ndash;`2.0` meters). |
| `--scale X` | Multiply every original value by X (default `1.0`, i.e. keep value). Useful to tune an existing `float2` scene. |
| `--suffix S` | Save each modified layer as `<stem><S><ext>` next to the original. Default: `_texture_scaled`. Original files are **not** touched; inter-layer references in the new files are rewritten to point at the suffixed siblings. |
| `--in-place` | Overwrite the original files. |
| `-y`, `--yes` | Skip the interactive confirmation prompt (used by the batch script). |

The script prints a **dry run** first, listing every edit and every
reference it would rewrite. Only after you say `y` does anything hit disk
(unless `--yes` is passed).

### Typical workflow

```bash
# 1. Normalize all texture tile sizes to 1m in sibling files
bash upgrade_scene_for_isaacsim5.sh \
    "/.../CraftBench/scene_06_.../export_version.usd" \
    --value 1.0

# ... opens Isaac Sim 5.1 on the new root ...
#     .../export_version_texture_scaled.usd
# ... tiles look too dense, try looser ...

# 2. Re-tune; rerun on the same scene with a different value
bash upgrade_scene_for_isaacsim5.sh \
    "/.../CraftBench/scene_06_.../export_version.usd" \
    --value 2.0

# 3. Once you've found a good value, overwrite in place
bash upgrade_scene_for_isaacsim5.sh \
    "/.../CraftBench/scene_06_.../export_version.usd" \
    --value 2.0 --in-place
```

Re-runs are idempotent: attributes already at the target value are detected
and skipped. The suffix flow creates new siblings on first run and updates
the existing siblings on subsequent runs (so `--value` tuning is fast).

## Batch usage

```bash
# Dry-run: just list the scenes that would be processed
bash upgrade_scenes_batch.sh \
    "/media/hollis/Extreme SSD/UrbanVerse-Scenes/CraftBench" \
    --dry-run

# Process every scene under the root, normalizing to 1m tiles, writing siblings
bash upgrade_scenes_batch.sh \
    "/media/hollis/Extreme SSD/UrbanVerse-Scenes/CraftBench" \
    --value 1.0

# Resume a partial run: skip scenes whose sibling already exists
bash upgrade_scenes_batch.sh \
    "/media/hollis/Extreme SSD/UrbanVerse-Scenes/CraftBench" \
    --value 1.0 --skip-done

# Custom root-layer filename (default is export_version.usd)
bash upgrade_scenes_batch.sh "/path/to/root" --pattern "scene.usd" --value 1.0
```

Flags:

| flag | effect |
| ---- | ------ |
| `--value V`, `--scale X` | Forwarded to every scene. |
| `--suffix S`, `--in-place` | Forwarded to every scene. |
| `--pattern P` | Root-layer filename to search for (default `export_version.usd`). |
| `--skip-done` | Skip scenes where `<stem><suffix>.usd` already exists beside the source USD. |
| `--dry-run` | List the scenes that would be processed and exit without running the patch. |

Behavior:

- Scenes are discovered with `find … -name <pattern> -print0`, so paths
  containing spaces or special characters are handled safely.
- Each scene runs in its own bash invocation; a failure on one scene does
  not abort the whole batch.
- The final summary prints counts and a list of any failed scenes; the
  script exits non-zero if any scene failed.

## What actually gets changed

The patcher:

1. Opens the root USD layer and walks its composition, collecting every
   USD-format sublayer (`.usd` / `.usda` / `.usdc` / `.usdz`). `.glb`,
   `.fbx`, and texture files are ignored &mdash; they do not carry the
   attributes we need to patch.
2. Identifies layers that directly contain a scalar `inputs:texture_scale`
   (type `int` / `float` / `double` / `half`) or an already-`float2` one.
3. Computes the transitive closure: any layer that references a patched
   layer also needs a suffixed copy (so the references can be rewritten
   to point at the new siblings).
4. For each affected layer:
   - For each patchable attribute, reads the scalar default, removes the
     old spec, and authors a fresh `float2` spec with
     `Gf.Vec2f(target, target)` where `target` = `--value` or
     `original * --scale`.
   - Rewrites every authored `Sdf.Reference` / `Sdf.Payload` / sublayer
     asset path whose target resolves to another affected layer, so the
     string now points at the suffixed sibling.
   - Writes the result either back to the original path (`--in-place`) or
     to `<stem><suffix><ext>` beside it.

Originals are untouched unless you pass `--in-place`.

## Troubleshooting

**"No module named 'pxr'"** &mdash; the bash wrapper handles this
automatically by `pip install usd-core`. If your env has network
restrictions, install it manually:

```bash
pip install usd-core
```

**Textures still look wrong after patching** &mdash; vMaterials'
`texture_scale` is a tile-size hint, not a UV multiplier, so very small
values (`0.001`) or very large values (`1000`) produce unhelpful results.
Typical working ranges:

| material kind | reasonable `--value` |
| ------------- | -------------------- |
| Paving stones, cobblestone, sidewalk | `0.5` &ndash; `2.0` |
| Concrete, asphalt | `1.0` &ndash; `3.0` |
| Grass, dirt | `1.0` &ndash; `5.0` |

If the scene authored per-material anisotropic values (e.g.
`float2(0.2, 1.0)` for a grooved surface), `--value` will flatten them
to isotropic. Use `--scale` instead to preserve the authored ratio.

**"Tried to assign 'int'(USD) to 'float2'(MDL)" still shows after
patching** &mdash; something in the Kit runtime cache is stale. Delete
`~/.cache/ov/Kit` (or the scene's fabric cache) and reload.

**Composition warnings about `.glb` payloads** &mdash; expected and
harmless. The patcher operates at the `Sdf.Layer` level without loading
`Usd.Stage`, so glTF payloads are never evaluated; any such warning you
see in Isaac Sim after the patch is unrelated to this tool.

---

## File layout

```
scripts/
├── upgrade_scene_for_isaacsim5.py    # the Python patcher
├── upgrade_scene_for_isaacsim5.sh    # single-scene wrapper (interactive)
├── upgrade_scenes_batch.sh           # batch wrapper (non-interactive)
└── upgrade_isaacsim5.md              # this file
```
