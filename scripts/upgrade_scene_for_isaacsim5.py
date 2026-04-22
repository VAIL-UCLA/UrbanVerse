#!/usr/bin/env python3
"""Convert USD Shader inputs:texture_scale from int/float -> float2.

Works at the Sdf layer level (not Usd.Stage), so it avoids loading
unresolvable payloads (e.g., .glb assets) and bypasses UsdShade's
``CreateInput`` helper which reuses the existing attribute's typeName.

Usage:
    python fix_texture_scale.py <path/to/scene.usd> [--dry-run] [--recurse-refs]

In Isaac Sim 5.1 the vMaterials MDL signature declares ``texture_scale`` as
``float2``; Isaac Sim 4.5 tolerated ``int``/``float`` via silent coercion.
Scenes exported with older tooling therefore render without texture tiling
on 5.1 and log:

    [UsdToMdl] ... 'texture_scale': Tried to assign 'int'(USD) to 'float2'(MDL).

This script rewrites every scalar ``inputs:texture_scale`` attribute spec
as ``Gf.Vec2f(v, v)`` with ``SdfValueTypeNames.Float2`` in the layer where
the opinion lives, preserving custom + connectability metadata.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pxr import Gf, Sdf


_USD_EXTS = {".usd", ".usda", ".usdc", ".usdz"}
_SCALAR_TYPES = {
    Sdf.ValueTypeNames.Int,
    Sdf.ValueTypeNames.Float,
    Sdf.ValueTypeNames.Double,
    Sdf.ValueTypeNames.Half,
}


def _all_prim_specs(layer: Sdf.Layer):
    """Yield every prim spec in the layer (depth-first)."""
    stack = list(layer.rootPrims)
    while stack:
        p = stack.pop()
        yield p
        stack.extend(p.nameChildren)


def _patch_prim_spec(prim_spec: Sdf.PrimSpec, dry_run: bool,
                     scale: float = 1.0, override_value: float | None = None) -> int:
    attr_spec = prim_spec.attributes.get("inputs:texture_scale")
    if attr_spec is None:
        return 0

    val = attr_spec.default
    if val is None:
        return 0

    # Case A: already Float2 — just rewrite the default value in place.
    if attr_spec.typeName == Sdf.ValueTypeNames.Float2:
        try:
            vx = float(val[0])
            vy = float(val[1])
        except (TypeError, IndexError):
            # Malformed: typeName is float2 but default was authored as a scalar.
            try:
                vx = vy = float(val)
            except (TypeError, ValueError):
                print(f"  {attr_spec.path}  skipped (unreadable Float2 default: {val!r})")
                return 0
        if override_value is not None:
            target_x = target_y = float(override_value)
        else:
            target_x, target_y = vx * scale, vy * scale
        if abs(target_x - vx) < 1e-9 and abs(target_y - vy) < 1e-9:
            return 0
        print(f"  {attr_spec.path}  Float2({vx}, {vy}) "
              f"-> Float2({target_x}, {target_y})")
        if not dry_run:
            attr_spec.default = Gf.Vec2f(target_x, target_y)
        return 1

    # Case B: scalar int/float/double/half — requires type change.
    if attr_spec.typeName not in _SCALAR_TYPES:
        return 0
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return 0
    if override_value is not None:
        target = float(override_value)
    else:
        target = fval * scale
    new_val = Gf.Vec2f(target, target)
    print(f"  {attr_spec.path}  {attr_spec.typeName}({val}) "
          f"-> Float2({target}, {target})")

    if dry_run:
        return 1

    custom = attr_spec.custom
    conn = None
    try:
        if attr_spec.HasInfo("connectability"):
            conn = attr_spec.GetInfo("connectability")
    except Exception:
        conn = None

    # Delete the scalar spec so we can re-author with a new typeName.
    # ``prim_spec.attributes`` is a read-only view — use RemoveProperty.
    prim_spec.RemoveProperty(attr_spec)
    new_attr = Sdf.AttributeSpec(
        prim_spec,
        "inputs:texture_scale",
        Sdf.ValueTypeNames.Float2,
        variability=Sdf.VariabilityVarying,
    )
    new_attr.default = new_val
    new_attr.custom = custom
    if conn is not None:
        try:
            new_attr.SetInfo("connectability", conn)
        except Exception:
            pass
    return 1


def _patch_layer(layer_path: Path, dry_run: bool,
                 scale: float = 1.0, override_value: float | None = None,
                 save_as: Path | None = None,
                 asset_path_rewrites: dict[str, str] | None = None) -> int:
    print(f"[patching] {layer_path}")
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None:
        print("  (skip, failed to open)")
        return 0
    fixed = 0
    for prim_spec in _all_prim_specs(layer):
        fixed += _patch_prim_spec(prim_spec, dry_run,
                                  scale=scale, override_value=override_value)

    rewrote = 0
    if asset_path_rewrites:
        for old, new in asset_path_rewrites.items():
            if layer.UpdateExternalReference(old, new):
                rewrote += 1
                print(f"  ref: {old} -> {new}")

    if not dry_run:
        if save_as is not None:
            layer.Export(str(save_as))
            print(f"  wrote {save_as}")
        elif fixed or rewrote:
            layer.Save()

    print(f"  fixed {fixed} attribute(s), rewrote {rewrote} reference(s)")
    return fixed


def _layer_has_patchable_attr(layer_path: Path) -> bool:
    """True if the layer has any scalar or Float2 inputs:texture_scale authored."""
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None:
        return False
    for prim_spec in _all_prim_specs(layer):
        attr = prim_spec.attributes.get("inputs:texture_scale")
        if attr is None:
            continue
        tn = attr.typeName
        if tn in _SCALAR_TYPES or tn == Sdf.ValueTypeNames.Float2:
            if attr.default is not None:
                return True
    return False


def _layer_authored_deps(layer_path: Path):
    """Yield (authored_path, resolved_abs_path) for each composition dep in layer."""
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None:
        return
    for sub in layer.subLayerPaths:
        yield (sub, layer.ComputeAbsolutePath(sub))
    for prim_spec in _all_prim_specs(layer):
        for items in (prim_spec.referenceList.prependedItems,
                      prim_spec.referenceList.appendedItems,
                      prim_spec.referenceList.addedItems,
                      prim_spec.referenceList.orderedItems,
                      prim_spec.payloadList.prependedItems,
                      prim_spec.payloadList.appendedItems,
                      prim_spec.payloadList.addedItems,
                      prim_spec.payloadList.orderedItems):
            for item in items:
                ap = getattr(item, "assetPath", None)
                if ap:
                    yield (ap, layer.ComputeAbsolutePath(ap))


def _suffixed_name(p: Path, suffix: str) -> Path:
    """/a/b/foo.usd  +  '_scaled' -> /a/b/foo_scaled.usd"""
    return p.with_name(p.stem + suffix + p.suffix)


def _gather_sublayers(root: Path) -> list[Path]:
    """All USD files reachable via sublayers/references/payloads."""
    seen: set[Path] = set()
    stack: list[Path] = [root.resolve()]
    result: list[Path] = []
    while stack:
        p = stack.pop()
        if p in seen or not p.is_file():
            continue
        if p.suffix.lower() not in _USD_EXTS:
            continue
        seen.add(p)
        result.append(p)
        try:
            layer = Sdf.Layer.FindOrOpen(str(p))
        except Exception as e:
            print(f"  (skip {p.name}: {e})")
            continue
        if layer is None:
            continue
        for ext_id in layer.GetCompositionAssetDependencies():
            try:
                sub_str = layer.ComputeAbsolutePath(ext_id)
            except Exception:
                continue
            sub = Path(sub_str)
            if sub.is_file() and sub.suffix.lower() in _USD_EXTS:
                stack.append(sub.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("usd", type=Path, help="root .usd/.usda/.usdc file")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    parser.add_argument("--recurse-refs", action="store_true",
                        help="also descend into referenced/payload sublayers")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiplier applied to the original scalar (default 1.0, i.e. keep value)")
    parser.add_argument("--value", type=float, default=None,
                        help="override: always write this value instead of scaling the original")
    parser.add_argument("--suffix", type=str, default=None,
                        help="save each modified layer to a sibling file with this suffix on its "
                             "stem (e.g. '_texture_scaled' -> 'foo_texture_scaled.usd'). Leaves "
                             "originals untouched and rewrites inter-layer references in the new files.")
    args = parser.parse_args()

    if not args.usd.is_file():
        sys.exit(f"error: {args.usd} not found")

    targets = [args.usd.resolve()]
    if args.recurse_refs:
        targets = _gather_sublayers(args.usd)
        print(f"discovered {len(targets)} layer(s) in composition")

    if args.suffix:
        # Save-as mode: figure out which layers are "affected" — either they directly
        # carry patchable attrs, or they transitively reference such a layer. Those
        # need suffixed copies with their refs rewritten.
        directly = {p for p in targets if _layer_has_patchable_attr(p)}
        affected = set(directly)
        # Transitive closure: any layer whose authored dep resolves into affected.
        changed = True
        while changed:
            changed = False
            for p in targets:
                if p in affected:
                    continue
                for _authored, resolved in _layer_authored_deps(p):
                    try:
                        if Path(resolved).resolve() in affected:
                            affected.add(p)
                            changed = True
                            break
                    except Exception:
                        continue

        print(f"affected layers (will be saved with suffix '{args.suffix}'): "
              f"{len(affected)}")
        for p in sorted(affected):
            print(f"  {p}")

        # Build per-layer rewrite maps: rewrite every authored asset path that
        # resolves to another affected layer -> suffixed authored path.
        def _rewrite_map_for(layer_path: Path) -> dict[str, str]:
            rewrites = {}
            for authored, resolved in _layer_authored_deps(layer_path):
                try:
                    if Path(resolved).resolve() in affected:
                        rewrites[authored] = _suffixed_name(
                            Path(authored), args.suffix).as_posix()
                except Exception:
                    continue
            return rewrites

        total = 0
        for p in sorted(affected):
            save_as = _suffixed_name(p, args.suffix)
            rewrites = _rewrite_map_for(p)
            total += _patch_layer(
                p, dry_run=args.dry_run,
                scale=args.scale, override_value=args.value,
                save_as=save_as, asset_path_rewrites=rewrites,
            )

        root_new = _suffixed_name(args.usd.resolve(), args.suffix)
        verb = "would fix" if args.dry_run else "fixed"
        print(f"\ndone: {verb} {total} attribute(s) across {len(affected)} "
              f"layer(s). Load this for debugging:")
        print(f"  {root_new}")
    else:
        total = 0
        for t in targets:
            total += _patch_layer(t, dry_run=args.dry_run,
                                  scale=args.scale, override_value=args.value)
        verb = "would fix" if args.dry_run else "fixed"
        print(f"\ndone: {verb} {total} texture_scale attribute(s) across "
              f"{len(targets)} layer(s)")


if __name__ == "__main__":
    main()
