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

Not every MDL declares ``texture_scale`` as ``float2``: some of the custom
materials shipped with UrbanVerse scenes declare it as a plain ``float``
(``uv * texture_scale``). For each Shader the script reads the MDL it points
at (``info:mdl:sourceAsset`` + ``subIdentifier``) and matches the declared
type - ``float2`` gets ``Float2``, ``float`` gets ``Float``. Shaders whose
MDL cannot be found or parsed fall back to ``Float2`` (the vMaterials case).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import re

from pxr import Gf, Sdf


_USD_EXTS = {".usd", ".usda", ".usdc", ".usdz"}
_MDL_TYPE_CACHE: dict[tuple[str, str], str | None] = {}


_NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]?"


def _mdl_texture_scale_decl(mdl_path: str, material: str) -> tuple[str, tuple[float, float] | None] | None:
    """(type, default) of ``texture_scale`` in ``export material <material>(...)``.

    type is 'float' or 'float2'; default is (x, y) or None if it could not be
    parsed. Returns None if the file is missing, the material is not found,
    or it has no ``texture_scale`` parameter.
    """
    key = (mdl_path, material)
    if key in _MDL_TYPE_CACHE:
        return _MDL_TYPE_CACHE[key]
    result = None
    try:
        text = Path(mdl_path).read_text(errors="replace")
        m = re.search(r"export\s+material\s+" + re.escape(material) + r"\s*\(", text)
        if m:
            # Walk the parameter list to its matching ')' (defaults like
            # float2(0.5f) and [[ annotations ]] nest parentheses).
            i, depth = m.end(), 1
            while i < len(text) and depth:
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                i += 1
            params = text[m.end():i]
            t = re.search(r"\b(float2|float)\s+texture_scale\b\s*(?:=\s*([^\[,]*?))?\s*(?:\[\[|,|$)",
                          params, re.S)
            if t:
                typ, default_src = t.group(1), (t.group(2) or "").strip()
                # "float2(0.5f)" / "float2(1.f, 2.f)" / bare "1.0" - the digits
                # in the constructor name itself must not be read as a value.
                inner = re.search(r"\bfloat2?\s*\(([^)]*)\)", default_src)
                default_src = inner.group(1) if inner else default_src
                nums = [float(n.rstrip("fF")) for n in re.findall(_NUM, default_src)]
                default = None
                if nums:
                    default = (nums[0], nums[1]) if len(nums) >= 2 else (nums[0], nums[0])
                result = (typ, default)
    except OSError:
        result = None
    _MDL_TYPE_CACHE[key] = result
    return result


def _shader_mdl_decl(prim_spec: Sdf.PrimSpec) -> tuple[str, tuple[float, float] | None] | None:
    src = prim_spec.attributes.get("info:mdl:sourceAsset")
    sub = prim_spec.attributes.get("info:mdl:sourceAsset:subIdentifier")
    if src is None or sub is None or src.default is None or not sub.default:
        return None
    ap = getattr(src.default, "path", None) or str(src.default).strip("@")
    if not ap:
        return None
    mdl_path = prim_spec.layer.ComputeAbsolutePath(ap)
    return _mdl_texture_scale_decl(mdl_path, str(sub.default))


def _shader_target_type(prim_spec: Sdf.PrimSpec) -> str:
    """'float' or 'float2' for this Shader's texture_scale, per its MDL; float2 if unknown."""
    decl = _shader_mdl_decl(prim_spec)
    return decl[0] if decl else "float2"
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
                     scale: float = 1.0, override_value: float | str | None = None,
                     keep_float2: bool = False, mdl_default_fallback: float = 1.0) -> int:
    attr_spec = prim_spec.attributes.get("inputs:texture_scale")
    if attr_spec is None:
        return 0

    val = attr_spec.default
    if val is None:
        return 0

    want = _shader_target_type(prim_spec)
    if override_value == "mdl-default":
        # Isaac Sim 4.5 rejected the scalar too (same UsdToMdl error) and fell
        # back to the MDL parameter default, so that default is what the
        # scenes actually looked like; reproduce it. Unknown -> fallback.
        decl = _shader_mdl_decl(prim_spec)
        if decl and decl[1] is not None:
            override_value = decl[1][0]  # isotropic; anisotropic defaults are rare
        else:
            override_value = mdl_default_fallback

    # Case A: already Float2 — just rewrite the default value in place.
    if attr_spec.typeName == Sdf.ValueTypeNames.Float2 and want == "float2":
        if keep_float2:
            return 0
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

    # Case B: type change needed. Scalar int/float/double/half -> the MDL's
    # declared type; a Float2 authored against a float MDL param -> Float.
    if attr_spec.typeName == Sdf.ValueTypeNames.Float2:
        try:
            fval = float(val[0])
        except (TypeError, IndexError, ValueError):
            return 0
    elif attr_spec.typeName in _SCALAR_TYPES:
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return 0
    else:
        return 0
    if want == "float" and attr_spec.typeName == Sdf.ValueTypeNames.Float:
        # Already the right scalar type; only the value policy applies.
        target = float(override_value) if override_value is not None else fval * scale
        if abs(target - fval) < 1e-9:
            return 0
        print(f"  {attr_spec.path}  Float({fval}) -> Float({target})")
        if not dry_run:
            attr_spec.default = float(target)
        return 1
    if override_value is not None:
        target = float(override_value)
    else:
        target = fval * scale
    if want == "float":
        new_type, new_val, shown = Sdf.ValueTypeNames.Float, float(target), f"Float({target})"
    else:
        new_type, new_val, shown = Sdf.ValueTypeNames.Float2, Gf.Vec2f(target, target), f"Float2({target}, {target})"
    print(f"  {attr_spec.path}  {attr_spec.typeName}({val}) -> {shown}  [mdl: {want}]")

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
        new_type,
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
                 scale: float = 1.0, override_value: float | str | None = None,
                 save_as: Path | None = None,
                 asset_path_rewrites: dict[str, str] | None = None,
                 keep_float2: bool = False, mdl_default_fallback: float = 1.0) -> int:
    print(f"[patching] {layer_path}")
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None:
        print("  (skip, failed to open)")
        return 0
    fixed = 0
    for prim_spec in _all_prim_specs(layer):
        fixed += _patch_prim_spec(prim_spec, dry_run,
                                  scale=scale, override_value=override_value,
                                  keep_float2=keep_float2,
                                  mdl_default_fallback=mdl_default_fallback)

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


def _value_arg(text: str):
    if text == "mdl-default":
        return text
    return float(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("usd", type=Path, help="root .usd/.usda/.usdc file")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    parser.add_argument("--recurse-refs", action="store_true",
                        help="also descend into referenced/payload sublayers")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiplier applied to the original scalar (default 1.0, i.e. keep value)")
    parser.add_argument("--value", type=_value_arg, default=None,
                        help="override: write this value instead of scaling the original. "
                             "'mdl-default' uses each shader's MDL parameter default, which is "
                             "what Isaac Sim 4.5 rendered after rejecting the scalar")
    parser.add_argument("--mdl-default-fallback", type=float, default=1.0,
                        help="value for --value mdl-default when the MDL cannot be read (default 1.0)")
    parser.add_argument("--keep-float2", action="store_true",
                        help="only retype broken scalar attributes; leave attributes that are "
                             "already float2 (i.e. authored correctly) untouched even with --value/--scale")
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
                keep_float2=args.keep_float2,
                mdl_default_fallback=args.mdl_default_fallback,
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
                                  scale=args.scale, override_value=args.value,
                                  keep_float2=args.keep_float2,
                                  mdl_default_fallback=args.mdl_default_fallback)
        verb = "would fix" if args.dry_run else "fixed"
        print(f"\ndone: {verb} {total} texture_scale attribute(s) across "
              f"{len(targets)} layer(s)")


if __name__ == "__main__":
    main()
