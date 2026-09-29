"""What the three scripts share: which scenes to work on, and where their files go."""
import argparse
import json
from pathlib import Path

PATTERN = "Collected_export_version/export_version.usd"  # a CraftBench scene's root layer
HERE = Path(__file__).resolve().parents[1]
MAPS = HERE / "maps"  # built by render_topdown.py
ROUTES = HERE / "routes"  # annotations and generated routes


def parser(doc: str) -> argparse.ArgumentParser:
    """A parser that takes the scenes to work on and where to find them."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", nargs="*", help="Scene id (scene_03), name, directory or root .usd. Default: every "
                                            "scene under --root.")
    p.add_argument("--root", help="Directory with one sub-directory per scene. Default: the urbanverse-scene "
                                  "toolkit's CraftBench folder.")
    p.add_argument("--pattern", default=PATTERN, help=f"Root layer relative to each scene dir (default {PATTERN}).")
    return p


def default_root() -> Path:
    """<cache>/CraftBench of the urbanverse-scene toolkit: where uvs.set() pointed it, else its default."""
    try:
        cache = Path(json.loads((Path.home() / ".cache" / "urbanverse_scenes_config.json").read_text())["cache_dir"])
    except (OSError, ValueError, KeyError):
        cache = Path.home() / ".cache" / "urbanverse_scenes"
    return cache / "CraftBench"


def find_scenes(a: argparse.Namespace) -> list:
    """(name, scene dir, root layer) of each scene given, or of every scene under --root if none is."""
    root = Path(a.root).expanduser().resolve() if a.root else default_root()
    known = sorted(d for d in root.iterdir() if (d / a.pattern).is_file()) if root.is_dir() else []
    if not a.scene:
        if not known:
            raise SystemExit(f"no scene under {root} has {a.pattern}: pass --root, or a scene's path")
        return [(d.name, d, d / a.pattern) for d in known]
    scenes = [find_scene(want, known, root, a.pattern) for want in a.scene]
    names = [name for name, _, _ in scenes]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise SystemExit(f"{', '.join(twice)}: given more than once (a scene is named after its directory)")
    return scenes


def find_scene(want: str, known: list, root: Path, pattern: str) -> tuple:
    path = Path(want).expanduser()
    if path.is_file():  # a root layer: the scene dir is where the pattern starts, if it matches
        usd = path.resolve()
        d = usd.parents[len(Path(pattern).parts) - 1] if usd.as_posix().endswith("/" + pattern) else usd.parent
        return d.name, d, usd
    if path.is_dir():
        if not (path / pattern).is_file():
            raise SystemExit(f"{path} has no {pattern}: pass the root .usd itself, or --pattern")
        d = path.resolve()
        return d.name, d, d / pattern
    hits = [d for d in known if d.name == want or d.name.startswith(want + "_")]
    if len(hits) != 1:
        raise SystemExit(f"{want!r} is not a path and matches {len(hits)} scenes under {root}")
    return hits[0].name, hits[0], hits[0] / pattern


def find_maps(names: list, maps: Path) -> list:
    """The map folders of the scenes named (by id or name), or all of them if none is named."""
    known = sorted(d for d in maps.iterdir() if (d / "map.json").is_file()) if maps.is_dir() else []
    if not known:
        raise SystemExit(f"no maps in {maps}: build them with render_topdown.py")
    if not names:
        return known
    found = []
    for want in names:
        hits = [d for d in known if d.name == want or d.name.startswith(want + "_")]
        if len(hits) != 1:
            raise SystemExit(f"{want!r} matches {len(hits)} maps in {maps}")
        found += hits
    return found
