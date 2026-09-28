"""What sanity_check_static.py, sanity_check_sim.py and sanity_check_render.py share: which
scenes to check, how an issue is recorded, and how a scene is checked in its own Kit process.

A scene's record is {"issues": [{"level", "check", "msg"}, ...], "stats": {...}}. The two Kit
checks start `<script> --one <job>` per scene: that child checks the scene inside Kit, writes
its record to job["result"] and exits; the parent then adds what Kit's log shows."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PATTERN = "Collected_export_version/export_version.usd"  # a CraftBench scene's root layer
MAX_LOAD = 900.0  # s: FAIL a scene that is still loading after this
STALL = 300.0  # s: kill a Kit process that has logged nothing for this long (hung)
TIMEOUT = 1800.0  # s: kill a Kit process that runs longer than this


# ── scenes ───────────────────────────────────────────────────────────────────────────────────

def parser(doc: str) -> argparse.ArgumentParser:
    """A parser that takes the scenes to check and where to find them."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", nargs="*", help="Scene id (scene_03), name, directory or root .usd. Default: every "
                                            "scene under --root, one by one.")
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
    if twice:  # its log, record and image would overwrite each other
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


# ── issues ───────────────────────────────────────────────────────────────────────────────────

def add(rec: dict, level: str, check: str, msg: str) -> None:
    """Record an issue. FAIL: the scene is broken. ERROR: the check could not finish. WARN: suspicious."""
    rec["issues"].append({"level": level, "check": check, "msg": msg})


def status(rec: dict) -> str:
    """The scene's worst issue level, or 'ok'."""
    levels = {i["level"] for i in rec["issues"]}
    return next((level for level in ("FAIL", "ERROR", "WARN") if level in levels), "ok")


def print_issues(rec: dict, indent: int, info: bool = False) -> None:
    for i in rec["issues"]:
        if info or i["level"] != "INFO":
            print(f"{' ' * indent}{i['level']:5} {i['check']}: {i['msg']}", flush=True)


def summary(tag: str, statuses: list, where: str = "") -> None:
    """Print how many scenes came out how, and exit 1 if any FAILed or had an ERROR."""
    n = Counter(statuses)
    print(f"[{tag}] {n['ok']} ok, {n['WARN']} WARN, {n['ERROR']} ERROR, {n['FAIL']} FAIL of {len(statuses)} "
          f"scene(s)" + (f"; {where}" if where else ""), flush=True)
    if n["FAIL"] or n["ERROR"]:
        raise SystemExit(1)


# ── one Kit process per scene ────────────────────────────────────────────────────────────────

def run_kit(script: str, job: dict, log: Path, tmp: Path) -> dict:
    """Run `script --one <job>`, which checks one scene inside Kit and writes its record to
    job['result'], and return that record with Kit's log judged and the peak RAM added. Kit now
    and then crashes while starting up, so a process that ends without a record gets a second try."""
    result = Path(job["result"])
    result.unlink(missing_ok=True)
    cmd = [sys.executable, script, "--one", json.dumps(job)]
    ended, killed, peak = run_process(cmd, log, tmp)
    second_try = not result.exists() and not killed
    if second_try:
        ended, killed, peak = run_process(cmd, log, tmp)
    if result.exists():
        rec = json.loads(result.read_text())
    else:
        rec = {"issues": [], "stats": {}}
        add(rec, "ERROR", "run", f"Kit process ended ({ended}) without a result")
    if second_try:
        add(rec, "INFO", "run", "Kit ended without a result on the first try; this is the second")
    rec["stats"]["peak_ram_gb"] = round(peak, 1)
    scan_log(log.read_text(errors="replace"), rec)
    return rec


def run_process(cmd: list, log: Path, tmp: Path) -> tuple:
    """Run a Kit process, its output to `log`. It is killed after TIMEOUT seconds, or after STALL
    seconds without output: after a GPU out-of-memory a single app.update() can block forever, so
    Kit cannot time itself out. Returns (how it ended, whether it was killed, peak RAM in GB)."""
    # The .glb importer extracts every model's textures to $TMPDIR/<hash>/textures and only removes
    # them on a clean shutdown, which the child skips: give it a TMPDIR that is deleted afterwards.
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        with open(log, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "TMPDIR": str(tmp)})
            start = last = time.perf_counter()
            size, peak = 0, 0.0
            while (rc := proc.poll()) is None:
                time.sleep(2)
                now = time.perf_counter()
                peak = max(peak, peak_ram_gb(proc.pid))
                if log.stat().st_size != size:
                    size, last = log.stat().st_size, now
                if now - last > STALL or now - start > TIMEOUT:
                    proc.kill()
                    proc.wait()
                    why = f"no output for {now - last:.0f}s" if now - last > STALL else f"after {now - start:.0f}s"
                    return f"killed, {why}", True, peak
        return rc, False, peak
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def peak_ram_gb(pid: int) -> float:
    """Peak resident memory of a running process (the kernel's high-water mark); 0 once it is gone."""
    try:
        with open(f"/proc/{pid}/status") as f:
            return next(int(line.split()[1]) for line in f if line.startswith("VmHWM:")) / 2**20
    except (OSError, StopIteration, ValueError):
        return 0.0


# Kit log lines that mean a scene is not loading as authored.
LOG_FAIL = {
    "usd->mdl type error": re.compile(r"Tried to assign a '\w+'\(USD\) to a '\w+'\(MDL\)"),
    "file format": re.compile(r"Cannot determine file format"),
    "missing file": re.compile(r"Could not open asset|Failed to open layer|Failed to resolve|Unresolved asset|"
                               r"Could not find file|file not found", re.I),
}
# The GPU ran out of memory: a property of this machine, not of the scene. It shows up as
# failed texture uploads and material snapshots, which only count against the scene without it.
LOG_GPU_OOM = re.compile(r"OUT_OF_DEVICE_MEMORY|Out of GPU memory|vkAllocateMemory failed|Unable to allocate buffer")
LOG_TEXTURE = re.compile(r"Texture upload failed|Failed texture loads reported|Failed to create source texture")
# Colliders PhysX rejected, e.g. a mesh without points or contactOffset <= restOffset.
LOG_PHYSX = re.compile(r"\[Error\] \[omni\.(physx|physicsschema)[\w.]*\]")
LOG_ERROR = re.compile(r"\[Error\] \[([\w.]+)\] (.*)")


def scan_log(text: str, rec: dict) -> None:
    """Judge a scene's Kit log, and count its error lines into the record."""
    lines = text.splitlines()

    def matching(rx: re.Pattern) -> list:
        return [line.strip() for line in lines if rx.search(line)]

    for name, rx in LOG_FAIL.items():
        if hits := matching(rx):
            add(rec, "FAIL", "log", f"{len(hits)} '{name}' line(s), e.g. {hits[0][:220]}")
    oom, tex = matching(LOG_GPU_OOM), matching(LOG_TEXTURE)
    if oom:
        add(rec, "WARN", "env", f"GPU ran out of memory ({len(oom)} lines, {len(tex)} texture failures), so what was "
                                f"rendered is incomplete: free the GPU, or lower sanity_check_render.py's --tex-budget")
    elif tex:
        add(rec, "FAIL", "log", f"{len(tex)} texture load failure(s), e.g. {tex[0][:220]}")
    if physx := matching(LOG_PHYSX):
        add(rec, "WARN", "log", f"{len(physx)} PhysX error(s), e.g. {physx[0][:220]}")
    errors = Counter(f"[{m.group(1)}] {m.group(2)[:160]}" for m in LOG_ERROR.finditer(text))
    rec["stats"]["log_errors"] = sum(errors.values())
    rec["log_errors"] = dict(errors.most_common(40))


# ── inside Kit ───────────────────────────────────────────────────────────────────────────────

def wait_for_load(update, ctx, rec: dict) -> None:
    """Call update() until the opened stage has loaded, or FAIL after MAX_LOAD seconds."""
    t0 = said = time.perf_counter()
    while True:
        update()
        _, loaded, total = ctx.get_stage_loading_status()
        now = time.perf_counter()
        if loaded >= total and now - t0 > 10:  # 0/0 once idle; give payload requests time to queue
            break
        if now - t0 > MAX_LOAD:
            add(rec, "FAIL", "load", f"still loading {loaded}/{total} files after {MAX_LOAD:.0f}s")
            break
        if now - said > 10:  # also keeps the log growing, which is how run_process() tells busy from hung
            said = now
            print(f"[sanity-child] loading {loaded}/{total} files, {now - t0:.0f}s", flush=True)
    rec["stats"]["load_s"] = round(time.perf_counter() - t0, 1)


def finish(rec: dict, result: str) -> None:
    """Hand the record to the parent and end the Kit process."""
    Path(result).write_text(json.dumps(rec, indent=1))
    print("[sanity-child] result written", flush=True)
    # Nothing to save, and Kit's shutdown can hang on Linux (seen with texture streaming on:
    # minutes of per-frame texture-upload errors inside app.close()).
    os._exit(0)
