#!/usr/bin/env python3
"""Update the sim-ready conversion progress bars in README.md.

Examples:
    python scripts/update_progress.py --add-assets 500       # converted 500 more assets
    python scripts/update_progress.py --add-scenarios 8      # converted 8 more scenarios
    python scripts/update_progress.py --assets-done 12000    # set absolute count
"""
import argparse
import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"
START = "<!-- sim-ready-progress:start -->"
END = "<!-- sim-ready-progress:end -->"
BAR_WIDTH = 25
DEFAULT_TOTALS = {"Assets": 102_445, "Scenarios": 160}


def bar_line(name, done, total):
    done = max(0, min(done, total))
    filled = round(BAR_WIDTH * done / total) if total else 0
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"**{name}:** `{bar}` {done:,} / {total:,} ({100 * done / total:.1f}%)"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets-done", type=int, help="set absolute converted-asset count")
    p.add_argument("--scenarios-done", type=int, help="set absolute converted-scenario count")
    p.add_argument("--add-assets", type=int, default=0, help="increment converted-asset count")
    p.add_argument("--add-scenarios", type=int, default=0, help="increment converted-scenario count")
    p.add_argument("--assets-total", type=int, help="override total asset count")
    p.add_argument("--scenarios-total", type=int, help="override total scenario count")
    args = p.parse_args()

    text = README.read_text(encoding="utf-8")
    block = re.search(re.escape(START) + r"(.*?)" + re.escape(END), text, re.S)
    if not block:
        raise SystemExit(f"progress markers not found in {README}")

    state = {}
    for name, default_total in DEFAULT_TOTALS.items():
        m = re.search(rf"\*\*{name}:\*\* `[█░]*` ([\d,]+) / ([\d,]+)", block.group(1))
        if m:
            state[name] = [int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))]
        else:
            state[name] = [0, default_total]

    if args.assets_total is not None:
        state["Assets"][1] = args.assets_total
    if args.scenarios_total is not None:
        state["Scenarios"][1] = args.scenarios_total
    if args.assets_done is not None:
        state["Assets"][0] = args.assets_done
    if args.scenarios_done is not None:
        state["Scenarios"][0] = args.scenarios_done
    state["Assets"][0] += args.add_assets
    state["Scenarios"][0] += args.add_scenarios

    lines = "\n\n".join(bar_line(n, d, t) for n, (d, t) in state.items())
    text = text[: block.start()] + f"{START}\n{lines}\n{END}" + text[block.end():]
    README.write_text(text, encoding="utf-8")
    for name, (done, total) in state.items():
        print(bar_line(name, done, total))


if __name__ == "__main__":
    main()
