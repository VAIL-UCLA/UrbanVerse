#!/usr/bin/env bash
# Upgrade a single UrbanVerse scene so it renders correctly in Isaac Sim 5.x
# by rewriting ``inputs:texture_scale`` on shader prims from scalar int/float
# to the ``float2`` that vMaterials now require. Wraps
# ``upgrade_scene_for_isaacsim5.py``.
#
# By default every modified layer is saved as a sibling with the suffix
# ``_texture_scaled`` on its stem, leaving the original files untouched.
#
# Usage:
#   bash upgrade_scene_for_isaacsim5.sh <path/to/export_version.usd> \
#       [--scale X | --value V] [--suffix S | --in-place] [--yes]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_FIX="${SCRIPT_DIR}/upgrade_scene_for_isaacsim5.py"

if [[ ! -f "${PY_FIX}" ]]; then
    echo "error: ${PY_FIX} not found (expected alongside this .sh)" >&2
    exit 1
fi

USD_PATH=""
SUFFIX="_texture_scaled"
ASSUME_YES=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale|--value)
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --suffix)
            SUFFIX="$2"; shift 2 ;;
        --in-place)
            SUFFIX=""; shift ;;
        -y|--yes)
            ASSUME_YES=1; shift ;;
        -h|--help)
            cat <<EOF
usage: $0 <path/to/export_version.usd> [options]

  positional:
    <path/to/export_version.usd>  root USD layer of the scene

  options:
    --scale X     multiply every original value by X (default 1)
    --value V     ignore originals; write float2(V, V) everywhere
    --suffix S    sibling-file suffix on each modified layer's stem
                  (default '_texture_scaled')
    --in-place    overwrite originals instead of creating siblings
    -y, --yes     skip the interactive confirmation prompt (non-interactive mode)

example:
    bash $0 /path/to/scene/Collected_export_version/export_version.usd --value 1.0 -y
EOF
            exit 0 ;;
        *)
            if [[ -z "${USD_PATH}" ]]; then USD_PATH="$1"; shift
            else echo "unknown positional: $1" >&2; exit 1; fi ;;
    esac
done

if [[ -z "${USD_PATH}" ]]; then
    echo "error: USD path is required. See --help." >&2
    exit 1
fi
if [[ ! -f "${USD_PATH}" ]]; then
    echo "error: USD not found: ${USD_PATH}" >&2
    exit 1
fi

if [[ -n "${SUFFIX}" ]]; then
    EXTRA_ARGS+=("--suffix" "${SUFFIX}")
fi

PY="${PYTHON:-python}"

echo "[info] python: $(command -v "${PY}")  $(${PY} --version 2>&1)"
echo "[info] USD scene: ${USD_PATH}"
echo

if ! "${PY}" -c "from pxr import Usd" >/dev/null 2>&1; then
    echo "==> installing usd-core (standalone USD Python bindings)"
    "${PY}" -m pip install --quiet usd-core
    if ! "${PY}" -c "from pxr import Usd" >/dev/null 2>&1; then
        echo "error: pxr still not importable after installing usd-core." >&2
        exit 1
    fi
fi

echo "==> dry run"
"${PY}" "${PY_FIX}" "${USD_PATH}" --recurse-refs --dry-run "${EXTRA_ARGS[@]}"

if [[ "${ASSUME_YES}" != "1" ]]; then
    echo
    read -r -p "Apply these edits? [y/N] " reply
    case "${reply}" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "aborted."; exit 0 ;;
    esac
fi

echo
echo "==> applying patch"
"${PY}" "${PY_FIX}" "${USD_PATH}" --recurse-refs "${EXTRA_ARGS[@]}"

echo
echo "[done] re-run your Isaac Sim 5.1 script on the new root USD above."
