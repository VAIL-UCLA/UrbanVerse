#!/usr/bin/env bash
# Batch-upgrade every UrbanVerse scene under a root directory. Finds each
# ``export_version.usd`` below the root and runs the per-scene upgrade
# non-interactively, collecting pass/fail counts at the end.
#
# Usage:
#   bash upgrade_scenes_batch.sh <root_dir> [--value V | --scale X]
#                                [--suffix S | --in-place] [--skip-done]
#                                [--pattern P] [--dry-run]
#
# Example:
#   bash upgrade_scenes_batch.sh "/media/hollis/Extreme SSD/UrbanVerse-Scenes/CraftBench" --value 1.0

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_SH="${SCRIPT_DIR}/upgrade_scene_for_isaacsim5.sh"

if [[ ! -x "${SINGLE_SH}" && ! -f "${SINGLE_SH}" ]]; then
    echo "error: ${SINGLE_SH} not found (expected alongside this .sh)" >&2
    exit 1
fi

ROOT_DIR=""
PATTERN="export_version.usd"
SUFFIX="_texture_scaled"
SKIP_DONE=0
DRY_RUN=0
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale|--value)
            FORWARD_ARGS+=("$1" "$2"); shift 2 ;;
        --suffix)
            SUFFIX="$2"; shift 2 ;;
        --in-place)
            SUFFIX=""; shift ;;
        --pattern)
            PATTERN="$2"; shift 2 ;;
        --skip-done)
            SKIP_DONE=1; shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        -h|--help)
            cat <<EOF
usage: $0 <root_dir> [options]

  positional:
    <root_dir>      directory to scan recursively for USD files

  options:
    --value V       pass value to the per-scene upgrade (tile size)
    --scale X       pass scale multiplier
    --suffix S      sibling-file suffix (default '_texture_scaled')
    --in-place      overwrite originals
    --pattern P     filename to look for under root_dir (default 'export_version.usd')
    --skip-done     skip scenes whose sibling <stem><suffix>.usd already exists
    --dry-run       list the scenes that would be processed, then exit

example:
    bash $0 "/media/hollis/Extreme SSD/UrbanVerse-Scenes/CraftBench" --value 1.0 --skip-done
EOF
            exit 0 ;;
        *)
            if [[ -z "${ROOT_DIR}" ]]; then ROOT_DIR="$1"; shift
            else echo "unknown positional: $1" >&2; exit 1; fi ;;
    esac
done

if [[ -z "${ROOT_DIR}" ]]; then
    echo "error: root_dir is required. See --help." >&2
    exit 1
fi
if [[ ! -d "${ROOT_DIR}" ]]; then
    echo "error: root_dir not a directory: ${ROOT_DIR}" >&2
    exit 1
fi

if [[ -n "${SUFFIX}" ]]; then
    FORWARD_ARGS+=("--suffix" "${SUFFIX}")
fi

echo "[info] root: ${ROOT_DIR}"
echo "[info] pattern: ${PATTERN}"
[[ -n "${SUFFIX}" ]] && echo "[info] suffix: ${SUFFIX}"
[[ "${SKIP_DONE}" == "1" ]] && echo "[info] --skip-done enabled"

# Collect scenes
mapfile -d '' SCENES < <(find "${ROOT_DIR}" -type f -name "${PATTERN}" -print0 | sort -z)

if [[ ${#SCENES[@]} -eq 0 ]]; then
    echo "no scenes found matching '${PATTERN}' under ${ROOT_DIR}"
    exit 0
fi

echo "[info] discovered ${#SCENES[@]} scene(s)"

# Optionally filter out already-processed scenes
TODO=()
for usd in "${SCENES[@]}"; do
    if [[ "${SKIP_DONE}" == "1" && -n "${SUFFIX}" ]]; then
        dir="$(dirname "${usd}")"
        stem="$(basename "${usd}" .usd)"
        sibling="${dir}/${stem}${SUFFIX}.usd"
        if [[ -f "${sibling}" ]]; then
            echo "  [skip] ${usd}  (sibling exists)"
            continue
        fi
    fi
    TODO+=("${usd}")
done

echo "[info] ${#TODO[@]} scene(s) to process"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
    for usd in "${TODO[@]}"; do echo "  ${usd}"; done
    exit 0
fi

PASS=0
FAIL=0
FAILED_SCENES=()
start_time=$(date +%s)

for i in "${!TODO[@]}"; do
    usd="${TODO[$i]}"
    n=$((i + 1))
    total=${#TODO[@]}
    echo
    echo "============================================================"
    echo "[${n}/${total}] ${usd}"
    echo "============================================================"
    if bash "${SINGLE_SH}" "${usd}" --yes "${FORWARD_ARGS[@]}"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED_SCENES+=("${usd}")
        echo "  [fail] ${usd}"
    fi
done

elapsed=$(( $(date +%s) - start_time ))
echo
echo "============================================================"
echo "batch done in ${elapsed}s:  ${PASS} ok, ${FAIL} failed (of ${#TODO[@]})"
echo "============================================================"
if (( FAIL > 0 )); then
    echo "failed scenes:"
    for f in "${FAILED_SCENES[@]}"; do echo "  ${f}"; done
    exit 1
fi
