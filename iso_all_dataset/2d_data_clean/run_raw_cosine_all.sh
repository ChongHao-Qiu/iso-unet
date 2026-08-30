#!/usr/bin/env bash
# Run raw-cosine analog maps over a list of baseline checkpoints.
#
# Usage:
#   ./run_raw_cosine_all.sh ckpt1.pt ckpt2.pt ...
#   ./run_raw_cosine_all.sh $(find /path/to/experiments -name model.pt)
#
# Each ckpt's output goes to <ckpt_dir>/raw_cosine/.
#
# Knobs you may want to tweak:
DATASET="${DATASET:-420ppm}"
SAMPLE_INDICES="${SAMPLE_INDICES:-100}"
REGIONS="${REGIONS:-all}"
OUT_SUBDIR="${OUT_SUBDIR:-raw_cosine}"
PY="${PY:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRAW_FEATURE="${SCRIPT_DIR}/draw_feature.py"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <ckpt1.pt> [ckpt2.pt ...]"
    echo
    echo "Env vars (defaults shown):"
    echo "  DATASET=$DATASET    SAMPLE_INDICES=$SAMPLE_INDICES"
    echo "  REGIONS=$REGIONS    OUT_SUBDIR=$OUT_SUBDIR"
    echo
    echo "Example:"
    echo "  $0 \$(find /home/.../experiments -name model.pt)"
    exit 1
fi

n_ok=0
n_fail=0
declare -a failed=()

for ckpt in "$@"; do
    [ -f "$ckpt" ] || { echo "[skip] not a file: $ckpt"; continue; }
    name="$(basename "$(dirname "$ckpt")")"
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "  $name"
    echo "  $ckpt"
    echo "════════════════════════════════════════════════════════════════"
    if "$PY" "$DRAW_FEATURE" \
            --ckpt "$ckpt" \
            --dataset "$DATASET" \
            --sample_indices "$SAMPLE_INDICES" \
            --regions "$REGIONS" \
            --no_center --no_zonal_demean --sharpen 1 \
            --cmap RdBu_r --vmin -1 --vmax 1 \
            --out_subdir "$OUT_SUBDIR" 2>&1 \
            | tail -25; then
        n_ok=$((n_ok+1))
    else
        n_fail=$((n_fail+1))
        failed+=("$name")
    fi
done

echo
echo "════════════════════════════════════════════════════════════════"
echo "  Done: $n_ok ok, $n_fail failed"
if [ $n_fail -gt 0 ]; then
    echo "  Failed: ${failed[*]}"
fi
echo "════════════════════════════════════════════════════════════════"
