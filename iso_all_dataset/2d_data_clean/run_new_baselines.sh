#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Train + eval the 5 newly added community baselines.
#
# Baselines:
#   AttU-Net       (Image_Segmentation repo)
#   DCSAU-Net      (DCSAU-Net repo)
#   TransUNet      (TransUNet repo)
#   UNO            (UNO repo)
#   UNet++         (UNetPlusPlus repo)
#
# OOD preset:  default  (ID 280/350/420/560, OOD 400/490/840)
# Output dir:  ./experiments
#
# Usage:
#   bash run_new_baselines.sh                # run all (skipping existing ckpts), seed=42
#   bash run_new_baselines.sh attunet_full_split   # run just one
#   FORCE=1 bash run_new_baselines.sh        # don't skip; retrain everything
#   SEED=123 bash run_new_baselines.sh       # seed=123 → adds -s123 suffix to directory
#   TRAIN_FRAC=<frac> bash run_new_baselines.sh  # sets the training split -> tf<NN> exp_tag suffix
#   SAVE_BASE=/path/to/dir bash run_new_baselines.sh   # custom output directory
#   combined: TRAIN_FRAC=<frac> SAVE_BASE=.../experiments_baselines SEED=40 bash ...
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config (all overridable via env vars) ──────────────────────────────
PY=python
WORK_DIR=.
CFG_DIR="$WORK_DIR/configs"
SAVE_BASE="${SAVE_BASE:-./experiments}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
# TF_TAG matches train.py's exp_tag computation: f'tf{int(round(train_frac*100)):02d}'
TF_TAG=$($PY -c "print(f'tf{int(round($TRAIN_FRAC*100)):02d}')")
SEED="${SEED:-42}"                         # default seed 42 (matches train.py's default)
# Non-default seed → adds -s{SEED} suffix to directory (matches train.py's auto-suffix)
if [[ "$SEED" == "42" ]]; then
  SEED_TAG=""
else
  SEED_TAG="-s${SEED}"
fi

MODELS_ALL=(
  attunet_full_split
  dcsaunet_full_split
  transunet_full_split
  uno_full_split
  unetpp_full_split
  segformer_full_split
  unet2d_vanilla_full_split
  canet_full_split
  fno2d_full_split
  unet2d_vanilla_12day_full_split
)

# Allow per-model override: `bash run_new_baselines.sh attunet_full_split uno_full_split`
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=("${MODELS_ALL[@]}")
fi

cd "$WORK_DIR"
mkdir -p "$SAVE_BASE"

# ── Loop ──────────────────────────────────────────────────────────────
for m in "${MODELS[@]}"; do
  EXP_TAG="${m}-${TF_TAG}${SEED_TAG}"
  EXP_DIR="$SAVE_BASE/$m/$EXP_TAG"
  CKPT="$EXP_DIR/model.pt"
  CFG="$CFG_DIR/${m}.yaml"

  if [[ ! -f "$CFG" ]]; then
    echo "[skip] no config: $CFG"
    continue
  fi

  echo
  echo "=================================================================="
  echo "  $(date '+%F %T')  MODEL: $m"
  echo "  exp_dir : $EXP_DIR"
  echo "=================================================================="

  # ── Train (skip if ckpt exists, unless FORCE=1) ────────────────────
  if [[ -f "$CKPT" && "${FORCE:-0}" != "1" ]]; then
    echo "  [skip train] $CKPT already exists  (use FORCE=1 to retrain)"
  else
    echo "  [train] $m  (seed=$SEED, train_frac=$TRAIN_FRAC, tag=$TF_TAG)"
    $PY train.py \
      --config "$CFG" \
      --train_frac "$TRAIN_FRAC" \
      --save_base "$SAVE_BASE" \
      --seed "$SEED"
  fi

  # ── Eval (always run — fast, idempotent) ───────────────────────────
  echo "  [eval] $m"
  $PY eval.py --ckpt "$CKPT"
done

echo
echo "=================================================================="
echo "  $(date '+%F %T')  ALL DONE"
echo "  Results in: $SAVE_BASE/"
echo "=================================================================="
