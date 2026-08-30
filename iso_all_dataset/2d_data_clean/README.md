# Training / evaluation pipeline

A unified training + eval entry point that shares the same code across single-frame and
sequence models. See the [project README](../../README.md) for installation, data layout,
and the ablation study; this file documents the pipeline's internals.

## Directory layout

```
2d_data_clean/
├── configs/                  ← one YAML per model (32 total)
│   ├── unet2d_vanilla_full_split.yaml     U-Net single-frame baseline
│   ├── unetpp_full_split.yaml             U-Net++ (base=64)
│   ├── fno2d_full_split.yaml              FNO2D single-frame baseline
│   ├── segformer_full_split.yaml          SegFormer (HuggingFace MiT-b0)
│   ├── canet_full_split.yaml              CA-Net (4 attention modules)
│   ├── convlstm_full_split.yaml           ConvLSTM (12-frame sequence)
│   ├── iso_unet_4way_ice_full_split.yaml  ★ ISO-UNet (ours) — the paper's Full model
│   ├── loo_L{1..5}_*.yaml                 leave-one-out ablations of the Full model
│   └── ...                                remaining baselines
│
├── data.py        ← REGISTRY + load_xr + normalize_inputs (shared by single-frame / sequence)
├── datasets.py    ← SingleFrameDataset + build_splits (SequenceDataset comes from iso_unet.data)
├── models.py      ← Model factory + resolve_model_kwargs + run_inference_unified
├── metrics.py     ← Regions (R01–R10) + R²/MSE/RMSE/MAE
├── visualize.py   ← spatial PDF plotter
│
├── train.py       ← Training entry  (writes model.pt + config.yaml + train.log + loss_curve.png)
├── eval.py        ← Eval entry      (reads ckpt + config, writes eval_results.json + spatial PDFs;
│                                     **does NOT touch train.log**, writes a separate eval.log)
├── draw_pics/     ← standalone figure scripts
└── README.md
```

## Usage

### Training

```bash
# ISO-UNet (ours)
python train.py --config configs/iso_unet_4way_ice_full_split.yaml

# U-Net baseline (single-frame)
python train.py --config configs/unet2d_vanilla_full_split.yaml

# Custom batch / lr / epochs  (CLI overrides YAML)
python train.py --config configs/canet_full_split.yaml \
    --batch_size 16 --lr 5e-4 --max_epochs 500

# Custom exp_tag (default is <model_name>-tf<NN>)
python train.py --config configs/loo_L1_no_stem.yaml \
    --exp_tag ablation-nostem
```

Training outputs go to `<save_base>/<model_name>/<exp_tag>/` (`--save_base`, default `./experiments`):
```
model.pt        ← weights
config.yaml     ← full training config, incl. resolved model_kwargs + norm stats (read by eval.py)
train.log       ← training log
loss_curve.png  ← loss + (λ schedule if prompt loss is used)
```

The data root defaults to `./data`; override it with `export ISO_DATASET_DIR=/path/to/data`.

### Eval

```bash
# Run eval — automatically reads config.yaml next to the ckpt
python eval.py --ckpt ./experiments/<model_name>/<exp_tag>/model.pt

# Custom output location (default is the ckpt's directory)
python eval.py --ckpt ... --output_dir ./reeval_v1

# Also save raw attention dict (only effective if model supports forward_with_attn)
python eval.py --ckpt ... --save_attn
```

Eval outputs (same schema for every model):
```
eval.log                                      ← eval-only log (train.log is untouched)
eval_results.json                             ← cross-model-comparable metrics
spatial/
  ├── <tag>_truth_pred.nc                     ← truth + pred for the 7 datasets
  ├── <tag>_metrics_2d.nc                     ← per-pixel R²/RMSE/MAE
  └── predict_<tag>_{R2,RMSE,MAE}_spatial.pdf ← x4c Robinson projection + coastlines
attention_raw/    (only with --save_attn)
  └── <tag>_attn.npz
```

## Design points

### 1. Train / eval are independent

- `train.py` only writes `train.log`
- `eval.py` only writes `eval.log`
- The two share `config.yaml` (train.py saves it → eval.py reads it), so re-evaluating a
  checkpoint never disturbs the training record.

### 2. One eval handles single-frame and sequence

`models.run_inference_unified` auto-detects:
- single-frame model: input `(B, C, H, W)`, output `(B, 1, H, W)`
- sequence model with dense output `(B, T, 1, H, W)`: take `[:, -1]` as the last frame
- sequence model with single output `(B, 1, H, W)` (ISO-UNet): use directly

All are normalized to `(N, H, W)` truth/pred → the same metrics apply.

### 3. The dataset switches automatically by model type

`dataset_kind` in the config decides:
- `single_frame` → `SingleFrameDataset` — `(x_t, y_t, co2)`
- `sequence`     → `iso_unet.data.SequenceDataset` — `(x_window, y_seq, co2)`, 12 frames

Each model registers a default in `models.DEFAULT_DATASET_KIND`; the config can override it.

### 4. Channel-dependent kwargs are resolved, not hard-coded

`landfrac_idx`, `pr_channels` and `ice_idx` are properties of the chosen `input_set`, not of
the model, so configs may leave them out. `models.resolve_model_kwargs(model_kwargs,
input_features, lr=...)` fills them in, and `train.py` calls it. Any standalone script that
rebuilds a model from a YAML should call it too:

```python
import yaml
from data import get_input_features
from models import build_model, resolve_model_kwargs

cfg   = yaml.safe_load(open('configs/iso_unet_4way_ice_full_split.yaml'))
feats = get_input_features(cfg['input_set'])
kw    = resolve_model_kwargs(cfg['model_kwargs'], feats, lr=cfg['lr'])
model = build_model(cfg['model_class'], kw)
```

### 5. How to add a new model

1. Keep the model code in the `iso_unet` package; it must be a Lightning Module.
2. Add an entry to `models.DEFAULT_DATASET_KIND`:
   `'iso_unet.your.YourClass': 'single_frame'` (or `'sequence'`).
3. Write `configs/your_model.yaml`:
   ```yaml
   model_name:   your_model
   model_class:  iso_unet.your.YourClass
   dataset_kind: single_frame    # optional; the factory looks up DEFAULT_DATASET_KIND
   input_set:    full_split      # or basic / full / split_pr / a custom list
   batch_size:   16
   lr:           1.0e-3
   max_epochs:   1000
   patience:     10
   model_kwargs:
     # ... model-specific parameters
   ```
4. Run `python train.py --config configs/your_model.yaml`.

`build_model` filters out any kwarg the class `__init__` does not accept (printing a notice),
so a shared config field will not break a model that ignores it.

### 6. CLI override

All top-level YAML parameters can be overridden via the CLI:
```bash
python train.py --config configs/unet2d_vanilla_full_split.yaml \
    --batch_size 32 --lr 5e-4 --max_epochs 200 --max_batch 1000
```

Parameters inside `model_kwargs` can only be changed via YAML — a deliberate choice to avoid
CLI explosion. This is why each ablation ships as its own config file.

## Notes

- Training uses `accelerator='auto'`: CUDA when available, otherwise CPU. The paper's runs
  used a single GPU; CPU works but is slow.
- Region metrics `R01`–`R10` are the 2 latitude × 5 longitude boxes defined in
  `metrics.define_regions()`.
- `--save_attn` only dumps the raw attention dict to npz; the figures in the paper are
  produced by the scripts in `draw_pics/` and `draw_feature.py` / `draw_inner.py`.
