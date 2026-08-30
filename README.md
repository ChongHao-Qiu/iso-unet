# ISO-UNet

<p align="center">
  <img src="img/model.png" alt="ISO-UNet architecture" width="900">
</p>

Region-aware U-Net for predicting precipitation oxygen isotopes (δ¹⁸O_p) from climate fields, with explicit land/ocean × wet/dry routing, sea-ice expert, and climate-state conditioning.

## Overview

The repository provides:

- **`iso_unet/`** — Python package with the proposed model (ISO-UNet: 4-way product routing + ice override + climate-state context + skip attention + bottleneck MoE) and 14 baselines.
- **`iso_all_dataset/2d_data_clean/`** — training / evaluation pipeline (PyTorch Lightning), per-model YAML configs, and plotting scripts.

### Baselines included

| Family | Models |
|---|---|
| **Single-frame (1 timestep in → 1 out)** | U-Net, U-Net++, Attention U-Net, CA-Net, DCSAU-Net, TransUNet, SegFormer, FNO, U-NO, ClimaX (single-day), Stormer |
| **Time-series (12 timesteps in → 1 out)** | ConvLSTM, Divided Space-Time Attention, Video Swin, Axial Transformer |
| **Ours** | ISO-UNet (`iso_unet.baseline_iso_unet.IsoUNetBaseline`) |

## Installation

```bash
# 1. Clone
git clone <repo-url> iso-unet && cd iso-unet

# 2. Create env (Python 3.10+)
conda create -n iso_unet python=3.10
conda activate iso_unet

# 3. Install PyTorch for your platform.
#    The paper's results used the CUDA 11.8 build:
pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 4. Install the remaining dependencies
pip install -r requirements.txt

# 5. Install the iso_unet package in editable mode
pip install -e iso_unet/
```

`requirements.txt` pins every package this code imports to the version used for the
paper's results.

Key dependencies: `torch`, `lightning`, `xarray`, `x4c`, `pyyaml`, `numpy`, `cartopy`. `transformers` is needed **only** for the SegFormer baseline; every other model runs without it.

Verify the install:

```bash
python -c "import iso_unet; from iso_unet.baseline_iso_unet import IsoUNetBaseline; print('ok', iso_unet.__version__)"
```

## Data

The pipeline reads regridded iCESM monthly history files. The data root is `DATASET_DIR` in
`iso_all_dataset/2d_data_clean/data.py`, which defaults to `./data` and can be overridden
without editing any source file:

```bash
export ISO_DATASET_DIR=/path/to/iCESM/data
```

The seven CO₂ scenarios map to subdirectories as follows (this mapping is the `REGISTRY` dict in the same file):

| Tag | CO₂ | Split (default preset) | `atm_dir` | File prefix (`glob_key`) |
|---|---|---|---|---|
| `280ppm` | 280 | ID | `iCESM-Pliocene_1x/atm` | `b.e13.B1850C5CN.ne30_g16.icesm13_ihesp.midPliocene.1x.001.cam.h0` |
| `350ppm` | 350 | ID | `iCESM-Pliocene_350ppm/atm` | `b.e13.B1850C5CN.ne30_g16.icesm131_ihesp.pW.Plio_350ppm.cam.h0` |
| `420ppm` | 420 | ID | `iCESM-MCO_1.5x/atm` | `b.e13.B1850C5.ne16_g16.icesm131_d18O_fixer.Miocene.1.5xCO2.005.cam.h0` |
| `560ppm` | 560 | ID | `iCESM-Pliocene_2x/atm` | `b.e13.B1850C5CN.ne30_g16.icesm13_ihesp.midPliocene.2x.001.cam.h0` |
| `400ppm` | 400 | OOD | `iCESM-Pliocene_400ppm/atm` | `b.e13.B1850C5CN.ne30_g16.icesm131_ihesp.pW.Plio_400ppm.cam.h0` |
| `490ppm` | 490 | OOD | `iCESM-Pliocene_490ppm/atm` | `b.e13.B1850C5CN.ne30_g16.icesm131_ihesp.pW.Plio_490ppm.cam.h0` |
| `840ppm` | 840 | OOD | `iCESM-MCO_3x/atm` | `b.e13.B1850C5.ne16_g16.icesm131_d18O_fixer.Miocene.3xCO2.005.cam.h0` |

Within each `atm_dir`, one file per variable is expected, named `<glob_key>.<VAR>_*_rgd.nc`:

```
<DATASET_DIR>/
└── iCESM-Pliocene_1x/atm/
    ├── b.e13....cam.h0.d18Op_000101-010012_rgd.nc     ← target
    ├── b.e13....cam.h0.TS_000101-010012_rgd.nc
    ├── b.e13....cam.h0.PRECL_000101-010012_rgd.nc
    ├── b.e13....cam.h0.PRECC_000101-010012_rgd.nc
    ├── b.e13....cam.h0.PS_000101-010012_rgd.nc
    ├── b.e13....cam.h0.TMQ_000101-010012_rgd.nc
    ├── b.e13....cam.h0.QFLX_000101-010012_rgd.nc
    ├── b.e13....cam.h0.FLUT_000101-010012_rgd.nc
    ├── b.e13....cam.h0.LANDFRAC_000101-010012_rgd.nc
    └── b.e13....cam.h0.aice_000101-010012_rgd.nc
```

All fields are on a regular 90 × 180 (2° × 2°) lat–lon grid, monthly resolution.

Channel names used by the configs map to CESM variables as: `tas`→`TS`, `precl`→`PRECL`, `precc`→`PRECC`, `pr`→`PRECL + PRECC`; `PS`, `TMQ`, `QFLX`, `FLUT`, `LANDFRAC`, `aice` keep their names. The `full_split` input set used throughout the paper is the 9 channels `[tas, precl, precc, PS, TMQ, QFLX, FLUT, LANDFRAC, aice]`, with `LANDFRAC` at index 7 and `aice` at index 8 (the routing code relies on this ordering; `train.py` auto-detects and corrects the indices).

## Training

```bash
cd iso_all_dataset/2d_data_clean/

python train.py \
    --config configs/iso_unet_4way_ice_full_split.yaml \
    --seed 42 \
    --save_base ./experiments
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--config` | (required) | YAML config path |
| `--save_base` | `./experiments` | Root dir for checkpoints |
| `--exp_tag` | `<model_name>-tf<NN>` | Override the run's subdirectory name |
| `--seed` | `42` | RNG seed (model save dir gets `-s<seed>` suffix when ≠ 42) |
| `--batch_size` | from config | Override batch size |
| `--lr` | from config | Override learning rate |
| `--max_epochs` | from config | Override max epochs |
| `--patience` | from config | Early-stop patience (epochs) |
| `--max_batch` | from config | Cap timesteps loaded per dataset (debugging) |
| `--ood_preset` | from config (`default`) | OOD split: `default` / `low_ood` / `high_ood` / `mid_ood` |
| `--no_lat_weights` | off | Disable cos(lat)-weighted MSE loss |

Output for each run is saved to:
```
<save_base>/<model_name>/<exp_tag>/
├── config.yaml          # frozen config (incl. normalization stats + param count)
├── model.pt             # best checkpoint
├── train.log            # training log
└── loss_curve.png       # train / val loss curve
```

### Example: ISO-UNet (ours)

```bash
python train.py \
    --config configs/iso_unet_4way_ice_full_split.yaml \
    --seed 42 \
    --save_base ./experiments
```

### Example: baselines

```bash
# U-Net (single-frame)
python train.py --config configs/unet2d_vanilla_full_split.yaml \
                --save_base ./experiments

# ConvLSTM (sequence)
python train.py --config configs/convlstm_full_split.yaml \
                --save_base ./experiments

# ClimaX (single-day variant)
python train.py --config configs/climax_single_day_full_split.yaml \
                --save_base ./experiments
```

## Evaluation

```bash
python eval.py --ckpt <save_base>/<model_name>/<exp_tag>/model.pt
```

Optional arguments:

| Argument | Description |
|---|---|
| `--ckpt` (required) | Path to `model.pt` |
| `--config` | Override config path (default: `<ckpt_dir>/config.yaml`) |
| `--output_dir` | Override output dir (default: same as `<ckpt_dir>`) |
| `--batch_size` | Override eval batch size |
| `--save_attn` | If model supports `forward_with_attn`, dump attention to NetCDF |
| `--seed` | RNG seed (default `42`) |

Output added to the ckpt dir:
```
<ckpt_dir>/
├── eval_results.json    # per-dataset RMSE / MAE / R² (overall + per-region)
├── eval.log
└── spatial/             # per-CO2 spatial RMSE / MAE / R² PDFs + NetCDFs
    ├── predict_280ppm_RMSE_spatial.pdf
    ├── predict_280ppm_truth_pred.nc
    └── ...
```

`eval_results.json` schema — `regions` holds the 10 boxes `R01`…`R10` defined by
`metrics.define_regions()` (2 latitude bands × 5 longitude bands):

```json
{
  "280ppm": {
    "overall": { "RMSE": 0.79, "MAE": 0.58, "R2": 0.77 },
    "regions": {
      "R01": { "RMSE": 0.71, "MAE": 0.52, "R2": 0.80 },
      "R02": { "...": "..." }
    }
  },
  "350ppm": { "...": "..." }
}
```

The reported ID / OOD averages are the unweighted means of the per-dataset `overall.RMSE` over the ID tags (280/350/420/560) and OOD tags (400/490/840) respectively.

## Configuration

Configs live in `iso_all_dataset/2d_data_clean/configs/*.yaml`. Each config pins the model class, its `model_kwargs`, and the training schedule. Run-level settings — the data split, `ood_preset`, `norm_mode` / `norm_ref`, `lat_weighted` — are deliberately **not** in the configs: they are CLI arguments with defaults in `train.py`, so the same config can be reused across splits. Below is `configs/iso_unet_4way_ice_full_split.yaml`, the **Full (Ours)** model, verbatim.

```yaml
model_name:   iso_unet_4way_ice_full_split
model_class:  iso_unet.baseline_iso_unet.IsoUNetBaseline
dataset_kind: sequence            # 'single_frame' or 'sequence' (T = seq_len)
input_set:    full_split          # feature set in data.py FEATURE_SETS (9 ch)
seq_len:      12                  # 12 monthly frames = 1 model year
batch_size:   4
lr:           1.0e-3
max_epochs:   1000
patience:     10                  # early stop on val loss
max_batch:    3000                # cap timesteps loaded per dataset
gradient_clip_val: 1.0

model_kwargs:
  landfrac_idx:           7       # index of LANDFRAC (train.py auto-detects)
  base_channels:          64
  d_state:                64      # climate-state embedding dim
  H:                      90
  W:                      180
  stem_k:                 4       # per-variable width in the disentangled stem
  use_disentangled_stem:  true
  use_temporal_context:   true    # climate-state encoder + FiLM + prompt loss
  use_bottleneck_moe:     true
  use_skip_attn:          true
  skip_attn_levels:       [1, 2, 3, 4]
  # ── 4-way product routing ──
  use_precip_routing:     true
  # pr_channels:          [1, 2]  # precl + precc (auto-detected)
  moe_mode:               product4
  # ── Sea-ice expert ──
  use_ice_routing:        true
  # ice_idx:              8       # aice (auto-detected)
  # ── Prompt-alignment loss ──
  lambda_prompt:          0.1
  lambda_prompt_final:    0.0
  lambda_decay_epochs:    50
  prompt_tau:             0.5
```

Building this config yields **129,841,853** parameters, matching the run recorded in the paper.

The run-level defaults `train.py` applies when the CLI does not override them are
`norm_mode: global`, `norm_ref: 280ppm`, `lat_weighted: true`, `ood_preset: default`
(see the `cfg.setdefault(...)` block near the top of `train.py`); each run's frozen
`config.yaml` records the values actually used, together with the resolved `model_kwargs`
and the normalization statistics.

### Per-model hyperparameters

Batch size and learning rate are **per model**, tuned so each baseline fits in GPU memory
and trains stably; they are not shared across the table. Every model uses the same schedule
otherwise (`max_epochs: 1000` with early stopping on validation loss, `gradient_clip_val: 1.0`,
`ood_preset: default`, `norm_mode: global`, cos(lat)-weighted MSE, and the same training split).
The values below are exactly what the released configs contain and what the recorded runs used:

| Model | Config | Kind | Batch | LR | Patience |
|---|---|---|---|---|---|
| **ISO-UNet (ours)** | `iso_unet_4way_ice_full_split` | sequence | **4** | 1e-3 | 10 |
| U-Net | `unet2d_vanilla_full_split` | single-frame | 8 | 1e-3 | 10 |
| U-Net (12-day) | `unet2d_vanilla_12day_full_split` | sequence | 8 | 1e-3 | 10 |
| U-Net++ | `unetpp_full_split` | single-frame | 8 | 1e-3 | 10 |
| Attention U-Net | `attunet_full_split` | single-frame | 8 | 1e-3 | 10 |
| CA-Net | `canet_full_split` | single-frame | 8 | 1e-3 | 10 |
| DCSAU-Net | `dcsaunet_full_split` | single-frame | 8 | 1e-3 | 10 |
| TransUNet | `transunet_full_split` | single-frame | 4 | 3e-4 | 10 |
| SegFormer | `segformer_full_split` | single-frame | 16 | 1e-4 | 10 |
| FNO | `fno2d_full_split` | single-frame | 16 | 1e-3 | 10 |
| U-NO | `uno_full_split` | single-frame | 8 | 1e-3 | 10 |
| ClimaX (single-day) | `climax_single_day_full_split` | single-frame | 16 | 3e-4 | 10 |
| ClimaX | `climax_full_split` | sequence | 4 | 1e-3 | 30 |
| Stormer | `stormer_full_split` | sequence | 4 | 3e-4 | 10 |
| ConvLSTM | `convlstm_full_split` | sequence | 8 | 1e-3 | 10 |
| Divided Space-Time | `divided_st_full_split` | sequence | 4 | 3e-4 | 10 |
| Video Swin | `video_swin_full_split` | sequence | 4 | 3e-4 | 10 |
| Axial Transformer | `axial_full_split` | sequence | 4 | 3e-4 | 10 |

All five ablation configs inherit the ISO-UNet row (batch 4, lr 1e-3, patience 10).

### Climate-state encoder

The climate-state encoder (`ClimateStateEncoder` in `iso_unet/iso_unet/baseline_iso_unet.py`) summarises the T−1 context frames into a `d_state`-dimensional vector, which conditions the bottleneck through FiLM, supplies the per-expert routing offsets, and is the quantity the prompt-alignment loss is defined on.

The variant used for **all results in the paper** is `time_pool: mean`, which is the default and is therefore not written out in the config above:

> time-average the context frames → 3-layer 2D CNN (5×5 s2 → 3×3 s2 → 3×3 s2, channels `32 → 32 → 64`, each BN+ReLU) → global average pool → 2-layer MLP → `d_state = 64`.

Two alternative temporal poolings are implemented and shipped as separate configs; they are **not** used for the main results:

| `time_pool` | Description | Config |
|---|---|---|
| `mean` | Time-average, then 2D CNN. **Default; used in the paper.** | `iso_unet_4way_ice_full_split.yaml` |
| `gru` | Per-frame 2D CNN → 1-layer GRU over time (order-aware) | `iso_unet_4way_ice_tgru_full_split.yaml` |
| `conv3d` | Joint 3D conv over (T, H, W) → 3D global average pool | `iso_unet_4way_ice_tcnv3d_full_split.yaml` |

All three output `(B, d_state)`, so downstream consumers are unaffected. Runs with a non-default `time_pool` get a `-tgru` / `-tcnv3d` suffix appended to `exp_tag` automatically.

## Ablation study

The leave-one-out ablation removes exactly one component from the full model, keeping every other setting identical (`base_channels: 64`, `d_state: 64`, `batch_size: 4`, `lr: 1e-3`, `ood_preset: default`, `norm_mode: global`, `lat_weighted: true`, early stop at `patience: 10`, and the same training split). One config file per row is provided:

| Row in paper | Config | Flag changed vs. Full | Params |
|---|---|---|---|
| **Full (Ours)** | `configs/iso_unet_4way_ice_full_split.yaml` | — | 129,841,853 |
| L1 − GroupConv | `configs/loo_L1_no_stem.yaml` | `use_disentangled_stem: false` | 129,879,961 |
| L2 − ClimateState | `configs/loo_L2_no_context.yaml` | `use_temporal_context: false` † | 129,663,921 |
| L3 − Bottleneck | `configs/loo_L3_no_moe.yaml` | `use_bottleneck_moe: false` ‡ | 34,665,189 |
| L4 − SkipConnection | `configs/loo_L4_no_skip_attn.yaml` | `use_skip_attn: false` | 125,919,173 |
| L5 − Ice Expert | `configs/loo_L5_no_ice.yaml` | `use_ice_routing: false` | 110,179,045 |

† Removing the climate-state encoder also removes everything defined on it — the bottleneck FiLM modulation, the per-expert climate-state offsets, and the prompt-alignment loss. The `lambda_prompt*` / `prompt_tau` keys are therefore dropped from that config.

‡ The sea-ice expert lives inside the bottleneck MoE, so disabling the MoE necessarily disables ice routing as well (`use_ice_routing: false` in that config).

The parameter counts above are what these configs build; they match the values recorded in the corresponding paper runs, and can be re-checked without any data:

```bash
cd iso_all_dataset/2d_data_clean/
python -c "
import yaml
from data import get_input_features
from models import build_model, resolve_model_kwargs
for name in ['iso_unet_4way_ice_full_split', 'loo_L1_no_stem', 'loo_L2_no_context',
             'loo_L3_no_moe', 'loo_L4_no_skip_attn', 'loo_L5_no_ice']:
    cfg = yaml.safe_load(open(f'configs/{name}.yaml'))
    feats = get_input_features(cfg['input_set'])
    kw = resolve_model_kwargs(cfg['model_kwargs'], feats, lr=cfg['lr'], verbose=False)
    m = build_model(cfg['model_class'], kw)
    print(f'{name:32s} {sum(p.numel() for p in m.parameters()):,}')
"
```

`resolve_model_kwargs` fills in the kwargs that depend on the channel ordering of the
chosen `input_set` (`landfrac_idx`, `pr_channels`, `ice_idx`); `train.py` calls the same
function, so this builds exactly the model that training would.

### Reproducing the ablation table

Each row is run over the 3 seeds **40, 42, 44**, and reported as mean ± std of test RMSE:

```bash
cd iso_all_dataset/2d_data_clean/

for CFG in iso_unet_4way_ice_full_split \
           loo_L1_no_stem loo_L2_no_context loo_L3_no_moe \
           loo_L4_no_skip_attn loo_L5_no_ice; do
  for SEED in 40 42 44; do
    python train.py --config configs/${CFG}.yaml \
                    --seed ${SEED} \
                    --save_base ./experiments_ablation
  done
done
```

Runs land in `./experiments_ablation/<model_name>/<exp_tag>/`, where `exp_tag` is `<model_name>-tf<NN>` plus a `-s<seed>` suffix for any seed other than the default 42. Then evaluate each:

```bash
for CKPT in ./experiments_ablation/*/*/model.pt; do
  python eval.py --ckpt "$CKPT"
done
```

Each run's `eval_results.json` carries the per-CO₂ `overall.RMSE` values that form the seven columns of the table; the ID and OOD columns are the means over {280, 350, 420, 560} and {400, 490, 840} ppm respectively.

For reference, the aggregated ID / OOD RMSE reported in the paper (mean ± std over the three seeds; lower is better):

| Model | ID | OOD |
|---|---|---|
| **Full (Ours)** | **0.776 ± 0.006** | **0.801 ± 0.011** |
| L1 − GroupConv | 0.780 ± 0.002 | 0.819 ± 0.012 |
| L2 − ClimateState | 0.791 ± 0.011 | 0.819 ± 0.020 |
| L3 − Bottleneck | 0.776 ± 0.003 | 0.814 ± 0.013 |
| L4 − SkipConnection | 0.784 ± 0.006 | 0.807 ± 0.006 |
| L5 − Ice Expert | 0.788 ± 0.012 | 0.824 ± 0.021 |

## Computational cost

ISO-UNet has the largest parameter count in the comparison (129.8 M), but 72.7 % of those
parameters sit in the five bottleneck experts, which run at the lowest spatial resolution
(6 × 12). Its inference cost is therefore only 1.61× a vanilla U-Net's (50.5 vs 31.4 GFLOPs),
and it uses fewer FLOPs than every spatio-temporal *attention* baseline while training
1.6–1.9× faster than them.

| Model | Params (M) | GFLOPs | Train (s/epoch) | Infer (s/epoch) |
|---|---:|---:|---:|---:|
| U-Net | 31.39 | 31.45 | 2.92 | 1.13 |
| DividedST | 11.48 | 61.06 | 16.62 | 5.68 |
| VideoSwin | 11.47 | 59.11 | 15.21 | 5.28 |
| Axial | 15.36 | 62.53 | 17.97 | 5.82 |
| **ISO-UNet** | **129.84** | **50.52** | **9.50** | **3.21** |

**→ Full table for all 17 models, the parameter breakdown, and how to re-run the benchmark:
[docs/computational-cost.md](docs/computational-cost.md)**

## OOD splits

The default OOD preset reserves 400 / 490 / 840 ppm as out-of-distribution test sets (ID = 280 / 350 / 420 / 560 ppm). Other presets (defined in `OOD_PRESETS` in `data.py`):

| Preset | ID (train) | OOD (held out) |
|---|---|---|
| `default` | 280, 350, 420, 560 | 400, 490, 840 |
| `low_ood` | 420, 490, 560, 840 | 280, 350, 400 |
| `high_ood` | 280, 350, 400, 420 | 490, 560, 840 |
| `mid_ood` | 280, 400, 490, 840 | 350, 420, 560 |

A non-default preset is appended to `exp_tag`, so different splits do not overwrite each other.

## Repository layout

```
iso-unet/
├── README.md                              # this file
├── docs/computational-cost.md             # params / FLOPs / runtime for all models
├── img/model.png                          # architecture diagram
├── requirements.txt                       # pinned dependencies
├── iso_unet/                              # ISO-UNet package (pip install -e .)
│   ├── README.md
│   ├── setup.py
│   └── iso_unet/
│       ├── __init__.py
│       ├── baseline_iso_unet.py           # Our model (ISO-UNet)
│       ├── baseline_*.py                  # 14 baselines
│       ├── convlstm.py
│       ├── data.py                        # Dataset + SequenceDataset
│       ├── model.py                       # FNO2d + framework
│       ├── trainer.py
│       └── utils.py
└── iso_all_dataset/
    └── 2d_data_clean/
        ├── train.py                       # CLI entry: train
        ├── eval.py                        # CLI entry: eval
        ├── data.py                        # dataset registry + loaders
        ├── datasets.py                    # split builders
        ├── models.py                      # model factory
        ├── metrics.py                     # RMSE/MAE/R² + region masks
        ├── visualize.py                   # spatial plots
        ├── configs/                       # per-model YAMLs (baselines + ablations)
        └── draw_pics/                     # standalone plot scripts
```
