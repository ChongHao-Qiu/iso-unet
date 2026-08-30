# iso_unet

Model package for **ISO-UNet**: region-aware U-Net for predicting precipitation
oxygen isotopes (δ¹⁸O_p) from climate fields, plus the 14 baselines used in the
paper.

This directory is the installable Python package. The training / evaluation
pipeline, configs and data loaders live one level up in
[`../iso_all_dataset/2d_data_clean/`](../iso_all_dataset/2d_data_clean), and the
full project README (installation, data layout, training, evaluation, ablations)
is at [`../README.md`](../README.md).

## Install

```bash
pip install -e .
```

## Contents

| Module | Contents |
|---|---|
| `baseline_iso_unet.py` | **ISO-UNet (ours)** — disentangled stem, climate-state encoder, 4-way product routing + sea-ice expert, skip attention, bottleneck MoE |
| `baseline_*.py` | Baselines: U-Net, U-Net++, Attention U-Net, CA-Net, DCSAU-Net, TransUNet, SegFormer, U-NO, ClimaX, Stormer, ConvLSTM, Divided Space-Time, Video Swin, Axial |
| `model.py` | FNO2d and simple linear-regression references |
| `data.py` | `Dataset` LightningDataModule and `SequenceDataset` (sliding-window sequence dataset) |
| `convlstm.py` | ConvLSTM cell / sequence wrapper |
| `trainer.py` | Lightning `Trainer` wrapper (early stopping, CSV logging, epoch summaries) |
| `utils.py` | Misc helpers |

## Quick check

```python
import torch, iso_unet
from iso_unet.baseline_iso_unet import IsoUNetBaseline

model = IsoUNetBaseline(n_inputs=9, base_channels=64, d_state=64, H=90, W=180,
                        landfrac_idx=7, use_precip_routing=True,
                        pr_channels=(1, 2), moe_mode='product4',
                        use_ice_routing=True, ice_idx=8)
y, state = model(torch.randn(2, 12, 9, 90, 180))
print(y.shape, state.shape)
```
