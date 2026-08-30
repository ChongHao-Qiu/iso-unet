# Computational cost

Parameter count, FLOPs and wall-clock cost for ISO-UNet and all baselines.
Back to the [project README](../README.md).

## Benchmark setup

All models are measured on the same input shape and batch size as their training
configuration in [`iso_all_dataset/2d_data_clean/configs/`](../iso_all_dataset/2d_data_clean/configs):
a 90 × 180 (2°) grid with 9 input channels, single-frame models taking one timestep and
spatio-temporal models a 12-frame window. GFLOPs are per forward pass at batch size 1;
train and inference times are seconds per epoch on a single GPU, measured with synthetic
tensors so that data I/O is excluded.

## Results

### Single-frame models (1 timestep in → 1 out)

| Model | Params (M) | GFLOPs | Train (s/epoch) | Infer (s/epoch) |
|---|---:|---:|---:|---:|
| U-Net | 31.39 | 31.45 | 2.92 | 1.13 |
| UNet++ | 24.83 | 19.46 | 3.64 | 1.50 |
| AttU-Net | 34.88 | 37.49 | 3.74 | 1.30 |
| CA-Net | 2.79 | 3.15 | 3.22 | 1.16 |
| DCSAU-Net | 2.62 | 4.39 | 5.80 | 1.95 |
| TransUNet | 105.20 | 21.29 | 15.91 | 4.74 |
| FNO | 0.13 | 0.11 | 0.49 | 0.16 |
| U-NO | 8.22 | 0.33 | 4.08 | 2.02 |
| ClimaX | 7.81 | 7.58 | 2.45 | 0.73 |
| Stormer | 5.36 | 7.59 | 3.35 | 1.13 |
| SegFormer | 3.72 | 0.90 | 1.64 | 0.48 |

### Spatio-temporal models (12 timesteps in)

| Model | Params (M) | GFLOPs | Train (s/epoch) | Infer (s/epoch) |
|---|---:|---:|---:|---:|
| U-Net (12t) | 31.45 | 33.55 | 2.92 | 1.14 |
| DividedST | 11.48 | 61.06 | 16.62 | 5.68 |
| ConvLSTM | 0.03 | 0.98 | 1.15 | 0.47 |
| VideoSwin | 11.47 | 59.11 | 15.21 | 5.28 |
| Axial | 15.36 | 62.53 | 17.97 | 5.82 |

### Ours

| Model | Params (M) | GFLOPs | Train (s/epoch) | Infer (s/epoch) |
|---|---:|---:|---:|---:|
| **ISO-UNet** | **129.84** | **50.52** | **9.50** | **3.21** |

## Discussion

Although ISO-UNet has the largest parameter count in the comparison (129.8 M), 72.7 % of
these parameters reside in the five bottleneck experts, which operate at the lowest spatial
resolution (6 × 12). Consequently its inference cost is only 1.61× that of a vanilla U-Net
(50.5 vs 31.4 GFLOPs), and it is in fact cheaper than every spatio-temporal *attention*
baseline — requiring fewer FLOPs than DividedST (61.1), VideoSwin (59.1) and Axial (62.5)
while training 1.6–1.9× faster and using 55–61 % of their peak memory. Parameter count is
therefore a poor proxy for the computational cost of this architecture: the mixture-of-experts
capacity is conditionally routed rather than densely applied.

### Where the parameters go

The 72.7 % figure is reproducible from the released config — the `regional` module holds the
four land/ocean × wet/dry experts plus the sea-ice expert:

| Submodule | Params | Share |
|---|---:|---:|
| `regional` (5 bottleneck experts) | 94,392,320 | 72.7 % |
| `bottleneck` | 14,159,872 | 10.9 % |
| `up4` | 9,439,232 | 7.3 % |
| `enc4` | 3,540,992 | 2.7 % |
| `skip_attn_e4` | 2,951,882 | 2.3 % |
| `up3` | 2,360,320 | 1.8 % |
| everything else | 2,997,235 | 2.3 % |
| **total** | **129,841,853** | **100 %** |

To reproduce this breakdown:

```bash
cd iso_all_dataset/2d_data_clean/
python -c "
import yaml
from data import get_input_features
from models import build_model, resolve_model_kwargs

cfg   = yaml.safe_load(open('configs/iso_unet_4way_ice_full_split.yaml'))
feats = get_input_features(cfg['input_set'])
kw    = resolve_model_kwargs(cfg['model_kwargs'], feats, lr=cfg['lr'], verbose=False)
net   = build_model(cfg['model_class'], kw).net

total = sum(p.numel() for p in net.parameters())
for name, mod in net.named_children():
    n = sum(p.numel() for p in mod.parameters())
    if n:
        print(f'{name:20s} {n:>12,}  {100*n/total:5.1f}%')
print(f'{\"total\":20s} {total:>12,}')
"
```

The bottleneck sits after four 2× downsamples. With the 90 × 180 input padded to 96 × 192,
that is a **6 × 12** feature map, so the experts — despite holding most of the weights —
contribute a small share of the total FLOPs.

## Re-running the benchmark

Peak GPU memory can be re-measured with the standalone script (synthetic tensors, no dataset
required):

```bash
cd iso_all_dataset/2d_data_clean/
python benchmark_gpu_mem.py --n_epochs 5 --n_steps 10
```

It writes a CSV with `n_params_M`, `peak_alloc_GB`, `peak_reserved_GB` and wall time per
config. A GPU is required; the numbers above were collected on a single device with all
other settings at their config defaults.
