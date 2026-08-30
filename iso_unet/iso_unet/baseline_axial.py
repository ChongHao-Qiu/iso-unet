"""
baseline_axial.py — Axial Attention baseline (EarthFormer self_pattern='axial').

forward(x):  (B, T=12, C=9, H=90, W=180) → (B, T=12, 1, H=90, W=180)
loss:        cos(lat) weighted MSE on all T frames
batch:       (x, y) or (x, y, co2) — co2 ignored
output kind: sequence-to-sequence (the eval pipeline takes y[:, -1])

The model body is built by `_earthformer_cuboid.build_cuboid_model` with
self_pattern='axial'. All EarthFormer hyperparams (base_units, depth, ...)
are forwarded via model_kwargs in the YAML config.

EarthFormer source path discovery: see _earthformer_cuboid._resolve_cuboid_src.
"""
from ._earthformer_cuboid import make_cuboid_baseline_class

AxialBaseline = make_cuboid_baseline_class('AxialBaseline', self_pattern='axial')


if __name__ == '__main__':
    import torch
    m = AxialBaseline(n_inputs=9, H=90, W=180, seq_len=12)
    x = torch.randn(2, 12, 9, 90, 180)
    y = m(x)
    n = sum(p.numel() for p in m.parameters())
    print(f'AxialBaseline  in={tuple(x.shape)} → out={tuple(y.shape)}  '
          f'params={n/1e6:.2f}M')
