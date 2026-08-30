"""
baseline_divided_st.py — Divided Space-Time Attention baseline
(EarthFormer self_pattern='divided_st_axial').

forward(x):  (B, T=12, C=9, H=90, W=180) → (B, T=12, 1, H=90, W=180)
loss:        cos(lat) weighted MSE on all T frames
batch:       (x, y) or (x, y, co2) — co2 ignored
output kind: sequence-to-sequence (the eval pipeline takes y[:, -1])

Wrapped by `_earthformer_cuboid.build_cuboid_model`; the only difference
from `AxialBaseline` is the self_pattern string.
"""
from ._earthformer_cuboid import make_cuboid_baseline_class

DividedSTBaseline = make_cuboid_baseline_class(
    'DividedSTBaseline', self_pattern='divided_st')


if __name__ == '__main__':
    import torch
    m = DividedSTBaseline(n_inputs=9, H=90, W=180, seq_len=12)
    x = torch.randn(2, 12, 9, 90, 180)
    y = m(x)
    n = sum(p.numel() for p in m.parameters())
    print(f'DividedSTBaseline  in={tuple(x.shape)} → out={tuple(y.shape)}  '
          f'params={n/1e6:.2f}M')
