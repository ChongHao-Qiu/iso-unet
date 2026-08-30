"""
draw_expert_focus.py — Paper-quality "expert focus map" visualization.

For each ISO-UNet 4-way expert, computes:
    effective_attn(h, w) = sigmoid_α(h, w) × routing_weight(h, w)

This "effective attention" is what the expert ACTUALLY contributes to the
final feature modulation — naturally confined to its physical regime.

Generates 3 figures:
    A) per-dataset 2×2 grid (4 experts) — one PDF per ppm
    B) HERO: 4 experts × 3 CO₂ regimes (paper figure)
    C) Bottleneck vs skip e1/e3 (per dataset)

Usage:
    python draw_expert_focus.py \\
        --ckpt /path/to/model.pt \\
        --out_dir /path/to/output

Requires: a ISO-UNet 4-way ckpt (moe_mode='product4').
"""
import os
import sys
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure local imports
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import x4c                                          # for Robinson projection + coastlines

from data import (REGISTRY, apply_ood_preset, REF_DATASET, compute_ref_stats,
                  get_input_features, load_xr)
from datasets import build_splits
from models import build_model


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description='Visualize ISO-UNet expert focus maps')
parser.add_argument('--ckpt', required=True,
                    help='Path to ISO-UNet 4-way model.pt')
parser.add_argument('--out_dir', default=None,
                    help='Output dir (default: <ckpt_dir>/expert_focus/)')
parser.add_argument('--level', type=int, default=3, choices=[1, 2, 3, 4],
                    help='Which skip-attn level for Fig A and B (default: e3)')
parser.add_argument('--hero_datasets', nargs='+',
                    default=['280ppm', '420ppm', '840ppm'],
                    help='Which datasets for Figure B hero plot (default: 280, 420, 840)')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Setup
# ════════════════════════════════════════════════════════════════════════════
ckpt_path = os.path.abspath(args.ckpt)
ckpt_dir  = os.path.dirname(ckpt_path)
out_dir   = args.out_dir or os.path.join(ckpt_dir, 'expert_focus')
os.makedirs(out_dir, exist_ok=True)

config_path = os.path.join(ckpt_dir, 'config.yaml')
assert os.path.isfile(config_path), f'no config.yaml at {config_path}'
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# Apply same OOD preset as training
apply_ood_preset(cfg.get('ood_preset', 'default'))

print(f'Loading datasets...')
input_features = cfg.get('input_features') or get_input_features(cfg['input_set'])
raw = {tag: load_xr(tag, max_batch=cfg['max_batch'], input_features=input_features)
       for tag in REGISTRY}

# Resolve norm stats
ref_stats = None
if cfg.get('norm_mode', 'per_dataset') == 'global':
    if 'norm_stats' in cfg and cfg['norm_stats']:
        ref_stats = cfg['norm_stats']
    else:
        ref_stats = compute_ref_stats(raw[cfg.get('norm_ref', REF_DATASET)],
                                       input_features, cfg['train_frac'])

# Build splits
splits = {}
for tag in REGISTRY:
    tr, va, te_a, te_b, stats, _ = build_splits(
        tag, raw[tag], cfg['train_frac'], cfg['seq_len'],
        input_features, cfg.get('dataset_kind', 'sequence'),
        ref_stats=ref_stats)
    splits[tag] = (tr, va, te_a, te_b, stats)

# Build + load model
mkw = dict(cfg['model_kwargs'])
mkw['n_inputs'] = len(input_features)
lat_weighted = cfg.get('lat_weighted', True)
lat_weights = np.cos(np.deg2rad(np.linspace(-89, 89, 90)))
lat_weights = lat_weights / lat_weights.mean()
weights_t = torch.as_tensor(lat_weights, dtype=torch.float32) if lat_weighted else None

model = build_model(cfg['model_class'], model_kwargs=mkw, weights=weights_t)
sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model.load_state_dict(sd, strict=False)
model.to(args.device).eval()

# Confirm it's a 4-way ISO-UNet
assert hasattr(model.net, 'regional') and model.net.regional is not None, (
    'Need a 4-way ISO-UNet ckpt')
mm = getattr(model.net, '_moe_mode', None)
assert mm == 'product4', f"This script needs moe_mode='product4', got '{mm}'"

print(f'Model loaded: {cfg["model_name"]}  moe_mode=product4')
print(f'OOD preset: {cfg.get("ood_preset", "default")}')


# ════════════════════════════════════════════════════════════════════════════
# Collect attention caches per dataset
# ════════════════════════════════════════════════════════════════════════════
LEVEL = args.level
print(f'\nCollecting attention from forward_with_attn (skip e{LEVEL})...')

attn_summary = {}   # tag → {α_LW, α_LD, α_OW, α_OD, lf, pr}
bottleneck_summary = {}  # tag → {b_LW, b_LD, b_OW, b_OD, lf_b, pr_b}

with torch.no_grad():
    for tag in REGISTRY:
        test_ds = splits[tag][2]
        loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)
        alphas  = {s: [] for s in ('LW', 'LD', 'OW', 'OD')}
        lf_e3   = []
        pr_e3   = []
        b_norms = {s: [] for s in ('LW', 'LD', 'OW', 'OD')}
        lf_b    = []
        pr_b    = []
        for batch in loader:
            x = batch[0].to(args.device)
            _, attn = model.forward_with_attn(x)
            # Skip-attn at chosen level
            for s in ('LW', 'LD', 'OW', 'OD'):
                a = attn[f'skip{LEVEL}_attn_{s}']
                alphas[s].append(a.squeeze(1).cpu().numpy())
            # Routing weights at this level (cached in the module)
            mod = getattr(model.net, f'skip_attn_e{LEVEL}')
            lf_e3.append(mod._last_landfrac_local.squeeze(1).cpu().numpy())
            pr_e3.append(mod._last_pr_local.squeeze(1).cpu().numpy())
            # Bottleneck expert ‖act‖
            for s in ('LW', 'LD', 'OW', 'OD'):
                b = attn[f'b_{s}']                          # (B, C, H_b, W_b)
                b_norms[s].append(b.norm(dim=1).cpu().numpy())
            lf_b.append(attn['landfrac_b'].squeeze(1).cpu().numpy())
            pr_b.append(attn['pr_b'].squeeze(1).cpu().numpy())

        attn_summary[tag] = {
            'α_LW': np.concatenate(alphas['LW'], axis=0).mean(axis=0),
            'α_LD': np.concatenate(alphas['LD'], axis=0).mean(axis=0),
            'α_OW': np.concatenate(alphas['OW'], axis=0).mean(axis=0),
            'α_OD': np.concatenate(alphas['OD'], axis=0).mean(axis=0),
            'lf':   np.concatenate(lf_e3, axis=0).mean(axis=0),
            'pr':   np.concatenate(pr_e3, axis=0).mean(axis=0),
        }
        bottleneck_summary[tag] = {
            'b_LW': np.concatenate(b_norms['LW'], axis=0).mean(axis=0),
            'b_LD': np.concatenate(b_norms['LD'], axis=0).mean(axis=0),
            'b_OW': np.concatenate(b_norms['OW'], axis=0).mean(axis=0),
            'b_OD': np.concatenate(b_norms['OD'], axis=0).mean(axis=0),
            'lf':   np.concatenate(lf_b, axis=0).mean(axis=0),
            'pr':   np.concatenate(pr_b, axis=0).mean(axis=0),
        }
        print(f'  {tag}  α shape {attn_summary[tag]["α_LW"].shape}  '
              f'b shape {bottleneck_summary[tag]["b_LW"].shape}')


# ════════════════════════════════════════════════════════════════════════════
# Helper: compute effective attention per expert
# ════════════════════════════════════════════════════════════════════════════
def routing_weight(s_dict, side):
    """Returns w(h,w) for one expert side, given a dict with 'lf' and 'pr'."""
    lf, pr = s_dict['lf'], s_dict['pr']
    if side == 'LW': return lf * pr
    if side == 'LD': return lf * (1.0 - pr)
    if side == 'OW': return (1.0 - lf) * pr
    if side == 'OD': return (1.0 - lf) * (1.0 - pr)
    raise ValueError(side)


def effective_α(s_dict, side):
    """effective_α = α × w (skip-attn level)"""
    return s_dict[f'α_{side}'] * routing_weight(s_dict, side)


def effective_b(s_dict, side):
    """Bottleneck effective contribution = ‖E‖ × w (proxy)"""
    return s_dict[f'b_{side}'] * routing_weight(s_dict, side)


def to_xr(arr_2d, lat_n=90, lon_n=180):
    """Upsample to 90×180 and wrap in xarray for x4c."""
    if arr_2d.shape != (lat_n, lon_n):
        t = torch.from_numpy(arr_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(lat_n, lon_n), mode='bilinear', align_corners=False)
        arr_2d = t.squeeze().numpy()
    lat = np.linspace(-89, 89, lat_n)
    lon = np.linspace(0, 360, lon_n, endpoint=False) + 1.0
    return xr.DataArray(arr_2d, dims=['lat', 'lon'],
                        coords={'lat': lat, 'lon': lon},
                        attrs={'long_name': 'effective α'})


# ════════════════════════════════════════════════════════════════════════════
# Color settings
# ════════════════════════════════════════════════════════════════════════════
SIDES = ['LW', 'LD', 'OW', 'OD']
SIDE_LABELS = {
    'LW': 'Land + Wet  (rainforest, monsoon)',
    'LD': 'Land + Dry  (desert, arid uplands)',
    'OW': 'Ocean + Wet (ITCZ, storm tracks)',
    'OD': 'Ocean + Dry (subtropical subsidence)',
}
# 4 distinct cmaps, monochrome, sequential
SIDE_CMAPS = {'LW': 'YlGn', 'LD': 'YlOrBr', 'OW': 'PuBu', 'OD': 'BuPu'}


# ════════════════════════════════════════════════════════════════════════════
# FIGURE A — Per-dataset 2×2 grid (4 experts), one PDF per ppm
# ════════════════════════════════════════════════════════════════════════════
print('\n[Figure A] Per-dataset 2×2 expert focus maps...')
fig_a_dir = os.path.join(out_dir, 'fig_A_per_dataset')
os.makedirs(fig_a_dir, exist_ok=True)

# Shared color levels for visual comparability
LEVELS_EFF = np.linspace(0, 0.5, 21)

for tag in REGISTRY:
    co2 = REGISTRY[tag]['co2']
    holdout = REGISTRY[tag]['holdout']
    flag = ' [OOD]' if holdout else ''

    fig, axes = x4c.plot_grid(2, 2, figsize=(13, 6), projection='Robinson')
    # x4c's plot_grid returns (fig, axes) — fall back if not available:
    # If x4c doesn't have plot_grid, use plain subplots
    try:
        pass
    except Exception:
        import cartopy.crs as ccrs
        fig, axes = plt.subplots(2, 2, figsize=(13, 6),
                                  subplot_kw={'projection': ccrs.Robinson()})

    for i, side in enumerate(SIDES):
        ax = axes.flatten()[i] if hasattr(axes, 'flatten') else axes[i//2, i%2]
        eff = effective_α(attn_summary[tag], side)
        da  = to_xr(eff)
        # x4c plot inside given ax
        fig_, _ = da.x.plot(cmap=SIDE_CMAPS[side], levels=LEVELS_EFF,
                            cbar_kwargs={'label': f'effective α_{side}'},
                            title=f'{side}: {SIDE_LABELS[side]}',
                            ax=ax, projection='Robinson',
                            extend='max')
        ax.coastlines(linewidth=0.5)

    fig.suptitle(f'{tag} ({co2} ppm){flag} — Expert Focus Maps  '
                 f'(effective α = sigmoid α × routing weight @ skip e{LEVEL})',
                 fontsize=12, y=0.99)
    fig.tight_layout()
    out_pdf = os.path.join(fig_a_dir, f'expert_focus_{tag}.pdf')
    fig.savefig(out_pdf, dpi=140, bbox_inches='tight')
    fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {os.path.basename(out_pdf)}')


# ════════════════════════════════════════════════════════════════════════════
# FIGURE B — HERO: 4 experts × N selected CO₂ regimes
# ════════════════════════════════════════════════════════════════════════════
print(f'\n[Figure B - HERO] 4 experts × {len(args.hero_datasets)} CO₂ regimes...')

import cartopy.crs as ccrs
hero_datasets = [t for t in args.hero_datasets if t in REGISTRY]
N_COLS = len(hero_datasets)
N_ROWS = 4    # 4 experts

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4.5 * N_COLS, 3.0 * N_ROWS),
                          subplot_kw={'projection': ccrs.Robinson()})
if N_ROWS == 1: axes = axes[None, :]
if N_COLS == 1: axes = axes[:, None]

for row, side in enumerate(SIDES):
    for col, tag in enumerate(hero_datasets):
        ax = axes[row, col]
        eff = effective_α(attn_summary[tag], side)
        da  = to_xr(eff)
        # Plot WITHOUT x4c's auto-fig — we use the provided ax
        # x4c API: da.x.plot(..., ax=ax)
        try:
            _, _ = da.x.plot(cmap=SIDE_CMAPS[side], levels=LEVELS_EFF,
                             cbar_kwargs={'label': ''} if col == N_COLS - 1 else None,
                             title='', ax=ax, projection='Robinson',
                             extend='max')
        except Exception:
            # Fallback: direct cartopy
            ax.coastlines(linewidth=0.5)
            mesh = ax.pcolormesh(da['lon'], da['lat'], da.values,
                                  transform=ccrs.PlateCarree(),
                                  cmap=SIDE_CMAPS[side],
                                  vmin=0, vmax=0.5)
            if col == N_COLS - 1:
                plt.colorbar(mesh, ax=ax, shrink=0.7, label='effective α')
        ax.coastlines(linewidth=0.5)

        # Row label (side) on left of row 0
        if col == 0:
            ax.text(-0.08, 0.5, side, fontsize=14, fontweight='bold',
                    va='center', ha='right', rotation=90,
                    transform=ax.transAxes)
        # Column header (CO₂ ppm) on row 0
        if row == 0:
            ax.set_title(f'{tag} ({REGISTRY[tag]["co2"]} ppm)' +
                         (' [OOD]' if REGISTRY[tag]['holdout'] else ''),
                         fontsize=12)

fig.suptitle(f'Expert Focus Maps Across CO₂ Regimes (skip e{LEVEL}, time-averaged)',
             fontsize=14, y=0.995)
fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.97])
out_hero = os.path.join(out_dir, 'fig_B_hero_4experts_x_co2.pdf')
fig.savefig(out_hero, dpi=160, bbox_inches='tight')
fig.savefig(out_hero.replace('.pdf', '.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print(f'  saved: {os.path.basename(out_hero)}')


# ════════════════════════════════════════════════════════════════════════════
# FIGURE C — Bottleneck (‖E‖) vs skip e3 (α) for a representative dataset
# ════════════════════════════════════════════════════════════════════════════
print(f'\n[Figure C] Bottleneck vs skip e{LEVEL} consistency...')
ref_tag = hero_datasets[len(hero_datasets) // 2]
print(f'  using {ref_tag} as representative')

fig, axes = plt.subplots(N_ROWS, 2, figsize=(11, 3.0 * N_ROWS),
                          subplot_kw={'projection': ccrs.Robinson()})

for row, side in enumerate(SIDES):
    # Left: bottleneck ‖E_LW‖ × w_b
    eff_b = effective_b(bottleneck_summary[ref_tag], side)
    da_b  = to_xr(eff_b)
    eff_a = effective_α(attn_summary[ref_tag], side)
    da_a  = to_xr(eff_a)

    for col, (da, title) in enumerate([(da_b, f'Bottleneck  ‖{side}‖·w'),
                                         (da_a, f'Skip e{LEVEL}  α·w')]):
        ax = axes[row, col]
        try:
            _, _ = da.x.plot(cmap=SIDE_CMAPS[side],
                             title=title if row == 0 else '',
                             ax=ax, projection='Robinson')
        except Exception:
            ax.coastlines(linewidth=0.5)
            mesh = ax.pcolormesh(da['lon'], da['lat'], da.values,
                                  transform=ccrs.PlateCarree(),
                                  cmap=SIDE_CMAPS[side])
            plt.colorbar(mesh, ax=ax, shrink=0.7)
        ax.coastlines(linewidth=0.5)
        if col == 0:
            ax.text(-0.08, 0.5, side, fontsize=14, fontweight='bold',
                    va='center', ha='right', rotation=90,
                    transform=ax.transAxes)
        if row == 0:
            ax.set_title(title, fontsize=12)

fig.suptitle(f'{ref_tag} ({REGISTRY[ref_tag]["co2"]} ppm) — Bottleneck vs skip e{LEVEL} '
             f'(both effective contributions)', fontsize=13, y=0.995)
fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.97])
out_c = os.path.join(out_dir, f'fig_C_bottleneck_vs_skip_e{LEVEL}.pdf')
fig.savefig(out_c, dpi=160, bbox_inches='tight')
fig.savefig(out_c.replace('.pdf', '.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print(f'  saved: {os.path.basename(out_c)}')


print(f'\n✓ All figures saved to: {out_dir}')
print(f'  - fig_A_per_dataset/expert_focus_<ppm>.pdf  (7 dataset 2×2 grids)')
print(f'  - fig_B_hero_4experts_x_co2.pdf             ← paper hero figure')
print(f'  - fig_C_bottleneck_vs_skip_e{LEVEL}.pdf       (consistency check)')
