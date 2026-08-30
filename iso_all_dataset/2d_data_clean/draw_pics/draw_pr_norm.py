"""
draw_pr_norm.py — visualize the normalized pr_mask used for wet/dry routing.

Matches exactly the normalization formula in RegionalAttentionUNet.forward:
    pr_raw  = PRECL + PRECC           (m/s, last frame)
    pr_flat = pr_raw.reshape(B, -1)
    pr_min  = pr_flat.min(dim=1)      # per-sample
    pr_max  = pr_flat.max(dim=1)
    pr_norm = (pr_raw - pr_min) / (pr_max - pr_min + eps)   # ∈ [0, 1]

This pr_norm is what the wet/dry expert uses as a routing mask.

Output: ./images/pr_norm/
"""
import os, sys, glob
import numpy as np
import xarray as xr
import x4c
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data import REGISTRY, DATASET_DIR


OUT_DIR = './images/pr_norm'
os.makedirs(OUT_DIR, exist_ok=True)

MAX_T = 2400


# ── Load helpers ──────────────────────────────────────────────────────────
def load_pr_total(tag):
    """Returns PRECT = PRECL + PRECC in m/s as xr.DataArray (T, lat, lon)."""
    cfg  = REGISTRY[tag]
    base = os.path.join(DATASET_DIR, cfg['atm_dir'])
    pfx  = cfg['glob_key']
    def _lv(v):
        files = sorted(glob.glob(os.path.join(base, f'{pfx}.{v}_*_rgd.nc')))
        return x4c.open_mfdataset(files, parallel=False)[v]
    precl = _lv('PRECL').isel(time=slice(0, MAX_T))
    precc = _lv('PRECC').isel(time=slice(0, MAX_T))
    return precl + precc


def per_frame_minmax_norm(da):
    """Per-timestep min-max normalize to [0, 1]. da: (T, lat, lon).
    Returns: (T, lat, lon) with each timestep ∈ [0, 1]."""
    arr = da.values     # (T, H, W)
    T_ = arr.shape[0]
    flat = arr.reshape(T_, -1)
    mn = flat.min(axis=1, keepdims=True)    # (T, 1)
    mx = flat.max(axis=1, keepdims=True)
    norm_flat = (flat - mn) / (mx - mn + 1e-15)
    norm = norm_flat.reshape(arr.shape)
    return xr.DataArray(norm.astype(np.float32),
                        dims=da.dims, coords=da.coords,
                        attrs={'units': 'normalized [0,1]',
                               'long_name': 'pr per-sample min-max normalized'})


def to_mmday(da):
    out = da * 86400.0 * 1000.0
    out.attrs['units'] = 'mm/day'
    return out


# ── x4c plot wrapper ──────────────────────────────────────────────────────
def _x4c_plot(da, title, cmap, levels, label, save_path, **extra):
    try:
        fig, ax = da.x.plot(cmap=cmap, levels=levels,
                            cbar_kwargs={'label': label},
                            title=title, **extra)
        ax.coastlines()
        x4c.savefig(fig, save_path)
        plt.close(fig)
        print(f'  saved: {os.path.basename(save_path)}')
    except Exception as e:
        print(f'  [warn] {os.path.basename(save_path)} failed: {e}')


# ── Settings ──────────────────────────────────────────────────────────────
RAW_LEVELS  = np.linspace(0, 12, 25)    # mm/day raw
NORM_LEVELS = np.linspace(0, 1, 21)     # normalized [0, 1]
DIFF_LEVELS = np.linspace(-0.4, 0.4, 21)

RAW_CMAP    = 'viridis'
NORM_CMAP   = 'plasma'                  # high-contrast highlights the routing mask
DIFF_CMAP   = 'RdBu_r'


# ── Plot 1: side-by-side raw vs normalized for sample frames ────────────────
def plot_sample_frames(tag, raw_da, norm_da, idxs=(0, 250, 1000, 2000)):
    """One figure per ppm: top row raw, bottom row normalized, 4 columns for 4 time frames."""
    raw_mm = to_mmday(raw_da)
    co2 = REGISTRY[tag]['co2']
    for idx in idxs:
        if idx >= raw_da.shape[0]: continue
        # raw
        _x4c_plot(raw_mm.isel(time=idx),
                  title=f'{tag} ({co2} ppm) — pr raw, t={idx} (mm/day)',
                  cmap=RAW_CMAP, levels=RAW_LEVELS, label='pr (mm/day)',
                  save_path=os.path.join(OUT_DIR, f'{tag}_t{idx}_raw.pdf'),
                  extend='max')
        # normalized
        _x4c_plot(norm_da.isel(time=idx),
                  title=f'{tag} ({co2} ppm) — pr normalized [0,1], t={idx}',
                  cmap=NORM_CMAP, levels=NORM_LEVELS, label='pr_norm ∈ [0,1]',
                  save_path=os.path.join(OUT_DIR, f'{tag}_t{idx}_norm.pdf'),
                  extend='neither')


# ── Plot 2: climatology of normalized ─────────────────────────────────────
def plot_climatology(tag, raw_da, norm_da):
    co2 = REGISTRY[tag]['co2']
    label = f'{tag} ({co2} ppm)'
    _x4c_plot(to_mmday(raw_da).mean('time'),
              title=f'{label} — raw pr CLIMATOLOGY (mm/day)',
              cmap=RAW_CMAP, levels=RAW_LEVELS, label='pr (mm/day)',
              save_path=os.path.join(OUT_DIR, f'{tag}_climatology_raw.pdf'),
              extend='max')
    _x4c_plot(norm_da.mean('time'),
              title=f'{label} — normalized pr CLIMATOLOGY (mean of per-frame [0,1])',
              cmap=NORM_CMAP, levels=NORM_LEVELS, label='⟨pr_norm⟩',
              save_path=os.path.join(OUT_DIR, f'{tag}_climatology_norm.pdf'),
              extend='neither')


# ── Plot 3: 280 vs 840 normalized climatology difference ──────────────────
def plot_diff_norm(tag_hi, norm_hi, tag_lo, norm_lo):
    co2_hi, co2_lo = REGISTRY[tag_hi]['co2'], REGISTRY[tag_lo]['co2']
    diff = norm_hi.mean('time') - norm_lo.mean('time')
    _x4c_plot(diff,
              title=f'normalized-pr climatology DIFF: {tag_hi}({co2_hi}) − {tag_lo}({co2_lo})',
              cmap=DIFF_CMAP, levels=DIFF_LEVELS, label='Δ ⟨pr_norm⟩',
              save_path=os.path.join(OUT_DIR, f'diff_norm_clim_{co2_hi}-{co2_lo}.pdf'),
              extend='both')


# ── Plot 4: per-frame max/min/range curves ─────────────────────────────────
def plot_minmax_curves(samples):
    """Show how the raw min/max varies per frame across 2 datasets."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, key in zip(axes, ['max', 'mean']):
        for tag, raw_da in samples.items():
            arr = raw_da.values.reshape(raw_da.shape[0], -1)   # (T, H*W)
            val = arr.max(axis=1) * 86400 * 1000 if key == 'max' else arr.mean(axis=1) * 86400 * 1000
            ax.plot(val, label=f'{tag} ({REGISTRY[tag]["co2"]} ppm)',
                     alpha=0.85, linewidth=0.8)
        ax.set_ylabel(f'{key} pr (mm/day)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    axes[1].set_xlabel('time step')
    fig.suptitle('Per-frame pr statistics (across full timeseries)\n'
                 'Top: max value used for normalization denominator\n'
                 'Bottom: spatial mean (informational only)', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, 'per_frame_max_curve.pdf')
    plt.savefig(out_path, dpi=140); plt.close()
    print(f'  saved: per_frame_max_curve.pdf')


# ── Plot 5: histogram of per-frame max — show variability ────────────────
def plot_max_histogram(samples):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, key, title in zip(axes,
                                ['max', 'min'],
                                ['Per-frame MAX (denominator of normalization)',
                                 'Per-frame MIN (subtracted before normalization)']):
        for tag, raw_da in samples.items():
            arr = raw_da.values.reshape(raw_da.shape[0], -1)
            val = arr.max(axis=1) if key == 'max' else arr.min(axis=1)
            val_mmday = val * 86400 * 1000
            ax.hist(val_mmday, bins=50, alpha=0.5,
                    label=f'{tag} ({REGISTRY[tag]["co2"]} ppm)')
        ax.set_xlabel('mm/day'); ax.set_ylabel('# frames')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle('Per-frame pr extremes (informs how "stretched" the normalized values are)',
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUT_DIR, 'per_frame_extremes_histogram.pdf')
    plt.savefig(out_path, dpi=140); plt.close()
    print(f'  saved: per_frame_extremes_histogram.pdf')


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print(f'OUTPUT_DIR: {OUT_DIR}')
    print(f'MAX_T: {MAX_T}')
    print('=' * 60)

    raw_data = {}; norm_data = {}
    for tag in ['280ppm', '840ppm']:
        print(f'\n[load] {tag}...')
        raw = load_pr_total(tag)
        print(f'  pr_raw shape={raw.shape}, range=[{float(raw.min()):.2e}, {float(raw.max()):.2e}]')
        norm = per_frame_minmax_norm(raw)
        raw_data[tag]  = raw
        norm_data[tag] = norm

    print('\n[plot] sample frames (raw + norm side-by-side)')
    for tag in ['280ppm', '840ppm']:
        plot_sample_frames(tag, raw_data[tag], norm_data[tag])

    print('\n[plot] climatologies')
    plot_climatology('280ppm', raw_data['280ppm'], norm_data['280ppm'])
    plot_climatology('840ppm', raw_data['840ppm'], norm_data['840ppm'])

    print('\n[plot] 840 - 280 normalized climatology diff')
    plot_diff_norm('840ppm', norm_data['840ppm'], '280ppm', norm_data['280ppm'])

    print('\n[plot] per-frame min/max curves & histogram')
    plot_minmax_curves(raw_data)
    plot_max_histogram(raw_data)

    # Stats summary
    print('\n[stats] per-frame pr extremes (mm/day):')
    for tag in ['280ppm', '840ppm']:
        arr = raw_data[tag].values.reshape(raw_data[tag].shape[0], -1) * 86400 * 1000
        maxs = arr.max(axis=1); mins = arr.min(axis=1)
        print(f'  {tag}: max ∈ [{maxs.min():.1f}, {maxs.max():.1f}] mm/day, '
              f'mean={maxs.mean():.1f}; min always ≈ 0 (mean={mins.mean():.4f})')

    print('\n' + '=' * 60)
    print(f'All figures saved to: {OUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
