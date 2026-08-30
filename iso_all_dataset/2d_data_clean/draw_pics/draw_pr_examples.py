"""
draw_pr_examples.py — plot PRECL / PRECC / PRECT / conv_frac climatology maps,
comparing 280ppm (low CO2, Pliocene) vs 840ppm (high CO2, Miocene).

Uses x4c.plot (Robinson projection + coastlines).
Output: ./images/pr/

Visual evidence for the question "can PRECL and PRECC just be summed?".
"""
import os, sys, glob
import numpy as np
import xarray as xr
import x4c
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import REGISTRY + DATASET_DIR from 2d_data_clean
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data import REGISTRY, DATASET_DIR


OUT_DIR = './images/pr'
os.makedirs(OUT_DIR, exist_ok=True)

# Use the first N steps to compute climatology (avoids being too slow; iCESM is usually already quasi-equilibrium)
MAX_T = 2400


# ── Load helpers ────────────────────────────────────────────────────────────

def load_pr_pair(tag):
    """Load PRECL + PRECC for a tag. Returns (precl, precc) DataArrays in m/s."""
    cfg  = REGISTRY[tag]
    base = os.path.join(DATASET_DIR, cfg['atm_dir'])
    pfx  = cfg['glob_key']
    def _lv(v):
        files = sorted(glob.glob(os.path.join(base, f'{pfx}.{v}_*_rgd.nc')))
        assert files, f'No files for {tag}/{v}'
        return x4c.open_mfdataset(files, parallel=False)[v]
    precl = _lv('PRECL').isel(time=slice(0, MAX_T))
    precc = _lv('PRECC').isel(time=slice(0, MAX_T))
    return precl, precc


def to_mmday(da):
    """m/s → mm/day (humans-readable units for precipitation)."""
    out = da * 86400.0 * 1000.0     # m/s × s/day × mm/m
    out.attrs['units'] = 'mm/day'
    return out


def conv_fraction(precl, precc, eps=1e-15):
    """Convective fraction: PRECC / (PRECL + PRECC), masked where total ≈ 0."""
    total = precl + precc
    frac  = (precc / (total + eps)).where(total > 1e-10, np.nan)
    return frac


# ── Plot helper using x4c ──────────────────────────────────────────────────

def _x4c_plot(da, title, cmap, levels, units_label, save_path, **extra):
    """Single-panel x4c plot with consistent style."""
    da_clean = da.copy()
    if 'units' in da.attrs:
        units_label = f'{units_label}'
    try:
        fig, ax = da_clean.x.plot(
            cmap=cmap,
            levels=levels,
            cbar_kwargs={'label': units_label},
            title=title,
            **extra,
        )
        ax.coastlines()
        x4c.savefig(fig, save_path)
        plt.close(fig)
        print(f'  saved: {os.path.basename(save_path)}')
    except Exception as e:
        print(f'  [warn] {os.path.basename(save_path)} failed: {e}')


# ── Main loop ───────────────────────────────────────────────────────────────

def compute_all(tag):
    """Load + compute climatologies for a single tag."""
    print(f'\n[load] {tag}...')
    precl, precc = load_pr_pair(tag)
    precl_mm = to_mmday(precl)
    precc_mm = to_mmday(precc)
    prect_mm = precl_mm + precc_mm
    # Conv fraction in m/s domain (stays ∈ [0,1] regardless of units)
    cfrac    = conv_fraction(precl, precc)
    # Climatologies
    return dict(
        precl_clim = precl_mm.mean('time'),
        precc_clim = precc_mm.mean('time'),
        prect_clim = prect_mm.mean('time'),
        cfrac_clim = cfrac.mean('time'),
        precl_mm   = precl_mm,
        precc_mm   = precc_mm,
        prect_mm   = prect_mm,
        cfrac      = cfrac,
    )


# Per-variable plot settings (consistent across tags)
PRECIP_CMAP = 'viridis'
PRECIP_LEVELS = np.linspace(0, 12, 25)      # 0-12 mm/day, ~typical max
CFRAC_CMAP  = 'RdYlBu'                       # red = convective dominant, blue = stratiform
CFRAC_LEVELS = np.linspace(0, 1, 21)

# Diff colorscales
PRECIP_DIFF_LEVELS = np.linspace(-6, 6, 25)  # symmetric for 840-280
CFRAC_DIFF_LEVELS  = np.linspace(-0.4, 0.4, 21)


def plot_per_tag(tag, data):
    """Save 4 PDFs per tag: PRECL, PRECC, PRECT, conv_frac climatology."""
    co2 = REGISTRY[tag]['co2']
    label = f'{tag} ({co2} ppm)'
    _x4c_plot(data['precl_clim'],
              title=f'{label} — PRECL (large-scale precip), climatology',
              cmap=PRECIP_CMAP, levels=PRECIP_LEVELS, units_label='mm/day',
              save_path=os.path.join(OUT_DIR, f'{tag}_PRECL_clim.pdf'),
              extend='max')
    _x4c_plot(data['precc_clim'],
              title=f'{label} — PRECC (convective precip), climatology',
              cmap=PRECIP_CMAP, levels=PRECIP_LEVELS, units_label='mm/day',
              save_path=os.path.join(OUT_DIR, f'{tag}_PRECC_clim.pdf'),
              extend='max')
    _x4c_plot(data['prect_clim'],
              title=f'{label} — PRECT = PRECL + PRECC, climatology',
              cmap=PRECIP_CMAP, levels=PRECIP_LEVELS, units_label='mm/day',
              save_path=os.path.join(OUT_DIR, f'{tag}_PRECT_clim.pdf'),
              extend='max')
    _x4c_plot(data['cfrac_clim'],
              title=f'{label} — convective fraction = PRECC / PRECT, climatology',
              cmap=CFRAC_CMAP, levels=CFRAC_LEVELS, units_label='fraction',
              save_path=os.path.join(OUT_DIR, f'{tag}_conv_frac_clim.pdf'),
              extend='neither')


def plot_diff(tag_hi, data_hi, tag_lo, data_lo):
    """Save 4 PDFs: 840ppm - 280ppm difference for each metric."""
    co2_hi = REGISTRY[tag_hi]['co2']
    co2_lo = REGISTRY[tag_lo]['co2']
    diff_label = f'{tag_hi} - {tag_lo} ({co2_hi} - {co2_lo} ppm)'
    for key, prefix, levels, units, cmap in [
        ('precl_clim', 'PRECL',     PRECIP_DIFF_LEVELS, 'Δ mm/day', 'RdBu_r'),
        ('precc_clim', 'PRECC',     PRECIP_DIFF_LEVELS, 'Δ mm/day', 'RdBu_r'),
        ('prect_clim', 'PRECT',     PRECIP_DIFF_LEVELS, 'Δ mm/day', 'RdBu_r'),
        ('cfrac_clim', 'conv_frac', CFRAC_DIFF_LEVELS,  'Δ fraction', 'RdBu_r'),
    ]:
        diff = data_hi[key] - data_lo[key]
        _x4c_plot(diff,
                  title=f'{prefix} change: {diff_label}',
                  cmap=cmap, levels=levels, units_label=units,
                  save_path=os.path.join(OUT_DIR, f'diff_{prefix}_840-280.pdf'),
                  extend='both')


def plot_sample_frames(tag, data, n_frames=2, seed=0):
    """Show a couple of single time-frame snapshots — to illustrate variability."""
    co2 = REGISTRY[tag]['co2']
    rng = np.random.RandomState(seed)
    n_t = data['precl_mm'].shape[0]
    idxs = sorted(rng.choice(n_t, size=n_frames, replace=False).tolist())
    for idx in idxs:
        _x4c_plot(data['precl_mm'].isel(time=idx),
                  title=f'{tag} ({co2} ppm) — PRECL snapshot t={idx}',
                  cmap=PRECIP_CMAP, levels=PRECIP_LEVELS, units_label='mm/day',
                  save_path=os.path.join(OUT_DIR, f'{tag}_PRECL_t{idx}.pdf'),
                  extend='max')
        _x4c_plot(data['precc_mm'].isel(time=idx),
                  title=f'{tag} ({co2} ppm) — PRECC snapshot t={idx}',
                  cmap=PRECIP_CMAP, levels=PRECIP_LEVELS, units_label='mm/day',
                  save_path=os.path.join(OUT_DIR, f'{tag}_PRECC_t{idx}.pdf'),
                  extend='max')


def main():
    print('=' * 60)
    print(f'OUTPUT_DIR: {OUT_DIR}')
    print(f'MAX_T (frames per dataset): {MAX_T}')
    print('=' * 60)

    # 1. Compute climatologies for both
    data_280 = compute_all('280ppm')
    data_840 = compute_all('840ppm')

    # 2. Per-tag climatology plots (4 each)
    print('\n[plot] per-tag climatologies')
    plot_per_tag('280ppm', data_280)
    plot_per_tag('840ppm', data_840)

    # 3. Differences (840 - 280)
    print('\n[plot] 840 - 280 difference maps')
    plot_diff('840ppm', data_840, '280ppm', data_280)

    # 4. Sample frames (1-2 each, show variability)
    print('\n[plot] sample snapshots (illustrate variability)')
    plot_sample_frames('280ppm', data_280, n_frames=2, seed=0)
    plot_sample_frames('840ppm', data_840, n_frames=2, seed=0)

    # 5. Summary stats printout
    print('\n[stats] climatology summary (global mean):')
    for tag, d in [('280ppm', data_280), ('840ppm', data_840)]:
        co2 = REGISTRY[tag]['co2']
        gm = lambda da: float(da.mean(['lat', 'lon']))
        print(f'  {tag} ({co2:4d} ppm): '
              f'PRECL={gm(d["precl_clim"]):.2f}, '
              f'PRECC={gm(d["precc_clim"]):.2f}, '
              f'PRECT={gm(d["prect_clim"]):.2f} mm/day, '
              f'conv_frac={gm(d["cfrac_clim"]):.3f}')

    print('\n' + '=' * 60)
    print(f'All figures saved to: {OUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
