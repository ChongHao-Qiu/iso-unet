"""
data.py — Data loading (REGISTRY + load_xr + normalize), shared by single-frame and sequence models.

Logic is identical to the legacy scripts:
    * 7 CO2 datasets (4 ID + 3 OOD)
    * chrono_split: train first N%, val 70-80%, test 80-100%
    * normalize_inputs: per-channel z-score; channels in SKIP_NORM are not normalized
"""
import os, glob
import numpy as np
import xarray as xr
import torch
import x4c


# ── Dataset root ─────────────────────────────────────────────────────────────
# Override without editing this file by exporting ISO_DATASET_DIR, e.g.
#     export ISO_DATASET_DIR=/path/to/iCESM/data
# The directory must contain the per-scenario subdirs listed in REGISTRY below
# (iCESM-Pliocene_1x/atm, iCESM-Pliocene_350ppm/atm, ...).
DATASET_DIR = os.environ.get('ISO_DATASET_DIR', './data')

REGISTRY = {
    '280ppm': dict(co2=280, holdout=False, atm_dir='iCESM-Pliocene_1x/atm',
                   glob_key='b.e13.B1850C5CN.ne30_g16.icesm13_ihesp.midPliocene.1x.001.cam.h0'),
    '350ppm': dict(co2=350, holdout=False, atm_dir='iCESM-Pliocene_350ppm/atm',
                   glob_key='b.e13.B1850C5CN.ne30_g16.icesm131_ihesp.pW.Plio_350ppm.cam.h0'),
    '420ppm': dict(co2=420, holdout=False, atm_dir='iCESM-MCO_1.5x/atm',
                   glob_key='b.e13.B1850C5.ne16_g16.icesm131_d18O_fixer.Miocene.1.5xCO2.005.cam.h0'),
    '560ppm': dict(co2=560, holdout=False, atm_dir='iCESM-Pliocene_2x/atm',
                   glob_key='b.e13.B1850C5CN.ne30_g16.icesm13_ihesp.midPliocene.2x.001.cam.h0'),
    '400ppm': dict(co2=400, holdout=True,  atm_dir='iCESM-Pliocene_400ppm/atm',
                   glob_key='b.e13.B1850C5CN.ne30_g16.icesm131_ihesp.pW.Plio_400ppm.cam.h0'),
    '490ppm': dict(co2=490, holdout=True,  atm_dir='iCESM-Pliocene_490ppm/atm',
                   glob_key='b.e13.B1850C5CN.ne30_g16.icesm131_ihesp.pW.Plio_490ppm.cam.h0'),
    '840ppm': dict(co2=840, holdout=True,  atm_dir='iCESM-MCO_3x/atm',
                   glob_key='b.e13.B1850C5.ne16_g16.icesm131_d18O_fixer.Miocene.3xCO2.005.cam.h0'),
}

# ─────────────────────────────────────────────────────────────────────────────
# OOD presets — rewrite the holdout flag in REGISTRY at runtime to realize
# different OOD splits.
# Usage: train.py / eval.py reads the `ood_preset` field from YAML → calls
#        apply_ood_preset() which mutates REGISTRY's holdout flags. All
#        downstream code (datasets / visualize / draw_inner) reads
#        REGISTRY[tag]['holdout'] directly and picks up the change.
#
# Each preset = list of OOD tags (the remaining 4 datasets are ID).
# Sorted by CO2 (280→840) it's clear which are held out:
#   default:  ID 280 350 ___ 420 ___ 560 ___       OOD ___ ___ 400 ___ 490 ___ 840  (mixed interpolation + extrapolation)
#   low_ood:  ID ___ ___ ___ 420 ___ 490 560 840   OOD 280 350 400 ___ ___ ___ ___  (low-CO2 extrapolation)
#   high_ood: ID 280 350 400 420                   OOD ___ ___ ___ 490 560 ___ 840  (high-CO2 extrapolation)
#   mid_ood:  ID 280 ___ ___ ___ 400 490 ___ 840   OOD ___ 350 ___ 420 ___ 560 ___  (mid-range interpolation)
# ─────────────────────────────────────────────────────────────────────────────
OOD_PRESETS = {
    'default':  ['400ppm', '490ppm', '840ppm'],
    'low_ood':  ['280ppm', '350ppm', '400ppm'],
    'high_ood': ['490ppm', '560ppm', '840ppm'],
    'mid_ood':  ['350ppm', '420ppm', '560ppm'],
}


def apply_ood_preset(preset):
    """Mutate REGISTRY in place: set holdout=True for tags in this preset, False for rest.

    Returns: (preset_name, sorted_ood_tags, sorted_id_tags) for logging.
    """
    if preset not in OOD_PRESETS:
        raise ValueError(
            f"Unknown ood_preset '{preset}'. Choices: {list(OOD_PRESETS.keys())}"
        )
    ood_tags = set(OOD_PRESETS[preset])
    missing  = ood_tags - set(REGISTRY)
    if missing:
        raise ValueError(f"ood_preset '{preset}' references unknown tags: {missing}")
    for tag in REGISTRY:
        REGISTRY[tag]['holdout'] = (tag in ood_tags)
    # Sort tags by co2 ppm for log-friendly output
    id_sorted  = sorted([t for t in REGISTRY if not REGISTRY[t]['holdout']],
                        key=lambda t: REGISTRY[t]['co2'])
    ood_sorted = sorted(list(ood_tags), key=lambda t: REGISTRY[t]['co2'])
    return preset, ood_sorted, id_sorted

# Input feature sets
#   'pr'     = PRECL + PRECC (combined; aligned with observed total precipitation)
#   'precl'  = PRECL only (large-scale precipitation)
#   'precc'  = PRECC only (convective precipitation)
#   to split them apart → use split_pr / full_split
FEATURE_SETS = {
    'basic':      ['tas', 'pr'],                                                          # 2 ch
    'full':       ['tas', 'pr', 'PS', 'TMQ', 'QFLX', 'FLUT', 'LANDFRAC', 'aice'],         # 8 ch
    'split_pr':   ['tas', 'precl', 'precc'],                                              # 3 ch — basic with pr split
    'full_split': ['tas', 'precl', 'precc', 'PS', 'TMQ', 'QFLX', 'FLUT', 'LANDFRAC', 'aice'],   # 9 ch
}

# Channels NOT to z-score — routing mask must retain its [0, 1] physical meaning
# (under norm_mode='global', tas is also normalized)
SKIP_NORM = {'LANDFRAC', 'aice'}

# Legacy skip set — used under norm_mode='per_dataset' (tas is also skipped, matching pre-change behavior)
# This was originally to prevent per-dataset normalization from erasing CO2-induced temperature differences.
LEGACY_SKIP_NORM = {'tas', 'LANDFRAC', 'aice'}

# Reference dataset for global normalization stats (μ, σ computed from its train slice)
# 280ppm = pre-industrial baseline → inputs from other CO2 datasets become "offset relative to baseline"
REF_DATASET = '280ppm'


def get_input_features(input_set):
    """input_set: str ('basic'/'full') or list of channel names."""
    if isinstance(input_set, str):
        return list(FEATURE_SETS[input_set])
    return list(input_set)


def load_xr(tag, max_batch=3000, input_features=None):
    """
    Dynamically load atm variables + d18Op (target) based on input_features.
    If input_features is None, load everything by default (8 ch 'full' + d18Op).
    """
    if input_features is None:
        input_features = FEATURE_SETS['full']
    cfg  = REGISTRY[tag]
    base = os.path.join(DATASET_DIR, cfg['atm_dir'])
    pfx  = cfg['glob_key']

    def lv(vname):
        files = sorted(glob.glob(os.path.join(base, f'{pfx}.{vname}_*_rgd.nc')))
        assert files, f'No files for {tag}/{vname}'
        return x4c.open_mfdataset(files, parallel=False)[vname]

    ds = xr.Dataset()
    # Target is always loaded
    ds['d18Op'] = lv('d18Op')
    # Map standard channel names → CESM names
    if 'tas'   in input_features: ds['tas']   = lv('TS')
    if 'pr'    in input_features: ds['pr']    = lv('PRECL') + lv('PRECC')   # combined version
    if 'precl' in input_features: ds['precl'] = lv('PRECL')                  # separate (large-scale)
    if 'precc' in input_features: ds['precc'] = lv('PRECC')                  # separate (convective)
    for v in ('PS', 'TMQ', 'QFLX', 'FLUT', 'LANDFRAC', 'aice'):
        if v in input_features: ds[v] = lv(v)

    ds = ds.isel(time=slice(0, max_batch))
    if 'aice' in ds:
        ds['aice'] = ds['aice'].fillna(0.0)
    return ds


def xr_to_tensors(ds_xr, input_features):
    """Stack input_features channels in order. Returns (x, y) tensors."""
    arrs = [ds_xr[v].values.astype(np.float32, copy=False) for v in input_features]
    x = np.stack(arrs, axis=1)                    # (T, n_inputs, H, W)
    y = ds_xr['d18Op'].values[:, None]            # (T, 1, H, W)
    return (torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32))


def chrono_split(n_time, train_frac):
    """Return (sl_train, sl_val, sl_test) — val/test fixed at 70-80% / 80-100%."""
    n_train      = int(n_time * train_frac)
    n_val_start  = int(n_time * 0.70)
    n_test_start = int(n_time * 0.80)
    return (slice(0,            n_train),
            slice(n_val_start,  n_test_start),
            slice(n_test_start, n_time))


def normalize_inputs(x_train, x_val, x_test, features, skip=None):
    """
    Per-dataset z-score (μ, σ from THIS dataset's train slice). Channels in skip are left untouched.
    In-place modification. Returns stats dict.

    Difference vs normalize_inputs_with_stats:
        per-dataset    : each dataset normalized separately (legacy behavior) → CO2-induced input differences are flattened
        with_stats     : normalize all datasets using *externally provided* ref_stats (e.g. computed from 280ppm) (new)
    """
    skip = skip if skip is not None else SKIP_NORM
    stats = {}
    for c, vname in enumerate(features):
        if vname in skip:
            stats[vname] = 'skipped'; continue
        mu = float(x_train[:, c].mean())
        sd = float(x_train[:, c].std()) + 1e-8
        for xt in (x_train, x_val, x_test):
            xt[:, c] = (xt[:, c] - mu) / sd
        stats[vname] = {'mu': mu, 'sd': sd}
    return stats


def compute_ref_stats(raw_ref, input_features, train_frac, skip=None):
    """
    Compute global normalization stats from REF dataset's train slice.
    Returns: {feature: {'mu': float, 'sd': float}}.
    """
    skip = skip if skip is not None else SKIP_NORM
    x, _ = xr_to_tensors(raw_ref, input_features)
    n_train = int(x.shape[0] * train_frac)
    x_train = x[:n_train]
    stats = {}
    for c, vname in enumerate(input_features):
        if vname in skip: continue
        mu = float(x_train[:, c].mean())
        sd = float(x_train[:, c].std()) + 1e-8
        stats[vname] = {'mu': mu, 'sd': sd}
    return stats


def normalize_inputs_with_stats(x_train, x_val, x_test, features, ref_stats, skip=None):
    """
    Apply *pre-computed* ref_stats to a single dataset's splits (global normalization).
    Use this for cross-dataset training: all datasets get normalized by the SAME (μ, σ).

    Returns stats dict (mirrors normalize_inputs format) — 'mu'/'sd' fields come from ref_stats
    (so the log shows what was applied), not from this dataset's local distribution.
    """
    skip = skip if skip is not None else SKIP_NORM
    stats = {}
    for c, vname in enumerate(features):
        if vname in skip:
            stats[vname] = 'skipped'; continue
        if vname not in ref_stats:
            raise KeyError(
                f'ref_stats missing channel "{vname}". '
                f'Available: {list(ref_stats.keys())}'
            )
        mu = ref_stats[vname]['mu']
        sd = ref_stats[vname]['sd']
        for xt in (x_train, x_val, x_test):
            xt[:, c] = (xt[:, c] - mu) / sd
        stats[vname] = {'mu': mu, 'sd': sd}
    return stats


def format_norm_log(stats, tag, co2, holdout, train_size, val_size, test_size):
    """Print a uniform per-dataset summary line."""
    norm_str = ', '.join([f'{v}={s["mu"]:+.2e}±{s["sd"]:.2e}'
                          for v, s in stats.items() if s != 'skipped'])
    skip_str = ', '.join([v for v, s in stats.items() if s == 'skipped'])
    flag = ' [OOD]' if holdout else ''
    return (f'  {tag} ({co2:4d} ppm): '
            f'train={train_size:4d}  val={val_size:4d}  test={test_size:4d}{flag}\n'
            f'      skip-norm: [{skip_str}]\n'
            f'      z-score  : [{norm_str}]')
