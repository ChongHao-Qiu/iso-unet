"""
metrics.py — computation only (regions, R², RMSE, MAE).

All visualization has been moved to:
    visualize.py                       — result-only viz (computed from truth/pred)
    iso_unet/baseline_<model>.py:
        draw_inner() method            — model-internal viz (attention / state, etc.)
"""
import numpy as np


# ── Regions: 5 lon × 2 lat = 10 regions, R01..R10 ────────────────────────────
def define_regions(H=90, W=180, n_lat=2, n_lon=5):
    """R01 is at the northwest corner (NH first, west to east)."""
    lat_edges     = np.linspace(0, H, n_lat + 1, dtype=int)
    lon_edges     = np.linspace(0, W, n_lon + 1, dtype=int)
    lat_edges_deg = np.linspace(-90, 90, n_lat + 1)
    lon_edges_deg = np.linspace(0, 360, n_lon + 1)
    regions, rid = {}, 1
    for i in range(n_lat - 1, -1, -1):
        for j in range(n_lon):
            regions[f'R{rid:02d}'] = {
                'id':        rid,
                'lat_slice': slice(lat_edges[i],   lat_edges[i+1]),
                'lon_slice': slice(lon_edges[j],   lon_edges[j+1]),
                'lat_deg':   (float(lat_edges_deg[i]),   float(lat_edges_deg[i+1])),
                'lon_deg':   (float(lon_edges_deg[j]),   float(lon_edges_deg[j+1])),
            }
            rid += 1
    return regions


# ── Tail (Q90) lat-weighted RMSE ─────────────────────────────────────────────
# Per-sample area-weighted top-(1-q) RMSE:
#   1. For each sample, compute err² = (truth - pred)²
#   2. Weighted by cos(lat) → use the area-weighted q-th quantile to find the threshold
#      (the top 10% of *area* exceeds the threshold, not "10% of grid cells")
#   3. Select pixels with e² ≥ threshold and compute the area-weighted RMSE
#   4. Average across samples
# Used to reveal model error on hard pixels (poles / land-sea boundaries / highlands) —
# ordinary RMSE is diluted by easy ocean pixels; Q90 spotlights the ~10% hard pixels.
def _q90_rmse_weighted(truth, pred, lat_values, q=0.9):
    """truth, pred: (N, H, W); lat_values: (H,) in degrees. Returns scalar."""
    err2 = (truth - pred) ** 2                                       # (N, H, W)
    w    = np.cos(np.deg2rad(lat_values)); w = w / w.mean()           # (H,)
    w_2d = np.broadcast_to(w[:, None], truth.shape[-2:])              # (H, W)
    w_flat  = w_2d.reshape(-1)                                        # (H*W,)
    e2_flat = err2.reshape(err2.shape[0], -1)                         # (N, H*W)
    per_sample = np.empty(e2_flat.shape[0])
    for n in range(e2_flat.shape[0]):
        e2 = e2_flat[n]
        # Area-weighted quantile: sort then interpolate using cumulative area
        order = np.argsort(e2)
        cum_w = np.cumsum(w_flat[order])
        thresh = np.interp(q * cum_w[-1], cum_w, e2[order])
        mask = e2 >= thresh
        # Area-weighted mean of selected err²
        per_sample[n] = np.sqrt(
            (e2[mask] * w_flat[mask]).sum() / w_flat[mask].sum()
        )
    return float(per_sample.mean())


# ── Overall + per-region weighted metrics ────────────────────────────────────
# Aligned with the official iso_unet Predictor.valid_metric():
#   1. Per-pixel metric (reduced over the time dimension)
#        MSE_pixel(h, w)   = mean_t (y - ŷ)²
#        RMSE_pixel(h, w)  = sqrt(MSE_pixel)
#        MAE_pixel(h, w)   = mean_t |y - ŷ|
#        R²_pixel(h, w)    = 1 - SS_res / SS_tot (per pixel over time)
#   2. Area-weighted spatial mean (using cos(lat))
#        global = Σ w(lat) · metric_pixel / Σ w(lat)
#
# Numerically different from the previous formulation (pooled weighted MSE → sqrt):
#   * official:    mean_pixels[ sqrt(MSE_pixel) ]              ← sqrt-then-mean
#   * old (mine):  sqrt( pool over all (t, h, w) of weighted MSE ) ← mean-then-sqrt
# Generally sqrt-then-mean > mean-then-sqrt (Jensen's inequality), but usually < 5% difference.
def _single_metrics(truth, pred, lat_values):
    """truth, pred: (N, H, W). lat_values: (H,) in degrees."""
    # 1. Per-pixel temporal stats
    err      = truth - pred                                          # (N, H, W)
    se       = err ** 2
    mse_px   = se.mean(axis=0)                                       # (H, W)
    rmse_px  = np.sqrt(mse_px)
    mae_px   = np.abs(err).mean(axis=0)                              # (H, W)
    # Per-pixel R²
    truth_t_mean = truth.mean(axis=0, keepdims=True)                 # (1, H, W)
    ss_res = se.sum(axis=0)                                          # (H, W)
    ss_tot = ((truth - truth_t_mean) ** 2).sum(axis=0)
    r2_px  = 1.0 - ss_res / np.where(ss_tot > 1e-12, ss_tot, np.nan) # NaN if flat

    # 2. Area-weighted spatial mean (NaN-safe so flat-pixel R² doesn't kill global mean)
    w    = np.cos(np.deg2rad(lat_values))                            # (H,)
    w_2d = np.broadcast_to(w[:, None], rmse_px.shape).copy()         # (H, W)
    def _wmean(m):
        valid = np.isfinite(m)
        if not valid.any(): return float('nan')
        return float((m[valid] * w_2d[valid]).sum() / w_2d[valid].sum())
    return {'R2':       _wmean(r2_px),
            'MSE':      _wmean(mse_px),
            'RMSE':     _wmean(rmse_px),
            'MAE':      _wmean(mae_px),
            'RMSE_q90': _q90_rmse_weighted(truth, pred, lat_values, q=0.9)}


def compute_metrics(truth, pred, lat_values, regions):
    out = {'overall': _single_metrics(truth, pred, lat_values), 'regions': {}}
    for name, info in regions.items():
        sub_t   = truth[:, info['lat_slice'], info['lon_slice']]
        sub_p   = pred [:, info['lat_slice'], info['lon_slice']]
        sub_lat = lat_values[info['lat_slice']]
        out['regions'][name] = _single_metrics(sub_t, sub_p, sub_lat)
    return out


# ── Per-pixel spatial metrics (used by visualize.draw_results) ─────────────
def _pixel_metrics_2d(truth, pred):
    n = truth.shape[0]
    truth_mean = truth.mean(axis=0, keepdims=True)
    ss_res = ((truth - pred) ** 2).sum(axis=0)
    ss_tot = ((truth - truth_mean) ** 2).sum(axis=0)
    r2   = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1e-12)
    mse  = ss_res / n
    rmse = np.sqrt(mse)
    mae  = np.abs(truth - pred).mean(axis=0)
    return {'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}


# ── Lat weights for training loss ───────────────────────────────────────────
def make_lat_weights(H=90):
    lat_values  = np.linspace(-89, 89, H)
    lat_weights = np.cos(np.deg2rad(lat_values))
    lat_weights /= lat_weights.mean()
    return lat_weights
