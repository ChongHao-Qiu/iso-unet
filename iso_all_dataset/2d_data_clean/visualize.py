"""
visualize.py — Result-only visualizations (plots computed from truth/pred).

Independent of model architecture; only needs (truth, pred) arrays to plot.
Covers **result visualizations** like "global R²/RMSE/MAE maps".

Kept separate from **model-internal visualizations** (e.g. attention / FiLM γ
/ climate state) — those live in each model's own file in its draw_inner() method.
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from metrics import _pixel_metrics_2d


def _save_spatial_nc(tag, truth, pred, save_dir):
    """Save raw truth_pred.nc + metrics_2d.nc."""
    H, W = truth.shape[1], truth.shape[2]
    lat  = np.linspace(-89, 89, H)
    lon  = np.linspace(1, 359, W)
    ds_tp = xr.Dataset(
        {'truth': (('time','lat','lon'), truth.astype(np.float32)),
         'pred':  (('time','lat','lon'), pred .astype(np.float32))},
        coords={'time': np.arange(truth.shape[0]), 'lat': lat, 'lon': lon},
    )
    ds_tp.to_netcdf(os.path.join(save_dir, f'{tag}_truth_pred.nc'))
    metrics_2d = _pixel_metrics_2d(truth, pred)
    ds_m = xr.Dataset(
        {m: (('lat','lon'), arr.astype(np.float32)) for m, arr in metrics_2d.items()},
        coords={'lat': lat, 'lon': lon},
    )
    ds_m.to_netcdf(os.path.join(save_dir, f'{tag}_metrics_2d.nc'))
    return lat, lon, metrics_2d


def _save_metric_pdfs(tag, lat, lon, metrics_2d, save_dir):
    """
    Same schema as exp-MCO-unet2d.py / fno2d.py etc.:
        predict_<tag>_R2_spatial.pdf
        predict_<tag>_RMSE_spatial.pdf
        predict_<tag>_MAE_spatial.pdf
    Uses x4c.plot's Robinson projection + coastlines.
    """
    import x4c
    plot_settings = {
        'R2': dict(
            cmap='RdBu_r', levels=np.linspace(0, 1, 21),
            cbar_kwargs={'ticks': np.linspace(0, 1, 6), 'label': r'$R^2$'},
            title=f'{tag}: $R^2$(Prediction, Truth)', extend='min'),
        'MSE': dict(
            cmap='RdYlGn_r', levels=np.linspace(0, 4, 21),
            cbar_kwargs={'ticks': np.linspace(0, 4, 5), 'label': r'MSE (‰$^2$)'},
            title=f'{tag}: MSE(Prediction, Truth)', extend='max'),
        'RMSE': dict(
            cmap='RdYlGn_r', levels=np.linspace(0, 4, 21),
            cbar_kwargs={'ticks': np.linspace(0, 4, 5), 'label': 'RMSE (‰)'},
            title=f'{tag}: RMSE(Prediction, Truth)', extend='max'),
        'MAE': dict(
            cmap='RdYlGn_r', levels=np.linspace(0, 3, 21),
            cbar_kwargs={'ticks': np.linspace(0, 3, 7), 'label': 'MAE (‰)'},
            title=f'{tag}: MAE(Prediction, Truth)', extend='max'),
    }
    for m, arr in metrics_2d.items():
        if m not in plot_settings: continue
        da = xr.DataArray(arr, dims=('lat', 'lon'),
                          coords={'lat': lat, 'lon': lon},
                          attrs={'long_name': m, 'units': '' if m == 'R2' else '‰'},
                          name=m)
        try:
            fig, ax = da.x.plot(**plot_settings[m])
            ax.coastlines()
            _pdf_path = os.path.join(save_dir, f'predict_{tag}_{m}_spatial.pdf')
            x4c.savefig(fig, _pdf_path)
            # Also save PNG for easy preview
            fig.savefig(_pdf_path[:-4] + '.png', bbox_inches='tight', dpi=140)
            plt.close(fig)
        except Exception as e:
            print(f'  [warn] {tag} {m} plot failed: {e}')


def draw_results(truth_pred_cache, output_dir, registry=None,
                 save_nc=True, save_pdf=True):
    """
    One-stop result-only visualization.

    Args:
        truth_pred_cache: dict of {tag: (truth_NxHxW, pred_NxHxW)} — matches eval.py's output
        output_dir:       root output directory (a spatial/ subdirectory is created)
        registry:         optional — used to flag OOD in prints; default None means no flagging
        save_nc:          whether to save NetCDF (truth_pred.nc + metrics_2d.nc)
        save_pdf:         whether to plot R²/RMSE/MAE x4c global map PDFs

    Side effects: writes into output_dir/spatial/:
        <tag>_truth_pred.nc                       N timesteps of truth & pred
        <tag>_metrics_2d.nc                       per-pixel R²/RMSE/MAE
        predict_<tag>_{R2,RMSE,MAE}_spatial.pdf   x4c Robinson global map
    """
    spatial_dir = os.path.join(output_dir, 'spatial')
    os.makedirs(spatial_dir, exist_ok=True)
    print(f'[draw_results] writing to {spatial_dir}/')
    for tag, (truth, pred) in truth_pred_cache.items():
        ood = registry and registry.get(tag, {}).get('holdout', False)
        flag = ' [OOD]' if ood else ''
        if save_nc:
            lat, lon, metrics_2d = _save_spatial_nc(tag, truth, pred, spatial_dir)
        else:
            H, W = truth.shape[1], truth.shape[2]
            lat = np.linspace(-89, 89, H); lon = np.linspace(1, 359, W)
            metrics_2d = _pixel_metrics_2d(truth, pred)
        if save_pdf:
            _save_metric_pdfs(tag, lat, lon, metrics_2d, spatial_dir)
        print(f'  {tag}{flag}: nc{"=Y" if save_nc else "=N"}, pdf{"=Y" if save_pdf else "=N"}')
    print(f'[draw_results] done')
