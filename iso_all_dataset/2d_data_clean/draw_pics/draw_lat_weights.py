"""
draw_lat_weights.py — visualize the lat-weight map used in weighted RMSE / Q90 RMSE.

Weight formula (from metrics.py:_single_metrics):
    w(lat) = cos(deg2rad(lat))
    w_normalized = w / w.mean()

Meaning: equatorial pixel weight ≈ 1.55, polar pixel ≈ 0.02.
Darker color = larger weight = pixel contributes more to RMSE.

Output: ./images/weight_rmse/
"""
import os
import numpy as np
import xarray as xr
import x4c
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


OUT_DIR = './images/weight_rmse'
os.makedirs(OUT_DIR, exist_ok=True)

H, W = 90, 180

# ⚠️ use the lat_values actually passed by eval.py to reflect the true situation (-90 to 90)
# Note: real NC data coords are -89 to 89, there is a bug here — but the paper numbers use this
# So this map reflects "how the metric is actually computed"
lat_values = np.linspace(-90, 90, H)            # ★ aligned with eval.py:185
lon_values = np.linspace(0, 360, W, endpoint=False) + 1.0
w_lat = np.cos(np.deg2rad(lat_values))          # (90,)
w_lat_norm = w_lat / w_lat.mean()               # normalized

# Broadcast to 2D (H, W)
w_2d = np.broadcast_to(w_lat_norm[:, None], (H, W)).copy()

print(f'Weight range: min={w_2d.min():.6f} ← polar weight = 0! (cos(±90°)=0)')
print(f'             max={w_2d.max():.4f}, mean={w_2d.mean():.4f}')
print(f'Lat |  weight  | meaning')
print('-' * 50)
for lat_check in [-89, -60, -30, 0, 30, 60, 89]:
    idx = np.abs(lat_values - lat_check).argmin()
    print(f'{lat_values[idx]:+5.0f}° | {w_lat_norm[idx]:7.4f}  '
          + ('← polar (small area)'   if abs(lat_check) > 80 else
             '← equator (large area)'  if abs(lat_check) < 10 else
             '← mid-latitude'))


# ── Wrap as xarray DataArray for x4c ────────────────────────────────────────
da = xr.DataArray(
    w_2d.astype(np.float32),
    dims=['lat', 'lon'],
    coords={'lat': lat_values, 'lon': lon_values},
    attrs={'long_name': 'cos(lat) area weight (normalized to mean=1)',
           'units': 'dimensionless'},
)


# ── Plot 1: full weight map (color = weight magnitude) ─────────────────────
print('\n[plot] lat_weight_2d.pdf — darker color, larger weight')
levels = np.linspace(0, 1.6, 17)
fig, ax = da.x.plot(cmap='YlOrBr', levels=levels,
                    cbar_kwargs={'label': 'cos(lat) weight (normalized, mean=1)'},
                    title='Lat-weight Map used in eval RMSE / Q90\n'
                          '(linspace(-90, 90, 90) → polar weight = 0)',
                    extend='neither')
ax.coastlines()
x4c.savefig(fig, os.path.join(OUT_DIR, 'lat_weight_2d.pdf'))
plt.close(fig)


# ── Plot 2: weight as fraction of equatorial maximum (highlights small areas) ──
print('[plot] lat_weight_log.pdf — log scale makes polar weights visible too')
da_log = np.log10(da + 1e-3)
da_log.attrs = {'long_name': 'log10(weight)'}
levels_log = np.linspace(-2, 0.3, 24)
fig, ax = da_log.x.plot(cmap='YlOrBr', levels=levels_log,
                        cbar_kwargs={'label': 'log10(cos(lat) weight)'},
                        title='Lat-weight Map (log scale)\n'
                              'Shows how dramatically polar pixels are downweighted')
ax.coastlines()
x4c.savefig(fig, os.path.join(OUT_DIR, 'lat_weight_log.pdf'))
plt.close(fig)


# ── Plot 3: 1D curve — weight vs latitude (cleanest view) ──────────────────
print('[plot] lat_weight_1d.pdf — 1D curve, most intuitive view')
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(lat_values, w_lat_norm, linewidth=2, color='#c8521b')
ax.fill_between(lat_values, 0, w_lat_norm, alpha=0.3, color='#f4a261')
ax.set_xlabel('Latitude (deg)')
ax.set_ylabel('Weight (cos(lat) normalized)')
ax.set_title('Per-latitude weight used in weighted RMSE / Q90 RMSE\n'
             '(equator ≈ 1.55, polar ≈ 0.02  → polar weight is ~75x smaller than equator)')
ax.grid(True, alpha=0.3)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='mean weight = 1')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlim(-90, 90)
ax.set_ylim(0, 1.7)
ax.legend(loc='upper right')
# Annotate extremes
ax.annotate(f'equator max = {w_lat_norm[len(w_lat_norm)//2]:.3f}',
            xy=(0, w_lat_norm[len(w_lat_norm)//2]),
            xytext=(15, 1.3),
            arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
            fontsize=10)
ax.annotate(f'polar min = {w_lat_norm[0]:.3f}',
            xy=(-89, w_lat_norm[0]),
            xytext=(-70, 0.5),
            arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
            fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'lat_weight_1d.pdf'), dpi=140)
plt.close(fig)


# ── Plot 4: cumulative area fraction (for understanding Q90 selection) ────
print('[plot] lat_weight_cumulative.pdf — geometric meaning of the Q90 selection threshold')
sorted_w = np.sort(w_lat_norm)[::-1]    # large → small
# repeat W times (every lat has W=180 cells)
all_w = np.concatenate([np.repeat(w_lat_norm[i], W) for i in range(H)])
sorted_all = np.sort(all_w)
cum_all = np.cumsum(sorted_all)
cum_frac = cum_all / cum_all[-1]
pixel_frac = np.arange(1, len(sorted_all) + 1) / len(sorted_all)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(pixel_frac * 100, cum_frac * 100, linewidth=2, color='#264653')
ax.fill_between(pixel_frac * 100, 0, cum_frac * 100, alpha=0.2, color='#2a9d8f')
ax.plot([0, 100], [0, 100], color='gray', linestyle=':', linewidth=1, label='equal-area line')
ax.axhline(10, color='red', linestyle='--', alpha=0.7,
           label='10% area (Q90 threshold)')
# Mark where 10% area is reached
idx10 = np.searchsorted(cum_frac, 0.10)
pct_pixel_for_10pct_area = pixel_frac[idx10] * 100
ax.axvline(pct_pixel_for_10pct_area, color='red', linestyle='--', alpha=0.7)
ax.scatter([pct_pixel_for_10pct_area], [10], color='red', s=80, zorder=5)
ax.annotate(f'10% area\n= about {pct_pixel_for_10pct_area:.1f}% of pixels\n(all concentrated at the poles)',
            xy=(pct_pixel_for_10pct_area, 10),
            xytext=(pct_pixel_for_10pct_area + 5, 25),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')
ax.set_xlabel('Cumulative fraction of pixels (sorted small → large weight) [%]')
ax.set_ylabel('Cumulative fraction of total area [%]')
ax.set_title('How quickly polar (small-weight) pixels accumulate area\n'
             '— it takes many polar pixels to make up 10% of the area')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'lat_weight_cumulative.pdf'), dpi=140)
plt.close(fig)


print(f'\n✓ All plots saved to: {OUT_DIR}')
print('  - lat_weight_2d.pdf       (Robinson projection, color = weight)')
print('  - lat_weight_log.pdf      (log scale, polar regions clearer)')
print('  - lat_weight_1d.pdf       (1D curve, most intuitive)')
print('  - lat_weight_cumulative.pdf (cumulative area curve, geometric meaning of Q90)')
