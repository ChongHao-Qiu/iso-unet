"""
draw_feature_rect.py

Climate-analog rectangular map generator.

Compared with the old Cartopy/Robinson version:
  - No Cartopy
  - No Robinson projection
  - No pcolormesh
  - No cyclic-point seam handling
  - Uses imshow on a rectangular lon-lat grid
  - Much less likely to create PDF/vector/masking artifacts

Keeps the same high-level CLI interface as the old draw_feature.py.
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import lightning as L

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. Feature layer registry
# ============================================================

MODEL_FEATURE_REGISTRY = {
    "iso_unet.baseline_iso_unet.IsoUNetBaseline":
        dict(path="net.up1"),
    "iso_unet.baseline_iso_unet_v2.IsoUNetBaselineV2":
        dict(path="net.up1"),
    "iso_unet.baseline_unet2d_vanilla.UNet2DVanillaBaseline":
        dict(path="net.up1"),
    "iso_unet.baseline_unet2d.UNet2DBaseline":
        dict(path="net.up1"),
    "iso_unet.baseline_attunet.AttentionUNetBaseline":
        dict(path="net.Up_conv2"),
    "iso_unet.baseline_canet.CANetBaseline":
        dict(path="net.scale_att"),
    "iso_unet.baseline_dcsaunet.DCSAUNetBaseline":
        dict(path="net.up4"),
    "iso_unet.baseline_unetpp.UNetPlusPlusBaseline":
        dict(path="net.nested.x_0_4"),
    "iso_unet.baseline_transunet.TransUNetBaseline":
        dict(path="net.decoder_blocks.3"),
    "iso_unet.baseline_uno.UNOBaseline":
        dict(path="net.fc1", perm="BHWC"),
    "iso_unet.model.FNO2d":
        dict(path="fc1", perm="BHWC"),
    "iso_unet.baseline_unet3d_temporal.TemporalUNet3DBaseline":
        dict(path="net.dec1", squeeze_t=True),
    "iso_unet.baseline_segformer.SegFormerBaseline":
        dict(path="seg.base_model.decode_head.batch_norm"),
    "iso_unet.baseline_climax.ClimaXBaseline":
        dict(
            path="arch.norm",
            token_grid=(18, 36),
            token_T=12,
            token_aggT="mean",
        ),
    "iso_unet.baseline_stormer.StormerBaseline":
        dict(
            path="arch.blocks.-1",
            token_grid=(18, 36),
            token_T=1,
        ),
}


DEFAULT_REGIONS = {
    "Nino3.4": (0, 215),
    "Nino1+2": (-5, 275),
    "Nino3": (0, 240),
    "Nino4": (0, 185),
    "US_West_Coast": (40, 239),
    "Indian_Monsoon": (17, 80),
    "EA_Monsoon": (32, 117),
    "Tibetan_Plateau": (34, 90),
    "Mediterranean": (37, 20),
    "Tropical_IO": (0, 75),
    "Amazon": (-7, 310),
    "Sahara": (22, 15),
    "Greenland": (71, 325),
    "Antarctica": (-80, 0),
    "Arctic": (80, 180),
}


# ============================================================
# 2. CLI: keep old interface
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Climate-analog rectangular map generator"
    )

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out_subdir", type=str, default="feature_analog_rect")
    parser.add_argument("--dataset", type=str, default="420ppm")
    parser.add_argument("--sample_indices", type=str, default="0,100,200,400")
    parser.add_argument("--n_avg", type=int, default=1)
    parser.add_argument("--regions", type=str, default="all")
    parser.add_argument("--queries", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--feature_layer",
        type=str,
        default="auto",
        help='auto or manual module path, e.g. "net.up1"',
    )

    parser.add_argument("--center", dest="center", action="store_true", default=True)
    parser.add_argument("--no_center", dest="center", action="store_false")

    parser.add_argument("--sharpen", type=float, default=4.0)

    parser.add_argument(
        "--zonal_demean",
        dest="zonal_demean",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no_zonal_demean",
        dest="zonal_demean",
        action="store_false",
    )

    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--cmap", type=str, default="RdBu_r")
    parser.add_argument("--top_pct", type=float, default=100.0)

    # New but harmless: better rectangular figure control
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--save_pdf", action="store_true", default=True)
    parser.add_argument("--no_save_pdf", dest="save_pdf", action="store_false")
    parser.add_argument("--show_grid", action="store_true", default=False)

    return parser


# ============================================================
# 3. Small utilities
# ============================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def resolve_submodule(model, dot_path):
    """
    Resolve paths like:
      net.up1
      net.decoder_blocks.3
      arch.blocks.-1
      net.nested.x_0_4
    """
    m = model

    for part in dot_path.split("."):
        try:
            idx = int(part)
            m = m[idx]
            continue
        except ValueError:
            pass

        if hasattr(m, part):
            m = getattr(m, part)
        else:
            try:
                m = m[part]
            except Exception as e:
                raise AttributeError(
                    f'Cannot resolve "{part}" on {type(m).__name__}; '
                    f'full path="{dot_path}"'
                ) from e

    return m


def normalize_feature_shape(
    feat,
    target_H,
    target_W,
    *,
    perm=None,
    squeeze_t=False,
    token_grid=None,
    token_T=None,
    token_aggT="mean",
):
    """
    Convert hooked feature into (B, C, H, W).
    """

    if perm == "BHWC":
        feat = feat.permute(0, 3, 1, 2).contiguous()

    if squeeze_t and feat.dim() == 5:
        feat = feat[:, :, -1].contiguous()

    if token_grid is not None:
        H_p, W_p = token_grid

        if feat.dim() != 3:
            raise RuntimeError(
                f"token feature should be 3D (B*T, L, D), got {feat.shape}"
            )

        BT, L, D = feat.shape

        if L != H_p * W_p:
            raise RuntimeError(
                f"token length mismatch: L={L}, H_p*W_p={H_p * W_p}"
            )

        T = int(token_T or 1)

        if BT % T != 0:
            raise RuntimeError(f"BT={BT} is not divisible by token_T={T}")

        B = BT // T
        feat = feat.reshape(B, T, L, D)

        if token_aggT == "mean":
            feat = feat.mean(dim=1)
        elif token_aggT == "last":
            feat = feat[:, -1]
        else:
            raise ValueError(f"Unknown token_aggT={token_aggT}")

        feat = feat.reshape(B, H_p, W_p, D)
        feat = feat.permute(0, 3, 1, 2).contiguous()

    if feat.dim() != 4:
        raise RuntimeError(f"Feature after adapters is not 4D: {feat.shape}")

    if feat.shape[-2:] != (target_H, target_W):
        feat = F.interpolate(
            feat.float(),
            size=(target_H, target_W),
            mode="bilinear",
            align_corners=False,
        )

    return feat


def make_lat_lon_grids(H, W):
    """
    Keep same convention as original script:
      lat: -89 to 89
      lon: 1, 3, 5, ..., 359 for W=180
    """
    lat = np.linspace(-89, 89, H)
    lon = np.linspace(0, 360, W, endpoint=False) + 360.0 / W / 2.0
    return lat, lon


def latlon_to_idx(lat, lon, lat_values, lon_values):
    lon = lon % 360.0
    h = int(np.argmin(np.abs(lat_values - lat)))
    w = int(np.argmin(np.abs(lon_values - lon)))
    return h, w


def safe_slug(name, max_len=32):
    return (
        name.lower()
        .replace("+", "p")
        .replace(".", "")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )[:max_len]


# ============================================================
# 4. Query construction
# ============================================================

def build_queries(args, lat_values, lon_values):
    queries = []

    if args.regions:
        if args.regions.strip().lower() == "all":
            selected = list(DEFAULT_REGIONS.keys())
        else:
            selected = [x.strip() for x in args.regions.split(",") if x.strip()]

        for name in selected:
            if name not in DEFAULT_REGIONS:
                print(f'[warn] unknown region "{name}", skipped')
                continue

            lat, lon = DEFAULT_REGIONS[name]
            lon = lon % 360.0
            h, w = latlon_to_idx(lat, lon, lat_values, lon_values)

            queries.append(
                {
                    "name": name,
                    "lat": float(lat),
                    "lon": float(lon),
                    "pixel_ids": [(h, w)],
                }
            )

    if args.queries:
        for q in args.queries.split(";"):
            q = q.strip()
            if not q:
                continue

            try:
                lat, lon = map(float, q.split(","))
            except Exception:
                print(f'[warn] bad query "{q}", skipped')
                continue

            lon_norm = lon % 360.0
            h, w = latlon_to_idx(lat, lon_norm, lat_values, lon_values)

            queries.append(
                {
                    "name": f"pt_{lat:+.0f}_{lon:+.0f}",
                    "lat": float(lat),
                    "lon": float(lon_norm),
                    "pixel_ids": [(h, w)],
                }
            )

    if not queries:
        raise RuntimeError("No valid queries. Use --regions or --queries.")

    return queries


# ============================================================
# 5. Analog map computation
# ============================================================

def compute_analog_map(
    feat_tensor,
    pixel_ids,
    *,
    center=True,
    sharpen=1.0,
    zonal_demean=False,
    top_pct=100.0,
):
    """
    feat_tensor: (C, H, W)

    Returns:
      sim: (H, W) numpy array
    """

    if not isinstance(feat_tensor, torch.Tensor):
        raise TypeError("feat_tensor must be a torch.Tensor")

    feat = feat_tensor.float()

    C, H, W = feat.shape

    if zonal_demean:
        # Remove per-latitude zonal mean.
        feat = feat - feat.mean(dim=2, keepdim=True)

    flat = feat.reshape(C, H * W)

    if center:
        # Remove global spatial feature mean per channel.
        flat = flat - flat.mean(dim=1, keepdim=True)

    flat_norm = F.normalize(flat, dim=0, eps=1e-8)

    q_flat_ids = []
    for h, w in pixel_ids:
        if not (0 <= h < H and 0 <= w < W):
            raise ValueError(f"query pixel {(h, w)} outside feature map {(H, W)}")
        q_flat_ids.append(h * W + w)

    q_feat = flat_norm[:, q_flat_ids].mean(dim=1)
    q_feat = F.normalize(q_feat, dim=0, eps=1e-8)

    sim = flat_norm.T @ q_feat
    sim = sim.reshape(H, W)

    if sharpen != 1.0:
        sim = sim.sign() * sim.abs().pow(sharpen)

    sim = sim.cpu().numpy()

    if top_pct < 100.0:
        top_pct = float(top_pct)
        if not (0 < top_pct <= 100):
            raise ValueError("--top_pct should be in (0, 100]")
        threshold = np.nanquantile(sim, 1.0 - top_pct / 100.0)
        sim = np.where(sim >= threshold, sim, np.nan)

    return sim


# ============================================================
# 6. Rectangular plotting
# ============================================================

_CUSTOM_CMAPS = {
    # blue -> purple -> red, no white middle, smooth in perceptual space
    "bpr":  [(0.05, 0.20, 0.65), (0.50, 0.10, 0.60), (0.80, 0.10, 0.20)],
    # deeper saturated version
    "bpr2": [(0.00, 0.10, 0.80), (0.45, 0.00, 0.55), (0.85, 0.00, 0.10)],
    # blue -> magenta -> red
    "bmr":  [(0.10, 0.30, 0.85), (0.75, 0.10, 0.75), (0.90, 0.15, 0.20)],
    # user picked: medium blue -> pale icy cyan-white (shifted to sim=-0.1
    # = cmap position 0.45) -> RdBu_r deep red at sim=+1
    "bir": [
        (0.00, (56/255, 105/255, 185/255)),   # sim=-1   → blue
        (0.45, (210/255, 235/255, 240/255)),  # sim=-0.1 → pale ice
        (1.00, (178/255,  24/255,  43/255)),  # sim=+1   → RdBu_r red
    ],
}


def prepare_cmap(cmap_name, mask_color="lightgrey"):
    from matplotlib.colors import LinearSegmentedColormap
    if cmap_name in _CUSTOM_CMAPS:
        cmap = LinearSegmentedColormap.from_list(
            cmap_name, _CUSTOM_CMAPS[cmap_name], N=256).copy()
    else:
        cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=mask_color, alpha=1.0)
    return cmap


# Pre-load coastline segments from cartopy's natural_earth shapefile.
# We extract coordinates ONCE and just plot them as line segments on plain
# matplotlib axes -- this gives us geographic context without using cartopy's
# projection/GeoAxes machinery (which was the source of the old artifacts).
_COASTLINE_SEGMENTS = None


def _load_coastline_segments():
    """Load Natural Earth coastlines as a list of (xs, ys) arrays in
    longitude [0, 360], latitude [-90, 90]. Lines that cross the antimeridian
    are split so they don't draw long horizontal connectors."""
    global _COASTLINE_SEGMENTS
    if _COASTLINE_SEGMENTS is not None:
        return _COASTLINE_SEGMENTS
    try:
        from cartopy.io.shapereader import natural_earth, Reader
        shp = natural_earth(resolution="110m", category="physical",
                            name="coastline")
        segments = []
        for geom in Reader(shp).geometries():
            geoms = (geom.geoms if hasattr(geom, "geoms") else [geom])
            for line in geoms:
                xs, ys = line.xy
                xs = np.asarray(xs)
                ys = np.asarray(ys)
                # convert from [-180, 180] to [0, 360]
                xs = np.where(xs < 0, xs + 360.0, xs)
                # split at any jump > 180° (anti-meridian crossings)
                jump = np.abs(np.diff(xs)) > 180.0
                splits = np.where(jump)[0] + 1
                xs_segs = np.split(xs, splits)
                ys_segs = np.split(ys, splits)
                for sx, sy in zip(xs_segs, ys_segs):
                    if len(sx) > 1:
                        segments.append((sx, sy))
        _COASTLINE_SEGMENTS = segments
    except Exception as e:
        print(f"[warn] coastlines unavailable ({type(e).__name__}: {e}) — "
              f"continuing without coastline overlay.")
        _COASTLINE_SEGMENTS = []
    return _COASTLINE_SEGMENTS


def _draw_coastlines(ax, linewidth=0.5, color="k", alpha=0.85, zorder=4):
    """Overlay coastline segments on a plain matplotlib axis with lon in
    [0, 360], lat in [-90, 90]."""
    for xs, ys in _load_coastline_segments():
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha,
                zorder=zorder)


def plot_rect_single(
    sim,
    q,
    save_path_png,
    *,
    lat_values,
    lon_values,
    vmin=-1.0,
    vmax=1.0,
    cmap="RdBu_r",
    dpi=220,
    show_grid=False,
    save_pdf=True,
):
    """
    Draw one rectangular lon-lat analog map.

    This uses imshow instead of pcolormesh.
    The plotted domain is:
      lon: 0 to 360
      lat: -90 to 90
    """

    cmap_obj = prepare_cmap(cmap)

    fig, ax = plt.subplots(figsize=(11.5, 5.2))

    masked = np.ma.masked_invalid(sim)

    im = ax.imshow(
        masked,
        origin="lower",
        extent=[0, 360, -90, 90],
        interpolation="nearest",
        aspect="auto",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )

    # Geographic context: coastlines drawn as plain matplotlib line segments
    # (pre-loaded from cartopy's Natural Earth shapefile; no GeoAxes).
    _draw_coastlines(ax, linewidth=0.55, color="k", alpha=0.9, zorder=4)

    ax.scatter(
        [q["lon"]],
        [q["lat"]],
        marker="*",
        s=180,
        c="yellow",
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
    )

    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_yticks(np.arange(-90, 91, 30))

    if show_grid:
        ax.grid(linewidth=0.4, alpha=0.35)

    ax.set_title(f"Climate analog: {q['name']}", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("cosine similarity")

    fig.tight_layout()

    fig.savefig(save_path_png, dpi=dpi, bbox_inches="tight")

    if save_pdf:
        save_path_pdf = save_path_png.replace(".png", ".pdf")
        fig.savefig(save_path_pdf, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


def plot_rect_combined(
    query_list,
    sim_list,
    save_path_png,
    *,
    lat_values,
    lon_values,
    dataset_label,
    n_cols=3,
    vmin=-1.0,
    vmax=1.0,
    cmap="RdBu_r",
    dpi=220,
    show_grid=False,
    save_pdf=True,
):
    """
    Draw multiple rectangular maps in one grid.
    """

    cmap_obj = prepare_cmap(cmap)

    N = len(query_list)
    n_rows = int(np.ceil(N / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 2.9 * n_rows),
        squeeze=False,
    )

    last_im = None

    for i in range(n_rows * n_cols):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r][c]

        if i >= N:
            ax.axis("off")
            continue

        q = query_list[i]
        sim = sim_list[i]

        masked = np.ma.masked_invalid(sim)

        last_im = ax.imshow(
            masked,
            origin="lower",
            extent=[0, 360, -90, 90],
            interpolation="nearest",
            aspect="auto",
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        # Thinner coastlines for grid panels
        _draw_coastlines(ax, linewidth=0.35, color="k", alpha=0.85, zorder=4)

        ax.scatter(
            [q["lon"]],
            [q["lat"]],
            marker="*",
            s=90,
            c="yellow",
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
        )

        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)

        ax.set_xticks([0, 120, 240, 360])
        ax.set_yticks([-60, 0, 60])

        if show_grid:
            ax.grid(linewidth=0.35, alpha=0.3)

        ax.set_title(q["name"], fontsize=10)

        if r == n_rows - 1:
            ax.set_xlabel("Lon")
        if c == 0:
            ax.set_ylabel("Lat")

    fig.suptitle(
        f"Decoder Feature Climate Analogs ({dataset_label})",
        fontsize=13,
        y=0.995,
    )

    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes,
            shrink=0.82,
            pad=0.012,
            location="right",
        )
        cbar.set_label("cosine similarity to query")

    fig.savefig(save_path_png, dpi=dpi, bbox_inches="tight")

    if save_pdf:
        save_path_pdf = save_path_png.replace(".png", ".pdf")
        fig.savefig(save_path_pdf, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.ckpt)
    ckpt_dir = os.path.dirname(ckpt_path)
    config_path = args.config or os.path.join(ckpt_dir, "config.yaml")
    out_dir = os.path.join(ckpt_dir, args.out_subdir)

    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "feature_analog_rect.log")
    log_file = open(log_path, "w", buffering=1)

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"[{datetime.datetime.now()}] feature_analog_rect.log -> {log_path}")
    print(f"CKPT       = {ckpt_path}")
    print(f"CONFIG     = {config_path}")
    print(f"OUTPUT_DIR = {out_dir}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data import (
        REGISTRY,
        apply_ood_preset,
        REF_DATASET,
        compute_ref_stats,
        get_input_features,
        load_xr,
    )
    from datasets import build_splits
    from models import build_model, get_dataset_kind
    from metrics import make_lat_weights

    L.seed_everything(args.seed, workers=True)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config found at {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("train_frac", 0.10)
    cfg.setdefault("seq_len", 12)
    cfg.setdefault("max_batch", 3000)
    cfg.setdefault("input_set", "basic")
    cfg.setdefault("batch_size", 8)
    cfg.setdefault("ood_preset", "default")
    cfg.setdefault("norm_mode", "global")
    cfg.setdefault("norm_ref", REF_DATASET)
    cfg.setdefault("lat_weighted", True)

    if args.dataset not in REGISTRY:
        raise ValueError(f"Unknown dataset {args.dataset}; choose from {list(REGISTRY)}")

    input_features = cfg.get("input_features") or get_input_features(cfg["input_set"])
    n_inputs = len(input_features)

    seq_len = int(cfg["seq_len"])
    train_frac = float(cfg["train_frac"])
    dataset_kind = get_dataset_kind(cfg["model_class"], cfg.get("dataset_kind"))

    apply_ood_preset(cfg["ood_preset"])

    print(f"DATASET = {args.dataset} ({REGISTRY[args.dataset]['co2']} ppm)")
    print(f"MODEL   = {cfg['model_class']}")

    print("\nLoading data...")
    raw_chosen = load_xr(
        args.dataset,
        max_batch=cfg["max_batch"],
        input_features=input_features,
    )

    ref_stats = None
    if cfg["norm_mode"] == "global":
        if "norm_stats" in cfg and cfg["norm_stats"]:
            ref_stats = cfg["norm_stats"]
        else:
            raw_ref = load_xr(
                cfg["norm_ref"],
                max_batch=cfg["max_batch"],
                input_features=input_features,
            )
            ref_stats = compute_ref_stats(raw_ref, input_features, train_frac)

    tr, va, te_a, te_b, stats, split_log = build_splits(
        args.dataset,
        raw_chosen,
        train_frac,
        seq_len,
        input_features,
        dataset_kind,
        ref_stats=ref_stats,
    )

    print(split_log)

    test_ds = te_a

    lat_weights_t = (
        torch.as_tensor(make_lat_weights(), dtype=torch.float32)
        if cfg["lat_weighted"]
        else None
    )

    model_kwargs = dict(cfg["model_kwargs"])
    model_kwargs["n_inputs"] = n_inputs

    model = build_model(
        cfg["model_class"],
        model_kwargs=model_kwargs,
        weights=lat_weights_t,
    )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        print(f"[warn] missing={len(missing)}, unexpected={len(unexpected)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Loaded model on {device}")

    model_class = cfg.get("model_class", "")

    if args.feature_layer == "auto":
        if model_class not in MODEL_FEATURE_REGISTRY:
            raise KeyError(
                f'Model "{model_class}" not in MODEL_FEATURE_REGISTRY. '
                f"Use --feature_layer manually."
            )

        feat_cfg = dict(MODEL_FEATURE_REGISTRY[model_class])
        feat_path = feat_cfg.pop("path")
    else:
        feat_path = args.feature_layer
        feat_cfg = {
            k: v
            for k, v in MODEL_FEATURE_REGISTRY.get(model_class, {}).items()
            if k != "path"
        }

    print(f"Hook target = {feat_path}")
    print(f"Adapters    = {feat_cfg if feat_cfg else '(none)'}")

    target_module = resolve_submodule(model, feat_path)

    H_TARGET = 90
    W_TARGET = 180
    lat_values, lon_values = make_lat_lon_grids(H_TARGET, W_TARGET)

    feat_cache = []

    def hook_fn(module, inp, out):
        feat_cache.append(out.detach())

    handle = target_module.register_forward_hook(hook_fn)

    def extract_features(indices):
        """
        Return features with shape:
          (N, C, H_TARGET, W_TARGET)
        """

        feats = []
        bs = int(args.batch_size)

        if dataset_kind == "single_frame":
            all_x = test_ds

            with torch.no_grad():
                for start in range(0, len(indices), bs):
                    idx = indices[start:start + bs]
                    x = all_x[idx].to(device)

                    feat_cache.clear()
                    _ = model(x)

                    if len(feat_cache) == 0:
                        raise RuntimeError("Hook did not capture any feature.")

                    raw_feat = feat_cache[0]
                    # ★ Some baselines (e.g. IsoUNet, AttUNet) pad input
                    # to (H+pad, W+pad) inside .net.forward before running
                    # the encoder/decoder; the hooked feature still carries
                    # this padding. Crop it back to (H, W) BEFORE any
                    # bilinear interpolation — otherwise pad-pixel feature
                    # noise leaks into the analog map at the boundaries.
                    if raw_feat.dim() == 4 and hasattr(model, "net") and \
                            hasattr(model.net, "_crop") and \
                            raw_feat.shape[-2:] != (H_TARGET, W_TARGET):
                        try:
                            raw_feat = model.net._crop(raw_feat)
                        except Exception as e:
                            print(f"[warn] _crop failed: {e}")
                    feat = normalize_feature_shape(
                        raw_feat,
                        H_TARGET,
                        W_TARGET,
                        **feat_cfg,
                    )

                    feats.append(feat.cpu())

        else:
            subset = torch.utils.data.Subset(test_ds, list(indices))
            loader = torch.utils.data.DataLoader(
                subset,
                batch_size=bs,
                shuffle=False,
                num_workers=0,
            )

            with torch.no_grad():
                for batch in loader:
                    x = batch[0].to(device)

                    feat_cache.clear()
                    _ = model(x)

                    if len(feat_cache) == 0:
                        raise RuntimeError("Hook did not capture any feature.")

                    raw_feat = feat_cache[0]
                    # ★ See note above — crop padding before bilinear resize.
                    if raw_feat.dim() == 4 and hasattr(model, "net") and \
                            hasattr(model.net, "_crop") and \
                            raw_feat.shape[-2:] != (H_TARGET, W_TARGET):
                        try:
                            raw_feat = model.net._crop(raw_feat)
                        except Exception as e:
                            print(f"[warn] _crop failed: {e}")
                    feat = normalize_feature_shape(
                        raw_feat,
                        H_TARGET,
                        W_TARGET,
                        **feat_cfg,
                    )

                    feats.append(feat.cpu())

        return torch.cat(feats, dim=0)

    if args.n_avg > 1:
        n = min(args.n_avg, len(test_ds))
        print(f"\nMode: average over first {n} samples")

        t0 = time.perf_counter()
        feats_n = extract_features(list(range(n)))
        runs = [(f"avg_{n}", feats_n.mean(dim=0))]
        print(f"Feature extraction done in {time.perf_counter() - t0:.1f}s")

    else:
        sample_indices = [
            int(x) for x in args.sample_indices.split(",") if x.strip()
        ]
        sample_indices = [i for i in sample_indices if 0 <= i < len(test_ds)]

        if len(sample_indices) == 0:
            raise RuntimeError(f"No valid sample index; test_ds length={len(test_ds)}")

        print(f"\nMode: per-sample for {sample_indices}")

        t0 = time.perf_counter()
        feats_n = extract_features(sample_indices)
        runs = [
            (f"sample_{idx}", feats_n[i])
            for i, idx in enumerate(sample_indices)
        ]
        print(f"Feature extraction done in {time.perf_counter() - t0:.1f}s")

    handle.remove()

    C, H, W = runs[0][1].shape
    print(f"\nFeature shape: C={C}, H={H}, W={W}")

    queries = build_queries(args, lat_values, lon_values)

    print(f"\nQueries: {len(queries)}")
    for q in queries:
        h, w = q["pixel_ids"][0]
        print(
            f"  {q['name']:20s} "
            f"lat={q['lat']:+6.1f}, lon={q['lon']:6.1f} "
            f"-> pixel=({h}, {w})"
        )

    print("\nPlot config:")
    print(f"  center       = {args.center}")
    print(f"  zonal_demean = {args.zonal_demean}")
    print(f"  sharpen      = {args.sharpen}")
    print(f"  top_pct      = {args.top_pct}")
    print(f"  cmap         = {args.cmap}")
    print(f"  vmin/vmax    = {args.vmin}/{args.vmax}")

    for run_name, feat_tensor in runs:
        run_dir = os.path.join(out_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n== {run_name} ==")

        sim_list = []

        for q in queries:
            sim = compute_analog_map(
                feat_tensor,
                q["pixel_ids"],
                center=args.center,
                sharpen=args.sharpen,
                zonal_demean=args.zonal_demean,
                top_pct=args.top_pct,
            )

            sim_list.append(sim)

            slug = safe_slug(q["name"])
            save_png = os.path.join(
                run_dir,
                f"analog_{slug}_{args.dataset}.png",
            )

            plot_rect_single(
                sim,
                q,
                save_png,
                lat_values=lat_values,
                lon_values=lon_values,
                vmin=args.vmin,
                vmax=args.vmax,
                cmap=args.cmap,
                dpi=args.dpi,
                show_grid=args.show_grid,
                save_pdf=args.save_pdf,
            )

            print(
                f"  {q['name']:20s} -> {os.path.basename(save_png)} "
                f"min={np.nanmin(sim):+.3f}, "
                f"max={np.nanmax(sim):+.3f}, "
                f"mean={np.nanmean(sim):+.3f}, "
                f"std={np.nanstd(sim):.3f}"
            )

        combined_png = os.path.join(
            run_dir,
            f"combined_{args.dataset}.png",
        )

        dataset_label = (
            f"{args.dataset} "
            f"({REGISTRY[args.dataset]['co2']} ppm) - {run_name}"
        )

        plot_rect_combined(
            queries,
            sim_list,
            combined_png,
            lat_values=lat_values,
            lon_values=lon_values,
            dataset_label=dataset_label,
            n_cols=3,
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            dpi=args.dpi,
            show_grid=args.show_grid,
            save_pdf=args.save_pdf,
        )

        print(f"  combined -> {os.path.basename(combined_png)}")

    print("\n" + "=" * 70)
    print(f"Done. Wrote {len(runs)} run dirs to:")
    print(f"  {out_dir}")
    print(f"Log:")
    print(f"  {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
