"""
draw_feature.py — Climate-Analog Map Generator (decoder feature similarity).

For a trained model, computes per-pixel feature embeddings just before the
regression head (`d1 = decoder last feature`), then for each query location
`(lat₀, lon₀)` shows the cosine similarity to ALL global locations on a
Robinson projection. Locations the model treats as "predictively equivalent
for δ¹⁸O" light up — revealing the model's internal regime structure.

Use case for paper:
  - Query a tropical land+wet point (Amazon) → expect Congo, SE Asia analogs
  - Query Antarctica → expect Greenland, sea-ice fringes (validates ice expert)
  - Query coastal point → expect other coastlines (validates 4-way at boundaries)

Doesn't touch:
  - eval.log / eval_results.json / spatial/predict_*  (eval pipeline files)
  - draw.log / attention/                              (draw_inner.py output)

Writes to:
  <ckpt_dir>/<out_subdir>/                              (default: feature_analog/)
    ├── analog_<query>_<dataset>.{pdf,png}              individual maps
    ├── combined_<dataset>.{pdf,png}                    grid of all queries
    └── feature_analog.log                              standalone log

Usage:
  python draw_feature.py --ckpt /path/to/model.pt
  python draw_feature.py --ckpt ... --dataset 420ppm     # use specific dataset
  python draw_feature.py --ckpt ... --n_avg 50           # avg features over 50 samples
  python draw_feature.py --ckpt ... --queries 5,-60 20,5 # custom query points (lat,lon)
"""
import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import yaml
import lightning as L

import matplotlib
matplotlib.use('Agg')


# ── MODEL_FEATURE_REGISTRY ───────────────────────────────────────────────
# Map each Lightning model class (the string used in YAML `model_class:`) to
# the dot-path of the submodule whose OUTPUT is the "last feature before the
# regression head" — what we want to compare via cosine analog maps.
#
# Special keys:
#   'path'       : dot-separated submodule path from the Lightning module root
#   'perm'       : if 'BHWC', permute (B,H,W,C) → (B,C,H,W) (UNO-style)
#   'squeeze_t'  : if True, feature is (B,C,T,H,W); take the LAST temporal step
#   'upsample'   : if True, bilinear upsample (H',W') → (H_target,W_target)
#
# Dot-path resolver handles three component types:
#   - attribute       : 'net.up1'        → getattr
#   - integer index   : 'decoder_blocks.3' → list[3]
#   - dict-style key  : 'nested.x_0_4'   → ModuleDict['x_0_4']
MODEL_FEATURE_REGISTRY = {
    # ── U-Net family: feature is (B, C, H, W) at full resolution ──────────
    'iso_unet.baseline_iso_unet.IsoUNetBaseline':
        dict(path='net.up1'),
    'iso_unet.baseline_iso_unet_v2.IsoUNetBaselineV2':
        dict(path='net.up1'),
    'iso_unet.baseline_unet2d_vanilla.UNet2DVanillaBaseline':
        dict(path='net.up1'),
    'iso_unet.baseline_unet2d.UNet2DBaseline':
        dict(path='net.up1'),
    'iso_unet.baseline_attunet.AttentionUNetBaseline':
        dict(path='net.Up_conv2'),
    'iso_unet.baseline_canet.CANetBaseline':
        dict(path='net.scale_att'),
    'iso_unet.baseline_dcsaunet.DCSAUNetBaseline':
        dict(path='net.up4'),
    # ── ModuleDict / ModuleList paths ─────────────────────────────────────
    'iso_unet.baseline_unetpp.UNetPlusPlusBaseline':
        dict(path='net.nested.x_0_4'),               # depth=4 final nested node
    'iso_unet.baseline_transunet.TransUNetBaseline':
        dict(path='net.decoder_blocks.3'),           # last of 4 decoder blocks
    # ── Operator-style / unusual shapes ──────────────────────────────────
    'iso_unet.baseline_uno.UNOBaseline':
        dict(path='net.fc1', perm='BHWC'),           # (B,H,W,C) → permute
    'iso_unet.baseline_unet3d_temporal.TemporalUNet3DBaseline':
        dict(path='net.dec1', squeeze_t=True),       # (B,C,T,H,W) → last T
    # ── SegFormer: hook after the fused/BN'd 256-ch features, BEFORE the
    # 1-ch classifier. _normalize_feature_shape will bilinear-upsample
    # from (H/4, W/4) → (H, W).
    'iso_unet.baseline_segformer.SegFormerBaseline':
        dict(path='seg.base_model.decode_head.batch_norm'),
    # ── ClimaX (ViT, variable-tokenized): hook the LayerNorm right after
    # all transformer blocks. Output is (B*T, L, D) with L = h_patch*w_patch.
    # token_grid + token_T + token_aggT reshape it back into spatial (B,D,H,W).
    # Defaults below match climax_full_split.yaml (patch_size=5, seq_len=12).
    'iso_unet.baseline_climax.ClimaXBaseline':
        dict(path='arch.norm',
             token_grid=(18, 36),         # 90/5, 180/5
             token_T=12,
             token_aggT='mean'),
    # ── Stormer (ViT, time-aggregated to single frame BEFORE blocks): hook
    # the LAST transformer block. Output is (B, L, D) with L = h_patch*w_patch
    # already (no T dim because our wrapper takes the last input frame).
    # Defaults below match stormer_full_split.yaml (patch_size=5).
    'iso_unet.baseline_stormer.StormerBaseline':
        dict(path='arch.blocks.-1',       # last block, depth-agnostic via -1
             token_grid=(18, 36),
             token_T=1),                   # already collapsed by our wrapper
    # NOTE: FNO2dSeq has no clear "decoder feature" (the spectral mixer
    # outputs directly to 1-channel) — add a custom entry only if needed.
}


def _resolve_submodule(model, dot_path):
    """Walk a dot-separated path against a model. Handles:
       - attribute lookup            (`net.up1`)
       - integer indices, +/-        (`decoder_blocks.3`, `blocks.-1`)
       - ModuleDict string keys      (`nested.x_0_4`)
    """
    m = model
    for part in dot_path.split('.'):
        # Try integer index first (allow leading '-' for "last")
        try:
            idx = int(part)
            m = m[idx]
            continue
        except (ValueError, TypeError):
            pass
        try:
            m = getattr(m, part)
        except AttributeError:
            try:
                m = m[part]                # ModuleDict key
            except (TypeError, KeyError):
                raise AttributeError(
                    f'Could not resolve "{part}" on {type(m).__name__} '
                    f'(full path: "{dot_path}")')
    return m


def _normalize_feature_shape(feat, target_H, target_W, *,
                              perm=None, squeeze_t=False,
                              token_grid=None, token_T=None, token_aggT='mean'):
    """Coerce a hooked feature to (B, C, H, W) at target spatial size.

    Supported adapter pipelines (applied in order if specified):
      perm='BHWC'                : (B, H, W, C) → (B, C, H, W)        (UNO)
      squeeze_t=True             : (B, C, T, H, W) → take last T      (UNet3D)
      token_grid=(H_p, W_p),
      token_T=T,
      token_aggT='mean'|'last'   : (B*T, L, D) → reshape into
                                    (B, D, H_p, W_p) by un-flattening BxT,
                                    aggregating over T, reshaping L=H_p·W_p.
                                    (ClimaX-style ViT tokens.)
    Then bilinear-upsample to (target_H, target_W) if needed.
    """
    if perm == 'BHWC':
        feat = feat.permute(0, 3, 1, 2).contiguous()
    if squeeze_t and feat.dim() == 5:
        feat = feat[:, :, -1].contiguous()
    if token_grid is not None:
        H_p, W_p = token_grid
        if feat.dim() != 3:
            raise RuntimeError(
                f'token_grid adapter expects 3D (B*T, L, D) feature, got '
                f'{tuple(feat.shape)}.')
        BT, L, D = feat.shape
        if L != H_p * W_p:
            raise RuntimeError(
                f'token_grid adapter: L={L} != H_p*W_p={H_p*W_p}.')
        T = token_T or 1
        B = BT // T
        if B * T != BT:
            raise RuntimeError(
                f'token_grid adapter: token_T={T} does not divide first dim {BT}.')
        feat = feat.reshape(B, T, L, D)
        if token_aggT == 'mean':
            feat = feat.mean(dim=1)
        elif token_aggT == 'last':
            feat = feat[:, -1]
        else:
            raise ValueError(f'Unknown token_aggT={token_aggT!r}')
        feat = feat.reshape(B, H_p, W_p, D).permute(0, 3, 1, 2).contiguous()
    if feat.dim() != 4:
        raise RuntimeError(f'feature shape {tuple(feat.shape)} after adapters '
                            f'is not 4D — cannot use for analog map.')
    if feat.shape[-2] != target_H or feat.shape[-1] != target_W:
        feat = F.interpolate(feat.float(), size=(target_H, target_W),
                              mode='bilinear', align_corners=False)
    return feat

import matplotlib.pyplot as plt


# ── Predefined physically meaningful regions (CENTER POINT: lat, lon °) ────
#  lon ∈ [0, 360).  Each region is represented by ONE center pixel.
DEFAULT_REGIONS = {
    # ENSO Pacific
    'Nino3.4'        : (  0, 215),   # ocean + wet (equatorial Pacific)
    'Nino1+2'        : ( -5, 275),
    'Nino3'          : (  0, 240),
    'Nino4'          : (  0, 185),
    # Mid-lat coastal & monsoon land
    'US_West_Coast'  : ( 40, 239),
    'Indian_Monsoon' : ( 17,  80),
    'EA_Monsoon'     : ( 32, 117),
    'Tibetan_Plateau': ( 34,  90),
    'Mediterranean'  : ( 37,  20),
    # Tropical land/ocean
    'Tropical_IO'    : (  0,  75),
    'Amazon'         : ( -7, 310),
    'Sahara'         : ( 22,  15),
    # Polar ice
    'Greenland'      : ( 71, 325),
    'Antarctica'     : (-80,   0),
    'Arctic'         : ( 80, 180),
}


# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Climate-analog map generator')
parser.add_argument('--ckpt',         type=str, required=True,
                    help='Trained model checkpoint (model.pt)')
parser.add_argument('--config',       type=str, default=None,
                    help='Override config path (default: <ckpt_dir>/config.yaml)')
parser.add_argument('--out_subdir',   type=str, default='feature_analog',
                    help='Output subdir under ckpt_dir (default: "feature_analog")')
parser.add_argument('--dataset',      type=str, default='420ppm',
                    help='Which dataset to compute analogs on (default: 420ppm, '
                         'a middle ID dataset)')
parser.add_argument('--sample_indices', type=str, default='0,100,200,400',
                    help='Comma-separated test sample indices to use (per-day mode). '
                         'Each idx produces a separate set of analog maps in '
                         '<out_dir>/sample_<idx>/. Default "0,100,200,400" = 4 days '
                         '(spanning the year if monthly test data).')
parser.add_argument('--n_avg',        type=int, default=1,
                    help='If >1, ignore --sample_indices and average features over '
                         '`n_avg` samples (starting at idx 0) into ONE set of maps '
                         'in <out_dir>/avg_<n>/. Default 1 = per-sample mode.')
parser.add_argument('--regions',      type=str, default='all',
                    help='Comma-separated region names from DEFAULT_REGIONS (boxes). '
                         'Use "all" for every region, empty string "" to skip. '
                         'Example: "Nino3.4,Amazon,US_West_Coast". Default "all".')
parser.add_argument('--queries',      type=str, default=None,
                    help='Custom POINT queries, semicolon-separated "lat,lon" pairs. '
                         'Example: "--queries=-3,-60;38,15;25,180". '
                         'Use the "=" form so argparse does not eat negative lats. '
                         'Added on top of --regions; star marker on map.')
parser.add_argument('--batch_size',   type=int, default=4)
parser.add_argument('--seed',         type=int, default=42)
parser.add_argument('--feature_layer',type=str, default='auto',
                    help='Submodule dot-path to hook for feature extraction. '
                         '"auto" (default) → look up MODEL_FEATURE_REGISTRY by '
                         'cfg["model_class"]. Manual path is dot-separated from '
                         '`model` root, e.g. "net.up1", "net.decoder_blocks.3", '
                         '"net.nested.x_0_4".')
parser.add_argument('--center',       dest='center', action='store_true', default=True,
                    help='Subtract spatial-mean feature per-channel before normalize. '
                         'Removes DC bias → sim reflects local DEVIATION from mean state '
                         '(like climatological anomalies). Default ON.')
parser.add_argument('--no_center',    dest='center', action='store_false',
                    help='Disable mean-centering (legacy raw cosine sim).')
parser.add_argument('--sharpen',      type=float, default=4.0,
                    help='Power scaling: sim ← sign(sim) * |sim|^p. p>1 sharpens by '
                         'suppressing middle-similarity pixels. Default 4 (try 8 for '
                         'extra-sharp, 1 to disable).')
parser.add_argument('--zonal_demean', dest='zonal_demean', action='store_true', default=False,
                    help='Subtract per-latitude zonal mean from features before cosine. '
                         'Removes the "subtropical-band-looks-all-same" effect — sim '
                         'reflects only departures from each latitude\'s baseline '
                         '(zonal anomaly, standard climate convention). Default OFF.')
parser.add_argument('--no_zonal_demean', dest='zonal_demean', action='store_false',
                    help='Disable zonal demean (default).')
parser.add_argument('--vmin',         type=float, default=-1.0,
                    help='Colorbar min. Default -1.0 (full diverging range). '
                         'Use 0 with a sequential cmap to focus on positive analogs.')
parser.add_argument('--vmax',         type=float, default=1.0,
                    help='Colorbar max. Default 1.0.')
parser.add_argument('--cmap',         type=str, default='RdBu_r',
                    help='Matplotlib colormap. Default "RdBu_r" (diverging, white at 0). '
                         'Sequential alternatives that avoid the white-at-zero look: '
                         '"YlOrRd", "magma", "plasma", "viridis", "Reds", "hot_r". '
                         'For sequential, also pass --vmin 0.')
parser.add_argument('--top_pct',      type=float, default=100.0,
                    help='Show only the top P%% pixels by similarity, mask the rest as '
                         'light grey. Default 100 = no mask. Try 10 or 20 to focus on '
                         'the strongest analogs.')
args = parser.parse_args()


# ── Tee stdout/stderr to feature_analog.log ───────────────────────────────
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()


ckpt_path   = os.path.abspath(args.ckpt)
ckpt_dir    = os.path.dirname(ckpt_path)
config_path = args.config or os.path.join(ckpt_dir, 'config.yaml')
out_dir     = os.path.join(ckpt_dir, args.out_subdir)
os.makedirs(out_dir, exist_ok=True)
log_path    = os.path.join(out_dir, 'feature_analog.log')

_log_file = open(log_path, 'w', buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)
print(f'[{datetime.datetime.now()}] feature_analog.log -> {log_path}')
print(f'OUTPUT_DIR  = {out_dir}/')
print(f'CKPT        = {ckpt_path}')

# Add this dir to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data      import (REGISTRY, apply_ood_preset, REF_DATASET, compute_ref_stats,
                       get_input_features, load_xr)
from datasets  import build_splits
from models    import build_model, get_dataset_kind
from metrics   import make_lat_weights


L.seed_everything(args.seed, workers=True)

# ── Load config ────────────────────────────────────────────────────────────
assert os.path.exists(config_path), f'No config at {config_path}'
with open(config_path) as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('train_frac',   0.10)
cfg.setdefault('seq_len',      12)
cfg.setdefault('max_batch',    3000)
cfg.setdefault('input_set',    'basic')
cfg.setdefault('batch_size',   8)
cfg.setdefault('ood_preset',   'default')
cfg.setdefault('norm_mode',    'global')
cfg.setdefault('norm_ref',     REF_DATASET)
cfg.setdefault('lat_weighted', True)

input_features = cfg.get('input_features') or get_input_features(cfg['input_set'])
n_inputs       = len(input_features)
seq_len        = int(cfg['seq_len'])
train_frac     = float(cfg['train_frac'])
dataset_kind   = get_dataset_kind(cfg['model_class'], cfg.get('dataset_kind'))

apply_ood_preset(cfg['ood_preset'])

# Resolve dataset
assert args.dataset in REGISTRY, f'Unknown dataset {args.dataset}. Choose from {list(REGISTRY)}'
print(f'DATASET     = {args.dataset}  ({REGISTRY[args.dataset]["co2"]} ppm)')

# ── Load + build splits ───────────────────────────────────────────────────
print('\nLoading data...')
raw_chosen = load_xr(args.dataset, max_batch=cfg['max_batch'],
                     input_features=input_features)

ref_stats = None
if cfg['norm_mode'] == 'global':
    if 'norm_stats' in cfg and cfg['norm_stats']:
        ref_stats = cfg['norm_stats']
    else:
        raw_ref = load_xr(cfg['norm_ref'], max_batch=cfg['max_batch'],
                          input_features=input_features)
        ref_stats = compute_ref_stats(raw_ref, input_features, train_frac)

tr, va, te_a, te_b, stats, log = build_splits(
    args.dataset, raw_chosen, train_frac, seq_len, input_features, dataset_kind,
    ref_stats=ref_stats)
print(log)

# For 'sequence': te_a is a torch Dataset (SequenceDataset). For
# 'single_frame': te_a is a (N, C, H, W) Tensor of eval_x. Both expose
# `len()` and integer-list indexing, so we treat them uniformly.
test_ds = te_a


# ── Build + load model ────────────────────────────────────────────────────
lat_weights_t = (torch.as_tensor(make_lat_weights(), dtype=torch.float32)
                 if cfg['lat_weighted'] else None)
mkw = dict(cfg['model_kwargs']); mkw['n_inputs'] = n_inputs

model = build_model(cfg['model_class'], model_kwargs=mkw, weights=lat_weights_t)
sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing or unexpected:
    print(f'  [warn] missing={len(missing)}, unexpected={len(unexpected)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()
print(f'\nLoaded model on {device}')


# ── Hook to grab the "last feature before regression head" ───────────────
# Strategy: look up the model's config in MODEL_FEATURE_REGISTRY, and resolve
# a dot-separated path against the Lightning module root. The user can also
# override by passing --feature_layer explicitly.
model_class = cfg.get('model_class', '')
if args.feature_layer == 'auto':
    if model_class not in MODEL_FEATURE_REGISTRY:
        print(f'[ERROR] model_class "{model_class}" is not in '
              f'MODEL_FEATURE_REGISTRY. Either:\n'
              f'  (1) add an entry to MODEL_FEATURE_REGISTRY in this file, or\n'
              f'  (2) pass --feature_layer <dot.path> explicitly.\n'
              f'Known model_class entries: '
              f'{list(MODEL_FEATURE_REGISTRY.keys())[:6]} ...')
        sys.exit(1)
    feat_cfg = dict(MODEL_FEATURE_REGISTRY[model_class])   # copy
    feat_path = feat_cfg.pop('path')
else:
    feat_path = args.feature_layer
    # Reuse registry adapters if available, else default to identity
    feat_cfg = {k: v for k, v in
                 MODEL_FEATURE_REGISTRY.get(model_class, {}).items()
                 if k != 'path'}

print(f'\nHook target  : {feat_path}')
print(f'Adapters     : {feat_cfg if feat_cfg else "(none)"}')

try:
    target_module = _resolve_submodule(model, feat_path)
except AttributeError as e:
    print(f'[ERROR] {e}')
    sys.exit(1)

H_TARGET, W_TARGET = 90, 180

_feat_cache = []
def _hook(module, inp, out):
    _feat_cache.append(out.detach())
handle = target_module.register_forward_hook(_hook)


# ── Helper: extract feature for given sample indices ─────────────────────
def _extract_feature_for_indices(indices):
    """Run forward on these test sample indices, return (n, C, H, W) feature
    tensor — normalized to (target_H, target_W) and (B, C, H, W) layout.
    Works for both 'sequence' (te_a is Dataset → DataLoader) and
    'single_frame' (te_a is Tensor → index directly)."""
    feats = []
    bs   = args.batch_size

    if dataset_kind == 'single_frame':
        # te_a is a (N, C, H, W) tensor; just slice it
        all_x = test_ds                           # alias
        with torch.no_grad():
            for i in range(0, len(indices), bs):
                idx = indices[i:i + bs]
                x = all_x[idx].to(device)
                _feat_cache.clear()
                _ = model(x)
                feat = _feat_cache[0]
                feat = _normalize_feature_shape(feat, H_TARGET, W_TARGET,
                                                 **feat_cfg)
                if (feat.shape[-2] != H_TARGET or feat.shape[-1] != W_TARGET) \
                        and hasattr(model, 'net') and hasattr(model.net, '_crop'):
                    feat = model.net._crop(feat)
                feats.append(feat.cpu())
    else:
        # sequence: te_a is a torch Dataset
        sub = torch.utils.data.Subset(test_ds, list(indices))
        loader = torch.utils.data.DataLoader(sub, batch_size=bs,
                                              shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                _feat_cache.clear()
                _ = model(x)
                feat = _feat_cache[0]
                feat = _normalize_feature_shape(feat, H_TARGET, W_TARGET,
                                                 **feat_cfg)
                if (feat.shape[-2] != H_TARGET or feat.shape[-1] != W_TARGET) \
                        and hasattr(model, 'net') and hasattr(model.net, '_crop'):
                    feat = model.net._crop(feat)
                feats.append(feat.cpu())
    return torch.cat(feats, dim=0)   # (n, C, H, W)


# ── Decide mode ────────────────────────────────────────────────────────────
if args.n_avg > 1:
    # Average mode: features averaged over first n_avg samples → single output dir
    n = min(args.n_avg, len(test_ds))
    print(f'\nMode: AVERAGE over {n} samples → single output')
    t0 = time.perf_counter()
    feats_n = _extract_feature_for_indices(list(range(n)))     # (n, C, H, W)
    runs = [('avg_{:d}'.format(n), feats_n.mean(dim=0))]
    print(f'  done in {time.perf_counter()-t0:.1f}s')
else:
    # Per-sample mode: one set of analog maps per requested sample idx
    sample_indices = [int(x) for x in args.sample_indices.split(',') if x.strip()]
    # Filter out-of-range
    sample_indices = [i for i in sample_indices if 0 <= i < len(test_ds)]
    if not sample_indices:
        print(f'[ERROR] No valid sample_indices (test_ds has {len(test_ds)} samples)')
        sys.exit(1)
    print(f'\nMode: PER-SAMPLE for indices {sample_indices}')
    t0 = time.perf_counter()
    feats_n = _extract_feature_for_indices(sample_indices)     # (len(idx), C, H, W)
    runs = [('sample_{:d}'.format(idx), feats_n[i])
            for i, idx in enumerate(sample_indices)]
    print(f'  done in {time.perf_counter()-t0:.1f}s')

handle.remove()

C, H, W = runs[0][1].shape
print(f'\nFeature: C={C}, H={H}, W={W}, {len(runs)} sets to plot')


# ── Resolve query points to pixel indices ────────────────────────────────
LAT_VALUES = np.linspace(-89, 89, H)         # NC data convention
LON_VALUES = np.linspace(0, 360, W, endpoint=False) + 1.0
def latlon_to_idx(lat, lon):
    """Convert (lat, lon) in degrees → (h, w) pixel idx (closest)."""
    lon = lon if lon >= 0 else lon + 360.0
    h = int(np.argmin(np.abs(LAT_VALUES - lat)))
    w = int(np.argmin(np.abs(LON_VALUES - lon)))
    return h, w


# ── Build query list (all single-pixel point queries) ───────────────────
# Each entry: {'name': str, 'lat': float, 'lon': float in [0,360), 'pixel_ids': [(h,w)]}
queries = []
if args.regions:
    if args.regions.strip().lower() == 'all':
        sel_names = list(DEFAULT_REGIONS.keys())
    else:
        sel_names = [s.strip() for s in args.regions.split(',') if s.strip()]
    for name in sel_names:
        if name not in DEFAULT_REGIONS:
            print(f'  [warn] unknown region "{name}", skip. '
                  f'Known: {list(DEFAULT_REGIONS.keys())}')
            continue
        la, lo = DEFAULT_REGIONS[name]
        lo_n = lo if lo >= 0 else lo + 360.0
        h, w = latlon_to_idx(la, lo_n)
        queries.append({'name': name, 'lat': la, 'lon': lo_n,
                         'pixel_ids': [(h, w)]})

if args.queries:
    for q in args.queries.split(';'):
        q = q.strip()
        if not q: continue
        try:
            la, lo = map(float, q.split(','))
        except Exception as e:
            print(f'  [warn] skip bad query "{q}": {e}'); continue
        lo_n = lo if lo >= 0 else lo + 360.0
        h, w = latlon_to_idx(la, lo_n)
        queries.append({'name': 'pt_{:+d}_{:+d}'.format(int(la), int(lo)),
                         'lat': la, 'lon': lo_n,
                         'pixel_ids': [(h, w)]})

if not queries:
    print('[ERROR] No queries selected. Use --regions or --queries.')
    sys.exit(1)

print(f'\nQueries ({len(queries)}):')
for q in queries:
    h, w = q['pixel_ids'][0]
    print(f'  {q["name"]:20s}  ({q["lat"]:+5.0f}°, {q["lon"]:5.0f}°)  → pixel ({h}, {w})')


# ── Compute analog map helper (cosine similarity to query) ────────────────
def _analog_map(feat_tensor, pixel_ids, *,
                 center=True, sharpen=1.0, zonal_demean=False, top_pct=100.0):
    """Cosine similarity to query, where query feature = mean over pixel_ids.
    feat_tensor: (C, H, W). Returns (H, W) numpy.

    zonal_demean=True : per-channel, per-latitude subtract zonal mean
                    f(c,h,w) ← f(c,h,w) − mean_w f(c,h,w)
                    Removes the zonal-band component that makes the subtropics
                    look uniform. The cosine then operates on "zonal anomaly"
                    features — the standard climate convention for stripping
                    out latitudinal effects.
    center=True   : subtract spatial-mean feature per channel before normalize.
                    Removes the global DC component → sim reflects how each pixel
                    DEVIATES from the climatological mean state. Sharpens contrast.
    sharpen=p > 1 : sim ← sign(sim) * |sim|^p. Suppresses middle-range
                    similarities (e.g. 0.6→0.13 for p=4) while keeping peaks
                    (0.95→0.81). Strong perceptual sharpening.

    Multi-pixel query: average L2-normalized features inside the box, then
    re-normalize. Equivalent to a "regional signature" centroid in the unit
    sphere — much more stable than a single-pixel query.
    """
    Cf, Hf, Wf = feat_tensor.shape
    f = feat_tensor
    if zonal_demean:
        f = f - f.mean(dim=2, keepdim=True)                 # remove per-row zonal mean
    flat = f.reshape(Cf, -1)
    if center:
        flat = flat - flat.mean(dim=1, keepdim=True)        # per-channel spatial mean
    flat_norm = F.normalize(flat, dim=0, eps=1e-8)
    q_ids = [h * Wf + w for (h, w) in pixel_ids]
    f_q = flat_norm[:, q_ids].mean(dim=1)                    # (C,) centroid
    f_q = F.normalize(f_q, dim=0, eps=1e-8)                  # back onto unit sphere
    sim = (flat_norm.T @ f_q).reshape(Hf, Wf)
    if sharpen != 1.0:
        sim = sim.sign() * sim.abs() ** sharpen
    sim = sim.numpy()
    if top_pct < 100.0:
        thresh = np.quantile(sim, 1.0 - top_pct / 100.0)
        sim = np.where(sim >= thresh, sim, np.nan)        # mask below-threshold
    return sim


# ── Plot helpers (rectangular PlateCarree, Pacific-centered) ──────────────
# Why rectangular and not Robinson:
#   - True rectangle (1:2 aspect) — no projection curvature
#   - No antimeridian seam artifact when the data wraps cyclically
#   - Easier to read latitude/longitude positions
#   - Visually clearer for paper figures
# Why central_longitude=180:
#   - Puts the Pacific (~lon 180°) at the figure center → the cyclic "cut"
#     falls at lon=0 (Greenwich), which has minimal land area in our domain
#   - Continents (Asia, Americas, Africa) stay un-split
def _draw_query_marker(ax, q, *, star_size=240):
    """Draw a yellow star at the query location."""
    import cartopy.crs as ccrs
    ax.scatter([q['lon']], [q['lat']],
               marker='*', s=star_size, c='yellow',
               edgecolors='black', linewidths=1.2,
               transform=ccrs.PlateCarree(), zorder=10)


def _setup_rect_axes(ax, with_gridlabels=True):
    """Configure a PlateCarree axis to look like a clean rectangular map.
    Adds coastlines, optional gridlines with labels, and ensures the full
    [-180, 180] × [-90, 90] extent is shown."""
    import cartopy.crs as ccrs
    ax.set_global()
    ax.coastlines(linewidth=0.6, color='#333333', zorder=5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=with_gridlabels,
                      linewidth=0.3, alpha=0.35, linestyle='--', color='gray')
    if with_gridlabels:
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}


def _plot_single_analog(sim, q, save_pdf, vmin=-1, vmax=1, cmap='RdBu_r'):
    """Single analog map — rectangular PlateCarree(central_longitude=180).
    NaN pixels (top_pct mask) render as light grey overlay."""
    import cartopy.crs as ccrs
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='lightgrey', alpha=0.45)

    fig = plt.figure(figsize=(11, 5.2))
    ax = fig.add_subplot(1, 1, 1,
                          projection=ccrs.PlateCarree(central_longitude=180))
    sim_m = np.ma.masked_invalid(sim)
    mesh = ax.pcolormesh(LON_VALUES, LAT_VALUES, sim_m,
                          transform=ccrs.PlateCarree(),
                          cmap=cmap_obj, vmin=vmin, vmax=vmax,
                          shading='auto')
    _setup_rect_axes(ax, with_gridlabels=True)
    _draw_query_marker(ax, q)
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical',
                         shrink=0.85, pad=0.03, aspect=25)
    cbar.set_label('cosine similarity', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(f'Climate analog: {q["name"]}', fontsize=12, pad=8)
    fig.savefig(save_pdf, dpi=140, bbox_inches='tight')
    fig.savefig(save_pdf.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)


def _plot_combined_grid(query_list, sim_list, save_pdf,
                         dataset_label, n_cols=3, vmin=-1, vmax=1,
                         cmap='RdBu_r'):
    """Paper-figure grid of all queries on rectangular PlateCarree panels."""
    import cartopy.crs as ccrs
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='lightgrey', alpha=0.45)

    N = len(query_list)
    n_rows = (N + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(6.0 * n_cols, 2.8 * n_rows + 0.6))

    for i, (q, sim) in enumerate(zip(query_list, sim_list)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1,
                              projection=ccrs.PlateCarree(central_longitude=180))
        sim_m = np.ma.masked_invalid(sim)
        mesh = ax.pcolormesh(LON_VALUES, LAT_VALUES, sim_m,
                              transform=ccrs.PlateCarree(),
                              cmap=cmap_obj, vmin=vmin, vmax=vmax,
                              shading='auto')
        # Per-panel: coastlines yes, gridlabels no (too cluttered in a grid)
        _setup_rect_axes(ax, with_gridlabels=False)
        _draw_query_marker(ax, q, star_size=160)
        ax.set_title(q['name'], fontsize=10, pad=4)

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.10, 0.012, 0.78])
    cbar = plt.colorbar(mesh, cax=cbar_ax)
    cbar.set_label('cosine similarity to query', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f'Climate Analogs  ({dataset_label})', fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.01, 0.90, 0.97])

    fig.savefig(save_pdf, dpi=160, bbox_inches='tight')
    fig.savefig(save_pdf.replace('.pdf', '.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


# ── Loop over runs (per-sample OR averaged) and produce analog maps ───────
print(f'\nSharpening config: center={args.center}, zonal_demean={args.zonal_demean}, '
      f'sharpen={args.sharpen}, vmin={args.vmin}, vmax={args.vmax}, cmap={args.cmap}')
print(f'\nComputing & saving analog maps to {out_dir}/...')
for run_name, feat_tensor in runs:
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f'\n── {run_name} ──')
    sim_list = []
    for q in queries:
        sim = _analog_map(feat_tensor, q['pixel_ids'],
                          center=args.center, sharpen=args.sharpen,
                          zonal_demean=args.zonal_demean,
                          top_pct=args.top_pct)
        sim_list.append(sim)
        slug = q['name'].lower().replace('+', 'p').replace('.', '').replace(
            '/', '_').replace(' ', '_').replace('(', '').replace(')', '')[:18]
        save_pdf = os.path.join(run_dir, f'analog_{slug}_{args.dataset}.pdf')
        _plot_single_analog(sim, q, save_pdf,
                            vmin=args.vmin, vmax=args.vmax, cmap=args.cmap)
        print(f'  {q["name"]:20s} → {run_name}/{os.path.basename(save_pdf)}  '
              f'(min={np.nanmin(sim):+.3f}  max={np.nanmax(sim):+.3f}  '
              f'mean={np.nanmean(sim):+.3f}  std={np.nanstd(sim):.3f})')

    combined_pdf = os.path.join(run_dir, f'combined_{args.dataset}.pdf')
    _plot_combined_grid(queries, sim_list, combined_pdf,
                         dataset_label=f'{args.dataset} ({REGISTRY[args.dataset]["co2"]} ppm) — {run_name}',
                         vmin=args.vmin, vmax=args.vmax, cmap=args.cmap)
    print(f'  combined grid → {run_name}/{os.path.basename(combined_pdf)}')


print(f'\n{"="*60}')
print(f'✓ Wrote {len(runs)} run dirs to {out_dir}/  (each: {len(queries)+1} figures)')
print(f'✓ Log: {log_path}')
print(f'✓ eval.log / draw.log / other PDFs NOT touched')
print(f'{"="*60}')
