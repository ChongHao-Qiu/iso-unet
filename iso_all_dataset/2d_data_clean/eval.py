"""
eval.py — Universal evaluation entry. Read ckpt + same-dir config.yaml,
run inference, write eval_results.json + spatial PDFs.  **DOES NOT touch
train.log** — eval-only log goes to eval.log (and also stdout).

Usage:
    python eval.py --ckpt experiments/<model_name>/<exp_tag>/model.pt

    # Custom output dir (default: ckpt directory)
    python eval.py --ckpt ... --output_dir ./reeval_dir

Behavior:
    * Automatically reads <ckpt_dir>/config.yaml to rebuild the model + dataset
    * Unified inference: single-frame → (B, 1, H, W); sequence → take last frame y_hat[:, -1]
    * Outputs share the same schema as all existing baselines:
        eval_results.json
        spatial/<tag>_truth_pred.nc + <tag>_metrics_2d.nc
        spatial/predict_<tag>_{R2,RMSE,MAE}_spatial.pdf
"""
import os, sys, glob, json, argparse, datetime, time
import yaml
import numpy as np
import xarray as xr
import torch
import lightning as L


_T0_PERF = time.perf_counter()

def _fmt_duration(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    if h: return f'{h}h {m}m {s}s'
    if m: return f'{m}m {s}s'
    return f'{s}s ({sec:.1f}s)'


class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()


# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt',         type=str, required=True,
                    help='trained model checkpoint (model.pt)')
parser.add_argument('--config',       type=str, default=None,
                    help='Override config path (default: <ckpt_dir>/config.yaml)')
parser.add_argument('--output_dir',   type=str, default=None,
                    help='Override output dir (default: <ckpt_dir>)')
parser.add_argument('--batch_size',   type=int, default=None,
                    help='Override eval batch size (default: from config or 8)')
parser.add_argument('--save_attn',    action='store_true',
                    help='If the model supports forward_with_attn, also export an attention dict')
parser.add_argument('--seed',         type=int, default=42)
args = parser.parse_args()

# Resolve paths
ckpt_dir   = os.path.dirname(os.path.abspath(args.ckpt))
config_path = args.config or os.path.join(ckpt_dir, 'config.yaml')
output_dir  = args.output_dir or ckpt_dir
os.makedirs(output_dir, exist_ok=True)

# Tee to eval.log (NOT train.log)
_log_path = os.path.join(output_dir, 'eval.log')
_log_file = open(_log_path, 'w', buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)
print(f'[{datetime.datetime.now()}] eval.log -> {_log_path}')

# Add this dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data      import (REGISTRY, apply_ood_preset,
                       REF_DATASET, compute_ref_stats,
                       get_input_features, load_xr)
from datasets  import build_splits
from models    import build_model, get_dataset_kind, run_inference_unified
from metrics   import compute_metrics, define_regions, make_lat_weights
from visualize import draw_results


L.seed_everything(args.seed, workers=True)

# ── Load config ─────────────────────────────────────────────────────────────
print(f'CKPT       = {args.ckpt}')
print(f'CONFIG     = {config_path}')
print(f'OUTPUT     = {output_dir}')

assert os.path.exists(config_path), f'No config at {config_path} — was the model trained with train.py?'
with open(config_path) as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('train_frac', 0.10)
cfg.setdefault('seq_len',    12)
cfg.setdefault('max_batch',  3000)
cfg.setdefault('input_set',  'basic')
cfg.setdefault('batch_size', 8)
cfg.setdefault('ood_preset', 'default')
# Default to global (consistent with train.py). For old ckpts trained with per_dataset,
# you must manually add to config.yaml:
#     norm_mode: per_dataset
# Otherwise inference will use global stats → input scale won't match training → wrong results.
cfg.setdefault('norm_mode', 'global')
cfg.setdefault('norm_ref',  REF_DATASET)
# Default True (legacy training was weighted). Old ckpts without a lat_weighted field → treated as True (correct).
# New ckpts trained unweighted will explicitly write lat_weighted: false in config.yaml.
cfg.setdefault('lat_weighted', True)

# ── Apply OOD preset (mutates REGISTRY's holdout, same as training) ────────
_ood_preset_name, _ood_tags, _id_tags = apply_ood_preset(cfg['ood_preset'])

batch_size       = args.batch_size or cfg['batch_size']
model_name       = cfg['model_name']
model_class_path = cfg['model_class']
input_features   = cfg.get('input_features') or get_input_features(cfg['input_set'])
n_inputs         = len(input_features)
dataset_kind     = get_dataset_kind(model_class_path, cfg.get('dataset_kind'))
seq_len          = int(cfg['seq_len'])
train_frac       = float(cfg['train_frac'])

print(f'MODEL        = {model_name}  ({model_class_path})')
print(f'TRAIN_FRAC   = {train_frac}  (same as training)')
print(f'INPUT_SET    = {cfg["input_set"]}  ({n_inputs} ch: {input_features})')
print(f'DATASET_KIND = {dataset_kind},  SEQ_LEN = {seq_len}')
print(f'OOD_PRESET   = {_ood_preset_name}')
print(f'   ID  = {_id_tags}')
print(f'   OOD = {_ood_tags}')
print(f'NORM_MODE    = {cfg["norm_mode"]}'
      + (f"  (ref={cfg['norm_ref']})" if cfg['norm_mode'] == 'global' else ''))
print(f'EVAL_BATCH   = {batch_size}')


# ── Load + build splits (same as training, so test set is identical) ───────
print('\nLoading datasets...')
raw = {}
for tag in REGISTRY:
    raw[tag] = load_xr(tag, max_batch=cfg['max_batch'], input_features=input_features)
    print(f'  {tag}: {len(raw[tag].time)} time steps')

# ── Resolve normalization stats (prefer saved in config.yaml; recompute if missing) ──
ref_stats = None
if cfg['norm_mode'] == 'global':
    ref_tag = cfg['norm_ref']
    if 'norm_stats' in cfg and cfg['norm_stats']:
        ref_stats = cfg['norm_stats']
        print(f'\n[norm] using saved global stats from config.yaml '
              f'({len(ref_stats)} channels)')
    else:
        assert ref_tag in REGISTRY, f"norm_ref='{ref_tag}' not in REGISTRY"
        ref_stats = compute_ref_stats(raw[ref_tag], input_features, train_frac)
        print(f'\n[norm] ⚠️  recomputed global stats from {ref_tag} train slice '
              f'(no norm_stats in config.yaml)')
        print(f'[norm] ⚠️  IF this ckpt was trained with per_dataset norm (old code), '
              f'this is WRONG — inputs will not match training scale.')
        print(f'[norm] ⚠️  FIX: add `norm_mode: per_dataset` to {config_path} and re-eval.')

print(f'\nBuilding splits (dataset_kind={dataset_kind}, train_frac={train_frac})...')
splits = {}
for tag in REGISTRY:
    tr, va, te_a, te_b, stats, log = build_splits(
        tag, raw[tag], train_frac, seq_len, input_features, dataset_kind,
        ref_stats=ref_stats)
    splits[tag] = (tr, va, te_a, te_b, stats)
    print(log)


# ── Build model + load weights ──────────────────────────────────────────────
# Must match training (cfg['lat_weighted']) so state_dict loads cleanly
# (weighted model has 'lat_weights' buffer in state_dict; unweighted doesn't)
lat_weights = make_lat_weights()
lat_weights_t = torch.as_tensor(lat_weights, dtype=torch.float32) if cfg['lat_weighted'] else None

mkw = dict(cfg.get('model_kwargs', {}))
mkw.setdefault('n_inputs', n_inputs)
mkw.setdefault('lr',       cfg.get('lr', 1e-3))

model = build_model(model_class_path, model_kwargs=mkw, weights=lat_weights_t)
state = torch.load(args.ckpt, map_location='cpu')
model.load_state_dict(state)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f'\nLoaded {n_params:,} params from {args.ckpt}  (device: {device})')


# ── Run inference + metrics ─────────────────────────────────────────────────
REGIONS    = define_regions()
# Aligned with the NC data's actual coordinates (NC has lat = [-89, -87, ..., 87, 89]).
# Previously using linspace(-90, 90, 90) caused polar weight = cos(±90°) = 0,
# meaning polar pixels were entirely excluded from RMSE.
lat_values = np.linspace(-89, 89, 90)

print('\n' + '=' * 60)
print(f'Evaluation [{model_name}]  (train_frac={train_frac})')
print(f'Protocol: last-frame of each test window  (matches MCO-242 schema)')
print('=' * 60)

results = {}
truth_pred_cache = {}    # reused by the spatial PDF section
n_test_per_dataset = {}

for tag, cfg_d in REGISTRY.items():
    split_label = 'OOD' if cfg_d['holdout'] else 'ID '
    # Get test input depending on dataset_kind
    tr, va, te_a, te_b, _ = splits[tag]
    if dataset_kind == 'single_frame':
        test_input = (te_a, te_b)               # (eval_x, eval_y)
        n_samples  = len(te_a)
    else:
        test_input = te_a                        # test_ds (SequenceDataset)
        n_samples  = len(te_a)
    n_test_per_dataset[tag] = n_samples

    truth, pred = run_inference_unified(model, test_input, device,
                                         batch_size=batch_size,
                                         dataset_kind=dataset_kind)
    truth_pred_cache[tag] = (truth, pred)
    metrics = compute_metrics(truth, pred, lat_values, REGIONS)
    results[tag] = {'co2': cfg_d['co2'], 'split': split_label.strip(),
                    'n_test': n_samples, **metrics}
    o = metrics['overall']
    print(f"  {tag:8s} ({cfg_d['co2']:4d} ppm) [{split_label}]  "
          f"overall  R2={o['R2']:+.4f}  MSE={o['MSE']:.4f}  "
          f"RMSE={o['RMSE']:.4f}  MAE={o['MAE']:.4f}")
    for region_name in ('R01', 'R03', 'R08'):
        r = metrics['regions'][region_name]
        info = REGIONS[region_name]
        print(f'           {region_name} (lat {info["lat_deg"][0]:+.0f}~{info["lat_deg"][1]:+.0f}, '
              f'lon {info["lon_deg"][0]:.0f}~{info["lon_deg"][1]:.0f}): '
              f"R2={r['R2']:+.4f}  RMSE={r['RMSE']:.4f}")

n_test_total = sum(n_test_per_dataset.values())
results['_meta'] = {
    'model':              model_name,
    'model_class':        model_class_path,
    'n_params':           int(n_params),
    'dataset_kind':       dataset_kind,
    'input_features':     input_features,
    'n_inputs':           n_inputs,
    'seq_len':            seq_len,
    'train_frac':         train_frac,
    'n_test_total':       int(n_test_total),
    'n_test_per_dataset': n_test_per_dataset,
    'eval_protocol':      'rolling-origin (last frame of each test window)',
    'ckpt':               os.path.abspath(args.ckpt),
}

with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved eval -> {output_dir}/eval_results.json')


# ── 1. Result-only viz (R²/RMSE/MAE global maps + .nc) ─────────────────────
#       This part is identical across models (only uses truth/pred)
draw_results(truth_pred_cache, output_dir, registry=REGISTRY)


# ── 2. Model-internal viz (attention / FiLM / climate state, etc.) ─────────
#       Done by each model's own draw_inner() method; skipped if unsupported
if args.save_attn:
    if hasattr(model, 'draw_inner') and callable(model.draw_inner):
        print(f'\n[--save_attn] {model_class_path}.draw_inner() — running model-internal viz')
        # Build test_sets_dict (consistent with model.dataset_kind)
        test_sets_dict = {}
        for tag in REGISTRY:
            tr, va, te_a, te_b, _ = splits[tag]
            if dataset_kind == 'single_frame':
                test_sets_dict[tag] = (te_a, te_b)
            else:
                test_sets_dict[tag] = te_a
        try:
            model.draw_inner(test_sets_dict, save_dir=output_dir,
                             registry=REGISTRY, batch_size=batch_size)
        except Exception as e:
            print(f'[warn] draw_inner failed: {type(e).__name__}: {e}')
    else:
        print(f'\n[--save_attn] {model_class_path} has no draw_inner() — skipped '
              f'(model-internal viz exists only on some models, e.g. IsoUNet)')


_total_dur = time.perf_counter() - _T0_PERF
print(f'\n{"="*60}')
print(f'[{datetime.datetime.now()}] eval finished')
print(f'  total wall: {_fmt_duration(_total_dur)}')
print(f'  outputs   : {output_dir}/')
print('=' * 60)
