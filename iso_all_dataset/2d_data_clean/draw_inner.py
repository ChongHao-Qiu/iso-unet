"""
draw_inner.py — Standalone "model-internal viz" runner for ISO-UNet ckpts.

Runs model.draw_inner() WITHOUT touching:
  - eval.log     (creates its own draw.log)
  - eval_results.json
  - spatial/predict_*  result-only PDFs

Writes to:
  <ckpt_dir>/<out_subdir>/spatial/           ← all attention / expert / sample PDFs
  <ckpt_dir>/draw.log                         ← its own log

Usage:
  python draw_inner.py --ckpt /path/to/model.pt
  python draw_inner.py --ckpt ... --out_subdir my_viz   # custom subdir name
  python draw_inner.py --ckpt ... --n_samples 10        # more per-sample plots
"""
import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
import yaml
import lightning as L


# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Standalone model-internal viz runner')
parser.add_argument('--ckpt',         type=str, required=True,
                    help='trained model checkpoint (model.pt)')
parser.add_argument('--config',       type=str, default=None,
                    help='Override config path (default: <ckpt_dir>/config.yaml)')
parser.add_argument('--out_subdir',   type=str, default='attention',
                    help='Subdir under ckpt_dir for output (default: "attention").'
                         ' Final write path = <ckpt_dir>/<out_subdir>/spatial/')
parser.add_argument('--batch_size',   type=int, default=None,
                    help='Override eval batch size (default: from config or 4)')
parser.add_argument('--n_samples',    type=int, default=6,
                    help='Per-sample viz: K random windows per dataset (default: 6)')
parser.add_argument('--seed',         type=int, default=42)
args = parser.parse_args()


# ── Tee stdout/stderr to draw.log (NOT eval.log!) ─────────────────────────
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()


# Resolve paths
ckpt_path   = os.path.abspath(args.ckpt)
ckpt_dir    = os.path.dirname(ckpt_path)
config_path = args.config or os.path.join(ckpt_dir, 'config.yaml')
out_dir     = os.path.join(ckpt_dir, args.out_subdir)   # ★ separate dir
os.makedirs(out_dir, exist_ok=True)
log_path    = os.path.join(ckpt_dir, 'draw.log')         # ★ separate log

_log_file = open(log_path, 'w', buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)
print(f'[{datetime.datetime.now()}] draw.log -> {log_path}')
print(f'OUTPUT_DIR  = {out_dir}/spatial/   (separate from eval output)')
print(f'CKPT        = {ckpt_path}')

# Add this dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data      import (REGISTRY, apply_ood_preset,
                       REF_DATASET, compute_ref_stats,
                       get_input_features, load_xr)
from datasets  import build_splits
from models    import build_model, get_dataset_kind
from metrics   import make_lat_weights


L.seed_everything(args.seed, workers=True)

# ── Load config ─────────────────────────────────────────────────────────────
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

batch_size       = args.batch_size or cfg['batch_size']
model_name       = cfg['model_name']
model_class_path = cfg['model_class']
input_features   = cfg.get('input_features') or get_input_features(cfg['input_set'])
n_inputs         = len(input_features)
dataset_kind     = get_dataset_kind(model_class_path, cfg.get('dataset_kind'))
seq_len          = int(cfg['seq_len'])
train_frac       = float(cfg['train_frac'])

# Apply OOD preset (mutate REGISTRY)
_ood_name, _ood_tags, _id_tags = apply_ood_preset(cfg['ood_preset'])

print(f'MODEL       = {model_name}')
print(f'OOD_PRESET  = {_ood_name}  (ID={_id_tags}, OOD={_ood_tags})')
print(f'TRAIN_FRAC  = {train_frac}')
print(f'NORM_MODE   = {cfg["norm_mode"]}')
print(f'LAT_WEIGHT  = {cfg["lat_weighted"]}')


# ── Load data + build splits (same as eval.py) ───────────────────────────
print('\nLoading datasets...')
raw = {}
for tag in REGISTRY:
    raw[tag] = load_xr(tag, max_batch=cfg['max_batch'], input_features=input_features)

# Norm stats
ref_stats = None
if cfg['norm_mode'] == 'global':
    if 'norm_stats' in cfg and cfg['norm_stats']:
        ref_stats = cfg['norm_stats']
        print(f'[norm] using saved global stats from config.yaml')
    else:
        ref_stats = compute_ref_stats(raw[cfg['norm_ref']], input_features, train_frac)
        print(f'[norm] recomputed global stats from {cfg["norm_ref"]} train slice')

print(f'\nBuilding splits...')
splits = {}
for tag in REGISTRY:
    tr, va, te_a, te_b, stats, log = build_splits(
        tag, raw[tag], train_frac, seq_len, input_features, dataset_kind,
        ref_stats=ref_stats)
    splits[tag] = (tr, va, te_a, te_b, stats)
    print(log)


# ── Build model + load weights ─────────────────────────────────────────────
lat_weights_t = (torch.as_tensor(make_lat_weights(), dtype=torch.float32)
                 if cfg['lat_weighted'] else None)
mkw = dict(cfg['model_kwargs'])
mkw['n_inputs'] = n_inputs

model = build_model(model_class_path, model_kwargs=mkw, weights=lat_weights_t)
sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing:    print(f'  [warn] missing keys: {len(missing)} (first: {missing[:2]})')
if unexpected: print(f'  [warn] unexpected keys: {len(unexpected)} (first: {unexpected[:2]})')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f'\nLoaded {n_params:,} params from {ckpt_path}  (device: {device})')


# ── Run model.draw_inner() ─────────────────────────────────────────────────
if not hasattr(model, 'draw_inner'):
    print(f'\n[warn] model class {model.__class__.__name__} has no draw_inner() method.')
    print('       Nothing to draw. Exit.')
    sys.exit(0)

# Build test_sets_dict (same shape draw_inner expects)
test_sets_dict = {}
for tag in REGISTRY:
    tr, va, te_a, te_b, _ = splits[tag]
    if dataset_kind == 'single_frame':
        # eval.py wraps (eval_x, eval_y) tuple; for draw_inner sequence-style is preferred
        # Try to pass the SingleFrame test set if it exists, else (eval_x, eval_y)
        test_sets_dict[tag] = (te_a, te_b)
    else:
        test_sets_dict[tag] = te_a       # SequenceDataset

t0 = time.perf_counter()
print(f'\n[{datetime.datetime.now()}] running model.draw_inner()')
print(f'  output → {out_dir}/spatial/')
print(f'  n_samples per dataset = {args.n_samples}, batch_size = {batch_size}')

try:
    model.draw_inner(
        test_sets_dict=test_sets_dict,
        save_dir=out_dir,                     # ★ separate from eval's spatial/
        registry=REGISTRY,
        batch_size=batch_size,
        n_samples=args.n_samples,
        seed=args.seed,
    )
except Exception as e:
    import traceback
    print(f'\n[ERROR] draw_inner failed: {type(e).__name__}: {e}')
    print(traceback.format_exc())
    sys.exit(1)

dur = time.perf_counter() - t0
print(f'\n[{datetime.datetime.now()}] draw_inner done ({dur:.1f}s)')
print(f'\n{"=" * 60}')
print(f'✓ Saved figures to: {out_dir}/spatial/')
print(f'✓ Log saved to:     {log_path}')
print(f'✓ eval.log + eval_results.json + spatial/predict_*  NOT touched')
print(f'{"=" * 60}')
