"""
train.py — Universal training entry. Reads YAML config, trains, and saves.

Usage:
    python train.py --config configs/unet2d_vanilla_full_split.yaml \\
                    --save_base ./experiments

Outputs:
    <save_base>/<model_name>/<exp_tag>/
        model.pt        ← trained weights
        config.yaml     ← training config (read by eval.py)
        train.log       ← training log
        loss_curve.png  ← loss curve

Design:
    * Single-frame and sequence models share the same script; selected by config.dataset_kind
    * Per-model-specific parameters live under YAML's model_kwargs
    * The CLI only exposes parameters that vary across models (train_frac / exp_tag / lr / ...)
"""
import os, sys, glob, json, argparse, datetime, time
import yaml
import numpy as np
import torch
import lightning as L
import iso_unet as ig


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
parser.add_argument('--config',         type=str,   required=True,
                    help='Path to YAML config (e.g. configs/unet2d.yaml)')
parser.add_argument('--exp_tag',        type=str,   default=None,
                    help='Override exp_tag (default: <model_name>-tf<NN>)')
parser.add_argument('--save_base',      type=str,
                    default='./experiments')
parser.add_argument('--train_frac',     type=float, default=None)
parser.add_argument('--batch_size',     type=int,   default=None)
parser.add_argument('--lr',             type=float, default=None)
parser.add_argument('--max_epochs',     type=int,   default=None)
parser.add_argument('--patience',       type=int,   default=None)
parser.add_argument('--max_batch',      type=int,   default=None,
                    help='max time steps per dataset (debug, default from config)')
parser.add_argument('--seed',           type=int,   default=42)
parser.add_argument('--ood_preset',     type=str,   default=None,
                    help="OOD split preset: 'default'/'low_ood'/'high_ood'/'mid_ood' "
                         "(overrides config; rare). Normally set via YAML 'ood_preset:' field.")
parser.add_argument('--no_lat_weights', action='store_true',
                    help='Disable cos(lat) weighting in training loss. Default: weighted.')
args = parser.parse_args()

# Add this dir to path so we can import data.py / models.py / metrics.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data     import (REGISTRY, OOD_PRESETS, apply_ood_preset,
                      REF_DATASET, compute_ref_stats,
                      get_input_features, load_xr)
from datasets import build_splits
from models   import build_model, get_dataset_kind, resolve_model_kwargs
from metrics  import make_lat_weights


# ── Load YAML config ────────────────────────────────────────────────────────
with open(args.config) as f:
    cfg = yaml.safe_load(f)

# CLI overrides
def _set(key, val):
    if val is not None: cfg[key] = val
_set('train_frac', args.train_frac)
_set('batch_size', args.batch_size)
_set('lr',         args.lr)
_set('max_epochs', args.max_epochs)
_set('patience',   args.patience)
_set('max_batch',  args.max_batch)
_set('ood_preset', args.ood_preset)

# Apply defaults
cfg.setdefault('train_frac',     0.10)
cfg.setdefault('batch_size',     8)
cfg.setdefault('lr',             1e-3)
cfg.setdefault('max_epochs',     1000)
cfg.setdefault('patience',       30)
cfg.setdefault('max_batch',      3000)
cfg.setdefault('seq_len',        12)
cfg.setdefault('gradient_clip_val', 1.0)
cfg.setdefault('input_set',      'basic')
cfg.setdefault('model_kwargs',   {})
cfg.setdefault('dataset_kind',   None)
cfg.setdefault('ood_preset',     'default')
cfg.setdefault('norm_mode',      'global')    # 'global' (default, μ/σ from REF_DATASET) | 'per_dataset' (legacy)
cfg.setdefault('norm_ref',       REF_DATASET) # which dataset to use as normalization reference (default '280ppm')
cfg.setdefault('lat_weighted',   True)        # True = cos(lat) area-weighted MSE; False = unweighted
if args.no_lat_weights:
    cfg['lat_weighted'] = False

# ── Apply OOD preset (mutates REGISTRY's holdout flags in place) ───────────
_ood_preset_name, _ood_tags, _id_tags = apply_ood_preset(cfg['ood_preset'])

model_name        = cfg['model_name']
model_class_path  = cfg['model_class']
input_features    = get_input_features(cfg['input_set'])
n_inputs          = len(input_features)
dataset_kind      = get_dataset_kind(model_class_path, cfg.get('dataset_kind'))
seq_len           = int(cfg['seq_len'])
train_frac        = float(cfg['train_frac'])

if args.exp_tag is not None:
    exp_tag = args.exp_tag
else:
    exp_tag = f'{model_name}-tf{int(round(train_frac * 100)):02d}'
    # Non-default preset → append to exp_tag so different OOD splits don't overwrite each other
    if cfg['ood_preset'] != 'default':
        exp_tag += f"-{cfg['ood_preset']}"
    # Unweighted training loss → append `-uw` (default is weighted, no suffix)
    if not cfg['lat_weighted']:
        exp_tag += '-uw'
    # Non-default 4-way mask_mode → append suffix (for ISO-UNet 4-way only)
    # mask_mode: 'none' (no suffix) | 'hard' (-mhard) | 'soft' (-msoft) | 'soft_boost' (-msbo)
    _mask_mode = cfg.get('model_kwargs', {}).get('mask_mode', 'none')
    exp_tag += {'none': '', 'hard': '-mhard',
                'soft': '-msoft', 'soft_boost': '-msbo'}.get(_mask_mode, '')
    # Non-default climate-state time pooling → suffix (for ISO-UNet only)
    # time_pool: 'mean' (no suffix, default) | 'gru' (-tgru) | 'conv3d' (-tcnv3d)
    _time_pool = cfg.get('model_kwargs', {}).get('time_pool', 'mean')
    exp_tag += {'mean': '', 'gru': '-tgru', 'conv3d': '-tcnv3d'}.get(_time_pool, '')
    # Non-default seed → append `-s{seed}` so multi-seed runs don't collide
    if args.seed != 42:
        exp_tag += f"-s{args.seed}"

SAVE_PATH = os.path.join(args.save_base, model_name, exp_tag)
os.makedirs(SAVE_PATH, exist_ok=True)

# Tee stdout/stderr to train.log
_log_path = os.path.join(SAVE_PATH, 'train.log')
_log_file = open(_log_path, 'w', buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

L.seed_everything(args.seed, workers=True)

print(f'[{datetime.datetime.now()}] train.log -> {_log_path}')
print(f'CONFIG       = {args.config}')
print(f'MODEL        = {model_name}  ({model_class_path})')
print(f'EXP_TAG      = {exp_tag}')
print(f'SAVE_PATH    = {SAVE_PATH}')
print(f'TRAIN_FRAC   = {train_frac}')
print(f'BATCH_SIZE   = {cfg["batch_size"]}, LR = {cfg["lr"]}, MAX_EPOCHS = {cfg["max_epochs"]}')
print(f'INPUT_SET    = {cfg["input_set"]}  ({n_inputs} ch: {input_features})')
print(f'DATASET_KIND = {dataset_kind},  SEQ_LEN = {seq_len}')
print(f'OOD_PRESET   = {_ood_preset_name}')
print(f'   ID  ({len(_id_tags )} datasets) = {_id_tags}')
print(f'   OOD ({len(_ood_tags)} datasets) = {_ood_tags}')
print(f'NORM_MODE    = {cfg["norm_mode"]}'
      + (f"  (ref={cfg['norm_ref']})" if cfg['norm_mode'] == 'global' else ''))
print(f'LAT_WEIGHTED = {cfg["lat_weighted"]}'
      + ('  (training MSE = area-weighted by cos(lat))' if cfg['lat_weighted']
         else '  ⚠️ training MSE = unweighted (each pixel counts equally)'))
print(f'MODEL_KWARGS = {cfg["model_kwargs"]}')


# ── Load + build splits ─────────────────────────────────────────────────────
print('\nLoading datasets...')
raw = {}
for tag in REGISTRY:
    raw[tag] = load_xr(tag, max_batch=cfg['max_batch'], input_features=input_features)
    print(f'  {tag}: {len(raw[tag].time)} time steps')

# ── Compute ref stats for global normalization (if norm_mode='global') ─────
ref_stats = None
if cfg['norm_mode'] == 'global':
    ref_tag = cfg['norm_ref']
    assert ref_tag in REGISTRY, (
        f"norm_ref='{ref_tag}' not in REGISTRY. Available: {list(REGISTRY)}"
    )
    ref_stats = compute_ref_stats(raw[ref_tag], input_features, train_frac)
    print(f'\nGlobal normalization stats (from {ref_tag} train slice, '
          f'applied to ALL datasets):')
    for f, s in ref_stats.items():
        print(f'  {f:10s} μ={s["mu"]:+.4e}  σ={s["sd"]:.4e}')
    cfg['norm_stats'] = ref_stats   # save into config.yaml for eval to read
elif cfg['norm_mode'] == 'per_dataset':
    print(f"\nUsing per-dataset normalization (legacy mode) — each dataset's "
          f"μ/σ comes from its own train slice")
else:
    raise ValueError(
        f"norm_mode must be 'global' or 'per_dataset', got '{cfg['norm_mode']}'"
    )

print(f'\nBuilding splits (dataset_kind={dataset_kind}, train_frac={train_frac})...')
splits = {}
for tag in REGISTRY:
    tr, va, te_a, te_b, stats, log = build_splits(
        tag, raw[tag], train_frac, seq_len, input_features, dataset_kind,
        ref_stats=ref_stats)
    splits[tag] = (tr, va, te_a, te_b, stats)
    print(log)


# ── DataModule ──────────────────────────────────────────────────────────────
class _DM(L.LightningDataModule):
    def __init__(self, splits, registry, batch_size):
        super().__init__()
        self.splits = splits; self.registry = registry; self.batch_size = batch_size
    def _concat(self, which):
        idx = {'train': 0, 'valid': 1}[which]
        subsets = [self.splits[t][idx] for t, c in self.registry.items() if not c['holdout']]
        return torch.utils.data.ConcatDataset(subsets)
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self._concat('train'),
            batch_size=self.batch_size, shuffle=True, num_workers=0)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self._concat('valid'),
            batch_size=self.batch_size, shuffle=False, num_workers=0)

dm = _DM(splits, REGISTRY, batch_size=cfg['batch_size'])
n_tr  = sum(len(splits[t][0]) for t, c in REGISTRY.items() if not c['holdout'])
n_val = sum(len(splits[t][1]) for t, c in REGISTRY.items() if not c['holdout'])
print(f'\nCombined train={n_tr},  val={n_val}')


# ── Build model ─────────────────────────────────────────────────────────────
lat_weights = make_lat_weights()
lat_weights_t = torch.as_tensor(lat_weights, dtype=torch.float32)

# Resolve the kwargs that depend on channel ordering (landfrac_idx / pr_channels /
# ice_idx). Shared with any standalone script that rebuilds a model from a config.
mkw = resolve_model_kwargs(cfg['model_kwargs'], input_features, lr=cfg['lr'])

# Training loss weight — cfg['lat_weighted']=True: weight by cos(lat); False: unweighted (model._loss takes the else branch)
_weights_for_model = lat_weights_t if cfg['lat_weighted'] else None
model = build_model(model_class_path, model_kwargs=mkw, weights=_weights_for_model)
n_params = sum(p.numel() for p in model.parameters())
print(f'\nModel: {model_class_path}  params: {n_params:,}')

# Save config alongside model.pt (eval.py reads this)
final_cfg = dict(cfg)
final_cfg['exp_tag']            = exp_tag
final_cfg['input_features']     = input_features
final_cfg['n_inputs']           = n_inputs
final_cfg['dataset_kind']       = dataset_kind
final_cfg['n_params']           = int(n_params)
final_cfg['model_kwargs']       = mkw
with open(os.path.join(SAVE_PATH, 'config.yaml'), 'w') as f:
    yaml.safe_dump(final_cfg, f, sort_keys=False, allow_unicode=True)
print(f'config saved -> {SAVE_PATH}/config.yaml')


# ── Train ───────────────────────────────────────────────────────────────────
trainer = ig.Trainer(
    log_dirpath='./logs',
    name=f'IsoGen-{exp_tag}',
    patience=cfg['patience'],
    min_epochs=1,
    max_epochs=cfg['max_epochs'],
    gradient_clip_val=cfg['gradient_clip_val'],
    enable_progress_bar=False,
    simple_epoch_log=True,
    simple_epoch_every=5,
)
_t_train_begin = time.perf_counter()
print(f'\n[{datetime.datetime.now()}] training started')
trainer.fit(model, datamodule=dm)
_train_dur  = time.perf_counter() - _t_train_begin
_last_epoch = trainer.current_epoch
print(f'[{datetime.datetime.now()}] training ended  '
      f'({_fmt_duration(_train_dur)}, {_last_epoch + 1} epochs)')

torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model.pt'))
print(f'Model saved -> {SAVE_PATH}/model.pt')


# ── Loss curve ──────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_vers = sorted(glob.glob('./logs/lightning_logs/version_*'), key=os.path.getmtime)
if _vers:
    try:
        _df = pd.read_csv(os.path.join(_vers[-1], 'metrics.csv'))
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        cols = [(c, c.replace('train_', 'Train ').replace('valid_', 'Val '))
                for c in _df.columns if c.startswith(('train_', 'valid_'))
                and not c.endswith(('_step',))]
        for col, lbl in cols:
            sub = _df[_df[col].notna()][['epoch', col]]
            if len(sub):
                axes[0].plot(sub['epoch'], sub[col], label=lbl, linewidth=1.2)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{exp_tag} — loss')
        axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
        try: axes[0].set_yscale('log')
        except Exception: pass
        if 'lambda_prompt' in _df.columns:
            sub = _df[_df['lambda_prompt'].notna()][['epoch', 'lambda_prompt']]
            if len(sub):
                axes[1].plot(sub['epoch'], sub['lambda_prompt'], color='#f0ad4e', marker='o', markersize=3)
                axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('λ prompt')
                axes[1].set_title('Prompt-loss weight')
                axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH, 'loss_curve.png'), dpi=140)
        plt.close()
        print(f'Loss curve -> {SAVE_PATH}/loss_curve.png')
    except Exception as e:
        print(f'[warn] loss curve plot failed: {e}')


_total_dur = time.perf_counter() - _T0_PERF
print(f'\n{"="*60}')
print(f'[{datetime.datetime.now()}] training script finished')
print(f'  total wall: {_fmt_duration(_total_dur)}')
print(f'  training:   {_fmt_duration(_train_dur)} ({_last_epoch + 1} epochs)')
print(f'  next step:  python eval.py --ckpt {SAVE_PATH}/model.pt')
print('=' * 60)
