"""
benchmark_gpu_mem.py
─────────────────────────────────────────────────────────────────────────────
STANDALONE peak-GPU-memory benchmark for all baselines.

  - Does NOT touch train.py / draw_feature.py / any existing code.
  - Uses synthetic random tensors of the right shape — does NOT load the
    real iso dataset, so there's no I/O overhead.
  - "Fake-trains" each model for N epochs × M steps with Adam,
    measuring torch.cuda.max_memory_allocated / max_memory_reserved.
  - Resets CUDA stats between baselines so measurements are independent.

Output: a printed table + (optional) CSV.

Usage:
  python benchmark_gpu_mem.py                            # default baseline list
  python benchmark_gpu_mem.py --configs climax,stormer   # subset
  python benchmark_gpu_mem.py --n_epochs 5 --n_steps 10
  python benchmark_gpu_mem.py --batch_size 4             # override YAML batch
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from models import build_model                                         # noqa: E402


CONFIGS_DIR = os.path.join(_SCRIPT_DIR, 'configs')

# Default list — every "headline" baseline that has a config. Add/remove freely.
DEFAULT_CONFIGS = [
    'unet2d_vanilla_full_split',
    'attunet_full_split',
    'dcsaunet_full_split',
    'canet_full_split',
    'unetpp_full_split',
    'transunet_full_split',
    'segformer_full_split',
    'uno_full_split',
    'unet3d_temporal_full_split',
    'climax_full_split',
    'stormer_full_split',
    'iso_unet_4way_ice_full_split',
]


def benchmark_one(cfg_path, n_epochs=5, n_steps=10, batch_size_override=None,
                   device=None, verbose=True):
    """Fake-train one config, return dict with peak-memory stats."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mkw = dict(cfg['model_kwargs'])
    mkw['n_inputs'] = 9
    dataset_kind = cfg.get('dataset_kind', 'sequence')
    seq_len      = int(cfg.get('seq_len', 12))
    batch_size   = int(batch_size_override or cfg.get('batch_size', 4))

    model = build_model(cfg['model_class'], model_kwargs=mkw, weights=None)
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Reset before measurement
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Synthetic shapes
    if dataset_kind == 'sequence':
        x_shape = (batch_size, seq_len, 9, 90, 180)
    else:
        x_shape = (batch_size, 9, 90, 180)
    y_shape = (batch_size, 1, 90, 180)

    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        for step in range(n_steps):
            x = torch.randn(*x_shape, device=device)
            y = torch.randn(*y_shape, device=device)
            optimizer.zero_grad()
            y_hat = model(x)
            if y_hat.dim() == 5:               # seq→seq baselines (UNet3D)
                y_hat = y_hat[:, -1]
            loss = F.mse_loss(y_hat, y)
            loss.backward()
            optimizer.step()
    dt = time.perf_counter() - t0

    if device.type == 'cuda':
        peak_alloc    = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(device)  / (1024 ** 3)
    else:
        peak_alloc = peak_reserved = float('nan')

    # Clean up so the next baseline starts fresh
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return {
        'model_name'       : cfg.get('model_name', os.path.basename(cfg_path)),
        'model_class'      : cfg['model_class'].split('.')[-1],
        'dataset_kind'     : dataset_kind,
        'n_params_M'       : n_params / 1e6,
        'batch_size'       : batch_size,
        'seq_len'          : seq_len if dataset_kind == 'sequence' else 1,
        'peak_alloc_GB'    : peak_alloc,
        'peak_reserved_GB' : peak_reserved,
        'wall_time_s'      : dt,
        'n_epochs'         : n_epochs,
        'n_steps_per_epoch': n_steps,
    }


def resolve_config_paths(names_or_paths):
    """Accept either 'climax' / 'climax_full_split' / 'climax_full_split.yaml'
    or an absolute path. Returns list of absolute YAML paths."""
    out = []
    for n in names_or_paths:
        n = n.strip()
        if not n:
            continue
        if os.path.isabs(n):
            out.append(n)
            continue
        if not n.endswith('.yaml'):
            n = n + '.yaml'
        full = os.path.join(CONFIGS_DIR, n)
        if not os.path.exists(full):
            # try as substring of any DEFAULT_CONFIGS entry
            matches = [c for c in DEFAULT_CONFIGS if n.replace('.yaml', '') in c]
            if len(matches) == 1:
                full = os.path.join(CONFIGS_DIR, matches[0] + '.yaml')
            else:
                raise FileNotFoundError(
                    f'Cannot resolve config "{n}". Tried "{full}". '
                    f'Matches in DEFAULT_CONFIGS: {matches}')
        out.append(full)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs',    type=str, default=None,
                   help='Comma-separated YAML names (e.g. "climax,stormer"). '
                        'Default = the full DEFAULT_CONFIGS list.')
    p.add_argument('--n_epochs',   type=int, default=5)
    p.add_argument('--n_steps',    type=int, default=10,
                   help='Synthetic-batch steps per epoch (default 10).')
    p.add_argument('--batch_size', type=int, default=None,
                   help='Override the YAML batch_size. None = use YAML.')
    p.add_argument('--out',        type=str,
                   default=os.path.join(_SCRIPT_DIR, 'gpu_mem_benchmark.csv'),
                   help='CSV output path. Set to "" to skip CSV.')
    args = p.parse_args()

    names = ([s for s in args.configs.split(',')] if args.configs
             else DEFAULT_CONFIGS[:])
    paths = resolve_config_paths(names)

    if not torch.cuda.is_available():
        print('[warn] CUDA not available — benchmark will be CPU-only and meaningless.')
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f'GPU: {torch.cuda.get_device_name(0)}  '
              f'(total {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)')

    print(f'Configs: {len(paths)}, fake-training {args.n_epochs} epochs '
          f'× {args.n_steps} steps each, batch override = {args.batch_size}\n')

    results = []
    for path in paths:
        name = os.path.basename(path).replace('.yaml', '')
        print(f'  {name:42s} ', end='', flush=True)
        try:
            r = benchmark_one(path, n_epochs=args.n_epochs,
                              n_steps=args.n_steps,
                              batch_size_override=args.batch_size,
                              device=device, verbose=False)
            results.append({'config': name, **r})
            print(f'OK  params={r["n_params_M"]:6.2f}M  bs={r["batch_size"]}  '
                  f'alloc={r["peak_alloc_GB"]:6.2f}GB  '
                  f'reserved={r["peak_reserved_GB"]:6.2f}GB  '
                  f'wall={r["wall_time_s"]:5.1f}s')
        except Exception as e:
            print(f'FAIL  {type(e).__name__}: {e}')
            traceback.print_exc()
            results.append({'config': name, 'error': str(e)})
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Pretty table
    print()
    print('=' * 116)
    print(f'{"config":<42} {"class":<28} {"params":>10} {"batch":>6} '
          f'{"alloc":>10} {"reserved":>10} {"wall":>8}')
    print('-' * 116)
    for r in results:
        if 'error' in r:
            print(f'{r["config"]:<42} ERROR: {r["error"][:60]}')
            continue
        print(f'{r["config"]:<42} {r["model_class"]:<28} '
              f'{r["n_params_M"]:>9.2f}M {r["batch_size"]:>6} '
              f'{r["peak_alloc_GB"]:>8.2f}GB {r["peak_reserved_GB"]:>8.2f}GB '
              f'{r["wall_time_s"]:>6.1f}s')
    print('=' * 116)

    # CSV
    if args.out:
        ok_results = [r for r in results if 'error' not in r]
        fieldnames = ['config', 'model_class', 'dataset_kind',
                      'n_params_M', 'batch_size', 'seq_len',
                      'peak_alloc_GB', 'peak_reserved_GB', 'wall_time_s',
                      'n_epochs', 'n_steps_per_epoch']
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in ok_results:
                writer.writerow({k: r.get(k, '') for k in fieldnames})
        print(f'\nCSV saved: {args.out}  ({len(ok_results)} rows)')


if __name__ == '__main__':
    main()
