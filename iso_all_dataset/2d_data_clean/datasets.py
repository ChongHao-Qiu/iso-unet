"""
datasets.py — two dataset types:
    1. SingleFrameDataset (dataset_kind='single_frame')
       Each sample is (x_t, y_t, co2_scalar). Used by single-frame baselines.
    2. SequenceDataset (dataset_kind='sequence')  ← from iso_unet.data
       Each sample is (x_window, y_seq, co2_scalar). Used by sequence models.

Same logic as the legacy scripts — this module just wraps "build splits" in a
unified interface shared by train.py / eval.py.
"""
import torch
from data import (REGISTRY, chrono_split, get_input_features, load_xr,
                  normalize_inputs, normalize_inputs_with_stats,
                  LEGACY_SKIP_NORM,
                  xr_to_tensors, format_norm_log)

# Sequence dataset — provided by the iso_unet package
from iso_unet.data import SequenceDataset


# ── Single-frame dataset (returns co2 for prompt loss; models that don't use it ignore it) ──
class SingleFrameDataset(torch.utils.data.Dataset):
    """Single-frame (x_t, y_t, co2_ppm) triple."""
    def __init__(self, x, y, co2_ppm):
        assert len(x) == len(y), f'len mismatch: x={len(x)}, y={len(y)}'
        self.x   = x        # (N, C, H, W)
        self.y   = y        # (N, 1, H, W)
        self.co2 = torch.tensor(co2_ppm, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.co2


# ── Unified build_splits ────────────────────────────────────────────────────
def build_splits(tag, raw_ds, train_frac, seq_len, input_features,
                 dataset_kind='single_frame', ref_stats=None):
    """
    Returns, depending on dataset_kind:
      single_frame  → (train_ds, valid_ds, eval_x, eval_y, stats)
        * train/val use SingleFrameDataset
        * eval_x/eval_y are tensors from the test slice with the first seq_len-1 frames skipped
          (aligned with MCO-242 last-frame eval)
      sequence      → (train_ds, valid_ds, test_ds, None, stats)
        * train/val/test all use SequenceDataset (returns the full seq_len-frame y)

    ref_stats:
        None  → per-dataset z-score (legacy behavior; each dataset computes its own μ/σ)
        dict  → global z-score; normalize all datasets using the (μ, σ) in the dict
    """
    cfg = REGISTRY[tag]
    co2 = cfg['co2']
    x, y = xr_to_tensors(raw_ds, input_features)
    n_time = x.shape[0]
    sl_tr, sl_val, sl_test = chrono_split(n_time, train_frac)

    x_tr,  y_tr  = x[sl_tr].clone(),   y[sl_tr].clone()
    x_val, y_val = x[sl_val].clone(),  y[sl_val].clone()
    x_te,  y_te  = x[sl_test].clone(), y[sl_test].clone()
    if x_tr.shape[0] < max(1, seq_len):
        raise ValueError(f'{tag}: train slice ({x_tr.shape[0]}) < seq_len ({seq_len}) '
                         f'(train_frac={train_frac} is too small)')

    if ref_stats is None:
        # per_dataset (legacy) — skip tas/LANDFRAC/aice; bit-level equivalent to pre-change behavior
        stats = normalize_inputs(x_tr, x_val, x_te, input_features, skip=LEGACY_SKIP_NORM)
    else:
        # global — only skip LANDFRAC/aice; tas is also normalized
        stats = normalize_inputs_with_stats(x_tr, x_val, x_te, input_features, ref_stats)

    if dataset_kind == 'single_frame':
        train_ds = SingleFrameDataset(x_tr,  y_tr,  co2_ppm=co2)
        valid_ds = SingleFrameDataset(x_val, y_val, co2_ppm=co2)
        # Test set: skip the first (seq_len-1) frames so eval frames align with MCO-242 last-frame setup
        offset = seq_len - 1
        eval_x = x_te[offset:]
        eval_y = y_te[offset:]
        log = format_norm_log(stats, tag, co2, cfg['holdout'],
                              len(train_ds), len(valid_ds), len(eval_x))
        return train_ds, valid_ds, eval_x, eval_y, stats, log

    elif dataset_kind == 'sequence':
        train_ds = SequenceDataset(x_tr,  y_tr,  co2_ppm=co2, seq_len=seq_len)
        valid_ds = SequenceDataset(x_val, y_val, co2_ppm=co2, seq_len=seq_len)
        test_ds  = SequenceDataset(x_te,  y_te,  co2_ppm=co2, seq_len=seq_len)
        log = format_norm_log(stats, tag, co2, cfg['holdout'],
                              len(train_ds), len(valid_ds), len(test_ds))
        return train_ds, valid_ds, test_ds, None, stats, log

    else:
        raise ValueError(f"dataset_kind must be 'single_frame' or 'sequence', got {dataset_kind}")
