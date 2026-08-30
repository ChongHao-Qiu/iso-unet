"""
_earthformer_cuboid.py
─────────────────────────────────────────────────────────────────────────────
PRIVATE helper for the three EarthFormer Cuboid Transformer baselines
(axial / divided_st / video_swin). Used by:
    baseline_axial.py
    baseline_divided_st.py
    baseline_video_swin.py

Centralises:
  1. Path discovery for the EarthFormer source tree (UCAR-hosted at
     `chenqi_isosim/isot/models/cfg/`). Tries:
       (a) $CHENQI_CUBOID  env var
       (b) iso_root/chenqi_isosim/isot/models/cfg
       (c) a few hardcoded fallback paths (Mac, UCAR HPC)
  2. The kwargs → CuboidTransformerModel constructor invocation, so the
     three baselines only need to specify `self_pattern` and forward their
     hyperparams.

EarthFormer is large external code (cuboid attention + dispatchers) — we do
NOT inline it. Instead the baselines load it via sys.path.
"""
from __future__ import annotations

import os
import sys


def _resolve_cuboid_src():
    """Locate the EarthFormer source tree (the `cfg/` directory whose
    `model/cuboid_transformer/earthformer/...` subtree contains the model
    code) and add it to sys.path. Priority:
      1. $CHENQI_CUBOID env var
      2. iso_root/chenqi_isosim/isot/models/cfg (relative to this file)
      3. Known absolute paths (UCAR HPC, local Mac)
      4. Already-importable `model.cuboid_transformer...` (no-op)
    """
    target_subpath = os.path.join(
        'model', 'cuboid_transformer', 'earthformer',
        'cuboid_transformer', 'cuboid_transformer.py',
    )

    candidates = []
    env_src = os.environ.get('CHENQI_CUBOID')
    if env_src:
        candidates.append(env_src)

    here = os.path.dirname(os.path.abspath(__file__))
    iso_root = os.path.abspath(os.path.join(here, '..', '..', '..'))
    candidates += [
        os.path.join(iso_root, 'chenqi_isosim', 'isot', 'models', 'cfg'),
        os.path.join(iso_root, '..', 'chenqi_isosim', 'isot', 'models', 'cfg'),
        '<EARTHFORMER_CFG_DIR>',                       # UCAR
        '<EARTHFORMER_CFG_DIR>',  # local
    ]

    for p in candidates:
        if p and os.path.exists(os.path.join(p, target_subpath)):
            if p not in sys.path:
                sys.path.insert(0, p)
            return p

    # Last try: maybe it's already on sys.path
    try:
        import model.cuboid_transformer.earthformer.cuboid_transformer.cuboid_transformer  # noqa: F401
        return None
    except ImportError:
        pass

    raise ImportError(
        'Cannot find EarthFormer source. Tried:\n  ' +
        '\n  '.join(c for c in candidates if c) +
        '\n\nFix one of:\n'
        '  (a) export CHENQI_CUBOID=/path/to/cfg     before running, or\n'
        '  (b) `pip install -e /path/to/cfg`        to register it, or\n'
        '  (c) git clone chenqi_isosim into <iso_root>/chenqi_isosim'
    )


# Trigger path discovery at import time
_EF_SRC = _resolve_cuboid_src()

from model.cuboid_transformer.earthformer.cuboid_transformer.cuboid_transformer import (  # noqa: E402
    CuboidTransformerModel,
)


# ── Hyperparam defaults (mirror the upstream YAML cfg_*_12to12.yaml) ─────
_CUBOID_DEFAULTS = dict(
    base_units                     = 64,
    block_units                    = None,
    scale_alpha                    = 1.0,
    enc_depth                      = (2, 2),
    dec_depth                      = (2, 2),
    enc_use_inter_ffn              = True,
    dec_use_inter_ffn              = True,
    dec_hierarchical_pos_embed     = False,
    downsample                     = 5,
    downsample_type                = 'patch_merge',
    upsample_type                  = 'upsample',
    num_heads                      = 2,
    attn_drop                      = 0.1,
    proj_drop                      = 0.1,
    ffn_drop                       = 0.1,
    num_global_vectors             = 0,
    use_dec_self_global            = False,
    dec_self_update_global         = True,
    use_dec_cross_global           = False,
    use_global_vector_ffn          = False,
    use_global_self_attn           = False,
    separate_global_qkv            = False,
    global_dim_ratio               = 1,
    ffn_activation                 = 'gelu',
    gated_ffn                      = False,
    norm_layer                     = 'layer_norm',
    padding_type                   = 'zeros',
    pos_embed_type                 = 't+hw',
    use_relative_pos               = True,
    self_attn_use_final_proj       = True,
    dec_use_first_self_attn        = False,
    z_init_method                  = 'zeros',
    initial_downsample_type        = 'conv',
    initial_downsample_activation  = 'leaky',
    initial_downsample_scale       = 6,
    initial_downsample_conv_layers = 2,
    final_upsample_conv_layers     = 1,
    checkpoint_level               = 0,
    attn_linear_init_mode          = '0',
    ffn_linear_init_mode           = '0',
    conv_init_mode                 = '0',
    down_up_linear_init_mode       = '0',
    norm_init_mode                 = '0',
    dec_cross_last_n_frames        = None,
    cross_pattern                  = 'cross_1x1',
)


def build_cuboid_model(self_pattern: str,
                        n_inputs: int = 9, out_channels: int = 1,
                        H: int = 90, W: int = 180, seq_len: int = 12,
                        cross_self_pattern: str = None,
                        **overrides) -> CuboidTransformerModel:
    """Build a CuboidTransformerModel matching the iso baseline configs.

    Args:
        self_pattern        : 'axial' | 'divided_st_axial' | 'video_swin'
                              — selects the encoder/decoder attention kernel.
        n_inputs, out_channels, H, W, seq_len : iso-side data shape.
        cross_self_pattern  : pattern for decoder cross-self attention.
                              Default = same as `self_pattern`.
        overrides           : any kwarg in _CUBOID_DEFAULTS may be overridden.
    """
    cfg = dict(_CUBOID_DEFAULTS)
    cfg.update(overrides)

    # Normalize ModuleList args
    enc_depth = list(cfg['enc_depth'])
    dec_depth = list(cfg['dec_depth'])
    num_blocks = len(enc_depth)

    if cross_self_pattern is None:
        cross_self_pattern = self_pattern
    cross_pattern = cfg['cross_pattern']

    enc_attn_patterns        = [self_pattern]        * num_blocks
    dec_self_attn_patterns   = [cross_self_pattern]  * num_blocks
    dec_cross_attn_patterns  = ([cross_pattern] * num_blocks
                                 if isinstance(cross_pattern, str)
                                 else list(cross_pattern))

    return CuboidTransformerModel(
        input_shape  = [seq_len, H, W, n_inputs],
        target_shape = [seq_len, H, W, out_channels],
        base_units                     = cfg['base_units'],
        block_units                    = cfg['block_units'],
        scale_alpha                    = cfg['scale_alpha'],
        enc_depth                      = enc_depth,
        dec_depth                      = dec_depth,
        enc_use_inter_ffn              = cfg['enc_use_inter_ffn'],
        dec_use_inter_ffn              = cfg['dec_use_inter_ffn'],
        dec_hierarchical_pos_embed     = cfg['dec_hierarchical_pos_embed'],
        downsample                     = cfg['downsample'],
        downsample_type                = cfg['downsample_type'],
        enc_attn_patterns              = enc_attn_patterns,
        dec_self_attn_patterns         = dec_self_attn_patterns,
        dec_cross_attn_patterns        = dec_cross_attn_patterns,
        dec_cross_last_n_frames        = cfg['dec_cross_last_n_frames'],
        dec_use_first_self_attn        = cfg['dec_use_first_self_attn'],
        num_heads                      = cfg['num_heads'],
        attn_drop                      = cfg['attn_drop'],
        proj_drop                      = cfg['proj_drop'],
        ffn_drop                       = cfg['ffn_drop'],
        upsample_type                  = cfg['upsample_type'],
        ffn_activation                 = cfg['ffn_activation'],
        gated_ffn                      = cfg['gated_ffn'],
        norm_layer                     = cfg['norm_layer'],
        num_global_vectors             = cfg['num_global_vectors'],
        use_dec_self_global            = cfg['use_dec_self_global'],
        dec_self_update_global         = cfg['dec_self_update_global'],
        use_dec_cross_global           = cfg['use_dec_cross_global'],
        use_global_vector_ffn          = cfg['use_global_vector_ffn'],
        use_global_self_attn           = cfg['use_global_self_attn'],
        separate_global_qkv            = cfg['separate_global_qkv'],
        global_dim_ratio               = cfg['global_dim_ratio'],
        initial_downsample_type        = cfg['initial_downsample_type'],
        initial_downsample_activation  = cfg['initial_downsample_activation'],
        initial_downsample_scale       = cfg['initial_downsample_scale'],
        initial_downsample_conv_layers = cfg['initial_downsample_conv_layers'],
        final_upsample_conv_layers     = cfg['final_upsample_conv_layers'],
        padding_type                   = cfg['padding_type'],
        z_init_method                  = cfg['z_init_method'],
        checkpoint_level               = cfg['checkpoint_level'],
        pos_embed_type                 = cfg['pos_embed_type'],
        use_relative_pos               = cfg['use_relative_pos'],
        self_attn_use_final_proj       = cfg['self_attn_use_final_proj'],
        attn_linear_init_mode          = cfg['attn_linear_init_mode'],
        ffn_linear_init_mode           = cfg['ffn_linear_init_mode'],
        conv_init_mode                 = cfg['conv_init_mode'],
        down_up_linear_init_mode       = cfg['down_up_linear_init_mode'],
        norm_init_mode                 = cfg['norm_init_mode'],
    )


# ── Shared Lightning-side methods (the 3 baselines paste these in) ────────
def make_cuboid_baseline_class(name: str, self_pattern: str):
    """Factory that builds a Lightning class for a given self_pattern.

    Used internally to avoid copy-pasting the same boilerplate across 3
    files. Each baseline_<x>.py still defines an explicit class (so that
    Lightning checkpoint resolution works on the class name) but its body
    is delegated here."""
    import torch
    import torch.nn.functional as F
    import lightning as L

    class _CuboidBaseline(L.LightningModule):
        def __init__(self,
                     n_inputs:     int = 9,
                     out_channels: int = 1,
                     H:            int = 90,
                     W:            int = 180,
                     seq_len:      int = 12,
                     weights             = None,
                     lr:           float = 1e-3,
                     **cuboid_kwargs):
            """All cuboid_kwargs are forwarded to build_cuboid_model — see
            _CUBOID_DEFAULTS for the full list of valid overrides."""
            super().__init__()
            self.save_hyperparameters(ignore=['weights'])
            self.lr      = lr
            self.seq_len = seq_len

            if weights is not None:
                self.register_buffer('lat_weights',
                                     torch.as_tensor(weights, dtype=torch.float32))
            else:
                self.lat_weights = None

            self.model = build_cuboid_model(
                self_pattern=self_pattern,
                n_inputs=n_inputs, out_channels=out_channels,
                H=H, W=W, seq_len=seq_len,
                **cuboid_kwargs,
            )

        def forward(self, x):
            # (B, T, C, H, W) → (B, T, H, W, C) → cuboid → (B, T, H, W, V_out) → (B, T, V_out, H, W)
            return self.model(
                x.permute(0, 1, 3, 4, 2).contiguous()
            ).permute(0, 1, 4, 2, 3).contiguous()

        @staticmethod
        def _unpack(batch):
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            return x, y

        def _loss(self, y_hat, y):
            # y_hat, y: (B, T, 1, H, W)
            if self.lat_weights is not None:
                loss = F.mse_loss(y_hat, y, reduction='none')
                w = self.lat_weights.view(1, 1, 1, -1, 1)
                return (loss * w).mean()
            return F.mse_loss(y_hat, y)

        def training_step(self, batch, batch_idx):
            x, y = self._unpack(batch)
            loss = self._loss(self(x), y)
            self.log('train_loss', loss, on_step=True, on_epoch=True,
                     prog_bar=True, sync_dist=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = self._unpack(batch)
            loss = self._loss(self(x), y)
            self.log('valid_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True)
            return loss

        def test_step(self, batch, batch_idx):
            x, y = self._unpack(batch)
            loss = self._loss(self(x), y)
            self.log('test_loss', loss, sync_dist=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    _CuboidBaseline.__name__     = name
    _CuboidBaseline.__qualname__ = name
    return _CuboidBaseline
