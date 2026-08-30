"""
baseline_climax.py
─────────────────────────────────────────────────────────────────────────────
SELF-CONTAINED ClimaX baseline (Microsoft 2023, https://arxiv.org/abs/2301.10343).

This file inlines every component needed from the upstream ClimaX repo so
that nothing outside the standard PyTorch / numpy / lightning stack is
required. No `climax`, `timm`, or `xformers` import is needed.

Source of the inlined code (verbatim, lightly cleaned, MIT/Meta licenses):
  • https://github.com/microsoft/ClimaX/blob/main/src/climax/arch.py
  • https://github.com/microsoft/ClimaX/blob/main/src/climax/climate_projection/arch.py
  • https://github.com/microsoft/ClimaX/blob/main/src/climax/utils/pos_embed.py

For us:
    forward(x):  (B, T, C, H, W) → (B, 1, H, W)
    loss:        cos(lat) weighted MSE
    batch:       (x, y) or (x, y, co2) — co2 ignored

NOTE: paleoclimate data is limited (~K monthly samples). Default
hyperparams below produce a shrunken "ClimaX-XS"; scale up via YAML if
needed.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# ═════════════════════════════════════════════════════════════════════════
# §1. Position embedding utilities  (inlined from climax.utils.pos_embed)
# ═════════════════════════════════════════════════════════════════════════
def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega                            # (D/2,)
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)                   # (M, D/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)            # (H*W, D)


def _get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)                       # w first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# ═════════════════════════════════════════════════════════════════════════
# §2. Lightweight replacements for timm modules
# ═════════════════════════════════════════════════════════════════════════
class PatchEmbed(nn.Module):
    """Per-variable image-to-patch embedding: one input channel per call."""
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C=1, H, W) → (B, embed_dim, H/p, W/p) → (B, L, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                  act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features    = out_features    or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    """Stochastic depth per sample."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class Attention(nn.Module):
    """Standard multi-head self-attention (uses PyTorch SDPA under the hood)."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop_p = attn_drop
        self.proj_drop   = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                              # each (B, H, N, C/H)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    """Standard pre-norm ViT block."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                  drop=0.0, attn_drop=0.0, drop_path=0.0,
                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1     = norm_layer(dim)
        self.attn      = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ═════════════════════════════════════════════════════════════════════════
# §3. ClimaX (variable-tokenized ViT) — base class
# ═════════════════════════════════════════════════════════════════════════
class _ClimaX(nn.Module):
    """ClimaX backbone — inlined verbatim from climax/arch.py with the
    parallel_patch_embed branch removed for simplicity (we always use
    the sequential per-variable PatchEmbed path)."""
    def __init__(
        self,
        default_vars,
        img_size=(32, 64),
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
    ):
        super().__init__()
        self.img_size     = tuple(img_size)
        self.patch_size   = patch_size
        self.default_vars = list(default_vars)

        # Per-variable patch embedding (one PatchEmbed per input variable)
        self.token_embeds = nn.ModuleList([
            PatchEmbed(self.img_size, patch_size, 1, embed_dim)
            for _ in self.default_vars
        ])
        self.num_patches = self.token_embeds[0].num_patches

        # Variable-identity embedding (which variable is this token?)
        self.var_embed, self.var_map = self._create_var_embedding(embed_dim)

        # Variable aggregation: learnable query + cross-attention over V
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg   = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # 2D positional embedding (over the patch grid)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim),
                                       requires_grad=True)
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,
                   qkv_bias=True, drop=drop_rate, drop_path=dpr[i],
                   norm_layer=nn.LayerNorm)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Generic prediction head (subclasses override this)
        head_layers = []
        for _ in range(decoder_depth):
            head_layers += [nn.Linear(embed_dim, embed_dim), nn.GELU()]
        head_layers.append(nn.Linear(embed_dim, len(self.default_vars) * patch_size ** 2))
        self.head = nn.Sequential(*head_layers)

        self._initialize_weights()

    # ── init ──────────────────────────────────────────────────────────────
    def _initialize_weights(self):
        pos_embed = _get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = _get_1d_sincos_pos_embed_from_grid(
            self.var_embed.shape[-1], np.arange(len(self.default_vars))
        )
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        for emb in self.token_embeds:
            w = emb.proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ── variable-id bookkeeping ──────────────────────────────────────────
    def _create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim),
                                  requires_grad=True)
        var_map = {v: i for i, v in enumerate(self.default_vars)}
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, variables, device):
        ids = np.array([self.var_map[v] for v in variables])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, variables):
        ids = self.get_var_ids(variables, var_emb.device)
        return var_emb[:, ids, :]

    # ── helpers ──────────────────────────────────────────────────────────
    def aggregate_variables(self, x):
        # x: (B, V, L, D) → (B, L, D) via cross-attention over V
        b, _, l, _ = x.shape
        x = torch.einsum('bvld->blvd', x)
        x = x.flatten(0, 1)                                   # (B*L, V, D)
        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)
        x = x.squeeze(1)                                       # (B*L, D)
        return x.unflatten(0, (b, l))                          # (B, L, D)

    def forward_encoder(self, x, lead_times, variables):
        # x: (B, V, H, W) — variables is a list/tuple of names matching V
        if isinstance(variables, list):
            variables = tuple(variables)
        var_ids = self.get_var_ids(variables, x.device)
        embeds = []
        for i in range(len(var_ids)):
            embeds.append(self.token_embeds[var_ids[i]](x[:, i:i + 1]))
        x = torch.stack(embeds, dim=1)                         # (B, V, L, D)
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)                          # (B, V, L, D)
        x = self.aggregate_variables(x)                         # (B, L, D)
        x = x + self.pos_embed
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ═════════════════════════════════════════════════════════════════════════
# §4. ClimaX-ClimateBench (adds time aggregation + linear head)
# ═════════════════════════════════════════════════════════════════════════
class ClimaXClimateBench(_ClimaX):
    """Inlined from climax/climate_projection/arch.py."""
    def __init__(
        self,
        default_vars,
        out_vars,
        img_size=(32, 64),
        time_history=1,
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
    ):
        assert out_vars is not None
        super().__init__(
            default_vars=default_vars,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
        )
        self.out_vars     = list(out_vars)
        self.time_history = time_history

        # Time aggregation: learnable query + cross-attention over T
        self.time_pos_embed = nn.Parameter(torch.zeros(1, time_history, embed_dim),
                                            requires_grad=True)
        self.time_agg   = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.time_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        time_pos_embed = _get_1d_sincos_pos_embed_from_grid(
            self.time_pos_embed.shape[-1], np.arange(time_history)
        )
        self.time_pos_embed.data.copy_(torch.from_numpy(time_pos_embed).float().unsqueeze(0))

        # Overwrite head with a single linear projection (B, D) → (B, H*W).
        # Caller reshapes that into (B, 1, H, W).
        self.head = nn.Linear(embed_dim, img_size[0] * img_size[1])

    def forward_encoder(self, x, lead_times, variables):
        # x: (B, T, V, H, W)
        if isinstance(variables, list):
            variables = tuple(variables)
        b, t = x.shape[0], x.shape[1]
        x = x.flatten(0, 1)                                    # (B*T, V, H, W)

        var_ids = self.get_var_ids(variables, x.device)
        embeds = []
        for i in range(len(var_ids)):
            embeds.append(self.token_embeds[var_ids[i]](x[:, i:i + 1]))
        x = torch.stack(embeds, dim=1)                          # (B*T, V, L, D)
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)
        x = self.aggregate_variables(x)                          # (B*T, L, D)
        x = x + self.pos_embed
        x = x.unflatten(0, (b, t))                              # (B, T, L, D)
        x = x + self.time_pos_embed.unsqueeze(2)

        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))
        lead_time_emb = lead_time_emb.unsqueeze(1).unsqueeze(2)
        x = x + lead_time_emb                                   # (B, T, L, D)

        x = x.flatten(0, 1)                                     # (B*T, L, D)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                                        # (B*T, L, D)

        x = x.unflatten(0, (b, t))                              # (B, T, L, D)
        x = x.mean(-2)                                          # (B, T, D) — avg over patches
        time_query = self.time_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.time_agg(time_query, x, x)                  # (B, 1, D)
        return x

    def forward_raw(self, x, lead_times, variables):
        """(B, T, V, H, W) → (B, 1, H, W)"""
        z = self.forward_encoder(x, lead_times, variables)       # (B, 1, D)
        preds = self.head(z)                                     # (B, 1, H*W)
        return preds.reshape(-1, 1, self.img_size[0], self.img_size[1])


# ═════════════════════════════════════════════════════════════════════════
# §5. Lightning wrapper for iso paleoclimate regression
# ═════════════════════════════════════════════════════════════════════════
DEFAULT_VAR_NAMES = ('tas', 'precl', 'precc', 'PS', 'TMQ',
                     'QFLX', 'FLUT', 'LANDFRAC', 'aice')


class ClimaXBaseline(L.LightningModule):
    """ClimaX wrapper.   forward(x): (B, T, C, H, W) → (B, 1, H, W)"""
    def __init__(
        self,
        n_inputs:      int = 9,
        out_channels:  int = 1,
        H:             int = 90,
        W:             int = 180,
        seq_len:       int = 12,
        patch_size:    int = 5,
        embed_dim:     int = 256,
        depth:         int = 4,
        decoder_depth: int = 2,
        num_heads:     int = 8,
        mlp_ratio:     float = 4.0,
        drop_path:     float = 0.1,
        drop_rate:     float = 0.1,
        var_names:     Optional[List[str]] = None,
        weights              = None,
        lr:            float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr
        self.seq_len = seq_len
        self.out_channels = out_channels   # currently hardcoded to 1 in head; kept for YAML API
        if out_channels != 1:
            raise NotImplementedError(
                'ClimaXBaseline currently only supports out_channels=1; the linear '
                'head maps embed_dim → H*W. Extend forward_raw if more outputs needed.'
            )

        if weights is not None:
            self.register_buffer('lat_weights',
                                 torch.as_tensor(weights, dtype=torch.float32))
        else:
            self.lat_weights = None

        if var_names is None:
            var_names = list(DEFAULT_VAR_NAMES)[:n_inputs]
        if len(var_names) != n_inputs:
            raise ValueError(f'var_names length {len(var_names)} != n_inputs={n_inputs}')
        self.var_names = tuple(var_names)
        self.out_var_names = ('d18Op',)

        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f'ClimaX patch_size={patch_size} must divide H={H} and W={W}. '
                f'Try patch_size ∈ {{2, 5, 6, 9, 10}} for (90, 180).')

        self.arch = ClimaXClimateBench(
            default_vars=list(self.var_names),
            out_vars=list(self.out_var_names),
            img_size=(H, W),
            time_history=seq_len,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
        )
        self.h_patch = H // patch_size
        self.w_patch = W // patch_size

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T, C = x.shape[:3]
        if T != self.seq_len:
            raise ValueError(f'expected seq_len={self.seq_len} got T={T}')
        if C != len(self.var_names):
            raise ValueError(f'expected {len(self.var_names)} variables got C={C}')
        lead_times = torch.zeros(B, device=x.device, dtype=x.dtype)
        return self.arch.forward_raw(x, lead_times, self.var_names)

    # ── Lightning hooks ──────────────────────────────────────────────────
    @staticmethod
    def _unpack(batch):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        if y.dim() == 5:
            y = y[:, -1]
        return x, y

    def _loss(self, y_hat, y):
        if self.lat_weights is not None:
            loss = F.mse_loss(y_hat, y, reduction='none')
            w = self.lat_weights.view(1, 1, -1, 1)
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


# ═════════════════════════════════════════════════════════════════════════
# §6. Single-day variant — ClimaXSingleDay (NO time aggregation)
# ═════════════════════════════════════════════════════════════════════════
class ClimaXSingleDay(_ClimaX):
    """Single-frame ClimaX backbone:  (B, V, H, W) → (B, 1, H, W).

    Same head philosophy as ClimaXClimateBench (mean-pool patch tokens →
    linear → H*W) but the time-aggregation cross-attention is dropped
    entirely — there is only one frame. Uses the base `_ClimaX.forward_encoder`,
    which is already single-frame (expects x: (B, V, H, W))."""
    def __init__(
        self,
        default_vars,
        out_vars,
        img_size=(32, 64),
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
    ):
        assert out_vars is not None
        super().__init__(
            default_vars=default_vars,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
        )
        self.out_vars = list(out_vars)
        # Overwrite patch-reconstruction head with a single linear projection
        # (B, D) → (B, H*W); caller reshapes to (B, 1, H, W).
        self.head = nn.Linear(embed_dim, img_size[0] * img_size[1])

    def forward_raw(self, x, lead_times, variables):
        """(B, V, H, W) → (B, 1, H, W)"""
        z = self.forward_encoder(x, lead_times, variables)   # (B, L, D)
        z = z.mean(dim=1)                                    # (B, D) — avg over patches
        preds = self.head(z)                                 # (B, H*W)
        return preds.reshape(-1, 1, self.img_size[0], self.img_size[1])


class ClimaXSingleDayBaseline(L.LightningModule):
    """Single-day ClimaX wrapper.  forward(x): (B, C, H, W) → (B, 1, H, W).

    Also accepts (B, 1, C, H, W) (e.g. a sequence dataset run with seq_len=1)
    and squeezes the singleton time axis. No `seq_len` arg — single frame only.
    """
    def __init__(
        self,
        n_inputs:      int = 9,
        out_channels:  int = 1,
        H:             int = 90,
        W:             int = 180,
        patch_size:    int = 5,
        embed_dim:     int = 256,
        depth:         int = 4,
        decoder_depth: int = 2,
        num_heads:     int = 8,
        mlp_ratio:     float = 4.0,
        drop_path:     float = 0.1,
        drop_rate:     float = 0.1,
        var_names:     Optional[List[str]] = None,
        weights              = None,
        lr:            float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr
        self.out_channels = out_channels   # head hardcoded to 1; kept for YAML API
        if out_channels != 1:
            raise NotImplementedError(
                'ClimaXSingleDayBaseline currently only supports out_channels=1; the '
                'linear head maps embed_dim → H*W. Extend forward_raw if more outputs needed.'
            )

        if weights is not None:
            self.register_buffer('lat_weights',
                                 torch.as_tensor(weights, dtype=torch.float32))
        else:
            self.lat_weights = None

        if var_names is None:
            var_names = list(DEFAULT_VAR_NAMES)[:n_inputs]
        if len(var_names) != n_inputs:
            raise ValueError(f'var_names length {len(var_names)} != n_inputs={n_inputs}')
        self.var_names = tuple(var_names)
        self.out_var_names = ('d18Op',)

        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f'ClimaX patch_size={patch_size} must divide H={H} and W={W}. '
                f'Try patch_size ∈ {{2, 5, 6, 9, 10}} for (90, 180).')

        self.arch = ClimaXSingleDay(
            default_vars=list(self.var_names),
            out_vars=list(self.out_var_names),
            img_size=(H, W),
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        if x.dim() == 5:                       # (B, T, C, H, W) — must have T=1
            if x.shape[1] != 1:
                raise ValueError(
                    f'ClimaXSingleDayBaseline is single-frame; got T={x.shape[1]} '
                    f'(expected 1). Use ClimaXBaseline for multi-frame input.')
            x = x[:, 0]                        # → (B, C, H, W)
        B, C = x.shape[:2]
        if C != len(self.var_names):
            raise ValueError(f'expected {len(self.var_names)} variables got C={C}')
        lead_times = torch.zeros(B, device=x.device, dtype=x.dtype)
        return self.arch.forward_raw(x, lead_times, self.var_names)

    # ── Lightning hooks ──────────────────────────────────────────────────
    @staticmethod
    def _unpack(batch):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        if y.dim() == 5:                       # (B, T, 1, H, W) → last frame
            y = y[:, -1]
        return x, y

    def _loss(self, y_hat, y):
        if self.lat_weights is not None:
            loss = F.mse_loss(y_hat, y, reduction='none')
            w = self.lat_weights.view(1, 1, -1, 1)
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


# ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    B, T, C, H, W = 2, 12, 9, 90, 180
    m = ClimaXBaseline(n_inputs=C, H=H, W=W, seq_len=T,
                        patch_size=5, embed_dim=128, depth=2, num_heads=4)
    x = torch.randn(B, T, C, H, W)
    y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f'ClimaXBaseline forward OK')
    print(f'  in : {tuple(x.shape)}')
    print(f'  out: {tuple(y.shape)}')
    print(f'  params: {n_params/1e6:.2f} M')

    # Single-day variant
    ms = ClimaXSingleDayBaseline(n_inputs=C, H=H, W=W,
                                 patch_size=5, embed_dim=128, depth=2, num_heads=4)
    xs = torch.randn(B, C, H, W)                 # single frame
    ys = ms(xs)
    ns = sum(p.numel() for p in ms.parameters())
    print(f'ClimaXSingleDayBaseline forward OK')
    print(f'  in : {tuple(xs.shape)}')
    print(f'  out: {tuple(ys.shape)}    ← (B, 1, H, W)')
    print(f'  also accepts (B,1,C,H,W): {tuple(ms(xs.unsqueeze(1)).shape)}')
    print(f'  params: {ns/1e6:.2f} M')
