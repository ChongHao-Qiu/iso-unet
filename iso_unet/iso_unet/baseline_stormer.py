"""
baseline_stormer.py
─────────────────────────────────────────────────────────────────────────────
SELF-CONTAINED Stormer baseline (Nguyen et al. 2023,
https://arxiv.org/abs/2312.03876).

This file inlines every Stormer component so that nothing outside the
standard PyTorch / numpy / lightning stack is required. No `stormer`,
`timm`, or `xformers` import is needed — the memory-efficient attention is
replaced by PyTorch's `scaled_dot_product_attention` (equivalent).

Source of the inlined code:
  • https://github.com/tung-nd/stormer/blob/main/stormer/models/hub/stormer.py
  • https://github.com/tung-nd/stormer/blob/main/stormer/models/hub/weather_embedding.py
  • https://github.com/tung-nd/stormer/blob/main/stormer/utils/pos_embed.py

Adaptations for iso paleoclimate δ¹⁸O_p regression:
    - Aggregate (B, T, V, H, W) → (B, V, H, W) by taking the LAST frame of
      the input window (Stormer is single-step).
    - Replace the prediction head to output 1 channel (d18Op) instead of
      len(variables) channels.
    - `time_interval = 0` (no forecast — we predict the same time).

forward(x):  (B, T, V, H, W) → (B, 1, H, W)
loss:        cos(lat) weighted MSE
batch:       (x, y) or (x, y, co2) — co2 ignored
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
# §1. Position embedding utilities (inlined from stormer.utils.pos_embed)
# ═════════════════════════════════════════════════════════════════════════
def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# ═════════════════════════════════════════════════════════════════════════
# §2. Lightweight replacements for timm modules
# ═════════════════════════════════════════════════════════════════════════
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                               kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)          # (B, L, D)


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


# ═════════════════════════════════════════════════════════════════════════
# §3. Stormer building blocks (adaLN-modulated DiT-style)
# ═════════════════════════════════════════════════════════════════════════
def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar time-interval into a vector."""
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Linear(1, hidden_size)

    def forward(self, t):
        return self.mlp(t.unsqueeze(-1))


class MemEffAttention(nn.Module):
    """Stormer's memory-efficient attention, with xformers' kernel replaced
    by PyTorch's `scaled_dot_product_attention` (functionally identical,
    same memory-efficient backend on CUDA)."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True,
                  attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.attn_drop_p = attn_drop
        self.proj_drop   = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        H = self.num_heads
        # qkv: (B, N, 3, H, C/H) → split → each (B, H, N, C/H)
        qkv = self.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )                                                        # (B, H, N, C/H)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    """Transformer block with adaLN-Zero conditioning (DiT-style)."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = MemEffAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate='tanh')        # noqa: E731
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden,
                        act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp (_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Predicts (B, L, V*p*p) tokens, ready for unpatchify."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.Identity()
        self.linear     = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ═════════════════════════════════════════════════════════════════════════
# §4. WeatherEmbedding (variable-tokenized patch embed + cross-attention agg)
# ═════════════════════════════════════════════════════════════════════════
class WeatherEmbedding(nn.Module):
    def __init__(self, variables, img_size, patch_size=2, embed_dim=1024, num_heads=16):
        super().__init__()
        self.img_size   = tuple(img_size)
        self.patch_size = patch_size
        self.variables  = list(variables)

        self.token_embeds = nn.ModuleList([
            PatchEmbed(self.img_size, patch_size, 1, embed_dim)
            for _ in self.variables
        ])
        self.num_patches = (self.img_size[0] // patch_size) * (self.img_size[1] // patch_size)

        self.channel_embed, self.channel_map = self._create_var_embedding(embed_dim)

        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.channel_agg   = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim),
                                       requires_grad=True)
        self._initialize_weights()

    def _initialize_weights(self):
        pos_embed = _get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        channel_embed = _get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1], np.arange(len(self.variables))
        )
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

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

    def _create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.variables), dim), requires_grad=True)
        var_map = {v: i for i, v in enumerate(self.variables)}
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, variables, device):
        ids = np.array([self.channel_map[v] for v in variables])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, variables):
        ids = self.get_var_ids(variables, var_emb.device)
        return var_emb[:, ids, :]

    def aggregate_variables(self, x):
        # x: (B, V, L, D) → (B, L, D)
        b, _, l, _ = x.shape
        x = torch.einsum('bvld->blvd', x)
        x = x.flatten(0, 1)                                      # (B*L, V, D)
        q = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.channel_agg(q, x, x)
        x = x.squeeze(1)
        return x.unflatten(0, (b, l))

    def forward(self, x, variables):
        if isinstance(variables, list):
            variables = tuple(variables)
        var_ids = self.get_var_ids(variables, x.device)
        embeds = []
        for i in range(len(var_ids)):
            embeds.append(self.token_embeds[var_ids[i]](x[:, i:i + 1]))
        x = torch.stack(embeds, dim=1)                            # (B, V, L, D)
        var_embed = self.get_var_emb(self.channel_embed, variables)
        x = x + var_embed.unsqueeze(2)
        x = x + self.pos_embed.unsqueeze(1)
        return self.aggregate_variables(x)                        # (B, L, D)


# ═════════════════════════════════════════════════════════════════════════
# §5. Stormer base class
# ═════════════════════════════════════════════════════════════════════════
class _Stormer(nn.Module):
    def __init__(
        self,
        in_img_size,
        variables,
        patch_size=2,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        if in_img_size[0] % patch_size != 0:
            pad = patch_size - in_img_size[0] % patch_size
            in_img_size = (in_img_size[0] + pad, in_img_size[1])
        self.in_img_size = tuple(in_img_size)
        self.variables   = list(variables)
        self.patch_size  = patch_size

        self.embedding         = WeatherEmbedding(variables=self.variables,
                                                    img_size=self.in_img_size,
                                                    patch_size=patch_size,
                                                    embed_dim=hidden_size,
                                                    num_heads=num_heads)
        self.embed_norm_layer  = nn.LayerNorm(hidden_size)
        self.t_embedder        = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.head = FinalLayer(hidden_size, patch_size, len(self.variables))
        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)
        trunc_normal_(self.t_embedder.mlp.weight, std=0.02)

        # adaLN-Zero: zero the last linear of every adaLN_modulation in blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)


# ═════════════════════════════════════════════════════════════════════════
# §6. Lightning wrapper for iso paleoclimate regression
# ═════════════════════════════════════════════════════════════════════════
DEFAULT_VAR_NAMES = ('tas', 'precl', 'precc', 'PS', 'TMQ',
                     'QFLX', 'FLUT', 'LANDFRAC', 'aice')


class StormerBaseline(L.LightningModule):
    """Stormer wrapper. forward(x): (B, T, C, H, W) → (B, 1, H, W).
    Only the LAST timestep of the input sequence is fed into the Stormer
    backbone (Stormer is single-step). The model's prediction head is
    replaced to emit 1 channel for d18Op."""
    def __init__(
        self,
        n_inputs:      int = 9,
        out_channels:  int = 1,
        H:             int = 90,
        W:             int = 180,
        seq_len:       int = 12,
        patch_size:    int = 5,
        hidden_size:   int = 256,
        depth:         int = 4,
        num_heads:     int = 8,
        mlp_ratio:     float = 4.0,
        var_names:     Optional[List[str]] = None,
        time_aggregation: str = 'last',
        weights              = None,
        lr:            float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr
        self.seq_len = seq_len
        self.out_channels = out_channels
        self.time_aggregation = time_aggregation
        self.patch_size = patch_size
        self.h_patch = H // patch_size
        self.w_patch = W // patch_size

        if weights is not None:
            self.register_buffer('lat_weights',
                                 torch.as_tensor(weights, dtype=torch.float32))
        else:
            self.lat_weights = None

        if var_names is None:
            var_names = list(DEFAULT_VAR_NAMES)[:n_inputs]
        if len(var_names) != n_inputs:
            raise ValueError(f'var_names length {len(var_names)} != n_inputs={n_inputs}')
        self.var_names = list(var_names)

        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f'Stormer patch_size={patch_size} must divide H={H} and W={W}.')

        self.arch = _Stormer(
            in_img_size=(H, W),
            variables=self.var_names,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # Replace the V-channel head with our 1-channel head
        self.arch.head = FinalLayer(hidden_size, patch_size, out_channels)
        nn.init.constant_(self.arch.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.arch.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.arch.head.linear.weight, 0)
        nn.init.constant_(self.arch.head.linear.bias, 0)

    # ── forward ──────────────────────────────────────────────────────────
    def _unpatchify_1ch(self, x):
        """(B, L, V_out * p * p) → (B, V_out, H, W) — bypasses Stormer's
        unpatchify which hardcodes V = len(variables)."""
        p = self.patch_size
        h, w, v = self.h_patch, self.w_patch, self.out_channels
        assert h * w == x.shape[1], (h, w, x.shape)
        x = x.reshape(x.shape[0], h, w, p, p, v)
        x = torch.einsum('nhwpqv->nvhpwq', x)
        return x.reshape(x.shape[0], v, h * p, w * p)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T, V = x.shape[:3]
        if V != len(self.var_names):
            raise ValueError(f'expected V={len(self.var_names)} got {V}')

        if self.time_aggregation == 'last':
            xt = x[:, -1]
        elif self.time_aggregation == 'mean':
            xt = x.mean(dim=1)
        else:
            raise ValueError(f'Unknown time_aggregation={self.time_aggregation!r}')

        h = self.arch.embedding(xt, self.var_names)
        h = self.arch.embed_norm_layer(h)
        time_interval = torch.zeros(B, device=xt.device, dtype=xt.dtype)
        c = self.arch.t_embedder(time_interval)
        for blk in self.arch.blocks:
            h = blk(h, c)
        h = self.arch.head(h, c)                                  # (B, L, V_out*p*p)
        return self._unpatchify_1ch(h)                            # (B, 1, H, W)

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


# ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    B, T, C, H, W = 2, 12, 9, 90, 180
    m = StormerBaseline(n_inputs=C, H=H, W=W, seq_len=T,
                         patch_size=5, hidden_size=128, depth=2, num_heads=4)
    x = torch.randn(B, T, C, H, W)
    y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f'StormerBaseline forward OK')
    print(f'  in : {tuple(x.shape)}')
    print(f'  out: {tuple(y.shape)}')
    print(f'  params: {n_params/1e6:.2f} M')
