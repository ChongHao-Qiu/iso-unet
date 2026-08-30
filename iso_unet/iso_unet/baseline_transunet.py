"""
baseline_transunet.py
─────────────────────────────────────────────────────────────────────────────
TransUNet (Chen et al., 2021, https://arxiv.org/abs/2102.04306) single-frame baseline.
Same I/O as baseline_unet2d_vanilla / unetpp / attunet / dcsaunet — plugs directly
into 2d_data_clean's train.py / eval.py.

Architecture = R50-ViT-B/16 hybrid (paper canonical):
    * R50 stem (ResNetV2 pre-activation bottleneck) — 3 stages / width 64 ->
        outputs 1024 ch at /16, skip features at /8 (512), /4 (256), /2 (64)
    * ViT encoder (12 layers, hidden=768, heads=12, mlp=3072) — applied on
        the 6x12 patch grid (for our 96x192 padded input)
    * UNet-style DecoderCup — 4 upsample steps with skip from R50, channels
        [256, 128, 64, 16], followed by a SegHead 1x1x16 -> out_channels

Code ported from baselines/TransUNet/networks/{vit_seg_modeling,vit_seg_modeling_resnet_skip,
vit_seg_configs}.py, with the following changes:
    1. **Fix non-square bug**:
       - DecoderCup originally hard-coded `h=w=sqrt(n_patch)` -> changed to (h_grid, w_grid).
       - ResNetV2 originally used `right_size = in_size/4/(i+1)`, assuming square -> removed
         the padding hack and used F.interpolate to align skip sizes in decoder blocks.
    2. **Changed maxpool padding 0 -> 1**: so the root pool produces a clean output on 96x192
       input (24x48 instead of 23x47), avoiding downstream per-layer size mismatches (since we
       train from scratch, this small bias does not affect accuracy).
    3. Removed unused dependencies (ml_collections / scipy / npz pretrained weight loading /
       attention-collecting vis), reduced to a pure-PyTorch single file.
    4. Added 90x180 -> pad-to-16 + crop-back; in_channels is configurable (the original
       hard-codes 3).
    5. Added Lightning wrapper + cos(lat) weighted MSE.

I/O:
    forward(x):  (B, C, H, W) -> (B, 1, H, W)
    batch (x, y) or (x, y, co2) both accepted (co2 ignored)
"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


# ══════════════════════════════════════════════════════════════════════
#  R50 hybrid stem (ResNetV2 pre-activation, weight-standardized conv)
# ══════════════════════════════════════════════════════════════════════

class _StdConv2d(nn.Conv2d):
    """Weight-standardized Conv2d (Big-Transfer style)."""
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def _conv3x3(ci, co, stride=1):
    return _StdConv2d(ci, co, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(ci, co, stride=1):
    return _StdConv2d(ci, co, kernel_size=1, stride=stride, padding=0, bias=False)


class _PreActBottleneck(nn.Module):
    """Pre-activation v2 bottleneck (cin -> cmid -> cmid -> cout, +residual)."""
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        # NOTE: bug-compatible with the original — gn1 uses cmid (the paper says cin, but the original repo uses cmid)
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = _conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = _conv3x3(cmid, cmid, stride)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = _conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = _conv1x1(cin, cout, stride)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.gn_proj(self.downsample(x))
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return self.relu(residual + y)


class _ResNetV2Hybrid(nn.Module):
    """
    R50 stem for TransUNet hybrid. 4 spatial levels:
        root (/2, ch=width) → pool → block1 (/4, ch=4*width) → block2 (/8, ch=8*width)
            → block3 (/16, ch=16*width)
    Returns (final_at_/16, [skip_at_/8, skip_at_/4, skip_at_/2]).
    """
    def __init__(self, in_channels=3, block_units=(3, 4, 9), width_factor=1):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', _StdConv2d(in_channels, width, kernel_size=7,
                                stride=2, bias=False, padding=3)),
            ('gn',   nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        # NOTE: the original uses maxpool padding=0 — at 224 input this gives 55x55 (off-by-1),
        # then a right_size hack forcibly pads back to 56. Here padding=1 keeps the size
        # naturally clean and avoids the right_size square-assumption bug.
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def _block(cin, cout, cmid, n_units, first_stride=1):
            units = [('unit1', _PreActBottleneck(cin=cin, cout=cout, cmid=cmid,
                                                 stride=first_stride))]
            for i in range(2, n_units + 1):
                units.append((f'unit{i:d}', _PreActBottleneck(
                    cin=cout, cout=cout, cmid=cmid)))
            return nn.Sequential(OrderedDict(units))

        self.body = nn.Sequential(OrderedDict([
            ('block1', _block(width,     width * 4,  width,     block_units[0])),
            ('block2', _block(width * 4, width * 8,  width * 2, block_units[1],
                              first_stride=2)),
            ('block3', _block(width * 8, width * 16, width * 4, block_units[2],
                              first_stride=2)),
        ]))

    def forward(self, x):
        features = []
        x = self.root(x);            features.append(x)    # /2,  ch=width
        x = self.pool(x)
        x = self.body.block1(x);     features.append(x)    # /4,  ch=4*width
        x = self.body.block2(x);     features.append(x)    # /8,  ch=8*width
        x = self.body.block3(x)                            # /16, ch=16*width
        return x, features[::-1]     # [/8, /4, /2]


# ══════════════════════════════════════════════════════════════════════
#  ViT transformer encoder
# ══════════════════════════════════════════════════════════════════════

class _Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_dropout=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.all_head  = num_heads * self.head_size

        self.query = nn.Linear(hidden_size, self.all_head)
        self.key   = nn.Linear(hidden_size, self.all_head)
        self.value = nn.Linear(hidden_size, self.all_head)
        self.out   = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attn_dropout)

    def _split(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, x):
        q = self._split(self.query(x))
        k = self._split(self.key(x))
        v = self._split(self.value(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
        probs  = self.dropout(F.softmax(scores, dim=-1))
        ctx    = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        ctx    = ctx.view(*ctx.size()[:-2], self.all_head)
        return self.dropout(self.out(ctx))


class _Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        return self.dropout(self.fc2(x))


class _Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim,
                 dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm  = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = _Attention(hidden_size, num_heads, attn_dropout)
        self.ffn  = _Mlp(hidden_size, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class _ViTEncoder(nn.Module):
    """Positional embed + N transformer blocks + final LN."""
    def __init__(self, hidden_size, num_heads, num_layers, mlp_dim,
                 n_patches, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            _Block(hidden_size, num_heads, mlp_dim, dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        x = self.dropout(x + self.pos_embed)
        for layer in self.layers:
            x = layer(x)
        return self.encoder_norm(x)


# ══════════════════════════════════════════════════════════════════════
#  Decoder cup (UNet-style with skip from R50)
# ══════════════════════════════════════════════════════════════════════

class _Conv2dReLU(nn.Sequential):
    def __init__(self, ci, co, k=3, p=1):
        super().__init__(
            nn.Conv2d(ci, co, k, padding=p, bias=False),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
        )


class _DecoderBlock(nn.Module):
    """
    Bilinear up + optional skip concat + 2× Conv-BN-ReLU.
    Use F.interpolate(size=skip.shape) for robust non-square / odd-dim alignment.
    """
    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.conv1 = _Conv2dReLU(in_ch + skip_ch, out_ch)
        self.conv2 = _Conv2dReLU(out_ch, out_ch)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                              align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear',
                              align_corners=False)
        return self.conv2(self.conv1(x))


# ══════════════════════════════════════════════════════════════════════
#  TransUNet main module
# ══════════════════════════════════════════════════════════════════════

class TransUNet(nn.Module):
    """
    R50-ViT TransUNet.

    Args:
        in_channels:        number of input channels
        out_channels:       number of output channels (regression = 1)
        H, W:               original spatial size (internally padded to a multiple of 16)
        hidden_size:        ViT hidden dim (default 768 = ViT-B)
        num_heads:          ViT attention heads (default 12)
        num_layers:         ViT transformer blocks (default 12 = ViT-B; ViT-L uses 24)
        mlp_dim:            ViT FFN inner dim (default 3072 = 4*hidden)
        dropout:            transformer dropout (default 0.1)
        attn_dropout:       attention prob dropout (default 0.0)
        resnet_block_units: number of units per R50 stage (default (3, 4, 9) = R50)
        resnet_width_factor: R50 width multiplier (default 1 -> 64 base)
        decoder_channels:   out_channels of the 4 decoder blocks (default (256,128,64,16))
        n_skip:             use the first n R50 skips (default 3 = all; the 4th decoder block has no skip)
    """
    def __init__(
        self,
        in_channels=2, out_channels=1, H=90, W=180,
        hidden_size=768, num_heads=12, num_layers=12, mlp_dim=3072,
        dropout=0.1, attn_dropout=0.0,
        resnet_block_units=(3, 4, 9), resnet_width_factor=1,
        decoder_channels=(256, 128, 64, 16), n_skip=3,
    ):
        super().__init__()
        self.H_in, self.W_in = H, W
        # pad to 16-multiple (R50 + 4 decoder up steps both work on /16 grid)
        self.H_p = ((H - 1) // 16 + 1) * 16
        self.W_p = ((W - 1) // 16 + 1) * 16
        self.pad_top    = (self.H_p - H) // 2
        self.pad_bottom = self.H_p - H - self.pad_top
        self.pad_left   = (self.W_p - W) // 2
        self.pad_right  = self.W_p - W - self.pad_left

        self.h_grid = self.H_p // 16
        self.w_grid = self.W_p // 16
        n_patches = self.h_grid * self.w_grid

        # R50 hybrid stem
        self.hybrid = _ResNetV2Hybrid(
            in_channels=in_channels,
            block_units=resnet_block_units,
            width_factor=resnet_width_factor,
        )
        hybrid_out_ch = self.hybrid.width * 16     # e.g. 1024

        # 1×1 conv "patch embedding" (since hybrid already /16)
        self.patch_embed = nn.Conv2d(hybrid_out_ch, hidden_size,
                                     kernel_size=1, stride=1)

        # ViT encoder
        self.encoder = _ViTEncoder(
            hidden_size=hidden_size, num_heads=num_heads,
            num_layers=num_layers, mlp_dim=mlp_dim,
            n_patches=n_patches,
            dropout=dropout, attn_dropout=attn_dropout,
        )

        # Decoder cup
        head_channels = 512
        self.conv_more = _Conv2dReLU(hidden_size, head_channels)

        # Skip channels (R50 outputs reversed: /8, /4, /2; 4th decoder gets 0)
        skip_channels = [self.hybrid.width * 8,    # /8  → 512
                         self.hybrid.width * 4,    # /4  → 256
                         self.hybrid.width,        # /2  →  64
                         0]
        # zero out skips beyond n_skip (last decoder blocks have no skip)
        n_skip = max(0, min(int(n_skip), 4))
        for i in range(4 - n_skip):
            skip_channels[3 - i] = 0
        self.n_skip = n_skip

        in_chs = [head_channels] + list(decoder_channels[:-1])
        self.decoder_blocks = nn.ModuleList([
            _DecoderBlock(ic, oc, sc)
            for ic, oc, sc in zip(in_chs, decoder_channels, skip_channels)
        ])

        # SegmentationHead (kernel 3, no upsample — decoder already at full res)
        self.seg_head = nn.Conv2d(decoder_channels[-1], out_channels,
                                  kernel_size=3, padding=1)

    def forward(self, x):
        x = F.pad(x,
                  (self.pad_left, self.pad_right,
                   self.pad_top,  self.pad_bottom),
                  mode='constant', value=0)

        # R50 → final /16 + 3 skips
        x, features = self.hybrid(x)                          # x: (B, 16w, h_g, w_g)
        x = self.patch_embed(x)                               # (B, hidden, h_g, w_g)
        B, C, H_g, W_g = x.shape
        tokens = x.flatten(2).transpose(-1, -2).contiguous()  # (B, n_patches, hidden)
        tokens = self.encoder(tokens)                         # (B, n_patches, hidden)
        x = tokens.transpose(-1, -2).contiguous().view(B, C, H_g, W_g)

        # Decoder cup
        x = self.conv_more(x)
        for i, block in enumerate(self.decoder_blocks):
            skip = features[i] if i < self.n_skip else None
            x = block(x, skip=skip)

        x = self.seg_head(x)
        return x[:, :,
                 self.pad_top : self.pad_top + self.H_in,
                 self.pad_left: self.pad_left + self.W_in]


# ══════════════════════════════════════════════════════════════════════
#  Lightning wrapper (same interface as the other baselines)
# ══════════════════════════════════════════════════════════════════════

class TransUNetBaseline(L.LightningModule):
    """
    TransUNet single-frame baseline.

        forward(x): (B, C, H, W) → (B, 1, H, W)
        loss:       cos(lat) weighted MSE
        batch:      (x, y) or (x, y, co2) — co2 ignored
    """
    def __init__(
        self,
        n_inputs:           int   = 2,
        out_channels:       int   = 1,
        H:                  int   = 90,
        W:                  int   = 180,
        hidden_size:        int   = 768,
        num_heads:          int   = 12,
        num_layers:         int   = 12,
        mlp_dim:            int   = 3072,
        dropout:            float = 0.1,
        attn_dropout:       float = 0.0,
        resnet_block_units: tuple = (3, 4, 9),
        resnet_width_factor: int  = 1,
        decoder_channels:   tuple = (256, 128, 64, 16),
        n_skip:             int   = 3,
        weights                   = None,
        lr:                 float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr

        if weights is not None:
            self.register_buffer('lat_weights',
                                 torch.as_tensor(weights, dtype=torch.float32))
        else:
            self.lat_weights = None

        self.net = TransUNet(
            in_channels=n_inputs, out_channels=out_channels,
            H=H, W=W,
            hidden_size=hidden_size, num_heads=num_heads,
            num_layers=num_layers, mlp_dim=mlp_dim,
            dropout=dropout, attn_dropout=attn_dropout,
            resnet_block_units=tuple(resnet_block_units),
            resnet_width_factor=resnet_width_factor,
            decoder_channels=tuple(decoder_channels),
            n_skip=n_skip,
        )

    def forward(self, x):
        return self.net(x)

    def _unpack(self, batch):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        if y.dim() == 5:                # sequence dataset misused -> last frame
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


# ══════════════════════════════════════════════════════════════════════
#  Smoke test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 9-ch climate input, 90×180
    B, C, H, W = 2, 9, 90, 180

    # ── ViT-B/16 default (paper canonical) ──
    print('=' * 60)
    print('R50-ViT-B/16 (paper canonical)')
    print('=' * 60)
    m = TransUNetBaseline(n_inputs=C, out_channels=1, H=H, W=W)
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, 1, H, W)
    y_hat = m(x)
    print(f'Input  : {tuple(x.shape)}')
    print(f'Output : {tuple(y_hat.shape)}   ← (B, 1, H, W) ✓')
    print(f'h_grid × w_grid = {m.net.h_grid} × {m.net.w_grid} '
          f'= {m.net.h_grid * m.net.w_grid} patches')
    print(f'Params : {sum(p.numel() for p in m.parameters()):,}')
    loss = m.training_step((x, y), 0)
    loss.backward()
    print(f'train_loss = {loss.item():.4f}, backward OK')

    # ── light variant (small ViT for prototyping) ──
    print('\n' + '=' * 60)
    print('Light variant (hidden=256, layers=4, heads=8)')
    print('=' * 60)
    m_light = TransUNetBaseline(
        n_inputs=C, out_channels=1, H=H, W=W,
        hidden_size=256, num_heads=8, num_layers=4, mlp_dim=512,
    )
    y_hat = m_light(x)
    print(f'Output : {tuple(y_hat.shape)}')
    print(f'Params : {sum(p.numel() for p in m_light.parameters()):,}')
