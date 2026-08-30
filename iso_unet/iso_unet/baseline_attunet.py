"""
baseline_attunet.py
─────────────────────────────────────────────────────────────────────────────
Attention U-Net (Oktay et al., 2018) single-frame baseline. Same I/O as
baseline_unet2d_vanilla / baseline_unetpp, plugs directly into 2d_data_clean's
train.py / eval.py.

Core: 5-level UNet (4 pools, filters = [c, 2c, 4c, 8c, 16c]) plus an additive
attention gate on each skip connection:
    g  = decoder feature (gating signal, from up-conv at the deeper level)
    x  = encoder skip
    a  = Sigmoid( Conv1x1( ReLU( Conv1x1(g) + Conv1x1(x) ) ) )
    x' = x * a                        <- then concat into decoder
The attention makes the decoder focus only on relevant skip regions (the paper
uses it for pancreas segmentation).

Implementation is stripped from AttU_Net in baselines/Image_Segmentation/network.py.
While porting:
    * Refactored into iso_unet-style _ConvBlock / _UpConv (BN+ReLU, bias=False)
    * Added H/W pad-to-16 (90x180 -> 96x192) + crop-back
    * Added Lightning wrapper + cos(lat) weighted MSE
    * Added forward_with_attn() exposing 4 attention maps (compatible with
      models.supports_attention_viz)

I/O:
    forward(x):  (B, C, H, W) -> (B, 1, H, W)
    batch (x, y) or (x, y, co2) both accepted (co2 ignored)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


# ── Basic blocks ────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """2× (Conv3×3 + BN + ReLU)."""
    def __init__(self, ci, co):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ci, co, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class _UpConv(nn.Module):
    """Bilinear upsample + Conv3×3 + BN + ReLU (channel reduction)."""
    def __init__(self, ci, co):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(ci, co, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(self.up(x))


class _AttentionGate(nn.Module):
    """
    Additive attention gate (Oktay et al. 2018, eq. 1-2).
        g:  gating signal, shape (B, F_g, H, W)
        x:  skip feature,  shape (B, F_l, H, W)  <- same spatial size (up-sample before gating)
    Returns:
        x' = x * α,        α ∈ [0,1] shape (B, 1, H, W) (spatial mask, channels broadcast)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, return_attn=False):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        a  = self.psi(self.relu(g1 + x1))     # (B, 1, H, W)
        out = x * a
        if return_attn:
            return out, a
        return out


# ── Main network ────────────────────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net, 5-level (4 pool) backbone.

    Args:
        in_channels:   number of input channels
        out_channels:  number of output channels (regression = 1)
        base_channels: level-1 width (filters = [c, 2c, 4c, 8c, 16c])
        H, W:          original spatial size; internally padded to be divisible by 16
        max_channels:  channel cap (to prevent the bottleneck from blowing up)
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=64,
                 H=90, W=180, max_channels=1024):
        super().__init__()
        self.H_in, self.W_in = H, W

        # 4 pools -> pad to a multiple of 16
        self.H_p = ((H - 1) // 16 + 1) * 16     # 90 → 96
        self.W_p = ((W - 1) // 16 + 1) * 16     # 180 → 192
        self.pad_top    = (self.H_p - H) // 2
        self.pad_bottom = self.H_p - H - self.pad_top
        self.pad_left   = (self.W_p - W) // 2
        self.pad_right  = self.W_p - W - self.pad_left

        c = base_channels
        filters = [min(c * (2 ** i), max_channels) for i in range(5)]
        # default base=64 → [64, 128, 256, 512, 1024]
        self.filters = filters

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.Conv1 = _ConvBlock(in_channels, filters[0])
        self.Conv2 = _ConvBlock(filters[0],  filters[1])
        self.Conv3 = _ConvBlock(filters[1],  filters[2])
        self.Conv4 = _ConvBlock(filters[2],  filters[3])
        self.Conv5 = _ConvBlock(filters[3],  filters[4])    # bottleneck

        # Decoder: up-conv → attention(g=up, x=skip) → concat → conv-block
        # F_int = F_l // 2 follows the paper convention
        self.Up5      = _UpConv(filters[4], filters[3])
        self.Att5     = _AttentionGate(F_g=filters[3], F_l=filters[3],
                                       F_int=filters[3] // 2)
        self.Up_conv5 = _ConvBlock(filters[3] * 2, filters[3])

        self.Up4      = _UpConv(filters[3], filters[2])
        self.Att4     = _AttentionGate(F_g=filters[2], F_l=filters[2],
                                       F_int=filters[2] // 2)
        self.Up_conv4 = _ConvBlock(filters[2] * 2, filters[2])

        self.Up3      = _UpConv(filters[2], filters[1])
        self.Att3     = _AttentionGate(F_g=filters[1], F_l=filters[1],
                                       F_int=filters[1] // 2)
        self.Up_conv3 = _ConvBlock(filters[1] * 2, filters[1])

        self.Up2      = _UpConv(filters[1], filters[0])
        self.Att2     = _AttentionGate(F_g=filters[0], F_l=filters[0],
                                       F_int=max(filters[0] // 2, 1))
        self.Up_conv2 = _ConvBlock(filters[0] * 2, filters[0])

        self.head = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def _pad(self, x):
        return F.pad(x,
                     (self.pad_left, self.pad_right,
                      self.pad_top,  self.pad_bottom),
                     mode='constant', value=0)

    def _crop(self, x):
        return x[:, :,
                 self.pad_top : self.pad_top + self.H_in,
                 self.pad_left: self.pad_left + self.W_in]

    def _encode(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.pool(x1))
        x3 = self.Conv3(self.pool(x2))
        x4 = self.Conv4(self.pool(x3))
        x5 = self.Conv5(self.pool(x4))
        return x1, x2, x3, x4, x5

    def forward(self, x):
        x = self._pad(x)
        x1, x2, x3, x4, x5 = self._encode(x)

        d5 = self.Up5(x5);  s4 = self.Att5(g=d5, x=x4)
        d5 = self.Up_conv5(torch.cat([s4, d5], dim=1))

        d4 = self.Up4(d5);  s3 = self.Att4(g=d4, x=x3)
        d4 = self.Up_conv4(torch.cat([s3, d4], dim=1))

        d3 = self.Up3(d4);  s2 = self.Att3(g=d3, x=x2)
        d3 = self.Up_conv3(torch.cat([s2, d3], dim=1))

        d2 = self.Up2(d3);  s1 = self.Att2(g=d2, x=x1)
        d2 = self.Up_conv2(torch.cat([s1, d2], dim=1))

        return self._crop(self.head(d2))

    @torch.no_grad()
    def forward_with_attn(self, x):
        """
        Return (y_hat, attn_dict) — attn_dict is used by eval --save_attn.
            attn_dict['att{2..5}']: spatial mask of shape (B, 1, h, w) per level, larger index = deeper level
        """
        x = self._pad(x)
        x1, x2, x3, x4, x5 = self._encode(x)

        d5 = self.Up5(x5);  s4, a5 = self.Att5(g=d5, x=x4, return_attn=True)
        d5 = self.Up_conv5(torch.cat([s4, d5], dim=1))

        d4 = self.Up4(d5);  s3, a4 = self.Att4(g=d4, x=x3, return_attn=True)
        d4 = self.Up_conv4(torch.cat([s3, d4], dim=1))

        d3 = self.Up3(d4);  s2, a3 = self.Att3(g=d3, x=x2, return_attn=True)
        d3 = self.Up_conv3(torch.cat([s2, d3], dim=1))

        d2 = self.Up2(d3);  s1, a2 = self.Att2(g=d2, x=x1, return_attn=True)
        d2 = self.Up_conv2(torch.cat([s1, d2], dim=1))

        y_hat = self._crop(self.head(d2))
        return y_hat, {'att5': a5, 'att4': a4, 'att3': a3, 'att2': a2}


# ── Lightning wrapper (same interface as UNet2DVanillaBaseline / UNetPlusPlusBaseline) ──

class AttentionUNetBaseline(L.LightningModule):
    """
    Attention U-Net single-frame baseline.

        forward(x): (B, C, H, W) → (B, 1, H, W)
        loss:       cos(lat) weighted MSE
        batch:      (x, y) or (x, y, co2) — co2 ignored
    """
    def __init__(
        self,
        n_inputs:      int   = 2,
        out_channels:  int   = 1,
        base_channels: int   = 64,
        H:             int   = 90,
        W:             int   = 180,
        max_channels:  int   = 1024,
        weights              = None,
        lr:            float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr

        if weights is not None:
            self.register_buffer('lat_weights',
                                 torch.as_tensor(weights, dtype=torch.float32))
        else:
            self.lat_weights = None

        self.net = AttentionUNet(
            in_channels=n_inputs, out_channels=out_channels,
            base_channels=base_channels, H=H, W=W,
            max_channels=max_channels,
        )

    def forward(self, x):
        return self.net(x)

    def forward_with_attn(self, x):
        return self.net.forward_with_attn(x)

    def _unpack(self, batch):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        if y.dim() == 5:                # sequence dataset misused -> take last frame
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


# ── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, C, H, W = 2, 9, 90, 180
    m = AttentionUNetBaseline(n_inputs=C, base_channels=64, H=H, W=W)
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, 1, H, W)
    y_hat = m(x)
    print(f'Input  : {tuple(x.shape)}')
    print(f'Output : {tuple(y_hat.shape)}   ← (B, 1, H, W) ✓')
    print(f'Filters: {m.net.filters}')
    print(f'Params : {sum(p.numel() for p in m.parameters()):,}')
    loss = m._loss(y_hat, y)
    print(f'MSE    : {loss.item():.4f}')

    loss = m.training_step((x, y), 0)
    loss.backward()
    print(f'train_loss = {loss.item():.4f}, backward OK')

    # attention maps
    y_hat2, attn = m.forward_with_attn(x)
    print(f'\nforward_with_attn:')
    for k, v in attn.items():
        print(f'  {k}: {tuple(v.shape)}   (range [{v.min():.3f}, {v.max():.3f}])')

    # smaller variant (base=32)
    m32 = AttentionUNetBaseline(n_inputs=C, base_channels=32, H=H, W=W)
    print(f'\nbase=32: params={sum(p.numel() for p in m32.parameters()):,}')
