"""
baseline_unet2d_vanilla.py
─────────────────────────────────────────────────────────────────────────────
4-level UNet 2D, sharing the **same UNet backbone** as
baseline_iso_unet.RegionalAttentionUNet (4-level encoder + bottleneck +
4-level decoder, default base_channels=64, filters=[64,128,256,512,1024]), but
with all attention / MoE / FiLM / climate state **stripped out** — a pure
vanilla UNet baseline.

Purpose:
    * Provide an architecture-matched baseline to measure how much the
      attention / other increments in iso_unet contribute (rather than
      comparing against the 3-level base=16 baseline_unet2d.py — that model is
      about 50x smaller, which is unfair).
    * Match the behavior of iso_unet's "everything off = vanilla" ablation
      flag, but as a standalone, minimal, citable model class.

I/O:
    forward(x):  (B, C, H, W) -> (B, 1, H, W)
    batch (x, y) or (x, y, co2) both accepted (co2 ignored)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


# ── Basic blocks (same as baseline_iso_unet) ───────────────────────────

class _ConvBlock(nn.Module):
    """2x (Conv3x3 + BN + ReLU)."""
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


class _UpConcat(nn.Module):
    """Decoder bilinear upsample + concat with skip + conv block."""
    def __init__(self, in_feat, skip_feat, out_feat):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = _ConvBlock(in_feat + skip_feat, out_feat)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


# ── Main network ────────────────────────────────────────────────────────────

class UNet2DVanilla(nn.Module):
    """
    4-level UNet 2D backbone, no fancy modules.

    Args:
        in_channels:   number of input channels (matches len(INPUT_FEATURES))
        out_channels:  number of output channels (regression task = 1)
        base_channels: UNet level-1 width (default 64, same as iso_unet)
                       filters = [base, 2c, 4c, 8c, 16c]
        H, W:          original spatial size; internally padded to be divisible by 16 (4 pools)
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=64,
                 H=90, W=180):
        super().__init__()
        self.H_in, self.W_in = H, W
        # 4 pools -> 1/16 downsampling; pad to a multiple of 16
        self.H_p = ((H - 1) // 16 + 1) * 16     # 90 -> 96
        self.W_p = ((W - 1) // 16 + 1) * 16     # 180 -> 192
        self.pad_top    = (self.H_p - H) // 2
        self.pad_bottom = self.H_p - H - self.pad_top
        self.pad_left   = (self.W_p - W) // 2
        self.pad_right  = self.W_p - W - self.pad_left

        c = base_channels
        filters = [c, c * 2, c * 4, c * 8, c * 16]

        # Encoder: 4 levels + bottleneck
        self.enc1 = _ConvBlock(in_channels, filters[0])
        self.enc2 = _ConvBlock(filters[0],  filters[1])
        self.enc3 = _ConvBlock(filters[1],  filters[2])
        self.enc4 = _ConvBlock(filters[2],  filters[3])
        self.bottleneck = _ConvBlock(filters[3], filters[4])

        self.pool = nn.MaxPool2d(2)

        # Decoder: 4 levels (bilinear upsample, no transposed conv)
        self.up4 = _UpConcat(filters[4], filters[3], filters[3])
        self.up3 = _UpConcat(filters[3], filters[2], filters[2])
        self.up2 = _UpConcat(filters[2], filters[1], filters[1])
        self.up1 = _UpConcat(filters[1], filters[0], filters[0])

        # Output 1×1 conv (no Softmax — regression)
        self.head = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Pad → (H_p, W_p)
        x = F.pad(x, (self.pad_left, self.pad_right,
                      self.pad_top,  self.pad_bottom),
                  mode='constant', value=0)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # Output + crop back
        out = self.head(d1)
        return out[:, :,
                   self.pad_top:self.pad_top + self.H_in,
                   self.pad_left:self.pad_left + self.W_in]


# ── Lightning wrapper (same interface as UNet2DBaseline) ────────────────────

class UNet2DVanillaBaseline(L.LightningModule):
    """
    UNet2D Vanilla — single-frame baseline, no attention/MoE/FiLM/state.
    Same backbone depth + width as baseline_iso_unet, but pure UNet processing.

        forward(x):  (B, C, H, W) -> (B, 1, H, W)
        loss: cos(lat) weighted MSE
        batch: (x, y) or (x, y, co2) — co2 is ignored
    """
    def __init__(
        self,
        n_inputs:      int   = 2,
        out_channels:  int   = 1,
        base_channels: int   = 64,
        H:             int   = 90,
        W:             int   = 180,
        # ── input_mode (aligned with baseline_iso_unet's same-named ablation) ──
        #     'last_only'  : take the last frame (B, T, C, H, W) -> (B, C, H, W) -> UNet
        #     'concat_all' : concat T frames along channels (B, T, C, H, W) -> (B, T*C, H, W) -> UNet
        input_mode:    str   = 'last_only',
        T:             int   = 12,
        weights              = None,
        lr:            float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr
        assert input_mode in ('last_only', 'concat_all'), (
            f"input_mode must be 'last_only' or 'concat_all', got '{input_mode}'")
        self.n_inputs    = n_inputs
        self._input_mode = input_mode
        self.T_expected  = int(T)
        # In concat_all mode, the UNet sees T*n_inputs channels (e.g. 12 frames x 9 ch = 108)
        in_ch = T * n_inputs if input_mode == 'concat_all' else n_inputs

        if weights is not None:
            self.register_buffer('lat_weights',
                                 torch.as_tensor(weights, dtype=torch.float32))
        else:
            self.lat_weights = None

        self.net = UNet2DVanilla(
            in_channels=in_ch, out_channels=out_channels,
            base_channels=base_channels, H=H, W=W,
        )

    def forward(self, x):
        # Handle 5D (sequence) input — collapse to 4D before feeding UNet
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            if self._input_mode == 'concat_all':
                assert T == self.T_expected, (
                    f'concat_all expected T={self.T_expected}, got T={T}')
                # (B, T, C, H, W) → (B, C, T, H, W) → (B, T*C, H, W)
                x = x.permute(0, 2, 1, 3, 4).reshape(B, T * C, H, W)
            else:
                # last_only — take the last frame
                x = x[:, -1]
        return self.net(x)

    def _unpack(self, batch):
        if len(batch) == 3:
            x, y, _ = batch                 # co2 ignored
        else:
            x, y = batch
        # If a sequence dataset is misused as input, take the last frame
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


# ── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, C, H, W = 2, 2, 90, 180
    m = UNet2DVanillaBaseline(n_inputs=C, base_channels=64, H=H, W=W)
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, 1, H, W)
    y_hat = m(x)
    print(f'Input  : {tuple(x.shape)}')
    print(f'Output : {tuple(y_hat.shape)}')
    print(f'Params : {sum(p.numel() for p in m.parameters()):,}')
    loss = m._loss(y_hat, y)
    print(f'MSE    : {loss.item():.4f}')

    loss = m.training_step((x, y), 0)
    loss.backward()
    print(f'train_loss = {loss.item():.4f}, backward OK')

    # n_inputs=8 (full set) compat
    m8 = UNet2DVanillaBaseline(n_inputs=8, base_channels=64, H=H, W=W)
    x8 = torch.randn(B, 8, H, W)
    print(f'\\nn_inputs=8: out={tuple(m8(x8).shape)}, '
          f'params={sum(p.numel() for p in m8.parameters()):,}')

    # base=32 (smaller) compat
    m32 = UNet2DVanillaBaseline(n_inputs=C, base_channels=32, H=H, W=W)
    print(f'base=32: params={sum(p.numel() for p in m32.parameters()):,}')
