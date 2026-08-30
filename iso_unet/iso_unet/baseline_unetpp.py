"""
baseline_unetpp.py
─────────────────────────────────────────────────────────────────────────────
UNet++ (Zhou et al., 2018) single-frame baseline. Same I/O as
baseline_unet2d_vanilla, plugs directly into 2d_data_clean's train.py / eval.py.

UNet++ core: nested dense skip connections — extra conv blocks are inserted
between each encoder/decoder level to form a dense feature aggregation pyramid.
Nodes x[i, j]:
    x[0, 0]  x[0, 1]  x[0, 2]  x[0, 3]  x[0, 4]   <- output (j=L)
    x[1, 0]  x[1, 1]  x[1, 2]  x[1, 3]
    x[2, 0]  x[2, 1]  x[2, 2]
    x[3, 0]  x[3, 1]
    x[4, 0]                                       <- bottleneck

    x[i, j] = ConvBlock( concat( x[i,0], x[i,1], ..., x[i,j-1], up(x[i+1,j-1]) ) )

Implementation stripped from baselines/UNetPlusPlus/pytorch/nnunet/network_architecture/
generic_UNetPlusPlus.py, but without the entire nnunet package — single-file,
self-contained, BatchNorm + ReLU (the original nnunet uses LeakyReLU + Dropout;
we use BN+ReLU without dropout, consistent with the other baselines).

I/O:
    forward(x): (B, C, H, W) -> (B, 1, H, W)
    batch (x, y) or (x, y, co2) both accepted (co2 ignored)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


# ── Basic conv block (same spec as baseline_unet2d_vanilla) ─────────────────

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


# ── UNet++ backbone ─────────────────────────────────────────────────────────

class UNetPlusPlus(nn.Module):
    """
    UNet++ with nested dense skip connections.

    Args:
        in_channels:      number of input channels
        out_channels:     number of output channels (regression = 1)
        base_channels:    width of the shallowest level (filters[i] = base * 2^i, capped at max_channels).
                          Default 64, matching the vanilla U-Net baseline width used in the paper.
        H, W:             original spatial size; internally padded to be divisible by 2^depth
        depth:            number of pools (default 4 -> 5 levels, aligned with vanilla UNet)
        max_channels:     channel cap per level (to prevent the deepest level from exploding)
        deep_supervision: True -> average outputs from all x[0, 1..depth] (UNet++ paper trick).
                          During inference, still returns a single (B, 1, H, W), preserving the pipeline.
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=64,
                 H=90, W=180, depth=4, max_channels=512,
                 deep_supervision=False):
        super().__init__()
        assert depth >= 1
        self.depth = depth
        self.H_in, self.W_in = H, W

        factor = 2 ** depth
        self.H_p = ((H - 1) // factor + 1) * factor      # 90 → 96 (depth=4)
        self.W_p = ((W - 1) // factor + 1) * factor      # 180 → 192
        self.pad_top    = (self.H_p - H) // 2
        self.pad_bottom = self.H_p - H - self.pad_top
        self.pad_left   = (self.W_p - W) // 2
        self.pad_right  = self.W_p - W - self.pad_left

        filters = [min(base_channels * (2 ** i), max_channels)
                   for i in range(depth + 1)]
        self.filters = filters

        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False)

        # Encoder backbone: x[i, 0]
        self.backbone = nn.ModuleList([
            _ConvBlock(in_channels if i == 0 else filters[i - 1], filters[i])
            for i in range(depth + 1)
        ])

        # Nested nodes: x[i, j] for j >= 1, i + j <= depth
        # input ch = j * filters[i] (prior siblings at this level)
        #          + filters[i + 1] (upsampled lower-level node)
        self.nested = nn.ModuleDict()
        for i in range(depth):
            for j in range(1, depth - i + 1):
                in_ch = j * filters[i] + filters[i + 1]
                self.nested[f'x_{i}_{j}'] = _ConvBlock(in_ch, filters[i])

        # Output head(s) at the top level
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.heads = nn.ModuleList([
                nn.Conv2d(filters[0], out_channels, kernel_size=1)
                for _ in range(depth)
            ])
        else:
            self.head = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # pad → (H_p, W_p)
        x = F.pad(x,
                  (self.pad_left, self.pad_right,
                   self.pad_top,  self.pad_bottom),
                  mode='constant', value=0)

        # encoder backbone — x[i, 0]
        nodes = {(0, 0): self.backbone[0](x)}
        for i in range(1, self.depth + 1):
            nodes[(i, 0)] = self.backbone[i](self.pool(nodes[(i - 1, 0)]))

        # nested fill — outer loop j (skip column), inner loop i (depth)
        for j in range(1, self.depth + 1):
            for i in range(self.depth - j + 1):
                prior = [nodes[(i, k)] for k in range(j)]
                up    = self.up(nodes[(i + 1, j - 1)])
                # Sizes are consistent under exact division; safety net for non-divisible H/W
                if up.shape[-2:] != prior[0].shape[-2:]:
                    up = F.interpolate(up, size=prior[0].shape[-2:],
                                       mode='bilinear', align_corners=False)
                nodes[(i, j)] = self.nested[f'x_{i}_{j}'](
                    torch.cat(prior + [up], dim=1))

        # head + crop back
        if self.deep_supervision:
            outs = [self.heads[j - 1](nodes[(0, j)])
                    for j in range(1, self.depth + 1)]
            out = torch.stack(outs, dim=0).mean(dim=0)
        else:
            out = self.head(nodes[(0, self.depth)])

        return out[:, :,
                   self.pad_top : self.pad_top + self.H_in,
                   self.pad_left: self.pad_left + self.W_in]


# ── Lightning wrapper (same interface as UNet2DVanillaBaseline) ─────────────

class UNetPlusPlusBaseline(L.LightningModule):
    """
    UNet++ single-frame baseline.

        forward(x): (B, C, H, W) → (B, 1, H, W)
        loss:       cos(lat) weighted MSE
        batch:      (x, y) or (x, y, co2) — co2 ignored
    """
    def __init__(
        self,
        n_inputs:         int   = 2,
        out_channels:     int   = 1,
        base_channels:    int   = 64,
        H:                int   = 90,
        W:                int   = 180,
        depth:            int   = 4,
        max_channels:     int   = 512,
        deep_supervision: bool  = False,
        weights                 = None,
        lr:               float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr

        if weights is not None:
            self.register_buffer('lat_weights',
                                 torch.as_tensor(weights, dtype=torch.float32))
        else:
            self.lat_weights = None

        self.net = UNetPlusPlus(
            in_channels=n_inputs, out_channels=out_channels,
            base_channels=base_channels, H=H, W=W,
            depth=depth, max_channels=max_channels,
            deep_supervision=deep_supervision,
        )

    def forward(self, x):
        return self.net(x)

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
    m = UNetPlusPlusBaseline(n_inputs=C, base_channels=64, H=H, W=W, depth=4)
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

    # deep supervision variant
    m_ds = UNetPlusPlusBaseline(n_inputs=C, base_channels=64, H=H, W=W,
                                depth=4, deep_supervision=True)
    print(f'\nDeep-supervision out: {tuple(m_ds(x).shape)}, '
          f'params={sum(p.numel() for p in m_ds.parameters()):,}')

    # smaller variant (base=16)
    m16 = UNetPlusPlusBaseline(n_inputs=C, base_channels=16, H=H, W=W)
    print(f'base=16: params={sum(p.numel() for p in m16.parameters()):,}')
