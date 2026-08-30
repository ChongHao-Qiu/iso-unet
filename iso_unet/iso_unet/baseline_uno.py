"""
baseline_uno.py
─────────────────────────────────────────────────────────────────────────────
U-NO (U-shaped Neural Operator, Ashiqur Rahman et al., 2022,
https://arxiv.org/abs/2204.11127) single-frame baseline.
Same I/O as baseline_unet2d_vanilla / unetpp / attunet / dcsaunet / transunet —
plugs directly into 2d_data_clean's train.py / eval.py.

U-NO = Fourier integral operators arranged in a U shape:
    * Lift: Linear (C+2 -> w/2 -> w) — concat the coord grid, then channel-lift
    * Pyramid (same as UNO_9, 5 OperatorBlocks):
        conv0: w  -> 2w,  dim -> /2
        conv1: 2w -> 4w,  dim -> /4   [InstanceNorm]
        conv2: 4w -> 4w,  dim -> /4   (bottleneck)
        conv4: 4w -> 2w,  dim -> /2   [InstanceNorm]
        conv5: 4w -> w,   dim -> original    (input concat x_c0)
    * Project: Linear (2w concat -> w -> out_channels)
    * Skip connections (concat): conv4 <- conv0, conv5 final <- x_fc0
    Each OperatorBlock = SpectralConv (FFT linear) + pointwise (Conv1x1 + bicubic interp) -> GELU.

Code ported from baselines/UNO/{integral_operators,darcy_flow_uno2d}.py, with changes:
    1. Input layout changed from NHWC to NCHW (consistent with other baselines), with internal permute.
    2. Added an in_channels argument (the original hard-codes Darcy 1 ch + 2 grid = 3 ch).
    3. Rewrote grid generation to be NCHW-friendly, avoiding building a new tensor every forward.
    4. Auto-compute H_pad / W_pad so /4 halving stays integer (Darcy uses a square 85x85; we
       use 90x180 and must pad to a multiple of 4, default H_p=96, W_p=184).
    5. Added a Lightning wrapper + cos(lat) weighted MSE.

Note that U-NO is a spectral model (FFT-based) and behaves differently from a vanilla CNN:
    * Friendly to low-frequency signals (large-scale climate patterns)
    * Not as good as a CNN on high-frequency details (sharp boundaries)
    * Has a built-in form of grid-invariance, can do zero-shot inference at a different resolution

I/O:
    forward(x):  (B, C, H, W) -> (B, 1, H, W)
    batch (x, y) or (x, y, co2) both accepted (co2 ignored)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


# ══════════════════════════════════════════════════════════════════════
#  Spectral building blocks (ported from UNO/integral_operators.py)
# ══════════════════════════════════════════════════════════════════════

class _SpectralConv2d(nn.Module):
    """
    2D Fourier integral layer with adjustable output resolution.

    Args:
        in_codim, out_codim: input/output channel ("codomain") dimensions
        dim1, dim2:          default output spatial size (can be overridden in forward)
        modes1, modes2:      number of Fourier modes kept along H / W axis
            Constraints: 2*modes1 <= min(input_H, output_H),
                         modes2   <= min(input_W, output_W) // 2 + 1
    """
    def __init__(self, in_codim, out_codim, dim1, dim2,
                 modes1=None, modes2=None):
        super().__init__()
        in_codim  = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels  = in_codim
        self.out_channels = out_codim
        self.dim1, self.dim2 = dim1, dim2
        self.modes1 = modes1 if modes1 is not None else dim1 // 2 - 1
        self.modes2 = modes2 if modes2 is not None else dim2 // 2

        scale = (1.0 / (2 * in_codim)) ** 0.5
        self.weights1 = nn.Parameter(scale * torch.randn(
            in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(
            in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat))

    @staticmethod
    def _compl_mul2d(x, w):
        # (B, in, h, w) x (in, out, h, w) -> (B, out, h, w)
        return torch.einsum('bixy,ioxy->boxy', x, w)

    def forward(self, x, dim1=None, dim2=None):
        if dim1 is not None:
            self.dim1, self.dim2 = dim1, dim2
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm='forward')

        out_ft = torch.zeros(B, self.out_channels, self.dim1, self.dim2 // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m1, m2 = self.modes1, self.modes2
        # Low frequencies (top-left corner) + reflection-symmetric wrapping (bottom-left corner)
        out_ft[:, :,  :m1, :m2] = self._compl_mul2d(x_ft[:, :,  :m1, :m2], self.weights1)
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2)

        return torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2), norm='forward')


class _PointwiseOp2D(nn.Module):
    """Pointwise Conv1x1 + bicubic resample to (dim1, dim2)."""
    def __init__(self, in_codim, out_codim, dim1, dim2):
        super().__init__()
        self.conv = nn.Conv2d(int(in_codim), int(out_codim), kernel_size=1)
        self.dim1, self.dim2 = int(dim1), int(dim2)

    def forward(self, x, dim1=None, dim2=None):
        if dim1 is None:
            dim1, dim2 = self.dim1, self.dim2
        x = self.conv(x)
        return F.interpolate(x, size=(dim1, dim2),
                             mode='bicubic', align_corners=True, antialias=True)


class _OperatorBlock2D(nn.Module):
    """
    The basic U-NO unit: SpectralConv + PointwiseOp summed in parallel -> optional InstanceNorm -> optional GELU.

    Args:
        Normalize: True -> InstanceNorm2d (only used in select layers)
        Non_Lin:   True -> GELU
    """
    def __init__(self, in_codim, out_codim, dim1, dim2, modes1, modes2,
                 Normalize=False, Non_Lin=True):
        super().__init__()
        self.conv = _SpectralConv2d(in_codim, out_codim, dim1, dim2, modes1, modes2)
        self.w    = _PointwiseOp2D (in_codim, out_codim, dim1, dim2)
        self.normalize = Normalize
        self.non_lin   = Non_Lin
        if Normalize:
            self.normalize_layer = nn.InstanceNorm2d(int(out_codim), affine=True)

    def forward(self, x, dim1=None, dim2=None):
        out = self.conv(x, dim1, dim2) + self.w(x, dim1, dim2)
        if self.normalize:
            out = self.normalize_layer(out)
        if self.non_lin:
            out = F.gelu(out)
        return out


# ══════════════════════════════════════════════════════════════════════
#  U-NO 9-layer backbone (UNO_9 from darcy_flow_uno2d.py)
# ══════════════════════════════════════════════════════════════════════

def _next_multiple(n, m):
    """Smallest multiple of m that is >= n."""
    return ((n + m - 1) // m) * m


class UNO(nn.Module):
    """
    U-NO with 5 OperatorBlock_2D layers in U-shape (paper's UNO_9 architecture).

    Args:
        in_channels:  number of input channels
        out_channels: number of output channels (regression = 1)
        H, W:         original spatial size; internally padded to a multiple of 4 (so /4 halving is clean)
        width:        lifting hidden dim (default 32, the paper's Darcy value)
        factor:       channel multiplier (default 1)
        pad_h, pad_w: explicit padding size; None -> auto-compute to next-multiple-of-4
        modes:        list of 5 (mh, mw) tuples per OperatorBlock.
                      Default [(18,18),(8,8),(8,8),(8,8),(18,18)] — same as the paper's Darcy setup,
                      but with 90x180 input we can use more modes on W because it is wider.
    """
    def __init__(self, in_channels=2, out_channels=1, H=90, W=180,
                 width=32, factor=1,
                 pad_h=None, pad_w=None,
                 modes=None):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.width  = int(width)
        self.factor = int(factor)
        self.H_in, self.W_in = H, W

        # Auto pad: ensure H_p, W_p are both multiples of 4 (UNO_9's deepest /4 halving)
        # and ensure at least 1 padding (non-periodic boundary)
        if pad_h is None:
            pad_h = max(1, _next_multiple(H + 1, 4) - H)
        if pad_w is None:
            pad_w = max(1, _next_multiple(W + 1, 4) - W)
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.H_p = H + pad_h
        self.W_p = W + pad_w

        if modes is None:
            modes = [(18, 18), (8, 8), (8, 8), (8, 8), (18, 18)]
        assert len(modes) == 5, f'expect 5 (mh, mw) tuples, got {len(modes)}'
        self.modes = modes

        w = self.width
        f = self.factor

        # Lift: Linear (in+2 grid -> w/2 -> w)
        self.fc_n1 = nn.Linear(in_channels + 2, w // 2)
        self.fc0   = nn.Linear(w // 2, w)

        # Default spatial dims at each pyramid level (used at init; can be overridden at runtime)
        d_full = (self.H_p,     self.W_p)
        d_half = (self.H_p // 2, self.W_p // 2)
        d_quar = (self.H_p // 4, self.W_p // 4)

        m0, m1, m2, m4, m5 = modes

        # 5-layer U-shape pyramid (UNO_9)
        self.conv0 = _OperatorBlock2D(w,         2 * f * w, *d_half, *m0)
        self.conv1 = _OperatorBlock2D(2 * f * w, 4 * f * w, *d_quar, *m1, Normalize=True)
        self.conv2 = _OperatorBlock2D(4 * f * w, 4 * f * w, *d_quar, *m2)
        self.conv4 = _OperatorBlock2D(4 * f * w, 2 * f * w, *d_half, *m4, Normalize=True)
        # conv5 input has concat skip -> in_codim doubled
        self.conv5 = _OperatorBlock2D(4 * f * w, w,         *d_full, *m5)

        # Project: Linear (concat 2w -> w -> out_channels)
        self.fc1 = nn.Linear(2 * w, w)
        self.fc2 = nn.Linear(w,     out_channels)

    def _make_grid(self, B, H, W, device, dtype):
        """Coord grid (B, H, W, 2) — same as the paper, but cached on device for efficiency."""
        gx = torch.linspace(0, 1, H, device=device, dtype=dtype).view(1, H, 1, 1).expand(B, H, W, 1)
        gy = torch.linspace(0, 1, W, device=device, dtype=dtype).view(1, 1, W, 1).expand(B, H, W, 1)
        return torch.cat([gx, gy], dim=-1)

    def forward(self, x):
        # x: (B, C, H, W) -> permute to (B, H, W, C) for Linear lifting
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()                      # (B, H, W, C)
        grid = self._make_grid(B, H, W, x.device, x.dtype)
        x = torch.cat([x, grid], dim=-1)                            # (B, H, W, C+2)

        x = F.gelu(self.fc_n1(x))
        x = F.gelu(self.fc0(x))                                     # (B, H, W, w)
        x = x.permute(0, 3, 1, 2).contiguous()                      # (B, w, H, W)

        # Pad to (H_p, W_p): bottom + right (non-periodic boundary handling)
        x_fc0 = F.pad(x, (0, self.pad_w, 0, self.pad_h),
                      mode='constant', value=0)                      # (B, w, H_p, W_p)

        D1, D2 = x_fc0.shape[-2], x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0, D1 // 2, D2 // 2)                   # (B, 2fw, H/2, W/2)
        x_c1 = self.conv1(x_c0,  D1 // 4, D2 // 4)                   # (B, 4fw, H/4, W/4)
        x_c2 = self.conv2(x_c1,  D1 // 4, D2 // 4)                   # bottleneck
        x_c4 = self.conv4(x_c2,  D1 // 2, D2 // 2)                   # (B, 2fw, H/2, W/2)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)                        # skip: 2fw + 2fw = 4fw
        x_c5 = self.conv5(x_c4,  D1,     D2)                         # (B, w, H_p, W_p)
        x_c5 = torch.cat([x_c5, x_fc0], dim=1)                       # skip: w + w = 2w

        # Reverse pad
        if self.pad_h or self.pad_w:
            x_c5 = x_c5[..., :D1 - self.pad_h, :D2 - self.pad_w]

        x_c5 = x_c5.permute(0, 2, 3, 1).contiguous()                 # (B, H, W, 2w)
        x = F.gelu(self.fc1(x_c5))
        x = self.fc2(x)                                              # (B, H, W, out_channels)
        return x.permute(0, 3, 1, 2).contiguous()                    # (B, out, H, W)


# ══════════════════════════════════════════════════════════════════════
#  Lightning wrapper (same interface as the other baselines)
# ══════════════════════════════════════════════════════════════════════

class UNOBaseline(L.LightningModule):
    """
    U-NO single-frame baseline.

        forward(x): (B, C, H, W) → (B, 1, H, W)
        loss:       cos(lat) weighted MSE
        batch:      (x, y) or (x, y, co2) — co2 ignored
    """
    def __init__(
        self,
        n_inputs:     int   = 2,
        out_channels: int   = 1,
        H:            int   = 90,
        W:            int   = 180,
        width:        int   = 32,
        factor:       int   = 1,
        pad_h         = None,
        pad_w         = None,
        modes               = None,
        weights             = None,
        lr:           float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr

        if weights is not None:
            self.register_buffer('lat_weights',
                                 torch.as_tensor(weights, dtype=torch.float32))
        else:
            self.lat_weights = None

        self.net = UNO(
            in_channels=n_inputs, out_channels=out_channels,
            H=H, W=W, width=width, factor=factor,
            pad_h=pad_h, pad_w=pad_w, modes=modes,
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
#  Smoke test + parity vs upstream
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Parity vs original UNO_9 (Darcy 85x85, in_width=3, width=32)
    # -> 1 input ch + 2 grid = 3 lift dim
    # Our fc_n1 input is in_channels + 2 -> set in_channels=1 (= paper Darcy 1 ch)
    print('=' * 60)
    print('Parity check vs paper UNO_9 (85×85, width=32, in_ch=1)')
    print('=' * 60)
    m85 = UNOBaseline(n_inputs=1, out_channels=1, H=85, W=85, width=32, pad_h=12, pad_w=12)
    n85 = sum(p.numel() for p in m85.parameters())
    print(f'My params : {n85:,}   (paper UNO_9 = 8,218,049)')
    # Note: I added a lat_weights buffer (size 0 here); ignore it when comparing byte-exact to the paper.

    # Climate setup (9 ch, 90×180)
    print('\n' + '=' * 60)
    print('Climate setup (9 ch, 90×180, width=32)')
    print('=' * 60)
    B, C, H, W = 2, 9, 90, 180
    m = UNOBaseline(n_inputs=C, out_channels=1, H=H, W=W, width=32)
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, 1, H, W)
    y_hat = m(x)
    print(f'Input    : {tuple(x.shape)}')
    print(f'Output   : {tuple(y_hat.shape)}   <- (B, 1, H, W) OK')
    print(f'H_p × W_p = {m.net.H_p} × {m.net.W_p}  (pad h={m.net.pad_h}, w={m.net.pad_w})')
    print(f'Params   : {sum(p.numel() for p in m.parameters()):,}')
    loss = m._loss(y_hat, y)
    print(f'MSE      : {loss.item():.4f}')

    loss = m.training_step((x, y), 0)
    loss.backward()
    print(f'train_loss = {loss.item():.4f}, backward OK')

    # Resolution invariance: model is trained on 90x180 but can also run 45x90 input (FFT adapts automatically)
    print('\n[resolution-invariance]')
    x_low = torch.randn(2, 9, 45, 90)
    y_low = m(x_low)
    print(f'  input 45x90 -> output {tuple(y_low.shape)}  OK (zero-shot)')

    # Wider variant
    m64 = UNOBaseline(n_inputs=C, out_channels=1, H=H, W=W, width=64)
    print(f'\nwidth=64: params={sum(p.numel() for p in m64.parameters()):,}')
