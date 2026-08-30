"""
baseline_dcsaunet.py
─────────────────────────────────────────────────────────────────────────────
DCSAU-Net (Xu et al., 2023, https://arxiv.org/abs/2202.00972) single-frame baseline.
Same I/O as baseline_unet2d_vanilla / baseline_unetpp / baseline_attunet —
plugs directly into 2d_data_clean's train.py / eval.py.

DCSAU-Net = "Deeper and more Compact Split-Attention U-Net":
    * PFC (Primary Feature Conservation) stem:
        7x7 conv -> depthwise-conv (with residual) -> pointwise-conv;
        used as the input feature extractor, replacing the typical UNet's double 3x3 conv.
    * CSA (Compound Split-Attention) blocks as the encoder/decoder backbone:
        Based on ResNeSt's SplAtConv2d (Zhang et al. 2020); the "compound" variant adds
        an extra conv2 branch (DCSAU-Net's change — sum the two splits x1, x2 and conv again).
    * 4 pools, 4 upsample-concats, same depth as the other baselines.

This implementation consolidates dependencies from the original 4 files
(DCSAU_Net.py + encoder.py + resnet.py + splat.py) into one, and:
    * Lets PFC accept any in_channels (the original hard-codes 3)
    * Adds H/W pad-to-16 (90x180 -> 96x192) + crop-back
    * Removes unused dead code (rectified-conv / dropblock / dilation-4)
    * Adds Lightning wrapper + cos(lat) weighted MSE
Retains the original spec: base_channels=64 PFC, ConvFea=[32,64,128,256,512],
            radix=2, cardinality=1, bottleneck_width=64, avd=True, avd_first=False.
The original (3ch input) is 2.6M params; with 9ch input here, the count is on the same order.

I/O:
    forward(x):  (B, C, H, W) -> (B, 1, H, W)
    batch (x, y) or (x, y, co2) both accepted (co2 ignored)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


# ══════════════════════════════════════════════════════════════════════
#  ResNeSt Split-Attention building blocks
# ══════════════════════════════════════════════════════════════════════

class _rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class _SplAtConv2d(nn.Module):
    """
    Split-Attention Conv2d (ResNeSt), DCSAU-Net "compound" variant — adds an
    extra self.conv2 branch as another convolution on x2, then merges the two
    splits via attention weights.
    """
    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, radix=2, reduction_factor=4,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix       = radix
        self.cardinality = groups
        self.channels    = channels

        self.conv = nn.Conv2d(in_channels, channels * radix, kernel_size,
                              stride, padding, dilation,
                              groups=groups * radix, bias=bias)
        self.bn0  = norm_layer(channels * radix)
        self.bn2  = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc1  = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1  = norm_layer(inter_channels)
        self.fc2  = nn.Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        self.rsoftmax = _rSoftMax(radix, groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation,
                               groups=groups * radix, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn0(self.conv(x)))
        batch, rchannel = x.shape[:2]
        x1, x2 = torch.split(x, rchannel // self.radix, dim=1)

        x2 = x2 + x1
        x2 = self.relu(self.bn2(self.conv2(x2)))

        splited = (x1, x2)
        gap = sum(splited)
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.relu(self.bn1(self.fc1(gap)))

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        attens = torch.split(atten, rchannel // self.radix, dim=1)

        out = sum([a * s for a, s in zip(attens, splited)])
        return out.contiguous()


class _CSABottleneck(nn.Module):
    """
    ResNeSt-style Bottleneck used in DCSAU-Net (compound split-attention).
    Layout: 1×1 conv → SplAt 3×3 → 1×1 conv (+ residual). expansion=4.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=2, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 norm_layer=nn.BatchNorm2d, custom=0):
        super().__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        if custom != 0:
            inplanes = custom

        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1   = norm_layer(group_width)
        self.radix = radix
        self.avd   = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.conv2 = _SplAtConv2d(
            group_width, group_width, kernel_size=3,
            stride=stride, padding=dilation, dilation=dilation,
            groups=cardinality, bias=False,
            radix=radix, norm_layer=norm_layer,
        )

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3   = norm_layer(planes * 4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)               # SplAt has its own bn + relu

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


# ══════════════════════════════════════════════════════════════════════
#  CSA layer builder (mirrors the original ResNet._make_layer + avg_down=True)
# ══════════════════════════════════════════════════════════════════════

class _CSALayerBuilder:
    """
    Stateful builder — self.inplanes is updated to the new output channels after
    each make_layer call, matching the original ResNet. The encoder layers share
    one builder state; the decoder explicitly overrides via the inchannel argument
    (since concat causes channel jumps, inplanes cannot simply accumulate).
    """
    def __init__(self, radix=2, cardinality=1, bottleneck_width=64,
                 avd=True, avd_first=False, stem_width=32,
                 norm_layer=nn.BatchNorm2d):
        self.radix             = radix
        self.cardinality       = cardinality
        self.bottleneck_width  = bottleneck_width
        self.avd               = avd
        self.avd_first         = avd_first
        self.inplanes          = stem_width * 2                # default 64
        self.norm_layer        = norm_layer

    def make_layer(self, planes, blocks, stride=1, dilation=1,
                   is_first=True, inchannel=0):
        block = _CSABottleneck
        out_ch = planes * block.expansion
        downsample = None

        if stride != 1 or self.inplanes != out_ch or inchannel != 0:
            if inchannel != 0:
                self.inplanes = inchannel
            # avg_down=True path
            kp = stride if dilation == 1 else 1
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=kp, stride=kp,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, out_ch, kernel_size=1, stride=1, bias=False),
                self.norm_layer(out_ch),
            )

        layers = [block(
            self.inplanes, planes, stride, downsample=downsample,
            radix=self.radix, cardinality=self.cardinality,
            bottleneck_width=self.bottleneck_width,
            avd=self.avd, avd_first=self.avd_first,
            dilation=1, is_first=is_first,
            norm_layer=self.norm_layer, custom=inchannel,
        )]
        self.inplanes = out_ch
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes,
                radix=self.radix, cardinality=self.cardinality,
                bottleneck_width=self.bottleneck_width,
                avd=self.avd, avd_first=self.avd_first,
                dilation=dilation,
                norm_layer=self.norm_layer,
            ))
        return nn.Sequential(*layers)


# ══════════════════════════════════════════════════════════════════════
#  PFC stem + Up concat
# ══════════════════════════════════════════════════════════════════════

class _PFC(nn.Module):
    """
    Primary Feature Conservation (DCSAU-Net paper Fig. 3):
        7x7 conv -> depthwise-conv (residual) -> pointwise-conv.
    Note: keeps the original's non-standard "Conv -> ReLU -> BN" order (matches the paper implementation).
    """
    def __init__(self, in_channels, channels=64, kernel_size=7):
        super().__init__()
        p = kernel_size // 2
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size, padding=p),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=p),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x) + residual
        x = self.pointwise(x)
        return x


class _Up(nn.Module):
    """Bilinear upsample + size-aligning pad + concat skip."""
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffX or diffY:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        return torch.cat([x2, x1], dim=1)


# ══════════════════════════════════════════════════════════════════════
#  DCSAU-Net backbone
# ══════════════════════════════════════════════════════════════════════

class DCSAUNet(nn.Module):
    """
    DCSAU-Net main module.

    Args:
        in_channels:   number of input channels
        out_channels:  number of output channels (regression = 1)
        H, W:          original spatial size; internally padded to a multiple of 16
        pfc_channels:  PFC stem width (default 64, matches the paper). Also used as input to out_conv.
        pfc_kernel:    PFC kernel size (default 7)

    Channel layout (identical to the original CSA encoder.py):
        x1=PFC=64 -> pool -> down1=128 -> pool -> down2=256 -> pool -> down3=512
            -> pool -> down4=512
        -> up_conv1(cat 512+512=1024) -> up1=256
        -> up_conv2(cat 256+256=512)  -> up2=128
        -> up_conv3(cat 128+128=256)  -> up3=64
        -> up_conv4(cat 64+64=128)    -> up4=64
        -> out_conv(64 -> out_channels)
    """
    def __init__(self, in_channels=2, out_channels=1, H=90, W=180,
                 pfc_channels=64, pfc_kernel=7):
        super().__init__()
        self.H_in, self.W_in = H, W
        self.H_p = ((H - 1) // 16 + 1) * 16
        self.W_p = ((W - 1) // 16 + 1) * 16
        self.pad_top    = (self.H_p - H) // 2
        self.pad_bottom = self.H_p - H - self.pad_top
        self.pad_left   = (self.W_p - W) // 2
        self.pad_right  = self.W_p - W - self.pad_left

        # PFC stem
        self.pfc = _PFC(in_channels, channels=pfc_channels, kernel_size=pfc_kernel)

        # CSA layers — channel plan hard-coded in the original paper
        ConvFea = [32, 64, 128, 256, 512]
        blocks  = [2, 2, 2, 2]
        # stem_width=32 -> builder.inplanes=64 (aligned with PFC output)
        builder = _CSALayerBuilder(stem_width=pfc_channels // 2)

        # Encoder
        self.down1 = builder.make_layer(ConvFea[0], blocks[0], is_first=False)         # 64 → 128
        self.down2 = builder.make_layer(ConvFea[1], blocks[1])                          # 128 → 256
        self.down3 = builder.make_layer(ConvFea[2], blocks[2], dilation=1)              # 256 → 512
        self.down4 = builder.make_layer(ConvFea[2], blocks[3], dilation=1)              # 512 → 512
        # Decoder (inchannel = channels after concat)
        self.up1   = builder.make_layer(ConvFea[1], blocks[0], dilation=1, inchannel=1024)  # → 256
        self.up2   = builder.make_layer(ConvFea[0], blocks[1], dilation=1, inchannel=512)   # → 128
        self.up3   = builder.make_layer(ConvFea[0] // 2, blocks[2], inchannel=256)          # → 64
        self.up4   = builder.make_layer(ConvFea[0] // 2, blocks[3],
                                        is_first=False, inchannel=128)                      # → 64

        self.maxpool  = nn.MaxPool2d(2)
        self.up_conv1 = _Up()
        self.up_conv2 = _Up()
        self.up_conv3 = _Up()
        self.up_conv4 = _Up()

        self.out_conv = nn.Conv2d(pfc_channels, out_channels, kernel_size=1)

        # paper-style init (same as the original ResNet)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = F.pad(x,
                  (self.pad_left, self.pad_right,
                   self.pad_top,  self.pad_bottom),
                  mode='constant', value=0)

        x1 = self.pfc(x)
        x2 = self.maxpool(x1)
        x3 = self.down1(x2)
        x4 = self.maxpool(x3)
        x5 = self.down2(x4)
        x6 = self.maxpool(x5)
        x7 = self.down3(x6)
        x8 = self.maxpool(x7)
        x9 = self.down4(x8)

        x10 = self.up_conv1(x9, x7)
        x11 = self.up1(x10)
        x12 = self.up_conv2(x11, x5)
        x13 = self.up2(x12)
        x14 = self.up_conv3(x13, x3)
        x15 = self.up3(x14)
        x16 = self.up_conv4(x15, x1)
        x17 = self.up4(x16)

        out = self.out_conv(x17)
        return out[:, :,
                   self.pad_top : self.pad_top + self.H_in,
                   self.pad_left: self.pad_left + self.W_in]


# ══════════════════════════════════════════════════════════════════════
#  Lightning wrapper (same interface as the other baselines)
# ══════════════════════════════════════════════════════════════════════

class DCSAUNetBaseline(L.LightningModule):
    """
    DCSAU-Net single-frame baseline.

        forward(x): (B, C, H, W) → (B, 1, H, W)
        loss:       cos(lat) weighted MSE
        batch:      (x, y) or (x, y, co2) — co2 ignored
    """
    def __init__(
        self,
        n_inputs:      int   = 2,
        out_channels:  int   = 1,
        H:             int   = 90,
        W:             int   = 180,
        pfc_channels:  int   = 64,
        pfc_kernel:    int   = 7,
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

        self.net = DCSAUNet(
            in_channels=n_inputs, out_channels=out_channels,
            H=H, W=W,
            pfc_channels=pfc_channels, pfc_kernel=pfc_kernel,
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
#  Smoke test + parity check vs upstream
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Same setup as the original (3 channels, 96x192) for the param-count check
    m3 = DCSAUNetBaseline(n_inputs=3, out_channels=1, H=96, W=192)
    print(f'[3ch, 96x192]  params: {sum(p.numel() for p in m3.parameters()):,}  '
          f'(original 2,598,785)')

    B, C, H, W = 2, 9, 90, 180
    m = DCSAUNetBaseline(n_inputs=C, out_channels=1, H=H, W=W)
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, 1, H, W)
    y_hat = m(x)
    print(f'\nInput  : {tuple(x.shape)}')
    print(f'Output : {tuple(y_hat.shape)}   ← (B, 1, H, W) ✓')
    print(f'Params : {sum(p.numel() for p in m.parameters()):,}')
    loss = m._loss(y_hat, y)
    print(f'MSE    : {loss.item():.4f}')

    loss = m.training_step((x, y), 0)
    loss.backward()
    print(f'train_loss = {loss.item():.4f}, backward OK')
