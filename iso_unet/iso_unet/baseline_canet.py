"""
baseline_canet.py
─────────────────────────────────────────────────────────────────────────────
CA-Net (Comprehensive Attention U-Net) ported to iso_unet for d18Op regression.

Original paper:
    "CA-Net: Comprehensive Attention Convolutional Neural Networks for Explainable
     Medical Image Segmentation" (TMI 2020), Gu et al.
Original code: https://github.com/HiLab-git/CA-Net  (medical seg for ISIC2018, 224x300)

Main changes (differences from the original):
    1. Output changed from segmentation (Softmax2d + n_classes) to regression (Linear, n_classes=1)
    2. The hard-coded AvgPool size in SE_Conv_Block -> AdaptiveAvgPool2d (works with any input size)
    3. The torch.rand().cuda() size compensation in UpCat -> F.interpolate (avoids the CUDA
       dependency and the injected random noise)
    4. Internally pad 90x180 input to 96x192 and crop back at the decoder tail to fit 5 pools
       (1/32 downsampling)
    5. Updated all deprecated PyTorch APIs (F.upsample -> F.interpolate, F.sigmoid ->
       torch.sigmoid, init.kaiming_normal_, init.constant_)
    6. Removed unused attention-mode variants from the original, keeping only:
       - GridAttention: mode='concatenation'
       - NONLocalBlock: mode='embedded_gaussian'
    7. Lightning wrapper (CANetBaseline) shares the UNet2DBaseline interface, plugs directly
       into SingleFrameDataset

I/O:
    forward(x):  (B, C, H=90, W=180) → (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


# ══════════════════════════════════════════════════════════════════════
#  Weight init (same as the original, but using modern PyTorch APIs)
# ══════════════════════════════════════════════════════════════════════

def _kaiming_init(m):
    cn = m.__class__.__name__
    if cn.find('Conv') != -1 and hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif cn.find('Linear') != -1 and hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif cn.find('BatchNorm') != -1 and hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


# ══════════════════════════════════════════════════════════════════════
#  Basic blocks
# ══════════════════════════════════════════════════════════════════════

def _conv3x3(ci, co, stride=1, bias=False, groups=1):
    return nn.Conv2d(ci, co, kernel_size=3, stride=stride, padding=1,
                     groups=groups, bias=bias)


class ConvBlock(nn.Module):
    """2x (Conv3x3 + BN + ReLU)."""
    def __init__(self, ci, co, drop_out=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ci, co, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
            nn.Conv2d(co, co, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
        )
        self.drop_out = drop_out
        self.dropout  = nn.Dropout2d(0.5) if drop_out else None

    def forward(self, x):
        x = self.conv(x)
        if self.drop_out:
            x = self.dropout(x)
        return x


class UpCat(nn.Module):
    """
    Deconv + concat (with skip). Same semantics as the original, but:
        * uses F.interpolate to handle size mismatch (instead of the original's
          torch.rand().cuda() noise padding)
    """
    def __init__(self, in_feat, out_feat, is_deconv=True):
        super().__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, inputs, down_outputs):
        outputs = self.up(down_outputs)
        # Align size to inputs (use bilinear rather than padding random noise)
        if outputs.shape[-2:] != inputs.shape[-2:]:
            outputs = F.interpolate(outputs, size=inputs.shape[-2:],
                                    mode='bilinear', align_corners=False)
        return torch.cat([inputs, outputs], dim=1)


class UnetDsv(nn.Module):
    """Deep-supervision projection: ci -> out_size channels, then upsample to scale_factor (H, W)."""
    def __init__(self, in_size, out_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)
        self.scale_factor = scale_factor   # (H, W) tuple

    def forward(self, x):
        x = self.conv(x)
        return F.interpolate(x, size=self.scale_factor, mode='bilinear', align_corners=False)


# ══════════════════════════════════════════════════════════════════════
#  Channel attention: SE_Conv_Block
#  Change: hardcoded AvgPool2d size -> AdaptiveAvgPool2d((1, 1))
# ══════════════════════════════════════════════════════════════════════

class SEConvBlock(nn.Module):
    """
    SE-style channel attention + bottleneck conv block.
        Adds an SE gate on top of the inplanes -> planes*2 -> planes residual block
        (sum of two channel-attention paths: average + max).
    """
    def __init__(self, inplanes, planes, stride=1, drop_out=False):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes * 2)
        self.bn2   = nn.BatchNorm2d(planes * 2)
        self.conv3 = _conv3x3(planes * 2, planes)
        self.bn3   = nn.BatchNorm2d(planes)
        self.drop_out = drop_out
        self.dropout  = nn.Dropout2d(0.5) if drop_out else None

        # * change: use adaptive pool to support any input size
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        # SE excitation MLP
        reduced = max(1, planes // 2)
        self.fc1 = nn.Linear(planes * 2, reduced)
        self.fc2 = nn.Linear(reduced, planes * 2)

        # Align residual channels
        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(
                nn.Conv2d(inplanes, planes * 2, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * 2),
            )

    def _se(self, pool_out, ref):
        """pool_out: (B, planes*2, 1, 1). Returns attention-weighted features."""
        b, c = pool_out.size(0), pool_out.size(1)
        s = pool_out.view(b, c)
        s = self.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return s * ref, s

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))                        # (B, planes*2, H, W)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        # Two SE paths: avg + max
        avg_out, avg_att = self._se(self.gap(out), out)
        max_out, max_att = self._se(self.gmp(out), out)
        att_weight = avg_att + max_att

        out = avg_out + max_out + residual
        out = self.relu(out)

        out = self.relu(self.bn3(self.conv3(out)))             # (B, planes, H, W)
        if self.drop_out:
            out = self.dropout(out)
        return out, att_weight


# ══════════════════════════════════════════════════════════════════════
#  Grid Attention (concatenation mode, 2D only)
#  The original's 5 modes are reduced to only 'concatenation' (the one used in CA-Net's main path)
# ══════════════════════════════════════════════════════════════════════

class GridAttention2D(nn.Module):
    """
    Attention U-Net-style grid attention.
        x: (B, in_ch,   H,  W )   — skip feature to be gated
        g: (B, gate_ch, H', W')   — gating signal (from a deeper layer)
        return: (W_y, attn_map)
    """
    def __init__(self, in_channels, gating_channels, inter_channels=None,
                 sub_sample_factor=(1, 1)):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(1, in_channels // 2)
        self.theta = nn.Conv2d(in_channels, inter_channels,
                               kernel_size=sub_sample_factor,
                               stride=sub_sample_factor, padding=0, bias=True)
        self.phi   = nn.Conv2d(gating_channels, inter_channels,
                               kernel_size=1, stride=1, padding=0, bias=True)
        self.psi   = nn.Conv2d(inter_channels, 1,
                               kernel_size=1, stride=1, padding=0, bias=True)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.apply(_kaiming_init)

    def forward(self, x, g):
        input_size = x.size()
        theta_x    = self.theta(x)
        # Upsample gating signal to theta_x's spatial size
        phi_g = F.interpolate(self.phi(g), size=theta_x.shape[2:],
                              mode='bilinear', align_corners=False)
        f         = F.relu(theta_x + phi_g, inplace=True)
        sigm_psi  = torch.sigmoid(self.psi(f))
        # Upsample back to x's size
        sigm_psi  = F.interpolate(sigm_psi, size=input_size[2:],
                                  mode='bilinear', align_corners=False)
        y         = sigm_psi.expand_as(x) * x
        return self.W(y), sigm_psi


class MultiAttentionBlock(nn.Module):
    """Two parallel GridAttention2D paths -> concat -> 1x1 conv fusion."""
    def __init__(self, in_size, gate_size, inter_size, sub_sample_factor=(1, 1)):
        super().__init__()
        self.gate1 = GridAttention2D(in_size, gate_size, inter_size, sub_sample_factor)
        self.gate2 = GridAttention2D(in_size, gate_size, inter_size, sub_sample_factor)
        self.combine = nn.Sequential(
            nn.Conv2d(in_size * 2, in_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True),
        )
        # cache for CA-Net Fig 5/10-style visualization (filled in forward)
        self._last_attn = None

    def forward(self, x, g):
        g1, a1 = self.gate1(x, g)
        g2, a2 = self.gate2(x, g)
        attn = torch.cat([a1, a2], dim=1)   # (B, 2, H, W) — dual-path α maps
        self._last_attn = attn.detach()
        return self.combine(torch.cat([g1, g2], dim=1)), attn


# ══════════════════════════════════════════════════════════════════════
#  Non-local block (embedded_gaussian, 2D only)
# ══════════════════════════════════════════════════════════════════════

class NonLocal2D(nn.Module):
    """Non-local block (Wang et al. 2018), embedded Gaussian variant."""
    def __init__(self, in_channels, inter_channels=None, sub_sample=False):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(1, in_channels // 2)
        self.in_channels    = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.phi   = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.W = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias,   0)

        if sub_sample:
            self.g   = nn.Sequential(self.g,   nn.MaxPool2d(2))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(2))
        # cache for CA-Net Fig 12-style visualization (||W_y||₂ ≈ "where NL added info")
        self._last_delta_norm = None

    def forward(self, x):
        B = x.size(0)
        g_x     = self.g(x).view(B, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(B, self.inter_channels, -1).permute(0, 2, 1)
        phi_x   = self.phi(x).view(B, self.inter_channels, -1)
        f       = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y       = torch.matmul(f_div_C, g_x).permute(0, 2, 1).contiguous()
        y       = y.view(B, self.inter_channels, *x.shape[2:])
        W_y     = self.W(y)
        self._last_delta_norm = W_y.detach().norm(dim=1)   # (B, H, W)
        return W_y + x


# ══════════════════════════════════════════════════════════════════════
#  Scale attention (CBAM-style channel+spatial, 4×4 dsv structure)
# ══════════════════════════════════════════════════════════════════════

class BasicConv(nn.Module):
    def __init__(self, ci, co, kernel_size, stride=1, padding=0, relu=True, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(co, eps=1e-5, momentum=0.01) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn   is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    """
    Channel attention operates on a reshape of 16 channels = 4 dsv x 4 chan.
    Keeps the original (4, 4) reshape structure — this is CA-Net's "scale x channel"
    dual-axis attention design.
    """
    def __init__(self, gate_channels=16, reduction_ratio=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, max(1, gate_channels // reduction_ratio)),
            nn.ReLU(),
            nn.Linear(max(1, gate_channels // reduction_ratio), gate_channels),
        )
        # cache for CA-Net Fig 8/12-style γ visualization
        self._last_gamma = None    # (B, 4) per-scale weights

    def forward(self, x):
        # global avg + max pool -> MLP -> reshape (4,4) -> mean over channel dim -> expand
        avg = F.adaptive_avg_pool2d(x, (1, 1))
        mx  = F.adaptive_max_pool2d(x, (1, 1))
        att = self.mlp(avg) + self.mlp(mx)
        B   = x.size(0)
        # (B, 16) → (B, 4, 4) [scale, channel] → avg over channel → (B, 4) → expand → (B, 16)
        att = att.reshape(B, 4, 4)
        gamma_per_scale = torch.sigmoid(att.mean(dim=2))   # (B, 4) — CA-Net γ
        self._last_gamma = gamma_per_scale.detach()
        avg_weight = att.mean(dim=2, keepdim=True).expand(B, 4, 4).reshape(B, 16)
        scale      = torch.sigmoid(avg_weight).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale, scale


class SpatialAtten(nn.Module):
    """Spatial attention, in_size=16 -> out_size=4 (reduction)."""
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = BasicConv(in_size, out_size, 3, padding=1, relu=True)
        self.conv2 = BasicConv(out_size, out_size, 1, padding=0, relu=True, bn=False)

    def forward(self, x):
        residual = x
        z = self.conv2(self.conv1(x))                  # (B, out_size=4, H, W)
        # Expand 4 channel attention along the dsv axis to 16 (4 dsv x 4 chan)
        att = torch.sigmoid(z).unsqueeze(2)             # (B, 4, 1, H, W)
        att = att.expand(att.size(0), 4, 4, att.size(3), att.size(4))
        att = att.reshape(att.size(0), 16, att.size(3), att.size(4))
        x_out = residual * att + residual
        return x_out, att


class ScaleAttenConvBlock(nn.Module):
    """Merge 4 dsv outputs (16 channels total) -> channel-gate -> spatial-atten -> 3x3 conv -> out_size."""
    def __init__(self, in_size=16, out_size=4):
        super().__init__()
        self.channel = ChannelGate(in_size, reduction_ratio=4)
        self.spatial = SpatialAtten(in_size, max(1, in_size // 4))
        self.conv3   = _conv3x3(in_size, out_size)
        self.bn3     = nn.BatchNorm2d(out_size)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out, _ = self.channel(x)
        out, _ = self.spatial(out)
        out += residual
        out = self.relu(out)
        out = self.relu(self.bn3(self.conv3(out)))
        return out


# ══════════════════════════════════════════════════════════════════════
#  Main: Comprehensive Attention U-Net (regression variant)
# ══════════════════════════════════════════════════════════════════════

class ComprehensiveAttnUNet(nn.Module):
    """
    CA-Net 5-level UNet + (grid attention + SE + non-local + scale attention).

    Args:
        in_channels:   input channels (e.g. 2 for tas+pr; 8 for full set)
        out_channels:  output channels (1 for d18Op regression)
        feature_scale: filters = [64, 128, 256, 512, 1024] / feature_scale
                       default 4 -> [16, 32, 64, 128, 256]
        H, W:          original input size (90, 180). Internally padded to (96, 192) before 5 pools.
        is_deconv:     whether UpCat uses ConvTranspose or bilinear upsample
        dsv_channels:  number of output channels per deep-supervision branch
                       (default 4, matches the original structure)
    """
    def __init__(self, in_channels=2, out_channels=1, feature_scale=4,
                 H=90, W=180, is_deconv=True, dsv_channels=4):
        super().__init__()
        self.H_in = H
        self.W_in = W
        # Internally pad to a size divisible by 32 (5 pools)
        self.H_padded = ((H + 31) // 32) * 32
        self.W_padded = ((W + 31) // 32) * 32
        self.pad_top    = (self.H_padded - H) // 2
        self.pad_bottom = self.H_padded - H - self.pad_top
        self.pad_left   = (self.W_padded - W) // 2
        self.pad_right  = self.W_padded - W - self.pad_left

        filters = [int(x / feature_scale) for x in [64, 128, 256, 512, 1024]]
        assert all(f > 0 for f in filters), f'feature_scale={feature_scale} too large, filters={filters}'

        # Encoder
        self.conv1 = ConvBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(filters[2], filters[3], drop_out=True)
        self.pool4 = nn.MaxPool2d(2)
        self.center = ConvBlock(filters[3], filters[4], drop_out=True)

        # Attention blocks (decoder skip-gates)
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2],
                                                   inter_size=filters[1])
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3],
                                                   inter_size=filters[2])
        self.nonlocal4_2 = NonLocal2D(in_channels=filters[4], inter_channels=filters[4] // 4)

        # Decoder
        self.up_concat4 = UpCat(filters[4], filters[3], is_deconv)
        self.up_concat3 = UpCat(filters[3], filters[2], is_deconv)
        self.up_concat2 = UpCat(filters[2], filters[1], is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], is_deconv)
        self.up4 = SEConvBlock(filters[4], filters[3], drop_out=True)
        self.up3 = SEConvBlock(filters[3], filters[2])
        self.up2 = SEConvBlock(filters[2], filters[1])
        self.up1 = SEConvBlock(filters[1], filters[0])

        # Deep supervision (each branch emits dsv_channels channels, upsampled to padded size)
        dsv_scale = (self.H_padded, self.W_padded)
        self.dsv4 = UnetDsv(filters[3], dsv_channels, dsv_scale)
        self.dsv3 = UnetDsv(filters[2], dsv_channels, dsv_scale)
        self.dsv2 = UnetDsv(filters[1], dsv_channels, dsv_scale)
        self.dsv1 = nn.Conv2d(filters[0], dsv_channels, kernel_size=1)

        # Scale attention (merges 4 dsv branches of 4*dsv_channels=16 inputs -> dsv_channels=4)
        n_dsv_total = 4 * dsv_channels
        self.scale_att = ScaleAttenConvBlock(in_size=n_dsv_total, out_size=dsv_channels)

        # Final regression head: dsv_channels → out_channels (NO softmax, regression)
        self.final = nn.Conv2d(dsv_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Pad → (H_padded, W_padded)
        x = F.pad(x, (self.pad_left, self.pad_right,
                      self.pad_top,  self.pad_bottom),
                  mode='constant', value=0)

        # Encoder
        c1 = self.conv1(x);     p1 = self.pool1(c1)
        c2 = self.conv2(p1);    p2 = self.pool2(c2)
        c3 = self.conv3(p2);    p3 = self.pool3(c3)
        c4 = self.conv4(p3);    p4 = self.pool4(c4)
        center = self.center(p4)

        # Decoder level 4
        u4 = self.up_concat4(c4, center)          # cat (c4, up(center))
        g4 = self.nonlocal4_2(u4)                  # non-local refine
        u4, _ = self.up4(g4)                       # SE conv block

        # Decoder level 3 (with grid attention on c3)
        gc3, _ = self.attentionblock3(c3, u4)
        u3 = self.up_concat3(gc3, u4)
        u3, _ = self.up3(u3)

        # Decoder level 2 (with grid attention on c2)
        gc2, _ = self.attentionblock2(c2, u3)
        u2 = self.up_concat2(gc2, u3)
        u2, _ = self.up2(u2)

        # Decoder level 1 (no attention, same as the original)
        u1 = self.up_concat1(c1, u2)
        u1, _ = self.up1(u1)

        # Deep supervision
        d4 = self.dsv4(u4)
        d3 = self.dsv3(u3)
        d2 = self.dsv2(u2)
        d1 = self.dsv1(u1)
        d_cat = torch.cat([d1, d2, d3, d4], dim=1)

        out = self.scale_att(d_cat)
        out = self.final(out)

        # Crop back to (H, W)
        out = out[:, :,
                  self.pad_top:self.pad_top + self.H_in,
                  self.pad_left:self.pad_left + self.W_in]
        return out

    @torch.no_grad()
    def forward_with_attn(self, x):
        """
        Run forward + return cached attention maps. Used for CA-Net paper-style
        interpretability visualizations:
            * 'SA2': (B, 2, H, W)  — attentionblock2 dual-path alpha maps (scale H/2)
            * 'SA3': (B, 2, H, W)  — attentionblock3 dual-path alpha maps (scale H/4)
            * 'NL_delta': (B, H, W) — L2 norm of the bottleneck non-local update
            * 'scale_gamma': (B, 4) — gamma weights of the 4 scales (CA-Net Fig 8/12)
        All spatial dims are cropped back to the original (H_in, W_in); scale_gamma is a scalar vector.
        """
        out = self.forward(x)
        attn = {
            'SA2':         self.attentionblock2._last_attn,
            'SA3':         self.attentionblock3._last_attn,
            'NL_delta':    self.nonlocal4_2._last_delta_norm,
            'scale_gamma': self.scale_att.channel._last_gamma,
        }
        # Crop spatial maps back to original (H_in, W_in) and resize the smaller-scale SA*/NL to (H_in, W_in)
        cropped = {'scale_gamma': attn['scale_gamma']}
        for k in ('SA2', 'SA3', 'NL_delta'):
            v = attn[k]
            if v is None:
                cropped[k] = None; continue
            if v.dim() == 3:                 # (B, H, W) → unsqueeze to (B, 1, H, W) for interpolate
                v = v.unsqueeze(1)
            v = F.interpolate(v, size=(self.H_padded, self.W_padded),
                              mode='bilinear', align_corners=False)
            v = v[:, :, self.pad_top:self.pad_top + self.H_in,
                       self.pad_left:self.pad_left + self.W_in]
            cropped[k] = v
        return out, cropped


# ══════════════════════════════════════════════════════════════════════
#  Lightning wrapper (same interface as UNet2DBaseline)
# ══════════════════════════════════════════════════════════════════════

class CANetBaseline(L.LightningModule):
    """
    CA-Net single-frame regression baseline.
        forward(x):  (B, C, H, W) -> (B, 1, H, W)
        batch: (x, y) or (x, y, co2),  co2 is ignored
        loss: cos(lat) weighted MSE
    """
    def __init__(
        self,
        n_inputs:      int   = 2,
        out_channels:  int   = 1,
        feature_scale: int   = 4,
        H:             int   = 90,
        W:             int   = 180,
        is_deconv:     bool  = True,
        dsv_channels:  int   = 4,
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

        self.net = ComprehensiveAttnUNet(
            in_channels=n_inputs, out_channels=out_channels,
            feature_scale=feature_scale,
            H=H, W=W, is_deconv=is_deconv, dsv_channels=dsv_channels,
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def forward_with_attn(self, x):
        return self.net.forward_with_attn(x)

    def _unpack(self, batch):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
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
    B, C, H, W = 2, 2, 90, 180
    m = CANetBaseline(n_inputs=C, feature_scale=4, H=H, W=W)

    x = torch.randn(B, C, H, W)
    y = torch.randn(B, 1, H, W)
    y_hat = m(x)
    print(f'Input  x    : {tuple(x.shape)}')
    print(f'Output y_hat: {tuple(y_hat.shape)}   ← (B, 1, H, W) ✓')
    print(f'Params      : {sum(p.numel() for p in m.parameters()):,}')

    loss = m._loss(y_hat, y)
    print(f'MSE loss    : {loss.item():.4f}')

    # Backward sanity
    loss = m.training_step((x, y), 0)
    loss.backward()
    print(f'train_loss  : {loss.item():.4f}, backward OK')

    # (x,y,co2) compat
    loss2 = m.training_step((x, y, torch.tensor([280., 420.])), 0)
    print(f'(x,y,co2) train_loss compat: {loss2.item():.4f}')
