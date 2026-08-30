"""
baseline_iso_unet.py
─────────────────────────────────────────────────────────────────────────────
Regional Attention UNet — temporal context UNet for d18Op regression.

Design goals (contrast with baseline_canet.py / baseline_unet2d.py):
    1. Input is a 12-frame sequence, only the last frame is predicted
       (the first 11 frames produce climate_state and do not directly enter the UNet)
    2. Attention uses MoE-style routing (land vs ocean experts), not a sigmoid-mask
       (CA-Net-style masks suppress regions → introduces bias for regression)
    3. Climate state modulates features at the bottleneck via FiLM (γ·x + β)
       (not gating, so no information is lost)
    4. The first encoder block uses a grouped conv (groups=C_in): each input feature does
       spatial conv independently → a 1×1 mixer performs controlled fusion
       (avoids mixing features with different physical units at the very first layer, as
       would happen with RGB-style conv)

Input convention:
    * x: (B, T, C, H, W)  — INPUT_SET 'full' defaults to C=8:
      [tas, pr, PS, TMQ, QFLX, FLUT, LANDFRAC, aice]
    * landfrac_idx: index of the LANDFRAC channel in the C dimension (default 6)
    * T defaults to 12, but any T works

I/O:
    forward(x):  (B, T, C, H, W) → (B, 1, H, W)  ← d18Op for the last frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

# Prompt alignment loss — organizes climate_state by log(CO2/280) distance.
def co2_prompt_alignment_loss(
    climate_states: torch.Tensor,   # (B, d_model)
    co2_ppm:        torch.Tensor,   # (B,)  CO2 concentration in ppm
    tau:            float = 0.5,    # temperature: smaller -> sharper similarity decay
) -> torch.Tensor:
    """CO2 prompt alignment loss.

    Goal: samples with closer CO2 concentrations should have higher
    cosine similarity between their climate_state embeddings.

    Target similarity matrix:
        target_sim[i,j] = exp(-|log(co2_i/280) - log(co2_j/280)| / tau)

    Physical motivation: radiative forcing dF ~ log(CO2/CO2_0), so a
    log-scale distance is more physically meaningful than raw ppm diff.

    Loss = MSE(actual cosine similarity, target similarity), excluding diagonal.
    """
    B = climate_states.shape[0]
    states     = F.normalize(climate_states, dim=-1)   # (B, d_model)
    sim_matrix = states @ states.T                     # (B, B)
    co2_log    = torch.log(co2_ppm / 280.0)
    co2_dist   = (co2_log.unsqueeze(0) - co2_log.unsqueeze(1)).abs()  # (B, B)
    target_sim = torch.exp(-co2_dist / tau)            # (B, B) in [0, 1]
    mask       = ~torch.eye(B, dtype=torch.bool, device=climate_states.device)
    return F.mse_loss(sim_matrix[mask], target_sim[mask])


# ── Save figure to BOTH .pdf and .png (one helper for all draw_inner methods) ──
def _save_fig_both(path, **kwargs):
    """Save the current figure to path (.pdf) and also generate matching .png.
    Use instead of plt.savefig(path, ...). PNG output uses same DPI as PDF."""
    import matplotlib.pyplot as plt
    plt.savefig(path, **kwargs)
    if path.lower().endswith('.pdf'):
        plt.savefig(path[:-4] + '.png', **kwargs)


# ── Input-mask for 4-way MoE / attention ──────────────────────────────
# Controlled by the single `mask_mode` parameter. When not 'none', the 4-way
# (LW/LD/OW/OD) and ice experts are all masked (LW/LD←land, OW/OD←ocean, ice←aice).
#
# mask_mode:
#   'none'       : no masking (existing behavior, default for backward compatibility)
#   'hard'       : multiplier = (m > 0.5).float()  → 0/1 binary, abrupt at boundaries (info loss)
#   'soft'       : multiplier = m                  → continuous ∈ [0,1], smooth at coastlines / ice edges
#   'soft_boost' : multiplier = 1 + γ · m          → no info loss, boost inside region (γ default 0.5)
#
# wet/dry is never fed an input mask; it is distinguished only at the output via routing
# weights (w_LW = lf·pr).
MASK_MODES = ('none', 'hard', 'soft', 'soft_boost')


def _apply_mask_kind(mask_mode, m, gamma=0.5):
    """
    Convert continuous m ∈ [0,1] into a multiplier (same shape as feat, used to multiply feat).
    m: e.g. lf (land mask), 1-lf (ocean), aice (ice region).
    gamma: only used by 'soft_boost' — boost strength.
    """
    if mask_mode == 'hard':
        return (m > 0.5).float()
    if mask_mode == 'soft':
        return m
    if mask_mode == 'soft_boost':
        return 1.0 + gamma * m
    raise ValueError(f"_apply_mask_kind: mask_mode='{mask_mode}' should not call this "
                     f"(only 'hard'/'soft'/'soft_boost' need multiplier). MASK_MODES={MASK_MODES}")


def _compute_input_masks(mask_mode, lf, gamma=0.5):
    """
    Returns (m_LW, m_LD, m_OW, m_OD) — each (B, 1, H, W) multipliers to apply to feat.
    Returns None if mask_mode='none' (caller skips masking).
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(f"Unknown mask_mode '{mask_mode}'. Choices: {MASK_MODES}")
    if mask_mode == 'none':
        return None
    land_mult  = _apply_mask_kind(mask_mode, lf,        gamma)
    ocean_mult = _apply_mask_kind(mask_mode, 1.0 - lf,  gamma)
    return land_mult, land_mult, ocean_mult, ocean_mult


def _compute_ice_mask(mask_mode, aice, gamma=0.5):
    """Returns multiplier (B, 1, H, W) for ice expert input, or None if no ice mask."""
    if mask_mode == 'none':
        return None
    return _apply_mask_kind(mask_mode, aice, gamma)


# ══════════════════════════════════════════════════════════════════════
#  Basic blocks
# ══════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """2x (Conv3x3 + BN + ReLU). Same as baseline_canet.ConvBlock."""
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


class UpConcat(nn.Module):
    """Decoder upsample + concat with skip (uses bilinear to avoid checkerboard)."""
    def __init__(self, in_feat, skip_feat, out_feat):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_feat + skip_feat, out_feat)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ══════════════════════════════════════════════════════════════════════
#  Goal 4: Disentangled stem — per-feature spatial conv with no early mixing
# ══════════════════════════════════════════════════════════════════════

class DisentangledStem(nn.Module):
    """
    First block: each input channel runs **independently** through 2 layers of 3×3 spatial
    conv, and a final 1×1 conv performs "controlled fusion" → base_channels.

    Differences from the original UNet2D:
        baseline_unet2d.py first layer is Conv2d(C_in=2, base=16, k=3) ← it mixes all input
            channels onto every output channel at the very first step (only reasonable when
            channels are semantically similar, e.g. RGB).
        Here: each input feature has its own k spatial filters, with no knowledge of the
            others; the 1×1 mixer then collapses (C·k) into base_channels — controlling
            when mixing happens.

    Args:
        n_inputs:   number of input channels (e.g. 8 for full)
        k:          internal width per input feature (default 4)
        base:       output channels (UNet level-1 width)
    """
    def __init__(self, n_inputs: int, k: int = 4, base: int = 32):
        super().__init__()
        self.n_inputs = n_inputs
        self.k = k
        inter = n_inputs * k
        self.depthwise = nn.Sequential(
            nn.Conv2d(n_inputs, inter, kernel_size=3, padding=1,
                      groups=n_inputs, bias=False),
            nn.BatchNorm2d(inter), nn.ReLU(inplace=True),
            nn.Conv2d(inter, inter, kernel_size=3, padding=1,
                      groups=n_inputs, bias=False),
            nn.BatchNorm2d(inter), nn.ReLU(inplace=True),
        )
        self.mixer = nn.Sequential(
            nn.Conv2d(inter, base, kernel_size=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, n_inputs, H, W)
        x = self.depthwise(x)   # (B, n_inputs*k, H, W) — each group of k corresponds to one input feature
        x = self.mixer(x)       # (B, base, H, W) — controlled fusion
        return x


# ══════════════════════════════════════════════════════════════════════
#  Goal: Climate state encoder — lightweight context summary
# ══════════════════════════════════════════════════════════════════════

TIME_POOLS = ('mean', 'gru', 'conv3d')


class ClimateStateEncoder(nn.Module):
    """
    Extracts a d_state-dim vector from T-1 context frames.

    time_pool:
        'mean'   : time-average then 2D CNN → GAP → MLP    (default, backward compat)
        'gru'    : per-frame 2D CNN → GRU over time → MLP  (sequence-aware, order matters)
        'conv3d' : 3D conv jointly over (T, H, W) → GAP3D → MLP  (spatio-temporal joint)

    All modes output (B, d_state) with the same shape; switching modes does not affect
    downstream consumers (FiLM, prompt loss).
    """
    def __init__(self, n_inputs: int, d_state: int = 64,
                 inter_channels: int = 32, time_pool: str = 'mean'):
        super().__init__()
        if time_pool not in TIME_POOLS:
            raise ValueError(f"time_pool must be one of {TIME_POOLS}, got '{time_pool}'")
        self.time_pool      = time_pool
        self.d_state        = d_state
        self.inter_channels = inter_channels

        if time_pool == 'mean':
            # Original: 2D CNN on time-mean
            self.encoder = nn.Sequential(
                nn.Conv2d(n_inputs, inter_channels, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, inter_channels * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(inter_channels * 2), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.mlp = nn.Sequential(
                nn.Linear(inter_channels * 2, d_state),
                nn.ReLU(inplace=True),
                nn.Linear(d_state, d_state),
            )

        elif time_pool == 'gru':
            # Per-frame 2D CNN → embedding (B, T, d_frame), then GRU aggregates over time
            self.frame_encoder = nn.Sequential(
                nn.Conv2d(n_inputs, inter_channels, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, inter_channels * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(inter_channels * 2), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            # 1-layer GRU, hidden=d_state. Input = per-frame embed dim
            self.gru = nn.GRU(inter_channels * 2, d_state,
                              num_layers=1, batch_first=True)
            self.mlp = nn.Linear(d_state, d_state)

        elif time_pool == 'conv3d':
            # 3D conv joint over (T, H, W)
            self.encoder = nn.Sequential(
                nn.Conv3d(n_inputs, inter_channels,
                          kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
                nn.BatchNorm3d(inter_channels), nn.ReLU(inplace=True),
                nn.Conv3d(inter_channels, inter_channels * 2,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(inter_channels * 2), nn.ReLU(inplace=True),
                nn.Conv3d(inter_channels * 2, inter_channels * 4,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(inter_channels * 4), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
            )
            self.mlp = nn.Sequential(
                nn.Linear(inter_channels * 4, d_state),
                nn.ReLU(inplace=True),
                nn.Linear(d_state, d_state),
            )

    def forward(self, x_context):
        """x_context: (B, T_context, C, H, W) → (B, d_state)"""
        if self.time_pool == 'mean':
            # time average → 2D CNN
            x_avg = x_context.mean(dim=1)
            return self.mlp(self.encoder(x_avg))

        if self.time_pool == 'gru':
            B, T, C, H, W = x_context.shape
            # encode each frame independently
            x = x_context.reshape(B * T, C, H, W)
            x = self.frame_encoder(x)             # (B*T, inter*2)
            x = x.reshape(B, T, -1)                # (B, T, d_frame)
            _, h = self.gru(x)                     # (1, B, d_state)
            return self.mlp(h.squeeze(0))          # (B, d_state)

        if self.time_pool == 'conv3d':
            # (B, T, C, H, W) → (B, C, T, H, W) for Conv3d
            x = x_context.permute(0, 2, 1, 3, 4)
            return self.mlp(self.encoder(x))


# ══════════════════════════════════════════════════════════════════════
#  Goal 2: Regional MoE attention — two-expert routing (land/ocean)
#          (no sigmoid suppression, no information loss)
# ══════════════════════════════════════════════════════════════════════

class RegionalMoE(nn.Module):
    """
    Mixture-of-Experts at bottleneck.

    Land/Ocean routing (default, always on):
        b_land   = LandExpert(b);  b_ocean = OceanExpert(b)
        routed_lo = landfrac · b_land + (1 - landfrac) · b_ocean

    Wet/Dry routing (additional pair when use_precip=True):
        b_wet    = WetExpert(b);   b_dry   = DryExpert(b)
        routed_wd = pr_norm · b_wet + (1 - pr_norm) · b_dry
        # pr_norm: pr mask min-max normalized to [0,1] per-sample per-timestep

    Final: routed = (routed_lo + routed_wd) / 2  if both else routed_lo
           out    = ReLU(routed + b)               # residual

    Difference from CA-Net attention (no sigmoid suppression): both experts process the
    full image and are soft-blended by the mask → no information is suppressed.
    """
    def __init__(self, channels: int, kernel_size: int = 3,
                 use_precip: bool = False):
        super().__init__()
        pad = (kernel_size - 1) // 2
        def expert():
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=kernel_size,
                          padding=pad, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=kernel_size,
                          padding=pad, bias=False),
                nn.BatchNorm2d(channels),
            )
        self.expert_land  = expert()
        self.expert_ocean = expert()
        self.use_precip   = use_precip
        if use_precip:
            self.expert_wet = expert()
            self.expert_dry = expert()
        self.relu = nn.ReLU(inplace=True)
        # cache for viz
        self._last_b_land     = None
        self._last_b_ocean    = None
        self._last_landfrac_b = None
        self._last_b_wet      = None
        self._last_b_dry      = None
        self._last_pr_b       = None

    def forward(self, b, landfrac_b, pr_b=None):
        """
        b:          (B, C, H_b, W_b)
        landfrac_b: (B, 1, H_b, W_b)   ∈ [0, 1]
        pr_b:       (B, 1, H_b, W_b)   ∈ [0, 1]  per-sample-normalized — optional
        """
        b_land  = self.expert_land(b)
        b_ocean = self.expert_ocean(b)
        routed_lo = landfrac_b * b_land + (1 - landfrac_b) * b_ocean

        if self.use_precip and pr_b is not None:
            b_wet  = self.expert_wet(b)
            b_dry  = self.expert_dry(b)
            routed_wd = pr_b * b_wet + (1 - pr_b) * b_dry
            routed = 0.5 * (routed_lo + routed_wd)
            self._last_b_wet = b_wet.detach()
            self._last_b_dry = b_dry.detach()
            self._last_pr_b  = pr_b.detach()
        else:
            routed = routed_lo
            self._last_b_wet = None
            self._last_b_dry = None
            self._last_pr_b  = None

        out = self.relu(routed + b)             # residual
        self._last_b_land     = b_land.detach()
        self._last_b_ocean    = b_ocean.detach()
        self._last_landfrac_b = landfrac_b.detach()
        return out


# ══════════════════════════════════════════════════════════════════════
#  Skip-connection Region Expert Attention (attention on low-resolution encoder/skip features)
#    * Two experts: land and ocean
#    * Each expert outputs a spatial attention map (B, 1, H, W) ∈ [0, 1]
#    * Soft-routed by landfrac@local + climate_state modulates the offset
#    * Residual form (scale init=0 → identity at the start of training, does not damage features)
#
#  Visualization: after running, the per-level attn_land / attn_ocean distributions are
#  available, answering the question "where does the model look when predicting over land /
#  over ocean".
# ══════════════════════════════════════════════════════════════════════

class RegionExpertAttention(nn.Module):
    """
    Region-conditional spatial attention with 2 routing axes:
        * Land/Ocean (LANDFRAC mask)        ← always on
        * Wet/Dry    (pr mask, [0,1])       ← additional when use_precip=True

    Each pair of experts produces a sigmoid attention map (B, 1, H, W) ∈ [0,1], which is
    then soft-blended by the corresponding mask; the results from the two routings are
    averaged (when both are enabled).

    Output: feat + scale · merged_attn · feat   (residual, scale init=0 → identity)
    """
    def __init__(self, channels, d_state=64, expert_hidden=None,
                 use_precip: bool = False):
        super().__init__()
        if expert_hidden is None:
            expert_hidden = max(8, channels // 4)
        def _expert():
            return nn.Sequential(
                nn.Conv2d(channels, expert_hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(expert_hidden), nn.ReLU(inplace=True),
                nn.Conv2d(expert_hidden, 1, kernel_size=1),
            )
        # Land/Ocean experts (default)
        self.expert_land  = _expert()
        self.expert_ocean = _expert()
        # Wet/Dry experts (optional)
        self.use_precip = use_precip
        if use_precip:
            self.expert_wet = _expert()
            self.expert_dry = _expert()
            # offsets output 4 values (land/ocean + wet/dry)
            self.state_mlp = nn.Linear(d_state, 4)
        else:
            self.state_mlp = nn.Linear(d_state, 2)
        nn.init.zeros_(self.state_mlp.weight)
        nn.init.zeros_(self.state_mlp.bias)

        # Residual scale (init 0 → identity)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Viz caches
        self._last_attn_land      = None
        self._last_attn_ocean     = None
        self._last_attn_wet       = None
        self._last_attn_dry       = None
        self._last_landfrac_local = None
        self._last_pr_local       = None
        self._last_merged_attn    = None

    def forward(self, feat, landfrac, climate_state, pr=None):
        """
        feat:          (B, C, H, W)
        landfrac:      (B, 1, H_in, W_in)  auto-resized to (H, W)
        climate_state: (B, d_state)
        pr:            (B, 1, H_in, W_in)  ∈ [0,1] per-sample-normalized — optional
        """
        B, C, H, W = feat.shape

        # 1. Resize landfrac
        if landfrac.shape[-2:] != (H, W):
            lf_local = F.interpolate(landfrac, size=(H, W),
                                     mode='bilinear', align_corners=False)
        else:
            lf_local = landfrac
        lf_local = lf_local.clamp(0.0, 1.0)

        # 2. Land/Ocean experts
        offsets = self.state_mlp(climate_state)   # (B, 2 or 4)
        a_l_logit = self.expert_land(feat)
        a_o_logit = self.expert_ocean(feat)
        attn_land  = torch.sigmoid(a_l_logit + offsets[:, 0].view(B,1,1,1))
        attn_ocean = torch.sigmoid(a_o_logit + offsets[:, 1].view(B,1,1,1))
        merged_lo  = lf_local * attn_land + (1 - lf_local) * attn_ocean

        # 3. Wet/Dry experts (optional)
        if self.use_precip and pr is not None:
            if pr.shape[-2:] != (H, W):
                pr_local = F.interpolate(pr, size=(H, W),
                                         mode='bilinear', align_corners=False)
            else:
                pr_local = pr
            pr_local = pr_local.clamp(0.0, 1.0)
            a_w_logit = self.expert_wet(feat)
            a_d_logit = self.expert_dry(feat)
            attn_wet = torch.sigmoid(a_w_logit + offsets[:, 2].view(B,1,1,1))
            attn_dry = torch.sigmoid(a_d_logit + offsets[:, 3].view(B,1,1,1))
            merged_wd = pr_local * attn_wet + (1 - pr_local) * attn_dry
            merged_attn = 0.5 * (merged_lo + merged_wd)
            self._last_attn_wet  = attn_wet.detach()
            self._last_attn_dry  = attn_dry.detach()
            self._last_pr_local  = pr_local.detach()
        else:
            merged_attn = merged_lo
            self._last_attn_wet  = None
            self._last_attn_dry  = None
            self._last_pr_local  = None

        # Caches for viz
        self._last_attn_land      = attn_land.detach()
        self._last_attn_ocean     = attn_ocean.detach()
        self._last_landfrac_local = lf_local.detach()
        self._last_merged_attn    = merged_attn.detach()

        # Residual modulation
        return feat + self.scale * merged_attn * feat


# ══════════════════════════════════════════════════════════════════════
#  Regional 4-Way Product MoE — (land/ocean) × (wet/dry) = 4 experts
#  Key differences from RegionalMoE:
#      RegionalMoE      : 2 experts or 2+2=4 experts (two independent additive paths)
#      Regional4WayMoE  : 4 experts, routing via outer product — 4 regimes fully decoupled
#  Use case: when the model should learn the four regimes "land-wet / land-dry /
#  ocean-wet / ocean-dry" separately.
# ══════════════════════════════════════════════════════════════════════

class Regional4WayMoE(nn.Module):
    """
    Product-routing MoE at bottleneck. 4 experts:
        E_LW (land+wet), E_LD (land+dry), E_OW (ocean+wet), E_OD (ocean+dry)

    Routing weights (sum to 1: (lf + (1-lf)) * (pr + (1-pr)) = 1):
        w_LW = lf · pr
        w_LD = lf · (1 - pr)
        w_OW = (1 - lf) · pr
        w_OD = (1 - lf) · (1 - pr)

    Output: ReLU(Σ w_i · E_i(b) + b)         # residual, identity-friendly

    ── Optional ice override (use_ice=True) ────────────────────────────
    Adds one extra expert E_ice; routing is an *additive override* (not part of the outer product):
        routed_4way = Σ w_i · E_i(b)             # 4-expert part computed as usual
        b_ice       = E_ice(b)
        merged      = (1 - aice_b) · routed_4way + aice_b · b_ice
        out         = ReLU(merged + b)
    Motivation: ice pixels are a small fraction and concentrated near the poles; an override
    is more reasonable than splitting weight uniformly via a partition.
    """
    def __init__(self, channels: int, kernel_size: int = 3,
                 use_ice: bool = False,
                 mask_mode: str = 'none',
                 mask_gamma: float = 0.5):
        super().__init__()
        if mask_mode not in MASK_MODES:
            raise ValueError(f"Regional4WayMoE: bad mask_mode '{mask_mode}', "
                             f"choices: {MASK_MODES}")
        self.mask_mode  = mask_mode
        self.mask_gamma = float(mask_gamma)
        pad = (kernel_size - 1) // 2
        def expert():
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=kernel_size,
                          padding=pad, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=kernel_size,
                          padding=pad, bias=False),
                nn.BatchNorm2d(channels),
            )
        self.expert_LW = expert()
        self.expert_LD = expert()
        self.expert_OW = expert()
        self.expert_OD = expert()
        self.use_ice = use_ice
        if use_ice:
            self.expert_ice = expert()
        self.relu = nn.ReLU(inplace=True)
        # viz caches
        self._last_b_LW = None
        self._last_b_LD = None
        self._last_b_OW = None
        self._last_b_OD = None
        self._last_b_ice      = None
        self._last_landfrac_b = None
        self._last_pr_b       = None
        self._last_aice_b     = None

    def forward(self, b, landfrac_b, pr_b, aice_b=None):
        """
        b:          (B, C, H_b, W_b)
        landfrac_b: (B, 1, H_b, W_b)  ∈ [0,1]
        pr_b:       (B, 1, H_b, W_b)  ∈ [0,1]  per-sample normalized
        aice_b:     (B, 1, H_b, W_b)  ∈ [0,1]  sea ice fraction — only used when use_ice=True
        """
        assert pr_b is not None, 'Regional4WayMoE requires pr_b'
        # Optional input mask for 4-way experts (mode: 'hard' / 'soft' / 'soft_boost', 'none' = skip)
        masks = _compute_input_masks(self.mask_mode, landfrac_b, gamma=self.mask_gamma)
        if masks is None:
            f_LW = f_LD = f_OW = f_OD = b
        else:
            m_LW, m_LD, m_OW, m_OD = masks
            f_LW = b * m_LW
            f_LD = b * m_LD
            f_OW = b * m_OW
            f_OD = b * m_OD
        b_LW = self.expert_LW(f_LW)
        b_LD = self.expert_LD(f_LD)
        b_OW = self.expert_OW(f_OW)
        b_OD = self.expert_OD(f_OD)

        w_LW = landfrac_b * pr_b
        w_LD = landfrac_b * (1.0 - pr_b)
        w_OW = (1.0 - landfrac_b) * pr_b
        w_OD = (1.0 - landfrac_b) * (1.0 - pr_b)

        routed_4way = w_LW * b_LW + w_LD * b_LD + w_OW * b_OW + w_OD * b_OD

        if self.use_ice:
            assert aice_b is not None, 'use_ice=True but aice_b is None'
            # ice expert input mask — uses the same mode + gamma as the 4-way mask
            ice_mult = _compute_ice_mask(self.mask_mode, aice_b, gamma=self.mask_gamma)
            f_ice = b if ice_mult is None else b * ice_mult
            b_ice  = self.expert_ice(f_ice)
            merged = (1.0 - aice_b) * routed_4way + aice_b * b_ice
            self._last_b_ice  = b_ice.detach()
            self._last_aice_b = aice_b.detach()
        else:
            merged = routed_4way
            self._last_b_ice  = None
            self._last_aice_b = None

        out = self.relu(merged + b)

        # caches
        self._last_b_LW       = b_LW.detach()
        self._last_b_LD       = b_LD.detach()
        self._last_b_OW       = b_OW.detach()
        self._last_b_OD       = b_OD.detach()
        self._last_landfrac_b = landfrac_b.detach()
        self._last_pr_b       = pr_b.detach()
        return out


class Region4WayExpertAttention(nn.Module):
    """
    Skip-connection 4-expert spatial attention with product routing.
    Mirrors the structure of RegionExpertAttention but uses 4 sigmoid attention maps with
    outer-product weights.

    Output: feat + scale · merged_attn · feat   (residual, scale init=0 → identity)

    ── Optional ice override (use_ice=True) ────────────────────────────
    Adds one ice attention map; merging uses an aice override:
        merged_4way = Σ w_i · α_i
        α_ice       = sigmoid(expert_ice(feat) + offset_ice)
        merged_attn = (1 - aice) · merged_4way + aice · α_ice
    """
    def __init__(self, channels, d_state=64, expert_hidden=None,
                 use_ice: bool = False,
                 mask_mode: str = 'none',
                 mask_gamma: float = 0.5):
        super().__init__()
        if mask_mode not in MASK_MODES:
            raise ValueError(f"Region4WayExpertAttention: bad mask_mode '{mask_mode}', "
                             f"choices: {MASK_MODES}")
        self.mask_mode  = mask_mode
        self.mask_gamma = float(mask_gamma)
        if expert_hidden is None:
            expert_hidden = max(8, channels // 4)
        def _expert():
            return nn.Sequential(
                nn.Conv2d(channels, expert_hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(expert_hidden), nn.ReLU(inplace=True),
                nn.Conv2d(expert_hidden, 1, kernel_size=1),
            )
        self.expert_LW = _expert()
        self.expert_LD = _expert()
        self.expert_OW = _expert()
        self.expert_OD = _expert()
        self.use_ice = use_ice
        if use_ice:
            self.expert_ice = _expert()
            # 5 climate-state offsets (4 + ice)
            self.state_mlp = nn.Linear(d_state, 5)
        else:
            self.state_mlp = nn.Linear(d_state, 4)
        nn.init.zeros_(self.state_mlp.weight)
        nn.init.zeros_(self.state_mlp.bias)
        # residual scale (init 0 → identity at init)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        # viz caches
        self._last_attn_LW         = None
        self._last_attn_LD         = None
        self._last_attn_OW         = None
        self._last_attn_OD         = None
        self._last_attn_ice        = None
        self._last_landfrac_local  = None
        self._last_pr_local        = None
        self._last_aice_local      = None
        self._last_merged_attn     = None

    def forward(self, feat, landfrac, climate_state, pr, aice=None):
        """
        feat:          (B, C, H, W)
        landfrac, pr:  (B, 1, *, *) auto-resized to (H, W)
        climate_state: (B, d_state)
        aice:          (B, 1, *, *) ∈ [0,1] — only used when use_ice=True
        """
        assert pr is not None, 'Region4WayExpertAttention requires pr mask'
        B, C, H, W = feat.shape

        # Resize masks to (H, W)
        if landfrac.shape[-2:] != (H, W):
            lf_local = F.interpolate(landfrac, size=(H, W),
                                     mode='bilinear', align_corners=False)
        else:
            lf_local = landfrac
        lf_local = lf_local.clamp(0.0, 1.0)
        if pr.shape[-2:] != (H, W):
            pr_local = F.interpolate(pr, size=(H, W),
                                     mode='bilinear', align_corners=False)
        else:
            pr_local = pr
        pr_local = pr_local.clamp(0.0, 1.0)

        # Optional input mask for 4-way experts (mode: 'hard' / 'soft' / 'soft_boost', 'none' = skip)
        masks = _compute_input_masks(self.mask_mode, lf_local, gamma=self.mask_gamma)
        if masks is None:
            f_LW = f_LD = f_OW = f_OD = feat
        else:
            m_LW, m_LD, m_OW, m_OD = masks
            f_LW = feat * m_LW
            f_LD = feat * m_LD
            f_OW = feat * m_OW
            f_OD = feat * m_OD

        # 4 expert logits + state offsets
        offsets = self.state_mlp(climate_state)            # (B, 4) or (B, 5)
        a_LW = torch.sigmoid(self.expert_LW(f_LW) + offsets[:, 0].view(B,1,1,1))
        a_LD = torch.sigmoid(self.expert_LD(f_LD) + offsets[:, 1].view(B,1,1,1))
        a_OW = torch.sigmoid(self.expert_OW(f_OW) + offsets[:, 2].view(B,1,1,1))
        a_OD = torch.sigmoid(self.expert_OD(f_OD) + offsets[:, 3].view(B,1,1,1))

        # Product routing
        w_LW = lf_local * pr_local
        w_LD = lf_local * (1.0 - pr_local)
        w_OW = (1.0 - lf_local) * pr_local
        w_OD = (1.0 - lf_local) * (1.0 - pr_local)
        merged_4way = w_LW * a_LW + w_LD * a_LD + w_OW * a_OW + w_OD * a_OD

        if self.use_ice:
            assert aice is not None, 'use_ice=True but aice is None'
            if aice.shape[-2:] != (H, W):
                aice_local = F.interpolate(aice, size=(H, W),
                                            mode='bilinear', align_corners=False)
            else:
                aice_local = aice
            aice_local = aice_local.clamp(0.0, 1.0)
            # ice expert input mask — uses the same mode + gamma as the 4-way mask
            ice_mult = _compute_ice_mask(self.mask_mode, aice_local, gamma=self.mask_gamma)
            f_ice = feat if ice_mult is None else feat * ice_mult
            a_ice = torch.sigmoid(self.expert_ice(f_ice) + offsets[:, 4].view(B,1,1,1))
            merged_attn = (1.0 - aice_local) * merged_4way + aice_local * a_ice
            self._last_attn_ice   = a_ice.detach()
            self._last_aice_local = aice_local.detach()
        else:
            merged_attn = merged_4way
            self._last_attn_ice   = None
            self._last_aice_local = None

        # caches
        self._last_attn_LW        = a_LW.detach()
        self._last_attn_LD        = a_LD.detach()
        self._last_attn_OW        = a_OW.detach()
        self._last_attn_OD        = a_OD.detach()
        self._last_landfrac_local = lf_local.detach()
        self._last_pr_local       = pr_local.detach()
        self._last_merged_attn    = merged_attn.detach()

        # Residual modulation
        return feat + self.scale * merged_attn * feat


# ══════════════════════════════════════════════════════════════════════
#  Goal 3: FiLM modulation — climate state modulates the bottleneck
# ══════════════════════════════════════════════════════════════════════

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (Perez et al. 2018):
        out = (1 + γ) · feature + β
    γ, β ∈ ℝ^C per sample, produced from climate_state via an MLP.

    Difference from attention gating:
        * Gating: sigmoid_mask · feature (∈ [0, ·]: always non-negative, may collapse to 0)
        * FiLM:   (1 + γ) · feature + β (γ ∈ ℝ: can be positive or negative, centered at 1+γ)
        → FiLM adjusts feature scale + offset without loss (regression-friendly).
    """
    def __init__(self, d_state: int, channels: int):
        super().__init__()
        self.gamma_mlp = nn.Linear(d_state, channels)
        self.beta_mlp  = nn.Linear(d_state, channels)
        # init: γ=0, β=0 → reduces to identity (does not damage features early in training)
        nn.init.zeros_(self.gamma_mlp.weight); nn.init.zeros_(self.gamma_mlp.bias)
        nn.init.zeros_(self.beta_mlp.weight);  nn.init.zeros_(self.beta_mlp.bias)
        # cache for viz
        self._last_gamma = None
        self._last_beta  = None

    def forward(self, feat, climate_state):
        # feat: (B, C, H, W);  climate_state: (B, d_state)
        gamma = self.gamma_mlp(climate_state).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta  = self.beta_mlp(climate_state).unsqueeze(-1).unsqueeze(-1)
        self._last_gamma = gamma.detach().squeeze(-1).squeeze(-1)
        self._last_beta  = beta.detach().squeeze(-1).squeeze(-1)
        return feat * (1.0 + gamma) + beta


# ══════════════════════════════════════════════════════════════════════
#  Main network
# ══════════════════════════════════════════════════════════════════════

class RegionalAttentionUNet(nn.Module):
    """
    Regional Attention UNet:
        forward(x):  (B, T, C, H, W) → (B, 1, H, W)
                     ↑ T-frame seq      ↑ last-frame d18Op

    Args:
        n_inputs:      C, number of input channels (default 8 for full INPUT_SET)
        landfrac_idx:  index of LANDFRAC in the C dimension (default 6, matches INPUT_SET['full'])
                       passing None disables region routing (falls back to standard UNet + FiLM)
        base_channels: UNet level-1 width (default 32)
        d_state:       climate_state dimension (default 64)
        H, W:          original spatial size (90, 180); internally padded to (96, 192) for 4 levels of pooling
    """
    def __init__(self, n_inputs: int = 8, landfrac_idx: int = 6,
                 base_channels: int = 64, d_state: int = 64,
                 H: int = 90, W: int = 180,
                 stem_k: int = 4,
                 # ── Ablation flags (all default to True = full model) ──
                 use_disentangled_stem: bool = True,
                 use_temporal_context:  bool = True,   # off = disable context+FiLM
                 use_bottleneck_moe:    bool = True,   # off = bottleneck uses plain conv
                 use_skip_attn:         bool = True,   # off = skips go directly into decoder (master switch)
                 skip_attn_levels:      tuple = (3, 4),  # which levels get skip attention
                 # ── Precip routing: add a wet/dry expert pair ──
                 use_precip_routing:    bool = False,
                 pr_channels:           tuple = None,
                 # ── MoE mode: switch for 4-way product routing ──
                 #     None              -> legacy behavior (depends on use_precip_routing)
                 #     'landocean'       -> same as use_precip_routing=False (2 experts)
                 #     'wetdry_parallel' -> same as use_precip_routing=True  (2+2 experts, additive)
                 #     'product4'        -> ★ new mode (4 experts, outer-product routing)
                 moe_mode:              str = None,
                 # ── Climate state temporal pooling: 'mean' (default) | 'gru' | 'conv3d' ──
                 time_pool:             str = 'mean',
                 # ── Ice routing override (additive, only effective when moe_mode='product4') ──
                 use_ice_routing:       bool = False,
                 ice_idx:               int  = None,
                 # ── Input-mask for 4-way + ice experts (only with moe_mode='product4') ──
                 # mask_mode: 'none' / 'hard' / 'soft' / 'soft_boost'
                 # mask_gamma: γ for 'soft_boost' (default 0.5); ignored for other modes
                 # When not 'none', all 4-way + ice are masked (LW/LD←land, OW/OD←ocean, ice←aice)
                 mask_mode:             str   = 'none',
                 mask_gamma:            float = 0.5,
                 # ── Input mode: ablation comparing against the climate-state setup ──
                 input_mode:            str = 'last_only',  # 'last_only' | 'concat_all'
                 T:                     int = 12):          # T must be known when using concat_all
        super().__init__()
        self.n_inputs     = n_inputs
        self.landfrac_idx = landfrac_idx
        # use_routing = "do we have the landfrac mask?"
        self.use_routing            = (landfrac_idx is not None)
        # Four component flags (the two region-aware ones also need use_routing=True)
        self._use_disentangled_stem = use_disentangled_stem
        self._use_temporal_context  = use_temporal_context
        self._use_bottleneck_moe    = use_bottleneck_moe and self.use_routing
        self._use_skip_attn         = use_skip_attn and self.use_routing
        # Skip-attention level set: only meaningful when _use_skip_attn=True
        self._skip_attn_levels = set(int(l) for l in skip_attn_levels) if self._use_skip_attn else set()
        assert all(l in (1, 2, 3, 4) for l in self._skip_attn_levels), \
            f'skip_attn_levels must only contain {1,2,3,4}, got {skip_attn_levels}'

        # Precip routing setup
        self._use_precip_routing = bool(use_precip_routing)
        if self._use_precip_routing:
            assert pr_channels is not None and len(pr_channels) > 0, \
                '--use_precip_routing=True but no pr_channels given (which input channels to sum as pr)'
            self.pr_channels = tuple(int(i) for i in pr_channels)
        else:
            self.pr_channels = None

        # ── Resolve moe_mode (None = legacy behavior, equivalent to use_precip_routing) ──
        # Legacy config (no moe_mode passed) → auto-derive; behavior fully equivalent to the pre-change version
        if moe_mode is None:
            moe_mode = 'wetdry_parallel' if self._use_precip_routing else 'landocean'
        assert moe_mode in ('landocean', 'wetdry_parallel', 'product4'), (
            f"moe_mode must be one of {{'landocean','wetdry_parallel','product4'}}, "
            f"got '{moe_mode}'"
        )
        # product4 mode needs a pr_mask (just like wetdry_parallel needs pr_channels)
        if moe_mode == 'product4':
            assert self._use_precip_routing, (
                "moe_mode='product4' requires use_precip_routing=True (needs pr_channels to compute the pr mask)"
            )
        # landocean mode overrides use_precip_routing → even if the user sets True, pr is not passed.
        # pr_channels is still computed (cheap), just unused; we don't raise to avoid noise.
        self._moe_mode = moe_mode

        # ── Ice routing override (additive, only valid under product4 mode) ──
        self._use_ice_routing = bool(use_ice_routing)
        if self._use_ice_routing:
            assert moe_mode == 'product4', (
                f"use_ice_routing=True only supports moe_mode='product4', "
                f"got moe_mode='{moe_mode}'"
            )
            assert ice_idx is not None, 'use_ice_routing=True but ice_idx is None'
            self.ice_idx = int(ice_idx)
        else:
            self.ice_idx = None

        # ── Input mask for 4-way + ice experts (only passed to modules when moe_mode='product4') ──
        if mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode must be one of {MASK_MODES}, got '{mask_mode}'")
        if mask_mode != 'none':
            assert moe_mode == 'product4', (
                f"mask_mode='{mask_mode}' only supports moe_mode='product4', "
                f"got moe_mode='{moe_mode}'"
            )
        self._mask_mode  = mask_mode
        self._mask_gamma = float(mask_gamma)

        # Input mode setup: 'last_only' (default) or 'concat_all' (12-day ablation)
        assert input_mode in ('last_only', 'concat_all'), \
            f"input_mode must be 'last_only' or 'concat_all', got '{input_mode}'"
        self._input_mode = input_mode
        self.T_expected = int(T)
        self._d_state   = int(d_state)
        # concat_all mode: stem sees T*n_inputs channels and the climate context is forcibly
        # disabled (the 12 frames are already concatenated and fed into the UNet, so the
        # separate climate_state path is unused — this is exactly the ablation comparison)
        if self._input_mode == 'concat_all':
            self._use_temporal_context = False     # forced override
            self._eff_n_inputs = self.T_expected * n_inputs   # e.g. 12 * 9 = 108
        else:
            self._eff_n_inputs = n_inputs

        self.H_in, self.W_in = H, W
        # 4-level UNet → 16x downsampling; pad to a multiple of 16
        self.H_p = ((H - 1) // 16 + 1) * 16   # 90 → 96
        self.W_p = ((W - 1) // 16 + 1) * 16   # 180 → 192
        self.pad_top    = (self.H_p - H) // 2
        self.pad_bottom = self.H_p - H - self.pad_top
        self.pad_left   = (self.W_p - W) // 2
        self.pad_right  = self.W_p - W - self.pad_left

        c = base_channels
        filters = [c, c * 2, c * 4, c * 8, c * 16]   # 32, 64, 128, 256, 512

        # A. Climate state encoder (only built when use_temporal_context=True)
        if self._use_temporal_context:
            if time_pool not in TIME_POOLS:
                raise ValueError(f"time_pool must be one of {TIME_POOLS}, got '{time_pool}'")
            self.climate_encoder = ClimateStateEncoder(n_inputs=n_inputs,
                                                        d_state=d_state,
                                                        time_pool=time_pool)
            self.film            = FiLM(d_state=d_state, channels=filters[4])
        else:
            self.climate_encoder = None
            self.film            = None

        # B. Disentangled stem (goal 4) — or fall back to a plain ConvBlock
        #    concat_all mode: stem sees T*n_inputs channels (12 frames × 9 features = 108 ch)
        #    last_only mode: stem sees n_inputs channels
        if self._use_disentangled_stem:
            self.stem = DisentangledStem(n_inputs=self._eff_n_inputs,
                                          k=stem_k, base=filters[0])
        else:
            self.stem = ConvBlock(self._eff_n_inputs, filters[0])

        # C. UNet encoder (skip level 0 — already handled by the stem)
        self.pool  = nn.MaxPool2d(2)
        self.enc2  = ConvBlock(filters[0], filters[1])
        self.enc3  = ConvBlock(filters[1], filters[2])
        self.enc4  = ConvBlock(filters[2], filters[3])
        self.bottleneck = ConvBlock(filters[3], filters[4])

        # D. Bottleneck regional routing (flag is independent from skip_attn)
        #    moe_mode='product4'  → Regional4WayMoE (4-expert outer product)
        #    other moe_mode       → RegionalMoE     (2 or 2+2 experts)
        if self._use_bottleneck_moe:
            if self._moe_mode == 'product4':
                self.regional = Regional4WayMoE(channels=filters[4],
                                                 use_ice=self._use_ice_routing,
                                                 mask_mode=self._mask_mode,
                                                 mask_gamma=self._mask_gamma)
            else:
                # 'landocean' → use_precip=False; 'wetdry_parallel' → use_precip=True
                _bottleneck_use_precip = (self._moe_mode == 'wetdry_parallel')
                self.regional = RegionalMoE(channels=filters[4],
                                             use_precip=_bottleneck_use_precip)
        else:
            self.regional = None

        # D2. Skip-connection region attention — configurable per level
        #     level 1 → e1 (96×192, base ch),    level 2 → e2 (48×96, 2c)
        #     level 3 → e3 (24×48, 4c),          level 4 → e4 (12×24, 8c)
        sa_d_state = d_state if self._use_temporal_context else 1
        _sa_channels = {1: filters[0], 2: filters[1], 3: filters[2], 4: filters[3]}
        for lv in (1, 2, 3, 4):
            attr_name = f'skip_attn_e{lv}'
            if lv in self._skip_attn_levels:
                if self._moe_mode == 'product4':
                    setattr(self, attr_name,
                            Region4WayExpertAttention(channels=_sa_channels[lv],
                                                       d_state=sa_d_state,
                                                       use_ice=self._use_ice_routing,
                                                       mask_mode=self._mask_mode,
                                                       mask_gamma=self._mask_gamma))
                else:
                    _skip_use_precip = (self._moe_mode == 'wetdry_parallel')
                    setattr(self, attr_name,
                            RegionExpertAttention(channels=_sa_channels[lv],
                                                  d_state=sa_d_state,
                                                  use_precip=_skip_use_precip))
            else:
                setattr(self, attr_name, None)

        # E. UNet decoder (4 levels)
        self.up4 = UpConcat(filters[4], filters[3], filters[3])
        self.up3 = UpConcat(filters[3], filters[2], filters[2])
        self.up2 = UpConcat(filters[2], filters[1], filters[1])
        self.up1 = UpConcat(filters[1], filters[0], filters[0])

        # F. Output head
        self.head = nn.Conv2d(filters[0], 1, kernel_size=1)

        # cache for viz
        self._last_climate_state = None

    # ─── helpers ───────────────────────────────────────────────
    def _pad(self, x):
        return F.pad(x, (self.pad_left, self.pad_right,
                         self.pad_top,  self.pad_bottom),
                     mode='constant', value=0)

    def _crop(self, x):
        return x[:, :,
                 self.pad_top:self.pad_top + self.H_in,
                 self.pad_left:self.pad_left + self.W_in]

    # ─── main forward ─────────────────────────────────────────
    def forward(self, x):
        """
        x: (B, T, C, H, W)  — T-frame sequence
        return: (B, 1, H, W) — last-frame d18Op prediction
        """
        if x.dim() == 4:
            # Backward-compatible single-frame mode (B, C, H, W): no context, climate_state = 0
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        assert C == self.n_inputs, f'expected C={self.n_inputs}, got {C}'

        # 1. Split: context (T-1 frames) and last frame  +  build stem input
        if self._input_mode == 'concat_all':
            # Concat 12 frames into (B, T*C, H, W) → feed directly to stem, no separate climate_state path.
            # This is the ablation demonstrating "direct embedding is worse than disentangled climate-state".
            assert T == self.T_expected, (
                f'concat_all expects T={self.T_expected}, got T={T} — T is fixed in this mode'
            )
            # permute (B,T,C,H,W) → (B,C,T,H,W) → reshape (B, T*C, H, W)
            stem_input = x.permute(0, 2, 1, 3, 4).reshape(B, T * C, H, W)
            climate_state = x.new_zeros(B, self._d_state)
        else:
            # 'last_only': existing logic
            stem_input = x[:, -1]                                  # (B, C, H, W)
            if self._use_temporal_context and T >= 2 and self.climate_encoder is not None:
                x_context = x[:, :-1]                              # (B, T-1, C, H, W)
                climate_state = self.climate_encoder(x_context)    # (B, d_state)
            else:
                climate_state = x.new_zeros(B, self._d_state)

        self._last_climate_state = climate_state.detach()
        # Keep a non-detached version during training for the prompt loss (None at eval time)
        self._train_climate_state = climate_state if self.training else None

        # 2. Extract LANDFRAC for region routing (static — any frame works)
        if self.use_routing:
            landfrac = x[:, 0, self.landfrac_idx:self.landfrac_idx + 1]   # (B, 1, H, W)
            landfrac = landfrac.clamp(0.0, 1.0)
        else:
            landfrac = None

        # 2b. Extract per-sample normalized pr mask for wet/dry routing (optional)
        # Use pr from the last frame (aligned with the target frame we predict), per-sample min-max → [0, 1]
        if self._use_precip_routing:
            pr_raw = x[:, -1, list(self.pr_channels)].sum(dim=1, keepdim=True)   # (B, 1, H, W)
            B_ = pr_raw.shape[0]
            pr_flat = pr_raw.reshape(B_, -1)
            pr_min = pr_flat.min(dim=1).values.view(B_, 1, 1, 1)
            pr_max = pr_flat.max(dim=1).values.view(B_, 1, 1, 1)
            pr_mask = (pr_raw - pr_min) / (pr_max - pr_min + 1e-15)   # ∈ [0, 1]
        else:
            pr_mask = None

        # 2c. Extract sea-ice mask for ice override routing (optional)
        # aice is in [0,1] (SKIP_NORM preserves the physical meaning); take the last frame and use directly as routing mask
        if self._use_ice_routing:
            aice_mask = x[:, -1, self.ice_idx:self.ice_idx + 1]   # (B, 1, H, W)
            aice_mask = aice_mask.clamp(0.0, 1.0)
        else:
            aice_mask = None

        # 3. Pad spatial → (H_p, W_p)
        stem_input = self._pad(stem_input)
        if landfrac is not None:
            landfrac = self._pad(landfrac)
        if pr_mask is not None:
            pr_mask = self._pad(pr_mask)
        if aice_mask is not None:
            aice_mask = self._pad(aice_mask)

        # 4. Stem (disentangled, per-feature spatial conv → 1×1 mixer)
        #    last_only:  stem input is (B, n_inputs, H, W) — the last frame
        #    concat_all: stem input is (B, T*n_inputs, H, W) — all frames concatenated
        e1 = self.stem(stem_input)                   # (B, base, H_p, W_p)

        # 5. UNet encoder
        e2 = self.enc2(self.pool(e1))                # (B, 2c, H_p/2, W_p/2)
        e3 = self.enc3(self.pool(e2))                # (B, 4c, H_p/4, W_p/4)
        e4 = self.enc4(self.pool(e3))                # (B, 8c, H_p/8, W_p/8)
        b  = self.bottleneck(self.pool(e4))          # (B, 16c, H_p/16, W_p/16)

        # 6. Bottleneck: regional MoE routing + FiLM
        if self.regional is not None:
            lf_b = F.interpolate(landfrac, size=b.shape[-2:],
                                 mode='bilinear', align_corners=False)
            pr_b = (F.interpolate(pr_mask, size=b.shape[-2:],
                                  mode='bilinear', align_corners=False)
                    if pr_mask is not None else None)
            # When using 4-way + ice override, also pass aice_b
            if isinstance(self.regional, Regional4WayMoE) and self.regional.use_ice:
                aice_b = F.interpolate(aice_mask, size=b.shape[-2:],
                                       mode='bilinear', align_corners=False)
                b = self.regional(b, lf_b, pr_b=pr_b, aice_b=aice_b)
            else:
                b = self.regional(b, lf_b, pr_b=pr_b)    # MoE routing (no information loss)
        if self.film is not None:
            b = self.film(b, climate_state)          # climate state modulation

        # 6b. Skip-connection region attention — applied at each enabled level
        if any(getattr(self, f'skip_attn_e{lv}') is not None for lv in (1, 2, 3, 4)):
            # Without context, give skip_attn a 1-d zero vector; its state_mlp(0)=0 → offset=0
            sa_state = climate_state if self._use_temporal_context else x.new_zeros(B, 1)
            def _apply(mod, feat):
                if mod is None: return feat
                # Duck-typing: any 4-way-style skip module accepts (feat, lf, state, pr, aice=)
                # Modules with use_ice=True require aice (Region4WayExpertAttention / Region4WayMoESkip)
                if getattr(mod, 'use_ice', False):
                    return mod(feat, landfrac, sa_state, pr=pr_mask, aice=aice_mask)
                return mod(feat, landfrac, sa_state, pr=pr_mask)
            e1 = _apply(self.skip_attn_e1, e1)
            e2 = _apply(self.skip_attn_e2, e2)
            e3 = _apply(self.skip_attn_e3, e3)
            e4 = _apply(self.skip_attn_e4, e4)

        # 7. UNet decoder
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # 8. Output head + crop
        out = self.head(d1)
        return self._crop(out)                        # (B, 1, H, W)

    @torch.no_grad()
    def forward_with_attn(self, x):
        """
        In addition to the main prediction, returns a viz dict:
            'climate_state':       (B, d_state)
            ── bottleneck routing ──
            'landfrac_b':          (B, 1, H_b, W_b)   bottleneck-res LANDFRAC mask
            'b_land':              (B, C_b, H_b, W_b) land-expert output
            'b_ocean':             (B, C_b, H_b, W_b) ocean-expert output
            ── bottleneck FiLM ──
            'film_gamma':          (B, C_b)
            'film_beta':           (B, C_b)
            ── skip attention at e3 (24×48) ──
            'skip3_attn_land':     (B, 1, H_3, W_3)
            'skip3_attn_ocean':    (B, 1, H_3, W_3)
            'skip3_merged':        (B, 1, H_3, W_3)
            ── skip attention at e4 (12×24) ──
            'skip4_attn_land':     (B, 1, H_4, W_4)
            'skip4_attn_ocean':    (B, 1, H_4, W_4)
            'skip4_merged':        (B, 1, H_4, W_4)
        """
        out = self.forward(x)
        attn = {'climate_state': self._last_climate_state,
                'film_gamma':    self.film._last_gamma if self.film is not None else None,
                'film_beta':     self.film._last_beta  if self.film is not None else None,
                'moe_mode':      self._moe_mode}
        # ── Bottleneck caches: dispatch by moe_mode ──────────────
        # Legacy keys are None under product4 mode; new keys are None under legacy modes.
        # Downstream viz should rely on None checks to determine which are available
        # (rather than the mode string).
        if self.regional is not None:
            if isinstance(self.regional, Regional4WayMoE):
                # product4: 4 expert caches (+ optional ice)
                attn['b_LW']      = self.regional._last_b_LW
                attn['b_LD']      = self.regional._last_b_LD
                attn['b_OW']      = self.regional._last_b_OW
                attn['b_OD']      = self.regional._last_b_OD
                attn['b_ice']     = self.regional._last_b_ice       # None if use_ice=False
                attn['aice_b']    = self.regional._last_aice_b
                attn['landfrac_b']= self.regional._last_landfrac_b
                attn['pr_b']      = self.regional._last_pr_b
                # Legacy keys → None (for downstream compatibility)
                for k in ('b_land','b_ocean','b_wet','b_dry'):
                    attn[k] = None
            else:
                # landocean / wetdry_parallel: legacy cache
                attn['b_land']    = self.regional._last_b_land
                attn['b_ocean']   = self.regional._last_b_ocean
                attn['landfrac_b']= self.regional._last_landfrac_b
                attn['b_wet']     = self.regional._last_b_wet      # may be None (no precip)
                attn['b_dry']     = self.regional._last_b_dry
                attn['pr_b']      = self.regional._last_pr_b
                for k in ('b_LW','b_LD','b_OW','b_OD','b_ice','aice_b'):
                    attn[k] = None
        else:
            for k in ('b_land','b_ocean','landfrac_b','b_wet','b_dry','pr_b',
                      'b_LW','b_LD','b_OW','b_OD','b_ice','aice_b'):
                attn[k] = None
        # ── Skip attention at each enabled level (e1..e4) ────────
        for lv in (1, 2, 3, 4):
            mod = getattr(self, f'skip_attn_e{lv}')
            if mod is None:
                for k in ('attn_land','attn_ocean','attn_wet','attn_dry','pr_local',
                          'merged','attn_LW','attn_LD','attn_OW','attn_OD',
                          'attn_ice','aice_local'):
                    attn[f'skip{lv}_{k}'] = None
                continue
            attn[f'skip{lv}_merged'] = mod._last_merged_attn
            if isinstance(mod, Region4WayExpertAttention):
                attn[f'skip{lv}_attn_LW']   = mod._last_attn_LW
                attn[f'skip{lv}_attn_LD']   = mod._last_attn_LD
                attn[f'skip{lv}_attn_OW']   = mod._last_attn_OW
                attn[f'skip{lv}_attn_OD']   = mod._last_attn_OD
                attn[f'skip{lv}_attn_ice']  = mod._last_attn_ice     # None if use_ice=False
                attn[f'skip{lv}_aice_local']= mod._last_aice_local
                attn[f'skip{lv}_pr_local']  = mod._last_pr_local
                attn[f'skip{lv}_lf_local']  = mod._last_landfrac_local   # ★ NEW for viz mask
                for k in ('attn_land','attn_ocean','attn_wet','attn_dry'):
                    attn[f'skip{lv}_{k}'] = None
            else:
                attn[f'skip{lv}_attn_land']  = mod._last_attn_land
                attn[f'skip{lv}_attn_ocean'] = mod._last_attn_ocean
                attn[f'skip{lv}_attn_wet']   = mod._last_attn_wet
                attn[f'skip{lv}_attn_dry']   = mod._last_attn_dry
                attn[f'skip{lv}_pr_local']   = mod._last_pr_local
                for k in ('attn_LW','attn_LD','attn_OW','attn_OD','attn_ice','aice_local'):
                    attn[f'skip{lv}_{k}'] = None
        return out, attn


# ══════════════════════════════════════════════════════════════════════
#  Lightning wrapper (same interface as ConvLSTMSeq / v10, but outputs a single frame)
# ══════════════════════════════════════════════════════════════════════

class IsoUNetBaseline(L.LightningModule):
    """
    Regional Attention UNet — Lightning wrapper.
        forward(x):  (B, T, C, H, W) → (B, 1, H, W)
        loss:        lat-weighted MSE on last frame only

    Batch formats:
        (x, y_seq)         where y_seq: (B, T, 1, H, W)  ← takes the last frame as supervision
        (x, y_seq, co2)    same as above, co2 is ignored
        (x, y_last)        where y_last: (B, 1, H, W)    ← already the last frame
    """
    def __init__(
        self,
        n_inputs:      int   = 8,
        landfrac_idx: int    = 6,
        base_channels: int   = 64,                   # default 64 (previously 32)
        d_state:       int   = 64,
        H:             int   = 90,
        W:             int   = 180,
        stem_k:        int   = 4,
        # Ablation flags
        use_disentangled_stem: bool = True,
        use_temporal_context:  bool = True,
        use_bottleneck_moe:    bool = True,
        use_skip_attn:         bool = True,
        skip_attn_levels:      tuple = (3, 4),       # which encoder levels get skip attention
        # Precip routing (new): add a wet/dry expert pair
        use_precip_routing:    bool = False,
        pr_channels:           tuple = None,
        # MoE mode: None (legacy behavior) | 'landocean' | 'wetdry_parallel' | 'product4'
        moe_mode:              str = None,
        # Climate state temporal pooling: 'mean' (default) | 'gru' | 'conv3d'
        time_pool:             str = 'mean',
        # Ice override (only valid when moe_mode='product4')
        use_ice_routing:       bool = False,
        ice_idx:               int  = None,
        # Input mask for 4-way + ice experts:
        #   'none' (default) | 'hard' (binary) | 'soft' (continuous m) | 'soft_boost' (1 + γ·m)
        mask_mode:             str   = 'none',
        mask_gamma:            float = 0.5,        # only used for mask_mode='soft_boost'
        # Input mode (ablation comparing against the climate-state setup)
        input_mode:            str = 'last_only',   # 'last_only' | 'concat_all'
        T:                     int = 12,            # used by concat_all
        # Prompt loss (climate state ↔ CO2 distance alignment)
        lambda_prompt:        float = 0.1,      # initial weight
        lambda_prompt_final:  float = 0.0,      # final weight after decay
        lambda_decay_epochs:  int   = 50,       # number of epochs for linear annealing (afterwards: stays at final)
        prompt_tau:           float = 0.5,      # similarity temperature
        weights              = None,
        lr:            float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr
        # prompt loss schedule
        self.lambda_prompt        = lambda_prompt
        self.lambda_prompt_final  = lambda_prompt_final
        self.lambda_decay_epochs  = max(1, lambda_decay_epochs)
        self.prompt_tau           = prompt_tau
        self._use_prompt_loss     = (use_temporal_context and lambda_prompt > 0)

        if weights is not None:
            self.register_buffer('lat_weights',
                                 torch.as_tensor(weights, dtype=torch.float32))
        else:
            self.lat_weights = None

        self.net = RegionalAttentionUNet(
            n_inputs=n_inputs, landfrac_idx=landfrac_idx,
            base_channels=base_channels, d_state=d_state,
            H=H, W=W, stem_k=stem_k,
            use_disentangled_stem=use_disentangled_stem,
            use_temporal_context=use_temporal_context,
            use_bottleneck_moe=use_bottleneck_moe,
            use_skip_attn=use_skip_attn,
            skip_attn_levels=skip_attn_levels,
            use_precip_routing=use_precip_routing,
            pr_channels=pr_channels,
            moe_mode=moe_mode,
            time_pool=time_pool,
            use_ice_routing=use_ice_routing,
            ice_idx=ice_idx,
            mask_mode=mask_mode,
            mask_gamma=mask_gamma,
            input_mode=input_mode,
            T=T,
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def forward_with_attn(self, x):
        return self.net.forward_with_attn(x)

    def _unpack(self, batch):
        """Supports (x, y) and (x, y, co2). co2 is also returned now (used by the prompt loss)."""
        if len(batch) == 3:
            x, y, co2 = batch
        else:
            x, y, co2 = batch[0], batch[1], None
        # If y is a sequence (B, T, 1, H, W), take the last frame
        if y.dim() == 5:
            y = y[:, -1]                    # (B, 1, H, W)
        return x, y, co2

    def _loss(self, y_hat, y):
        # y_hat, y: (B, 1, H, W)
        if self.lat_weights is not None:
            loss = F.mse_loss(y_hat, y, reduction='none')
            w = self.lat_weights.view(1, 1, -1, 1)
            return (loss * w).mean()
        return F.mse_loss(y_hat, y)

    def current_lambda_prompt(self) -> float:
        """
        Linear decay from lambda_prompt → lambda_prompt_final over lambda_decay_epochs.
        After decay, stays at lambda_prompt_final.
        """
        ep = self.current_epoch
        progress = min(1.0, ep / self.lambda_decay_epochs)
        return self.lambda_prompt + (self.lambda_prompt_final - self.lambda_prompt) * progress

    def training_step(self, batch, batch_idx):
        x, y, co2 = self._unpack(batch)
        y_hat = self(x)
        mse_loss = self._loss(y_hat, y)
        total = mse_loss

        # Prompt loss: only computed when (1) temporal context is used and (2) the batch has co2
        if self._use_prompt_loss and co2 is not None:
            state = self.net._train_climate_state    # (B, d_state), non-detached
            if state is not None and state.shape[0] >= 2:   # need at least 2 samples for a pair
                prompt_loss = co2_prompt_alignment_loss(state, co2, tau=self.prompt_tau)
                lam = self.current_lambda_prompt()
                total = mse_loss + lam * prompt_loss
                self.log('train_prompt',  prompt_loss, on_step=True, on_epoch=True,
                         prog_bar=True, sync_dist=True)
                self.log('lambda_prompt', lam, on_step=False, on_epoch=True, sync_dist=True)

        self.log('train_mse',  mse_loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log('train_loss', total,    on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return total

    def validation_step(self, batch, batch_idx):
        x, y, _ = self._unpack(batch)
        loss = self._loss(self(x), y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = self._unpack(batch)
        loss = self._loss(self(x), y)
        self.log('test_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ══════════════════════════════════════════════════════════════════
    #  draw_inner() — model-internal viz (derived from model attention/state)
    #  Kept separate from result-only viz (R²/RMSE, etc.); called alongside draw_results() in eval.py
    # ══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def draw_inner(self, test_sets_dict, save_dir, registry,
                   batch_size: int = 4, n_samples: int = 6, seed: int = 42):
        """
        Generate model-internal visualizations for this RegionalAttentionUNet:
            spatial/regional_routing.pdf         — bottleneck MoE land/ocean
            spatial/wetdry_routing.pdf           — (if use_precip) bottleneck wet/dry
            spatial/film_gamma_beta.pdf          — FiLM γ/β comparison across datasets
            spatial/skip_attention.pdf           — skip attention land/ocean (averaged)
            spatial/skip_attention_diff.pdf      — land - ocean attention diff
            spatial/skip_attention_wetdry.pdf    — (if use_precip) skip wet/dry attention
            spatial/skip_attention_wetdry_diff.pdf — (if use_precip) wet - dry diff
            spatial/climate_state_embedding.png  — PCA + t-SNE of climate states
            spatial/climate_states.npz           — raw climate state arrays
            spatial/samples/samples_<tag>.pdf    — K individual-window viz per dataset
            spatial/samples/samples_wetdry_<tag>.pdf — (if use_precip) wet/dry per-sample

        Args:
            test_sets_dict: {tag: test_input}; for IsoUNet (sequence model), test_input
                            is a SequenceDataset instance.
            save_dir:       output root (a spatial/ subdirectory is created).
            registry:       REGISTRY dict (used to read co2 / holdout to flag OOD).
            batch_size:     inference batch size.
            n_samples:      per-sample viz: sample N random windows per dataset.
            seed:           seed for per-sample sampling.

        Side effects: writes multiple PDFs + npz under save_dir/spatial/.
        """
        import os
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        device = next(self.parameters()).device
        out_dir = os.path.join(save_dir, 'spatial')
        os.makedirs(out_dir, exist_ok=True)
        print(f'  [draw_inner: RegionalAttentionUNet] writing to {out_dir}/')

        # ── 1. Collect averaged attention statistics for each dataset ──
        attn_summary = {}
        for tag in registry:
            test_ds = test_sets_dict[tag]
            attn_summary[tag] = self._collect_avg_attn(test_ds, device, batch_size)
            cs = attn_summary[tag]['climate_state']
            gam = attn_summary[tag]['film_gamma']
            print(f'    {tag}: ‖state‖={np.linalg.norm(cs, axis=1).mean():.3f}, '
                  f'|γ|={np.abs(gam).mean():.4f}')

        # ── 2. Climate states npz + PCA/t-SNE ──
        self._plot_climate_state_embedding(attn_summary, registry, out_dir)

        # ── 3. Bottleneck MoE: regional_routing.pdf (land/ocean) ──
        self._plot_regional_routing(attn_summary, registry, out_dir)

        # ── 3b. Bottleneck MoE: wetdry_routing.pdf (only if use_precip parallel) ──
        any_wet = any('wet_norm_mean' in attn_summary[t] for t in registry)
        if any_wet:
            self._plot_wetdry_routing(attn_summary, registry, out_dir)
            print(f'    wetdry_routing.pdf written (use_precip_routing=True, parallel)')

        # ── 3c. Bottleneck MoE: 4way_routing.pdf (only if moe_mode='product4') ──
        any_4way = any('LW_norm_mean' in attn_summary[t] for t in registry)
        if any_4way:
            self._plot_4way_routing(attn_summary, registry, out_dir)
            print(f'    4way_routing.pdf written (moe_mode=product4)')

        # ── 3d. Bottleneck ice override: ice_routing.pdf (only if use_ice_routing=True) ──
        any_ice = any('ice_norm_mean' in attn_summary[t] for t in registry)
        if any_ice:
            self._plot_ice_routing(attn_summary, registry, out_dir)
            print(f'    ice_routing.pdf written (use_ice_routing=True)')

        # ── 4. FiLM γ/β: film_gamma_beta.pdf ──
        self._plot_film_gamma_beta(attn_summary, registry, out_dir)

        # ── 5. Skip attention (averaged): skip_attention[_diff].pdf ──
        self._plot_skip_attention(attn_summary, registry, out_dir)

        # ── 5b. Skip attention wet/dry (only if use_precip parallel) ──
        any_skip_wet = any(
            f'skip{lv}_wet_mean' in attn_summary[t]
            for t in registry for lv in (1, 2, 3, 4)
        )
        if any_skip_wet:
            self._plot_skip_attention_wetdry(attn_summary, registry, out_dir)
            print(f'    skip_attention_wetdry[_diff].pdf written')

        # ── 5c. Skip attention 4-way product (only if moe_mode='product4') ──
        any_skip_4way = any(
            f'skip{lv}_LW_mean' in attn_summary[t]
            for t in registry for lv in (1, 2, 3, 4)
        )
        if any_skip_4way:
            self._plot_skip_attention_4way(attn_summary, registry, out_dir)
            print(f'    skip_attention_4way.pdf written')

        # ── 5d. Skip ice attention (only if use_ice_routing=True) ──
        any_skip_ice = any(
            f'skip{lv}_ice_mean' in attn_summary[t]
            for t in registry for lv in (1, 2, 3, 4)
        )
        if any_skip_ice:
            self._plot_skip_attention_ice(attn_summary, registry, out_dir)
            print(f'    skip_attention_ice.pdf written')

        # ── 6. Per-sample (no averaging): samples/samples_<tag>.pdf ──
        samples_dir = os.path.join(out_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        print(f'    per-sample (n_samples={n_samples}, seed={seed}) → {samples_dir}/')
        for tag, cfg in registry.items():
            test_ds = test_sets_dict[tag]
            samples = self._collect_individual_samples(
                test_ds, device, n_samples=n_samples, seed=seed,
            )
            self._plot_individual_samples(
                samples, tag, cfg['co2'], cfg['holdout'], samples_dir,
            )
            # If wet/dry caches present in samples → also write samples_wetdry_*
            if any(s.get('b_wet_norm') is not None for s in samples):
                self._plot_individual_samples_wetdry(
                    samples, tag, cfg['co2'], cfg['holdout'], samples_dir,
                )
            # If 4-way caches present → also write samples_4way_*
            if any(s.get('b_LW_norm') is not None for s in samples):
                self._plot_individual_samples_4way(
                    samples, tag, cfg['co2'], cfg['holdout'], samples_dir,
                )
            # If e1/e2 skip attention present (4-way) → also write samples_4way_e12_*
            if any(s.get('skip1_LW') is not None or s.get('skip2_LW') is not None
                   for s in samples):
                self._plot_individual_samples_4way_e12(
                    samples, tag, cfg['co2'], cfg['holdout'], samples_dir,
                )
            idxs = [s['idx'] for s in samples]
            print(f'      {tag}: samples_{tag}.pdf  (idx={idxs})')

        print(f'  [draw_inner: RegionalAttentionUNet] done')

    # ───── Private helper methods (not part of the forward path; used only by draw_inner) ─────

    @torch.no_grad()
    def _collect_avg_attn(self, test_ds, device, batch_size):
        """Run forward_with_attn over one dataset and return a stats dict averaged over time.
        Includes wet/dry routing caches if enabled.
        """
        import numpy as np
        loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
        cstates, gammas, betas = [], [], []
        # land/ocean (2-expert bottleneck)
        land_norms, ocean_norms = [], []
        # wet/dry (parallel 2-expert bottleneck)
        wet_norms, dry_norms = [], []
        # 4-way (product4 bottleneck)
        LW_norms, LD_norms, OW_norms, OD_norms = [], [], [], []
        ice_norms, aice_b_accum = [], []   # ice override caches
        pr_b_accum = []
        # skip attentions (land/ocean + wet/dry + 4-way + ice)
        skip_sides = ('land', 'ocean', 'wet', 'dry', 'LW', 'LD', 'OW', 'OD', 'ice')
        skips = {f'skip{lv}_{side}': [] for lv in (1, 2, 3, 4) for side in skip_sides}
        skip_aice = {f'skip{lv}_aice': [] for lv in (1, 2, 3, 4)}
        skip_lf   = {f'skip{lv}_lf':   [] for lv in (1, 2, 3, 4)}     # for viz mask
        lf_b = None
        for batch in loader:
            x = batch[0].to(device)
            _, attn = self.forward_with_attn(x)
            cstates.append(attn['climate_state'].cpu().numpy())
            if attn['film_gamma'] is not None:
                gammas.append(attn['film_gamma'].cpu().numpy())
                betas .append(attn['film_beta'].cpu().numpy())
            # Bottleneck land/ocean (legacy 2-expert path)
            if attn.get('b_land') is not None:
                land_norms.append(attn['b_land'].norm(dim=1).cpu().numpy())
                ocean_norms.append(attn['b_ocean'].norm(dim=1).cpu().numpy())
                if lf_b is None:
                    lf_b = attn['landfrac_b'].squeeze(1)[0].cpu().numpy()
            # Bottleneck wet/dry (2+2-expert parallel path)
            if attn.get('b_wet') is not None:
                wet_norms.append(attn['b_wet'].norm(dim=1).cpu().numpy())
                dry_norms.append(attn['b_dry'].norm(dim=1).cpu().numpy())
            # Bottleneck 4-way product (product4 path)
            if attn.get('b_LW') is not None:
                LW_norms.append(attn['b_LW'].norm(dim=1).cpu().numpy())
                LD_norms.append(attn['b_LD'].norm(dim=1).cpu().numpy())
                OW_norms.append(attn['b_OW'].norm(dim=1).cpu().numpy())
                OD_norms.append(attn['b_OD'].norm(dim=1).cpu().numpy())
                if lf_b is None and attn.get('landfrac_b') is not None:
                    lf_b = attn['landfrac_b'].squeeze(1)[0].cpu().numpy()
            # Bottleneck ice override (use_ice=True path)
            if attn.get('b_ice') is not None:
                ice_norms.append(attn['b_ice'].norm(dim=1).cpu().numpy())
                aice_b_accum.append(attn['aice_b'].squeeze(1).cpu().numpy())
            # pr_b: shared (cached by both wetdry_parallel and product4)
            if attn.get('pr_b') is not None:
                pr_b_accum.append(attn['pr_b'].squeeze(1).cpu().numpy())
            # Skip attentions (per level, per side)
            for lv in (1, 2, 3, 4):
                # Legacy land/ocean
                if attn.get(f'skip{lv}_attn_land') is not None:
                    skips[f'skip{lv}_land' ].append(attn[f'skip{lv}_attn_land' ].squeeze(1).cpu().numpy())
                    skips[f'skip{lv}_ocean'].append(attn[f'skip{lv}_attn_ocean'].squeeze(1).cpu().numpy())
                # Legacy wet/dry parallel
                if attn.get(f'skip{lv}_attn_wet') is not None:
                    skips[f'skip{lv}_wet'].append(attn[f'skip{lv}_attn_wet'].squeeze(1).cpu().numpy())
                    skips[f'skip{lv}_dry'].append(attn[f'skip{lv}_attn_dry'].squeeze(1).cpu().numpy())
                # 4-way product
                if attn.get(f'skip{lv}_attn_LW') is not None:
                    for s in ('LW', 'LD', 'OW', 'OD'):
                        skips[f'skip{lv}_{s}'].append(attn[f'skip{lv}_attn_{s}'].squeeze(1).cpu().numpy())
                    # Also cache lf at this skip level (for viz mask)
                    if attn.get(f'skip{lv}_lf_local') is not None:
                        skip_lf[f'skip{lv}_lf'].append(attn[f'skip{lv}_lf_local'].squeeze(1).cpu().numpy())
                # Ice override on skip (only if use_ice=True)
                if attn.get(f'skip{lv}_attn_ice') is not None:
                    skips[f'skip{lv}_ice'].append(attn[f'skip{lv}_attn_ice'].squeeze(1).cpu().numpy())
                    skip_aice[f'skip{lv}_aice'].append(attn[f'skip{lv}_aice_local'].squeeze(1).cpu().numpy())
        out = {
            'climate_state': np.concatenate(cstates, axis=0),
            'film_gamma':    np.concatenate(gammas,  axis=0) if gammas else None,
            'film_beta':     np.concatenate(betas,   axis=0) if betas  else None,
        }
        if land_norms:
            out['land_norm_mean']  = np.concatenate(land_norms,  axis=0).mean(axis=0)
            out['ocean_norm_mean'] = np.concatenate(ocean_norms, axis=0).mean(axis=0)
            out['landfrac_b']      = lf_b
        if wet_norms:
            out['wet_norm_mean']   = np.concatenate(wet_norms, axis=0).mean(axis=0)
            out['dry_norm_mean']   = np.concatenate(dry_norms, axis=0).mean(axis=0)
        if LW_norms:
            out['LW_norm_mean']    = np.concatenate(LW_norms, axis=0).mean(axis=0)
            out['LD_norm_mean']    = np.concatenate(LD_norms, axis=0).mean(axis=0)
            out['OW_norm_mean']    = np.concatenate(OW_norms, axis=0).mean(axis=0)
            out['OD_norm_mean']    = np.concatenate(OD_norms, axis=0).mean(axis=0)
            out['landfrac_b']      = lf_b
        if ice_norms:
            out['ice_norm_mean']   = np.concatenate(ice_norms, axis=0).mean(axis=0)
            out['aice_b_mean']     = np.concatenate(aice_b_accum, axis=0).mean(axis=0)
        if pr_b_accum:
            out['pr_b_mean']       = np.concatenate(pr_b_accum, axis=0).mean(axis=0)
        for k, v in skips.items():
            if v: out[f'{k}_mean'] = np.concatenate(v, axis=0).mean(axis=0)
        for k, v in skip_aice.items():
            if v: out[f'{k}_mean'] = np.concatenate(v, axis=0).mean(axis=0)
        for k, v in skip_lf.items():
            if v: out[f'{k}_mean'] = np.concatenate(v, axis=0).mean(axis=0)
        return out

    @torch.no_grad()
    def _collect_individual_samples(self, test_ds, device, n_samples=6, seed=42):
        """Randomly draw n_samples windows and run forward_with_attn individually (no averaging)."""
        import numpy as np
        n_total = len(test_ds)
        rng = np.random.RandomState(seed)
        indices = sorted(rng.choice(n_total, size=min(n_samples, n_total),
                                    replace=False).tolist())
        # find LANDFRAC index from input — try common positions
        landfrac_idx = self.net.landfrac_idx if self.net.landfrac_idx is not None else 6

        samples = []
        for idx in indices:
            item = test_ds[idx]
            x = item[0].unsqueeze(0).to(device)
            y_seq = item[1]
            y_hat, attn = self.forward_with_attn(x)

            def _get(k):
                v = attn.get(k)
                return v.cpu().numpy() if v is not None else None

            s = {
                'idx': idx,
                'TS_last':       x[0, -1, 0].cpu().numpy(),
                'landfrac_full': x[0, -1, landfrac_idx].cpu().numpy(),
                'truth':         y_seq[-1, 0].numpy(),
                'pred':          y_hat[0, 0].cpu().numpy(),
                # land/ocean (2-expert)
                'b_land_norm':   _get('b_land'),
                'b_ocean_norm':  _get('b_ocean'),
                'landfrac_b':    _get('landfrac_b'),
                'skip3_land':    _get('skip3_attn_land'),
                'skip3_ocean':   _get('skip3_attn_ocean'),
                'skip4_land':    _get('skip4_attn_land'),
                'skip4_ocean':   _get('skip4_attn_ocean'),
                # wet/dry parallel (may be None)
                'b_wet_norm':    _get('b_wet'),
                'b_dry_norm':    _get('b_dry'),
                'pr_b':          _get('pr_b'),
                'skip3_wet':     _get('skip3_attn_wet'),
                'skip3_dry':     _get('skip3_attn_dry'),
                'skip4_wet':     _get('skip4_attn_wet'),
                'skip4_dry':     _get('skip4_attn_dry'),
                # 4-way product (may be None — only set in product4 mode)
                'b_LW_norm':     _get('b_LW'),
                'b_LD_norm':     _get('b_LD'),
                'b_OW_norm':     _get('b_OW'),
                'b_OD_norm':     _get('b_OD'),
                'skip1_LW':      _get('skip1_attn_LW'),
                'skip1_LD':      _get('skip1_attn_LD'),
                'skip1_OW':      _get('skip1_attn_OW'),
                'skip1_OD':      _get('skip1_attn_OD'),
                'skip2_LW':      _get('skip2_attn_LW'),
                'skip2_LD':      _get('skip2_attn_LD'),
                'skip2_OW':      _get('skip2_attn_OW'),
                'skip2_OD':      _get('skip2_attn_OD'),
                'skip3_LW':      _get('skip3_attn_LW'),
                'skip3_LD':      _get('skip3_attn_LD'),
                'skip3_OW':      _get('skip3_attn_OW'),
                'skip3_OD':      _get('skip3_attn_OD'),
                'skip4_LW':      _get('skip4_attn_LW'),
                'skip4_LD':      _get('skip4_attn_LD'),
                'skip4_OW':      _get('skip4_attn_OW'),
                'skip4_OD':      _get('skip4_attn_OD'),
                # Ice override (may be None — only set if use_ice_routing=True)
                'b_ice_norm':    _get('b_ice'),
                'aice_b':        _get('aice_b'),
                'skip1_ice':     _get('skip1_attn_ice'),
                'skip2_ice':     _get('skip2_attn_ice'),
                'skip3_ice':     _get('skip3_attn_ice'),
                'skip4_ice':     _get('skip4_attn_ice'),
                'pr_full':       attn['regional_pr_full'][0,0].cpu().numpy()
                                  if attn.get('regional_pr_full') is not None else None,
            }
            if s['b_land_norm'] is not None:
                s['b_land_norm']  = np.linalg.norm(s['b_land_norm'][0], axis=0)
                s['b_ocean_norm'] = np.linalg.norm(s['b_ocean_norm'][0], axis=0)
                s['landfrac_b']   = s['landfrac_b'][0, 0]
            if s['b_wet_norm'] is not None:
                s['b_wet_norm']   = np.linalg.norm(s['b_wet_norm'][0], axis=0)
                s['b_dry_norm']   = np.linalg.norm(s['b_dry_norm'][0], axis=0)
                s['pr_b']         = s['pr_b'][0, 0]
            if s['b_LW_norm'] is not None:
                s['b_LW_norm']   = np.linalg.norm(s['b_LW_norm'][0], axis=0)
                s['b_LD_norm']   = np.linalg.norm(s['b_LD_norm'][0], axis=0)
                s['b_OW_norm']   = np.linalg.norm(s['b_OW_norm'][0], axis=0)
                s['b_OD_norm']   = np.linalg.norm(s['b_OD_norm'][0], axis=0)
                # Squeeze (1, 1, H, W) → (H, W); if the legacy land/ocean path did not run, these are still 4D
                if s.get('landfrac_b') is not None and s['landfrac_b'].ndim == 4:
                    s['landfrac_b'] = s['landfrac_b'][0, 0]
                if s.get('pr_b') is not None and s['pr_b'].ndim == 4:
                    s['pr_b'] = s['pr_b'][0, 0]
            if s.get('b_ice_norm') is not None:
                s['b_ice_norm']  = np.linalg.norm(s['b_ice_norm'][0], axis=0)
                if s.get('aice_b') is not None:
                    s['aice_b'] = s['aice_b'][0, 0]
            for k in ('skip3_land','skip3_ocean','skip4_land','skip4_ocean',
                      'skip3_wet','skip3_dry','skip4_wet','skip4_dry',
                      'skip1_LW','skip1_LD','skip1_OW','skip1_OD',
                      'skip2_LW','skip2_LD','skip2_OW','skip2_OD',
                      'skip3_LW','skip3_LD','skip3_OW','skip3_OD',
                      'skip4_LW','skip4_LD','skip4_OW','skip4_OD',
                      'skip1_ice','skip2_ice','skip3_ice','skip4_ice'):
                if s[k] is not None: s[k] = s[k][0, 0]
            samples.append(s)
        return samples

    @staticmethod
    def _plot_climate_state_embedding(attn_summary, registry, out_dir):
        """PCA + t-SNE of climate states across all datasets."""
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # Save raw climate_states.npz
        np.savez(os.path.join(out_dir, 'climate_states.npz'),
                 **{tag: attn_summary[tag]['climate_state'] for tag in registry},
                 co2={tag: registry[tag]['co2'] for tag in registry})

        all_states, all_co2, all_tag, all_split = [], [], [], []
        for tag in registry:
            arr = attn_summary[tag]['climate_state']
            n = arr.shape[0]
            all_states.append(arr)
            all_co2 .append(np.full(n, registry[tag]['co2']))
            all_split.append(np.full(n, 'OOD' if registry[tag]['holdout'] else 'ID', dtype='<U3'))
            all_tag .append(np.full(n, tag, dtype='<U8'))
        all_states = np.concatenate(all_states, axis=0)
        all_co2    = np.concatenate(all_co2)
        all_tag    = np.concatenate(all_tag)

        from sklearn.decomposition import PCA
        from sklearn.manifold   import TSNE

        sorted_tags = sorted(registry.keys(), key=lambda t: registry[t]['co2'])
        co2_vals    = [registry[t]['co2'] for t in sorted_tags]
        norm = mcolors.Normalize(vmin=min(co2_vals), vmax=max(co2_vals))
        tag_color = {t: cm.viridis(norm(registry[t]['co2'])) for t in registry}
        tag_marker = {t: ('^' if registry[t]['holdout'] else 'o') for t in registry}

        pca = PCA(n_components=2); coords_p = pca.fit_transform(all_states)
        perp = max(5, min(30, all_states.shape[0] // 4))
        coords_t = TSNE(n_components=2, perplexity=perp, init='pca',
                        learning_rate='auto', random_state=42).fit_transform(all_states)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax_, coords, title in [
            (axes[0], coords_p, f'PCA (var={pca.explained_variance_ratio_[0]*100:.1f}/{pca.explained_variance_ratio_[1]*100:.1f}%)'),
            (axes[1], coords_t, f't-SNE (perp={perp})'),
        ]:
            for t in sorted_tags:
                m = (all_tag == t)
                if not m.any(): continue
                ax_.scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.7,
                            color=tag_color[t], marker=tag_marker[t], edgecolors='none',
                            label=f"{t} ({registry[t]['co2']} ppm)"
                                  + (' [OOD]' if registry[t]['holdout'] else ''))
            ax_.set_title(title); ax_.grid(True, alpha=0.3)
            ax_.legend(fontsize=8, loc='best', framealpha=0.85)
        fig.suptitle('climate_state (circle=ID, triangle=OOD; color=CO2)')
        plt.tight_layout()
        _save_fig_both(os.path.join(out_dir, 'climate_state_embedding.png'), dpi=140)
        plt.close()

    @staticmethod
    def _plot_regional_routing(attn_summary, registry, out_dir):
        """7 datasets × 3 cols: LANDFRAC@b, Land-expert, Ocean-expert."""
        import os
        import matplotlib.pyplot as plt
        n_tags = len(registry)
        fig, axes = plt.subplots(n_tags, 3, figsize=(12, 2.4 * n_tags))
        if n_tags == 1: axes = axes.reshape(1, -1)
        for r, tag in enumerate(registry):
            info = attn_summary[tag]
            if 'land_norm_mean' not in info:
                for c in range(3): axes[r, c].axis('off')
                axes[r, 0].text(0.5, 0.5, f'{tag}: no MoE', ha='center', va='center',
                                transform=axes[r, 0].transAxes)
                continue
            ld, oc, lf = info['land_norm_mean'], info['ocean_norm_mean'], info['landfrac_b']
            axes[r, 0].imshow(lf, origin='lower', cmap='Greys', aspect='auto')
            axes[r, 1].imshow(ld, origin='lower', cmap='YlOrBr', aspect='auto')
            axes[r, 2].imshow(oc, origin='lower', cmap='Blues', aspect='auto')
            if r == 0:
                for c, t in enumerate(['LANDFRAC@bottleneck', 'Land-expert ‖act‖', 'Ocean-expert ‖act‖']):
                    axes[r, c].set_title(t, fontsize=9)
            axes[r, 0].set_ylabel(f'{tag}\n({registry[tag]["co2"]} ppm)', fontsize=8)
            for c in range(3):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        plt.tight_layout()
        _save_fig_both(os.path.join(out_dir, 'regional_routing.pdf'), dpi=140)
        plt.close()

    @staticmethod
    def _plot_film_gamma_beta(attn_summary, registry, out_dir):
        """Distribution of FiLM γ/β over bottleneck channels."""
        import os
        import matplotlib.pyplot as plt
        if not any(attn_summary[t].get('film_gamma') is not None for t in registry):
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for tag in registry:
            g_arr = attn_summary[tag].get('film_gamma')
            b_arr = attn_summary[tag].get('film_beta')
            if g_arr is None: continue
            g = g_arr.mean(axis=0); b = b_arr.mean(axis=0)
            marker = '^' if registry[tag]['holdout'] else 'o'
            axes[0].plot(g, label=tag, alpha=0.7, marker=marker, markersize=3)
            axes[1].plot(b, label=tag, alpha=0.7, marker=marker, markersize=3)
        axes[0].set_title('FiLM γ (per channel)'); axes[0].set_ylabel('γ')
        axes[1].set_title('FiLM β');               axes[1].set_ylabel('β')
        for ax_ in axes:
            ax_.set_xlabel('bottleneck channel')
            ax_.legend(fontsize=8); ax_.grid(True, alpha=0.3)
            ax_.axhline(0, color='k', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        _save_fig_both(os.path.join(out_dir, 'film_gamma_beta.pdf'), dpi=140)
        plt.close()

    @staticmethod
    def _plot_skip_attention(attn_summary, registry, out_dir):
        """
        Averaged skip attention maps + land-ocean diff. Auto-detects which levels are enabled.
        """
        import os
        import matplotlib.pyplot as plt

        enabled_levels = []
        for lv in (1, 2, 3, 4):
            if any(f'skip{lv}_land_mean' in attn_summary[t] for t in registry):
                enabled_levels.append(lv)
        if not enabled_levels:
            return

        n_tags = len(registry)
        n_cols = 2 * len(enabled_levels)
        fig, axes = plt.subplots(n_tags, n_cols, figsize=(n_cols * 1.7, n_tags * 1.5))
        if n_tags == 1: axes = axes.reshape(1, -1)
        for r, tag in enumerate(registry):
            info = attn_summary[tag]
            for li, lv in enumerate(enabled_levels):
                col_l, col_o = 2 * li, 2 * li + 1
                m_l = info.get(f'skip{lv}_land_mean');  m_o = info.get(f'skip{lv}_ocean_mean')
                if m_l is None:
                    axes[r, col_l].axis('off'); axes[r, col_o].axis('off'); continue
                axes[r, col_l].imshow(m_l, origin='lower', cmap='inferno', vmin=0, vmax=1, aspect='auto')
                axes[r, col_o].imshow(m_o, origin='lower', cmap='inferno', vmin=0, vmax=1, aspect='auto')
                if r == 0:
                    axes[r, col_l].set_title(f'skip e{lv}  land-α', fontsize=9)
                    axes[r, col_o].set_title(f'skip e{lv}  ocean-α', fontsize=9)
            axes[r, 0].set_ylabel(f'{tag}\n({registry[tag]["co2"]} ppm)' +
                                  (' [OOD]' if registry[tag]['holdout'] else ''), fontsize=8)
            for c in range(n_cols):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        plt.tight_layout()
        _save_fig_both(os.path.join(out_dir, 'skip_attention.pdf'), dpi=140)
        plt.close()

        # Diff plots: land - ocean
        fig, axes = plt.subplots(n_tags, len(enabled_levels),
                                  figsize=(len(enabled_levels) * 2.2, n_tags * 1.5))
        if n_tags == 1 and len(enabled_levels) == 1: axes = axes.reshape(1, 1)
        elif n_tags == 1: axes = axes.reshape(1, -1)
        elif len(enabled_levels) == 1: axes = axes.reshape(-1, 1)
        for r, tag in enumerate(registry):
            info = attn_summary[tag]
            for li, lv in enumerate(enabled_levels):
                m_l = info.get(f'skip{lv}_land_mean'); m_o = info.get(f'skip{lv}_ocean_mean')
                if m_l is None: axes[r, li].axis('off'); continue
                d = m_l - m_o
                vmax = max(abs(d).max(), 1e-6)
                axes[r, li].imshow(d, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
                if r == 0: axes[r, li].set_title(f'skip e{lv}  (land - ocean)', fontsize=9)
                axes[r, li].set_xticks([]); axes[r, li].set_yticks([])
            axes[r, 0].set_ylabel(f'{tag}', fontsize=8)
        plt.tight_layout()
        _save_fig_both(os.path.join(out_dir, 'skip_attention_diff.pdf'), dpi=140)
        plt.close()

    @staticmethod
    def _plot_wetdry_routing(attn_summary, registry, out_dir):
        """
        7 datasets × 3 cols (parallel to _plot_regional_routing):
        pr@bottleneck (mean), Wet-expert ‖act‖, Dry-expert ‖act‖.
        """
        import os
        import matplotlib.pyplot as plt
        n_tags = len(registry)
        fig, axes = plt.subplots(n_tags, 3, figsize=(12, 2.4 * n_tags))
        if n_tags == 1: axes = axes.reshape(1, -1)
        for r, tag in enumerate(registry):
            info = attn_summary[tag]
            if 'wet_norm_mean' not in info:
                for c in range(3): axes[r, c].axis('off')
                axes[r, 0].text(0.5, 0.5, f'{tag}: no wet/dry routing',
                                ha='center', va='center',
                                transform=axes[r, 0].transAxes)
                continue
            wn, dn, prb = info['wet_norm_mean'], info['dry_norm_mean'], info['pr_b_mean']
            axes[r, 0].imshow(prb, origin='lower', cmap='Blues',   aspect='auto', vmin=0, vmax=1)
            axes[r, 1].imshow(wn,  origin='lower', cmap='YlOrBr',  aspect='auto')
            axes[r, 2].imshow(dn,  origin='lower', cmap='Purples', aspect='auto')
            if r == 0:
                for c, t in enumerate(['pr@bottleneck (mean, ∈[0,1])',
                                       'Wet-expert ‖act‖',
                                       'Dry-expert ‖act‖']):
                    axes[r, c].set_title(t, fontsize=9)
            axes[r, 0].set_ylabel(f'{tag}\n({registry[tag]["co2"]} ppm)' +
                                  (' [OOD]' if registry[tag]['holdout'] else ''), fontsize=8)
            for c in range(3):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        fig.suptitle('Bottleneck Wet/Dry MoE Routing (time-averaged)', fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig_both(os.path.join(out_dir, 'wetdry_routing.pdf'), dpi=140)
        plt.close()

    @staticmethod
    def _plot_skip_attention_wetdry(attn_summary, registry, out_dir):
        """Parallel to _plot_skip_attention, but uses wet/dry attention maps."""
        import os
        import matplotlib.pyplot as plt
        enabled_levels = []
        for lv in (1, 2, 3, 4):
            if any(f'skip{lv}_wet_mean' in attn_summary[t] for t in registry):
                enabled_levels.append(lv)
        if not enabled_levels:
            return

        n_tags = len(registry)
        n_cols = 2 * len(enabled_levels)
        fig, axes = plt.subplots(n_tags, n_cols, figsize=(n_cols * 1.7, n_tags * 1.5))
        if n_tags == 1: axes = axes.reshape(1, -1)
        for r, tag in enumerate(registry):
            info = attn_summary[tag]
            for li, lv in enumerate(enabled_levels):
                col_w, col_d = 2 * li, 2 * li + 1
                m_w = info.get(f'skip{lv}_wet_mean'); m_d = info.get(f'skip{lv}_dry_mean')
                if m_w is None:
                    axes[r, col_w].axis('off'); axes[r, col_d].axis('off'); continue
                axes[r, col_w].imshow(m_w, origin='lower', cmap='inferno', vmin=0, vmax=1, aspect='auto')
                axes[r, col_d].imshow(m_d, origin='lower', cmap='inferno', vmin=0, vmax=1, aspect='auto')
                if r == 0:
                    axes[r, col_w].set_title(f'skip e{lv}  wet-α', fontsize=9)
                    axes[r, col_d].set_title(f'skip e{lv}  dry-α', fontsize=9)
            axes[r, 0].set_ylabel(f'{tag}\n({registry[tag]["co2"]} ppm)' +
                                  (' [OOD]' if registry[tag]['holdout'] else ''), fontsize=8)
            for c in range(n_cols):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        fig.suptitle('Skip Wet/Dry Attention (time-averaged)', fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig_both(os.path.join(out_dir, 'skip_attention_wetdry.pdf'), dpi=140)
        plt.close()

        # Diff: wet - dry
        fig, axes = plt.subplots(n_tags, len(enabled_levels),
                                  figsize=(len(enabled_levels) * 2.2, n_tags * 1.5))
        if n_tags == 1 and len(enabled_levels) == 1: axes = axes.reshape(1, 1)
        elif n_tags == 1: axes = axes.reshape(1, -1)
        elif len(enabled_levels) == 1: axes = axes.reshape(-1, 1)
        for r, tag in enumerate(registry):
            info = attn_summary[tag]
            for li, lv in enumerate(enabled_levels):
                m_w = info.get(f'skip{lv}_wet_mean'); m_d = info.get(f'skip{lv}_dry_mean')
                if m_w is None: axes[r, li].axis('off'); continue
                d = m_w - m_d
                vmax = max(abs(d).max(), 1e-6)
                axes[r, li].imshow(d, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
                if r == 0: axes[r, li].set_title(f'skip e{lv}  (wet - dry)', fontsize=9)
                axes[r, li].set_xticks([]); axes[r, li].set_yticks([])
            axes[r, 0].set_ylabel(f'{tag}', fontsize=8)
        fig.suptitle('Skip Wet - Dry Attention Difference', fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig_both(os.path.join(out_dir, 'skip_attention_wetdry_diff.pdf'), dpi=140)
        plt.close()

    @staticmethod
    def _plot_individual_samples_wetdry(samples, tag, co2, holdout, out_dir):
        """K rows × 7 cols: per-window wet/dry attention subset."""
        import os
        import matplotlib.pyplot as plt
        K = len(samples)
        if K == 0: return
        n_cols = 7
        fig, axes = plt.subplots(K, n_cols, figsize=(n_cols * 1.7, K * 1.6))
        if K == 1: axes = axes.reshape(1, -1)
        col_titles = ['pr@bottleneck', 'Bot. Wet‖act‖', 'Bot. Dry‖act‖',
                      'Skip e3 wet-α', 'Skip e3 dry-α',
                      'Skip e4 wet-α', 'Skip e4 dry-α']
        col_cmaps  = ['Blues',  'YlOrBr', 'Purples',
                      'inferno','inferno',
                      'inferno','inferno']
        for r, s in enumerate(samples):
            cells = [s.get('pr_b'),
                     s.get('b_wet_norm'), s.get('b_dry_norm'),
                     s.get('skip3_wet'), s.get('skip3_dry'),
                     s.get('skip4_wet'), s.get('skip4_dry')]
            for c, m in enumerate(cells):
                ax = axes[r, c]
                if m is None: ax.axis('off'); continue
                kw = {'origin': 'lower', 'cmap': col_cmaps[c], 'aspect': 'auto'}
                if c == 0: kw['vmin'], kw['vmax'] = 0.0, 1.0       # pr_b normalized
                elif c >= 3: kw['vmin'], kw['vmax'] = 0.0, 1.0     # sigmoid attention
                ax.imshow(m, **kw)
                if r == 0: ax.set_title(col_titles[c], fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            axes[r, 0].set_ylabel(f'#{s["idx"]}', fontsize=8, rotation=0,
                                  labelpad=18, va='center')
        flag = ' [OOD]' if holdout else ''
        fig.suptitle(f'{tag} ({co2} ppm){flag} — wet/dry routing per sample', fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig_both(os.path.join(out_dir, f'samples_wetdry_{tag}.pdf'), dpi=120)
        plt.close()

    @staticmethod
    def _plot_4way_routing(attn_summary, registry, out_dir):
        """
        Product-4 MoE @ bottleneck: 7 datasets × 6 cols:
            LANDFRAC@b | pr@b | ‖E_LW‖ | ‖E_LD‖ | ‖E_OW‖ | ‖E_OD‖
        LW=land+wet, LD=land+dry, OW=ocean+wet, OD=ocean+dry.
        Visualization mask: ‖E_LW‖/‖E_LD‖ × (lf>0.5);  ‖E_OW‖/‖E_OD‖ × (lf<=0.5)
        which makes the physical regimes clearer.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        n_tags = len(registry)
        fig, axes = plt.subplots(n_tags, 6, figsize=(18, 2.4 * n_tags))
        if n_tags == 1: axes = axes.reshape(1, -1)
        col_titles = ['LANDFRAC@b', 'pr@b ∈[0,1]',
                      'E_LW ‖act‖·land', 'E_LD ‖act‖·land',
                      'E_OW ‖act‖·ocean', 'E_OD ‖act‖·ocean']
        col_cmaps  = ['Greys', 'Blues', 'YlGn', 'YlOrBr', 'BuPu', 'Purples']
        for r, tag in enumerate(registry):
            info = attn_summary[tag]
            if 'LW_norm_mean' not in info:
                for c in range(6): axes[r, c].axis('off')
                axes[r, 0].text(0.5, 0.5, f'{tag}: no 4-way MoE',
                                ha='center', va='center',
                                transform=axes[r, 0].transAxes)
                continue
            # Hard binary masks at bottleneck resolution
            lf_b = info['landfrac_b']
            land_mask  = (lf_b > 0.5).astype(np.float32)
            ocean_mask = 1.0 - land_mask
            cells = [
                lf_b,                                      # col 0: raw LANDFRAC@b
                info['pr_b_mean'],                         # col 1: pr@b
                info['LW_norm_mean'] * land_mask,          # col 2: LW · land
                info['LD_norm_mean'] * land_mask,          # col 3: LD · land
                info['OW_norm_mean'] * ocean_mask,         # col 4: OW · ocean
                info['OD_norm_mean'] * ocean_mask,         # col 5: OD · ocean
            ]
            for c, m in enumerate(cells):
                kw = {'origin': 'lower', 'cmap': col_cmaps[c], 'aspect': 'auto'}
                if c in (0, 1):     # masks ∈ [0,1]
                    kw['vmin'], kw['vmax'] = 0.0, 1.0
                axes[r, c].imshow(m, **kw)
                if r == 0: axes[r, c].set_title(col_titles[c], fontsize=9)
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, 0].set_ylabel(f'{tag}\n({registry[tag]["co2"]} ppm)' +
                                  (' [OOD]' if registry[tag]['holdout'] else ''), fontsize=8)
        fig.suptitle('Bottleneck 4-Way MoE — ‖E‖ masked by land/ocean for viz '
                     '(time-averaged)', fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig_both(os.path.join(out_dir, '4way_routing.pdf'), dpi=140)
        plt.close()

    @staticmethod
    def _plot_skip_attention_4way(attn_summary, registry, out_dir):
        """
        Skip-conn 4-way attention: 4 columns per enabled level (LW/LD/OW/OD).
        Visualization mask: raw α multiplied by a hard binary (lf>0.5) land/ocean mask,
        which makes the physical regimes clearer.
        - LW/LD only shown on land pixels (lf>0.5)
        - OW/OD only shown on ocean pixels (lf<=0.5)
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        enabled_levels = []
        for lv in (1, 2, 3, 4):
            if any(f'skip{lv}_LW_mean' in attn_summary[t] for t in registry):
                enabled_levels.append(lv)
        if not enabled_levels:
            return

        n_tags = len(registry)
        n_cols = 4 * len(enabled_levels)
        fig, axes = plt.subplots(n_tags, n_cols, figsize=(n_cols * 1.6, n_tags * 1.5))
        if n_tags == 1: axes = axes.reshape(1, -1)
        for r, tag in enumerate(registry):
            info = attn_summary[tag]
            for li, lv in enumerate(enabled_levels):
                # Build land/ocean hard binary masks at this skip level (if available)
                lf = info.get(f'skip{lv}_lf_mean')
                land_mask  = (lf > 0.5).astype(np.float32) if lf is not None else None
                ocean_mask = 1.0 - land_mask if land_mask is not None else None
                for ks, side in enumerate(('LW', 'LD', 'OW', 'OD')):
                    col = 4 * li + ks
                    m = info.get(f'skip{lv}_{side}_mean')
                    if m is None: axes[r, col].axis('off'); continue
                    # Apply viz mask: LW/LD on land, OW/OD on ocean
                    if land_mask is not None:
                        if side in ('LW', 'LD'):
                            m_show = m * land_mask
                        else:    # OW, OD
                            m_show = m * ocean_mask
                    else:
                        m_show = m
                    axes[r, col].imshow(m_show, origin='lower', cmap='inferno',
                                         vmin=0, vmax=1, aspect='auto')
                    if r == 0:
                        axes[r, col].set_title(f'e{lv} α_{side}', fontsize=8)
                    axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
            axes[r, 0].set_ylabel(f'{tag}\n({registry[tag]["co2"]} ppm)' +
                                  (' [OOD]' if registry[tag]['holdout'] else ''), fontsize=8)
        fig.suptitle('Skip 4-Way Attention — raw α masked by land/ocean for viz '
                     '(LW/LD on land, OW/OD on ocean; time-averaged)', fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig_both(os.path.join(out_dir, 'skip_attention_4way.pdf'), dpi=140)
        plt.close()

    @staticmethod
    def _plot_ice_routing(attn_summary, registry, out_dir):
        """
        Ice override @ bottleneck: 7 datasets × 3 cols:
            aice@b (mask)  |  ‖E_ice‖·ice_mask  |  override gate (= aice)
        Visualization mask: ‖E_ice‖ × (aice>0.5) — shown only on ice-covered pixels to
        highlight the physical regime.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        n_tags = len(registry)
        fig, axes = plt.subplots(n_tags, 3, figsize=(12, 2.4 * n_tags))
        if n_tags == 1: axes = axes.reshape(1, -1)
        col_titles = ['aice@b ∈[0,1]',
                      'Ice-expert ‖act‖·ice_mask',
                      'Override gate (= aice mean)']
        col_cmaps  = ['Blues', 'cool', 'Blues']
        for r, tag in enumerate(registry):
            info = attn_summary[tag]
            if 'ice_norm_mean' not in info:
                for c in range(3): axes[r, c].axis('off')
                axes[r, 0].text(0.5, 0.5, f'{tag}: no ice override',
                                ha='center', va='center',
                                transform=axes[r, 0].transAxes)
                continue
            aice_b = info['aice_b_mean']
            ice_mask = (aice_b > 0.5).astype(np.float32)
            cells = [
                aice_b,                                    # col 0: raw aice@b
                info['ice_norm_mean'] * ice_mask,          # col 1: E_ice · ice_mask
                aice_b,                                    # col 2: override gate
            ]
            for c, m in enumerate(cells):
                kw = {'origin': 'lower', 'cmap': col_cmaps[c], 'aspect': 'auto'}
                if c in (0, 2): kw['vmin'], kw['vmax'] = 0.0, 1.0
                axes[r, c].imshow(m, **kw)
                if r == 0: axes[r, c].set_title(col_titles[c], fontsize=9)
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, 0].set_ylabel(f'{tag}\n({registry[tag]["co2"]} ppm)' +
                                  (' [OOD]' if registry[tag]['holdout'] else ''), fontsize=8)
        fig.suptitle('Bottleneck Ice Override — ‖E_ice‖ masked by aice for viz '
                     '(time-averaged)', fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig_both(os.path.join(out_dir, 'ice_routing.pdf'), dpi=140)
        plt.close()

    @staticmethod
    def _plot_skip_attention_ice(attn_summary, registry, out_dir):
        """
        Skip 4-way + ice attention: each enabled level shows attn_ice·ice_mask + aice gate.
        Visualization mask: α_ice × (aice>0.5) — shown only on ice-covered pixels.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        enabled_levels = []
        for lv in (1, 2, 3, 4):
            if any(f'skip{lv}_ice_mean' in attn_summary[t] for t in registry):
                enabled_levels.append(lv)
        if not enabled_levels:
            return

        n_tags = len(registry)
        n_cols = 2 * len(enabled_levels)
        fig, axes = plt.subplots(n_tags, n_cols, figsize=(n_cols * 1.7, n_tags * 1.5))
        if n_tags == 1: axes = axes.reshape(1, -1)
        for r, tag in enumerate(registry):
            info = attn_summary[tag]
            for li, lv in enumerate(enabled_levels):
                col_a, col_g = 2 * li, 2 * li + 1
                m_a = info.get(f'skip{lv}_ice_mean')
                m_g = info.get(f'skip{lv}_aice_mean')
                if m_a is None:
                    axes[r, col_a].axis('off'); axes[r, col_g].axis('off'); continue
                # Apply ice mask to α (hard binary at aice>0.5)
                ice_mask = (m_g > 0.5).astype(np.float32) if m_g is not None else None
                m_a_show = m_a * ice_mask if ice_mask is not None else m_a
                axes[r, col_a].imshow(m_a_show, origin='lower', cmap='cool',
                                       vmin=0, vmax=1, aspect='auto')
                axes[r, col_g].imshow(m_g, origin='lower', cmap='Blues',
                                       vmin=0, vmax=1, aspect='auto')
                if r == 0:
                    axes[r, col_a].set_title(f'e{lv} α_ice·ice', fontsize=8)
                    axes[r, col_g].set_title(f'e{lv} aice gate', fontsize=8)
                axes[r, col_a].set_xticks([]); axes[r, col_a].set_yticks([])
                axes[r, col_g].set_xticks([]); axes[r, col_g].set_yticks([])
            axes[r, 0].set_ylabel(f'{tag}\n({registry[tag]["co2"]} ppm)' +
                                  (' [OOD]' if registry[tag]['holdout'] else ''), fontsize=8)
        fig.suptitle('Skip Ice Override — α_ice masked by aice for viz '
                     '(time-averaged)', fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig_both(os.path.join(out_dir, 'skip_attention_ice.pdf'), dpi=140)
        plt.close()

    @staticmethod
    def _plot_individual_samples_4way(samples, tag, co2, holdout, out_dir):
        """K rows × 14 cols: full 4-way attention path per individual window."""
        import os
        import matplotlib.pyplot as plt
        K = len(samples)
        if K == 0: return
        col_titles = ['LANDFRAC@b', 'pr@b',
                      'Bot E_LW', 'Bot E_LD', 'Bot E_OW', 'Bot E_OD',
                      'skip e3 α_LW', 'skip e3 α_LD', 'skip e3 α_OW', 'skip e3 α_OD',
                      'skip e4 α_LW', 'skip e4 α_LD', 'skip e4 α_OW', 'skip e4 α_OD']
        col_cmaps  = ['Greys', 'Blues',
                      'YlGn', 'YlOrBr', 'BuPu', 'Purples',
                      'inferno','inferno','inferno','inferno',
                      'inferno','inferno','inferno','inferno']
        n_cols = len(col_titles)
        fig, axes = plt.subplots(K, n_cols, figsize=(n_cols * 1.4, K * 1.4))
        if K == 1: axes = axes.reshape(1, -1)
        for r, s in enumerate(samples):
            cells = [s.get('landfrac_b'), s.get('pr_b'),
                     s.get('b_LW_norm'), s.get('b_LD_norm'),
                     s.get('b_OW_norm'), s.get('b_OD_norm'),
                     s.get('skip3_LW'), s.get('skip3_LD'),
                     s.get('skip3_OW'), s.get('skip3_OD'),
                     s.get('skip4_LW'), s.get('skip4_LD'),
                     s.get('skip4_OW'), s.get('skip4_OD')]
            for c, m in enumerate(cells):
                ax = axes[r, c]
                if m is None: ax.axis('off'); continue
                kw = {'origin': 'lower', 'cmap': col_cmaps[c], 'aspect': 'auto'}
                if c in (0, 1):       # LANDFRAC / pr_b
                    kw['vmin'], kw['vmax'] = 0.0, 1.0
                elif c >= 6:          # skip-conn sigmoid attention
                    kw['vmin'], kw['vmax'] = 0.0, 1.0
                ax.imshow(m, **kw)
                if r == 0: ax.set_title(col_titles[c], fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            axes[r, 0].set_ylabel(f'#{s["idx"]}', fontsize=8, rotation=0,
                                  labelpad=18, va='center')
        flag = ' [OOD]' if holdout else ''
        fig.suptitle(f'{tag} ({co2} ppm){flag} — 4-way product routing per sample',
                     fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig_both(os.path.join(out_dir, f'samples_4way_{tag}.pdf'), dpi=120)
        plt.close()

    @staticmethod
    def _plot_individual_samples_4way_e12(samples, tag, co2, holdout, out_dir):
        """
        K rows × 8 (or +2 with ice) cols: per-sample e1 + e2 4-way attention.
        Skips levels whose data is None (e.g. only e1 enabled → 4 cols + maybe ice).
        """
        import os
        import matplotlib.pyplot as plt
        K = len(samples)
        if K == 0: return

        # Detect which sides exist (LW/LD/OW/OD + optional ice) per level
        sides_4way = ('LW', 'LD', 'OW', 'OD')
        has_e1_4way = any(s.get('skip1_LW') is not None for s in samples)
        has_e2_4way = any(s.get('skip2_LW') is not None for s in samples)
        has_e1_ice  = any(s.get('skip1_ice') is not None for s in samples)
        has_e2_ice  = any(s.get('skip2_ice') is not None for s in samples)
        if not (has_e1_4way or has_e2_4way):
            return     # no e1/e2 attention → skip this PDF entirely

        col_titles = []
        col_cmaps  = []
        col_keys   = []   # which key of the sample dict each column corresponds to
        if has_e1_4way:
            for s_ in sides_4way:
                col_titles.append(f'e1 α_{s_}'); col_cmaps.append('inferno')
                col_keys.append(f'skip1_{s_}')
        if has_e1_ice:
            col_titles.append('e1 α_ice'); col_cmaps.append('cool')
            col_keys.append('skip1_ice')
        if has_e2_4way:
            for s_ in sides_4way:
                col_titles.append(f'e2 α_{s_}'); col_cmaps.append('inferno')
                col_keys.append(f'skip2_{s_}')
        if has_e2_ice:
            col_titles.append('e2 α_ice'); col_cmaps.append('cool')
            col_keys.append('skip2_ice')

        n_cols = len(col_titles)
        fig, axes = plt.subplots(K, n_cols, figsize=(n_cols * 1.6, K * 1.5))
        if K == 1:     axes = axes.reshape(1, -1)
        if n_cols == 1: axes = axes.reshape(-1, 1)
        for r, s in enumerate(samples):
            for c, key in enumerate(col_keys):
                ax = axes[r, c]
                m = s.get(key)
                if m is None: ax.axis('off'); continue
                ax.imshow(m, origin='lower', cmap=col_cmaps[c],
                          vmin=0.0, vmax=1.0, aspect='auto')
                if r == 0: ax.set_title(col_titles[c], fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            axes[r, 0].set_ylabel(f'#{s["idx"]}', fontsize=8, rotation=0,
                                  labelpad=18, va='center')
        flag = ' [OOD]' if holdout else ''
        fig.suptitle(f'{tag} ({co2} ppm){flag} — e1/e2 skip attention per sample',
                     fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig_both(os.path.join(out_dir, f'samples_4way_e12_{tag}.pdf'), dpi=120)
        plt.close()

    @staticmethod
    def _plot_individual_samples(samples, tag, co2, holdout, out_dir):
        """K rows × 9 cols figure: full attention path per individual window."""
        import os
        import matplotlib.pyplot as plt
        K = len(samples)
        if K == 0: return
        n_cols = 9
        fig, axes = plt.subplots(K, n_cols, figsize=(n_cols * 1.7, K * 1.6))
        if K == 1: axes = axes.reshape(1, -1)
        col_titles = ['TS (input)', 'd18Op truth', 'd18Op pred',
                      'Bot. Land‖act‖', 'Bot. Ocean‖act‖',
                      'Skip e3 land-α', 'Skip e3 ocean-α',
                      'Skip e4 land-α', 'Skip e4 ocean-α']
        col_cmaps  = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'YlOrBr', 'Blues',
                      'inferno', 'inferno', 'inferno', 'inferno']
        for r, s in enumerate(samples):
            cells = [s['TS_last'], s['truth'], s['pred'],
                     s['b_land_norm'], s['b_ocean_norm'],
                     s['skip3_land'], s['skip3_ocean'],
                     s['skip4_land'], s['skip4_ocean']]
            if cells[1] is not None and cells[2] is not None:
                vmax_d = max(abs(cells[1]).max(), abs(cells[2]).max())
            else:
                vmax_d = None
            for c, m in enumerate(cells):
                ax = axes[r, c]
                if m is None: ax.axis('off'); continue
                kw = {'origin': 'lower', 'cmap': col_cmaps[c], 'aspect': 'auto'}
                if c in (1, 2) and vmax_d is not None:
                    kw['vmin'], kw['vmax'] = -vmax_d, vmax_d
                elif c >= 5:
                    kw['vmin'], kw['vmax'] = 0.0, 1.0
                ax.imshow(m, **kw)
                if r == 0: ax.set_title(col_titles[c], fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            axes[r, 0].set_ylabel(f'#{s["idx"]}', fontsize=8, rotation=0,
                                  labelpad=18, va='center')
        flag = ' [OOD]' if holdout else ''
        fig.suptitle(f'{tag} ({co2} ppm){flag} — {K} individual samples (raw, no avg)', fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig_both(os.path.join(out_dir, f'samples_{tag}.pdf'), dpi=120)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  Smoke test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    B, T, C, H, W = 2, 12, 8, 90, 180

    m = IsoUNetBaseline(n_inputs=C, landfrac_idx=6,
                             base_channels=32, d_state=64, H=H, W=W)
    x = torch.randn(B, T, C, H, W)
    x[:, :, 6] = torch.bernoulli(0.3 * torch.ones(B, T, H, W))  # fake landfrac
    y_seq  = torch.randn(B, T, 1, H, W)
    y_last = y_seq[:, -1]

    y_hat = m(x)
    print(f'Input  : {tuple(x.shape)}  (T-frame seq)')
    print(f'Output : {tuple(y_hat.shape)}  ← (B, 1, H, W) last frame')
    print(f'Params : {sum(p.numel() for p in m.parameters()):,}')

    loss = m._loss(y_hat, y_last)
    print(f'MSE    : {loss.item():.4f}')

    # batch with y_seq (will auto-take last frame)
    loss = m.training_step((x, y_seq), 0)
    loss.backward()
    print(f'train_loss (y_seq mode): {loss.item():.4f}, backward OK')

    # (x, y_seq, co2) compat
    loss2 = m.training_step((x, y_seq, torch.tensor([280., 420.])), 0)
    print(f'(x,y,co2) compat: {loss2.item():.4f}')

    # forward_with_attn
    m.eval()
    _, attn = m.forward_with_attn(x)
    print(f'\nViz dict keys: {list(attn.keys())}')
    for k, v in attn.items():
        print(f'  {k:14s} : {tuple(v.shape) if v is not None else "None"}')
