import torch
import lightning as L
from . import utils

class SpectralConv2d(L.LightningModule):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = torch.nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = torch.nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(L.LightningModule):
    def __init__(self, n_inputs, modes1, modes2, width, weights=None):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.weights = weights
        # self.unwrap_lon = unwrap_lon

        # self.norm = torch.nn.InstanceNorm2d(n_inputs)
        # self.norm = torch.nn.BatchNorm2d(n_inputs)
        self.fc0 = torch.nn.Linear(n_inputs, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w1 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w2 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w3 = torch.nn.Conv2d(self.width, self.width, 1)

        self.fc1 = torch.nn.Linear(self.width, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        # x = self.norm(x)
        x = x.permute(0, 2, 3, 1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = torch.nn.functional.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = torch.nn.functional.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = torch.nn.functional.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)

        x = x.permute(0, 3, 1, 2)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        # if self.unwrap_lon:
        #     y = utils.unwrap_lon(y)
        #     y_hat = utils.unwrap_lon(y_hat)

        if self.weights is not None:
            loss = torch.nn.functional.mse_loss(y_hat, y, reduction='none')
            loss = (loss*self.weights.view(1,1,-1,1)).mean()
        else:
            loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        # if self.unwrap_lon:
        #     y = utils.unwrap_lon(y)
        #     y_hat = utils.unwrap_lon(y_hat)

        if self.weights is not None:
            loss = torch.nn.functional.mse_loss(y_hat, y, reduction='none')
            loss = (loss*self.weights.view(1,1,-1,1)).mean()
        else:
            loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        # if self.unwrap_lon:
        #     y = utils.unwrap_lon(y)
        #     y_hat = utils.unwrap_lon(y_hat)

        if self.weights is not None:
            loss = torch.nn.functional.mse_loss(y_hat, y, reduction='none')
            loss = (loss*self.weights.view(1,1,-1,1)).mean()
        else:
            loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log('test_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class LinearReg2d(L.LightningModule):
    def __init__(self, n_inputs, weights=None):
        super().__init__()
        self.fc = torch.nn.Linear(n_inputs, 1)
        self.weights = weights

    def forward(self, x):
        batch = x.shape[0]
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        y_flat = self.fc(x_flat)
        y = y_flat.reshape(batch, x.shape[2], x.shape[3], 1).permute(0, 3, 1, 2)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)

        if self.weights is not None:
            loss = torch.nn.functional.mse_loss(y_hat, y, reduction='none')
            loss = (loss*self.weights.view(1,1,-1,1)).mean()
        else:
            loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)

        if self.weights is not None:
            loss = torch.nn.functional.mse_loss(y_hat, y, reduction='none')
            loss = (loss*self.weights.view(1,1,-1,1)).mean()
        else:
            loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)

        if self.weights is not None:
            loss = torch.nn.functional.mse_loss(y_hat, y, reduction='none')
            loss = (loss*self.weights.view(1,1,-1,1)).mean()
        else:
            loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log('test_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

class LinearReg2dPerGrid(L.LightningModule):
    def __init__(self, n_inputs, H, W, weights=None):
        super().__init__()

        # One set of parameters per grid cell
        self.weight = torch.nn.Parameter(torch.zeros(n_inputs, H, W))
        self.bias   = torch.nn.Parameter(torch.zeros(1, H, W))

        self.weights = weights

    def forward(self, x):
        # x: (B, C, H, W)

        y = (x * self.weight.unsqueeze(0)).sum(dim=1, keepdim=True)
        y = y + self.bias.unsqueeze(0)

        return y

    def _loss(self, y_hat, y):
        if self.weights is not None:
            loss = torch.nn.functional.mse_loss(y_hat, y, reduction='none')
            loss = (loss * self.weights.view(1,1,-1,1)).mean()
        else:
            loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        loss = self._loss(self(x), y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        loss = self._loss(self(x), y)
        self.log('valid_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        loss = self._loss(self(x), y)
        self.log('test_loss', loss, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


########################### Below is the DisentangledTransformer model, with physics constraints and interpretability outputs ###########################
"""
iso_m_v1.py  —  Disentangled Spatiotemporal Attribution (DSTA)
Patch-based Transformer + three-path interpretability decomposition.

Usage:
  model   = DisentangledTransformer(n_inputs=len(x_norm.keys()), H=90, W=180, weights=data['1.5x'].w)
  trainer = ig.Trainer(name='IsoGen-dsta')
  trainer.fit(model, data['1.5x'])

patch_size speed reference (H=90, W=180):
  patch_size=10 ->  9x18 =  162 tokens  * default, fastest
  patch_size=6  -> 15x30 =  450 tokens    better accuracy
  patch_size=1  -> 90x180= 16200 tokens   slowest, not recommended
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


# ══════════════════════════════════════════════════════════════════
#  Path 1: Linear interpretable module
# ══════════════════════════════════════════════════════════════════

class FeatureInteractionModule(nn.Module):
    """
    Each grid point independently learns linear coefficients for the variables
    plus a bilinear interaction term. The coefficient map (H, W) can be
    visualized directly and has clear physical meaning.

    Physical constraints (enforced by the loss):
      weights[:,0] > 0  -> TS coefficient is positive (higher temperature -> larger d18O)
      interact_weight sparse -> only activated in regions of genuine synergy
    """
    def __init__(self, n_inputs: int, H: int, W: int):
        super().__init__()
        self.weights         = nn.Parameter(torch.zeros(1, n_inputs, H, W))
        self.interact_weight = nn.Parameter(torch.zeros(1, 1, H, W))
        nn.init.normal_(self.weights, std=0.01)

    def forward(self, x):
        # x: (B, C, H, W)
        linear_contrib = (self.weights * x).sum(dim=1, keepdim=True)  # (B,1,H,W)

        if x.shape[1] >= 2:
            interact_contrib = self.interact_weight * x[:, 0:1] * x[:, 1:2]
        else:
            interact_contrib = torch.zeros_like(linear_contrib)

        out = linear_contrib + interact_contrib

        return out, {
            'weights_map'    : self.weights.squeeze(0).detach(),        # (C,H,W)
            'interact_weight': self.interact_weight.squeeze().detach(), # (H,W)
        }


# ══════════════════════════════════════════════════════════════════
#  Path 2: Patch-based Transformer backbone
# ══════════════════════════════════════════════════════════════════

class PatchGeoPositionalEncoding(nn.Module):
    """Each patch uses its row/column index as a geographic positional encoding."""
    def __init__(self, d_model: int, n_patch_h: int, n_patch_w: int):
        super().__init__()
        self.lat_embed = nn.Embedding(n_patch_h, d_model // 2)
        self.lon_embed = nn.Embedding(n_patch_w, d_model // 2)

        lat_idx = torch.arange(n_patch_h).unsqueeze(1).expand(n_patch_h, n_patch_w).reshape(-1)
        lon_idx = torch.arange(n_patch_w).unsqueeze(0).expand(n_patch_h, n_patch_w).reshape(-1)
        self.register_buffer('lat_idx', lat_idx)
        self.register_buffer('lon_idx', lon_idx)

    def forward(self, x):
        # x: (B, n_patches, d_model)
        geo_e = torch.cat([
            self.lat_embed(self.lat_idx),
            self.lon_embed(self.lon_idx),
        ], dim=-1).unsqueeze(0)          # (1, n_patches, d_model)
        return x + geo_e


class GeoSpatialTransformer(nn.Module):
    """
    Splits the spatial grid into patches, each patch becomes a token.
    This drastically reduces the number of tokens and accelerates the Transformer.

    patch_size=10: 90x180 -> 9x18 = 162 tokens (~100x faster than per-grid-cell)
    """
    def __init__(self, n_inputs, d_model, nhead, num_layers, H, W,
                 patch_size=10, dropout=0.1):
        super().__init__()
        assert H % patch_size == 0, f"H={H} is not divisible by patch_size={patch_size}"
        assert W % patch_size == 0, f"W={W} is not divisible by patch_size={patch_size}"

        self.H          = H
        self.W          = W
        self.d_model    = d_model
        self.patch_size = patch_size
        self.n_patch_h  = H // patch_size
        self.n_patch_w  = W // patch_size

        patch_dim = n_inputs * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.geo_pe      = PatchGeoPositionalEncoding(d_model, self.n_patch_h, self.n_patch_w)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN, training is more stable
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj  = nn.Linear(d_model, d_model * patch_size * patch_size)
        self._attn_weights = None 

    # def forward(self, x):
    #     B, C, H, W = x.shape
    #     P  = self.patch_size
    #     Ph = self.n_patch_h
    #     Pw = self.n_patch_w

    #     # Split into patches: (B,C,H,W) -> (B, Ph*Pw, C*P*P)
    #     x_p = x.reshape(B, C, Ph, P, Pw, P)
    #     x_p = x_p.permute(0, 2, 4, 1, 3, 5).reshape(B, Ph * Pw, C * P * P)

    #     # Patch embedding + positional encoding + Transformer
    #     tokens = self.patch_embed(x_p)     # (B, n_patches, d_model)
    #     tokens = self.geo_pe(tokens)
    #     tokens = self.transformer(tokens)

    #     # Restore spatial dims: (B, n_patches, d_model*P*P) -> (B, d_model, H, W)
    #     tokens = self.output_proj(tokens)                              # (B, Ph*Pw, d_model*P*P)
    #     tokens = tokens.reshape(B, Ph, Pw, self.d_model, P, P)
    #     out    = tokens.permute(0, 3, 1, 4, 2, 5).reshape(B, self.d_model, H, W)

    #     return out   # (B, d_model, H, W)

    def forward(self, x):
        # Patch-splitting and embedding code unchanged...
        B, C, H, W = x.shape
        P  = self.patch_size
        Ph = self.n_patch_h
        Pw = self.n_patch_w

        x_p = x.reshape(B, C, Ph, P, Pw, P)
        x_p = x_p.permute(0, 2, 4, 1, 3, 5).reshape(B, Ph * Pw, C * P * P)
        tokens = self.patch_embed(x_p)
        tokens = self.geo_pe(tokens)

        # ── Change here: run layer by layer, also grab the last layer's attention weights ──
        for i, layer in enumerate(self.transformer.layers):
            if i == len(self.transformer.layers) - 1:
                # Last layer: call manually to grab attn_weights
                tokens, attn = layer.self_attn(
                    tokens, tokens, tokens,
                    need_weights=True,
                    average_attn_weights=True,   # averaged across heads
                )
                tokens = layer.norm1(tokens)
                tokens = tokens + layer.dropout(layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(tokens)))
                ))
                tokens = layer.norm2(tokens)
                self._attn_weights = attn.detach()  # (B, n_patches, n_patches)
            else:
                tokens = layer(tokens)

        tokens = self.output_proj(tokens)
        tokens = tokens.reshape(B, Ph, Pw, self.d_model, P, P)
        out    = tokens.permute(0, 3, 1, 4, 2, 5).reshape(B, self.d_model, H, W)
        return out

    def get_attention_map(self, query_patch_idx):
        """
        query_patch_idx: index of a patch.
        Returns the weights with which that patch attends to all patches globally,
        shape (n_patch_h, n_patch_w).
        """
        # _attn_weights: (B, n_patches, n_patches)
        attn = self._attn_weights.mean(0)           # batch-averaged (n_patches, n_patches)
        attn_row = attn[query_patch_idx]             # (n_patches,) this patch's attention over all patches
        return attn_row.reshape(self.n_patch_h, self.n_patch_w)  # (Ph, Pw)



# ══════════════════════════════════════════════════════════════════
#  Path 3: Local vs nonlocal decomposition
# ══════════════════════════════════════════════════════════════════

class SpatialLocalNonlocalModule(nn.Module):
    """
    alpha(H,W): near 1 -> dominated by local topography / surface
                near 0 -> dominated by large-scale teleconnections
    """
    def __init__(self, channels: int, H: int, W: int):
        super().__init__()
        self.local_branch    = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.nonlocal_conv   = nn.Conv2d(channels, channels, kernel_size=1)
        self.nonlocal_expand = nn.Conv2d(channels, channels, kernel_size=1)
        self.alpha_raw       = nn.Parameter(torch.zeros(1, 1, H, W))

    def forward(self, x):
        local_feat    = self.local_branch(x)
        nonlocal_feat = self.nonlocal_conv(x)
        nonlocal_feat = F.adaptive_avg_pool2d(nonlocal_feat, 1)   # global pool
        nonlocal_feat = self.nonlocal_expand(nonlocal_feat).expand_as(local_feat)

        alpha = torch.sigmoid(self.alpha_raw)                      # (1,1,H,W)
        out   = alpha * local_feat + (1 - alpha) * nonlocal_feat
        return out, alpha.squeeze().detach()                       # (H,W)


# ══════════════════════════════════════════════════════════════════
#  Main model
# ══════════════════════════════════════════════════════════════════

class DisentangledTransformer(L.LightningModule):
    """
    Usage:
      model = DisentangledTransformer(
          n_inputs=len(x_norm.keys()), H=90, W=180, weights=data['1.5x'].w,
          d_model=64, nhead=4, num_layers=2, patch_size=10,
      )
      trainer = ig.Trainer(name='IsoGen-dsta')
      trainer.fit(model, data['1.5x'])
    """
    def __init__(
        self,
        n_inputs     : int,
        H            : int   = 90,
        W            : int   = 180,
        weights              = None,
        d_model      : int   = 64,
        nhead        : int   = 4,
        num_layers   : int   = 2,
        patch_size   : int   = 10,
        dropout      : float = 0.1,
        lambda_mono  : float = 0.1,
        lambda_sparse: float = 0.01,
        lr           : float = 1e-4,
        use_path1    : bool  = True,
        use_path2    : bool  = True,
        use_path3    : bool  = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.H             = H
        self.W             = W
        self.lr            = lr
        self.lambda_mono   = lambda_mono
        self.lambda_sparse = lambda_sparse
        self.use_path1     = use_path1
        self.use_path2     = use_path2
        self.use_path3     = use_path3

        assert use_path1 or use_path2, "at least one of path1 or path2 must be enabled"

        # Latitude weighting (no gradient)
        w = torch.as_tensor(weights, dtype=torch.float32) if weights is not None else torch.ones(H)
        self.register_buffer('lat_weights', w)

        if use_path1:
            self.feature_interaction = FeatureInteractionModule(n_inputs, H, W)

        if use_path2:
            self.transformer_backbone = GeoSpatialTransformer(
                n_inputs, d_model, nhead, num_layers, H, W, patch_size, dropout)
            if use_path3:
                self.spatial_decomp = SpatialLocalNonlocalModule(d_model, H, W)
            self.head = nn.Sequential(
                nn.Conv2d(d_model, 32, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(32, 1, kernel_size=1),
            )

        self._interp = {}

    def forward(self, x):
        y_hat = 0

        # Path 1: linear decomposition
        if self.use_path1:
            feat_linear, feat_dict = self.feature_interaction(x)    # (B,1,H,W)
            y_hat = y_hat + feat_linear
            self._interp.update({
                'weights_map'    : feat_dict['weights_map'],
                'interact_weight': feat_dict['interact_weight'],
            })
            if x.shape[1] >= 1:
                self._interp['ts_weight'] = feat_dict['weights_map'][0]
            if x.shape[1] >= 2:
                self._interp['prect_weight'] = feat_dict['weights_map'][1]

        # Path 2: Patch Transformer
        if self.use_path2:
            x_tr = self.transformer_backbone(x)                     # (B,d_model,H,W)

            # Path 3: local/nonlocal decomposition
            if self.use_path3:
                x_decomp, alpha = self.spatial_decomp(x_tr)
                self._interp['alpha'] = alpha
            else:
                x_decomp = x_tr

            y_hat = y_hat + self.head(x_decomp)

        return y_hat

    # ── Loss ─────────────────────────────────────────────────────

    def _weighted_mse(self, pred, target):
        loss = F.mse_loss(pred, target, reduction='none')        # (B,1,H,W)
        w    = self.lat_weights.view(1, 1, -1, 1)
        return (loss * w).mean()

    def _constraint_loss(self):
        if not self.use_path1:
            return 0.0
        mono_loss   = F.relu(-self.feature_interaction.weights[:, 0:1]).mean()
        sparse_loss = self.feature_interaction.interact_weight.abs().mean()
        return self.lambda_mono * mono_loss + self.lambda_sparse * sparse_loss

    def _total_loss(self, pred, target):
        pred_loss = self._weighted_mse(pred, target)
        return pred_loss + self._constraint_loss(), pred_loss

    # ── Lightning steps ──────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        total, pred = self._total_loss(self(x), y)
        self.log('train_loss',       pred,  on_step=True,  on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log('train_total_loss', total, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return total

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        _, pred = self._total_loss(self(x), y)
        self.log('valid_loss', pred, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        _, pred = self._total_loss(self(x), y)
        self.log('test_loss', pred, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)
        return {'optimizer': opt, 'lr_scheduler': sch}

    # ── Interpretability interface ───────────────────────────────

    def get_interpretability_maps(self) -> dict:
        """
        After trainer.fit() finishes, run forward once on any batch, then call this function.

        Returns:
          ts_weight      (H,W)  temperature effect strength
          prect_weight   (H,W)  precipitation effect strength
          interact_weight(H,W)  TS x PRECT synergy region
          alpha          (H,W)  local (1) vs teleconnection (0)
          weights_map    (C,H,W) coefficients for all variables
        """
        return {k: v.cpu() for k, v in self._interp.items() if isinstance(v, torch.Tensor)}