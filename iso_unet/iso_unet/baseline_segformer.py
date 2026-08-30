"""
baseline_segformer.py
─────────────────────────────────────────────────────────────────────────────
SegFormer single-frame baseline (huggingface mit-b0).
Same interface as FNO2d / UNet2D single-frame -> head-to-head with v10 / MCO-242.

Reference: chenqi_isosim/isot/models/segformer.py.

Dependency: pip install 'transformers<4.50' (4.49 is OK on PyTorch 2.3)

I/O (as a Lightning module):
    forward(x):  x (B, C=2, H=90, W=180)  → y_hat (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from transformers import SegformerForSemanticSegmentation


# ══════════════════════════════════════════════════════════════════════
#  SegFormer single-frame wrapper
# ══════════════════════════════════════════════════════════════════════

class SegFormer2D(nn.Module):
    """
    Simple wrapper around HuggingFace SegformerForSemanticSegmentation:
        1. Use nvidia/mit-b0 pretrained encoder (ImageNet)
        2. num_channels=n_inputs, num_labels=1 -> retrain decode_head
        3. After forward, bilinear upsample back to original (H, W)
    """
    def __init__(self, n_inputs: int = 2, num_labels: int = 1, pretrained: str = 'nvidia/mit-b0'):
        super().__init__()
        self.base_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_channels=n_inputs,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return: (B, num_labels, H, W)  — SegFormer downsamples 4x internally, then bilinear upsample back to (H, W)
        """
        B, C, H, W = x.shape
        logits = self.base_model(x).logits                  # (B, num_labels, H/4, W/4)
        return F.interpolate(logits, size=(H, W),
                             mode='bilinear', align_corners=False)


# ══════════════════════════════════════════════════════════════════════
#  Lightning baseline
# ══════════════════════════════════════════════════════════════════════

class SegFormerBaseline(L.LightningModule):
    """
    SegFormer single-frame baseline.
    forward(x):  x (B, C=2, H, W) → y_hat (B, 1, H, W)
    """
    def __init__(
        self,
        n_inputs:   int   = 2,
        num_labels: int   = 1,
        pretrained: str   = 'nvidia/mit-b0',
        weights           = None,
        lr:         float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr

        if weights is not None:
            self.register_buffer('lat_weights', torch.as_tensor(weights, dtype=torch.float32))
        else:
            self.lat_weights = None

        self.seg = SegFormer2D(n_inputs=n_inputs, num_labels=num_labels, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg(x)

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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._unpack(batch)
        loss = self._loss(self(x), y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
    B, C, H, W = 4, 2, 90, 180
    m = SegFormerBaseline(n_inputs=C)

    x = torch.randn(B, C, H, W)
    y = torch.randn(B, 1, H, W)
    y_hat = m(x)

    print(f'Input  x   : {tuple(x.shape)}')
    print(f'Output y_hat: {tuple(y_hat.shape)}    ← (B, 1, H, W) ✓')
    print(f'Params      : {sum(p.numel() for p in m.parameters()):,}')
    loss = m._loss(y_hat, y)
    print(f'MSE loss    : {loss.item():.4f}')
