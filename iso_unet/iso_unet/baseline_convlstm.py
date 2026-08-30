"""
baseline_convlstm.py — ConvLSTM sequence baseline.

Thin wrapper around `convlstm.ConvLSTMSeq` (which already exists as a
Lightning module) that:
  - Accepts the standard iso baseline `__init__` API (n_inputs, H, W,
    seq_len, weights, lr, ...) for consistency with the other baselines.
  - Handles BOTH (x, y) and (x, y, co2) batch formats (the upstream
    ConvLSTMSeq.training_step unpacks (x, y) only, which breaks on
    SequenceDataset which yields the 3-tuple).

forward(x):  (B, T=12, C=9, H=90, W=180) → (B, T=12, 1, H=90, W=180)
loss:        cos(lat) weighted MSE on all T frames
batch:       (x, y) or (x, y, co2) — co2 ignored
output kind: sequence-to-sequence (eval pipeline takes y[:, -1])
"""
from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F

from .convlstm import ConvLSTMSeq


class ConvLSTMBaseline(L.LightningModule):
    """
    Forward(x): (B, T, C, H, W) → (B, T, 1, H, W)
    """
    def __init__(
        self,
        n_inputs:         int = 9,
        out_channels:     int = 1,
        H:                int = 90,
        W:                int = 180,
        seq_len:          int = 12,
        hidden_channels:  int = 16,
        lstm_channels:    int = 16,
        kernel_size:      int = 3,
        weights                = None,
        lr:               float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr      = lr
        self.seq_len = seq_len
        self.H, self.W = H, W
        self.out_channels = out_channels

        if weights is not None:
            self.register_buffer('lat_weights',
                                 torch.as_tensor(weights, dtype=torch.float32))
        else:
            self.lat_weights = None

        # ConvLSTMSeq carries its own forward; we just borrow its inner net
        # by instantiating it and stealing encoder/convlstm/decoder. Passing
        # weights=None so we manage lat_weighting at this level.
        seq_model = ConvLSTMSeq(
            n_inputs=n_inputs,
            hidden_channels=hidden_channels,
            lstm_channels=lstm_channels,
            kernel_size=kernel_size,
            weights=None,
            lr=lr,
        )
        self.encoder  = seq_model.encoder
        self.convlstm = seq_model.convlstm
        self.decoder  = seq_model.decoder

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:                                   # (B, C, H, W) → T=1
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape

        # time-distributed encoder
        x_flat = x.reshape(B * T, C, H, W)
        feat   = self.encoder(x_flat)                       # (B*T, hidden, H/4, W/4)
        feat   = feat.reshape(B, T, *feat.shape[1:])         # (B, T, hidden, H/4, W/4)

        feat   = self.convlstm(feat)                        # (B, T, lstm, H/4, W/4)
        Bt, Tt, Ct, Hp, Wp = feat.shape
        y      = self.decoder(feat.reshape(Bt * Tt, Ct, Hp, Wp))   # (B*T, 1, ., .)
        y      = y.reshape(Bt, Tt, *y.shape[1:])             # (B, T, 1, ., .)

        if y.shape[-2:] != (H, W):
            B_, T_, C_, hP, wP = y.shape
            y = y.reshape(B_ * T_, C_, hP, wP)
            y = F.interpolate(y, size=(H, W), mode='bilinear',
                              align_corners=False)
            y = y.reshape(B_, T_, C_, H, W)
        return y                                            # (B, T, 1, H, W)

    # ── batch helpers ────────────────────────────────────────────────────
    @staticmethod
    def _unpack(batch):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        return x, y

    def _loss(self, y_hat, y):
        # y_hat, y: (B, T, 1, H, W)
        if self.lat_weights is not None:
            loss = F.mse_loss(y_hat, y, reduction='none')
            w = self.lat_weights.view(1, 1, 1, -1, 1)
            return (loss * w).mean()
        return F.mse_loss(y_hat, y)

    # ── Lightning hooks ──────────────────────────────────────────────────
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


if __name__ == '__main__':
    m = ConvLSTMBaseline(n_inputs=9, H=90, W=180, seq_len=12)
    x = torch.randn(2, 12, 9, 90, 180)
    y = m(x)
    n = sum(p.numel() for p in m.parameters())
    print(f'ConvLSTMBaseline  in={tuple(x.shape)} → out={tuple(y.shape)}  '
          f'params={n/1e6:.2f}M')
