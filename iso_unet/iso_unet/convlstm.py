import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class ConvLSTM2dCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, state):
        h_prev, c_prev = state
        gates = self.conv(torch.cat([x, h_prev], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch, H, W, device, dtype):
        h = torch.zeros(batch, self.hidden_channels, H, W, device=device, dtype=dtype)
        c = torch.zeros(batch, self.hidden_channels, H, W, device=device, dtype=dtype)
        return h, c


class ConvLSTM2d(nn.Module):
    """Single-layer ConvLSTM over a (B, T, C, H, W) sequence."""
    def __init__(self, in_channels, hidden_channels, kernel_size=3, return_sequences=False, bias=True):
        super().__init__()
        self.cell = ConvLSTM2dCell(in_channels, hidden_channels, kernel_size, bias)
        self.return_sequences = return_sequences

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, _, H, W = x.shape
        h, c = self.cell.init_state(B, H, W, x.device, x.dtype)
        outs = []
        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
            if self.return_sequences:
                outs.append(h)
        if self.return_sequences:
            return torch.stack(outs, dim=1)
        return h


def _time_distributed(layer, x):
    """Apply a per-frame layer to (B, T, C, H, W) input."""
    B, T = x.shape[0], x.shape[1]
    y = layer(x.reshape(B * T, *x.shape[2:]))
    return y.reshape(B, T, *y.shape[1:])


class ConvLSTM(L.LightningModule):
    """
    ConvLSTM-based emulator.

    Forward accepts either:
      - (B, C, H, W)        : single frame, treated as T=1 (drop-in for the
                              standard iso_unet Dataset interface).
      - (B, T, C, H, W)     : sequence of T frames (use a dataset that yields
                              temporal windows to actually leverage recurrence).

    Output: (B, 1, H, W)
    """
    def __init__(self, n_inputs, hidden_channels=16, lstm_channels=16,
                 kernel_size=3, weights=None, lr=1e-3):
        super().__init__()
        self.weights = weights
        self.lr = lr

        self.encoder = nn.Sequential(
            nn.Conv2d(n_inputs, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.convlstm = ConvLSTM2d(
            in_channels=hidden_channels,
            hidden_channels=lstm_channels,
            kernel_size=kernel_size,
            return_sequences=False,
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(lstm_channels, hidden_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, C, H, W)

        B, T, C, H, W = x.shape

        feat = _time_distributed(self.encoder, x)         # (B, T, hidden, H/4, W/4)
        feat = self.convlstm(feat)                        # (B, lstm, H/4, W/4)
        y = self.decoder(feat)                            # (B, 1, H', W')

        if y.shape[-2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
        return y

    def _loss(self, y_hat, y):
        if self.weights is not None:
            loss = F.mse_loss(y_hat, y, reduction='none')
            loss = (loss * self.weights.view(1, 1, -1, 1)).mean()
        else:
            loss = F.mse_loss(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._loss(self(x), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self._loss(self(x), y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self._loss(self(x), y)
        self.log('test_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ConvLSTMSeq(L.LightningModule):
    """
    ConvLSTM 12->12 sequence-out variant (same dense supervision setup as the v8 transformer).

    Forward: (B, T, C, H, W) -> (B, T, 1, H, W)   * outputs all T frames

    Loss: 5D MSE, T frames weighted equally.

    Batch format: (x, y_seq) or (x, y_seq, co2_ppm) both accepted (co2 is ignored because
    ConvLSTM has no prompt mechanism. This lets us feed SequenceDataset directly,
    aligned with the v8 experiments.)

    Differences from the original ConvLSTM:
        * convlstm layer uses return_sequences=True
        * decoder is applied time-distributed to each of the T frames
        * loss is computed over (B, T, 1, H, W)
    """
    def __init__(self, n_inputs, hidden_channels=16, lstm_channels=16,
                 kernel_size=3, weights=None, lr=1e-3):
        super().__init__()
        self.weights = weights
        self.lr      = lr

        self.encoder = nn.Sequential(
            nn.Conv2d(n_inputs, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.convlstm = ConvLSTM2d(
            in_channels=hidden_channels,
            hidden_channels=lstm_channels,
            kernel_size=kernel_size,
            return_sequences=True,        # * key: return hidden state at every frame
        )

        # Same decoder architecture as single-frame ConvLSTM, just applied time-distributed in forward
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(lstm_channels, hidden_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W) or (B, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)              # (B, 1, C, H, W)
        B, T, C, H, W = x.shape

        feat = _time_distributed(self.encoder, x)        # (B, T, hidden, H/4, W/4)
        feat = self.convlstm(feat)                        # (B, T, lstm, H/4, W/4)
        y    = _time_distributed(self.decoder, feat)     # (B, T, 1, H', W')

        # ConvTranspose may not exactly recover (H, W) — interpolate each frame independently
        if y.shape[-2:] != (H, W):
            B_, T_, C_, hP, wP = y.shape
            y = y.reshape(B_ * T_, C_, hP, wP)
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            y = y.reshape(B_, T_, C_, H, W)
        return y

    def _loss(self, y_hat, y):
        # y_hat, y: (B, T, 1, H, W)
        if self.weights is not None:
            loss = F.mse_loss(y_hat, y, reduction='none')
            loss = (loss * self.weights.view(1, 1, 1, -1, 1)).mean()
        else:
            loss = F.mse_loss(y_hat, y)
        return loss

    def _unpack(self, batch):
        """Supports both (x, y) and (x, y, co2) batch formats. co2 will be ignored."""
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        return x, y

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
