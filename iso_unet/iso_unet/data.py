import torch
import lightning as L
from . import utils

class Dataset(L.LightningDataModule):
    def __init__(self, ds, input_vns, output_vns, train_frac=0.7, val_frac=0.1, test_frac=0.2,
                 batch_size=128, num_workers=4,
                 x_norm=None, y_norm=None, x_stats=None, y_stats=None,
                 climo=None, get_anom=True, normalize_mode='global',
                 verbose=False, shuffle_train_set=True):
        super().__init__()
        self.ds = ds.copy()
        self.get_anom = get_anom
        self.climo = climo
        self.normalize_mode = normalize_mode
        self.input_vns = input_vns
        self.output_vns = output_vns
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train_set = shuffle_train_set
        self.x_norm = x_norm
        self.y_norm = y_norm

        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, 'Train, val, and test fractions must sum to 1.'
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac

        # Allow external mean and std
        self.x_stats = x_stats
        self.y_stats = y_stats

        # Settings
        self.verbose = verbose

    def setup(self, stage=None):
        if self.get_anom:
            if self.climo is None:
                self.climo = {}
                for vn in self.input_vns:
                    self.climo[vn] = self.ds[vn].groupby('time.month').mean('time').load()
                for vn in self.output_vns:
                    self.climo[vn] = self.ds[vn].groupby('time.month').mean('time').load()

            for vn in self.input_vns:
                self.ds[vn] = self.ds[vn].groupby('time.month') - self.climo[vn]
            for vn in self.output_vns:
                self.ds[vn] = self.ds[vn].groupby('time.month') - self.climo[vn]

        x = torch.stack([torch.from_numpy(self.ds[vn].values).float() for vn in self.input_vns], dim=1)
        y = torch.stack([torch.from_numpy(self.ds[vn].values).float() for vn in self.output_vns], dim=1)
        # print(torch.isnan(x).any(), torch.isnan(y).any())

        # Weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lats = torch.tensor(self.ds[self.output_vns[0]].lat.values)
        w = torch.cos(torch.deg2rad(lats)).to(device)

        if self.verbose:
            print(f'{x.shape = }')
            print(f'{y.shape = }')
            print(f'{w.shape = }')

        dataset = torch.utils.data.TensorDataset(x, y)

        # save the x, y, and weights
        self.x = x
        self.y = y
        self.w = w

        # Pure chronological split (no randomness):
        #   [   train   |  val  |  test  ]
        #     first 70%     mid 10%   last 20%
        # Pure sequential split -> results are identical across scripts/runs, no RNG state dependency.
        n_total = len(dataset)
        n_test  = int(n_total * self.test_frac)
        n_val   = int(n_total * self.val_frac)
        n_train = n_total - n_test - n_val

        ds_idx     = list(range(n_total))
        train_idx  = ds_idx[:n_train]
        val_idx    = ds_idx[n_train:n_train + n_val]
        test_idx   = ds_idx[n_train + n_val:]

        # Backwards-compatible aliases for legacy references to train_valid_idx / test_idx
        self.train_valid_idx = train_idx + val_idx
        self.test_idx        = test_idx

        self.train_set = torch.utils.data.Subset(dataset, train_idx)
        self.valid_set = torch.utils.data.Subset(dataset, val_idx)
        self.test_set  = torch.utils.data.Subset(dataset, test_idx)

        train_x = torch.stack([self.train_set[i][0] for i in range(len(self.train_set))])
        train_y = torch.stack([self.train_set[i][1] for i in range(len(self.train_set))])

        if self.normalize_mode == 'global':
            reduce_dims = (0, 1, 2)
        elif self.normalize_mode == 'local':
            reduce_dims = (0,)
        else:
            raise ValueError(f'Unknown normalize_mode: {self.normalize_mode}')

        # X stats
        if self.x_stats is None:
            self.x_stats = {}
            for i, vn in enumerate(self.input_vns):
                cfg = self.x_norm[vn]
                xi = train_x[:, i]  # [N, H, W]
                if cfg['type'] == 'standard':
                    self.x_stats[vn] = {
                        'mean': xi.mean(dim=reduce_dims),
                        'std': xi.std(dim=reduce_dims),
                    }
                elif cfg['type'] == 'minmax':
                    self.x_stats[vn] = {
                        'min': cfg.get('min', xi.amin(dim=reduce_dims)),
                        'max': cfg.get('max', xi.amax(dim=reduce_dims)),
                    }
                elif cfg['type'] == 'none':
                    self.x_stats[vn] = {}

        # Y stats
        if self.y_stats is None:
            self.y_stats = {}
            for i, vn in enumerate(self.output_vns):
                cfg = self.y_norm[vn]
                yi = train_y[:, i]
                if cfg['type'] == 'standard':
                    self.y_stats[vn] = {
                        'mean': yi.mean(dim=reduce_dims),
                        'std': yi.std(dim=reduce_dims),
                    }
                elif cfg['type'] == 'minmax':
                    self.y_stats[vn] = {
                        'min': cfg.get('min', yi.amin(dim=reduce_dims)),
                        'max': cfg.get('max', yi.amax(dim=reduce_dims)),
                    }
                elif cfg['type'] == 'none':
                    self.y_stats[vn] = {}

        self.train_set = self.normalize_subset(self.train_set)
        self.valid_set = self.normalize_subset(self.valid_set)
        self.test_set = self.normalize_subset(self.test_set)

    def normalize_subset(self, subset):
        out = []
        for x, y in subset:
            x_list = []
            for i, vn in enumerate(self.input_vns):
                cfg = self.x_norm[vn]
                xi = x[i]
                if cfg['type'] == 'standard':
                    s = self.x_stats[vn]
                    xi = (xi - s['mean']) / s['std']
                elif cfg['type'] == 'minmax':
                    s = self.x_stats[vn]
                    xi = (xi - s['min']) / (s['max'] - s['min'])
                x_list.append(xi)

            y_list = []
            for i, vn in enumerate(self.output_vns):
                cfg = self.y_norm[vn]
                yi = y[i]
                if cfg['type'] == 'standard':
                    s = self.y_stats[vn]
                    yi = (yi - s['mean']) / s['std']
                elif cfg['type'] == 'minmax':
                    s = self.y_stats[vn]
                    yi = (yi - s['min']) / (s['max'] - s['min'])
                y_list.append(yi)

            out.append((torch.stack(x_list), torch.stack(y_list)))

        return out

    def denormalize(self, y_hat, vn):
        cfg = self.y_norm[vn]
        s = self.y_stats[vn]

        if cfg['type'] == 'standard':
            return y_hat * s['std'] + s['mean']

        elif cfg['type'] == 'minmax':
            return y_hat * (s['max'] - s['min']) + s['min']

        return y_hat

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_train_set)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)


# class Dataset(L.LightningDataModule):
#     def __init__(self, ds, input_vns, output_vns, train_frac=0.7, val_frac=0.1, test_frac=0.2,
#                  x_mean=None, x_std=None, y_mean=None, y_std=None, batch_size=128, num_workers=4,
#                  wrap_lon=False, verbose=False, shuffle_train_set=True, normalize_x=True, normalize_y=True):
#         super().__init__()
#         self.ds = ds
#         self.input_vns = input_vns
#         self.output_vns = output_vns
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.shuffle_train_set = shuffle_train_set
#         self.normalize_x = normalize_x
#         self.normalize_y = normalize_y

#         assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, 'Train, val, and test fractions must sum to 1.'
#         self.train_frac = train_frac
#         self.val_frac = val_frac
#         self.test_frac = test_frac

#         # Allow external mean and std
#         self.x_mean = x_mean
#         self.x_std = x_std
#         self.y_mean = y_mean
#         self.y_std = y_std

#         # Settings
#         self.wrap_lon = wrap_lon
#         self.verbose = verbose

#     def setup(self, stage=None):
#         x = torch.stack([torch.from_numpy(self.ds[vn].values).float() for vn in self.input_vns], dim=1)
#         y = torch.stack([torch.from_numpy(self.ds[vn].values).float() for vn in self.output_vns], dim=1)
#         # print(torch.isnan(x).any(), torch.isnan(y).any())

#         # Weights
#         lats = torch.tensor(self.ds[self.output_vns[0]].lat.values)
#         w = torch.cos(torch.deg2rad(lats)).to('cuda')

#         if self.wrap_lon:
#             x = utils.wrap_lon(x)
#             y = utils.wrap_lon(y)

#         if self.verbose:
#             print(f'{x.shape = }')
#             print(f'{y.shape = }')
#             print(f'{w.shape = }')

#         dataset = torch.utils.data.TensorDataset(x, y)

#         # save the x, y, and weights
#         self.x = x
#         self.y = y
#         self.w = w

#         # Compute dataset indices
#         ds_idx = list(range(len(dataset)))
#         test_size = int(len(dataset) * self.test_frac)
#         train_valid_size = len(dataset) - test_size
        
#         # Split train+val and test sets
#         self.train_valid_idx = ds_idx[:train_valid_size]
#         self.test_idx = ds_idx[train_valid_size:]
        
#         train_valid_set = torch.utils.data.Subset(dataset, self.train_valid_idx)
#         self.test_set = torch.utils.data.Subset(dataset, self.test_idx)
        
#         # Further split into train and validation sets
#         train_size = int(self.train_frac / (self.train_frac + self.val_frac) * train_valid_size)
#         val_size = train_valid_size - train_size
#         self.train_set, self.valid_set = torch.utils.data.random_split(train_valid_set, [train_size, val_size])

#         # If mean/std not provided, compute from training set
#         if self.x_mean is None or self.x_std is None:
#             train_x = torch.stack([self.train_set[i][0] for i in range(len(self.train_set))])
#             self.x_mean, self.x_std = train_x.mean(dim=0), train_x.std(dim=0)

#         if self.y_mean is None or self.y_std is None:
#             train_y = torch.stack([self.train_set[i][1] for i in range(len(self.train_set))])
#             self.y_mean, self.y_std = train_y.mean(dim=0), train_y.std(dim=0)

#         # Normalize datasets using specified or computed statistics
#         self.train_set = self.normalize_subset(self.train_set, normalize_x=self.normalize_x, normalize_y=self.normalize_y)
#         self.valid_set = self.normalize_subset(self.valid_set, normalize_x=self.normalize_x, normalize_y=self.normalize_y)
#         self.test_set = self.normalize_subset(self.test_set, normalize_x=self.normalize_x, normalize_y=self.normalize_y)

#     def normalize_subset(self, subset, normalize_x=True, normalize_y=True):
#         normalized_data = []
#         for i in range(len(subset)):
#             x, y = subset[i]
#             if normalize_x: x = (x - self.x_mean) / self.x_std
#             if normalize_y: y = (y - self.y_mean) / self.y_std
#             normalized_data.append((x, y))
#         return normalized_data

#     def denormalize(self, y_hat):
#         """Convert predictions back to original scale."""
#         return y_hat * self.y_std + self.y_mean

#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_train_set)

#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers)

#     def test_dataloader(self):
#         return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)


# ══════════════════════════════════════════════════════════════════════
#  Sequence dataset — sliding window over time
# ══════════════════════════════════════════════════════════════════════

class SequenceDataset(torch.utils.data.Dataset):
    """Sliding-window sequence dataset used by all sequence models (ISO-UNet,
    ConvLSTM, Divided Space-Time, Video Swin, Axial, ...).

    Given per-timestep tensors x (N, C, H, W) and y (N, 1, H, W) from one CO2
    dataset, sample `idx` is the window [idx, idx + seq_len).

    __getitem__(idx) -> (x_window, y_seq, co2_ppm)
        x_window: (T, C, H, W)   T = seq_len input frames
        y_seq:    (T, 1, H, W)   all T target frames (models that predict only
                                 the last frame simply index y_seq[:, -1])
        co2_ppm:  ()             scalar tensor, used by the prompt-alignment loss

    Args:
        x_tensor: (N, C, H, W) input fields, already normalized.
        y_tensor: (N, 1, H, W) target d18Op.
        co2_ppm:  scalar CO2 concentration of the source dataset.
        seq_len:  window length T (default 12, i.e. one model year of monthly data).
    """

    def __init__(self, x_tensor, y_tensor, co2_ppm, seq_len: int = 12):
        assert len(x_tensor) == len(y_tensor), (
            f'length mismatch: x={len(x_tensor)}, y={len(y_tensor)}')
        assert len(x_tensor) >= seq_len, (
            f'not enough frames: {len(x_tensor)} < seq_len={seq_len}')
        self.x   = x_tensor
        self.y   = y_tensor
        self.co2 = torch.tensor(co2_ppm, dtype=torch.float32)
        self.T   = seq_len
        self.n   = len(x_tensor) - seq_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x_win = self.x[idx: idx + self.T]      # (T, C, H, W)
        y_seq = self.y[idx: idx + self.T]      # (T, 1, H, W)
        return x_win, y_seq, self.co2


# Backwards-compatible alias: older scripts imported this as `SequenceDatasetV8`.
SequenceDatasetV8 = SequenceDataset
