from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from models.prelude.model import dummyLightning


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, seq_len, feat)
        # y: (N,)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class omni(dummyLightning):
    """Simple omni experiment runner.

    - `prepare_data(path)` reads a parquet of OHLCV with columns
      ['id', 'minute', 'open','high','low','close','volume'] (common schema)
    - builds sequences of length `seq_len` and a next-step regression target
    - provides `train_dataset` and `val_dataset` attributes

    The model implements a small MLP encoder and a regression head and
    implements `step` to return {'loss': tensor} as expected by
    `dummyLightning`.
    """

    def __init__(self, config):
        super().__init__(config)
        # small encoder; will be recompiled/converted in `activate`
        self.encoder: Optional[nn.Module] = None

    def prepare_data(self, path: Optional[str] = None, seq_len: int = 64, quant_bins: int = 256, train_frac: float = 0.9):
        # normalize path input
        if path is None:
            path = Path.home() / 'h' / 'data' / 'a_1min.pq'
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"parquet file not found: {path}")

        df = pl.read_parquet(str(path))

        # Expect at least: id, minute, open, high, low, close, volume
        # Sort and compute features
        df = df.sort(['id', 'minute'])

        # We'll compute per-row features: close (raw), zscore (per-id), return_1, return_30
        # first compute close shift features per id
        df = df.with_columns([
            pl.col('close').alias('close'),
            pl.col('volume').alias('volume')
        ])

        # compute returns
        df = df.with_columns([
            (pl.col('close') - pl.col('close').shift(1)).alias('ret_1'),
            (pl.col('close') - pl.col('close').shift(30)).alias('ret_30')
        ])

        # fillna for returns (beginning of each series)
        df = df.with_columns([
            pl.col('ret_1').fill_null(0.0),
            pl.col('ret_30').fill_null(0.0)
        ])

        # split train/val by timestamp percentile per id to avoid leakage
        # compute per-id cutoff minute
        cutoffs = (
            df.group_by('id')
              .agg(pl.col('minute').quantile(train_frac).alias('cut'))
        )

        df = df.join(cutoffs, on='id')
        df = df.with_columns((pl.col('minute') <= pl.col('cut')).alias('is_train'))

        # per-stock mean/std on training portion for zscore
        stats = (
            df.filter(pl.col('is_train'))
              .group_by('id')
              .agg([
                  pl.col('close').mean().alias('mean_close'),
                  pl.col('close').std().alias('std_close')
              ])
        )

        df = df.join(stats, on='id')
        df = df.with_columns(((pl.col('close') - pl.col('mean_close')) / (pl.col('std_close') + 1e-6)).fill_null(0.0).alias('z_close'))

        # Quantize global percentiles using training close distribution
        train_close = df.filter(pl.col('is_train'))['close'].to_numpy()
        quantiles = np.quantile(train_close, np.linspace(0, 1, quant_bins + 1))

        def quantize_array(arr: np.ndarray) -> np.ndarray:
            # returns int tokens in [0, quant_bins-1]
            tokens = np.digitize(arr, quantiles[1:-1], right=True)
            return tokens.astype(np.int64)

        # Build sliding windows per id
        seqs = []
        targets = []
        is_trains = []

        # iterate ids to build sliding windows
        for _id in df['id'].unique().to_list():
            sub = df.filter(pl.col('id') == _id).sort('minute')
            close = np.asarray(sub['close'])
            z_close = np.asarray(sub['z_close'])
            ret1 = np.asarray(sub['ret_1'])
            ret30 = np.asarray(sub['ret_30'])

            n = len(close)
            if n <= seq_len:
                continue

            # quantized tokens for close (unused here but kept for future)
            q_close = quantize_array(close)

            for i in range(n - seq_len - 1):
                seq_close = close[i:i+seq_len]
                seq_z = z_close[i:i+seq_len]
                seq_ret1 = ret1[i:i+seq_len]
                seq_ret30 = ret30[i:i+seq_len]
                # features: raw close, z_close, ret1, ret30
                feat = np.stack([seq_close, seq_z, seq_ret1, seq_ret30], axis=-1)
                target = z_close[i+seq_len]  # next-step z-scored close
                seqs.append(feat)
                targets.append(target)
                # mark train/val by the last index
                is_trains.append(bool(sub['is_train'][i+seq_len]))

        if len(seqs) == 0:
            raise ValueError('no sequences generated from data - check seq_len and data length')

        X = np.stack(seqs)  # (N, seq_len, feat)
        y = np.array(targets, dtype=np.float32)
        is_trains = np.array(is_trains, dtype=bool)
        mask = is_trains

        X_train, y_train = X[mask], y[mask]
        X_val, y_val = X[np.logical_not(mask)], y[np.logical_not(mask)]

        self.train_dataset = TimeSeriesDataset(X_train, y_train)
        self.val_dataset = TimeSeriesDataset(X_val, y_val)

        # build a tiny encoder if not present
        feat_dim = X.shape[-1]
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        if self.encoder is None:
            raise RuntimeError('encoder not initialized; call prepare_data first')
        return self.encoder(x)

    def step(self, batch):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        return {'loss': loss}
