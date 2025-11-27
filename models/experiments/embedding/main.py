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
        # y: (N,) or (N, pred_len) for sequence prediction
        self.X = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        # Handle both single target and multi-target cases
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(-1)  # (N,) -> (N, 1)

        self.y = y_tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
