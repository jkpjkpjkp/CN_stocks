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
