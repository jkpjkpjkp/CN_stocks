import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import Embedding, Module, Linear, CrossEntropyLoss
import mlflow
import random
import polars as pl
from datetime import date, time, datetime
from ..prelude.model import dummyLightning

class chrono(dummyLightning):
    def __init__(self, config, trunk):
        super().__init__(config)
        self.trunk = trunk
        df = pl.scan_parquet('../data/cleaned_1min.pq')
        train = df.filter(pl.col.datetime.dt.date() < date(2023, 1, 1))
        val = df.filter(pl.col.datetime.dt.date() >= date(2023, 1, 1)
               ).filter(pl.col.datetime.dt.date() < date(2024, 1, 1))