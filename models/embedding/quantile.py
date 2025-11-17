import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import Embedding, Module, Linear, CrossEntropyLoss
import mlflow
import random

from ..prelude.model import dummyLightning
from ..prelude.data import halfdayData

class _quantile_30min(Dataset, halfdayData):
    def __init__(self, filename):
        super().__init__(filename)
        self.q = np.load('./.results/128th_quantiles_of_1min_ret.npy')
        self.q30 = np.load('./.results/q30.npy')
    
    def __getitem__(self, idx):
        x = self.data[idx * self.seq_len : (idx+1) * self.seq_len]
        y = np.prod(np.lib.stride_tricks.sliding_window_view(x, 30), axis=-1).flatten()
        y = np.searchsorted(self.q30, y)
        x = np.searchsorted(self.q, x)
        return x, y

class quantile_30min(dummyLightning):
    def __init__(self, config, trunk):
        super().__init__(config)
        self.trunk = trunk
        self.train_dataset = _quantile_30min('../data/train.npy')
        self.val_dataset = _quantile_30min('../data/val.npy')
        
        self.emb1 = Embedding(config.vocab_size, config.hidden_size)
        self.emb30 = Embedding(config.vocab_size, config.hidden_size)
        self.readout = Linear(config.hidden_size, config.vocab_size)
    
    def pre_proc(self, x1, x30):
        b = x1.shape[0]
        x1 = self.emb1(x1)
        x30 = self.emb30(x30)
        return x1 + torch.concat((torch.zeros((b, 29, self.config.hidden_size), device=x30.device, dtype=x30.dtype), x30), dim=1)
    
    def forward(self, x1, x30):
        emb = self.pre_proc(x1, x30)
        x = self.trunk(emb)
        return self.readout(x)
    
    def step(self, batch):
        x, y = batch
        y_hat = self(x[:, :-30], y[:, :-30])
        loss = nn.CrossEntropyLoss()(y_hat.view(-1, 128), y[:, 1:].contiguous().view(-1))
        return {
            'loss': loss,
            'logits': y_hat,
        }


if __name__ == '__main__':
    from ..prelude.config import transformerConfig
    from ..prelude.tm import tm
    config = transformerConfig(
        batch_size=1024
    )
    model = quantile_30min(config, tm(config))
    
    model.fit()
    breakpoint()