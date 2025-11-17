import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import Embedding, Module, Linear, CrossEntropyLoss
import mlflow
import random

from ..prelude.model import dummyLightning

class quantile_1min(Dataset):
    def __init__(self, config, filename='../data/train.npy'):
        super().__init__()
        self.data = np.load(filename)
        self.q = np.load('./.results/128th_quantiles_of_1min_ret.npy')
        self.seq_len=config.seq_len
        assert self.data.shape[0] % self.seq_len == 0

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx * self.seq_len : (idx+1) * self.seq_len]
        x = np.searchsorted(self.q, x)
        return x

class _quantile_30min(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.data = np.load(filename) if isinstance(filename, str) else filename
        self.q = np.load('./.results/128th_quantiles_of_1min_ret.npy')
        self.q30 = np.load('./.results/q30.npy')
    
    def __len__(self):
        return len(self.data) // 119

    def __getitem__(self, idx):
        x = self.data[idx * 119 : (idx+1) * 119]
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
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        
        self.emb1 = Embedding(config.vocab_size, config.hidden_size)
        self.emb30 = Embedding(config.vocab_size, config.hidden_size)
        self.readout = Linear(config.hidden_size, config.vocab_size)
        
    
    def training_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    
    def validation_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def forward(self, x1, x30):
        b = x1.shape[0]
        x1 = self.emb1(x1)
        x30 = self.emb30(x30)
        return x1 + torch.concat((torch.zeros((b, 29, self.config.hidden_size), device=x30.device, dtype=x30.dtype), x30), dim=1)
    def step(self, batch):
        x, y = batch
        emb = self(x[:, :-30], y[:, :-30])
        x = self.trunk(emb)
        y_hat = self.readout(x)
        loss = nn.CrossEntropyLoss()(y_hat.view(-1, 128), y[:, 1:].contiguous().view(-1))
        return {
            'loss': loss,
            'logits': y_hat,
        }
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr)
    

if __name__ == '__main__':
    from ..prelude.config import transformerConfig
    from ..tm import tm
    config = transformerConfig()
    model = quantile_30min(config, tm(config))
    
    mlflow.set_experiment("quantile_30min")
    model.fit()
    breakpoint()