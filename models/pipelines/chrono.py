import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from ..prelude.model import dummyLightning
from ..prelude.data import halfdayData

class _chrono(Dataset, halfdayData):
    def __getitem__(self, idx):
        x = self.data[idx * self.seq_len : (idx+1) * self.seq_len]
        y = np.prod(np.lib.stride_tricks.sliding_window_view(x, 30), axis=-1).flatten()
        return x, y

def quantile_loss(y_hat, y):
    a = F.relu(y_hat - y.unsqueeze(-1))
    b = F.relu(y.unsqueeze(-1) - y_hat)
    
    quantiles = torch.arange(0, 1, 1/(y_hat.shape[-1]+1), device=y_hat.device)[1:]
    quantiles = quantiles[None, None, :]
    return (quantiles * a + (1 - quantiles) * b).mean()

class chrono(dummyLightning):
    def __init__(self, config, trunk):
        super().__init__(config)
        self.trunk = trunk
        self.train_dataset = _chrono('../data/train.npy')
        self.val_dataset = _chrono('../data/val.npy')
        
        self.l1 = nn.Conv1d(config.patch_size, config.intermediate_size, config.patch_size)
        self.l2 = nn.Linear(config.intermediate_size, config.hidden_dim)
        self.s1 = nn.Conv1d(config.patch_size, config.hidden_dim, config.patch_size)
        
        self.l3 = nn.Linear(config.hidden_dim, config.intermediate_size)
        self.l4 = nn.Linear(config.intermediate_size, config.vocab_size * config.patch_size)
        self.s2 = nn.Linear(config.hidden_dim, config.vocab_size * config.patch_size)
        
    def pre_proc(self, x):
        x = x.view(x.shape[0], self.config.patch_size, -1)
        x = self.s1(x).transpose(1, 2) + self.l2(F.relu(self.l1(x).transpose(1, 2)))
        return x
    
    def forward(self, x):
        return self.readout(self.trunk(self.pre_proc(x)))
    
    def readout(self, x):
        x = self.s2(x) + self.l4(F.relu(self.l3(x)))
        x = x.view(x.shape[0], -1, self.config.vocab_size)
        return x
    
    def step(self, batch):
        x, y = batch
        y_hat = self(x[:, :-30])
        loss = quantile_loss(y_hat, y[:, 1:])
        return loss
    
if __name__ == '__main__':
    from ..prelude.model.config import transformerConfig
    from ..prelude.TM import TM
    
    class chronoConfig(transformerConfig):
        patch_size = 1
        vocab_size = 10
    
    config = chronoConfig()
    model = chrono(config, TM(config))
    
    model.fit()