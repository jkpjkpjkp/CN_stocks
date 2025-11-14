import torch
from torch import nn
from torch.nn import functional as F, Module
from torch.utils.data import Dataset
from lightning import LightningModule, LightningDataModule as DataModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import Callback, RichProgressBar, ModelCheckpoint
from lightning.pytorch.utilities import grad_norm
import numpy as np
from einops import rearrange
import random
import matplotlib.pyplot as plt
from transformers import PreTrainedModel, PreTrainedConfig

class tickConfig(PreTrainedConfig):
    model_type = 'tick',
    vocab_size = 1024,
    hidden_size = 512,
    num_layers = 8,
    num_heads = 8,
    device = 'cuda',


class mha(Module):
    def __init__(self, config: tickConfig, cos, sin):
        super().__init__()
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.num_heads = config.num_heads
        self.cos, self.sin = cos, sin
        self.config = config
    
    def apply_rotary_emb(self, x, cos, sin):
        l = x.shape[1]
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
        y1 = x1 * cos[:l] + x2 * sin[:l] # rotate pairs of dims
        y2 = x1 * (-sin[:l]) + x2 * cos[:l]
        out = torch.cat([y1, y2], -1) # re-assemble
        assert out.shape[2:] == (self.config.hidden_size,)
        out = out.to(x.dtype)
        return out

    def forward(self, x):
        q, k, v = torch.chunk(self.qkv(x), 3, -1)
        q = self.apply_rotary_emb(q, self.cos, self.sin)
        k = self.apply_rotary_emb(k, self.cos, self.sin)
        b = x.shape[0]
        
        shape = (b, -1, 32, 4)
        q = rearrange(q.view(*shape), 'b l d h -> b h l d')
        k = rearrange(k.view(*shape), 'b l d h -> b h l d')
        v = rearrange(v.view(*shape), 'b l d h -> b h l d')

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous()
        y = y.view(b, -1, 128)
        y = self.fc2(y)
        y = F.silu(y)
        return y

class TransformerDecoderLayer(Module):
    def __init__(self, config: tickConfig, cos, sin):
        super().__init__()
        self.mha = mha(config, cos, sin)
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
    
    def forward(self, x):
        y = self.mha(x)
        z = self.fc1(x)
        z = F.silu(z)
        z = self.fc2(z)
        return y + z

class tickModel(PreTrainedModel, LightningModule):
    config_class = tickConfig
    
    def precompute_freqs(self, config: tickConfig):
        channel_range = torch.arange(0, config.hidden_size, 2, dtype=torch.float32, device=config.device)
        inv_freq = 1.0 / (10000 ** (channel_range / config.hidden_size))
        t = torch.arange(119, dtype=torch.float32, device=config.device)
        freqs = torch.outer(t, inv_freq)
        return freqs.cos().bfloat16(), freqs.sin().bfloat16()
    
    def __init__(self, config: tickConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        cos, sin = self.precompute_freqs(config)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config, cos, sin) for _ in range(config.num_layers)
        ])
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def custom_optimizer(self):
        # 
        return torch.optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8)