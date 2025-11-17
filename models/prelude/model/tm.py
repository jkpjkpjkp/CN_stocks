import torch
from torch import nn
from torch.nn import functional as F, Module
from einops import rearrange
from .main import dummyLightning

def apply_rotary_emb(x, cos, sin):
    l = x.shape[1]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos[:l] + x2 * sin[:l] # rotate pairs of dims
    y2 = x1 * (-sin[:l]) + x2 * cos[:l]
    out = torch.cat([y1, y2], -1) # re-assemble
    out = out.to(x.dtype)
    return out

class mha(Module):
    def __init__(self, config):
        super().__init__()
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        device = config.device
        channel_range = torch.arange(0, config.hidden_size, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (10000 ** (channel_range / config.hidden_size))
        t = torch.arange(config.seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        self.cos, self.sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()

        self.num_heads = config.num_heads
        assert config.hidden_size % config.num_heads == 0
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size

    def forward(self, x):
        q, k, v = torch.chunk(self.qkv(x), 3, -1)
        q = apply_rotary_emb(q, self.cos, self.sin)
        k = apply_rotary_emb(k, self.cos, self.sin)
        b = x.shape[0]
        
        shape = (b, -1, self.head_dim, self.num_heads)
        q = rearrange(q.view(*shape), 'b l d h -> b h l d')
        k = rearrange(k.view(*shape), 'b l d h -> b h l d')
        v = rearrange(v.view(*shape), 'b l d h -> b h l d')

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous()
        y = y.view(b, -1, self.hidden_size)
        y = self.fc2(y)
        y = F.silu(y)
        return y

class decoderLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.attn = mha(config)
        self.l1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.l2 = nn.Linear(config.intermediate_size, config.hidden_size)
        if config.norm == 'LayerNorm':
            self.norm1 = nn.LayerNorm(config.hidden_size)
            self.norm2 = nn.LayerNorm(config.hidden_size)
        elif config.norm == 'RMSNorm':
            self.norm1 = self.norm2 = F.rms_norm
        else:
            raise ValueError(f'Unknown norm: {config.norm}')
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.l2(F.silu(self.l1(self.norm2(x))))
        return x

class tm(dummyLightning):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([decoderLayer(config) for _ in range(config.layers)])
        self.optimizers()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
