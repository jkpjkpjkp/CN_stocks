"""TODO: decoderLayer vs decoderBlock"""
import torch
from torch import nn
from torch.nn import functional as F, Module
from einops import rearrange
from .main import dummyLightning
from typing import Optional


def apply_rotary_emb(x, cos, sin):
    l = x.shape[-2]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos[:l] + x2 * sin[:l]  # rotate pairs of dims
    y2 = x1 * (-sin[:l]) + x2 * cos[:l]
    out = torch.cat([y1, y2], -1)  # re-assemble
    out = out.to(x.dtype)
    return out


class Rope(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        device = getattr(config, 'device', 'cuda')
        if getattr(config, 'standard_rope', True):
            attn_dim = config.hidden_dim
        else:
            attn_dim = config.head_dim * config.num_heads
        channel_range = torch.arange(0, attn_dim, 2, dtype=torch.float32,
                                     device=device)
        inv_freq = 1.0 / (10000 ** (channel_range / attn_dim))
        t = torch.arange(config.seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        self.cos, self.sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()

    def forward(self, x):
        return apply_rotary_emb(x, self.cos, self.sin)


class mha(Module):
    def __init__(self, config):
        super().__init__()
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        device = getattr(config, 'device', 'cuda')
        channel_range = torch.arange(0, config.hidden_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (10000 ** (channel_range / config.hidden_dim))
        t = torch.arange(config.seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        self.cos, self.sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()

        self.num_heads = config.num_heads
        assert config.hidden_dim % config.num_heads == 0
        self.head_dim = config.hidden_dim // config.num_heads
        self.hidden_dim = config.hidden_dim

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
        y = y.view(b, -1, self.hidden_dim)
        y = self.fc2(y)
        y = F.silu(y)
        return y


class decoderLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.attn = mha(config)
        intermediate_size = getattr(config, 'intermediate_size', getattr(config, 'interim_dim', config.hidden_dim * 4))
        self.l1 = nn.Linear(config.hidden_dim, intermediate_size)
        self.l2 = nn.Linear(intermediate_size, config.hidden_dim)
        norm = getattr(config, 'norm', 'LayerNorm')
        if norm == 'LayerNorm':
            self.norm1 = nn.LayerNorm(config.hidden_dim)
            self.norm2 = nn.LayerNorm(config.hidden_dim)
        elif norm == 'RMSNorm':
            self.norm1 = self.norm2 = F.rms_norm
        else:
            raise ValueError(f'Unknown norm: {norm}')

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.l2(F.silu(self.l1(self.norm2(x))))
        return x


class decoderBlock(dummyLightning):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        attn_dim = self.num_heads * self.head_dim
        self.attn_dim = attn_dim
        self.q_proj = nn.Linear(self.hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, attn_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, self.hidden_dim, bias=False)
        self.rope = Rope(config)
        if config.qk_norm:
            self.q_norm = nn.LayerNorm(attn_dim)
            self.k_norm = nn.LayerNorm(attn_dim)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, config.interim_dim),
            nn.SiLU(),
            nn.Linear(config.interim_dim, self.hidden_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (batch, seq_len, hidden_dim)
        mask: (batch, seq_len) or (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        residual = x

        x = self.norm1(x)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if not self.standard_rope:
            q = self.rope(q)
            k = self.rope(k)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.standard_rope:
            q = self.rope(q)
            k = self.rope(k)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)\
                       .transpose(1, 2).contiguous()\
                       .view(batch_size, seq_len, self.attn_dim)

        x = residual + self.o_proj(attn_output)

        x = x + self.ffn(self.norm2(x))

        return x


class TM(dummyLightning):
    def __init__(self, config):
        super().__init__(config)
        num_layers = getattr(config, 'layers', getattr(config, 'num_layers', 6))
        self.layers = nn.ModuleList([decoderBlock(config) for _ in range(num_layers)])
        self.optimizers()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
