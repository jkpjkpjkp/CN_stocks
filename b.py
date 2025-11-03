import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from lightning import LightningModule as Module, LightningDataModule as DataModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import Callback, RichProgressBar
from lightning.pytorch.utilities import grad_norm
import numpy as np

def _precompute_rotary_embeddings(seq_len, head_dim, base=4242, device='cuda'):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        return cos, sin

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 3
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], -1) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class ds(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.data = np.load(filename)
    
    def __len__(self):
        return len(self.data) // 119

    def __getitem__(self, idx):
        x = self.data[idx * 119 : (idx+1) * 119]
        x = torch.tensor(x)
        x = (torch.clamp(x, 0.99, 1.01) - 1) * 420
        return x

class tm(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 128)
        self.qkv = nn.Linear(128, 3 * 128)
        self.cos, self.sin = _precompute_rotary_embeddings(118, 128 / 4)
        self.num_heads = 4
        device='cuda'
        channel_range = torch.arange(0, 128, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (10000 ** (channel_range / 128))

        t = torch.arange(118, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        self.cos, self.sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()

        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

        self.norm = nn.LayerNorm(128)
    
    def rope(self, x):
        dtype = x.dtype
        x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x = torch.view_as_real(x * self.freq.unsqueeze(0)).flatten(3)
        return x.to(dtype)
    
    def forward(self, x):
        b = x.shape[0]
        assert x.shape == (b, 118)

        x = self.fc1(x.unsqueeze(-1))
        x = F.silu(x)

        q, k, v = torch.chunk(self.qkv(x), 3, -1)
        q = apply_rotary_emb(q, self.cos, self.sin)
        k = apply_rotary_emb(k, self.cos, self.sin)

        shape = (b, 118, 32, 4)
        q = q.view(*shape)
        k = k.view(*shape)
        v = v.view(*shape)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, 118, 128)
        y = self.fc2(y)
        y = F.silu(y)

        x = y
        y = self.norm(y)
        q, k, v = torch.chunk(self.qkv(x), 3, -1)
        q = apply_rotary_emb(q, self.cos, self.sin)
        k = apply_rotary_emb(k, self.cos, self.sin)

        shape = (b, 118, 32, 4)
        q = q.view(*shape)
        k = k.view(*shape)
        v = v.view(*shape)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, 118, 128)
        y = self.fc2(y)
        y = F.silu(y)

        y = self.fc3(x + y).squeeze(-1)

        return y
    
    def training_step(self, batch, batch_idx):
        y = self(batch[:, :-1])
        loss = nn.HuberLoss()(batch[:, 1:] * 3, y * 3)
        if batch_idx % 10 == 0:
            self.log('train/loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = self(batch[:, :-1])
        loss = nn.HuberLoss()(batch[:, 1:] * 3, y * 3)
        if batch_idx % 10 == 0:
            self.log('eval/loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-4)

class loggingMixin(Callback):
    def __init__(self, every_n_steps):
        super().__init__()
        self.every_n_steps = every_n_steps

    def on_before_optimizer_step(self, model, closure, optimizer):
        layer_norms = grad_norm(closure, norm_type=2)
        self.log_dict(layer_norms, on_step=True, on_epoch=True, logger=True)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    model = tm()
    trainer = Trainer(
        max_epochs=42,
        gradient_clip_val=2.,
        callbacks=[RichProgressBar(), loggingMixin(every_n_steps=20)],
        logger=MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs"),
    )

    trainer.fit(model, DataModule.from_datasets(ds('../data/train.npy'), ds('../data/eval.npy'), batch_size=2048))

    breakpoint()