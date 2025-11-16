import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch import multinomial
from lightning import LightningModule as Module, LightningDataModule as DataModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import Callback, RichProgressBar, ModelCheckpoint
import numpy as np
from einops import rearrange
import random
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def apply_rotary_emb(x, cos, sin):
    l = x.shape[1]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos[:l] + x2 * sin[:l] # rotate pairs of dims
    y2 = x1 * (-sin[:l]) + x2 * cos[:l]
    out = torch.cat([y1, y2], -1) # re-assemble
    assert out.shape[2:] == (128,)
    assert out.shape[1] <= 118
    out = out.to(x.dtype)
    return out

class ds(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.data = np.load(filename)
        self.q = np.load('q.npy')

    def __len__(self):
        return len(self.data) // 119

    def __getitem__(self, idx):
        x = self.data[idx * 119 : (idx+1) * 119]
        x = np.searchsorted(self.q, x)
        return x

class mha(Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.qkv = nn.Linear(128, 3 * 128)
        self.fc2 = nn.Linear(128, 128)
        
        channel_range = torch.arange(0, 128, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (10000 ** (channel_range / 128))
        t = torch.arange(118, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        self.cos, self.sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()

        self.num_heads = 4

    def forward(self, x, return_kv=False):
        q, k, v = torch.chunk(self.qkv(x), 3, -1)
        q = apply_rotary_emb(q, self.cos, self.sin)
        k = apply_rotary_emb(k, self.cos, self.sin)
        b = x.shape[0]

        if return_kv:
            kv = (k, v)
        
        shape = (b, -1, 32, 4)
        q = rearrange(q.view(*shape), 'b l d h -> b h l d')
        k = rearrange(k.view(*shape), 'b l d h -> b h l d')
        v = rearrange(v.view(*shape), 'b l d h -> b h l d')

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous()
        y = y.view(b, -1, 128)
        y = self.fc2(y)
        y = F.silu(y)
        return y if not return_kv else (y, kv)

    def generate(self, x, kv):
        q, k, v = torch.chunk(self.qkv(x.unsqueeze(1)), 3, -1)

        l = kv[0].shape[1]
        q = apply_rotary_emb(q, self.cos[l:], self.sin[l:])
        k = apply_rotary_emb(k, self.cos[l:], self.sin[l:])
        ret_kv = (k, v)
        k = torch.concat(kv[0], k, dim=1)
        v = torch.concat(kv[1], v, dim=1)

        b = x.shape[0]
        shape = (b, -1, 32, 4)
        q = rearrange(q.view(*shape), 'b l d h -> b h l d')
        k = rearrange(k.view(*shape), 'b l d h -> b h l d')
        v = rearrange(v.view(*shape), 'b l d h -> b h l d')
        y = F.scaled_dot_product_attention(q, k, v)

        y = y.transpose(1, 2).contiguous()
        # assert y.shape == (b, 118, 4, 32)
        y = y.view(b, -1, 128)
        y = self.fc2(y)
        y = F.silu(y)
        return y, ret_kv

class tm(Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.emb = nn.Embedding(128, 128)
        
        self.attn1 = mha(device=device)
        self.attn2 = mha(device=device)

        self.l1 = nn.Linear(128, 256)
        self.l2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 128)

        self.norm = nn.LayerNorm(128)
    
    def forward(self, x):
        x = self.emb(x)

        x = x + self.attn1(x)
        x = x + self.l2(F.silu(self.l1(x)))

        x = x + self.attn2(self.norm(x))
        x = self.fc3(x)

        return x
        
    def generate(self, x):
        x = self.emb(x)

        f, kv1 = self.attn1(x, return_kv=True)
        x = x + f
        x = x + self.l2(F.silu(self.l1(x)))

        f, kv2 = self.attn2(self.norm(x), return_kv=True)
        x = x + f
        x = self.fc3(x)

        p = x[:, -1, :]
        
        p = p.tile(64, 1)
        kv1 = (kv1[0].tile(64, 1, 1), kv1[1].tile(64, 1, 1))
        kv2 = (kv2[0].tile(64, 1, 1), kv2[1].tile(64, 1, 1))

        logits = []
        seqs = []

        for _ in range(29):
            p = F.softmax(p, dim=-1)
            y = multinomial(p, num_samples=1)
            seqs.append(y)
            logits.append(p.gather(dim=1, index=y).squeeze(1))

            x = self.emb(y)
            f, kv1 = self.attn1.generate(y, kv1)
            x = x + f
            x = x + self.l2(F.silu(self.l1(x)))

            f, kv2 = self.attn2.generate(self.norm(x), kv2)
            x = x + f
            x = self.fc3(x)

            p = x.squeeze(1)

        return seqs, logits

    
    def training_step(self, batch, batch_idx):
        seqs, logits = self.generate(batch[:, :-30])
        

    def validation_step(self, batch, batch_idx):
        y = self(batch[:, :-1])
        loss = nn.CrossEntropyLoss()(y.view(-1, 128), batch[:, 1:].contiguous().view(-1))
        if batch_idx % 10 == 0:
            self.log('val/loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return {
            'optimizer': optimizer,
        }

class loggingMixin(Callback):
    def __init__(self, every_n_steps):
        super().__init__()
        self.every_n_steps = every_n_steps

    def on_before_optimizer_step(self, model, closure, optimizer):
        layer_norms = grad_norm(closure, norm_type=2)
        self.log_dict(layer_norms, on_step=True, on_epoch=True, logger=True)
    
    def on_train_batch_end(self, trainer: Trainer, pl_module: Module, outputs, batch, batch_idx: int) -> None:
        fig, ax = plt.subplots()
        ax.plot(np.arange(118), outputs['sample'][1:], label='gt')
        ax.plot(np.arange(118), outputs['pred'], label='pred')
        trainer.logger.experiment.log_figure(trainer.logger.run_id, fig, f"train/plt{batch_idx}.png") 
        plt.close(fig)
    
    def on_val_batch_end(self, trainer: Trainer, pl_module: Module, outputs, batch, batch_idx: int) -> None:
        fig, ax = plt.subplots()
        ax.plot(np.arange(118), outputs['sample'][1:], label='gt')
        ax.plot(np.arange(118), outputs['pred'], label='pred')
        trainer.logger.experiment.log_figure(trainer.logger.run_id, fig, f"val/plt{batch_idx}.png") 
        plt.close(fig)

 
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    model = tm()
    data = DataModule.from_datasets(ds('../data/train.npy'), ds('../data/eval.npy'), batch_size=4096, num_workers=32)
    trainer = Trainer(
        max_epochs=42,
        gradient_clip_val=1.,
        callbacks=[
            RichProgressBar(), 
            loggingMixin(every_n_steps=20),
            ModelCheckpoint(dirpath="./mlruns/models/", save_top_k=2, monitor="val/loss"),
        ],
        logger=MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./mlruns", artifact_location='./ml-runs/artifacts/'),
    )

    trainer.fit(model, data)