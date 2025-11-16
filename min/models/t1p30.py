import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from lightning import LightningModule as Module, LightningDataModule as DataModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
import numpy as np
import random
import os
import subprocess

from min.t1p1 import mha, loggingMixin

# @numba.jit(nopython=True)
def item_factory(slice, q, q30):
    x = slice
    y = np.prod(np.lib.stride_tricks.sliding_window_view(x, 30), axis=-1).flatten()
    y = np.searchsorted(q30, y)
    x = np.searchsorted(q, x)
    return x, y

class ds(Dataset):
    def __init__(self, filename, x=None):
        super().__init__()
        self.data = np.load(filename) if filename else x
        self.q = np.load('q.npy')
        self.q30 = np.load('q30.npy')

    @classmethod
    def from_array(cls, x):
        return cls(None, x)
    
    def __len__(self):
        return len(self.data) // 119

    def __getitem__(self, idx):
        return item_factory(self.data[idx * 119 : (idx+1) * 119], self.q, self.q30)
        x = self.data[idx * 119 : (idx+1) * 119]
        y = np.prod(np.lib.stride_tricks.sliding_window_view(x, 30), axis=-1).flatten()
        y = np.searchsorted(self.q30, y)
        x = np.searchsorted(self.q, x)
        return x, y

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class t30m(Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.emb1 = nn.Embedding(128, 128)
        self.emb30 = nn.Embedding(128, 128)
        
        self.attn1 = mha(device=device)
        self.attn2 = mha(device=device)
        self.attn3 = mha(device=device)
        self.attn4 = mha(device=device)
  
        self.l1 = nn.Linear(128, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 256)
        self.l6 = nn.Linear(256, 128)
        self.l7 = nn.Linear(128, 256)

        self.readout = nn.Linear(256, 128)

        self.norm = nn.LayerNorm(128)
    
    def forward(self, x1, x30):
        b = x1.shape[0]
        x1 = self.emb1(x1)
        x30 = self.emb30(x30)
        x = x1 + torch.concat((torch.zeros((b, 29, 128), device=x30.device, dtype=x30.dtype), x30), dim=1)
        
        x = x + self.attn1(x)
        x = x + self.l2(F.silu(self.l1(x)))

        x = x + self.attn2(self.norm(x))
        x = x + self.l4(F.silu(self.l3(x)))

        x = x + self.attn3(self.norm(x))
        x = x + self.l6(F.silu(self.l5(x)))

        x = x + self.attn4(self.norm(x))
        x = self.readout(F.silu(self.l7(x)))

        return x
    
    def training_step(self, batch, batch_idx, train = 'train'):
        x, y = batch
        y_hat = self(x[:, :-30], y[:, :-30])
        loss = nn.CrossEntropyLoss()(y_hat.view(-1, 128), y[:, 1:].contiguous().view(-1))
        if batch_idx % 10 == 0:
            self.log(f'{train}/loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return {
            'loss': loss,
            'logits': y_hat,
        }
    
    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, train='val')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        muon = None
        return {
            'optimizer': optimizer,
        }

def main(checkpoint_dir):
    model = t30m()

    torch.set_float32_matmul_precision('medium')
    data = DataModule.from_datasets(
        ds('../data/train.npy'), 
        ds('../data/eval.npy'), 
        batch_size=4096, 
        num_workers=32,
    )
    trainer = Trainer(
        max_epochs=62,
        gradient_clip_val=1.,
        precision="bf16-mixed",
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        callbacks=[
            RichProgressBar(),
            loggingMixin(every_n_steps=20),
            ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=2, monitor="val/loss"),
        ],
        logger=MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./mlruns", artifact_location='./ml-runs/artifacts/'),
    )

    trainer.fit(model, data)

if __name__ == '__main__':
    checkpoint_dir = '.checkpoint/1-30-4-128-p-30'
    os.makedirs(checkpoint_dir, exist_ok=True)
    subprocess.run(['cp', __file__, checkpoint_dir])
    main(checkpoint_dir)