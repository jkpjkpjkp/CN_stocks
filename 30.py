import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler
from lightning import LightningModule as Module, LightningDataModule as DataModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import Callback, RichProgressBar, ModelCheckpoint
from lightning.pytorch.utilities import grad_norm
import numpy as np
from einops import rearrange
import random
import matplotlib.pyplot as plt

from minute.b import mha, loggingMixin

class ds(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.data = np.load(filename)
        self.q = np.load('q.npy')
        self.q30 = np.load('q30.npy')
        self
    def __len__(self):
        return len(self.data) // 119

    def __getitem__(self, idx):
        x = self.data[idx * 119 : (idx+1) * 119]
        x = np.searchsorted(self.q, x)
        y = np.prod(np.lib.stride_tricks.sliding_window_view(x, 30), axis=-1).flatten()
        y = np.searchsorted(self.q30, y)
        return x, y

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class t30m(Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.emb = nn.Embedding(128, 128)
        
        self.attn1 = mha(device=device)
        self.attn2 = mha(device=device)

        self.l1 = nn.Linear(128, 256)
        self.l2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 128)

        self.norm = nn.LayerNorm(128)
        self.fc1 = nn.Linear(1, 128)
    
    def forward(self, x):
        with torch.no_grad():
            x = self.emb(x)

            x = x + self.attn1(x)
            x = x + self.l2(F.silu(self.l1(x)))

            x = x + self.attn2(self.norm(x))
        
        x = self.fc3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x[:, :-35])
        loss = nn.CrossEntropyLoss()(y_hat.view(-1, 128), y[:, 6:].contiguous().view(-1))
        if batch_idx % 10 == 0:
            self.log('train/loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x[:, :-35])
        loss = nn.CrossEntropyLoss()(y_hat.view(-1, 128), y[:, 5:].contiguous().view(-1))
        if batch_idx % 10 == 0:
            self.log('val/loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return {
            'optimizer': optimizer,
        }

def main():
    model = t30m.load_from_checkpoint('ml-runs/models/epoch=40-step=213610.ckpt')
    
    torch.set_float32_matmul_precision('medium')
    data = DataModule.from_datasets(ds('../data/train.npy'), ds('../data/eval.npy'), batch_size=4096, num_workers=16)
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

if __name__ == '__main__':
    main()